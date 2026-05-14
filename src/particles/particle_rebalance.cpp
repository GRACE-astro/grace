/**
 * @file particle_rebalance.cpp
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_rebalance.hh>
#include <grace/particles/particle_distributor.hh>
#include <grace/particles/particle_owner_search.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace grace {
namespace particles {

Kokkos::View<int*, grace::default_space>
compute_export_ranks_quad_owner(const tracer_container_t<>& tr)
{
    const std::size_t n = tr.size();
    Kokkos::View<int*, grace::default_space> ranks("export_ranks_quad_owner", n);
    if (n == 0) return ranks;

    auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.pos);
    std::vector<std::array<double, 3>> positions(n);
    for (std::size_t i = 0; i < n; ++i) {
        positions[i] = {h_pos(i, 0), h_pos(i, 1), h_pos(i, 2)};
    }
    std::vector<owner_t> owners;
    fluid_topology_shadow_t::get().find_owners_batch(positions, owners);

    auto h_ranks = Kokkos::create_mirror_view(ranks);
    for (std::size_t i = 0; i < n; ++i) {
        h_ranks(i) = owners[i].rank; // -1 propagates as "drop"
    }
    Kokkos::deep_copy(ranks, h_ranks);
    return ranks;
}

rebalance_decision_t
decide_rebalance_quad_owner(const tracer_container_t<>& tr,
                            double imbalance_threshold)
{
    rebalance_decision_t d;
    const std::size_t n        = tr.size();
    const int         self_rank = parallel::mpi_comm_rank();

    d.export_ranks = Kokkos::View<int*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("decide_export_ranks"), n);
    d.local_quads  = Kokkos::View<int*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("decide_local_quads"), n);

    // Local count of tracers that would migrate. Excludes -1 (off-domain
    // drops) — those aren't a load-balance concern, they're cull events.
    long long local_n_migrating = 0;

    if (n > 0) {
        auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         tr.pos);
        std::vector<std::array<double, 3>> positions(n);
        for (std::size_t i = 0; i < n; ++i) {
            positions[i] = {h_pos(i, 0), h_pos(i, 1), h_pos(i, 2)};
        }
        std::vector<owner_t> owners;
        fluid_topology_shadow_t::get().find_owners_batch(positions, owners);

        auto h_er = Kokkos::create_mirror_view(d.export_ranks);
        auto h_lq = Kokkos::create_mirror_view(d.local_quads);
        for (std::size_t i = 0; i < n; ++i) {
            h_er(i) = owners[i].rank;
            // local_quad is meaningful only on the owning rank.
            h_lq(i) = (owners[i].rank == self_rank) ? owners[i].local_quad : -1;
            if (owners[i].rank != self_rank && owners[i].rank >= 0) {
                ++local_n_migrating;
            }
        }
        Kokkos::deep_copy(d.export_ranks, h_er);
        Kokkos::deep_copy(d.local_quads,  h_lq);
    }

    // Allreduce {total, n_migrating} so every rank gets the same imbalance
    // and the same should_migrate decision. Must run unconditionally.
    long long lg[2] = { static_cast<long long>(n), local_n_migrating };
    long long gg[2] = { 0, 0 };
    MPI_Allreduce(lg, gg, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    d.imbalance      = (gg[0] > 0) ? static_cast<double>(gg[1])
                                     / static_cast<double>(gg[0])
                                   : 0.0;
    d.should_migrate = (d.imbalance > imbalance_threshold);
    return d;
}

void update_owner_only(tracer_container_t<>&        tr,
                       const rebalance_decision_t&  decision)
{
    const std::size_t n = tr.size();
    if (n == 0) return;
    auto own_r = tr.owner_rank;
    auto own_q = tr.owner_local_quad;
    auto er    = decision.export_ranks;
    auto lq    = decision.local_quads;
    Kokkos::parallel_for("update_owner_only",
        Kokkos::RangePolicy<grace::default_execution_space>(0, n),
        KOKKOS_LAMBDA(const int i) {
            // -1 entries (off-domain drops) get carried through; the next
            // flag_outside_domain + cull pass deals with them. For everything
            // else, owner_rank now points at the rank that owns the fluid
            // quad containing this tracer, so the next fetch routes there
            // even though the persistent data hasn't been moved yet.
            own_r(i) = er(i);
            own_q(i) = lq(i);
        });
}

void migrate_topology(MPI_Comm                                       comm,
                      tracer_container_t<>&                          tr,
                      Kokkos::View<int*, grace::default_space>       export_ranks)
{
    // No early return on n_in == 0: distribution_plan_t's ctor posts
    // MPI_Alltoall, and migrate(...) posts MPI_Ialltoallv. Both are
    // collective and must be entered by every rank, even those with empty
    // input. The distributor handles all-zero counts correctly.
    distribution_plan_t plan(comm, export_ranks);
    const std::size_t n_out = plan.total_num_import();

    // Allocate destination container. resize() (not resize_preserving): the
    // migration repopulates everything from network.
    tracer_container_t<> nu;
    nu.resize(n_out);
    nu.set_id_counter(tr.id_counter()); // counter is rank-local, survives migration

    // Always enter the migrate calls — Ialltoallv is collective.
    migrate(plan, tr.pos,    nu.pos);
    migrate(plan, tr.id,     nu.id);
    migrate(plan, tr.status, nu.status);
    // owner_rank/owner_local_quad and samples intentionally not migrated:
    // owner_rank gets set to self below; owner_local_quad is re-resolved
    // from position; samples get refilled by next fetch.

    // Post-migration topology fix-up: owner_rank = self, owner_local_quad
    // resolved from positions on the receiver side.
    if (n_out > 0) {
        const int self_rank = parallel::mpi_comm_rank();
        Kokkos::deep_copy(nu.owner_rank, self_rank);

        auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nu.pos);
        std::vector<std::array<double, 3>> positions(n_out);
        for (std::size_t i = 0; i < n_out; ++i) {
            positions[i] = {h_pos(i, 0), h_pos(i, 1), h_pos(i, 2)};
        }
        std::vector<owner_t> owners;
        fluid_topology_shadow_t::get().find_owners_batch(positions, owners);

        auto h_quad = Kokkos::create_mirror_view(nu.owner_local_quad);
        for (std::size_t i = 0; i < n_out; ++i) {
            // Under quad-owner strategy this matches; under other strategies
            // it may be -1, and the fetch path falls back to slow search.
            h_quad(i) = (owners[i].rank == self_rank) ? owners[i].local_quad : -1;
        }
        Kokkos::deep_copy(nu.owner_local_quad, h_quad);
    }

    tr = std::move(nu);
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
