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

void migrate_topology(MPI_Comm                                       comm,
                      tracer_container_t<>&                          tr,
                      Kokkos::View<int*, grace::default_space>       export_ranks)
{
    const std::size_t n_in = tr.size();
    if (n_in == 0) return;

    distribution_plan_t plan(comm, export_ranks);
    const std::size_t n_out = plan.total_num_import();

    // Allocate destination container. resize() (not resize_preserving): the
    // migration repopulates everything from network.
    tracer_container_t<> nu;
    nu.resize(n_out);
    nu.set_id_counter(tr.id_counter()); // counter is rank-local, survives migration

    if (n_out > 0 || n_in > 0) {
        migrate(plan, tr.pos,    nu.pos);
        migrate(plan, tr.id,     nu.id);
        migrate(plan, tr.status, nu.status);
        // owner_rank/owner_local_quad and samples intentionally not migrated:
        // owner_rank gets set to self below; owner_local_quad is re-resolved
        // from position; samples get refilled by next fetch.
    }

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
