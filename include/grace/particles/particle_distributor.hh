/**
 * @file particle_distributor.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Cabana-free MPI distributor for trivially-copyable POD elements.
 *
 * Replaces what we previously got from Cabana::Distributor. The plan is built
 * from a per-element destination-rank view (export_ranks); elements with
 * export_rank<0 are dropped. The plan can then be used to migrate any
 * Kokkos::View<T*> of trivially-copyable T from the source layout to the
 * imported layout.
 *
 * Implementation: build per-rank send/recv counts via MPI_Alltoall, then a
 * device-side pack kernel writes elements into the send buffer in the order
 * MPI_Alltoallv expects. GPU-aware MPI is assumed (same path GRACE uses for
 * fluid halos).
 */
#ifndef GRACE_PARTICLES_PARTICLE_DISTRIBUTOR_HH
#define GRACE_PARTICLES_PARTICLE_DISTRIBUTOR_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <mpi.h>

namespace grace {
namespace particles {

class distribution_plan_t {
  public:
    /// Build a plan from a per-element destination-rank view.
    /// Elements with export_ranks(i) < 0 are dropped (not exported).
    distribution_plan_t(MPI_Comm comm,
                        Kokkos::View<int*, grace::default_space> export_ranks);

    /// Number of elements this rank will receive after migrate.
    std::size_t total_num_import() const noexcept { return _total_recv; }

    /// Number of elements this rank will send (sum over destinations).
    std::size_t total_num_export() const noexcept { return _total_send; }

    /// Number of input elements (matches export_ranks.size() at construction).
    std::size_t total_num_input() const noexcept { return _n_input; }

  private:
    MPI_Comm    _comm;
    int         _nproc       = 0;
    int         _rank        = 0;
    std::size_t _n_input     = 0;
    std::size_t _total_send  = 0;
    std::size_t _total_recv  = 0;

    // Per-rank counts and prefix-sum offsets, host-side (small, ~O(nproc)).
    std::vector<int> _send_counts;
    std::vector<int> _send_offsets;
    std::vector<int> _recv_counts;
    std::vector<int> _recv_offsets;

    // Per-input-element destination slot in the packed send buffer
    // (or -1 if element is dropped).
    Kokkos::View<int*, grace::default_space> _send_perm;

    // Internal helper exposed to the typed migrate() template.
    template <typename T> friend void migrate(
        const distribution_plan_t&,
        Kokkos::View<T*, grace::default_space>,
        Kokkos::View<T*, grace::default_space>);

    void migrate_raw(const void* src_data, std::size_t bytes_per_elem,
                     void* dst_data) const;
};

/// Migrate a Kokkos::View<T*> of trivially-copyable T according to the plan.
/// `dst` must already be sized to plan.total_num_import().
template <typename T>
void migrate(const distribution_plan_t& plan,
             Kokkos::View<T*, grace::default_space> src,
             Kokkos::View<T*, grace::default_space> dst)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "particles::migrate<T>: T must be trivially copyable.");
    plan.migrate_raw(static_cast<const void*>(src.data()),
                     sizeof(T),
                     static_cast<void*>(dst.data()));
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_DISTRIBUTOR_HH
