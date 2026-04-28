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
#include <grace/utils/grace_utils.hh>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <mpi.h>

namespace grace {
namespace particles {

class distribution_plan_t;

/// RAII handle around an in-flight non-blocking MPI_Ialltoallv. Owns the
/// device-resident send buffer until wait() so MPI can read from it
/// asynchronously. The receive buffer (dst from the migrate_async call) is
/// caller-owned and must not be touched until wait() returns — same semantics
/// as MPI_Irecv.
class migrate_handle_t {
  public:
    migrate_handle_t() = default;
    migrate_handle_t(const migrate_handle_t&) = delete;
    migrate_handle_t& operator=(const migrate_handle_t&) = delete;
    migrate_handle_t(migrate_handle_t&& other) noexcept;
    migrate_handle_t& operator=(migrate_handle_t&& other) noexcept;
    ~migrate_handle_t(); // calls wait() if still in-flight

    /// Block until the underlying MPI_Ialltoallv completes.
    void wait();

    /// Non-blocking poll. Returns true if the transfer has completed.
    /// Calling wait()/test() after the handle has completed is a no-op.
    bool test();

    /// True iff the handle has no outstanding MPI request.
    bool completed() const noexcept { return _completed; }

  private:
    friend class distribution_plan_t;
    MPI_Request _req = MPI_REQUEST_NULL;
    Kokkos::View<char*, grace::default_space> _send_buf; // kept alive in flight
    bool _completed = true;
};

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

    // Internal helpers exposed to the typed migrate templates.
    template <typename T> friend void migrate(
        const distribution_plan_t&,
        Kokkos::View<T*, grace::default_space>,
        Kokkos::View<T*, grace::default_space>);

    template <typename T> friend migrate_handle_t migrate_async(
        const distribution_plan_t&,
        Kokkos::View<T*, grace::default_space>,
        Kokkos::View<T*, grace::default_space>);

    template <typename T, std::size_t N> friend void migrate(
        const distribution_plan_t&,
        Kokkos::View<T*[N], grace::default_space>,
        Kokkos::View<T*[N], grace::default_space>);

    /// Pack the source bytes into a fresh send buffer and post MPI_Ialltoallv
    /// into the (caller-owned) dst buffer. Returns a handle owning the send
    /// buffer + the MPI request.
    migrate_handle_t migrate_async_raw(const void* src_data,
                                       std::size_t bytes_per_elem,
                                       void* dst_data) const;
};

/// Asynchronous migrate. Returns immediately after posting the underlying
/// MPI_Ialltoallv. Caller must not touch `dst` until handle.wait() / .test().
template <typename T>
[[nodiscard]] migrate_handle_t migrate_async(
    const distribution_plan_t& plan,
    Kokkos::View<T*, grace::default_space> src,
    Kokkos::View<T*, grace::default_space> dst)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "particles::migrate_async<T>: T must be trivially copyable.");
    // Defense in depth: we treat src/dst as flat byte buffers via .data().
    // Any non-contiguous view (e.g. a strided subview into a wider parent)
    // would silently corrupt the transfer.
    ASSERT(src.span_is_contiguous(),
           "particles::migrate_async: src view is not contiguous.");
    ASSERT(dst.span_is_contiguous(),
           "particles::migrate_async: dst view is not contiguous.");
    return plan.migrate_async_raw(static_cast<const void*>(src.data()),
                                  sizeof(T),
                                  static_cast<void*>(dst.data()));
}

/// Synchronous migrate. Thin wrapper around migrate_async + handle.wait().
template <typename T>
void migrate(const distribution_plan_t& plan,
             Kokkos::View<T*, grace::default_space> src,
             Kokkos::View<T*, grace::default_space> dst)
{
    auto handle = migrate_async(plan, src, dst);
    handle.wait();
}

/// Synchronous migrate for multi-component views (Kokkos::View<T*[N]>).
/// Treats each row of N components as one packed element of size N*sizeof(T).
///
/// CRITICAL: the byte-pack inside migrate_async_raw assumes element i's N
/// components live contiguously at offset [i*N, i*N+N). That is only true
/// for LayoutRight. The Kokkos default device layout is LayoutLeft on
/// CUDA/HIP, where the storage is per-component-major (x[0..N-1],
/// y[0..N-1], z[0..N-1]) — so a naive byte-pack would scramble component
/// indices into element indices and silently corrupt every migrated row.
///
/// We therefore stage src/dst through explicit LayoutRight buffers and let
/// Kokkos::deep_copy do the layout transpose. The rank-2 migrate is only
/// used on small-element trajectory data (e.g. tr.pos), so the extra two
/// deep_copies are not on a hot path.
///
/// No async overload: the staging buffers would have to outlive the MPI
/// request and the post-wait copy-out, complicating the handle without
/// meaningfully improving overlap for the tracer use case.
template <typename T, std::size_t N>
void migrate(const distribution_plan_t& plan,
             Kokkos::View<T*[N], grace::default_space> src,
             Kokkos::View<T*[N], grace::default_space> dst)
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "particles::migrate<T,N>: T must be trivially copyable.");
    using lr_view_t = Kokkos::View<T*[N], Kokkos::LayoutRight,
                                   grace::default_space>;
    lr_view_t src_lr(Kokkos::ViewAllocateWithoutInitializing("migrate_src_lr"),
                     src.extent(0));
    lr_view_t dst_lr(Kokkos::ViewAllocateWithoutInitializing("migrate_dst_lr"),
                     dst.extent(0));
    Kokkos::deep_copy(src_lr, src);
    auto handle = plan.migrate_async_raw(
        static_cast<const void*>(src_lr.data()),
        N * sizeof(T),
        static_cast<void*>(dst_lr.data()));
    handle.wait();
    Kokkos::deep_copy(dst, dst_lr);
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_DISTRIBUTOR_HH
