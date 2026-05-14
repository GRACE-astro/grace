/**
 * @file particle_distributor.cpp
 * @brief Implementation of distribution_plan_t and migrate_handle_t.
 *
 * Plan construction (constructor):
 *   1. Per-destination-rank counts on this rank.
 *   2. MPI_Alltoall to learn how many elements each rank will send to us.
 *   3. Build a per-input permutation that places each element into the
 *      correct slot of the (eventually packed) send buffer in destination-
 *      rank order, so that MPI_Alltoallv can ship contiguous slabs per peer.
 *
 * Migrate (per call):
 *   1. Allocate a device-resident send buffer.
 *   2. Pack kernel: byte-memcpy each input element into its destination slot.
 *   3. Kokkos::fence to flush the pack before MPI reads from send_buf.
 *   4. Post MPI_Ialltoallv. The send buffer lives inside the migrate_handle_t
 *      so it stays alive until wait().
 *   5. Synchronous migrate() = migrate_async() + handle.wait().
 *
 * GPU-aware MPI assumed (same path as fluid halos).
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_distributor.hh>
#include <grace/utils/grace_utils.hh>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace grace {
namespace particles {

//*****************************************************************************
// migrate_handle_t lifecycle.
//*****************************************************************************

migrate_handle_t::migrate_handle_t(migrate_handle_t&& other) noexcept
  : _req(other._req)
  , _send_buf(std::move(other._send_buf))
  , _completed(other._completed)
{
    other._req       = MPI_REQUEST_NULL;
    other._completed = true;
}

migrate_handle_t& migrate_handle_t::operator=(migrate_handle_t&& other) noexcept
{
    if (this == &other) return *this;
    wait(); // ensure any prior in-flight transfer is flushed before overwrite
    _req       = other._req;
    _send_buf  = std::move(other._send_buf);
    _completed = other._completed;
    other._req       = MPI_REQUEST_NULL;
    other._completed = true;
    return *this;
}

migrate_handle_t::~migrate_handle_t()
{
    wait();
}

void migrate_handle_t::wait()
{
    if (_completed) return;
    MPI_Wait(&_req, MPI_STATUS_IGNORE);
    _req       = MPI_REQUEST_NULL;
    _completed = true;
    // _send_buf can now be released; happens automatically when handle dies
    // or is reassigned.
}

bool migrate_handle_t::test()
{
    if (_completed) return true;
    int flag = 0;
    MPI_Test(&_req, &flag, MPI_STATUS_IGNORE);
    if (flag) {
        _req       = MPI_REQUEST_NULL;
        _completed = true;
    }
    return _completed;
}

//*****************************************************************************
// distribution_plan_t constructor.
//*****************************************************************************

distribution_plan_t::distribution_plan_t(
    MPI_Comm comm,
    Kokkos::View<int*, grace::default_space> export_ranks)
    : _comm(comm)
    , _n_input(export_ranks.size())
{
    MPI_Comm_size(comm, &_nproc);
    MPI_Comm_rank(comm, &_rank);

    auto h_ranks = Kokkos::create_mirror_view(export_ranks);
    Kokkos::deep_copy(h_ranks, export_ranks);

    _send_counts.assign(_nproc, 0);
    for (std::size_t i = 0; i < _n_input; ++i) {
        const int r = h_ranks(i);
        if (r >= 0 && r < _nproc) ++_send_counts[r];
    }

    _send_offsets.assign(_nproc, 0);
    for (int r = 1; r < _nproc; ++r)
        _send_offsets[r] = _send_offsets[r - 1] + _send_counts[r - 1];
    _total_send = static_cast<std::size_t>(_send_offsets[_nproc - 1]
                                          + _send_counts[_nproc - 1]);

    _recv_counts.assign(_nproc, 0);
    MPI_Alltoall(_send_counts.data(), 1, MPI_INT,
                 _recv_counts.data(), 1, MPI_INT, comm);

    _recv_offsets.assign(_nproc, 0);
    for (int r = 1; r < _nproc; ++r)
        _recv_offsets[r] = _recv_offsets[r - 1] + _recv_counts[r - 1];
    _total_recv = static_cast<std::size_t>(_recv_offsets[_nproc - 1]
                                          + _recv_counts[_nproc - 1]);

    _send_perm = Kokkos::View<int*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_distrib_send_perm"),
        _n_input);
    auto h_perm = Kokkos::create_mirror_view(_send_perm);

    std::vector<int> cursor(_nproc, 0);
    for (std::size_t i = 0; i < _n_input; ++i) {
        const int r = h_ranks(i);
        if (r < 0 || r >= _nproc) {
            h_perm(i) = -1;
        } else {
            h_perm(i) = _send_offsets[r] + cursor[r];
            ++cursor[r];
        }
    }
    Kokkos::deep_copy(_send_perm, h_perm);
}

//*****************************************************************************
// migrate_async_raw: pack + post MPI_Ialltoallv, return handle.
//*****************************************************************************

migrate_handle_t distribution_plan_t::migrate_async_raw(
    const void* src_data,
    std::size_t bytes_per_elem,
    void* dst_data) const
{
    using exec_space = typename grace::default_space::execution_space;

    migrate_handle_t handle;
    handle._send_buf = Kokkos::View<char*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_distrib_send_buf"),
        _total_send * bytes_per_elem);

    auto perm_v = _send_perm;
    auto sbuf_v = handle._send_buf;
    const std::size_t n_in = _n_input;
    const char* src_bytes  = static_cast<const char*>(src_data);

    Kokkos::parallel_for("particle_distrib_pack",
        Kokkos::RangePolicy<exec_space>(0, n_in),
        KOKKOS_LAMBDA(const std::size_t i) {
            const int slot = perm_v(i);
            if (slot < 0) return;
            const char* src_elem = src_bytes + i * bytes_per_elem;
            char*       dst_elem = sbuf_v.data() + std::size_t(slot) * bytes_per_elem;
            for (std::size_t b = 0; b < bytes_per_elem; ++b) {
                dst_elem[b] = src_elem[b];
            }
        });
    // Required: pack must be visible to MPI before posting Ialltoallv.
    Kokkos::fence();

    std::vector<int> sbcnt(_nproc), sboff(_nproc);
    std::vector<int> rbcnt(_nproc), rboff(_nproc);
    for (int r = 0; r < _nproc; ++r) {
        sbcnt[r] = static_cast<int>(_send_counts[r] * bytes_per_elem);
        sboff[r] = static_cast<int>(_send_offsets[r] * bytes_per_elem);
        rbcnt[r] = static_cast<int>(_recv_counts[r] * bytes_per_elem);
        rboff[r] = static_cast<int>(_recv_offsets[r] * bytes_per_elem);
    }

    MPI_Ialltoallv(handle._send_buf.data(), sbcnt.data(), sboff.data(), MPI_BYTE,
                   dst_data,                rbcnt.data(), rboff.data(), MPI_BYTE,
                   _comm, &handle._req);
    handle._completed = false;
    return handle;
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
