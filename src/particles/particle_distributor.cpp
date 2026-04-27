/**
 * @file particle_distributor.cpp
 * @brief Implementation of distribution_plan_t.
 *
 * Three steps:
 *   1. Tally per-destination-rank counts on this rank.
 *   2. MPI_Alltoall to learn how many elements each rank will send to us.
 *   3. Build a per-input permutation that places each element into the
 *      correct slot of the (eventually packed) send buffer in destination-
 *      rank order, so that MPI_Alltoallv can ship contiguous slabs per peer.
 *
 * The permutation lives on device. The actual pack kernel runs in
 * migrate_raw() per migrate call (one byte-memcpy parallel_for, then one
 * MPI_Alltoallv).
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_distributor.hh>
#include <grace/utils/grace_utils.hh>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

namespace grace {
namespace particles {

distribution_plan_t::distribution_plan_t(
    MPI_Comm comm,
    Kokkos::View<int*, grace::default_space> export_ranks)
    : _comm(comm)
    , _n_input(export_ranks.size())
{
    MPI_Comm_size(comm, &_nproc);
    MPI_Comm_rank(comm, &_rank);

    // ---------------------------------------------------------
    // 1. Per-destination counts. Done host-side: ranks vector is short
    //    (size = _n_input), and we need to host-reduce to _nproc bins.
    // ---------------------------------------------------------
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

    // ---------------------------------------------------------
    // 2. Per-source counts via MPI_Alltoall.
    // ---------------------------------------------------------
    _recv_counts.assign(_nproc, 0);
    MPI_Alltoall(_send_counts.data(), 1, MPI_INT,
                 _recv_counts.data(), 1, MPI_INT, comm);

    _recv_offsets.assign(_nproc, 0);
    for (int r = 1; r < _nproc; ++r)
        _recv_offsets[r] = _recv_offsets[r - 1] + _recv_counts[r - 1];
    _total_recv = static_cast<std::size_t>(_recv_offsets[_nproc - 1]
                                          + _recv_counts[_nproc - 1]);

    // ---------------------------------------------------------
    // 3. Build per-input-element send-buffer slot permutation. We track a
    //    running cursor per-destination on host (small, _nproc ints) and
    //    write the per-input slot index into _send_perm.
    // ---------------------------------------------------------
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

void distribution_plan_t::migrate_raw(const void* src_data,
                                      std::size_t bytes_per_elem,
                                      void* dst_data) const
{
    using exec_space = typename grace::default_space::execution_space;

    // Allocate the send buffer (device-resident).
    Kokkos::View<char*, grace::default_space> send_buf(
        Kokkos::ViewAllocateWithoutInitializing("particle_distrib_send_buf"),
        _total_send * bytes_per_elem);

    // Pack: for each input element with valid destination, copy bytes into the
    // destination slot of the send buffer. byte-memcpy is fine on device for
    // small element sizes (~tens of bytes).
    auto perm = _send_perm;
    const std::size_t n_in = _n_input;
    const char* src_bytes = static_cast<const char*>(src_data);

    Kokkos::View<int*, grace::default_space> perm_v = perm;
    Kokkos::View<char*, grace::default_space> sbuf_v = send_buf;

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
    Kokkos::fence();

    // Per-rank byte-counts/offsets for MPI_Alltoallv.
    std::vector<int> sbcnt(_nproc), sboff(_nproc);
    std::vector<int> rbcnt(_nproc), rboff(_nproc);
    for (int r = 0; r < _nproc; ++r) {
        sbcnt[r] = static_cast<int>(_send_counts[r] * bytes_per_elem);
        sboff[r] = static_cast<int>(_send_offsets[r] * bytes_per_elem);
        rbcnt[r] = static_cast<int>(_recv_counts[r] * bytes_per_elem);
        rboff[r] = static_cast<int>(_recv_offsets[r] * bytes_per_elem);
    }

    // GPU-aware MPI: pass device pointers directly. Same path as fluid halos.
    MPI_Alltoallv(send_buf.data(), sbcnt.data(), sboff.data(), MPI_BYTE,
                  dst_data,        rbcnt.data(), rboff.data(), MPI_BYTE,
                  _comm);
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
