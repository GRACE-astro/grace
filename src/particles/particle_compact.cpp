/**
 * @file particle_compact.cpp
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_compact.hh>
#include <grace/particles/particle_utilities.hh>

#include <Kokkos_Core.hpp>

#include <cstdint>

namespace grace {
namespace particles {

std::size_t compact(tracer_container_t<>& tr)
{
    const std::size_t n = tr.size();
    if (n == 0) return 0;

    // Scan: write_idx[i] = number of alive (PARTICLE_DEFAULT) particles
    // strictly before i. After the scan, total_alive holds the count.
    Kokkos::View<int*, grace::default_space> write_idx("compact_write_idx", n);
    auto status = tr.status;
    int total_alive = 0;
    Kokkos::parallel_scan("compact_scan",
        Kokkos::RangePolicy<grace::default_execution_space>(0, n),
        KOKKOS_LAMBDA(const int i, int& acc, const bool final) {
            if (final) write_idx(i) = acc;
            if (status(i) == PARTICLE_DEFAULT) ++acc;
        }, total_alive);

    if (total_alive == static_cast<int>(n)) return 0;

    // Allocate compacted container. Preserve id counter — compaction does
    // NOT make freed ids available for reuse (would break global uniqueness).
    tracer_container_t<> nu;
    nu.resize(static_cast<std::size_t>(total_alive));
    nu.set_id_counter(tr.id_counter());

    // Capture each field individually rather than the whole container by
    // value — the container has 16 views and capturing by-struct in a
    // Kokkos::parallel_for lambda risks bloating the device kernel arg list.
    auto src_pos      = tr.pos;
    auto src_id       = tr.id;
    auto src_status   = tr.status;
    auto src_or       = tr.owner_rank;
    auto src_oq       = tr.owner_local_quad;
    auto src_alpha    = tr.sample_alpha;
    auto src_beta     = tr.sample_beta;
    auto src_v        = tr.sample_v;
    auto src_W        = tr.sample_W;
    auto src_rho      = tr.sample_rho;
    auto src_temp     = tr.sample_temp;
    auto src_ye       = tr.sample_ye;
    auto src_entropy  = tr.sample_entropy;
    auto src_press    = tr.sample_press;
    auto src_eps      = tr.sample_eps;
    auto src_B        = tr.sample_B;
    auto dst_pos      = nu.pos;
    auto dst_id       = nu.id;
    auto dst_status   = nu.status;
    auto dst_or       = nu.owner_rank;
    auto dst_oq       = nu.owner_local_quad;
    auto dst_alpha    = nu.sample_alpha;
    auto dst_beta     = nu.sample_beta;
    auto dst_v        = nu.sample_v;
    auto dst_W        = nu.sample_W;
    auto dst_rho      = nu.sample_rho;
    auto dst_temp     = nu.sample_temp;
    auto dst_ye       = nu.sample_ye;
    auto dst_entropy  = nu.sample_entropy;
    auto dst_press    = nu.sample_press;
    auto dst_eps      = nu.sample_eps;
    auto dst_B        = nu.sample_B;

    Kokkos::parallel_for("compact_gather",
        Kokkos::RangePolicy<grace::default_execution_space>(0, n),
        KOKKOS_LAMBDA(const int i) {
            if (src_status(i) != PARTICLE_DEFAULT) return;
            const int j = write_idx(i);
            for (int d = 0; d < 3; ++d) dst_pos(j, d)  = src_pos(i, d);
            for (int d = 0; d < 3; ++d) dst_beta(j, d) = src_beta(i, d);
            for (int d = 0; d < 3; ++d) dst_v(j, d)    = src_v(i, d);
            for (int d = 0; d < 3; ++d) dst_B(j, d)    = src_B(i, d);
            dst_id(j)       = src_id(i);
            dst_status(j)   = src_status(i);
            dst_or(j)       = src_or(i);
            dst_oq(j)       = src_oq(i);
            dst_alpha(j)    = src_alpha(i);
            dst_W(j)        = src_W(i);
            dst_rho(j)      = src_rho(i);
            dst_temp(j)     = src_temp(i);
            dst_ye(j)       = src_ye(i);
            dst_entropy(j)  = src_entropy(i);
            dst_press(j)    = src_press(i);
            dst_eps(j)      = src_eps(i);
        });
    Kokkos::fence();

    const std::size_t culled = n - static_cast<std::size_t>(total_alive);
    tr = std::move(nu);
    return culled;
}

void append(tracer_container_t<>&                      tr,
            int                                        rank,
            Kokkos::View<double*[3], Kokkos::HostSpace> new_positions)
{
    const std::size_t n_old = tr.size();
    const std::size_t n_new = new_positions.extent(0);
    if (n_new == 0) return;

    // Reserve fresh local ids before the resize so a failure here doesn't
    // leave the container in a half-grown state.
    const uint32_t id_base = tr.next_id_range(static_cast<uint32_t>(n_new));
    tr.resize_preserving(n_old + n_new);

    auto h_pos      = Kokkos::create_mirror_view(tr.pos);
    auto h_id       = Kokkos::create_mirror_view(tr.id);
    auto h_status   = Kokkos::create_mirror_view(tr.status);
    auto h_or       = Kokkos::create_mirror_view(tr.owner_rank);
    auto h_oq       = Kokkos::create_mirror_view(tr.owner_local_quad);
    Kokkos::deep_copy(h_pos,    tr.pos);
    Kokkos::deep_copy(h_id,     tr.id);
    Kokkos::deep_copy(h_status, tr.status);
    Kokkos::deep_copy(h_or,     tr.owner_rank);
    Kokkos::deep_copy(h_oq,     tr.owner_local_quad);

    for (std::size_t k = 0; k < n_new; ++k) {
        const std::size_t i = n_old + k;
        h_pos(i, 0) = new_positions(k, 0);
        h_pos(i, 1) = new_positions(k, 1);
        h_pos(i, 2) = new_positions(k, 2);
        h_id(i)     = (static_cast<uint64_t>(rank) << 32) |
                      static_cast<uint64_t>(id_base + k);
        h_status(i) = PARTICLE_DEFAULT;
        h_or(i)     = rank;
        h_oq(i)     = -1;
    }

    Kokkos::deep_copy(tr.pos,              h_pos);
    Kokkos::deep_copy(tr.id,               h_id);
    Kokkos::deep_copy(tr.status,           h_status);
    Kokkos::deep_copy(tr.owner_rank,       h_or);
    Kokkos::deep_copy(tr.owner_local_quad, h_oq);
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
