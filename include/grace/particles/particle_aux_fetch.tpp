/**
 * @file particle_aux_fetch.tpp
 * @brief Templated implementation of fetch_at_positions().
 *
 * Included from particle_aux_fetch.hh — do not include directly.
 */
#ifndef GRACE_PARTICLES_PARTICLE_AUX_FETCH_TPP
#define GRACE_PARTICLES_PARTICLE_AUX_FETCH_TPP

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_distributor.hh>
#include <grace/particles/particle_owner_search.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/utils/grace_utils.hh>

#include <Kokkos_Core.hpp>

namespace grace {
namespace particles {

namespace detail {

// Wire-format request POD. Trivially copyable, sized for tight pack.
struct alignas(8) fetch_request_t {
    int    orig_rank;
    int    orig_idx;
    int    quad_local;
    int    _pad;
    double pos[3];
};
static_assert(std::is_trivially_copyable_v<fetch_request_t>);

// Wire-format response POD, parametrized by field count.
template <int N_FIELDS>
struct alignas(8) fetch_response_t {
    int    orig_rank;
    int    orig_idx;
    double values[N_FIELDS];
};

struct quad_geom_dev_t {
    Kokkos::View<double*[6], grace::default_space> bbox;
    Kokkos::View<double*,    grace::default_space> dx;
};

inline quad_geom_dev_t copy_geom_to_device()
{
    auto& sh = fluid_topology_shadow_t::get();
    const auto& geom = sh.local_geometry();
    int n_local = static_cast<int>(geom.size());

    quad_geom_dev_t out;
    out.bbox = Kokkos::View<double*[6], grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_bbox"), n_local);
    out.dx = Kokkos::View<double*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_dx"), n_local);

    auto h_bbox = Kokkos::create_mirror_view(out.bbox);
    auto h_dx   = Kokkos::create_mirror_view(out.dx);
    for (int q = 0; q < n_local; ++q) {
        h_bbox(q, 0) = geom[q].bbox.xlo;
        h_bbox(q, 1) = geom[q].bbox.ylo;
        h_bbox(q, 2) = geom[q].bbox.zlo;
        h_bbox(q, 3) = geom[q].bbox.xhi;
        h_bbox(q, 4) = geom[q].bbox.yhi;
        h_bbox(q, 5) = geom[q].bbox.zhi;
        h_dx(q)      = geom[q].dx_cell;
    }
    Kokkos::deep_copy(out.bbox, h_bbox);
    Kokkos::deep_copy(out.dx,   h_dx);
    return out;
}

template <int N_FIELDS>
struct field_specs_dev_t {
    Kokkos::View<int*, grace::default_space> source;
    Kokkos::View<int*, grace::default_space> idx;
};

template <int N_FIELDS>
inline field_specs_dev_t<N_FIELDS> copy_field_specs_to_device(
    const std::array<fetch_field_spec_t, N_FIELDS>& fields)
{
    field_specs_dev_t<N_FIELDS> out;
    out.source = Kokkos::View<int*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_field_source"),
        N_FIELDS);
    out.idx = Kokkos::View<int*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_field_idx"),
        N_FIELDS);
    auto h_source = Kokkos::create_mirror_view(out.source);
    auto h_idx    = Kokkos::create_mirror_view(out.idx);
    for (int f = 0; f < N_FIELDS; ++f) {
        h_source(f) = static_cast<int>(fields[f].source);
        h_idx(f)    = fields[f].var_idx;
    }
    Kokkos::deep_copy(out.source, h_source);
    Kokkos::deep_copy(out.idx,    h_idx);
    return out;
}

} // namespace detail

template <int N_FIELDS>
void fetch_at_positions(
    MPI_Comm comm,
    std::size_t n_particles,
    const std::array<fetch_field_spec_t, N_FIELDS>& fields,
    Kokkos::View<int*,         grace::default_space> owner_rank,
    Kokkos::View<int*,         grace::default_space> owner_local_quad,
    Kokkos::View<double*[3],   grace::default_space> positions,
    Kokkos::View<double**,     grace::default_space> out)
{
    using exec_space   = typename grace::default_space::execution_space;
    using request_t    = detail::fetch_request_t;
    using response_t   = detail::fetch_response_t<N_FIELDS>;

    int my_rank = -1;
    MPI_Comm_rank(comm, &my_rank);

    // -------------------------------------------------------------------
    // 1. Pack forward request buffer + export-rank vector.
    // -------------------------------------------------------------------
    Kokkos::View<request_t*, grace::default_space> req(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_req"),
        n_particles);
    Kokkos::View<int*, grace::default_space> export_ranks(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_export"),
        n_particles);

    Kokkos::parallel_for("particle_fetch_pack_request",
        Kokkos::RangePolicy<exec_space>(0, n_particles),
        KOKKOS_LAMBDA(const int i) {
            request_t r;
            r.orig_rank = my_rank;
            r.orig_idx  = i;
            r._pad      = 0;
            r.quad_local = owner_local_quad(i);
            r.pos[0] = positions(i, 0);
            r.pos[1] = positions(i, 1);
            r.pos[2] = positions(i, 2);
            req(i) = r;
            export_ranks(i) = owner_rank(i); // -1 → distributor drops the elem
        });
    Kokkos::fence();

    // -------------------------------------------------------------------
    // 2. Forward distributor: ship requests to fluid owners.
    // -------------------------------------------------------------------
    distribution_plan_t fwd(comm, export_ranks);
    Kokkos::View<request_t*, grace::default_space> req_recv(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_req_recv"),
        fwd.total_num_import());
    migrate(fwd, req, req_recv);

    // -------------------------------------------------------------------
    // 3. Owner-side fulfillment: trilinear interp per requested field.
    // -------------------------------------------------------------------
    const std::size_t n_recv = req_recv.size();
    Kokkos::View<response_t*, grace::default_space> resp(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_resp"),
        n_recv);

    auto& vars = grace::variable_list::get();
    auto state_view = vars.getstate();
    auto aux_view   = vars.getaux();
    const int ngz_  = grace::amr::get_n_ghosts();

    auto geom_dev   = detail::copy_geom_to_device();
    auto specs_dev  = detail::copy_field_specs_to_device<N_FIELDS>(fields);

    auto bbox_v   = geom_dev.bbox;
    auto dx_v     = geom_dev.dx;
    auto src_v    = specs_dev.source;
    auto idx_v    = specs_dev.idx;
    const int n_local_quads = static_cast<int>(bbox_v.extent(0));

    Kokkos::parallel_for("particle_fetch_fulfill",
        Kokkos::RangePolicy<exec_space>(0, n_recv),
        KOKKOS_LAMBDA(const int r) {
            request_t  in  = req_recv(r);
            response_t out_r;
            out_r.orig_rank = in.orig_rank;
            out_r.orig_idx  = in.orig_idx;

            const double x = in.pos[0];
            const double y = in.pos[1];
            const double z = in.pos[2];

            // Owner-side local-quad resolution. The requester only knows the
            // owner_rank; the local_quad index is meaningful only on the owner.
            // Trust the requester's `quad_local` hint if it actually contains
            // the point (e.g. self-fetches where quad_local was cached); else
            // brute-force scan through the local bbox table. v1 linear scan;
            // optimize when N_local_quads × N_recv becomes a profile spike.
            int q = in.quad_local;
            bool valid = (q >= 0 && q < n_local_quads
                && x >= bbox_v(q, 0) && x < bbox_v(q, 3)
                && y >= bbox_v(q, 1) && y < bbox_v(q, 4)
                && z >= bbox_v(q, 2) && z < bbox_v(q, 5));
            if (!valid) {
                q = -1;
                for (int qi = 0; qi < n_local_quads; ++qi) {
                    if (x >= bbox_v(qi, 0) && x < bbox_v(qi, 3) &&
                        y >= bbox_v(qi, 1) && y < bbox_v(qi, 4) &&
                        z >= bbox_v(qi, 2) && z < bbox_v(qi, 5)) {
                        q = qi;
                        break;
                    }
                }
            }
            if (q < 0) {
                for (int f = 0; f < N_FIELDS; ++f) out_r.values[f] = 0.0;
                resp(r) = out_r;
                return;
            }
            const double xlo    = bbox_v(q, 0);
            const double ylo    = bbox_v(q, 1);
            const double zlo    = bbox_v(q, 2);
            const double inv_dx = 1.0 / dx_v(q);

            const double cx = (x - xlo) * inv_dx - 0.5;
            const double cy = (y - ylo) * inv_dx - 0.5;
            const double cz = (z - zlo) * inv_dx - 0.5;
            const int    bi = static_cast<int>(Kokkos::floor(cx));
            const int    bj = static_cast<int>(Kokkos::floor(cy));
            const int    bk = static_cast<int>(Kokkos::floor(cz));
            const double fx = cx - static_cast<double>(bi);
            const double fy = cy - static_cast<double>(bj);
            const double fz = cz - static_cast<double>(bk);

            // Interior coords [0, nx) → view coords [ngz, ngz+nx).
            const int vi = bi + ngz_;
            const int vj = bj + ngz_;
            const int vk = bk + ngz_;

            const double w000 = (1.0-fx)*(1.0-fy)*(1.0-fz);
            const double w100 =      fx *(1.0-fy)*(1.0-fz);
            const double w010 = (1.0-fx)*     fy *(1.0-fz);
            const double w110 =      fx *     fy *(1.0-fz);
            const double w001 = (1.0-fx)*(1.0-fy)*     fz ;
            const double w101 =      fx *(1.0-fy)*     fz ;
            const double w011 = (1.0-fx)*     fy *     fz ;
            const double w111 =      fx *     fy *     fz ;

            // Local trilinear interp helper. Inlined per call site below; the
            // 8 view reads at fixed (vidx, q) are all stride-1 in i.
            auto trilin_state = [&](int vidx) -> double {
                return  w000 * state_view(vi  , vj  , vk  , vidx, q)
                      + w100 * state_view(vi+1, vj  , vk  , vidx, q)
                      + w010 * state_view(vi  , vj+1, vk  , vidx, q)
                      + w110 * state_view(vi+1, vj+1, vk  , vidx, q)
                      + w001 * state_view(vi  , vj  , vk+1, vidx, q)
                      + w101 * state_view(vi+1, vj  , vk+1, vidx, q)
                      + w011 * state_view(vi  , vj+1, vk+1, vidx, q)
                      + w111 * state_view(vi+1, vj+1, vk+1, vidx, q);
            };
            auto trilin_aux = [&](int vidx) -> double {
                return  w000 * aux_view(vi  , vj  , vk  , vidx, q)
                      + w100 * aux_view(vi+1, vj  , vk  , vidx, q)
                      + w010 * aux_view(vi  , vj+1, vk  , vidx, q)
                      + w110 * aux_view(vi+1, vj+1, vk  , vidx, q)
                      + w001 * aux_view(vi  , vj  , vk+1, vidx, q)
                      + w101 * aux_view(vi+1, vj  , vk+1, vidx, q)
                      + w011 * aux_view(vi  , vj+1, vk+1, vidx, q)
                      + w111 * aux_view(vi+1, vj+1, vk+1, vidx, q);
            };

            // Lazy derived-field cache: computed once per request when the
            // first DERIVED_* field is encountered, reused by the others.
            // Computes v^i = Z^i / W where W = sqrt(1 + γ_ij Z^i Z^j).
            bool   have_derived = false;
            double dvx = 0.0, dvy = 0.0, dvz = 0.0, dW = 1.0;
            auto ensure_derived = [&]() {
                if (have_derived) return;
                have_derived = true;
                const double Zx = trilin_aux(ZVECX_);
                const double Zy = trilin_aux(ZVECY_);
                const double Zz = trilin_aux(ZVECZ_);
#ifdef GRACE_ENABLE_COWLING_METRIC
                // γ_ij is in the state array directly.
                const double gxx = trilin_state(GXX_);
                const double gxy = trilin_state(GXY_);
                const double gxz = trilin_state(GXZ_);
                const double gyy = trilin_state(GYY_);
                const double gyz = trilin_state(GYZ_);
                const double gzz = trilin_state(GZZ_);
#elif defined(GRACE_ENABLE_Z4C_METRIC) || defined(GRACE_ENABLE_BSSN_METRIC)
                // γ_ij = γ̃_ij / χ.
                const double chi  = trilin_state(CHI_);
                const double inv_chi = (chi > 0.0) ? 1.0 / chi : 1.0;
                const double gxx = trilin_state(GTXX_) * inv_chi;
                const double gxy = trilin_state(GTXY_) * inv_chi;
                const double gxz = trilin_state(GTXZ_) * inv_chi;
                const double gyy = trilin_state(GTYY_) * inv_chi;
                const double gyz = trilin_state(GTYZ_) * inv_chi;
                const double gzz = trilin_state(GTZZ_) * inv_chi;
#else
                // No 3+1 metric available — assume flat γ_ij = δ_ij.
                const double gxx = 1.0, gyy = 1.0, gzz = 1.0;
                const double gxy = 0.0, gxz = 0.0, gyz = 0.0;
#endif
                const double Z2 = gxx*Zx*Zx + gyy*Zy*Zy + gzz*Zz*Zz
                                + 2.0*(gxy*Zx*Zy + gxz*Zx*Zz + gyz*Zy*Zz);
                dW = Kokkos::sqrt(1.0 + Z2);
                const double inv_W = 1.0 / dW;
                dvx = Zx * inv_W;
                dvy = Zy * inv_W;
                dvz = Zz * inv_W;
            };

            for (int f = 0; f < N_FIELDS; ++f) {
                const int src  = src_v(f);
                const int vidx = idx_v(f);
                double v;
                switch (src) {
                    case 0: v = trilin_state(vidx); break;
                    case 1: v = trilin_aux  (vidx); break;
                    case 2: ensure_derived(); v = dvx; break;
                    case 3: ensure_derived(); v = dvy; break;
                    case 4: ensure_derived(); v = dvz; break;
                    case 5: ensure_derived(); v = dW;  break;
                    default: v = 0.0; break;
                }
                out_r.values[f] = v;
            }
            resp(r) = out_r;
        });
    Kokkos::fence();

    // -------------------------------------------------------------------
    // 4. Reverse distributor: ship responses back to originating ranks.
    // -------------------------------------------------------------------
    Kokkos::View<int*, grace::default_space> resp_export(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_resp_export"),
        n_recv);
    Kokkos::parallel_for("particle_fetch_build_resp_export",
        Kokkos::RangePolicy<exec_space>(0, n_recv),
        KOKKOS_LAMBDA(const int r) {
            resp_export(r) = resp(r).orig_rank;
        });
    Kokkos::fence();

    distribution_plan_t rev(comm, resp_export);
    Kokkos::View<response_t*, grace::default_space> resp_recv(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_resp_recv"),
        rev.total_num_import());
    migrate(rev, resp, resp_recv);

    // -------------------------------------------------------------------
    // 5. Write back into the caller's output view, indexed by orig_idx.
    // -------------------------------------------------------------------
    const std::size_t n_back = resp_recv.size();
    Kokkos::parallel_for("particle_fetch_writeback",
        Kokkos::RangePolicy<exec_space>(0, n_back),
        KOKKOS_LAMBDA(const int r) {
            response_t in_r = resp_recv(r);
            for (int f = 0; f < N_FIELDS; ++f) {
                out(in_r.orig_idx, f) = in_r.values[f];
            }
        });
    Kokkos::fence();
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_AUX_FETCH_TPP
