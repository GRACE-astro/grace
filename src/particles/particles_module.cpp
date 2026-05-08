/**
 * @file particles_module.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Top-level particle subsystem entry point.
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_utilities.hh>
#include <grace/particles/particles_module.hh>
#include <grace/particles/particle_storage.hh>
#include <grace/particles/particle_advance.hh>
#include <grace/particles/particle_owner_search.hh>
#include <grace/particles/particle_rebalance.hh>
#include <grace/particles/particle_compact.hh>
#include <grace/particles/particle_hdf5_output.hh>
#include <grace/particles/particle_checkpoint.hh>
#include <grace/errors/error.hh>

#include <grace/config/config_parser.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/runtime_functions.hh>
#include <grace/system/print.hh>

#ifdef GRACE_ENABLE_GRMHD
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/amr/amr_functions.hh>
#endif

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <mpi.h>

namespace grace {
namespace particles {

/// Domain-wide BC config for position transforms. Cached at module init.
/// Convention (matches grace::amr): `reflection_*` means the LOW face along
/// that axis is a symmetry plane; the HIGH face is outflow. `periodic_*`
/// applies to both faces along that axis. periodic and reflection on the
/// same axis is invalid; init asserts.
struct position_bcs_t {
    double xlo = 0, ylo = 0, zlo = 0;
    double xhi = 0, yhi = 0, zhi = 0;
    bool   periodic_x = false, periodic_y = false, periodic_z = false;
    bool   reflect_x  = false, reflect_y  = false, reflect_z  = false;
};

class particles_module_impl_t {
  public:
    bool                 enabled                  = false;
    int                  rebalance_every          = 1;
    int                  compact_every            = 0;
    int                  output_every             = 0;
    double               min_quad_width           = 0.0; // refreshed at init + on_regrid
    std::string          rebalance_strategy;
    double               rebalance_imbalance_thr  = 0.05; // strategy-specific; cached at init
    std::string          seeding_mode;
    double               seeding_rho_threshold    = 0.0;
    int                  seeding_oversample       = 10;
    std::string          output_directory;
    std::string          output_basename;
    position_bcs_t       bcs;
    tracer_container_t<> tracers;
};

particles_module_t::particles_module_t()
  : _impl(new particles_module_impl_t())
{}

particles_module_t::~particles_module_t() {
    delete _impl;
}

particles_module_t& particles_module_t::get() {
    static particles_module_t instance;
    return instance;
}

namespace {

/// Deterministic per-rank seed: spread N tracers over the local quads,
/// using a low-discrepancy 3D offset inside each quad's bbox.
/// owner_rank/owner_local_quad are set so the first advance_step runs
/// without needing an immediate refresh.
void seed_local(tracer_container_t<>& tr, std::size_t n_per_rank) {
    auto& sh = fluid_topology_shadow_t::get();
    sh.refresh();
    const auto& geom = sh.local_geometry();
    if (geom.empty() || n_per_rank == 0) {
        tr.resize(0);
        return;
    }
    const int rank = parallel::mpi_comm_rank();

    tr.resize(n_per_rank);
    // After seeding, the next freshly-appended tracer gets local id n_per_rank.
    tr.set_id_counter(static_cast<uint32_t>(n_per_rank));
    auto h_pos        = Kokkos::create_mirror_view(tr.pos);
    auto h_id         = Kokkos::create_mirror_view(tr.id);
    auto h_status     = Kokkos::create_mirror_view(tr.status);
    auto h_owner_rank = Kokkos::create_mirror_view(tr.owner_rank);
    auto h_owner_quad = Kokkos::create_mirror_view(tr.owner_local_quad);

    // Halton-ish offsets in (0,1) along each axis. Avoid edges so we don't
    // land on a quad face and risk dual-ownership ambiguity.
    auto halton = [](std::size_t i, int base) {
        double f = 1.0;
        double r = 0.0;
        std::size_t k = i + 1;
        while (k > 0) {
            f /= base;
            r += f * (k % base);
            k /= base;
        }
        return r;
    };

    for (std::size_t i = 0; i < n_per_rank; ++i) {
        const std::size_t q = i % geom.size();
        const std::size_t s = i / geom.size();
        const auto& g = geom[q];
        const double fx = halton(s, 2);
        const double fy = halton(s, 3);
        const double fz = halton(s, 5);
        h_pos(i, 0) = g.bbox.xlo + fx * (g.bbox.xhi - g.bbox.xlo);
        h_pos(i, 1) = g.bbox.ylo + fy * (g.bbox.yhi - g.bbox.ylo);
        h_pos(i, 2) = g.bbox.zlo + fz * (g.bbox.zhi - g.bbox.zlo);
        // Globally-unique id: (rank << 32) | local_index. Cheap, monotonic,
        // collision-free across ranks at this scale.
        h_id(i)         = (static_cast<uint64_t>(rank) << 32) | static_cast<uint32_t>(i);
        h_status(i)     = 0;
        h_owner_rank(i) = rank;
        h_owner_quad(i) = static_cast<int32_t>(q);
    }
    Kokkos::deep_copy(tr.pos,              h_pos);
    Kokkos::deep_copy(tr.id,               h_id);
    Kokkos::deep_copy(tr.status,           h_status);
    Kokkos::deep_copy(tr.owner_rank,       h_owner_rank);
    Kokkos::deep_copy(tr.owner_local_quad, h_owner_quad);
}

#ifdef GRACE_ENABLE_GRMHD
/// Density-weighted seed: oversample Halton candidates per local quad,
/// trilinear-interp AUX rho at each, accept those above the threshold.
/// Designed for nucleosynthesis post-processing where uniform seeding
/// wastes most tracers in low-density gas. The interp uses cell-centered
/// AUX storage on the host quad — the candidate's quad is known by
/// construction, so no global owner search is needed.
///
/// Best-effort on count: if too few candidates pass the threshold the
/// container ends up shorter than n_per_rank and we log a warning. Tune
/// seeding_oversample_factor or seeding_rho_threshold to recover.
void seed_density_weighted(tracer_container_t<>& tr,
                           std::size_t n_per_rank,
                           double      rho_threshold,
                           std::size_t oversample)
{
    auto& sh = fluid_topology_shadow_t::get();
    sh.refresh();
    const auto& geom = sh.local_geometry();
    if (geom.empty() || n_per_rank == 0) {
        tr.resize(0);
        tr.set_id_counter(0);
        return;
    }
    const int    rank        = parallel::mpi_comm_rank();
    const std::size_t n_cand = n_per_rank * oversample;

    // 1. Generate candidate positions on host (Halton spread across local
    //    quads, same low-discrepancy pattern as the uniform seed).
    auto halton = [](std::size_t i, int base) {
        double f = 1.0;
        double r = 0.0;
        std::size_t k = i + 1;
        while (k > 0) { f /= base; r += f * (k % base); k /= base; }
        return r;
    };
    Kokkos::View<double*[3], grace::default_space>
        cand_pos("seed_cand_pos", n_cand);
    Kokkos::View<int*, grace::default_space>
        cand_quad("seed_cand_quad", n_cand);
    auto h_cand_pos  = Kokkos::create_mirror_view(cand_pos);
    auto h_cand_quad = Kokkos::create_mirror_view(cand_quad);
    for (std::size_t i = 0; i < n_cand; ++i) {
        const std::size_t q = i % geom.size();
        const std::size_t s = i / geom.size();
        const auto& g = geom[q];
        const double fx = halton(s, 2);
        const double fy = halton(s, 3);
        const double fz = halton(s, 5);
        h_cand_pos(i, 0) = g.bbox.xlo + fx * (g.bbox.xhi - g.bbox.xlo);
        h_cand_pos(i, 1) = g.bbox.ylo + fy * (g.bbox.yhi - g.bbox.ylo);
        h_cand_pos(i, 2) = g.bbox.zlo + fz * (g.bbox.zhi - g.bbox.zlo);
        h_cand_quad(i)   = static_cast<int>(q);
    }
    Kokkos::deep_copy(cand_pos,  h_cand_pos);
    Kokkos::deep_copy(cand_quad, h_cand_quad);

    // 2. Stage per-quad geometry on device for the interp kernel: quad
    //    bbox lo + dx_cell. We only need lo + dx; the candidate is by
    //    construction inside the assigned quad's bbox.
    const std::size_t n_q = geom.size();
    Kokkos::View<double*[3], grace::default_space>
        bbox_lo_v("seed_bbox_lo", n_q);
    Kokkos::View<double*, grace::default_space>
        dx_v("seed_dx", n_q);
    auto h_bbox_lo = Kokkos::create_mirror_view(bbox_lo_v);
    auto h_dx      = Kokkos::create_mirror_view(dx_v);
    for (std::size_t q = 0; q < n_q; ++q) {
        h_bbox_lo(q, 0) = geom[q].bbox.xlo;
        h_bbox_lo(q, 1) = geom[q].bbox.ylo;
        h_bbox_lo(q, 2) = geom[q].bbox.zlo;
        h_dx(q)         = geom[q].dx_cell;
    }
    Kokkos::deep_copy(bbox_lo_v, h_bbox_lo);
    Kokkos::deep_copy(dx_v,      h_dx);

    // 3. Trilinear-interp AUX rho at each candidate.
    auto& vars      = grace::variable_list::get();
    auto  aux_view  = vars.getaux();
    const int ngz_  = grace::amr::get_n_ghosts();

    Kokkos::View<double*, grace::default_space>
        cand_rho("seed_cand_rho", n_cand);

    Kokkos::parallel_for("seed_density_weighted_eval",
        Kokkos::RangePolicy<grace::default_execution_space>(0, n_cand),
        KOKKOS_LAMBDA(const int i) {
            const int q = cand_quad(i);
            const double x = cand_pos(i, 0);
            const double y = cand_pos(i, 1);
            const double z = cand_pos(i, 2);
            const double xlo    = bbox_lo_v(q, 0);
            const double ylo    = bbox_lo_v(q, 1);
            const double zlo    = bbox_lo_v(q, 2);
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
            cand_rho(i) =
                  w000 * aux_view(vi  , vj  , vk  , RHO_, q)
                + w100 * aux_view(vi+1, vj  , vk  , RHO_, q)
                + w010 * aux_view(vi  , vj+1, vk  , RHO_, q)
                + w110 * aux_view(vi+1, vj+1, vk  , RHO_, q)
                + w001 * aux_view(vi  , vj  , vk+1, RHO_, q)
                + w101 * aux_view(vi+1, vj  , vk+1, RHO_, q)
                + w011 * aux_view(vi  , vj+1, vk+1, RHO_, q)
                + w111 * aux_view(vi+1, vj+1, vk+1, RHO_, q);
        });
    Kokkos::fence();

    // 4. Host-side accept loop: take the first n_per_rank candidates with
    //    rho >= threshold, in generation order. Generation order has good
    //    spatial spread (Halton across quads), so the accepted subset
    //    inherits that — no second-pass shuffling required.
    auto h_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cand_rho);
    std::vector<std::size_t> accepted_idx;
    accepted_idx.reserve(n_per_rank);
    for (std::size_t i = 0; i < n_cand && accepted_idx.size() < n_per_rank; ++i) {
        if (h_rho(i) >= rho_threshold) accepted_idx.push_back(i);
    }
    const std::size_t n_acc = accepted_idx.size();
    if (n_acc < n_per_rank) {
        GRACE_INFO("Particles: density_weighted seeding accepted only {} of "
                   "{} target tracers on this rank (threshold={:g}, "
                   "oversample={}). Bump seeding_oversample_factor or lower "
                   "seeding_rho_threshold.",
                   n_acc, n_per_rank, rho_threshold, oversample);
    }

    // 5. Populate the tracer container from accepted candidates.
    tr.resize(n_acc);
    tr.set_id_counter(static_cast<uint32_t>(n_acc));
    auto h_pos        = Kokkos::create_mirror_view(tr.pos);
    auto h_id         = Kokkos::create_mirror_view(tr.id);
    auto h_status     = Kokkos::create_mirror_view(tr.status);
    auto h_owner_rank = Kokkos::create_mirror_view(tr.owner_rank);
    auto h_owner_quad = Kokkos::create_mirror_view(tr.owner_local_quad);
    for (std::size_t k = 0; k < n_acc; ++k) {
        const std::size_t i = accepted_idx[k];
        h_pos(k, 0) = h_cand_pos(i, 0);
        h_pos(k, 1) = h_cand_pos(i, 1);
        h_pos(k, 2) = h_cand_pos(i, 2);
        h_id(k)         = (static_cast<uint64_t>(rank) << 32)
                          | static_cast<uint32_t>(k);
        h_status(k)     = 0;
        h_owner_rank(k) = rank;
        h_owner_quad(k) = h_cand_quad(i);
    }
    Kokkos::deep_copy(tr.pos,              h_pos);
    Kokkos::deep_copy(tr.id,               h_id);
    Kokkos::deep_copy(tr.status,           h_status);
    Kokkos::deep_copy(tr.owner_rank,       h_owner_rank);
    Kokkos::deep_copy(tr.owner_local_quad, h_owner_quad);
}
#endif // GRACE_ENABLE_GRMHD

/// Read global domain bounds and BC flags from the parser. Bounds live in
/// amr.{xmin,xmax,ymin,ymax,zmin,zmax} and are authoritative — no Allreduce
/// needed.
position_bcs_t discover_bcs() {
    position_bcs_t b;
    b.xlo = grace::get_param<double>("amr", "xmin");
    b.xhi = grace::get_param<double>("amr", "xmax");
    b.ylo = grace::get_param<double>("amr", "ymin");
    b.yhi = grace::get_param<double>("amr", "ymax");
    b.zlo = grace::get_param<double>("amr", "zmin");
    b.zhi = grace::get_param<double>("amr", "zmax");

    b.periodic_x = grace::get_param<bool>("amr", "periodic_x");
    b.periodic_y = grace::get_param<bool>("amr", "periodic_y");
    b.periodic_z = grace::get_param<bool>("amr", "periodic_z");
    b.reflect_x  = grace::get_param<bool>("amr", "reflection_symmetries", "x");
    b.reflect_y  = grace::get_param<bool>("amr", "reflection_symmetries", "y");
    b.reflect_z  = grace::get_param<bool>("amr", "reflection_symmetries", "z");

    ASSERT(!(b.periodic_x && b.reflect_x),
           "particles: periodic_x and reflection_symmetries.x are mutually exclusive");
    ASSERT(!(b.periodic_y && b.reflect_y),
           "particles: periodic_y and reflection_symmetries.y are mutually exclusive");
    ASSERT(!(b.periodic_z && b.reflect_z),
           "particles: periodic_z and reflection_symmetries.z are mutually exclusive");
    return b;
}

/// Smallest quad-width currently in the global mesh (min over all ranks of
/// each quad's bbox extent / npoints_block). Allreduced. Used as the
/// threshold for the Regime-3 drift assertion: per-step displacement
/// > 0.5 * min_quad_width indicates a tracer crossed an entire (finest)
/// quad in one step, which is incompatible with the fluid CFL and
/// indicates a bug or pathological dt.
double compute_min_quad_width() {
    auto& sh = fluid_topology_shadow_t::get();
    const auto& geom = sh.local_geometry();
    double m = std::numeric_limits<double>::infinity();
    for (const auto& g : geom) {
        m = std::min(m, g.dx_cell);
    }
    MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return m;
}

/// Regime-3 hard error: any tracer whose per-step displacement exceeds
/// half the finest quad-width has either jumped multiple cells (CFL
/// violation), or the dt scaling is broken. Either way, silent corruption
/// is worse than a loud abort.
void assert_no_regime3_drift(const tracer_container_t<>& tr,
                             Kokkos::View<double*[3], grace::default_space> src_pos,
                             double min_quad_width)
{
    // Threshold = 0.5 * min_quad_width per axis. Generous enough that
    // tracers in legitimate Regime 1 (sub-cell drift) never trip.
    const std::size_t n = tr.size();
    const double thr = 0.5 * min_quad_width;
    auto dst_pos = tr.pos;
    int n_violations = 0;
    if (n > 0) {
        Kokkos::parallel_reduce("regime3_drift_check",
            Kokkos::RangePolicy<grace::default_execution_space>(0, n),
            KOKKOS_LAMBDA(const int i, int& acc) {
                const double dx = dst_pos(i, 0) - src_pos(i, 0);
                const double dy = dst_pos(i, 1) - src_pos(i, 1);
                const double dz = dst_pos(i, 2) - src_pos(i, 2);
                if (Kokkos::fabs(dx) > thr || Kokkos::fabs(dy) > thr ||
                    Kokkos::fabs(dz) > thr) ++acc;
            }, n_violations);
    }
    // Allreduce must run on every rank, regardless of local n: a sibling
    // rank with non-empty tr would otherwise hang here.
    int global_violations = 0;
    MPI_Allreduce(&n_violations, &global_violations, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (global_violations > 0) {
        ERROR("Particles: " << global_violations
              << " tracer(s) drifted > 0.5 * min_quad_width="
              << min_quad_width << " in one step. Likely a CFL violation"
              " or buggy dt; aborting before silent corruption.");
    }
}

/// Apply periodic wrap (both faces) or reflection (low face only) per axis,
/// in place. Position-only — for passive tracers no velocity flip is
/// required because the next fetch resamples the fluid v at the post-BC
/// position, and the fluid itself honors the same symmetries by
/// construction. PIC species would need to flip the normal velocity here.
void apply_position_bcs(tracer_container_t<>& tr, const position_bcs_t& b) {
    const std::size_t n = tr.size();
    if (n == 0) return;

    auto pos = tr.pos;
    const double xlo = b.xlo, xhi = b.xhi;
    const double ylo = b.ylo, yhi = b.yhi;
    const double zlo = b.zlo, zhi = b.zhi;
    const double Lx = xhi - xlo, Ly = yhi - ylo, Lz = zhi - zlo;
    const bool px = b.periodic_x, py = b.periodic_y, pz = b.periodic_z;
    const bool rx = b.reflect_x,  ry = b.reflect_y,  rz = b.reflect_z;

    Kokkos::parallel_for("apply_position_bcs",
        Kokkos::RangePolicy<grace::default_execution_space>(0, n),
        KOKKOS_LAMBDA(const int i) {
            // Periodic: wrap into [lo, hi). One CFL step crosses at most one
            // cell, so a single subtract/add suffices — but use a while-style
            // guarded form to stay correct if a future change ever lifts the
            // single-cell guarantee. Compiler folds this for the common case.
            if (px) {
                while (pos(i, 0) >= xhi) pos(i, 0) -= Lx;
                while (pos(i, 0) <  xlo) pos(i, 0) += Lx;
            } else if (rx && pos(i, 0) < xlo) {
                pos(i, 0) = 2.0 * xlo - pos(i, 0);
            }
            if (py) {
                while (pos(i, 1) >= yhi) pos(i, 1) -= Ly;
                while (pos(i, 1) <  ylo) pos(i, 1) += Ly;
            } else if (ry && pos(i, 1) < ylo) {
                pos(i, 1) = 2.0 * ylo - pos(i, 1);
            }
            if (pz) {
                while (pos(i, 2) >= zhi) pos(i, 2) -= Lz;
                while (pos(i, 2) <  zlo) pos(i, 2) += Lz;
            } else if (rz && pos(i, 2) < zlo) {
                pos(i, 2) = 2.0 * zlo - pos(i, 2);
            }
        });
}

/// Per-rank summary of one cull pass. Lets rebalance() Allreduce a single
/// fixed-size payload to print one line of diagnostic. Bounding box of
/// flagged positions distinguishes "drifted off the high face" (genuine
/// outflow) from "scattered through the interior" (find_owner bug).
struct cull_summary_t {
    long long n_flagged    = 0;
    double    bbox_lo[3]   = { std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity() };
    double    bbox_hi[3]   = { -std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity() };
};

/// Mark tracers whose position no longer maps to any rank as
/// PARTICLE_OUTSIDE_DOMAIN before rebalancing. They get dropped by the
/// distributor (export rank -1). Returns a per-rank summary including the
/// bbox of flagged positions, for diagnostic logging in the caller.
cull_summary_t flag_outside_domain(tracer_container_t<>& tr,
                                   Kokkos::View<int*, grace::default_space> export_ranks)
{
    cull_summary_t s;
    const std::size_t n = tr.size();
    if (n == 0) return s;
    auto status = tr.status;
    auto pos    = tr.pos;

    // Single fused reduce: count + bbox of flagged positions in one pass.
    // Kokkos doesn't have a built-in tuple reducer for our payload, so we
    // mirror the relevant slices to the host and do the bookkeeping there.
    // Tracer counts are small (O(N/rank) ~ thousands) so the host pass is
    // cheap; if this ever becomes hot we can rewrite with a custom reducer.
    int n_flagged = 0;
    Kokkos::parallel_reduce("flag_outside_domain_count",
        Kokkos::RangePolicy<grace::default_execution_space>(0, n),
        KOKKOS_LAMBDA(const int i, int& acc) {
            if (export_ranks(i) < 0) {
                status(i) = PARTICLE_OUTSIDE_DOMAIN;
                ++acc;
            }
        }, n_flagged);
    s.n_flagged = static_cast<long long>(n_flagged);

    if (n_flagged > 0) {
        auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pos);
        auto h_xr  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, export_ranks);
        for (std::size_t i = 0; i < n; ++i) {
            if (h_xr(i) < 0) {
                for (int d = 0; d < 3; ++d) {
                    s.bbox_lo[d] = std::min(s.bbox_lo[d], h_pos(i, d));
                    s.bbox_hi[d] = std::max(s.bbox_hi[d], h_pos(i, d));
                }
            }
        }
    }
    return s;
}

void rebalance(tracer_container_t<>& tr, const std::string& strategy,
               double imbalance_threshold)
{
    // No early return on tr.size()==0: decide_rebalance_quad_owner posts an
    // Allreduce and a sibling rank with non-empty tr must not be left
    // waiting alone. The decision struct's views are size-n (size 0 if
    // tr is empty) — all subsequent kernels are safe on empty inputs.
    rebalance_decision_t decision;
    if (strategy == "quad_owner") {
        decision = decide_rebalance_quad_owner(tr, imbalance_threshold);
    } else {
        ERROR("Unknown particle rebalance strategy: " << strategy);
    }

    const cull_summary_t local = flag_outside_domain(tr, decision.export_ranks);

    // Diagnostic: aggregate cull count + global bbox of culled positions.
    // The bbox lets us see WHERE in the domain culls occur — concentrated
    // on a face means genuine outflow; scattered through the interior
    // means find_owner is rejecting valid positions.
    long long global_culled = 0;
    double    global_lo[3], global_hi[3];
    double    neg_local_lo[3] = { -local.bbox_lo[0], -local.bbox_lo[1], -local.bbox_lo[2] };
    double    neg_global_lo[3];
    MPI_Allreduce(&local.n_flagged, &global_culled, 1, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(neg_local_lo,    neg_global_lo, 3, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local.bbox_hi,   global_hi,     3, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    for (int d = 0; d < 3; ++d) global_lo[d] = -neg_global_lo[d];

    if (global_culled > 0) {
        GRACE_INFO("Particles: rebalance culled {} tracer(s) globally "
                   "(out-of-domain). Culled bbox: x=[{:g},{:g}] y=[{:g},{:g}] z=[{:g},{:g}].",
                   global_culled,
                   global_lo[0], global_hi[0],
                   global_lo[1], global_hi[1],
                   global_lo[2], global_hi[2]);
    }

    // Always refresh routing so subsequent fetches go to the live owner,
    // even when we skip the data shuffle. Local kernel; no MPI.
    update_owner_only(tr, decision);

    if (decision.should_migrate) {
        migrate_topology(MPI_COMM_WORLD, tr, decision.export_ranks);
        GRACE_VERBOSE("Particles: migrated (imbalance={:.3f} > threshold).",
                      decision.imbalance);
    } else {
        GRACE_VERBOSE("Particles: skipped migration (imbalance={:.3f} "
                      "below threshold).", decision.imbalance);
    }
}

} // namespace

void particles_module_t::initialize(hid_t restore_file_id) {
    _impl->enabled = grace::get_param<bool>("particles", "enabled");
    if (!_impl->enabled) {
        GRACE_INFO("Particle subsystem disabled by config.");
        return;
    }
    _impl->rebalance_strategy =
        grace::get_param<std::string>("particles", "rebalance_strategy");
    if (_impl->rebalance_strategy == "quad_owner") {
        _impl->rebalance_imbalance_thr = grace::get_param<double>(
            "particles", "quad_owner", "imbalance_threshold");
    } else {
        ERROR("Unknown particles.rebalance_strategy: "
              << _impl->rebalance_strategy);
    }
    _impl->rebalance_every =
        grace::get_param<int>("particles", "rebalance_every");
    _impl->compact_every =
        grace::get_param<int>("particles", "compact_every");
    _impl->output_every =
        grace::get_param<int>("particles", "output_every");
    _impl->output_directory =
        grace::get_param<std::string>("particles", "output_directory");
    _impl->output_basename =
        grace::get_param<std::string>("particles", "output_basename");
    _impl->seeding_mode =
        grace::get_param<std::string>("particles", "seeding_mode");
    _impl->seeding_rho_threshold =
        grace::get_param<double>("particles", "seeding_rho_threshold");
    _impl->seeding_oversample =
        grace::get_param<int>("particles", "seeding_oversample_factor");

    _impl->bcs = discover_bcs();
    _impl->min_quad_width = compute_min_quad_width();

    bool restored = false;
    if (restore_file_id >= 0) {
        // Restart path: replace would-be seed contents with checkpointed
        // tracers. load_particles_from_checkpoint refreshes the shadow and
        // rebalances under the current partition. Returns false on
        // pre-particle-subsystem checkpoints — fall back to seeding.
        restored = load_particles_from_checkpoint(restore_file_id);
    }
    if (!restored) {
        // Fresh-start path: dispatch on seeding_mode.
        const int n_per_rank =
            grace::get_param<int>("particles", "n_tracers_per_rank");
        if (_impl->seeding_mode == "uniform") {
            seed_local(_impl->tracers, static_cast<std::size_t>(n_per_rank));
        } else if (_impl->seeding_mode == "density_weighted") {
#ifdef GRACE_ENABLE_GRMHD
            seed_density_weighted(
                _impl->tracers,
                static_cast<std::size_t>(n_per_rank),
                _impl->seeding_rho_threshold,
                static_cast<std::size_t>(_impl->seeding_oversample));
#else
            ERROR("particles: seeding_mode=density_weighted requires "
                  "GRACE_ENABLE_GRMHD (rho is a GRMHD primitive).");
#endif
        } else {
            ERROR("Unknown particles.seeding_mode: " << _impl->seeding_mode);
        }
    }

    GRACE_INFO("Particle subsystem enabled: {} tracers/rank seeded "
               "(mode={}); strategy={}, imbalance_threshold={:.3f}; "
               "domain [{:g},{:g}]x[{:g},{:g}]x[{:g},{:g}], "
               "periodic {}{}{} reflect {}{}{}.",
               _impl->tracers.size(),
               _impl->seeding_mode,
               _impl->rebalance_strategy,
               _impl->rebalance_imbalance_thr,
               _impl->bcs.xlo, _impl->bcs.xhi,
               _impl->bcs.ylo, _impl->bcs.yhi,
               _impl->bcs.zlo, _impl->bcs.zhi,
               _impl->bcs.periodic_x ? "x" : "-",
               _impl->bcs.periodic_y ? "y" : "-",
               _impl->bcs.periodic_z ? "z" : "-",
               _impl->bcs.reflect_x  ? "x" : "-",
               _impl->bcs.reflect_y  ? "y" : "-",
               _impl->bcs.reflect_z  ? "z" : "-");
}

void particles_module_t::finalize() {
    _impl->tracers = tracer_container_t<>{};
    _impl->enabled = false;
}

void particles_module_t::on_regrid() {
    if (!_impl->enabled) return;
    // Shadow must refresh even if there are zero tracers, so that subsequent
    // seeding / append still reads valid geometry.
    fluid_topology_shadow_t::get().refresh();
    _impl->min_quad_width = compute_min_quad_width();

    auto& tr = _impl->tracers;
    // No early return on tr.size()==0: rebalance is collective. A sibling
    // rank may have surviving tracers and would hang waiting for us.
    rebalance(tr, _impl->rebalance_strategy, _impl->rebalance_imbalance_thr);
    // Samples are stale-or-zero post-rebalance; the next advance_step fetches
    // fresh ones at the top of the next evolve() call, before any tracer
    // output fires (output is gated inside advance_step, after the fetch).
}

bool particles_module_t::enabled() const noexcept {
    return _impl->enabled;
}

std::size_t particles_module_t::local_count() const noexcept {
    return _impl->tracers.size();
}

tracer_container_t<>& particles_module_t::tracers() noexcept {
    return _impl->tracers;
}

const tracer_container_t<>& particles_module_t::tracers() const noexcept {
    return _impl->tracers;
}

void particles_module_t::advance_step(double dt) {
    if (!_impl->enabled) return;
    auto& tr = _impl->tracers;

    const size_t iter = grace::get_iteration();

    // Rebalance: re-resolve ownership and migrate data to new owners.
    // Without this, tracers that drift into other ranks' quads still fetch
    // correctly (owner_rank gets refreshed inside compute_export_ranks_*),
    // but the data lives on the wrong rank — bad load balance once the fluid
    // clusters. Subsumes the old refresh_owners path. Schedule decision is
    // uniform across ranks (iter is global), so this is collective-safe.
    if (_impl->rebalance_every > 0 &&
        iter % static_cast<size_t>(_impl->rebalance_every) == 0) {
        rebalance(tr, _impl->rebalance_strategy, _impl->rebalance_imbalance_thr);
    }

    // Compaction: drop dead tracers (OUTSIDE_DOMAIN, etc.). Local-only — no
    // collectives — so a per-rank tr.size()==0 short-circuit is fine here.
    if (tr.size() > 0 && _impl->compact_every > 0 &&
        iter % static_cast<size_t>(_impl->compact_every) == 0) {
        const std::size_t culled = compact(tr);
        if (culled > 0) {
            GRACE_VERBOSE("Particles: compacted {} dead tracers (iter {}).",
                          culled, iter);
        }
    }

    // Snapshot pre-push positions for the Regime-3 drift assertion. Cheap
    // — one device-side copy — and lets us catch CFL violations or buggy
    // dt scaling before they corrupt trajectories silently. Allocating a
    // size-0 view is fine; needed because advance_substep is collective and
    // every rank must enter it whether or not it owns tracers.
    Kokkos::View<double*[3], grace::default_space>
        src_pos_snapshot("regime3_src_pos", tr.size());
    if (tr.size() > 0) {
        Kokkos::deep_copy(src_pos_snapshot, tr.pos);
    }

    // Collective: fluid fetch + RHS push. Must be entered by every rank.
    advance_substep(MPI_COMM_WORLD, dt, /*dtfact=*/1.0, tr.pos, tr.pos, tr);

    // Collective: contains an Allreduce.
    assert_no_regime3_drift(tr, src_pos_snapshot, _impl->min_quad_width);

    // Apply position-only BCs after the push so the next step's rebalance
    // sees in-domain positions. Periodic axes wrap; reflection axes mirror
    // the low face. Outflow tracers (genuine domain exit on a non-periodic,
    // non-reflecting face) are left as-is for the next rebalance to flag.
    apply_position_bcs(tr, _impl->bcs);

    if (_impl->output_every > 0 &&
        iter % static_cast<size_t>(_impl->output_every) == 0) {
        // Collective MPI-IO; every rank participates regardless of local
        // tr.size(). The schedule check uses the global iter so it's uniform.
        write_particle_snapshot(tr, _impl->output_directory,
                                _impl->output_basename);
    }
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
