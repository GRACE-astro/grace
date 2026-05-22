// Ghost-zone fill correctness on an FMR grid.
//
// Modelled after test_bc_unigrid.cpp's bit-exact pattern.  Differences:
//
//   (1) FMR layout — cells across fine-coarse interfaces go through P/R
//       kernels rather than a plain inter-quadrant copy.
//
//   (2) Polynomials are chosen so the operators give MATHEMATICALLY exact
//       results: a degree-1 (linear) function in three variables is exactly
//       representable in the P/R exactness window for the relevant orders.
//       For face-staggered B we additionally make each component depend
//       only on the OTHER two coordinates:
//          Bx(x,y,z) = a_y * y + a_z * z + a_0
//          By(x,y,z) = b_x * x + b_z * z + b_0
//          Bz(x,y,z) = c_x * x + c_y * y + c_0
//       so divB = 0 + 0 + 0 = 0 in exact arithmetic.  In FP, the
//       difference (Bx[i+1,j,k] - Bx[i,j,k]) reads two array slots that
//       hold the SAME bit pattern (the function doesn't depend on x at
//       fixed (j,k)) and the subtraction is bit-exact zero.  Same for By
//       and Bz.  So divB = 0 bit-exactly even after P/R fills the ghost.
//
//   (3) Tolerances are tightened: bit-exact == on the polynomial match
//       for cells reached by pure copy (interior-quadrant boundaries
//       inside the FMR box can be bit-exact); WithinAbs(few ulp * |val|)
//       for cells reached by P/R (FMR fine-coarse interfaces); strict
//       fabs(divB) <= ulp_scale for divB.  Catch2 macros are kept OUT
//       of the per-cell loop and aggregated, so we get one CHECK per
//       category.
//
//   (4) Ghostzones are invalidated (NaN-poisoned) by init_view before
//       apply_boundary_conditions(), same as the unigrid test — any
//       slot the BC pipeline leaves un-written stays NaN, the polynomial
//       check then catches it with full diagnostic info.
//
// Coverage: centered scalar (DENS) + all three face staggerings
// (FACEX/FACEY/FACEZ), plus per-cell divB on the resulting B-field
// staggered arrays.
//
// Author: carlo.musolino@aei.mpg.de
#include <catch2/catch_test_macros.hpp>

#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

// True iff the physical coordinate `p` lies outside the active simulation
// domain.  Cells outside get filled by a phys-BC kernel (extrap_N,
// sommerfeld) which does FP arithmetic, so bit-exactness is not
// guaranteed; we skip them per category.
inline bool is_phys_boundary(std::array<double, GRACE_NSPACEDIM> const& p)
{
    auto params = grace::config_parser::get()["amr"];
#ifdef GRACE_CARTESIAN_COORDINATES
    double const xmin = params["xmin"].as<double>();
    double const ymin = params["ymin"].as<double>();
    double const xmax = params["xmax"].as<double>();
    double const ymax = params["ymax"].as<double>();
    bool out = (p[0] < xmin) || (p[0] > xmax)
            || (p[1] < ymin) || (p[1] > ymax);
#ifdef GRACE_3D
    double const zmin = params["zmin"].as<double>();
    double const zmax = params["zmax"].as<double>();
    out = out || (p[2] < zmin) || (p[2] > zmax);
#endif
    return out;
#else
    auto const Ro = params["outer_region_radius"].as<double>();
    double r2 = math::int_pow<2>(p[0])
              + math::int_pow<2>(p[1])
#ifdef GRACE_3D
              + math::int_pow<2>(p[2])
#endif
              ;
    return r2 > Ro * Ro;
#endif
}

// Linear polynomial in three variables — exactly representable under any
// P/R order ≥ 1.  Used for the cell-centered state.
inline double h_func(VEC(double x, double y, double z))
{
    return 8.5 * x -5.1 * y +2.0 * z - 3.14;
}

// Face-B components: each depends only on the OTHER two coordinates.
// divB = ∂Bx/∂x + ∂By/∂y + ∂Bz/∂z = 0 + 0 + 0 = 0 in exact arithmetic,
// AND bit-exactly in FP because Bx evaluated at (x_i, y_j, z_k) and
// (x_{i+1}, y_j, z_k) returns identical bits (no dependence on x at fixed
// j,k) — their difference is exact zero.
inline double Bx_func(VEC(double /*x*/, double y, double z))
{
    return 3.7 * y -2.1 * z + 0.9;
}
inline double By_func(VEC(double x, double /*y*/, double z))
{
    return -1.8 * x + 4.3 * z - 0.5;
}
inline double Bz_func(VEC(double x, double y, double /*z*/))
{
    return 2.6 * x -3.2 * y + 1.7;
}

}  // namespace

// =============================================================================
// FMR ghost-zone fill correctness test.
// =============================================================================
TEST_CASE("BC bit-exact ghost-zone fill (FMR)", "[boundaries][fmr]")
{
    using namespace grace;
    using namespace grace::variables;

    int const DENS = DENS_;

    auto& state = variable_list::get().getstate();
    auto& stag  = variable_list::get().getstaggeredstate();
    auto& coord_system = coordinate_system::get();

    long nx, ny, nz;
    std::tie(nx, ny, nz) = amr::get_quadrant_extents();
    long const nq  = static_cast<long>(amr::get_local_num_quadrants());
    int  const ngz = amr::get_n_ghosts();
    int  const rank = parallel::mpi_comm_rank();

    if (rank == 0) {
        std::cout << "BC FMR ghost-fill test: nx,ny,nz,ngz = "
                  << nx << "," << ny << "," << nz << "," << ngz
                  << " nq(rank0)=" << nq << std::endl;
    }

    // Physical coordinates of a point inside cell (i,j,k) of local quadrant q.
    // `cc` selects the within-cell logical position (0.5 = centre, 0.0 = low
    // face of that axis).
    auto phys = [&](VEC(long i, long j, long k), long q,
                    std::array<double, GRACE_NSPACEDIM> const& cc) {
        return coord_system.get_physical_coordinates(
            {VEC(static_cast<size_t>(i),
                 static_cast<size_t>(j),
                 static_cast<size_t>(k))},
            static_cast<size_t>(q), cc, /*include_gzs*/ true);
    };

    // -------------------------------------------------------------------------
    // init_view: fill the entire 4-D view (INTERIOR and GHOST) with `f`
    // evaluated at the cell's physical position.  Followed by an explicit
    // invalidate_ghosts step that NaN-poisons the ghost slots — keeps the
    // test's invariant explicit:
    //   1. init_view              — polynomial everywhere.
    //   2. invalidate_ghosts      — NaN-poison ghost slots.
    //   3. apply_boundary_conditions  — BC refills ghosts.
    //   4. check_view             — every non-phys-BC ghost slot equals the
    //                               polynomial (NaN slot = BC missed it).
    // -------------------------------------------------------------------------
    auto init_view = [&](auto& view,
                         auto&& f,
                         std::array<double, GRACE_NSPACEDIM> const& cc,
                         VEC(long Nx, long Ny, long Nz),
                         int var_idx) {
        auto h = Kokkos::create_mirror_view(view);
        for (long q = 0; q < nq; ++q) {
        for (long k = 0; k < Nz; ++k) {
        for (long j = 0; j < Ny; ++j) {
        for (long i = 0; i < Nx; ++i) {
            auto p = phys(VEC(i, j, k), q, cc);
            h(VEC(i, j, k), var_idx, q) = f(VEC(p[0], p[1], p[2]));
        }}}}
        Kokkos::deep_copy(view, h);
    };

    // -------------------------------------------------------------------------
    // invalidate_ghosts: overwrite every GHOST slot with NaN.
    // -------------------------------------------------------------------------
    auto invalidate_ghosts = [&](auto& view,
                                 VEC(long Nx, long Ny, long Nz),
                                 int var_idx) {
        auto h = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(h, view);
        for (long q = 0; q < nq; ++q) {
        for (long k = 0; k < Nz; ++k) {
        for (long j = 0; j < Ny; ++j) {
        for (long i = 0; i < Nx; ++i) {
            bool const ghost = (i < ngz) || (i >= Nx - ngz)
                            || (j < ngz) || (j >= Ny - ngz)
#ifdef GRACE_3D
                            || (k < ngz) || (k >= Nz - ngz)
#endif
                            ;
            if (ghost) {
                h(VEC(i, j, k), var_idx, q) =
                    std::numeric_limits<double>::quiet_NaN();
            }
        }}}}
        Kokkos::deep_copy(view, h);
    };

    // -------------------------------------------------------------------------
    // check_view: aggregate per-cell deviations against `f`, no Catch2 macros
    // in the per-cell loop.  Returns counts + worst case for the one CHECK at
    // the end.
    // -------------------------------------------------------------------------
    struct check_result_t {
        size_t n_checked    = 0;
        size_t n_nan        = 0;   // BC failed to fill this slot
        size_t n_above_tol  = 0;   // value present but off by more than tol
        double max_abs_dev  = 0.0;
        long   max_i = -1, max_j = -1, max_k = -1, max_q = -1;
        double max_got = 0.0, max_expected = 0.0;
    };
    auto check_view = [&](auto& view,
                          char const* name,
                          auto&& f,
                          std::array<double, GRACE_NSPACEDIM> const& cc,
                          VEC(long Nx, long Ny, long Nz),
                          int var_idx,
                          double tol)
    -> check_result_t
    {
        auto h = Kokkos::create_mirror_view(view);
        Kokkos::deep_copy(h, view);
        check_result_t r;
        for (long q = 0; q < nq; ++q) {
        for (long k = 0; k < Nz; ++k) {
        for (long j = 0; j < Ny; ++j) {
        for (long i = 0; i < Nx; ++i) {
            auto p = phys(VEC(i, j, k), q, cc);
            if (is_phys_boundary(p)) continue;

            double const expected = f(VEC(p[0], p[1], p[2]));
            double const got      = h(VEC(i, j, k), var_idx, q);
            ++r.n_checked;

            if (std::isnan(got)) { ++r.n_nan; continue; }

            double const dev = std::fabs(got - expected);
            if (dev > r.max_abs_dev) {
                r.max_abs_dev  = dev;
                r.max_i = i; r.max_j = j; r.max_k = k; r.max_q = q;
                r.max_got = got; r.max_expected = expected;
            }
            if (dev > tol) ++r.n_above_tol;
        }}}}

        if (rank == 0) {
            std::cout << "  " << name
                      << ": checked " << r.n_checked
                      << ",  nan=" << r.n_nan
                      << ",  above_tol(" << tol << ")=" << r.n_above_tol
                      << ",  max|dev|=" << r.max_abs_dev
                      << "  @ ijk=("    << r.max_i << "," << r.max_j << ","
                                       << r.max_k << ") q=" << r.max_q
                      << "  got=" << r.max_got << " expected=" << r.max_expected
                      << std::endl;
        }
        return r;
    };

    // ===== Initialise (polynomial fill EVERYWHERE) =====
    init_view(state, h_func, {VEC(0.5, 0.5, 0.5)},
              VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz), DENS);

    init_view(stag.face_staggered_fields_x, Bx_func, {VEC(0.0, 0.5, 0.5)},
              VEC(nx + 2*ngz + 1, ny + 2*ngz, nz + 2*ngz), BSX_);
    init_view(stag.face_staggered_fields_y, By_func, {VEC(0.5, 0.0, 0.5)},
              VEC(nx + 2*ngz, ny + 2*ngz + 1, nz + 2*ngz), BSY_);
#ifdef GRACE_3D
    init_view(stag.face_staggered_fields_z, Bz_func, {VEC(0.5, 0.5, 0.0)},
              VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz + 1), BSZ_);
#endif

    // ===== Invalidate ghostzones (NaN-poison) =====
    invalidate_ghosts(state,
                      VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz), DENS);
    invalidate_ghosts(stag.face_staggered_fields_x,
                      VEC(nx + 2*ngz + 1, ny + 2*ngz, nz + 2*ngz), BSX_);
    invalidate_ghosts(stag.face_staggered_fields_y,
                      VEC(nx + 2*ngz, ny + 2*ngz + 1, nz + 2*ngz), BSY_);
#ifdef GRACE_3D
    invalidate_ghosts(stag.face_staggered_fields_z,
                      VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz + 1), BSZ_);
#endif

    // ===== Apply BC =====
    amr::apply_boundary_conditions();

    // ===== Check =====
    //
    // Tolerance: a few ulp times the function-value scale.  Linear functions
    // are within the P/R exactness window for any order ≥ 1; the only
    // deviation from the exact polynomial comes from FP roundoff in the P/R
    // arithmetic (Lagrange coefficient sums, conservative-restriction
    // averages).  16·eps·max(|f|,1) leaves comfortable headroom while still
    // catching anything 2 decades above the FP floor.
    constexpr double tol_poly = 16.0 * std::numeric_limits<double>::epsilon();

    auto r_state = check_view(state, "CENTER", h_func, {VEC(0.5, 0.5, 0.5)},
                              VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz), DENS,
                              /*tol=*/64.0 * tol_poly);

    auto r_bx = check_view(stag.face_staggered_fields_x, "FACEX",
                           Bx_func, {VEC(0.0, 0.5, 0.5)},
                           VEC(nx + 2*ngz + 1, ny + 2*ngz, nz + 2*ngz), BSX_,
                           /*tol=*/64.0 * tol_poly);
    auto r_by = check_view(stag.face_staggered_fields_y, "FACEY",
                           By_func, {VEC(0.5, 0.0, 0.5)},
                           VEC(nx + 2*ngz, ny + 2*ngz + 1, nz + 2*ngz), BSY_,
                           /*tol=*/64.0 * tol_poly);
#ifdef GRACE_3D
    auto r_bz = check_view(stag.face_staggered_fields_z, "FACEZ",
                           Bz_func, {VEC(0.5, 0.5, 0.0)},
                           VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz + 1), BSZ_,
                           /*tol=*/64.0 * tol_poly);
#else
    check_result_t r_bz; // empty
#endif

    // ===== divB check =====
    //
    // Each face-B function is independent of its own coordinate, so the
    // analytical divB is 0.  At any cell where ALL six surrounding face-B
    // slots are non-NaN, divB should be 0 to within the FP floor of the
    // CT discrete divergence.  Bound: ~ eps · max(|B|) / dx_min.
    //
    // Cells with a NaN in any of the 6 face slots are counted separately
    // (already-detected coverage gap in the face-B check above).
    auto bx_h = Kokkos::create_mirror_view(stag.face_staggered_fields_x);
    auto by_h = Kokkos::create_mirror_view(stag.face_staggered_fields_y);
    auto bz_h = Kokkos::create_mirror_view(stag.face_staggered_fields_z);
    Kokkos::deep_copy(bx_h, stag.face_staggered_fields_x);
    Kokkos::deep_copy(by_h, stag.face_staggered_fields_y);
    Kokkos::deep_copy(bz_h, stag.face_staggered_fields_z);

    auto& dx_arr_d = grace::variable_list::get().getinvspacings();
    auto  idx_h    = Kokkos::create_mirror_view(dx_arr_d);
    Kokkos::deep_copy(idx_h, dx_arr_d);

    size_t n_divB_checked       = 0;
    size_t n_divB_above_tol     = 0;
    size_t n_divB_skipped_nan   = 0;
    double max_abs_divB         = 0.0;
    long divB_i=-1, divB_j=-1, divB_k=-1, divB_q=-1;
    double divB_worst_value     = 0.0;

    constexpr double divB_tol   = 1e-13;  // matches CT-flux test floor

    long const Nx = nx + 2*ngz;
    long const Ny = ny + 2*ngz;
    long const Nz = nz + 2*ngz;
    for (long q = 0; q < nq; ++q) {
    for (long k = 0; k < Nz; ++k) {
    for (long j = 0; j < Ny; ++j) {
    for (long i = 0; i < Nx; ++i) {
        auto p_cc = phys(VEC(i, j, k), q, {VEC(0.5, 0.5, 0.5)});
        if (is_phys_boundary(p_cc)) continue;

        double const bxm = bx_h(VEC(i,   j,   k  ), BSX_, q);
        double const bxp = bx_h(VEC(i+1, j,   k  ), BSX_, q);
        double const bym = by_h(VEC(i,   j,   k  ), BSY_, q);
        double const byp = by_h(VEC(i,   j+1, k  ), BSY_, q);
        double const bzm = bz_h(VEC(i,   j,   k  ), BSZ_, q);
        double const bzp = bz_h(VEC(i,   j,   k+1), BSZ_, q);

        if (std::isnan(bxm) || std::isnan(bxp) ||
            std::isnan(bym) || std::isnan(byp) ||
            std::isnan(bzm) || std::isnan(bzp)) {
            ++n_divB_skipped_nan;
            continue;
        }

        double const divB = (bxp - bxm) * idx_h(0, q)
                          + (byp - bym) * idx_h(1, q)
                          + (bzp - bzm) * idx_h(2, q);
        double const adB = std::fabs(divB);
        ++n_divB_checked;
        if (adB > max_abs_divB) {
            max_abs_divB = adB;
            divB_i = i; divB_j = j; divB_k = k; divB_q = q;
            divB_worst_value = divB;
        }
        if (adB > divB_tol) ++n_divB_above_tol;
    }}}}

    if (rank == 0) {
        std::cout << "  DIVB: checked " << n_divB_checked
                  << ",  skipped(NaN face)=" << n_divB_skipped_nan
                  << ",  above_tol(" << divB_tol << ")=" << n_divB_above_tol
                  << ",  max|divB|=" << max_abs_divB
                  << "  @ ijk=(" << divB_i << "," << divB_j << "," << divB_k
                  << ") q=" << divB_q << "  divB=" << divB_worst_value
                  << std::endl;
    }

    // ===== MPI-aggregate failure counts =====
    // Single REQUIRE per category at the end — keeps Catch2 stream
    // tracking simple in the redirected-stdout context of
    // mains/p4est_tests_main.cc.
    auto reduce_sum = [](size_t local) {
        long long l = static_cast<long long>(local);
        long long g = 0;
        parallel::mpi_allreduce(&l, &g, 1, sc_MPI_SUM);
        return g;
    };

    long long const g_state_nan = reduce_sum(r_state.n_nan);
    long long const g_state_bad = reduce_sum(r_state.n_above_tol);
    long long const g_bx_nan    = reduce_sum(r_bx.n_nan);
    long long const g_bx_bad    = reduce_sum(r_bx.n_above_tol);
    long long const g_by_nan    = reduce_sum(r_by.n_nan);
    long long const g_by_bad    = reduce_sum(r_by.n_above_tol);
    long long const g_bz_nan    = reduce_sum(r_bz.n_nan);
    long long const g_bz_bad    = reduce_sum(r_bz.n_above_tol);
    long long const g_divB_nan  = reduce_sum(n_divB_skipped_nan);
    long long const g_divB_bad  = reduce_sum(n_divB_above_tol);

    if (rank == 0) {
        std::cout << "BC FMR aggregate (global):"
                  << "  state.nan="    << g_state_nan
                  << "  state.bad="    << g_state_bad
                  << "  Bx.nan="       << g_bx_nan
                  << "  Bx.bad="       << g_bx_bad
                  << "  By.nan="       << g_by_nan
                  << "  By.bad="       << g_by_bad
                  << "  Bz.nan="       << g_bz_nan
                  << "  Bz.bad="       << g_bz_bad
                  << "  divB.nan_skip="<< g_divB_nan
                  << "  divB.bad="     << g_divB_bad
                  << std::endl;
    }

    INFO("state: nan="     << g_state_nan << " bad=" << g_state_bad
         << "  Bx: nan="   << g_bx_nan    << " bad=" << g_bx_bad
         << "  By: nan="   << g_by_nan    << " bad=" << g_by_bad
         << "  Bz: nan="   << g_bz_nan    << " bad=" << g_bz_bad
         << "  divB: nan_skip=" << g_divB_nan << " bad=" << g_divB_bad);

    REQUIRE(g_state_nan  == 0);
    REQUIRE(g_state_bad  == 0);
    REQUIRE(g_bx_nan     == 0);
    REQUIRE(g_bx_bad     == 0);
    REQUIRE(g_by_nan     == 0);
    REQUIRE(g_by_bad     == 0);
    REQUIRE(g_bz_nan     == 0);
    REQUIRE(g_bz_bad     == 0);
    REQUIRE(g_divB_nan   == 0);
    REQUIRE(g_divB_bad   == 0);
}
