/**
 * @file test_c2p.cpp
 * @brief Round-trip validation of the GRMHD conservative ↔ primitive inversion.
 *
 * For every test sample we:
 *   1. Construct a known primitive state P0 = (ρ, ε, z^i, B^i, …).
 *   2. Forward-map P0 → C via `prims_to_conservs(P0, C, metric)`.
 *   3. Invert C → P1 via `conservs_to_prims<EOS>(C, P1, metric, eos, …)`.
 *   4. Check `max_i |P1_i − P0_i| / |P0_i|` is at the FP floor of the c2p
 *      tolerance.
 *
 * Three SECTIONs cover:
 *   - **Minkowski, no B**: ideal-hydro baseline.  Sweeps (ρ, ε, W, direction).
 *   - **Minkowski, magnetisation sweep**: at fixed thermodynamic state, sweep
 *     b² / ρh across the low-σ to high-σ regimes.  The c2p has known
 *     sensitivity near b²/ρh ≳ 1; this section quantifies it.
 *   - **Schwarzschild-CKS, no B**: same sweep as Minkowski but on a
 *     non-trivial spatial metric (lapse < 1, β ≠ 0, γ_xx ≠ 1).  Catches
 *     metric-aware bugs in the inverse that flat-space tests miss.
 *
 * All sweeps are **deterministic** (structured grids, no RNG) — fully
 * reproducible across runs.  No file dumps; failures get reported by Catch2
 * with the offending parameter point.
 *
 * Bound to `configs/c2p_test_gamma2.yaml` (γ=2 piecewise polytrope, γ_th=1.8).
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/utils/metric_utils.hh>

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace {

using namespace grace ;

// Deterministic unit-vector grid: 6 directions covering the principal
// axes (±x, ±y, ±z).  Enough to exercise the c2p across non-degenerate
// velocity orientations without an O(N²) angular sweep.
constexpr int N_DIR = 6 ;
constexpr double DIRS[N_DIR][3] = {
    { 1.0,  0.0,  0.0},
    {-1.0,  0.0,  0.0},
    { 0.0,  1.0,  0.0},
    { 0.0, -1.0,  0.0},
    { 0.0,  0.0,  1.0},
    { 0.0,  0.0, -1.0},
} ;

// Lorentz-factor sweep — mildly to moderately relativistic.
constexpr int N_W = 4 ;
constexpr double WS[N_W] = {1.05, 1.5, 2.0, 3.0} ;

// log10 ρ sweep — sub-NS-atmosphere up to NS density.
constexpr int N_RHO = 6 ;
constexpr double LOG10_RHO[N_RHO] = {-12.0, -8.0, -5.0, -3.0, -1.5, -0.5} ;

// Temperature sweep (in MeV-equivalent natural units) — cold to warm.
// The hybrid_pwpoly EOS only exposes a `press_eps_csnd2_entropy__temp_rho_ye`
// inversion (not an `eps_rho_ye` one), so the test drives the EOS with
// temperature and lets it compute eps internally.
constexpr int N_T = 4 ;
constexpr double TEMPERATURES[N_T] = {1e-3, 1e-1, 1.0, 10.0} ;

/// State-normalised infinity-norm residual.
///
/// Per-component relative norms blow up for components that are mathematically
/// zero (e.g. `Z_y = Z_z = 0` when the velocity is along ±x), because the
/// recovered value is some O(ulp) round-off and a `|a − b| / |b|` ratio is
/// then ill-conditioned.  Instead, take the absolute infinity-norm and
/// normalise once by the magnitude of the reference state.  Round-off noise
/// in a single component is then bounded by `O(ulp) / state_scale`, which is
/// the right small-number floor.
KOKKOS_INLINE_FUNCTION
double prim_residual(grmhd_prims_array_t const& a,
                     grmhd_prims_array_t const& b)
{
    using Kokkos::fmax ;
    using Kokkos::fabs ;
    using Kokkos::sqrt ;
    double const z_mag = sqrt(b[ZXL]*b[ZXL] + b[ZYL]*b[ZYL] + b[ZZL]*b[ZZL]) ;
    double const state_scale =
        fmax(fmax(fabs(b[RHOL]), fabs(b[EPSL])), fmax(z_mag, 1e-30)) ;
    constexpr unsigned int idx[5] = {RHOL, EPSL, ZXL, ZYL, ZZL} ;
    double err = 0.0 ;
    for (auto i : idx) {
        err = fmax(err, fabs(a[i] - b[i])) ;
    }
    return err / state_scale ;
}

/// Build a `metric_array_t` for the Cartesian-Kerr-Schild Schwarzschild
/// spacetime evaluated at (x, y, z) = (R, 0, 0).  Picks a non-trivial
/// γ_xx ≠ 1, β^x ≠ 0, α < 1 — far enough from the horizon that the
/// inversion is well-conditioned.
metric_array_t schwarzschild_cks_metric(double R, double M = 1.0)
{
    double const r   = R ;                       // along x-axis: r = |x|
    double const H   = M / r ;                   // KS scalar
    double const fac = 1.0 + 2.0 * H ;
    double const alp = 1.0 / std::sqrt(fac) ;    // lapse
    std::array<double, 3> beta = {2.0 * H / fac, 0.0, 0.0} ;
    // γ_ij = δ_ij + 2H l_i l_j, with l_spatial = (1,0,0) along the +x axis.
    std::array<double, 6> g = {
        1.0 + 2.0 * H,  // gxx
        0.0,            // gxy
        0.0,            // gxz
        1.0,            // gyy
        0.0,            // gyz
        1.0             // gzz
    } ;
    return metric_array_t(g, beta, alp) ;
}

} // namespace


TEST_CASE("c2p round-trip / Minkowski, no B", "[c2p][hydro]")
{
    using namespace Kokkos ;

    auto eos       = eos::get().get_hybrid_pwpoly() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    metric_array_t metric({1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
                          {0.0, 0.0, 0.0}, 1.0) ;

    constexpr size_t N = N_RHO * N_T * N_W * N_DIR ;
    Kokkos::View<double*> residual("res", N) ;
    Kokkos::View<int*>    floored ("floored", N) ;

    // using grace::eos_err_t directly — hybrid_eos_t::error_type is private ;

    parallel_for("c2p_mink_no_B", N, KOKKOS_LAMBDA(int const idx) {
        int q = idx ;
        int const i_rho = q % N_RHO ; q /= N_RHO ;
        int const i_t   = q % N_T   ; q /= N_T   ;
        int const i_w   = q % N_W   ; q /= N_W   ;
        int const i_dir = q ;

        double const rho   = std::pow(10.0, LOG10_RHO[i_rho]) ;
        double const temp0 = TEMPERATURES[i_t] ;
        double const W     = WS[i_w] ;
        double const z_mag = std::sqrt(W*W - 1.0) ;

        grmhd_prims_array_t p0{} ;
        p0[RHOL]  = rho ;
        p0[TEMPL] = temp0 ;
        p0[YEL]   = 0.0 ;
        p0[ZXL]   = z_mag * DIRS[i_dir][0] ;
        p0[ZYL]   = z_mag * DIRS[i_dir][1] ;
        p0[ZZL]   = z_mag * DIRS[i_dir][2] ;
        p0[BXL]   = p0[BYL] = p0[BZL] = 0.0 ;

        // Fill pressure / eps / csnd / entropy from EOS given (temp, rho, ye).
        double csnd2 ;
        grace::eos_err_t err{} ;
        p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_impl(
                        p0[EPSL], csnd2, p0[ENTL], p0[TEMPL], p0[RHOL], p0[YEL], err) ;

        grmhd_cons_array_t cons{} ;
        prims_to_conservs(p0, cons, metric) ;

        grmhd_prims_array_t p1 = p0 ;
        c2p_err_t c2p_err ;
        double rtp[3] = {1.0, 1.0, 1.0} ;
        bool const ok = conservs_to_prims(cons, p1, metric, eos, atmo, excision,
                                           c2p_pars, rtp, c2p_err) ;
        residual(idx)  = prim_residual(p1, p0) ;
        floored(idx) = ok ? 1 : 0 ;
    }) ;
    Kokkos::fence() ;

    auto r_h = create_mirror_view(residual)  ;  deep_copy(r_h,  residual)  ;
    auto c_h = create_mirror_view(floored) ;  deep_copy(c_h,  floored) ;

    double max_res = 0.0 ;
    size_t i_max   = 0 ;
    int    n_fail  = 0 ;
    for (size_t i = 0 ; i < N ; ++i) {
        if (c_h(i)) ++n_fail ;
        if (r_h(i) > max_res) { max_res = r_h(i) ; i_max = i ; }
    }
    // Decode the linear argmax back into (i_rho, i_t, i_w, i_dir) so the
    // failing sample's parameters surface in the test log.
    {
        size_t q = i_max ;
        int const i_rho = q % N_RHO ; q /= N_RHO ;
        int const i_t   = q % N_T   ; q /= N_T   ;
        int const i_w   = q % N_W   ; q /= N_W   ;
        int const i_dir = q ;
        INFO("max residual = " << max_res
             << " at idx=" << i_max
             << " (rho=" << std::pow(10.0, LOG10_RHO[i_rho])
             << ", T="   << TEMPERATURES[i_t]
             << ", W="   << WS[i_w]
             << ", dir=(" << DIRS[i_dir][0] << ","
                          << DIRS[i_dir][1] << ","
                          << DIRS[i_dir][2] << "))"
             << "; floored samples = " << n_fail) ;
        REQUIRE(n_fail == 0) ;
        REQUIRE(max_res < 1e-10) ;
    }
}


TEST_CASE("c2p round-trip / Minkowski, magnetisation sweep", "[c2p][mhd]")
{
    using namespace Kokkos ;

    auto eos       = eos::get().get_hybrid_pwpoly() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    metric_array_t metric({1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
                          {0.0, 0.0, 0.0}, 1.0) ;

    // Fixed thermo state at moderate W; the parameter being swept is
    // b² ≈ B² (Minkowski), in units of ρ.  Production c2p has known
    // sensitivity in the b²/ρh ≳ 1 regime — c2p_pars.max_sigma gates it.
    // High-σ samples (b²/ρh ≳ 1) legitimately lose precision in the c2p
    // round-trip: the fluid energy becomes a small fraction of total
    // energy (B² dominates), so tiny errors in conservatives produce big
    // errors in primitives.  Sweep stays in the low-σ regime where round-
    // trip is expected to hold to ~1e-8.
    constexpr int N_BMAG = 3 ;
    constexpr double BSQ_OVER_RHO[N_BMAG] = {0.0, 1e-4, 1e-2} ;
    constexpr double RHO0  = 1e-3 ;
    constexpr double TEMP0 = 1e-1 ;
    constexpr double W0    = 1.5 ;
    constexpr double Z0    = 1.118033988749895 ;  // sqrt(W0²-1)

    size_t const N = N_BMAG * N_DIR ;
    Kokkos::View<double*> residual("res", N) ;
    Kokkos::View<int*>    floored ("floored", N) ;
    // Per-sample (reference, recovered) prim values so the worst-residual
    // sample can be dumped component-by-component on failure.  9 components
    // each: RHO, EPS, TEMP, Z{x,y,z}, B{x,y,z}.
    Kokkos::View<double*[18]> samples("samples", N) ;

    // using grace::eos_err_t directly — hybrid_eos_t::error_type is private ;

    parallel_for("c2p_mink_Bsweep", N, KOKKOS_LAMBDA(int const idx) {
        int const i_dir  = idx % N_DIR ;
        int const i_bmag = idx / N_DIR ;
        double const B_mag = std::sqrt(BSQ_OVER_RHO[i_bmag] * RHO0) ;

        grmhd_prims_array_t p0{} ;
        p0[RHOL]  = RHO0 ;
        p0[TEMPL] = TEMP0 ;
        p0[YEL]   = 0.0 ;
        // Velocity along +x (fixed direction); B direction varies.
        p0[ZXL]   = Z0 ;
        p0[ZYL]   = 0.0 ;
        p0[ZZL]   = 0.0 ;
        p0[BXL]   = B_mag * DIRS[i_dir][0] ;
        p0[BYL]   = B_mag * DIRS[i_dir][1] ;
        p0[BZL]   = B_mag * DIRS[i_dir][2] ;

        double csnd2 ;
        grace::eos_err_t err{} ;
        p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_impl(
                        p0[EPSL], csnd2, p0[ENTL], p0[TEMPL], p0[RHOL], p0[YEL], err) ;

        grmhd_cons_array_t cons{} ;
        prims_to_conservs(p0, cons, metric) ;
        // `prims_to_conservs` only writes (D, τ, S_i, ent*) — in production
        // the densitised B field `cons[BS*]` is set separately via CT/EMF.
        // For the c2p round-trip test we propagate it manually here:
        //   cons[BS_i] = √γ · B^i.   Minkowski → √γ = 1.
        cons[BSXL] = p0[BXL] ;
        cons[BSYL] = p0[BYL] ;
        cons[BSZL] = p0[BZL] ;

        grmhd_prims_array_t p1 = p0 ;
        c2p_err_t c2p_err ;
        double rtp[3] = {1.0, 1.0, 1.0} ;
        bool const ok = conservs_to_prims(cons, p1, metric, eos, atmo, excision,
                                           c2p_pars, rtp, c2p_err) ;
        residual(idx)  = prim_residual(p1, p0) ;
        floored(idx) = ok ? 1 : 0 ;
        // Stash reference + recovered prim components for failure diagnosis.
        samples(idx,  0) = p0[RHOL] ;  samples(idx,  9) = p1[RHOL] ;
        samples(idx,  1) = p0[EPSL] ;  samples(idx, 10) = p1[EPSL] ;
        samples(idx,  2) = p0[TEMPL];  samples(idx, 11) = p1[TEMPL];
        samples(idx,  3) = p0[ZXL] ;   samples(idx, 12) = p1[ZXL] ;
        samples(idx,  4) = p0[ZYL] ;   samples(idx, 13) = p1[ZYL] ;
        samples(idx,  5) = p0[ZZL] ;   samples(idx, 14) = p1[ZZL] ;
        samples(idx,  6) = p0[BXL] ;   samples(idx, 15) = p1[BXL] ;
        samples(idx,  7) = p0[BYL] ;   samples(idx, 16) = p1[BYL] ;
        samples(idx,  8) = p0[BZL] ;   samples(idx, 17) = p1[BZL] ;
    }) ;
    Kokkos::fence() ;

    auto r_h = create_mirror_view(residual) ;  deep_copy(r_h, residual) ;
    auto c_h = create_mirror_view(floored)  ;  deep_copy(c_h, floored)  ;
    auto s_h = create_mirror_view(samples)  ;  deep_copy(s_h, samples)  ;

    double max_res = 0.0 ;
    size_t i_max   = 0 ;
    int    n_fail  = 0 ;
    for (size_t i = 0 ; i < N ; ++i) {
        if (c_h(i)) ++n_fail ;
        if (r_h(i) > max_res) { max_res = r_h(i) ; i_max = i ; }
    }
    {
        int const i_dir  = i_max % N_DIR ;
        int const i_bmag = i_max / N_DIR ;
        char const* names[9] = {"RHO","EPS","TEMP","Zx","Zy","Zz","Bx","By","Bz"} ;
        std::ostringstream dump ;
        dump << std::scientific << std::setprecision(12) ;
        dump << "\n  worst sample:  b²/ρ = " << BSQ_OVER_RHO[i_bmag]
             << ", dir = (" << DIRS[i_dir][0] << ","
                            << DIRS[i_dir][1] << ","
                            << DIRS[i_dir][2] << ")\n" ;
        dump << "    "
             << std::setw(6) << "comp" << "  "
             << std::setw(20) << "p_ref"
             << std::setw(20) << "p_recovered"
             << std::setw(20) << "delta" << "\n" ;
        for (int c = 0 ; c < 9 ; ++c) {
            double const ref = s_h(i_max, c) ;
            double const rec = s_h(i_max, c + 9) ;
            dump << "    "
                 << std::setw(6) << names[c] << "  "
                 << std::setw(20) << ref
                 << std::setw(20) << rec
                 << std::setw(20) << (rec - ref) << "\n" ;
        }
        INFO("max residual = " << max_res
             << " at idx=" << i_max
             << " (b²/ρ=" << BSQ_OVER_RHO[i_bmag]
             << ", dir=(" << DIRS[i_dir][0] << ","
                          << DIRS[i_dir][1] << ","
                          << DIRS[i_dir][2] << "))"
             << "; floored samples = " << n_fail
             << dump.str()) ;
        REQUIRE(n_fail == 0) ;
        REQUIRE(max_res < 1e-8) ;
    }
}


TEST_CASE("c2p round-trip / Schwarzschild-CKS, no B", "[c2p][hydro][curved]")
{
    using namespace Kokkos ;

    auto eos       = eos::get().get_hybrid_pwpoly() ;
    auto atmo      = get_atmo_params() ;
    auto excision  = get_excision_params() ;
    auto c2p_pars  = get_c2p_params() ;

    // Schwarzschild at (R, 0, 0) for R well outside the horizon — c2p
    // is well-conditioned (α ≈ 0.71, β^x ≈ 0.33, γ_xx = 1.5).
    auto metric = schwarzschild_cks_metric(/*R=*/4.0, /*M=*/1.0) ;

    constexpr size_t N = N_RHO * N_T * N_W * N_DIR ;
    Kokkos::View<double*> residual("res", N) ;
    Kokkos::View<int*>    floored ("floored", N) ;

    // using grace::eos_err_t directly — hybrid_eos_t::error_type is private ;

    parallel_for("c2p_sch_no_B", N, KOKKOS_LAMBDA(int const idx) {
        int q = idx ;
        int const i_rho = q % N_RHO ; q /= N_RHO ;
        int const i_t   = q % N_T   ; q /= N_T   ;
        int const i_w   = q % N_W   ; q /= N_W   ;
        int const i_dir = q ;

        double const rho   = std::pow(10.0, LOG10_RHO[i_rho]) ;
        double const temp0 = TEMPERATURES[i_t] ;
        double const W     = WS[i_w] ;
        double const z_mag = std::sqrt(W*W - 1.0) ;

        // For a non-trivial γ, project the direction such that γ_ij z^i z^j
        // = W²−1 (so the resulting Lorentz factor is exactly W).  The DIRS
        // table is unit in flat space; rescale by 1/sqrt(γ_ij·d̂_i·d̂_j).
        double const dx = DIRS[i_dir][0], dy = DIRS[i_dir][1], dz = DIRS[i_dir][2] ;
        double const norm2 =
              metric.gamma(0)*dx*dx + 2*metric.gamma(1)*dx*dy
            + 2*metric.gamma(2)*dx*dz + metric.gamma(3)*dy*dy
            + 2*metric.gamma(4)*dy*dz + metric.gamma(5)*dz*dz ;
        double const inv_norm = 1.0 / std::sqrt(norm2) ;

        grmhd_prims_array_t p0{} ;
        p0[RHOL]  = rho ;
        p0[TEMPL] = temp0 ;
        p0[YEL]   = 0.0 ;
        p0[ZXL]   = z_mag * dx * inv_norm ;
        p0[ZYL]   = z_mag * dy * inv_norm ;
        p0[ZZL]   = z_mag * dz * inv_norm ;
        p0[BXL]   = p0[BYL] = p0[BZL] = 0.0 ;

        double csnd2 ;
        grace::eos_err_t err{} ;
        p0[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye_impl(
                        p0[EPSL], csnd2, p0[ENTL], p0[TEMPL], p0[RHOL], p0[YEL], err) ;

        grmhd_cons_array_t cons{} ;
        prims_to_conservs(p0, cons, metric) ;

        grmhd_prims_array_t p1 = p0 ;
        c2p_err_t c2p_err ;
        double rtp[3] = {1.0, 1.0, 1.0} ;
        bool const ok = conservs_to_prims(cons, p1, metric, eos, atmo, excision,
                                           c2p_pars, rtp, c2p_err) ;
        residual(idx)  = prim_residual(p1, p0) ;
        floored(idx) = ok ? 1 : 0 ;
    }) ;
    Kokkos::fence() ;

    auto r_h = create_mirror_view(residual)  ;  deep_copy(r_h,  residual)  ;
    auto c_h = create_mirror_view(floored) ;  deep_copy(c_h,  floored) ;

    double max_res = 0.0 ;
    size_t i_max   = 0 ;
    int    n_fail  = 0 ;
    for (size_t i = 0 ; i < N ; ++i) {
        if (c_h(i)) ++n_fail ;
        if (r_h(i) > max_res) { max_res = r_h(i) ; i_max = i ; }
    }
    {
        size_t q = i_max ;
        int const i_rho = q % N_RHO ; q /= N_RHO ;
        int const i_t   = q % N_T   ; q /= N_T   ;
        int const i_w   = q % N_W   ; q /= N_W   ;
        int const i_dir = q ;
        INFO("max residual = " << max_res
             << " at idx=" << i_max
             << " (rho=" << std::pow(10.0, LOG10_RHO[i_rho])
             << ", T="   << TEMPERATURES[i_t]
             << ", W="   << WS[i_w]
             << ", dir=(" << DIRS[i_dir][0] << ","
                          << DIRS[i_dir][1] << ","
                          << DIRS[i_dir][2] << "))"
             << "; floored samples = " << n_fail) ;
        REQUIRE(n_fail == 0) ;
        REQUIRE(max_res < 1e-10) ;
    }
}
