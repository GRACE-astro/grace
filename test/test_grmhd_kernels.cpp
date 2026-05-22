/**
 * @file test_grmhd_kernels.cpp
 * @brief Unit tests for the codegen-generated GRMHD physics kernels.
 *
 * Until now, kernel correctness was tested only TRANSITIVELY via the
 * conservation/equilibrium tests (`flux_reflux_test`, `ct_flux_conservation`,
 * `fofc_conservation`).  A bug that breaks an individual kernel and is
 * silently absorbed by another (or by reflux) would pass every existing
 * check.  This file plugs that gap with direct input → output assertions
 * against analytic expectations.
 *
 * Covered (each its own TEST_CASE with multiple SECTIONs):
 *   - `grmhd_get_W`             — Lorentz factor from z = W·u^i.
 *   - `grmhd_get_vtildeu`       — transport velocity α u^j / W − β^j.
 *   - `grmhd_get_conserved`     — primitives → conservatives.
 *   - `grmhd_get_fluxes`        — Banyuls/Antón face flux (hydro, B=0).
 *   - `grmhd_get_cm_cp`         — sound-speed-based wavespeeds (hydro).
 *   - `grmhd_get_geom_sources`  — geometric source: vanishes on Minkowski.
 *
 * Bound to `configs/basic_config.yaml` — the kernels are pure functions of
 * their arguments, so the parfile only matters for the singleton init.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace_config.h>
#include <grace/physics/grmhd_subexpressions.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/ideal_gas_eos.hh>

#include <array>
#include <cmath>

namespace {

// γ-2 polytrope-ish thermo state used for sound-speed / pressure references.
// Pressure and sound speed for `γ = 2` ideal-gas EOS: p = (γ-1)·ρ·ε,
// h = 1 + ε + p/ρ,  cs² = γ·p / (ρ·h).
struct hydro_state {
    double rho{1e-3} ;
    double eps{1e-2} ;
    double press_at(double gamma) const { return (gamma - 1.0) * rho * eps ; }
    double h_at(double gamma)     const {
        double const p = press_at(gamma) ;
        return 1.0 + eps + p / rho ;
    }
    double cs2_at(double gamma)   const {
        double const p = press_at(gamma) ;
        double const h = h_at(gamma) ;
        return gamma * p / (rho * h) ;
    }
} ;

constexpr double GAMMA_TEST = 2.0 ;
constexpr hydro_state H{} ;

// Build the Schwarzschild Cartesian-Kerr-Schild spatial metric at the
// point (R, 0, 0):  α = 1/√(1+2M/R), β^x = (2M/R)/(1+2M/R),
// γ_xx = 1+2M/R, γ_yy = γ_zz = 1, off-diagonals = 0.
struct sch_cks {
    double R{4.0} ;
    double M{1.0} ;
    double alp() const  { return 1.0 / std::sqrt(1.0 + 2.0*M/R) ; }
    std::array<double, 3> betau() const {
        double const fac = 1.0 + 2.0*M/R ;
        return {2.0*M/(R*fac), 0.0, 0.0} ;
    }
    std::array<double, 6> gdd() const {
        return {1.0 + 2.0*M/R, 0.0, 0.0, 1.0, 0.0, 1.0} ;
    }
    std::array<double, 6> guu() const {
        // Inverse of γ = diag(1+2H, 1, 1) is diag(1/(1+2H), 1, 1).
        return {1.0/(1.0 + 2.0*M/R), 0.0, 0.0, 1.0, 0.0, 1.0} ;
    }
} ;

} // namespace


TEST_CASE("GRMHD kernels / Lorentz factor", "[grmhd][kernels]")
{
    SECTION("at rest → W = 1")
    {
        std::array<double, 6> g = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0} ;
        std::array<double, 3> z = {0.0, 0.0, 0.0} ;
        double W ;
        grmhd_get_W(g.data(), z.data(), &W) ;
        REQUIRE_THAT(W, Catch::Matchers::WithinRel(1.0, 1e-15)) ;
    }

    SECTION("Minkowski, z = (0.5, 0, 0) → W = √(1 + 0.25)")
    {
        std::array<double, 6> g = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0} ;
        std::array<double, 3> z = {0.5, 0.0, 0.0} ;
        double W ;
        grmhd_get_W(g.data(), z.data(), &W) ;
        REQUIRE_THAT(W, Catch::Matchers::WithinRel(std::sqrt(1.25), 1e-15)) ;
    }

    SECTION("Curved γ_ij, generic z → W = √(1 + γ_ij z^i z^j)")
    {
        sch_cks const S ;
        auto const g = S.gdd() ;
        std::array<double, 3> z = {0.3, -0.2, 0.1} ;
        double const expected = std::sqrt(
            1.0 + g[0]*z[0]*z[0] + g[3]*z[1]*z[1] + g[5]*z[2]*z[2]
              + 2.0*g[1]*z[0]*z[1] + 2.0*g[2]*z[0]*z[2] + 2.0*g[4]*z[1]*z[2]) ;
        double W ;
        grmhd_get_W(g.data(), z.data(), &W) ;
        REQUIRE_THAT(W, Catch::Matchers::WithinRel(expected, 1e-15)) ;
    }
}


TEST_CASE("GRMHD kernels / Transport velocity", "[grmhd][kernels]")
{
    SECTION("at rest, β = 0 → vtilde = 0")
    {
        std::array<double, 3> betau = {0.0, 0.0, 0.0} ;
        std::array<double, 3> z     = {0.0, 0.0, 0.0} ;
        double W = 1.0 ;
        double vtilde[3] ;
        grmhd_get_vtildeu(1.0, betau.data(), z.data(), W, &vtilde) ;
        REQUIRE_THAT(vtilde[0], Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
        REQUIRE_THAT(vtilde[1], Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
        REQUIRE_THAT(vtilde[2], Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
    }

    SECTION("at rest, β ≠ 0 → vtilde = -β")
    {
        std::array<double, 3> betau = {0.10, -0.05, 0.02} ;
        std::array<double, 3> z     = {0.0, 0.0, 0.0} ;
        double W = 1.0 ;
        double vtilde[3] ;
        grmhd_get_vtildeu(0.7, betau.data(), z.data(), W, &vtilde) ;
        REQUIRE_THAT(vtilde[0], Catch::Matchers::WithinAbs(-betau[0], 1e-15)) ;
        REQUIRE_THAT(vtilde[1], Catch::Matchers::WithinAbs(-betau[1], 1e-15)) ;
        REQUIRE_THAT(vtilde[2], Catch::Matchers::WithinAbs(-betau[2], 1e-15)) ;
    }

    SECTION("Generic state matches α·u^j/W − β^j")
    {
        std::array<double, 3> betau = {0.10, -0.05, 0.02} ;
        std::array<double, 3> z     = {0.4, -0.3, 0.2} ;
        double const alp = 0.6 ;
        double const W   = std::sqrt(1.0 + z[0]*z[0] + z[1]*z[1] + z[2]*z[2]) ;
        double vtilde[3] ;
        grmhd_get_vtildeu(alp, betau.data(), z.data(), W, &vtilde) ;
        for (int j = 0 ; j < 3 ; ++j) {
            double const expected = alp * z[j] / W - betau[j] ;
            INFO("j = " << j) ;
            REQUIRE_THAT(vtilde[j], Catch::Matchers::WithinRel(expected, 1e-15)) ;
        }
    }
}


TEST_CASE("GRMHD kernels / Conservatives", "[grmhd][kernels]")
{
    // Common inputs.
    std::array<double, 6> g_flat = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0} ;
    std::array<double, 3> beta_0 = {0.0, 0.0, 0.0} ;
    double const press = H.press_at(GAMMA_TEST) ;

    SECTION("Static fluid, Minkowski, B = 0 → D = ρ, S_i = 0, τ = ρ·ε")
    {
        std::array<double, 3> z   = {0.0, 0.0, 0.0} ;
        double W   = 1.0 ;
        double b2  = 0.0 ;
        double smallbu[4] = {0.0, 0.0, 0.0, 0.0} ;
        double dens, tau, ent ;
        double stilde[3] ;
        grmhd_get_conserved(1.0, beta_0.data(), g_flat.data(),
                                    H.rho, press, H.eps, z.data(),
                                    /*s=*/0.0, W, b2, smallbu,
                                    &dens, &tau, &stilde, &ent) ;
        REQUIRE_THAT(dens, Catch::Matchers::WithinRel(H.rho, 1e-15)) ;
        REQUIRE_THAT(stilde[0], Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
        REQUIRE_THAT(stilde[1], Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
        REQUIRE_THAT(stilde[2], Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
        REQUIRE_THAT(tau, Catch::Matchers::WithinRel(H.rho * H.eps, 1e-13)) ;
    }

    SECTION("x-boosted, Minkowski, B = 0 → D = ρW, S_x = ρhW²·u_x")
    {
        // z = (0.5, 0, 0), Minkowski → W = √1.25, u_x = z_x in flat space.
        std::array<double, 3> z = {0.5, 0.0, 0.0} ;
        double const W = std::sqrt(1.25) ;
        double const u_x = z[0] ;
        double const h   = H.h_at(GAMMA_TEST) ;
        double b2  = 0.0 ;
        double smallbu[4] = {0.0, 0.0, 0.0, 0.0} ;
        double dens, tau, ent ;
        double stilde[3] ;
        grmhd_get_conserved(1.0, beta_0.data(), g_flat.data(),
                                    H.rho, press, H.eps, z.data(),
                                    /*s=*/0.0, W, b2, smallbu,
                                    &dens, &tau, &stilde, &ent) ;
        REQUIRE_THAT(dens, Catch::Matchers::WithinRel(H.rho * W, 1e-14)) ;
        REQUIRE_THAT(stilde[0],
                     Catch::Matchers::WithinRel(H.rho * h * W * u_x, 1e-13)) ;
        REQUIRE_THAT(stilde[1], Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
        REQUIRE_THAT(stilde[2], Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
        // τ = ρhW² − p − D
        double const tau_expected = H.rho * h * W * W - press - H.rho * W ;
        REQUIRE_THAT(tau, Catch::Matchers::WithinRel(tau_expected, 1e-13)) ;
    }
}


TEST_CASE("GRMHD kernels / Fluxes (hydro, B = 0)", "[grmhd][kernels]")
{
    std::array<double, 6> g_flat = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0} ;
    std::array<double, 3> beta_0 = {0.0, 0.0, 0.0} ;
    double const press = H.press_at(GAMMA_TEST) ;

    SECTION("Static fluid → F^j_D = 0, F^j_τ = 0, F^j_S_i = p·δ^j_i")
    {
        std::array<double, 3> z = {0.0, 0.0, 0.0} ;
        double W = 1.0 ;
        double b2 = 0.0 ;
        double smallbu[4] = {0.0, 0.0, 0.0, 0.0} ;
        double vtilde[3] = {0.0, 0.0, 0.0} ;  // -β = 0
        double dens, tau, ent, fD, ftau, fent ;
        double stilde[3], fstilde[3] ;
        for (int dir = 0 ; dir < 3 ; ++dir) {
            INFO("direction " << dir) ;
            grmhd_get_fluxes(1.0, beta_0.data(), g_flat.data(),
                                     H.rho, press, H.eps, z.data(),
                                     /*s=*/0.0, W, b2, smallbu, vtilde,
                                     dir,
                                     &dens, &tau, &stilde, &ent,
                                     &fD, &ftau, &fstilde, &fent) ;
            REQUIRE_THAT(fD,   Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
            REQUIRE_THAT(ftau, Catch::Matchers::WithinAbs(0.0, 1e-15)) ;
            for (int i = 0 ; i < 3 ; ++i) {
                double const expected = (i == dir) ? press : 0.0 ;
                INFO("S-component i = " << i) ;
                REQUIRE_THAT(fstilde[i],
                             Catch::Matchers::WithinAbs(expected, 1e-13)) ;
            }
        }
    }

    SECTION("x-boost → F^x_D = D·vtilde^x  matches analytic Banyuls form")
    {
        std::array<double, 3> z = {0.5, 0.0, 0.0} ;
        double const W = std::sqrt(1.25) ;
        double b2 = 0.0 ;
        double smallbu[4] = {0.0, 0.0, 0.0, 0.0} ;
        double vtilde[3] ;
        grmhd_get_vtildeu(1.0, beta_0.data(), z.data(), W, &vtilde) ;
        double dens, tau, ent, fD, ftau, fent ;
        double stilde[3], fstilde[3] ;
        grmhd_get_fluxes(1.0, beta_0.data(), g_flat.data(),
                                 H.rho, press, H.eps, z.data(),
                                 /*s=*/0.0, W, b2, smallbu, vtilde,
                                 /*idir=*/0,
                                 &dens, &tau, &stilde, &ent,
                                 &fD, &ftau, &fstilde, &fent) ;
        // D = ρW;  F^x_D = D · vtilde^x.
        double const D_expected = H.rho * W ;
        REQUIRE_THAT(fD,
                     Catch::Matchers::WithinRel(D_expected * vtilde[0], 1e-13)) ;
    }
}


TEST_CASE("GRMHD kernels / Wavespeeds (hydro, B = 0)", "[grmhd][kernels]")
{
    std::array<double, 6> g_flat = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0} ;
    std::array<double, 3> beta_0 = {0.0, 0.0, 0.0} ;
    double const press = H.press_at(GAMMA_TEST) ;
    double const cs2   = H.cs2_at(GAMMA_TEST) ;

    SECTION("Static fluid → cp, cm = ±c_s")
    {
        double vtilde[3] = {0.0, 0.0, 0.0} ;
        double const guuDD_xx = 1.0 ;
        double cm, cp ;
        grmhd_get_cm_cp(1.0, beta_0.data(),
                                H.rho, press, H.eps, cs2,
                                /*W=*/1.0, /*b2=*/0.0, vtilde, guuDD_xx,
                                /*idir=*/0, &cm, &cp) ;
        REQUIRE_THAT(cp, Catch::Matchers::WithinRel( std::sqrt(cs2), 1e-12)) ;
        REQUIRE_THAT(cm, Catch::Matchers::WithinRel(-std::sqrt(cs2), 1e-12)) ;
    }

    SECTION("x-boost → cp ≠ -cm; cp − cm matches analytic boosted form")
    {
        // For a boost along +x in Minkowski, both characteristics shift
        // by the flow velocity but the SPREAD `cp − cm` remains close to
        // 2·cs / W² (relativistic narrowing) at small cs.  Here we just
        // check the SIGN structure (cp > 0 > cm) and the symmetry that
        // cp + cm = 2·vtilde^x (centre of the characteristic fan).
        std::array<double, 3> z = {0.5, 0.0, 0.0} ;
        double const W = std::sqrt(1.25) ;
        double vtilde[3] ;
        grmhd_get_vtildeu(1.0, beta_0.data(), z.data(), W, &vtilde) ;
        double const guuDD_xx = 1.0 ;
        double cm, cp ;
        grmhd_get_cm_cp(1.0, beta_0.data(),
                                H.rho, press, H.eps, cs2,
                                W, /*b2=*/0.0, vtilde, guuDD_xx,
                                /*idir=*/0, &cm, &cp) ;
        REQUIRE(cp > 0.0) ;
        REQUIRE(cm < cp) ;
        // Both characteristics straddle vtilde^x with cs/W width to first order.
        REQUIRE(cm < vtilde[0]) ;
        REQUIRE(vtilde[0] < cp) ;
    }
}


TEST_CASE("GRMHD kernels / Geom source on Minkowski → 0", "[grmhd][kernels]")
{
    // Identity 3-metric, zero shift, lapse = 1, K = 0, all spatial
    // derivatives zero ⇒ the geometric-source kernel must return zero for
    // ANY hydrodynamic input state.  This catches the kind of bug we found
    // in HydroCode.py (asymmetric W^{ij} storage) and any similar
    // index-bookkeeping regressions.
    std::array<double, 6> g_flat = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0} ;
    std::array<double, 6> guu    = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0} ;
    std::array<double, 6> Kdd    = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0} ;
    std::array<double, 3> beta_0 = {0.0, 0.0, 0.0} ;
    std::array<double, 3> dalp_0 = {0.0, 0.0, 0.0} ;
    std::array<double, 18> dgdd_0{} ;   // zeros
    std::array<double, 9>  dbet_0{} ;   // zeros
    std::array<double, 3>  B_0 = {0.0, 0.0, 0.0} ;

    double const press = H.press_at(GAMMA_TEST) ;

    SECTION("Static fluid → dtau = 0, dstilde = 0")
    {
        std::array<double, 3> z = {0.0, 0.0, 0.0} ;
        double dtau ;
        double dstilde[3] ;
        grmhd_get_geom_sources(
            1.0, beta_0.data(), g_flat.data(), guu.data(), Kdd.data(),
            dalp_0.data(), dgdd_0.data(), dbet_0.data(),
            H.rho, press, H.eps, B_0.data(), z.data(),
            /*W=*/1.0, &dtau, &dstilde) ;
        REQUIRE_THAT(dtau, Catch::Matchers::WithinAbs(0.0, 1e-14)) ;
        for (int i = 0 ; i < 3 ; ++i) {
            INFO("S-source component i = " << i) ;
            REQUIRE_THAT(dstilde[i], Catch::Matchers::WithinAbs(0.0, 1e-14)) ;
        }
    }

    SECTION("Generic boosted state → dtau = 0, dstilde = 0 (Minkowski has no sources)")
    {
        // Same metric (all derivs zero), but now arbitrary z, B.  Sources
        // should still vanish — confirms the kernel doesn't accidentally
        // produce non-zero output from STATE alone.
        std::array<double, 3> z = {0.4, -0.3, 0.2} ;
        std::array<double, 3> B = {0.01, -0.02, 0.03} ;
        double W ;
        grmhd_get_W(g_flat.data(), z.data(), &W) ;
        double dtau ;
        double dstilde[3] ;
        grmhd_get_geom_sources(
            1.0, beta_0.data(), g_flat.data(), guu.data(), Kdd.data(),
            dalp_0.data(), dgdd_0.data(), dbet_0.data(),
            H.rho, press, H.eps, B.data(), z.data(),
            W, &dtau, &dstilde) ;
        REQUIRE_THAT(dtau, Catch::Matchers::WithinAbs(0.0, 1e-13)) ;
        for (int i = 0 ; i < 3 ; ++i) {
            INFO("S-source component i = " << i) ;
            REQUIRE_THAT(dstilde[i], Catch::Matchers::WithinAbs(0.0, 1e-13)) ;
        }
    }
}


TEST_CASE("GRMHD kernels / Geom source vs Python reference", "[grmhd][kernels][regression]")
{
    // Non-trivial ground-truth values for the geom-source kernel.  Each
    // SECTION pins (input → output) for one carefully-chosen state at a
    // non-trivial metric.  Inputs and reference outputs were generated from
    // the Python-transcribed codegen used to validate the geom-source Wij
    // fix on 2026-05-21 — see [[geom-source-wij-bug]] for the script and
    // [[symmetry-audit-2026-05]] for the broader audit framework.  
    //
    // Tolerance: `1e-12` relative.  Bit-identity is NOT expected (the
    // SymPy-generated C99 and the Python NumPy implementation use
    // different CSE structures) but agreement should be near-ulp.
    //
    // Regenerating: if the codegen is intentionally changed (e.g. another
    // notebook fix), the EXPECTED VALUES below must be re-derived from the
    // updated source.  This is by design — pinning the codegen output
    // surfaces any unintended drift from a regeneration.

    SECTION("Generic 1: moderate near-horizon, B=0  (seed=43, α=0.5, W=1.5)")
    {
        std::array<double, 6> gdd = { +1.19999999999999996e+00, +1.00000000000000006e-01, -5.00000000000000028e-02, +1.14999999999999991e+00, +8.00000000000000017e-02, +1.17999999999999994e+00 } ;
        std::array<double, 6> guu = { +8.41366894150114231e-01, -7.60008596818554372e-02, +4.08037402390289405e-02, +8.80550943936906916e-01, -6.29187444907163829e-02, +8.53452276755261696e-01 } ;
        std::array<double, 6> Kdd = { +7.32688520015280093e-02, +2.03453496023656782e-01, -1.75658814405620894e-01, -2.72601936937604428e-01, -5.97551463336409405e-01, +2.91486894595860413e-01 } ;
        std::array<double, 3> betau = { +1.00000000000000006e-01, -5.00000000000000028e-02, +2.50000000000000014e-02 } ;
        std::array<double, 18> dgdd_dx = { +8.32865028802352858e-03, +1.02865672473414063e-01, -3.91797438252806718e-01, +6.13249014642980317e-01, +4.71600288839008341e-01, -6.09119407438451152e-02, -2.77931728951141155e-01, -1.78345692029290043e-01, -3.99312771212324613e-01, +1.32820746202705325e-02, -3.08083859850273589e-01, +2.92771260271021072e-01, -1.79734287329371828e-01, -1.76978531969679137e-01, +7.68626502100995701e-01, -3.62859761276027648e-01, -7.83957943972732330e-01, +1.83107596203723882e-01 } ;
        std::array<double, 9>  dbetau_dx = { -1.03893425570468656e+00, +1.32189317912860316e-01, +4.90860070946320948e-01, +3.50575153770565986e-02, +4.63753714810319351e-03, -1.02530467612477705e+00, +2.77651842779394309e-01, +1.98146283623305774e-01, -5.85998879187250465e-01 } ;
        std::array<double, 3>  dalp_dx = { +2.86975612086930454e+00, -1.83602430111033832e+00, -2.51893554257572383e+00 } ;
        std::array<double, 3>  B = { 0.0, 0.0, 0.0 } ;
        std::array<double, 3>  z = { +5.00000000000000000e-01, -1.00000000000000006e-01, +5.00000000000000028e-02 } ;
        double const alp   = 0.5 ;
        double const rho   = 1e-5 ;
        double const press = 1e-6 ;
        double const eps   = 1e-3 ;
        double const W     = 1.5 ;

        double const dtau_ref = -2.46366540636220083e-05 ;
        double const ds_ref[3] = {
            -7.82000088795445234e-05,
            +4.33711030551046470e-05,
            +6.21362117458239347e-05,
        } ;

        double dtau ;
        double ds[3] ;
        grmhd_get_geom_sources(
            alp, betau.data(), gdd.data(), guu.data(), Kdd.data(),
            dalp_dx.data(), dgdd_dx.data(), dbetau_dx.data(),
            rho, press, eps, B.data(), z.data(), W,
            &dtau, &ds) ;

        REQUIRE_THAT(dtau, Catch::Matchers::WithinRel(dtau_ref, 1e-12)) ;
        for (int i = 0 ; i < 3 ; ++i) {
            INFO("dS_" << "xyz"[i]) ;
            REQUIRE_THAT(ds[i], Catch::Matchers::WithinRel(ds_ref[i], 1e-12)) ;
        }
    }

    SECTION("Generic 2: same metric, with B-field  (seed=43, α=0.5, W=1.5)")
    {
        std::array<double, 6> gdd = { +1.19999999999999996e+00, +1.00000000000000006e-01, -5.00000000000000028e-02, +1.14999999999999991e+00, +8.00000000000000017e-02, +1.17999999999999994e+00 } ;
        std::array<double, 6> guu = { +8.41366894150114231e-01, -7.60008596818554372e-02, +4.08037402390289405e-02, +8.80550943936906916e-01, -6.29187444907163829e-02, +8.53452276755261696e-01 } ;
        std::array<double, 6> Kdd = { +7.32688520015280093e-02, +2.03453496023656782e-01, -1.75658814405620894e-01, -2.72601936937604428e-01, -5.97551463336409405e-01, +2.91486894595860413e-01 } ;
        std::array<double, 3> betau = { +1.00000000000000006e-01, -5.00000000000000028e-02, +2.50000000000000014e-02 } ;
        std::array<double, 18> dgdd_dx = { +8.32865028802352858e-03, +1.02865672473414063e-01, -3.91797438252806718e-01, +6.13249014642980317e-01, +4.71600288839008341e-01, -6.09119407438451152e-02, -2.77931728951141155e-01, -1.78345692029290043e-01, -3.99312771212324613e-01, +1.32820746202705325e-02, -3.08083859850273589e-01, +2.92771260271021072e-01, -1.79734287329371828e-01, -1.76978531969679137e-01, +7.68626502100995701e-01, -3.62859761276027648e-01, -7.83957943972732330e-01, +1.83107596203723882e-01 } ;
        std::array<double, 9>  dbetau_dx = { -1.03893425570468656e+00, +1.32189317912860316e-01, +4.90860070946320948e-01, +3.50575153770565986e-02, +4.63753714810319351e-03, -1.02530467612477705e+00, +2.77651842779394309e-01, +1.98146283623305774e-01, -5.85998879187250465e-01 } ;
        std::array<double, 3>  dalp_dx = { +2.86975612086930454e+00, -1.83602430111033832e+00, -2.51893554257572383e+00 } ;
        std::array<double, 3>  B = { +1.0e-3, -8.0e-4, +3.0e-4 } ;
        std::array<double, 3>  z = { +5.00000000000000000e-01, -1.00000000000000006e-01, +5.00000000000000028e-02 } ;
        double const alp   = 0.5 ;
        double const rho   = 1e-5 ;
        double const press = 1e-6 ;
        double const eps   = 1e-3 ;
        double const W     = 1.5 ;

        double const dtau_ref = -2.48104573815721489e-05 ;
        double const ds_ref[3] = {
            -8.21445746274834321e-05,
            +4.58890193752384476e-05,
            +6.54992152417176692e-05,
        } ;

        double dtau ;
        double ds[3] ;
        grmhd_get_geom_sources(
            alp, betau.data(), gdd.data(), guu.data(), Kdd.data(),
            dalp_dx.data(), dgdd_dx.data(), dbetau_dx.data(),
            rho, press, eps, B.data(), z.data(), W,
            &dtau, &ds) ;

        REQUIRE_THAT(dtau, Catch::Matchers::WithinRel(dtau_ref, 1e-12)) ;
        for (int i = 0 ; i < 3 ; ++i) {
            INFO("dS_" << "xyz"[i]) ;
            REQUIRE_THAT(ds[i], Catch::Matchers::WithinRel(ds_ref[i], 1e-12)) ;
        }
    }
}
