/**
 * @file bns_nurates.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 *         Marie Cassing (mcassing@itp.uni-frankfurt.de)
 * @brief  GRACE interface to the bns_nurates neutrino opacity library.
 *         Runs as a Kokkos device kernel (GPU + CPU portable).
 * @date 2026-03-26
 *
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas.
 * Copyright (C) 2023 Carlo Musolino
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef GRACE_PHYS_BNS_NURATES_HH
#define GRACE_PHYS_BNS_NURATES_HH

#include <grace_config.h>

#include <Kokkos_Core.hpp>

// bns_nurates public headers (added via target_include_directories)
#include <bns_nurates.hpp>
#include <m1_opacities.hpp>
#include <distribution.hpp>


#include <grace/physics/eos/physical_constants.hh>
#include <grace/physics/eos/unit_system.hh>
#include <grace/config/config_parser.hh>

namespace grace {

// =============================================================================
// Device-safe quadrature storage
//
// MyQuadrature holds raw pointers to heap memory, which are not safe to
// capture in a Kokkos lambda on GPU. This struct stores the quadrature
// points and weights in a plain fixed-size array (stack/value-copyable),
// and reconstructs a MyQuadrature pointing into those arrays on demand.
//
// MAXN = 32 is well above the 6-10 points ever used; increase if needed.
// =============================================================================
struct DeviceQuadrature
{
    static constexpr int MAXN = 32;
    double points[MAXN]  = {};
    double weights[MAXN] = {};
    int    n             = 0;

    /// Fill from a host-side MyQuadrature (after GaussLegendre() has been called).
    void fill(const MyQuadrature& q)
    {
        n = q.nx;
        for (int i = 0; i < n; ++i) {
            points[i]  = q.points[i];  // fixed array, direct copy
            weights[i] = q.w[i];       // field is 'w' not 'weights'
        }
    }

    /// Reconstruct a MyQuadrature on device pointing into our internal arrays.
    /// The returned struct is valid only for the lifetime of this DeviceQuadrature.
    KOKKOS_INLINE_FUNCTION
    MyQuadrature to_bns_quad() const
    {
        MyQuadrature q;
        q.nx    = n;
        q.ny    = 1;
        q.nz    = 1;
        q.dim   = 1;
        q.type  = kGauleg;
        q.alpha = 0.0;
        q.x1    = 0.0;  q.x2 = 1.0;
        q.y1    = 0.0;  q.y2 = 0.0;
        q.z1    = 0.0;  q.z2 = 0.0;
        // Both are fixed arrays — copy element by element, no pointer assignment
        for (int i = 0; i < n; ++i) {
            q.points[i] = points[i];
            q.w[i]      = weights[i];
        }
        return q;
    }
};

// =============================================================================
// Host helper: build and return a DeviceQuadrature ready for device capture
// =============================================================================
inline DeviceQuadrature make_device_quadrature(int n_points = 6)
{
    MyQuadrature hq;
    hq.nx    = n_points;
    hq.ny    = 1;
    hq.nz    = 1;
    hq.dim   = 1;
    hq.type  = kGauleg;
    hq.alpha = 0.0;
    hq.x1    = 0.0;
    hq.x2    = 1.0;
    hq.y1    = hq.y2 = 0.0;
    hq.z1    = hq.z2 = 0.0;
    GaussLegendre(&hq);

    DeviceQuadrature dq;
    dq.fill(hq);
    return dq;
}

// =============================================================================
// Unit conversion helpers (bns_nurates CGS/MeV <-> GRACE code units)
//
// bns_nurates internal units (after the 1e21 rescaling in the library):
//   nb   : nm^-3   (baryon number density;  1 nm^-3 = 1e21 cm^-3)
//   J    : MeV nm^-3
//   n    : nm^-3
//   kappa: nm^-1   (multiply by 1e7 to get cm^-1)
//   eta  : MeV nm^-3 s^-1  (energy emissivity; multiply by 1e21 to get MeV cm^-3 s^-1)
//   eta_0: nm^-3 s^-1      (number emissivity; multiply by 1e21 to get cm^-3 s^-1)
//
// GRACE code units follow the GR geometric convention encoded in the
// RHOGF / EPSGF / TIMEGF / LENGTHGF constants below.
// =============================================================================
namespace bns_unit_conv {

// Factors from eas_neutrino_rates_analytic.hh (nu_constants namespace)
// Repeated here for locality; they must stay in sync.
constexpr double LENGTHGF = 6.77269222552442e-6;  // cm  -> code length
constexpr double TIMEGF   = 2.03040204956746e5;   // s   -> code time
constexpr double RHOGF    = 1.61887093132742e-18; // g/cm^3 -> code density
constexpr double EPSGF    = 1.11265005605362e-21; // erg/g  -> code spec. energy
constexpr double mev_to_erg = 1.60217733e-6;      // erg/MeV
constexpr double mnuc_cgs   = 1.6726219e-24;      // g  (proton mass, approx nucleon)
constexpr double avogadro   = 6.02214076e23;      // mol^-1

/// kappa [cm^-1] -> code units [1/code_length]
/// bns_nurates returns nm^-1; multiply by 1e7 first to get cm^-1,
/// then divide by LENGTHGF to get code units.
KOKKOS_INLINE_FUNCTION constexpr double kappa_to_code()
{
    return 1e7 / LENGTHGF;
}

/// Energy emissivity [MeV nm^-3 s^-1] -> code units
/// Steps: MeV->erg (*mev_to_erg), nm^-3->cm^-3 (*1e21),
///        then apply GRACE density/energy/time conversion.
KOKKOS_INLINE_FUNCTION constexpr double eta_E_to_code()
{
    // [MeV nm^-3 s^-1] * 1e21 = [MeV cm^-3 s^-1]
    // [MeV cm^-3 s^-1] * mev_to_erg = [erg cm^-3 s^-1]
    // GRACE energy emissivity = Q * RHOGF * EPSGF / TIMEGF  (matches analytic file)
    return 1e21 * mev_to_erg * RHOGF * EPSGF / TIMEGF;
}

/// Number emissivity [nm^-3 s^-1] -> code units
/// Steps: nm^-3->cm^-3 (*1e21), then GRACE number emissivity conversion.
KOKKOS_INLINE_FUNCTION constexpr double eta_N_to_code()
{
    // [nm^-3 s^-1] * 1e21 = [cm^-3 s^-1]
    // GRACE number emissivity = R * mnuc_cgs * RHOGF / TIMEGF
    return 1e21 * mnuc_cgs * RHOGF / TIMEGF;
}

/// rho [code] -> baryon number density [nm^-3] for bns_nurates input
KOKKOS_INLINE_FUNCTION double rho_code_to_nb_nm3(double rho_code)
{
    // rho [code] / RHOGF = rho [g/cm^3]
    // rho [g/cm^3] * avogadro [mol^-1] ~ nb [cm^-3]  (using mnuc ~ 1/avogadro g)
    // nb [cm^-3] * 1e-21 = nb [nm^-3]
    return (rho_code / RHOGF) * avogadro * 1e-21;
}

/// J [code energy density] -> J [MeV nm^-3] for bns_nurates input
KOKKOS_INLINE_FUNCTION double J_code_to_mev_nm3(double J_code, double uconv_energy_density)
{
    // J_code [code] * uconv_energy_density = J [erg/cm^3]
    // J [erg/cm^3] / mev_to_erg = J [MeV/cm^3]
    // J [MeV/cm^3] * 1e-21 = J [MeV/nm^3]
    return J_code * uconv_energy_density / mev_to_erg * 1e-21;
}

/// n [GRACE NRAD code units] -> n [nm^-3] for bns_nurates input
/// GRACE stores NRAD_code = N_cgs * mnuc_cgs * RHOGF
/// (follows from R_to_code = R_cgs * mnuc_cgs * RHOGF / TIMEGF in the analytic file).
/// Therefore N_cgs [cm^-3] = NRAD_code / (mnuc_cgs * RHOGF),
/// and n_nm3 = N_cgs * 1e-21.
KOKKOS_INLINE_FUNCTION double n_code_to_nm3(double n_code)
{
    return n_code * 1e-21 / (mnuc_cgs * RHOGF);
}

} // namespace bns_unit_conv


// =============================================================================
// Main GRACE kernel functor: compute_bns_nurates_eas
//
// Designed to be called from set_m1_eas as a drop-in alongside the analytic
// and weakhub paths. Captures everything by value so it is safe on device.
// =============================================================================
template <typename eos_t>
struct compute_bns_nurates_eas
{
    // ---- captured state ----
    eos_t           eos;
    var_array_t     aux;
    var_array_t     state;
    DeviceQuadrature dquad;

    // Unit conversion cache (computed once on host, captured by value)
    double uconv_energy_density; // code energy density -> erg/cm^3  (= 1/(RHOGF*EPSGF))

    OpacityFlags  opacity_flags_init;
    OpacityParams opacity_params_init;


    // ---- constructor (host only) ----
    compute_bns_nurates_eas(
        const eos_t&          eos_,
        var_array_t           aux_,
        var_array_t           state_,
        double                mass_scale,
        bool                  beta_decay,
        bool                  bremsstrahlung,
        bool                  pair_annihilation,
        bool                  iso,
        bool                  inelastic,
        int                   n_quad_points = 6)
        : eos(eos_)
        , aux(aux_)
        , state(state_)
        , dquad(make_device_quadrature(n_quad_points))
    {
        using namespace grace::physical_constants;

        // Build a temporary unit_system to get the conversion factors.
        // GRACE code length = Msun_to_cm * mass_scale (same as eas_neutrino_rates_analytic)
        const double Msun_cgs   = 1.988475e33;         // g
        const double G_cgs      = 6.67430e-8;          // cm^3 g^-1 s^-2
        const double c_cgs      = 2.99792458e10;       // cm/s
        const double L_cm       = G_cgs * Msun_cgs / (c_cgs * c_cgs) * mass_scale;
        const double t_s        = L_cm / c_cgs;
        const double M_g        = Msun_cgs * mass_scale;
        const double rho_cgs_   = M_g / (L_cm * L_cm * L_cm);
        const double eps_cgs_   = c_cgs * c_cgs;       // erg/g

        uconv_energy_density = rho_cgs_ * eps_cgs_;    // code -> erg/cm^3  (= 1/(RHOGF*EPSGF))

        opacity_flags_init = opacity_flags_default_none;  // host-only, safe here
        opacity_flags_init.use_abs_em         = beta_decay        ? 1 : 0;
        opacity_flags_init.use_brem           = bremsstrahlung    ? 1 : 0;
        opacity_flags_init.use_pair           = pair_annihilation ? 1 : 0;
        opacity_flags_init.use_iso            = iso               ? 1 : 0;
        opacity_flags_init.use_inelastic_scatt = inelastic        ? 1 : 0;
        
        opacity_params_init = opacity_params_default_none;  // host-only, safe here
        opacity_params_init.use_dU             = 1;
        opacity_params_init.use_decay          = 1;
        opacity_params_init.use_WM_ab          = 1;
        opacity_params_init.use_WM_sc          = 1;
        opacity_params_init.use_NN_medium_corr = 1;
        opacity_params_init.brem_implementation = BREM_GP19;
    }

    // ---- device kernel ----
    KOKKOS_INLINE_FUNCTION
    void operator()(int const i, int const j, int const k, int const q) const
    {
        // ----------------------------------------------------------------
        // 1. Read primitive variables from aux
        // ----------------------------------------------------------------
        const double rhoL  = aux(i, j, k, RHO_,  q);
        const double tempL = aux(i, j, k, TEMP_,  q); // MeV
        const double yeL   = aux(i, j, k, YE_,    q);

        // ----------------------------------------------------------------
        // 2. EOS call: chemical potentials
        // ----------------------------------------------------------------
        double mup = 0., mun = 0.;
        double Xa = 0., Xh = 0., Xn = 0., Xp = 0.;
        double Abar = 1., Zbar = 1.;
        eos_err_t err;
        double rho_loc = rhoL, T_loc = tempL, ye_loc = yeL;
        const double mue = eos.mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye(
            mup, mun, Xa, Xh, Xn, Xp, Abar, Zbar,
            T_loc, rho_loc, ye_loc, err);

        // ----------------------------------------------------------------
        // 3. Metric and M1 primitives via GRACE closure
        // ----------------------------------------------------------------
        metric_array_t metric;
        FILL_METRIC_ARRAY(metric, state, q, i, j, k);

        m1_prims_array_t pnue, pnua, pnux;
        FILL_M1_PRIMS_ARRAY(pnue, state, aux, q, 0, VEC(i, j, k));
        FILL_M1_PRIMS_ARRAY(pnua, state, aux, q, 1, VEC(i, j, k));
        FILL_M1_PRIMS_ARRAY(pnux, state, aux, q, 2, VEC(i, j, k));

        const double inv_sqrtg = 1.0 / metric.sqrtg();
        for (int iv = 0; iv < 5; ++iv) {
            pnue[iv] *= inv_sqrtg;
            pnua[iv] *= inv_sqrtg;
            pnux[iv] *= inv_sqrtg;
        }

        m1_closure_t cle{pnue, metric}, cla{pnua, metric}, clx{pnux, metric};
        cle.update_closure(0, true);
        cla.update_closure(0, true);
        clx.update_closure(0, true);

        // ----------------------------------------------------------------
        // 4. Build GreyOpacityParams (all on device, stack allocated)
        // ----------------------------------------------------------------
        GreyOpacityParams eas_pars;
        eas_pars.opacity_flags = opacity_flags_init;
        eas_pars.opacity_pars  = opacity_params_init;

        // EOS parameters
        // dU and dm_eff: not returned by current GRACE tabulated EOS interface;
        // set to zero (TODO: extend EOS interface when available).
        eas_pars.eos_pars.dU     = 0.0;
        eas_pars.eos_pars.dm_eff = 0.0;
        eas_pars.eos_pars.nb   = bns_unit_conv::rho_code_to_nb_nm3(rhoL); // nm^-3
        eas_pars.eos_pars.temp = tempL;   // MeV
        eas_pars.eos_pars.yp   = yeL;
        eas_pars.eos_pars.yn   = 1.0 - yeL;
        eas_pars.eos_pars.mu_e = mue;     // MeV
        eas_pars.eos_pars.mu_p = mup;     // MeV
        eas_pars.eos_pars.mu_n = mun;     // MeV

        // Unit conversions: code -> bns_nurates input units
        //
        // J [code energy density] -> [MeV nm^-3]:
        //   J_code * (1/(RHOGF*EPSGF)) [erg/cm^3] / mev_to_erg [erg/MeV] * 1e-21 [cm^3/nm^3]
        //
        // n [code number density] -> [nm^-3]:
        //   GRACE stores NRAD such that NRAD_code = N_cgs * mnuc_cgs * RHOGF
        //   (consistent with R_to_code = R_cgs * mnuc_cgs * RHOGF / TIMEGF in analytic file).
        //   Therefore: N_cgs = NRAD_code / (mnuc_cgs * RHOGF)
        //   and: n_nm3 = N_cgs * 1e-21 = NRAD_code / (mnuc_cgs * RHOGF) * 1e-21
        const double fconv_J = uconv_energy_density / bns_unit_conv::mev_to_erg * 1e-21;
        const double fconv_n = 1e-21 / (bns_unit_conv::mnuc_cgs * bns_unit_conv::RHOGF);

        // M1 energy densities J [MeV nm^-3]
        eas_pars.m1_pars.J[id_nue]  = cle.J * fconv_J;
        eas_pars.m1_pars.J[id_anue] = cla.J * fconv_J;
        eas_pars.m1_pars.J[id_nux]  = clx.J * fconv_J;
        eas_pars.m1_pars.J[id_anux] = clx.J * fconv_J; // same species in GRACE

        // M1 number densities n [nm^-3]
        eas_pars.m1_pars.n[id_nue]  = (pnue[NRADL] / cle.Gamma) * fconv_n;
        eas_pars.m1_pars.n[id_anue] = (pnua[NRADL] / cla.Gamma) * fconv_n;
        eas_pars.m1_pars.n[id_nux]  = (pnux[NRADL] / clx.Gamma) * fconv_n;
        eas_pars.m1_pars.n[id_anux] = (pnux[NRADL] / clx.Gamma) * fconv_n; // same species in GRACE

        // Eddington factors from GRACE closure
        eas_pars.m1_pars.chi[id_nue]  = cle.chi;
        eas_pars.m1_pars.chi[id_anue] = cla.chi;
        eas_pars.m1_pars.chi[id_nux]  = clx.chi;
        eas_pars.m1_pars.chi[id_anux] = clx.chi; // same species in GRACE

        // ----------------------------------------------------------------
        // 5. Out-of-equilibrium distribution parameters from actual M1 data
        //    (mwe.cpp Part 2 pattern — feeds actual J, n, chi to bns_nurates)
        // ----------------------------------------------------------------
        eas_pars.distr_pars = CalculateDistrParamsFromM1(
            &eas_pars.m1_pars, &eas_pars.eos_pars);

        // ----------------------------------------------------------------
        // 8. Call bns_nurates
        // ----------------------------------------------------------------
        MyQuadrature dq = dquad.to_bns_quad();
        const M1Opacities eas = ComputeM1Opacities(&dq, &dq, &eas_pars);

        // ----------------------------------------------------------------
        // 9. Convert outputs to GRACE code units and write to aux
        // ----------------------------------------------------------------
        // bns_nurates returns:
        //   kappa / kappa_0  in nm^-1  -> code: * kappa_to_code()
        //   eta              in MeV nm^-3 s^-1  -> code: * eta_E_to_code()
        //   eta_0            in nm^-3 s^-1       -> code: * eta_N_to_code()
        constexpr double kc  = bns_unit_conv::kappa_to_code();   // nm^-1 -> code
        constexpr double ec  = bns_unit_conv::eta_E_to_code();   // MeV nm^-3 s^-1 -> code
        constexpr double enc = bns_unit_conv::eta_N_to_code();   // nm^-3 s^-1 -> code

        // Energy absorption opacity [code]
        aux(i, j, k, KAPPAA1_, q) = eas.kappa_a[id_nue]  * kc;
        aux(i, j, k, KAPPAA2_, q) = eas.kappa_a[id_anue] * kc;
        aux(i, j, k, KAPPAA3_, q) = (eas.kappa_a[id_nux] + eas.kappa_a[id_anux]) * kc;

        // Number absorption opacity [code]
        aux(i, j, k, KAPPAAN1_, q) = eas.kappa_0_a[id_nue]  * kc;
        aux(i, j, k, KAPPAAN2_, q) = eas.kappa_0_a[id_anue] * kc;
        aux(i, j, k, KAPPAAN3_, q) = (eas.kappa_0_a[id_nux] + eas.kappa_0_a[id_anux]) * kc;

        // Scattering opacity [code]
        aux(i, j, k, KAPPAS1_, q) = eas.kappa_s[id_nue]  * kc;
        aux(i, j, k, KAPPAS2_, q) = eas.kappa_s[id_anue] * kc;
        aux(i, j, k, KAPPAS3_, q) = (eas.kappa_s[id_nux] + eas.kappa_s[id_anux]) * kc;

        // Energy emissivity [code]
        aux(i, j, k, ETA1_, q) = eas.eta[id_nue]  * ec;
        aux(i, j, k, ETA2_, q) = eas.eta[id_anue] * ec;
        aux(i, j, k, ETA3_, q) = (eas.eta[id_nux] + eas.eta[id_anux]) * ec;

        // Number emissivity [code]
        aux(i, j, k, ETAN1_, q) = eas.eta_0[id_nue]  * enc;
        aux(i, j, k, ETAN2_, q) = eas.eta_0[id_anue] * enc;
        aux(i, j, k, ETAN3_, q) = (eas.eta_0[id_nux] + eas.eta_0[id_anux]) * enc;

#ifdef M1_NU_FIVESPECIES
        // In 5-species mode NUX above already carries only tau+antitau (2 species).
        // NUMU and NUMUBAR are currently zero in bns_nurates equilibrium mode
        // (no muon EOS). Set them explicitly.
        aux(i, j, k, KAPPAA4_,  q) = 0.0;
        aux(i, j, k, KAPPAAN4_, q) = 0.0;
        aux(i, j, k, KAPPAS4_,  q) = 0.0;
        aux(i, j, k, ETA4_,     q) = 0.0;
        aux(i, j, k, ETAN4_,    q) = 0.0;

        aux(i, j, k, KAPPAA5_,  q) = 0.0;
        aux(i, j, k, KAPPAAN5_, q) = 0.0;
        aux(i, j, k, KAPPAS5_,  q) = 0.0;
        aux(i, j, k, ETA5_,     q) = 0.0;
        aux(i, j, k, ETAN5_,    q) = 0.0;
#endif
    } // operator()
};   // struct compute_bns_nurates_eas


// =============================================================================
// Entry point: set_m1_eas_bns_nurates<eos_t>()
//
// Called from auxiliaries.cpp inside the GRACE_ENABLE_M1 block,
// after m1_eq_system.compute_auxiliaries has run (so closures are available).
// =============================================================================
template <typename eos_t>
void set_m1_eas_bns_nurates(
    grace::var_array_t& state,
    grace::var_array_t& aux)
{
    using namespace grace;

    const double mass_scale =
        grace::get_param<double>("coordinate_system", "mass_scale");
    const bool beta_decay =
        grace::get_param<bool>("m1", "eas", "beta_decay");
    const bool bremsstrahlung =
        grace::get_param<bool>("m1", "eas", "bremsstrahlung");
    const bool pair_annihilation =
        grace::get_param<bool>("m1", "eas", "pair_annihilation");
    // iso and inelastic scattering are always on for neutrino transport
    constexpr bool use_iso      = true;
    constexpr bool use_inelastic = true;
    constexpr int  n_quad        = 6;

    auto eos = eos::get().get_eos<eos_t>();

    // Build the functor on host (constructor initialises quadrature + unit factors)
    compute_bns_nurates_eas<eos_t> functor(
        eos, aux, state, mass_scale,
        beta_decay, bremsstrahlung, pair_annihilation,
        use_iso, use_inelastic, n_quad);

    int64_t nx, ny, nz;
    std::tie(nx, ny, nz) = amr::get_quadrant_extents();
    const int     ngz = amr::get_n_ghosts();
    const int64_t nq  = amr::get_local_num_quadrants();

    Kokkos::MDRangePolicy<Kokkos::Rank<4>, grace::default_execution_space> policy(
        {0,       0,       0,       0 },
        {nx+2*ngz, ny+2*ngz, nz+2*ngz, nq});

    Kokkos::parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "set_m1_eas_bns_nurates"),
        policy, functor);

    Kokkos::fence();
}

} // namespace grace

#endif // GRACE_PHYS_BNS_NURATES_HH
