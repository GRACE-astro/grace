/**
 * @file eas_neutrino_rates_analytic.hh
 * @author Marie Cassing (mcassing@itp.uni-frankfurt.de)
 * @brief Analytic neutrino emissivities/opacities
 * @date 2026-02-02
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *                                    
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *   
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *   
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */

#ifndef GRACE_PHYSICS_NEUTRINO_RATES_ANALYTIC_HH
#define GRACE_PHYSICS_NEUTRINO_RATES_ANALYTIC_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/physical_constants.hh>
#include <grace/physics/eos/unit_system.hh>

#include <grace/config/config_parser.hh>
#include <grace/physics/grace_weakhub_table.hh>

#include <array>
#include <cmath>
#include <cstdint>

// Prefer Kokkos device-safe math overloads for CUDA/HIP builds.
#include <Kokkos_MathematicalFunctions.hpp>

namespace grace {

// -----------------------------------------------------------------------------
// Species indexing 
// -----------------------------------------------------------------------------
enum nu_species : int {
  NUE    = 0,
  NUEBAR,
  NUMU,
  NUMUBAR,
  NUX,
  NUMSPECIES
};

// -----------------------------------------------------------------------------
// Output bundle for a single species (what GRACE stores in aux arrays)
//   - eta_E    : energy emissivity (code units)
//   - eta_N    : number emissivity (code units)
//   - kappa_a  : energy absorption opacity (1/code_length)
//   - kappa_n  : number absorption opacity (1/code_length)
//   - kappa_s  : (energy-weighted) scattering opacity (1/code_length)
// -----------------------------------------------------------------------------
struct nu_rates_out {
  double eta_E{0.0};
  double eta_N{0.0};
  double kappa_a{0.0};
  double kappa_n{0.0};
  double kappa_s{0.0};
};

// -----------------------------------------------------------------------------
// Constants (cgs/MeV)
// -----------------------------------------------------------------------------
namespace nu_constants {

constexpr double pi        = 3.1415926535897932384626433832795;
constexpr double clight     = 2.99792458e10;      // cm/s
constexpr double sigma_0   = 1.76e-44;           // cm^2
constexpr double alpha    = 1.23;               // axial coupling g_A (FIL naming)
constexpr double hc_mevcm  = 1.23984172e-10;     // MeV*cm
constexpr double me_mev    = 0.510998910;        // MeV
constexpr double mev_to_erg= 1.60217733e-6;      // erg/MeV
constexpr double erg_to_mev= 1.0 / mev_to_erg;

// Added by Ken
constexpr double Msun_cgs = 1.988475e33; // g
constexpr double G_cgs = 6.67430e-8; // g
constexpr double Msun_to_cm = G_cgs * Msun_cgs / (clight*clight); // g
constexpr double Msun_to_s = Msun_to_cm / clight;
constexpr double Msun_to_erg = Msun_cgs*clight*clight; // erg
constexpr double eV_to_erg = 1.602176634e-12;                 // erg/eV
constexpr double eV_to_g   = eV_to_erg / (clight * clight);   // g/eV
constexpr double MeV_to_g  = eV_to_g * 1e6;                   // g/MeV
constexpr double mp_MeV    = 938.27208943;                    // MeV
constexpr double mp_cgs    = mp_MeV * MeV_to_g;               // g
constexpr double mnuc_cgs  = mp_cgs;                          // nucleon mass (approx)
constexpr double avogadro = 6.02214076e23; // 1/mol

// LENGTHGF: cm -> CU  (1 cm = LENGTHGF CU)
constexpr double LENGTHGF = 6.77269222552442e-6;   // cm to CU
// TIMEGF: s -> CU
constexpr double TIMEGF   = 2.03040204956746e5;    // s to CU
// RHOGF: g/cm^3 -> CU
constexpr double RHOGF    = 1.61887093132742e-18;  // g/cm^3 to CU
// EPSGF: erg/g -> CU  (= 1/c^2)
constexpr double EPSGF    = 1.11265005605362e-21;  // erg/g to CU


// Neutron-proton mass difference (MeV)
constexpr double Qnp = 1.29333236;

// Weak couplings in Ruffert+ 
// Cv = 1/2 + 2 sin^2(theta_W) ~ 0.96 ; Ca = 1/2.
constexpr double Cv  = 0.96;
constexpr double Ca  = 0.50;

// Fine structure constant
constexpr double fsc = 1.0 / 137.035999084;

// Plasmon constant gamma_0 (Ruffert+)?
// used as gamma = gamma_0 * sqrt( (pi^2 + 3 eta_e^2)/3 ).
constexpr double gamma_0 = 5.565e-2;

// Beta constant (like in FIL, Ruffert)
constexpr double beta =
    pi * clight * (1.0 + 3.0 * alpha * alpha) * sigma_0 /
    (hc_mevcm * hc_mevcm * hc_mevcm * me_mev * me_mev);

} // namespace nu_constants

// -----------------------------------------------------------------------------
// Small math helpers 
// -----------------------------------------------------------------------------
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double safe_pos(double x, double tiny = 1e-80) {
    return (x > tiny ? x : tiny);
}

template <int P>
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double ipow(double x) {
    if constexpr (P == 0) return 1.0;
    if constexpr (P == 1) return x;
    if constexpr (P == 2) return x * x;
    if constexpr (P == 3) return x * x * x;
    if constexpr (P == 4) { double x2 = x * x; return x2 * x2; }
    if constexpr (P == 5) { double x2 = x * x; return x2 * x2 * x; }
    if constexpr (P == 6) { double x3 = x * x * x; return x3 * x3; }
    double r = 1.0;
    for (int i = 0; i < P; ++i) r *= x;
    return r;
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE bool finite_pos(double x) {
    return (x > 0.0) && ::isfinite(x);
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double safe_inv_pos_finite(double x) {
    // returns 1/x if x is finite and > 0, else 0
    return finite_pos(x) ? (1.0 / x) : 0.0;
}
// -----------------------------------------------------------------------------
// Unit conversions (GRACE code units <-> cgs/MeV)
// -----------------------------------------------------------------------------

// Code length L = Msun_to_cm * mass_scale
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double code_length_to_cm(double mass_scale) {
    using namespace grace::physical_constants;
    return nu_constants::Msun_to_cm * mass_scale;
}

// Code time t = Msun_to_s * mass_scale
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double code_time_to_s(double mass_scale) {
    using namespace grace::physical_constants;
    return nu_constants::Msun_to_s * mass_scale;
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double rho_code_to_cgs(double rho_code, double mass_scale) {
    using namespace grace::physical_constants;
    const double L = code_length_to_cm(mass_scale);
    //return rho_code * nu_constants::Msun_cgs / (L * L * L);
    return rho_code / nu_constants::RHOGF;
}

// Temperature conversion, ok already in MeV
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double temp_code_to_mev(double temp_code) {
    using namespace grace::physical_constants;
    return temp_code;
    // constexpr double erg_per_mev = 1.602176634e-6;
    // return (k_cgs * temp_code) / erg_per_mev;
}

// Q [MeV cm^-3 s^-1] -> code energy emissivity
//GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double Q_mev_to_code(double Q_mev, double mass_scale) {
//    using namespace grace::physical_constants;
//    const double L = code_length_to_cm(mass_scale);
//    const double t = code_time_to_s(mass_scale);
//    const double Q_erg = Q_mev * nu_constants::mev_to_erg;
//    return Q_erg * (L * L * L) * t / nu_constants::Msun_to_erg;
//}
//
//// R [cm^-3 s^-1] -> code number emissivity
//GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double R_to_code(double R_cgs, double mass_scale) {
//    const double L = code_length_to_cm(mass_scale);
//    const double t = code_time_to_s(mass_scale);
//    return R_cgs * (L * L * L) * t * (nu_constants::mnuc_cgs / nu_constants::Msun_cgs); // KEN added * (mnuc_cgs / Msun_cgs)
//}
//
//// kappa [cm^-1] -> code [1/code_length]
//GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double kappa_to_code(double kappa_cgs, double mass_scale) {
//    const double L = code_length_to_cm(mass_scale);
//    return kappa_cgs * L;
//}

// Q [MeV cm^-3 s^-1] -> code units
// Matches Margherita: Q_cactus = Q * mev_to_erg * RHOGF * EPSGF / TIMEGF
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double Q_mev_to_code(double Q_mev, double /*mass_scale*/) {
    using namespace nu_constants;
    return Q_mev * mev_to_erg * RHOGF * EPSGF / TIMEGF;
}

// R [cm^-3 s^-1] -> code units
// Matches Margherita: R_cactus = R * mnuc_cgs * RHOGF / TIMEGF
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double R_to_code(double R_cgs, double /*mass_scale*/) {
    using namespace nu_constants;
    return R_cgs * mnuc_cgs * RHOGF / TIMEGF;
}

// kappa [cm^-1] -> code units [1/code_length]
// Matches Margherita: kappa_cactus = kappa / LENGTHGF
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double kappa_to_code(double kappa_cgs, double /*mass_scale*/) {
    using namespace nu_constants;
    return kappa_cgs / LENGTHGF;
}


// -----------------------------------------------------------------------------
// Fermi-Dirac integrals: Takahashi+ (1978) fits 
// FD2...FD5 and the ratios.
// -----------------------------------------------------------------------------
namespace fermi {

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE constexpr double eta_min() { return 1.e-3; }

template <int N>
struct FD_no_exp_inv;

//GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD2_get(double eta) {
//    const double e = Kokkos::exp(eta);
//    if (eta > eta_min()) {
//        const double num = 0.5 * ipow<3>(eta) + 2.1646 * eta;
//        const double den = 1.0 - Kokkos::exp(-1.6854 * eta);
//        return num / den;
//    }
//    return 2.0 * e / (1.0 + 0.1604 * Kokkos::exp(0.8612 * eta));
//}
//template <> struct FD_no_exp_inv<2> {
//GRACE_HOST_DEVICE static GRACE_ALWAYS_INLINE double get(double eta) {
//    return 0.5 * (1.0 + 0.1604 * Kokkos::exp(0.8612 * eta));
//}
//};
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD2_get(double eta) {
    if (eta > eta_min()) {
        return (3.2899 * eta + (1.0/3.0) * eta * eta * eta) /
               (1.0 - Kokkos::exp(-1.8246 * eta));
    }
    return 2.0 * Kokkos::exp(eta) / (1.0 + 0.1092 * Kokkos::exp(0.8908 * eta));
}
template <> struct FD_no_exp_inv<2> {
GRACE_HOST_DEVICE static GRACE_ALWAYS_INLINE double get(double eta) {
    return 0.5 * (1.0 + 0.1092 * Kokkos::exp(0.8908 * eta));
}
};


//GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD3_get(double eta) {
//  const double e = Kokkos::exp(eta);
//  if (eta > eta_min()) {
//    const double num = (1.0/3.0) * ipow<3>(eta) + 2.8244 * eta + 5.6804;
//    const double den = 1.0 + Kokkos::exp(-1.9039 * eta);
//    return num / den;
//  }
//  return 6.0 * e / (1.0 + 0.0559 * Kokkos::exp(0.9069 * eta));
//}
//template <> struct FD_no_exp_inv<3> {
//GRACE_HOST_DEVICE static GRACE_ALWAYS_INLINE double get(double eta) {
//    return (1.0/6.0) * (1.0 + 0.0559 * Kokkos::exp(0.9069 * eta));
//}
//};

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD3_get(double eta) {
    if (eta > eta_min()) {
        const double eta2 = eta * eta;
        return (11.3644 + 4.9348 * eta2 + 0.25 * eta2 * eta2) /
        (1.0 + Kokkos::exp(-1.9039 * eta));
    }
    return 6.0 * Kokkos::exp(eta) / (1.0 + 0.0559 * Kokkos::exp(0.9069 * eta));
}
template <> struct FD_no_exp_inv<3> {
    GRACE_HOST_DEVICE static GRACE_ALWAYS_INLINE double get(double eta) {
        return (1.0/6.0) * (1.0 + 0.0559 * Kokkos::exp(0.9069 * eta));
    }
};

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD4_get(double eta) {
    const double e = Kokkos::exp(eta);
    if (eta > eta_min()) {
        const double num = 0.2 * ipow<5>(eta) + 6.5797 * ipow<3>(eta) + 45.4576 * eta;
        const double den = 1.0 - Kokkos::exp(-1.9484 * eta);
        return num / den;
    }
    return 24.0 * e / (1.0 + 0.0287 * Kokkos::exp(0.9257 * eta));
}
template <> struct FD_no_exp_inv<4> {
GRACE_HOST_DEVICE static GRACE_ALWAYS_INLINE double get(double eta) {
    return (1.0/24.0) * (1.0 + 0.0287 * Kokkos::exp(0.9257 * eta));
}
};

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD5_get(double eta) {
    const double e = Kokkos::exp(eta);
    if (eta > eta_min()) {
        const double num = (1.0/6.0) * ipow<5>(eta) + 8.2247 * ipow<3>(eta) + 113.6439 * eta + 236.5323;
        const double den = 1.0 + Kokkos::exp(-1.9727 * eta);
        return num / den;
    }
    return 120.0 * e / (1.0 + 0.0147 * Kokkos::exp(0.9431 * eta));
}
template <> struct FD_no_exp_inv<5> {
GRACE_HOST_DEVICE static GRACE_ALWAYS_INLINE double get(double eta) {
    return (1.0/120.0) * (1.0 + 0.0147 * Kokkos::exp(0.9431 * eta));
}
};

template <int N>
struct FD { GRACE_HOST_DEVICE static GRACE_ALWAYS_INLINE double get(double eta); };
template <> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD<2>::get(double eta) { return FD2_get(eta); }
template <> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD<3>::get(double eta) { return FD3_get(eta); }
template <> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD<4>::get(double eta) { return FD4_get(eta); }
template <> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double FD<5>::get(double eta) { return FD5_get(eta); }

template <int N, int M>
struct FDR {
GRACE_HOST_DEVICE static GRACE_ALWAYS_INLINE double get(double eta) {
    if (eta > eta_min()) return FD<N>::get(eta) / FD<M>::get(eta);
    return FD_no_exp_inv<M>::get(eta) / FD_no_exp_inv<N>::get(eta);
}
};

} // namespace fermi

// -----------------------------------------------------------------------------
// Fugacity/composition state 
// Similar to FIL's Fugacities<T>
// -----------------------------------------------------------------------------
struct fugacity_state {
    double rho_code{0.0};
    double temp_code{0.0};
    double ye{0.0};
    double ymu{0.0};
    double mass_scale{1.0};

    double rho_cgs{0.0};
    double temp_mev{0.0};

    double mu_e{0.0};
    double mu_p{0.0};
    double mu_n{0.0};
    double Xa{0.0}, Xh{0.0}, Xn{0.0}, Xp{0.0};
    double Abar{1.0}, Zbar{1.0};

    std::array<double, NUMSPECIES> eta_nu{{0,0,0,0,0}};
    double eta_e{0.0};
    double eta_hat{0.0};
    std::array<double, NUMSPECIES> tau_n{{0,0,0,0,0}};

    double nb{0.0};
    double eta_np{0.0};
    double eta_pn{0.0};
};

// -----------------------------------------------------------------------------
// Blackbody (equilibrium) densities for Kirchhoff law, in units:
//   - number:  [cm^-3]
//   - energy:  [MeV cm^-3]
// -----------------------------------------------------------------------------
//GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double black_body_number(double g_nu, double T_mev, double eta) {
//    using namespace nu_constants;
//    const double T = safe_pos(T_mev);
//    const double pref = g_nu / (2.0 * pi * pi) * ipow<3>(T / hc_mevcm);
//    return pref * fermi::FD<2>::get(eta);
//}
//
//GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double black_body_energy(double g_nu, double T_mev, double eta) {
//    using namespace nu_constants;
//    const double T = safe_pos(T_mev);
//    const double pref = g_nu / (2.0 * pi * pi) * ipow<3>(T / hc_mevcm);
//    return pref * T * fermi::FD<3>::get(eta);
//}
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double black_body_number(double g_nu, double T_mev, double eta) {
    using namespace nu_constants;
    const double T = safe_pos(T_mev);
    // 4π·c/(ħc)³ · T³ · F₂  →  [cm⁻² s⁻¹], matches Fortran BB_n_mev convention
    // κ_n = R [cm⁻³ s⁻¹] / B_n [cm⁻² s⁻¹] = [cm⁻¹]  ✓
    const double pref = g_nu * 4.0 * pi * clight / (hc_mevcm * hc_mevcm * hc_mevcm);
    return pref * T * T * T * fermi::FD<2>::get(eta);
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double black_body_energy(double g_nu, double T_mev, double eta) {
    using namespace nu_constants;
    const double T = safe_pos(T_mev);
    // 4π·c/(ħc)³ · T⁴ · F₃  →  [MeV cm⁻² s⁻¹], matches Fortran BB_e_mev convention
    // κ_a = Q [MeV cm⁻³ s⁻¹] / B_e [MeV cm⁻² s⁻¹] = [cm⁻¹]  ✓
    const double pref = g_nu * 4.0 * pi * clight / (hc_mevcm * hc_mevcm * hc_mevcm);
    return pref * T * T * T * T * fermi::FD<3>::get(eta);
}


// -----------------------------------------------------------------------------
// Optical depth (tau) handling 
//  - tau_init : an approximate tau used to suppress fugacities
//  - tau_post : an optional post-rate estimate (e.g. kappa_tot * dr)
// Policy interface:
//   double tau_init(double rho_code, const double* xyz_code, double mass_scale,
//                   int species, double rho_cgs) const;
//   double tau_post(double kappa_tot_cgs, double rho_code, const double* xyz_code,
//                   double mass_scale, int species, double rho_cgs) const;
// -----------------------------------------------------------------------------

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double compute_analytic_tau_from_rho_cgs(double rho_cgs) {
    // Deaton+ 2013 fit (log10 tau vs log10 rho) for cold NS-like profiles.
    const double rcgs = safe_pos(rho_cgs);
    const double log10_tau = 0.96 * ((Kokkos::log(rcgs) / Kokkos::log(10.0)) - 11.7);
    const double tau = Kokkos::exp(Kokkos::log(10.0) * log10_tau);
    return (::isfinite(tau) && tau > 0.0) ? tau : 0.0;
}

struct tau_policy_none {
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double tau_init(double /*rho_code*/, const double* /*xyz_code*/,
                                           double /*mass_scale*/, int /*species*/,
                                           double /*rho_cgs*/) const {
        return 0.0;
    }
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double tau_post(double /*kappa_tot_cgs*/, double /*rho_code*/,
                                           const double* /*xyz_code*/, double /*mass_scale*/,
                                           int /*species*/, double /*rho_cgs*/) const {
        return 0.0;
        }
};

struct tau_policy_analytic_density {
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double tau_init(double /*rho_code*/, const double* /*xyz_code*/,
                                           double /*mass_scale*/, int /*species*/,
                                           double rho_cgs) const {
        return compute_analytic_tau_from_rho_cgs(rho_cgs);
    }
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double tau_post(double /*kappa_tot_cgs*/, double /*rho_code*/,
                                           const double* /*xyz_code*/, double /*mass_scale*/,
                                           int /*species*/, double rho_cgs) const {
        return compute_analytic_tau_from_rho_cgs(rho_cgs);
        }
};

struct tau_policy_local_spherical {
    // Outer radius in code units (same coordinates as xyz). If <=0, tau_post returns 0.
    double r_outer_code{0.0};
    // Optional seed for tau_init.
    bool seed_with_analytic{true};

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double tau_init(double /*rho_code*/, const double* /*xyz_code*/,
                                           double /*mass_scale*/, int /*species*/,
                                           double rho_cgs) const {
        return seed_with_analytic ? compute_analytic_tau_from_rho_cgs(rho_cgs) : 0.0;
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double tau_post(double kappa_tot_cgs, double /*rho_code*/,
                                           const double* xyz_code, double mass_scale,
                                           int /*species*/, double /*rho_cgs*/) const {
        if (!(r_outer_code > 0.0) || !(kappa_tot_cgs > 0.0)) return 0.0;
        const double r_code = Kokkos::sqrt(xyz_code[0]*xyz_code[0] + xyz_code[1]*xyz_code[1] + xyz_code[2]*xyz_code[2]);
        const double dr_code = (r_outer_code > r_code) ? (r_outer_code - r_code) : 0.0;
        const double dr_cm = dr_code * code_length_to_cm(mass_scale);
        const double tau = kappa_tot_cgs * dr_cm;
        return (::isfinite(tau) && tau > 0.0) ? tau : 0.0;
    }
};

template <typename eos_t, typename tau_policy_t>
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE fugacity_state make_fugacity_state(
    const eos_t& eos,
    double rho_code,
    double temp_code,
    double ye,
    double ymu,
    double mass_scale,
    const double* xyz_code,
    const tau_policy_t& tau_policy)
{
    using namespace nu_constants;
    fugacity_state F;
    F.rho_code = rho_code;
    F.temp_code = temp_code;
    F.ye = ye;
    F.ymu = ymu;
    F.mass_scale = mass_scale;

    F.rho_cgs  = rho_code_to_cgs(rho_code, mass_scale);
    F.temp_mev = temp_code_to_mev(temp_code);

    double rho_eos = rho_code;
    double T_eos   = temp_code;
    double ye_eos  = ye;
    eos_err_t err;
    // TODO!! EOS framework and get chm. pot!
    // Called it here like in FIL
    F.mu_e = eos.mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye(
        F.mu_p, F.mu_n,
        F.Xa, F.Xh, F.Xn, F.Xp,
        F.Abar, F.Zbar,
        T_eos, rho_eos, ye_eos, err);
    // TODO MUONS: when leptonic eos call here to get eta_mu
    // FIL: eta[NUMU] = eta[MUON] + eta[PROTON] - eta[NEUTRON] - Qnp;

    F.Xa = (F.Xa > 0.0 ? F.Xa : 0.0);
    F.Xh = (F.Xh > 0.0 ? F.Xh : 0.0);
    F.Xn = (F.Xn > 0.0 ? F.Xn : 0.0);
    F.Xp = (F.Xp > 0.0 ? F.Xp : 0.0);
    if (!(::isfinite(F.Abar) && F.Abar > 0.0)) F.Abar = 1.0;
    if (!(::isfinite(F.Zbar) && F.Zbar > 0.0)) F.Zbar = 1.0;

    using namespace grace::physical_constants;
    F.nb = safe_pos(F.rho_cgs) * nu_constants::avogadro ;// nu_constants::mnuc_cgs;

    // ---- DEBUG ----
    if(rho_code > 1.23857e-03 && false ) {
        Kokkos::printf("=== make_fugacity_state CENTER DEBUG ===\n");
        Kokkos::printf("  rho_code = %e\n", rho_code);
        Kokkos::printf("  rho_cgs  = %e\n", F.rho_cgs);
        Kokkos::printf("  temp_mev = %e\n", F.temp_mev);
        Kokkos::printf("  ye       = %e\n", ye);
        Kokkos::printf("  mu_e     = %e\n", F.mu_e);
        Kokkos::printf("  mu_p     = %e\n", F.mu_p);
        Kokkos::printf("  mu_n     = %e\n", F.mu_n);
        Kokkos::printf("  Xn       = %e\n", F.Xn);
        Kokkos::printf("  Xp       = %e\n", F.Xp);
        Kokkos::printf("  nb       = %e\n", F.nb);
    }
    // ---- end of first debug block, put the rest AFTER eta_np/eta_pn are computed ----

    const double mu_hat = F.mu_n - F.mu_p - Qnp;
    const double mu_nue = F.mu_e + F.mu_p - F.mu_n - Qnp;
    const double mu_nuebar = -mu_nue;

    // Muonic species: default heavy-lepton neutrinos have zero equilibrium
    // chemical potential.  A genuine 5-species muonic-EOS path must replace
    // mu_numu by
    //     mu_mu + mu_p - mu_n - Qnp
    // and mu_numubar by -mu_numu.  The current GRACE EOS call above does not
    // return mu_mu, so the analytic 5-species mode is non-muonic unless that
    // EOS interface is extended.
    double mu_numu = 0.0;
    double mu_numubar = 0.0;
    double mu_nux = 0.0;

    const double T = safe_pos(F.temp_mev);
    F.eta_e   = F.mu_e / T;
    F.eta_hat = mu_hat / T;
    F.eta_nu[NUE]     = mu_nue / T;
    F.eta_nu[NUEBAR]  = mu_nuebar / T;
    F.eta_nu[NUMU]    = mu_numu / T;
    F.eta_nu[NUMUBAR] = mu_numubar / T;
    F.eta_nu[NUX]     = mu_nux / T;

    const double Yp = F.Xp;
    const double Yn = F.Xn;
    const double denom_np = (Kokkos::exp(-F.eta_hat) - 1.0);
    const double denom_pn = (Kokkos::exp(+F.eta_hat) - 1.0);
    F.eta_np = (denom_np != 0.0) ? (F.nb * (Yp - Yn) / denom_np) : 0.0;
    F.eta_pn = (denom_pn != 0.0) ? (F.nb * (Yn - Yp) / denom_pn) : 0.0;
    if (F.rho_cgs < 2.e11) {
        F.eta_pn = F.nb * Yp;
        F.eta_np = F.nb * Yn;
    }
    if (!::isfinite(F.eta_np) || !(F.eta_np > 0.0)) F.eta_np = 0.0;
    if (!::isfinite(F.eta_pn) || !(F.eta_pn > 0.0)) F.eta_pn = 0.0;

    // ---- DEBUG ----
    if(rho_code > 1.23857e-03 && false) {
        Kokkos::printf("  eta_hat  = %e\n", F.eta_hat);
        Kokkos::printf("  eta_e    = %e\n", F.eta_e);
        Kokkos::printf("  eta_nu[NUE]    = %e\n", F.eta_nu[NUE]);
        Kokkos::printf("  eta_nu[NUEBAR] = %e\n", F.eta_nu[NUEBAR]);
        Kokkos::printf("  eta_pn   = %e  (proton phase-space factor; drives NUE emission)\n", F.eta_pn);
        Kokkos::printf("  eta_np   = %e  (neutron phase-space factor; drives NUEBAR emission)\n", F.eta_np);
        Kokkos::printf("========================================\n");
    }
    // ---- end DEBUG ----


    // ---------------------------------------------------------------------------
    // Optical-depth based suppression of neutrino fugacities (Foucart/Bollig trick):
    //   eta_nu -> eta_nu * (1 - exp(-tau_nu)) interpolates thin/thick limits.
    // ---------------------------------------------------------------------------
    for (int s = 0; s < NUMSPECIES; ++s) {
        const double tau0 = tau_policy.tau_init(rho_code, xyz_code, mass_scale, s, F.rho_cgs);
        F.tau_n[s] = (::isfinite(tau0) && tau0 > 0.0) ? tau0 : 0.0;
    }

    // Apply suppression (electron types always; muon types only when 5 species)
    {
        const double fac_nue    = 1.0 - Kokkos::exp(-F.tau_n[NUE]);
        const double fac_nuebar = 1.0 - Kokkos::exp(-F.tau_n[NUEBAR]);
        if (::isfinite(fac_nue))    F.eta_nu[NUE]    *= fac_nue;
        if (::isfinite(fac_nuebar)) F.eta_nu[NUEBAR] *= fac_nuebar;

        #ifdef M1_NU_FIVESPECIES
        const double fac_numu    = 1.0 - Kokkos::exp(-F.tau_n[NUMU]);
        const double fac_numubar = 1.0 - Kokkos::exp(-F.tau_n[NUMUBAR]);
        if (::isfinite(fac_numu))    F.eta_nu[NUMU]    *= fac_numu;
        if (::isfinite(fac_numubar)) F.eta_nu[NUMUBAR] *= fac_numubar;
        #endif
    }

    return F;
    }

// -----------------------------------------------------------------------------
// Microphysics kernels (analytic rates)
// All rates are returned in units:
//   - Q: MeV cm^-3 s^-1
//   - R: cm^-3 s^-1
//   - opacities: cm^-1
// -----------------------------------------------------------------------------
struct rates_accum {
    std::array<double, NUMSPECIES> Q{{0,0,0,0,0}};
    std::array<double, NUMSPECIES> R{{0,0,0,0,0}};
    std::array<double, NUMSPECIES> kappa_a{{0,0,0,0,0}};
    std::array<double, NUMSPECIES> kappa_n{{0,0,0,0,0}};
    std::array<double, NUMSPECIES> kappa_s{{0,0,0,0,0}};
};

struct nu_rates_all_out {
    std::array<nu_rates_out, NUMSPECIES> out;
};

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE void add_charged_current_absorption_opacity(const fugacity_state& F, rates_accum& out) {
    using namespace nu_constants;
    constexpr double abs_const = 0.25 * (1.0 + 3.0 * alpha * alpha) * sigma_0;
    std::array<double, NUMSPECIES> zeta{{0,0,0,0,0}};

    const double block_n = 1.0 + Kokkos::exp(F.eta_e - fermi::FDR<5,4>::get(F.eta_nu[NUE]));
    const double absorb_e = (block_n > 0.0) ? (F.eta_np * abs_const / block_n) : 0.0;
    if (finite_pos(absorb_e)) zeta[NUE] += absorb_e;

    const double block_p = 1.0 + Kokkos::exp(-F.eta_e - fermi::FDR<5,4>::get(F.eta_nu[NUEBAR]));
    const double absorb_a = (block_p > 0.0) ? (F.eta_pn * abs_const / block_p) : 0.0;
    if (finite_pos(absorb_a)) zeta[NUEBAR] += absorb_a;

    const double tfac2 = ipow<2>(F.temp_mev / me_mev);
    for (int i = NUE; i < NUX; ++i) {
        const double nfac = tfac2 * fermi::FDR<4,2>::get(F.eta_nu[i]);
        const double efac = tfac2 * fermi::FDR<5,3>::get(F.eta_nu[i]);
        out.kappa_n[i] += zeta[i] * nfac;
        out.kappa_a[i] += zeta[i] * efac;
    }
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE void add_scattering_opacity(const fugacity_state& F, rates_accum& out) {
    using namespace nu_constants;
    constexpr double Cs_n = (1.0 + 5.0 * alpha * alpha) / 24.0 * sigma_0;
    constexpr double Cs_p = (4.0 * (Cv - 1.0) * (Cv - 1.0) + 5.0 * alpha * alpha) / 24.0 * sigma_0;
    double C_nucleus = (1.0/16.0) * sigma_0 * F.Abar * ipow<2>(1.0 - F.Zbar / F.Abar);
    if (!::isfinite(C_nucleus)) C_nucleus = 0.0;

    double ymp = F.Xp / (1.0 + (2.0/3.0) * Kokkos::max(0.0, F.mu_p / safe_pos(F.temp_mev)));
    double ymn = F.Xn / (1.0 + (2.0/3.0) * Kokkos::max(0.0, F.mu_n / safe_pos(F.temp_mev)));
    if (!::isfinite(ymp)) ymp = F.Xp;
    if (!::isfinite(ymn)) ymn = F.Xn;

    const double tfac2 = ipow<2>(F.temp_mev / me_mev);
    for (int i = NUE; i <= NUX; ++i) {
        const double efac = tfac2 * fermi::FDR<5,3>::get(F.eta_nu[i]);
        double ks = 0.0;
        ks += F.nb * Cs_n * ymn;
        ks += F.nb * Cs_p * ymp;
        ks += C_nucleus * F.nb * F.Xh;
        ks *= efac;
        if (finite_pos(ks)) out.kappa_s[i] += ks;
    }
}

// -----------------------------------------------------------------------------
// Charged-current emission 
// -----------------------------------------------------------------------------
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE void add_charged_current_emission(const fugacity_state& F, rates_accum& out) {
    using namespace nu_constants;

    // Ruffert+ (B3-4)
    double block_factor_nue = 1.0 + Kokkos::exp(F.eta_nu[NUE] - fermi::FDR<5,4>::get(F.eta_e));
    double block_factor_anue = 1.0 + Kokkos::exp(F.eta_nu[NUEBAR] - fermi::FDR<5,4>::get(-F.eta_e));
    if (!(block_factor_nue > 0.0))  block_factor_nue  = 1.0;
    if (!(block_factor_anue > 0.0)) block_factor_anue = 1.0;

    // Ruffert+ (B1-2) and (B15)
    // e- + p -> n + nue      : use eta_pn (proton availability)
    // e+ + n -> p + nuebar   : use eta_np (neutron availability)
    const double Re = beta * F.eta_pn * ipow<5>(F.temp_mev) * fermi::FD<4>::get(F.eta_e)  / block_factor_nue;
    const double Ra = beta * F.eta_np * ipow<5>(F.temp_mev) * fermi::FD<4>::get(-F.eta_e) / block_factor_anue;
    const double Qe = beta * F.eta_pn * ipow<6>(F.temp_mev) * fermi::FD<5>::get(F.eta_e)  / block_factor_nue;
    const double Qa = beta * F.eta_np * ipow<6>(F.temp_mev) * fermi::FD<5>::get(-F.eta_e) / block_factor_anue;

    if (Re > 0.0 && ::isfinite(Re)) out.R[NUE]    += Re;
    if (Ra > 0.0 && ::isfinite(Ra)) out.R[NUEBAR] += Ra;
    if (Qe > 0.0 && ::isfinite(Qe)) out.Q[NUE]    += Qe;
    if (Qa > 0.0 && ::isfinite(Qa)) out.Q[NUEBAR] += Qa;
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE void add_pair_process_emission(const fugacity_state& F, rates_accum& out) {
    using namespace nu_constants;

    std::array<double, NUMSPECIES> block{{1,1,1,1,1}};
    const double eta_m = fermi::FDR<4,3>::get(F.eta_e);
    const double eta_p = fermi::FDR<4,3>::get(-F.eta_e);
    for (int i = NUE; i <= NUX; ++i) {
        block[i] = 1.0 + Kokkos::exp(F.eta_nu[i] - 0.5 * (eta_m + eta_p));
        if (!(block[i] > 0.0)) block[i] = 1.0;
    }

    const double eps_const = 8.0 * pi / ipow<3>(hc_mevcm);
    const double eps_m = eps_const * ipow<4>(F.temp_mev) * fermi::FD<3>::get(F.eta_e);
    const double eps_p = eps_const * ipow<4>(F.temp_mev) * fermi::FD<3>::get(-F.eta_e);
    const double eps_fraction = 0.5 * F.temp_mev * (eta_m + eta_p);

    const double pair_const = sigma_0 * clight / ipow<2>(me_mev) * eps_m * eps_p;

    const double R_pair = pair_const * (ipow<2>(Cv - Ca) + ipow<2>(Cv + Ca)) /
                        (36.0 * block[NUE] * block[NUEBAR]);
    //KEN
    if (::isfinite(R_pair) && (R_pair > 0.0)) {
        //out.R[NUE] += R_pair;
        //out.R[NUEBAR] += R_pair;
        //out.Q[NUE] += R_pair * eps_fraction;
        //out.Q[NUEBAR] += R_pair * eps_fraction;
    }

    const double pair_heavy_coupling =
        ipow<2>(Cv - Ca) + ipow<2>(Cv + Ca - 2.0);

    #ifdef M1_NU_FIVESPECIES
    // In 5-species mode:
    //   NUMU    = nu_mu
    //   NUMUBAR = anti-nu_mu
    //   NUX     = nu_tau + anti-nu_tau
    // Therefore NUX carries only two heavy-lepton species, not all four.
    const double R_pair_x = (1.0/18.0) * pair_const * pair_heavy_coupling /
                          (block[NUX] * block[NUX]);
    #else
    // In 3-species mode NUX represents all four heavy-lepton species.
    const double R_pair_x = (1.0/9.0) * pair_const * pair_heavy_coupling /
                          (block[NUX] * block[NUX]);
    #endif

    if (::isfinite(R_pair_x) && (R_pair_x > 0.0)) {
        out.R[NUX] += R_pair_x;
        out.Q[NUX] += R_pair_x * eps_fraction;
    }

    #ifdef M1_NU_FIVESPECIES
    const double R_pair_numu = (1.0/36.0) * pair_const * pair_heavy_coupling /
                             (block[NUMU] * block[NUMUBAR]);
    if (::isfinite(R_pair_numu) && (R_pair_numu > 0.0)) {
        out.R[NUMU]    += R_pair_numu;
        out.R[NUMUBAR] += R_pair_numu;
        out.Q[NUMU]    += R_pair_numu * eps_fraction;
        out.Q[NUMUBAR] += R_pair_numu * eps_fraction;
    }
    #endif
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE void add_plasmon_decay_emission(const fugacity_state& F, rates_accum& out) {
    using namespace nu_constants;

    const double gamma = gamma_0 * Kokkos::sqrt((pi*pi + 3.0 * F.eta_e * F.eta_e) / 3.0);
    std::array<double, NUMSPECIES> block{{1,1,1,1,1}};
    for (int i = NUE; i <= NUX; ++i) {
        block[i] = 1.0 + Kokkos::exp(F.eta_nu[i] - (1.0 + 0.5 * gamma * gamma / (1.0 + gamma)));
        if (!(block[i] > 0.0)) block[i] = 1.0;
    }

    const double gamma_const =
        ipow<3>(pi) * sigma_0 * clight /
        (ipow<2>(me_mev) * 3.0 * fsc * ipow<6>(hc_mevcm)) *
        ipow<6>(gamma) * Kokkos::exp(-gamma) * (1.0 + gamma) * ipow<8>(F.temp_mev);

    const double R_gamma = ipow<2>(Cv) * gamma_const / (block[NUE] * block[NUEBAR]);
    const double Q_gamma = F.temp_mev * 0.5 * (2.0 + gamma * gamma / (1.0 + gamma));
    // TODO check if plasmon adds to nue, nuebar
    if (::isfinite(R_gamma) && (R_gamma > 0.0)) {
        out.R[NUE] += R_gamma;
        out.R[NUEBAR] += R_gamma;
        out.Q[NUE] += Q_gamma * R_gamma;
        out.Q[NUEBAR] += Q_gamma * R_gamma;
    }

    #ifdef M1_NU_FIVESPECIES
    // In 5-species mode NUX is only tau + anti-tau, so use degeneracy 2.
    const double R_gamma_x = 2.0 * ipow<2>(Cv - 1.0) * gamma_const /
                           (block[NUX] * block[NUX]);
    #else
    // In 3-species mode NUX represents all four heavy-lepton species.
    const double R_gamma_x = 4.0 * ipow<2>(Cv - 1.0) * gamma_const /
                           (block[NUX] * block[NUX]);
    #endif

    if (::isfinite(R_gamma_x) && (R_gamma_x > 0.0)) {
        out.R[NUX] += R_gamma_x;
        out.Q[NUX] += Q_gamma * R_gamma_x;
    }

    #ifdef M1_NU_FIVESPECIES
    const double R_gamma_numu = ipow<2>(Cv - 1.0) * gamma_const /
                              (block[NUMU] * block[NUMUBAR]);
    if (::isfinite(R_gamma_numu) && (R_gamma_numu > 0.0)) {
        out.R[NUMU]    += R_gamma_numu;
        out.R[NUMUBAR] += R_gamma_numu;
        out.Q[NUMU]    += Q_gamma * R_gamma_numu;
        out.Q[NUMUBAR] += Q_gamma * R_gamma_numu;
    }
    #endif
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE void add_brems_emission(const fugacity_state& F, rates_accum& out) {
    using namespace nu_constants;
    const double factorY = 0.5 * (ipow<2>(F.Xn) + ipow<2>(F.Xp) + (28.0/3.0) * F.Xn * F.Xp);
    const double R_brems =
        0.231 * (2.0778e2 * erg_to_mev) * factorY * ipow<2>(F.rho_cgs) * ipow<4>(F.temp_mev) * Kokkos::sqrt(safe_pos(F.temp_mev));
    const double Q_brems = R_brems * F.temp_mev / 0.231 * 0.504;
    if (::isfinite(R_brems) && (R_brems > 0.0) && ::isfinite(Q_brems) && (Q_brems > 0.0)) {
        #ifdef M1_NU_FIVESPECIES
        // In 5-species mode:
        //   NUMU    = one heavy species
        //   NUMUBAR = one heavy species
        //   NUX     = tau + anti-tau, i.e. two heavy species
        out.R[NUMU]    += R_brems;
        out.Q[NUMU]    += Q_brems;
        out.R[NUMUBAR] += R_brems;
        out.Q[NUMUBAR] += Q_brems;
        out.R[NUX]     += 2.0 * R_brems;
        out.Q[NUX]     += 2.0 * Q_brems;
        #else
        // In 3-species mode NUX represents all four heavy-lepton species.
        out.R[NUX] += 4.0 * R_brems;
        out.Q[NUX] += 4.0 * Q_brems;
        #endif
  }
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE void apply_kirchhoff(const fugacity_state& F,
                                             const std::array<double, NUMSPECIES>& g_nu,
                                             rates_accum& out) {
    for (int i = NUE; i <= NUX; ++i) {

        const double Ber_eq = black_body_energy(g_nu[i], F.temp_mev, F.eta_nu[i]);
        const double invBer = safe_inv_pos_finite(Ber_eq);

        // If invBer==0 -> k_a becomes 0 (no divide, no branch)
        const double k_a = out.Q[i] * invBer;
        // Store only if k_a >0 drop this if out.Q is finite I guess
        if (finite_pos(k_a)) out.kappa_a[i] = k_a;

        const double Bn_eq = black_body_number(g_nu[i], F.temp_mev, F.eta_nu[i]);
        const double invBn = safe_inv_pos_finite(Bn_eq);

        const double k_n = out.R[i] * invBn;
        if (finite_pos(k_n)) out.kappa_n[i] = k_n;
    }
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE void add_kirchhoff_absorption_opacity_from_QR(
                        const fugacity_state& F,
                        const std::array<double, NUMSPECIES>& g_nu,
                        const std::array<double, NUMSPECIES>& Q_in,
                        const std::array<double, NUMSPECIES>& R_in,
                        std::array<double, NUMSPECIES>& kappa_a_add,
                        std::array<double, NUMSPECIES>& kappa_n_add) {
    for (int s = NUE; s <= NUX; ++s) {
        const double Ber_eq = black_body_energy(g_nu[s], F.temp_mev, F.eta_nu[s]);
        const double Bn_eq = black_body_number(g_nu[s], F.temp_mev, F.eta_nu[s]);
        const double invBer = safe_inv_pos_finite(Ber_eq);
        const double invBn = safe_inv_pos_finite(Bn_eq);
        const double ka = Q_in[s] * invBer;
        const double kn = R_in[s] * invBn;
        kappa_a_add[s] = finite_pos(ka) ? ka : 0.0;
        kappa_n_add[s] = finite_pos(kn) ? kn : 0.0;
    }
}

GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
void add_kirchhoff_emission_from_absorption_opacity(
    const fugacity_state& F,
    const std::array<double, NUMSPECIES>& g_nu,
    const std::array<double, NUMSPECIES>& kappa_a_in,
    const std::array<double, NUMSPECIES>& kappa_n_in,
    std::array<double, NUMSPECIES>& Q_add,
    std::array<double, NUMSPECIES>& R_add)
{
    for (int s = 0; s < NUMSPECIES; ++s) {
        if (!(g_nu[s] > 0.0)) {
            Q_add[s] = 0.0;
            R_add[s] = 0.0;
            continue;
        }
        const double Ber_eq = black_body_energy(g_nu[s], F.temp_mev, F.eta_nu[s]);
        const double Bn_eq  = black_body_number(g_nu[s], F.temp_mev, F.eta_nu[s]);
        const double Q = kappa_a_in[s] * Ber_eq;
        const double R = kappa_n_in[s] * Bn_eq;
        Q_add[s] = finite_pos(Q) ? Q : 0.0;
        R_add[s] = finite_pos(R) ? R : 0.0;
    }
}

// -----------------------------------------------------------------------------
// compute opacities/emissivities for one species.
// -----------------------------------------------------------------------------

template <typename tau_policy_t>
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE nu_rates_all_out compute_all_species_weakhub(
    const grace::weakhub::device_handle& weakhub,
    double rho_code,
    double temp_code,
    double ye,
    double ymu,
    double mass_scale,
    bool plasmon_decay,
    bool bremsstrahlung,
    bool pair_annihilation,
    const double* xyz_code,
    const tau_policy_t& tau_policy,
    bool apply_temp_correction)
{
    fugacity_state F{};
    F.rho_code = rho_code;
    F.temp_code = temp_code;
    F.ye = ye;
    F.ymu = ymu;
    F.mass_scale = mass_scale;
    F.rho_cgs = rho_code_to_cgs(rho_code, mass_scale);
    F.temp_mev = temp_code_to_mev(temp_code);
    F.eta_nu.fill(0.0);
    F.tau_n.fill(0.0);

    const auto tbl = weakhub.lookup(F.rho_cgs, F.temp_mev, ye, ymu);
    rates_accum rates{};
    for (int s = NUE; s <= NUX; ++s) {
        rates.kappa_a[s] = tbl.kappa_a_en[s];
        rates.kappa_n[s] = tbl.kappa_a_num[s];
        rates.kappa_s[s] = tbl.kappa_s[s];
    }

    std::array<double, NUMSPECIES> g_nu{{1,1,0,0,4}};
    #ifdef M1_NU_FIVESPECIES
    g_nu = {{1,1,1,1,2}};
    #endif
    for (int s = NUE; s <= NUX; ++s) {
        const double Ber_eq = black_body_energy(g_nu[s], F.temp_mev, F.eta_nu[s]);
        const double Bn_eq = black_body_number(g_nu[s], F.temp_mev, F.eta_nu[s]);
        rates.Q[s] = rates.kappa_a[s] * Ber_eq;
        rates.R[s] = rates.kappa_n[s] * Bn_eq;
    }
    
    rates_accum extra{};
    if (pair_annihilation) add_pair_process_emission(F, extra);
    if (plasmon_decay) add_plasmon_decay_emission(F, extra);
    if (bremsstrahlung) add_brems_emission(F, extra);
    
    std::array<double, NUMSPECIES> kappa_a_add{{0,0,0,0,0}}, kappa_n_add{{0,0,0,0,0}};
    add_kirchhoff_absorption_opacity_from_QR(F, g_nu, extra.Q, extra.R, kappa_a_add, kappa_n_add);
    for (int s = NUE; s <= NUX; ++s) {
        rates.Q[s] += extra.Q[s];
        rates.R[s] += extra.R[s];
        rates.kappa_a[s] += kappa_a_add[s];
        rates.kappa_n[s] += kappa_n_add[s];
    }
    #ifdef M1_NU_FIVESPECIES
    if (F.rho_cgs < 1.0e10 || F.temp_mev < 2.5) {
        rates.Q[NUMU] = rates.R[NUMU] = rates.kappa_a[NUMU] = rates.kappa_n[NUMU] = 0.0;
        rates.Q[NUMUBAR] = rates.R[NUMUBAR] = rates.kappa_a[NUMUBAR] = rates.kappa_n[NUMUBAR] = 0.0;
    }
    #endif
    
    for (int s = NUE; s <= NUX; ++s) {
        const double kappa_tot = rates.kappa_a[s] + rates.kappa_s[s];
        (void)tau_policy.tau_post(kappa_tot, rho_code, xyz_code, mass_scale, s, F.rho_cgs);
    }
    
    if (apply_temp_correction) {
        for (int s = NUE; s <= NUX; ++s) {
        const double Rloc = rates.R[s], Qloc = rates.Q[s];
        if (Rloc > 0.0 && Qloc > 0.0) {
            const double eps_mev = Qloc / Rloc;
            const double Tnu_mev = fermi::FDR<2,3>::get(F.eta_nu[s]) * eps_mev;
            double fact = 1.0;
            if (Kokkos::isfinite(Tnu_mev) && Tnu_mev > 0.0) {
            const double ratio = Tnu_mev / safe_pos(F.temp_mev);
            fact = ratio * ratio;
            if (fact < 1.0) fact = 1.0;
            }
            if (Kokkos::isfinite(fact) && fact > 0.0) {
            // TODO is it correct like that?
            if(s == NUX){            
                rates.kappa_s[s] *= fact;
            }else{
                rates.Q[s]       *= fact;
                rates.R[s]       *= fact;
                rates.kappa_a[s] *= fact;
                rates.kappa_n[s] *= fact;
                rates.kappa_s[s] *= fact;
            }
            //rates.Q[s] *= fact; rates.R[s] *= fact;
            //rates.kappa_a[s] *= fact; rates.kappa_n[s] *= fact; rates.kappa_s[s] *= fact;
            }
        }
        }
    }
    
    nu_rates_all_out all{};
    for (int s = NUE; s <= NUX; ++s) {
        nu_rates_out out{};
        out.eta_E   = Q_mev_to_code(rates.Q[s], mass_scale);
        out.eta_N   = R_to_code(rates.R[s], mass_scale);
        out.kappa_a = kappa_to_code(rates.kappa_a[s], mass_scale);
        out.kappa_n = kappa_to_code(rates.kappa_n[s], mass_scale);
        out.kappa_s = kappa_to_code(rates.kappa_s[s], mass_scale);
        if (!Kokkos::isfinite(out.eta_E)) out.eta_E = 0.0;
        if (!Kokkos::isfinite(out.eta_N)) out.eta_N = 0.0;
        if (!Kokkos::isfinite(out.kappa_a)) out.kappa_a = 0.0;
        if (!Kokkos::isfinite(out.kappa_n)) out.kappa_n = 0.0;
        if (!Kokkos::isfinite(out.kappa_s)) out.kappa_s = 0.0;
        all.out[s] = out;
    }
    return all;
}

// -----------------------------------------------------------------------------
// Compute *all* species at once (CUDA-friendly; avoids per-species EOS calls).
// -----------------------------------------------------------------------------
template <typename eos_t, typename tau_policy_t>
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE nu_rates_all_out compute_all_species(
    const eos_t& eos,
    double rho_code,
    double temp_code,
    double ye,
    double ymu,
    double mass_scale,
    bool beta_decay,
    bool plasmon_decay,
    bool bremsstrahlung,
    bool pair_annihilation,
    const double* xyz_code,
    const tau_policy_t& tau_policy,
    bool apply_temp_correction)
{
    fugacity_state F = make_fugacity_state(eos, rho_code, temp_code, ye, ymu, mass_scale, xyz_code, tau_policy);

    rates_accum rates;

    // Charged-current beta processes are handled explicitly:
    //   emissivities from Ruffert/Rosswog formulae,
    //   absorption opacities from the inverse charged-current reactions.
    // Do not overwrite these opacities by Kirchhoff below.
   // if (beta_decay) {
    //    add_charged_current_emission(F, rates);
    //    add_charged_current_absorption_opacity(F, rates);
    //}

    // TEST
    // COMPUTE BETA EMISSION FROM KIRCHHOFF AND OPACITIES

    if (beta_decay) {
        // First compute charged-current absorption opacities:
        //   ν_e  + n -> p + e-
        //   ν̄_e + p -> n + e+
        add_charged_current_absorption_opacity(F, rates);
        // Then compute beta emissivities from Kirchhoff:
        //   Q = kappa_a * B_E
        //   R = kappa_n * B_N
        std::array<double, NUMSPECIES> g_nu{{1,1,0,0,4}};
        #ifdef M1_NU_FIVESPECIES
            g_nu = {{1,1,1,1,2}};
        #endif
    
        std::array<double, NUMSPECIES> Q_beta{{0,0,0,0,0}};
        std::array<double, NUMSPECIES> R_beta{{0,0,0,0,0}};
    
        add_kirchhoff_emission_from_absorption_opacity(
            F, g_nu, rates.kappa_a, rates.kappa_n, Q_beta, R_beta);   

        rates.Q[NUE]    += Q_beta[NUE];
        rates.R[NUE]    += R_beta[NUE];  
        rates.Q[NUEBAR] += Q_beta[NUEBAR];
        rates.R[NUEBAR] += R_beta[NUEBAR];
    }

    // Scattering is independent of thermal emissivity.
    add_scattering_opacity(F, rates);

    // Thermal pair/plasmon/bremsstrahlung channels provide emissivities.
    // Their inverse absorption opacities are added through Kirchhoff, like in FIL.
    rates_accum thermal_rates;
    if (pair_annihilation) add_pair_process_emission(F, thermal_rates);
    if (plasmon_decay)     add_plasmon_decay_emission(F, thermal_rates);
    if (bremsstrahlung)    add_brems_emission(F, thermal_rates);


    std::array<double, NUMSPECIES> g_nu{{1,1,0,0,4}};
#ifdef M1_NU_FIVESPECIES
    g_nu = {{1,1,1,1,2}};
#endif

    std::array<double, NUMSPECIES> kappa_a_thermal{{0,0,0,0,0}};
    std::array<double, NUMSPECIES> kappa_n_thermal{{0,0,0,0,0}};
    add_kirchhoff_absorption_opacity_from_QR(
        F, g_nu, thermal_rates.Q, thermal_rates.R,
        kappa_a_thermal, kappa_n_thermal);

    for (int s = 0; s < NUMSPECIES; ++s) {
        rates.Q[s]       += thermal_rates.Q[s];
        rates.R[s]       += thermal_rates.R[s];
        rates.kappa_a[s] += kappa_a_thermal[s];
        rates.kappa_n[s] += kappa_n_thermal[s];
    }

    // Tau post-estimate (optional; now not stored)
    // TODO implement proper tau path integral handeling here
    for (int s = 0; s < NUMSPECIES; ++s) {
        const double kappa_tot = rates.kappa_a[s] + rates.kappa_s[s];
        (void)tau_policy.tau_post(kappa_tot, rho_code, xyz_code, mass_scale, s, F.rho_cgs);
    }

    // Neutrino-temperature correction 
    if (apply_temp_correction) {
        for (int s = 0; s < NUMSPECIES; ++s) {
            const double Rloc = rates.R[s];
            const double Qloc = rates.Q[s];
            if (Rloc > 0.0 && Qloc > 0.0) {
                // TODO: calculate eps from W(E-Fv)/N
                // here eps just approximate
                const double eps_mev = Qloc / Rloc;
                const double Tnu_mev = fermi::FDR<2,3>::get(F.eta_nu[s]) * eps_mev;
                const double Tf_mev  = safe_pos(F.temp_mev);
                double fact = 1.0;
                if (::isfinite(Tnu_mev) && Tnu_mev > 0.0) {
                    const double ratio = Tnu_mev / Tf_mev;
                    fact = ratio * ratio;
                    if (fact < 1.0) fact = 1.0;
                }
                if (::isfinite(fact) && fact > 0.0) {
                    // TODO is it correct like that?
                    if(s == NUX){            
                        rates.kappa_s[s] *= fact;
                    }else{
                        rates.Q[s]       *= fact;
                        rates.R[s]       *= fact;
                        rates.kappa_a[s] *= fact;
                        rates.kappa_n[s] *= fact;
                        rates.kappa_s[s] *= fact;
                    }
                }
            }
        }
    }

    nu_rates_all_out all;
    for (int s = 0; s < NUMSPECIES; ++s) {
        nu_rates_out out;
        out.eta_E   = Q_mev_to_code(rates.Q[s], mass_scale);
        out.eta_N   = R_to_code(rates.R[s], mass_scale);
        out.kappa_a = kappa_to_code(rates.kappa_a[s], mass_scale);
        out.kappa_n = kappa_to_code(rates.kappa_n[s], mass_scale);
        out.kappa_s = kappa_to_code(rates.kappa_s[s], mass_scale);

        if (!::isfinite(out.eta_E)) out.eta_E = 0.0;
        if (!::isfinite(out.eta_N)) out.eta_N = 0.0;
        if (!::isfinite(out.kappa_a)) out.kappa_a = 0.0;
        if (!::isfinite(out.kappa_n)) out.kappa_n = 0.0;
        if (!::isfinite(out.kappa_s)) out.kappa_s = 0.0;
        all.out[s] = out;
    }
    return all;
}

// Backwards-compatible wrapper
template <typename eos_t, typename tau_policy_t>
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE nu_rates_out compute_species(
    const eos_t& eos,
    double rho_code,
    double temp_code,
    double ye,
    double ymu,
    double mass_scale,
    bool beta_decay,
    bool plasmon_decay,
    bool bremsstrahlung,
    bool pair_annihilation,
    int species,
    const double* xyz_code,
    const tau_policy_t& tau_policy,
    bool apply_temp_correction)
{
    const nu_rates_all_out all = compute_all_species(
        eos, rho_code, temp_code, ye, ymu, mass_scale,
        beta_decay, plasmon_decay, bremsstrahlung, pair_annihilation,
        xyz_code, tau_policy, apply_temp_correction);
    return all.out[species];
}

} // namespace grace

#endif // GRACE_PHYSICS_NEUTRINO_RATES_ANALYTIC_HH

/*
--- WEAKHUB
m1:
  eas:
    kind: "weakhub"
    use_weakhub: true
    use_analytic: false
    weakhub_table: "/path/to/weakhub_table.h5"
--- ANALYTIC
m1:
  eas:
    kind: "analytic"
    use_weakhub: false
    use_analytic: true

*/
