/**
 * @file eas_policies.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2024-05-13
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

#ifndef GRACE_PHYSICS_EAS_POLICIES_HH
#define GRACE_PHYSICS_EAS_POLICIES_HH

#include <grace_config.h>

#include <grace/physics/m1_helpers.hh>
#include <grace/physics/m1.hh>
#include <grace/physics/eas_neutrino_rates_analytic.hh>  

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/physical_constants.hh>
#include <grace/physics/eos/unit_system.hh>

#include <grace/config/config_parser.hh>

#include <string> 

namespace grace {

struct test_eas_op {
    enum test_t  {
        ZERO_EAS=0,
        LARGE_KS,
        EMITTING_SPHERE,
        SHADOW_CAST,
        COUPLING_TEST
    } ; 
    test_eas_op(
        grace::var_array_t _aux 
    ) : aux(_aux) 
    {
        auto _which_test = grace::get_param<std::string>(
            "m1", "id_type"
        ) ; 
        if (_which_test == "straight_beam" or 
            _which_test == "curved_beam" ) 
        {
            which_test = ZERO_EAS ;
        } else if (
            _which_test == "scattering"
            or _which_test == "moving_scattering"
        ) {
            which_test = LARGE_KS ; 
            _ks_value = grace::get_param<double>("m1","scattering_test","k_s") ; 
        } else if (
            _which_test == "shadow"
        ) {
            which_test = SHADOW_CAST; 
        } else if ( 
            _which_test == "emitting_sphere"
        ) {
            which_test = EMITTING_SPHERE ; 
            _emitting_sphere_temperature = grace::get_param<double>("m1","emitting_sphere_test","temperature") ;
            _emitting_sphere_cross_section = grace::get_param<double>("m1","emitting_sphere_test","cross_section") ; 
        } else if ( _which_test == "coupling_test") {
            which_test = COUPLING_TEST ; 
        } else {
            ERROR("Unknown m1 test") ; 
        }
    }

    void KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(const int i, const int j, const int k), int64_t q
        , double* xyz
    ) const
    {
        auto u = Kokkos::subview(aux,VEC(i,j,k),Kokkos::ALL(),q) ; 
        double r=0;
        switch (which_test) {
            case ZERO_EAS:
            u(KAPPAA1_) = u(KAPPAS1_) = u(ETA1_) = u(KAPPAAN1_) = u(ETAN1_) = 0. ; 
            break ; 
            case LARGE_KS:
            u(KAPPAA1_) = u(ETA1_) = u(KAPPAAN1_) = u(ETAN1_) = 0. ; 
            u(KAPPAS1_) = _ks_value ; 
            break ; 
            case SHADOW_CAST:
            // we assume pcoords is cartesian 
            r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ;
            u(KAPPAA1_) = u(KAPPAS1_) = u(ETA1_) = u(KAPPAAN1_) = u(ETAN1_) = 0. ; 
            if ( r<0.046875 ) {
                u(KAPPAA1_) = 1e06 ; 
                u(KAPPAAN1_) = 1e06 ; 
            } 
            break ; 
            case EMITTING_SPHERE:
            // we assume pcoords is cartesian 
            r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ; 
            u(KAPPAA1_) = u(KAPPAS1_) = u(ETA1_) = u(KAPPAAN1_) = u(ETAN1_) = 0. ; 
            if ( r < 1. ) {
                double T = _emitting_sphere_temperature ; 
                double sigma = _emitting_sphere_cross_section ; 
                // we set the rates according to LTE, 
                // for simplicity the Stefan Boltzmann constant is 1 here
                u(KAPPAA1_) = _emitting_sphere_cross_section ; 
                u(ETA1_) = SQR(T)*SQR(T) * _emitting_sphere_cross_section ; 
                
                u(KAPPAAN1_) = _emitting_sphere_cross_section ; 
                u(ETAN1_) = SQR(T)*T * _emitting_sphere_cross_section ; 
            } 
            break;
            case COUPLING_TEST:
            r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ; 
            aux(i,j,k,KAPPAS1_,q) = 0.0 ;
            if ( r < 1.0 ) {
                aux(i,j,k,KAPPAA1_,q) = 1.0 ; 
                aux(i,j,k,KAPPAAN1_,q) = 1.0; 
                if ( r < 0.5 ) {
                    // effectively T = 1 
                    aux(i,j,k,ETA1_,q) = 0.01  ; 
                    aux(i,j,k,ETAN1_,q) = 0.01 ; 
                } else {
                    double T = 1. - (r-0.5)/0.5 ;
                    aux(i,j,k,ETA1_,q) = 0.01  * T * T * T * T; 
                    aux(i,j,k,ETAN1_,q) = 0.01  * T * T * T ; 
                }
            } else {
                aux(i,j,k,KAPPAA1_,q) = 0.0 ; 
                aux(i,j,k,KAPPAAN1_,q) = 0.0; 
                aux(i,j,k,ETA1_,q) = 0.0;
                aux(i,j,k,ETAN1_,q) = 0.0;
            }
            
            break ; 
        }
        #ifdef M1_NU_THREESPECIES
        aux(i,j,k,KAPPAA2_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAAN2_,q) = aux(i,j,k,KAPPAA1_,q); 
        aux(i,j,k,KAPPAS2_,q) = aux(i,j,k,KAPPAS1_,q);
        aux(i,j,k,ETA2_,q) = aux(i,j,k,ETA1_,q); 
        aux(i,j,k,ETAN2_,q) = aux(i,j,k,ETAN1_,q);
        aux(i,j,k,KAPPAA3_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAAN3_,q) = aux(i,j,k,KAPPAA1_,q); 
        aux(i,j,k,KAPPAS3_,q) = aux(i,j,k,KAPPAS1_,q);
        aux(i,j,k,ETA3_,q) = aux(i,j,k,ETA1_,q); 
        aux(i,j,k,ETAN3_,q) = aux(i,j,k,ETAN1_,q);
        #endif
        #ifdef M1_NU_FIVESPECIES
        aux(i,j,k,KAPPAA4_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAAN4_,q) = aux(i,j,k,KAPPAA1_,q); 
        aux(i,j,k,KAPPAS4_,q) = aux(i,j,k,KAPPAS1_,q);
        aux(i,j,k,ETA4_,q) = aux(i,j,k,ETA1_,q); 
        aux(i,j,k,ETAN4_,q) = aux(i,j,k,ETAN1_,q);
        aux(i,j,k,KAPPAA5_,q) = aux(i,j,k,KAPPAA1_,q);
        aux(i,j,k,KAPPAAN5_,q) = aux(i,j,k,KAPPAA1_,q); 
        aux(i,j,k,KAPPAS5_,q) = aux(i,j,k,KAPPAS1_,q);
        aux(i,j,k,ETA5_,q) = aux(i,j,k,ETA1_,q); 
        aux(i,j,k,ETAN5_,q) = aux(i,j,k,ETAN1_,q);
        #endif
    }

    var_array_t aux ; 
    test_t which_test;
    double _ks_value ; 
    double _emitting_sphere_cross_section, _emitting_sphere_temperature; 
} ; 


struct photon_eas_op {
    photon_eas_op(
        var_array_t _aux
    ) 
     : mass_scale(grace::get_param<double>("coordinate_system","mass_scale")) 
     , aux(_aux)
    {} 

    void KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(const int i, const int j, const int k), int64_t q
        , double* xyz
    ) const
    {
        #if 0
        using namespace grace::physical_constants ; 
        auto const Munit = mass_scale * Msun_si ; 
        auto code_units = unit_system::make(
            G_si * Munit / SQR(c_si),
            G_si * Munit / (SQR(c_si)*c_si),
            Munit,
            -1 // not needed
        ); 
        auto uconv = CGS_units / code_units; 

        // now we compute everything in CGS and convert 
        
        double mu = 0.5 ; // fully ionized, fixme 
        double const rho = aux(VEC(i,j,k),RHO_,q)  * Msun_cgs / (Msun_to_cm*SQR(Msun_to_cm*mass_scale)) ; 
        double const T   = k_cgs * aux(VEC(i,j,k),TEMP_,q) / (mu * mp_cgs);// erg
        // Energy rates 
        double eta_cgs = 1.4e-27 * sqrt(T/k_cgs) * SQR(rho)/me_cgs/mp_cgs ; // gaunt factor 1
        // Kirchoff law 
        // 2 *( k_cgs *T )**4 * np.pi**4 / ( 15 * clight**2 * h_cgs**3)
        double BB = 2 * SQR(M_PI*T)*SQR(M_PI*T) / ( 15. * SQR(c_cgs) * SQR(h_cgs) * h_cgs) ; 
        // Planck mean opacity
        double kappa_cgs = eta_cgs / BB ;
        double kappa_r_cgs =  1.7e-25 * Kokkos::pow(T,-7./2.) * ;

        double BBn = 4 * SQR(T)*T / (SQR(c_cgs)*SQR(h_cgs)*h_cgs) * 1.20206 ; // numerical factor is Zeta(3)

        aux(i,j,k,ETA1_,q)    = eta_cgs  / Msun_to_erg * SQR(Msun_to_cm)*Msun_to_cm * Msun_to_s * SQR(mass_scale) * mass_scale ;
        aux(i,j,k,KAPPAA1_,q) = kappa_cgs * Msun_to_cm * mass_scale ;
        aux(i,j,k,KAPPAR1_,q) =  // Rosseland mean 
        // Number rates 
        // The integral over frequency is IR divergence, we cutoff at the plasma frequency 
        // and plop the result (~log(h nu_min/(kT))) into the Gaunt factor 
        // now -- we don't really care about eta anyway, eta/kappa is unaffected by this 
        // for now we prentend the gaunt factor stays one and sweep this under the rug 
        // but FIXME please 
        aux(i,j,k,ETAN1_,q)    = eta_cgs / T * SQR(Msun_to_cm)*Msun_to_cm * Msun_to_s * SQR(mass_scale) * SQR(mass_scale) / Msun_to_erg ; 
        aux(i,j,k,KAPPAAN1_,q) = eta_cgs / T / BBn * Msun_to_cm * mass_scale ; 
        // Scattering 
        aux(i,j,k,KAPPAS1_,q) = 0;//rho/me_cgs * sigma_T * Msun_to_cm * mass_scale; 
        #endif 
    }

    var_array_t aux ; 
    double mass_scale; 
} ; 

//------------------------------------------------------------------------------
// Neutrino EAS operator 
//------------------------------------------------------------------------------
template <typename eos_t>
struct neutrinos_eas_op 
    : public eos_base_t<eos_t> 
{
  using tau_kind_t = int;
  enum : tau_kind_t {
    TAU_NONE = 0,
    TAU_ANALYTIC_DENSITY = 1,
    TAU_LOCAL_SPHERICAL = 2
  };
  enum : int {
    BETAEQ_OFF = 0,
    BETAEQ_CHEMICAL = 1
  };

    neutrinos_eas_op(var_array_t _aux)
      : eos(eos::get().get_eos<eos_t>()),
        aux(_aux),
        mass_scale(grace::get_param<double>("coordinate_system", "mass_scale")),
        beta_decay(grace::get_param<bool>("m1", "eas", "beta_decay")),
        plasmon_decay(grace::get_param<bool>("m1", "eas", "plasmon_decay")),
        bremsstrahlung(grace::get_param<bool>("m1", "eas", "bremsstrahlung")),
        pair_annihilation(grace::get_param<bool>("m1", "eas", "pair_annihilation")),
        apply_temp_correction(grace::get_param<bool>("m1", "eas", "temperature_correction")),
        //KEN turned off weakhub
        //use_weakhub(parse_use_weakhub_host()),
        betaeq_mode(parse_betaeq_mode_host()),
        tau_kind(parse_tau_kind_host())
        // betaeq_mode(BETAEQ_OFF),
        // tau_kind(TAU_NONE),
        //KEN turned off weakhub
        //weakhub(grace::weakhub::get_device_handle())
    {
    // Host-only parsing: strings are not device-friendly.
    // TODO: parser optimize
    //betaeq_mode = parse_betaeq_mode_host();
    //tau_kind = parse_tau_kind_host();
    spherical_tau.r_outer_code = grace::get_param<double>("m1", "eas", "tau_outer_radius_code");
    spherical_tau.seed_with_analytic = true;
  }

    int parse_betaeq_mode_host() const {
        try {
            const auto mode = grace::get_param<std::string>("m1", "eas", "betaeq_policy");
            if (mode == "chemical") return BETAEQ_CHEMICAL;
        } catch(...) {}
        return BETAEQ_OFF;
    }

    tau_kind_t parse_tau_kind_host() const {
        try {
            const auto mode = grace::get_param<std::string>("m1", "eas", "tau_policy");
            if (mode == "analytic_density") return TAU_ANALYTIC_DENSITY;
            else if (mode == "local_spherical") return TAU_LOCAL_SPHERICAL;
        } catch(...) {}
        return TAU_NONE;
    }

    bool parse_use_weakhub_host() const {
        return grace::weakhub::weakhub_enabled_from_params();
    }

    // --- beta equilibrium: mu_e + mu_p - mu_n - Qnp = 0 (chemical equilibrium) ---
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double beta_eq_residual(double rho, double T, double Ye) const {
        double mu_p=0.0, mu_n=0.0;
        double Xa=0.0, Xh=0.0, Xn=0.0, Xp=0.0, Abar=1.0, Zbar=1.0;
        eos_err_t err;
        double ye_loc = Ye, T_loc = T, rho_loc = rho;
        const double mu_e = eos.mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye(mu_p, mu_n, Xa, Xh, Xn, Xp, Abar, Zbar, T_loc, rho_loc, ye_loc, err);
        //if (err != eos_err_t{} || !::isfinite(mu_e) || !::isfinite(mu_p) || !::isfinite(mu_n)) return 0.0;
        // KEN could not make this work 
        if ( !::isfinite(mu_e) || !::isfinite(mu_p) || !::isfinite(mu_n)) return 0.0;
        return (mu_e + mu_p - mu_n - nu_constants::Qnp);
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double find_ye_betaeq(double rho, double T, double Ye0) const {
        // Device-friendly bisection. If no bracket -> return Ye0.
        double a = 1.0e-6, b = 0.60;
        double fa = beta_eq_residual(rho, T, a), fb = beta_eq_residual(rho, T, b);
        if (!::isfinite(fa) || !::isfinite(fb) || fa * fb > 0.0) return Ye0;

        double left = a, right = b, fleft = fa, mid = Ye0;
        for (int it = 0; it < 40; ++it) {
            mid = 0.5 * (left + right);
            const double fm = beta_eq_residual(rho, T, mid);
            if (!::isfinite(fm) || fm == 0.0) break;
            if (fleft * fm <= 0.0) right = mid; else { left = mid; fleft = fm; }
            if ((right - left) < 1.0e-8) break;
        }
        return mid;
    }

    // Main Kernel
    void KOKKOS_INLINE_FUNCTION operator()(VEC(const int i, const int j, const int k), int64_t q, double* xyz) const {
       const double rho = aux(VEC(i,j,k),RHO_,q);
       const double T   = aux(VEC(i,j,k),TEMP_,q);
       double Ye        = aux(VEC(i,j,k),YE_,q);
       double Ymu       = 0.0;
       #ifdef M1_NU_FIVESPECIES
        Ymu = aux(VEC(i,j,k),YMU_,q);
        #endif
        if (betaeq_mode == BETAEQ_CHEMICAL) Ye = find_ye_betaeq(rho, T, Ye);

        tau_policy_none tau_none{};
        tau_policy_analytic_density tau_ana{};
        nu_rates_all_out all{};
    
        //if (use_weakhub && weakhub.valid) {
        if (false) {
            switch (tau_kind) {
            case TAU_LOCAL_SPHERICAL:
            all = compute_all_species_weakhub(weakhub, rho, T, Ye, Ymu, mass_scale, plasmon_decay, bremsstrahlung, pair_annihilation, xyz, spherical_tau, apply_temp_correction);
            break;
            case TAU_ANALYTIC_DENSITY:
            all = compute_all_species_weakhub(weakhub, rho, T, Ye, Ymu, mass_scale, plasmon_decay, bremsstrahlung, pair_annihilation, xyz, tau_ana, apply_temp_correction);
            break;
            default:
            all = compute_all_species_weakhub(weakhub, rho, T, Ye, Ymu, mass_scale, plasmon_decay, bremsstrahlung, pair_annihilation, xyz, tau_none, apply_temp_correction);
            break;
            }
        } else {
            switch (tau_kind) {
            case TAU_LOCAL_SPHERICAL:
            all = compute_all_species(eos, rho, T, Ye, Ymu, mass_scale, beta_decay, plasmon_decay, bremsstrahlung, pair_annihilation, xyz, spherical_tau, apply_temp_correction);
            break;
            case TAU_ANALYTIC_DENSITY:
            all = compute_all_species(eos, rho, T, Ye, Ymu, mass_scale, beta_decay, plasmon_decay, bremsstrahlung, pair_annihilation, xyz, tau_ana, apply_temp_correction);
            break;
            default:
            all = compute_all_species(eos, rho, T, Ye, Ymu, mass_scale, beta_decay, plasmon_decay, bremsstrahlung, pair_annihilation, xyz, tau_none, apply_temp_correction);
            break;
            }
        }
        #ifdef M1_NU_FIVESPECIES
        { const nu_rates_out r = all.out[NUE];     aux(i,j,k,ETA1_,q)=r.eta_E; aux(i,j,k,KAPPAA1_,q)=r.kappa_a; aux(i,j,k,KAPPAS1_,q)=r.kappa_s; aux(i,j,k,ETAN1_,q)=r.eta_N; aux(i,j,k,KAPPAAN1_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUEBAR];  aux(i,j,k,ETA2_,q)=r.eta_E; aux(i,j,k,KAPPAA2_,q)=r.kappa_a; aux(i,j,k,KAPPAS2_,q)=r.kappa_s; aux(i,j,k,ETAN2_,q)=r.eta_N; aux(i,j,k,KAPPAAN2_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUMU];    aux(i,j,k,ETA3_,q)=r.eta_E; aux(i,j,k,KAPPAA3_,q)=r.kappa_a; aux(i,j,k,KAPPAS3_,q)=r.kappa_s; aux(i,j,k,ETAN3_,q)=r.eta_N; aux(i,j,k,KAPPAAN3_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUMUBAR]; aux(i,j,k,ETA4_,q)=r.eta_E; aux(i,j,k,KAPPAA4_,q)=r.kappa_a; aux(i,j,k,KAPPAS4_,q)=r.kappa_s; aux(i,j,k,ETAN4_,q)=r.eta_N; aux(i,j,k,KAPPAAN4_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUX];     aux(i,j,k,ETA5_,q)=r.eta_E; aux(i,j,k,KAPPAA5_,q)=r.kappa_a; aux(i,j,k,KAPPAS5_,q)=r.kappa_s; aux(i,j,k,ETAN5_,q)=r.eta_N; aux(i,j,k,KAPPAAN5_,q)=r.kappa_n; }
        #elif defined(M1_NU_THREESPECIES)
        { const nu_rates_out r = all.out[NUE];    aux(i,j,k,ETA1_,q)=r.eta_E; aux(i,j,k,KAPPAA1_,q)=r.kappa_a; aux(i,j,k,KAPPAS1_,q)=r.kappa_s; aux(i,j,k,ETAN1_,q)=r.eta_N; aux(i,j,k,KAPPAAN1_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUEBAR]; aux(i,j,k,ETA2_,q)=r.eta_E; aux(i,j,k,KAPPAA2_,q)=r.kappa_a; aux(i,j,k,KAPPAS2_,q)=r.kappa_s; aux(i,j,k,ETAN2_,q)=r.eta_N; aux(i,j,k,KAPPAAN2_,q)=r.kappa_n; }
        { const nu_rates_out r = all.out[NUX];    aux(i,j,k,ETA3_,q)=r.eta_E; aux(i,j,k,KAPPAA3_,q)=r.kappa_a; aux(i,j,k,KAPPAS3_,q)=r.kappa_s; aux(i,j,k,ETAN3_,q)=r.eta_N; aux(i,j,k,KAPPAAN3_,q)=r.kappa_n; }
        #endif
    }

  eos_t eos;
  var_array_t aux;
  double mass_scale;
  bool beta_decay, plasmon_decay, bremsstrahlung, pair_annihilation;
  bool apply_temp_correction;
  bool use_weakhub;
  int betaeq_mode;
  tau_kind_t tau_kind;
  grace::weakhub::device_handle weakhub;
  grace::tau_policy_local_spherical spherical_tau{};
};

} /* namespace grace */

#endif /*GRACE_PHYSICS_EAS_POLICIES_HH*/
