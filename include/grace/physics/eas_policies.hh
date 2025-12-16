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

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/physical_constants.hh>

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
            u(KAPPAA_) = u(KAPPAS_) = u(ETA_) = u(KAPPAAN_) = u(ETAN_) = 0. ; 
            break ; 
            case LARGE_KS:
            u(KAPPAA_) = u(ETA_) = u(KAPPAAN_) = u(ETAN_) = 0. ; 
            u(KAPPAS_) = _ks_value ; 
            break ; 
            case SHADOW_CAST:
            // we assume pcoords is cartesian 
            r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ;
            u(KAPPAA_) = u(KAPPAS_) = u(ETA_) = u(KAPPAAN_) = u(ETAN_) = 0. ; 
            if ( r<0.046875 ) {
                u(KAPPAA_) = 1e06 ; 
                u(KAPPAAN_) = 1e06 ; 
            } 
            break ; 
            case EMITTING_SPHERE:
            // we assume pcoords is cartesian 
            r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ; 
            u(KAPPAA_) = u(KAPPAS_) = u(ETA_) = u(KAPPAAN_) = u(ETAN_) = 0. ; 
            if ( r < 1. ) {
                double T = _emitting_sphere_temperature ; 
                double sigma = _emitting_sphere_cross_section ; 
                // we set the rates according to LTE, 
                // for simplicity the Stefan Boltzmann constant is 1 here
                u(KAPPAA_) = _emitting_sphere_cross_section ; 
                u(ETA_) = SQR(T)*SQR(T) * _emitting_sphere_cross_section ; 
                
                u(KAPPAAN_) = _emitting_sphere_cross_section ; 
                u(ETAN_) = SQR(T)*T * _emitting_sphere_cross_section ; 
            } 
            break;
            case COUPLING_TEST:
            r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ; 
            aux(i,j,k,KAPPAS_,q) = 0.0 ;
            if ( r < 1.0 ) {
                aux(i,j,k,KAPPAA_,q) = 1.0 ; 
                aux(i,j,k,KAPPAAN_,q) = 1.0; 
                if ( r < 0.5 ) {
                    // effectively T = 1 
                    aux(i,j,k,ETA_,q) = 0.01  ; 
                    aux(i,j,k,ETAN_,q) = 0.01 ; 
                } else {
                    double T = 1. - (r-0.5)/0.5 ;
                    aux(i,j,k,ETA_,q) = 0.01  * T * T * T * T; 
                    aux(i,j,k,ETAN_,q) = 0.01  * T * T * T ; 
                }
            } else {
                aux(i,j,k,KAPPAA_,q) = 0.0 ; 
                aux(i,j,k,KAPPAAN_,q) = 0.0; 
                aux(i,j,k,ETA_,q) = 0.0;
                aux(i,j,k,ETAN_,q) = 0.0;
            }
            
            break ; 
        }
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
        using namespace grace::physical_constants ; 
        double mu = 0.5 ; // fully ionized, fixme 
        double const rho = aux(VEC(i,j,k),RHO_,q)  * Msun_cgs / (Msun_to_cm*SQR(Msun_to_cm*mass_scale)) ; 
        double const T   = k_cgs * aux(VEC(i,j,k),TEMP_,q) / (mu * mp_cgs);// erg
        // Energy rates 
        double eta_cgs = 1.4e-27 * sqrt(T/k_cgs) * SQR(rho)/me_cgs/mp_cgs ; // gaunt factor 1
        // Kirchoff law 
        // 2 *( k_cgs *T )**4 * np.pi**4 / ( 15 * clight**2 * h_cgs**3)
        double BB = 2 * SQR(M_PI*T)*SQR(M_PI*T) / ( 15. * SQR(clight) * SQR(h_cgs) * h_cgs) ; 
        // Planck mean opacity
        double kappa_cgs = eta_cgs / BB ; 

        double BBn = 4 * SQR(T)*T / (SQR(clight)*SQR(h_cgs)*h_cgs) * 1.20206 ; // numerical factor is Zeta(3)

        aux(i,j,k,ETA_,q)    = eta_cgs  / Msun_to_erg * SQR(Msun_to_cm)*Msun_to_cm * Msun_to_s * SQR(mass_scale) * mass_scale ;
        aux(i,j,k,KAPPAA_,q) = kappa_cgs * Msun_to_cm * mass_scale ;
        // Number rates 
        // The integral over frequency is IR divergence, we cutoff at the plasma frequency 
        // and plop the result (~log(h nu_min/(kT))) into the Gaunt factor 
        // now -- we don't really care about eta anyway, eta/kappa is unaffected by this 
        // for now we prentend the gaunt factor stays one and sweep this under the rug 
        // but FIXME please 
        aux(i,j,k,ETAN_,q)    = eta_cgs / T * SQR(Msun_to_cm)*Msun_to_cm * Msun_to_s * SQR(mass_scale) * SQR(mass_scale) / Msun_to_erg ; 
        aux(i,j,k,KAPPAAN_,q) = eta_cgs / T / BBn * Msun_to_cm * mass_scale ; 
        // Scattering 
        aux(i,j,k,KAPPAS_,q) = 0;//rho/me_cgs * sigma_T * Msun_to_cm * mass_scale; 
    }

    var_array_t aux ; 
    double mass_scale; 
} ; 

} /* namespace grace */

#endif /*GRACE_PHYSICS_EAS_POLICIES_HH*/