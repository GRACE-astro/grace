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

#include <grace/config/config_parser.hh>

#include <string> 

namespace grace {

struct test_eas_op {
    enum test_t : int {
        ZERO_EAS=0,
        LARGE_KS,
        EMITTING_SPHERE,
        SHADOW_CAST
    }
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
            which_test = SHADOW_CAST 
        } else if ( 
            _which_test == "emitting_sphere"
        ) {
            which_test = EMITTING_SPHERE ; 
            _emitting_sphere_keta = grace::get_param<double>("m1","emitting_sphere_test","kappa_eta") ; 
        } else {
            ERROR("Unknown m1 test") ; 
        }
    }

    void KOKKOS_INLINE_FUNCTION
    operator() (
        VEC(const int i, const int j, const int k), int64_t q,
        , double* xyz
    ) 
    {
        auto u = Kokkos::subview(aux,VEC(i,j,k),Kokkos::ALL(),q) ; 

        switch (which_test) {
            case ZERO_EAS:
            u(KAPPAA_) = u(KAPPAS_) = u(ETA_) = 0. ; 
            break ; 
            case LARGE_KS:
            u(KAPPAA_) = u(ETA_) = 0. ; 
            u(KAPPAS_) = _ks_value ; 
            break ; 
            case SHADOW_CAST:
            // we assume pcoords is cartesian 
            double const r = sqrt(
                SQR(xyz[0]+0.2) + SQR(xyz[1]) + SQR(xyz[2])
            ) ;
            u(KAPPAA_) = u(KAPPAS_) = u(ETA_) = 0. ; 
            if ( r<0.07 ) {
                u(KAPPAA_) = 1e06 ; 
            } 
            break ; 
            case EMITTING_SPHERE:
            // we assume pcoords is cartesian 
            double const r = sqrt(
                SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2])
            ) ; 
            u(KAPPAA_) = u(KAPPAS_) = u(ETA_) = 0. ; 
            if ( r < 1. ) {
                u(KAPPAA_) = u(ETA_) = _emitting_sphere_keta ; 
            } 
            break ; 
        }
    }

    var_array_t aux ; 
    test_t which_test
    double _ks_value ; 
    double _emitting_sphere_keta ; 
} ; 

} /* namespace grace */

#endif /*GRACE_PHYSICS_EAS_POLICIES_HH*/