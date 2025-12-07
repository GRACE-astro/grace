/**
 * @file ideal_gas_eos.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-28
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#ifndef GRACE_PHYSICS_EOS_IDEAL_GAS_HH
#define GRACE_PHYSICS_EOS_IDEAL_GAS_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/physics/eos/eos_base.hh>

#include <Kokkos_Core.hpp>

namespace grace {

class ideal_gas_eos_t 
    : public eos_base_t<ideal_gas_eos_t> 
{
    ideal_gas_eos_t() = default ; 

    ideal_gas_eos_t( double _gamma_th_m1 
                    , double _ent_min
                    , double baryon_mass
                    , double c2p_eps_max )
     : eos_base_t<ideal_gas_eos_t>{ 0, _cold_eos.eos_rhomax, _cold_eos.eos_rhomin
                                           , 1e99, 0
                                           , 1e99, 0
                                           , baryon_mass
                                           , 0
                                           , c2p_eps_max
                                           , 1.0
                                           , 1.e99
                                           , false 
                                           , false }
     , gamma_th(_gamma_th_m1+1.)
     , gamma_th_m1(_gamma_th_m1)
     , entropy_min(_ent_min)
    {}

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press__eps_rho_ye_impl(double& eps, double& rho, double& ye, error_type& err) const 
    {
        limit_rho(rho, err) ;
        limit_eps(eps, err) ; 
        return rho * eps * gamma_th_m1 ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press_temp__eps_rho_ye_impl(double& temp, double& eps, double& rho, double& ye, error_type& err) const
    {
        limit_rho(rho, err) ;
        limit_eps(eps, err) ; 
        temp = temp__eps(eps) ; 
        return eps * rho * gamma_th_m1 ; 
    }


    void KOKKOS_INLINE_FUNCTION
    limit_rho(double& rho, error_type& err) const {
        err = (rho<rho_min)*EOS_RHO_TOO_LOW + (rho>rho_max)*ERR_RHO_TOO_HIGH ; 
        rho = fmin(rho_max,fmax(rho,rho_min)) ; 
    }

    void KOKKOS_INLINE_FUNCTION
    limit_eps(double& eps, error_type& err) const {
        err = (eps>this->c2p_eps_max)*ERR_EPS_TOO_HIGH ; 
        eps = fmin(this->c2p_eps_max,fmax(eps,0.0)) ; 
    }

    double KOKKOS_INLINE_FUNCTION
    temp__eps(double eps) const {
        return eps/gamma_th_m1 ; 
    }


    double gamma, gamma_m1, entropy_min ; 

    static constexpr double rho_min{1e-100} ; 
    static constexpr double rho_max{1e100}  ; 
} ; 

}

#endif /* GRACE_PHYSICS_EOS_IDEAL_GAS_HH */