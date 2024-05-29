/**
 * @file piecewise_polytropic_eos.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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

#ifndef GRACE_PHYSICS_EOS_PWPOLY_EOS_HH
#define GRACE_PHYSICS_EOS_PWPOLY_EOS_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/physics/eos/eos_base.hh>

#include <Kokkos_Core.hpp>

namespace grace {

class piecewise_polytropic_eos_t
{
    using error_type = unsigned int ; 
 public:
    static constexpr unsigned int max_n_pieces = 8 ; 

    piecewise_polytropic_eos_t() = default ; 

    piecewise_polytropic_eos_t(
          Kokkos::View<double [max_n_pieces], grace::default_space> k
        , Kokkos::View<double [max_n_pieces], grace::default_space> gamma
        , Kokkos::View<double [max_n_pieces], grace::default_space> rho 
        , Kokkos::View<double [max_n_pieces], grace::default_space> eps 
        , Kokkos::View<double [max_n_pieces], grace::default_space> press
        , unsigned int n_pieces 
        , double rhomax
        , double rhomin )
      : _k(k), _gamma(gamma), _rho(rho), _eps(eps), _press(press)
      , num_pieces(n_pieces), eos_rhomax(rhomax), eos_rhomin(rhomin)
    {} 

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press_cold_eps_cold__rho(double& eps_cold, double& rho, error_type& err) const 
    {
        auto idx = find_index_rho(rho, err) ; 
        auto press_cold = _k(idx) * Kokkos::pow(rho, _gamma(idx)) ; 
        eps_cold = _eps(idx) + press_cold / (rho*(_gamma(idx)-1.)) ;
        return press_cold ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    dpress_cold_drho__rho(double& rho, error_type& err) const {
        auto idx = find_index_rho(rho, err) ; 
        auto press_cold = _k(idx) * Kokkos::pow(rho, _gamma(idx)) ; 
        return _gamma(idx) * press_cold / rho ; 
    }

 private:
    

    Kokkos::View<double [max_n_pieces], grace::default_space> _k     ; 
    Kokkos::View<double [max_n_pieces], grace::default_space> _gamma ; 
    Kokkos::View<double [max_n_pieces], grace::default_space> _rho   ;
    Kokkos::View<double [max_n_pieces], grace::default_space> _eps   ;
    Kokkos::View<double [max_n_pieces], grace::default_space> _press ; 

    unsigned int num_pieces ; 

    unsigned int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    find_index_rho(double& rho, error_type& err) const {
        err = rho < eos_rhomin ? 1 : 0; 
        err = rho > eos_rhomax ? 2 : 0; 

        rho = math::max(eos_rhomin, math::min(eos_rhomax, rho)) ; 
        for( int ii=0; ii<num_pieces-1; ++ii) {
            if( rho > _rho(ii) and rho < _rho(ii+1) ) {
                return ii ; 
            }
        }
        return num_pieces - 1 ;
    } 

 public:
    double eos_rhomin ; 
    double eos_rhomax ; 


} ; 

}

#endif /* GRACE_PHYSICS_EOS_PWPOLY_EOS_HH */