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
#include <grace/data_structures/memory_defaults.hh>
#include <grace/utils/rootfinding.hh>

#include <Kokkos_Core.hpp>

namespace grace {
/**
 * @brief Concrete cold EOS type corresponding to 
 *        a (piece-wise) polytrope. This class 
 *        should not be used on its own but rather
 *        serve as a template parameter for the 
 *        <code>hybrid_eos_t</code> class.
 * \ingroup eos
 * This class implements the minimum interface required
 * for the hybrid thermal extension.
 */
class piecewise_polytropic_eos_t
{
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
    press_cold_eps_cold__rho(double& eps_cold, double& rho) const 
    {
        auto idx = find_index_rho(rho) ; 
        double const press_cold = _k(idx) * Kokkos::pow(rho, _gamma(idx)) ; 
        eps_cold = _eps(idx) + press_cold / (rho*(_gamma(idx)-1.)) ;
        return press_cold ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    dpress_cold_drho__rho(double& rho) const {
        auto idx = find_index_rho(rho) ; 
        auto press_cold = _k(idx) * Kokkos::pow(rho, _gamma(idx)) ; 
        return _gamma(idx) * press_cold / rho ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    rho__press_cold(double & press) const {
        unsigned int idx = find_index_press(press) ; 
        return Kokkos::pow(press/_k(idx), 1./_gamma(idx)) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    rho__energy_cold(double & e) const {
        auto func = [=,this] (double rho, double& f, double& df) {
            unsigned int idx = find_index_rho(rho) ; 
            double g = _gamma(idx) ; 
            double K = _k(idx) ; 
            double ei = rho * ( 1. + _eps(idx) ) ; 

            f = (ei - e)+ K/(g-1.)  * pow(rho,g) ; 
            df = 1. + _eps(idx) + g*K/(g-1.) * pow(rho,g-1.) ; 
        } ;
        int rerr ; 
        // TODO: should we rootfind in logrho? 
        // rho can never exceed e = rho ( 1 + eps )
        // and the function is monotonic
        auto rho = utils::rootfind_newton_raphson(
            0.0, e, func, 30, 1e-15, rerr
        ) ; 
        if ( rerr != 0 ) {
            return _rho(0) ; 
        }
        return rho ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    energy_cold__press_cold(double & press) const {
        if(press<=0) return 0.0;
        unsigned int idx = find_index_press(press) ; 
        double const rho =  Kokkos::pow(press/_k(idx), 1./_gamma(idx));
        return rho + rho * _eps(idx) + press/(_gamma(idx)-1.) ; 
    }

 private:
    

    Kokkos::View<double [max_n_pieces], grace::default_space> _k     ; 
    Kokkos::View<double [max_n_pieces], grace::default_space> _gamma ; 
    Kokkos::View<double [max_n_pieces], grace::default_space> _rho   ;
    Kokkos::View<double [max_n_pieces], grace::default_space> _eps   ;
    Kokkos::View<double [max_n_pieces], grace::default_space> _press ; 

    unsigned int num_pieces ; 

    unsigned int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    find_index_rho(double& rho) const {

        rho = fmax(eos_rhomin, fmin(eos_rhomax, rho)) ; 
        for( int ii=0; ii<num_pieces-1; ++ii) {
            if( rho > _rho(ii) and rho <= _rho(ii+1) ) {
                return ii ; 
                break ; 
            }
        }
        return num_pieces - 1 ;
    } 

    unsigned int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    find_index_press(double& press) const {
        if( press < _press(0) ) {
            press = _press(0) ; 
            return 0 ; 
        }
        
        for( int ii=0; ii<num_pieces-1; ++ii) {
            if( press > _press(ii) and press <= _press(ii+1)) {
                return ii;
                break ; 
            }
        }
        return num_pieces-1 ;
    } 

    unsigned int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    find_index_ener(double& e) const {
        double emin = _rho(0) * (1.0 + _eps(0));
        double emax = _rho(num_pieces-1) * (1.0 + _eps(num_pieces-1));
        if( e < emin ) {
            e = emin ; 
            return 0 ; 
        } else if ( e > emax ) {
            e = emax ; 
            return num_pieces-1;
        }

        for( int ii=0; ii<num_pieces-1; ++ii) {
            if( e > _rho(ii) * (1.0 + _eps(ii)) and e <= _rho(ii+1) * (1. + _eps(ii+1))) {
                return ii ;
                break ; 
            }
        }
        return num_pieces-1 ;
    } 

    public:
    double eos_rhomin, eos_rhomax ; 
    double h_minimum ; 

} ; 

} /* namespace grace */

#endif /* GRACE_PHYSICS_EOS_PWPOLY_EOS_HH */