/**
 * @file ideal_gas_eos.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-04-08
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

#ifndef GRACE_PHYSICS_IDEAL_GAS_EOS_HH
#define GRACE_PHYSICS_IDEAL_GAS_EOS_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/physics/eos/eos_base.hh>

#include <grace/config/config_parser.hh>

#include <Kokkos_Core.hpp>

namespace grace {
/**
 * @brief Concrete EOS type corresponding to 
 *        an ideal gas.
 * \ingroup eos
 * The methods of this class are explicit implementations
 * of public methods from <code>eos_base_t</code>.
 */
class ideal_gas_eos_t 
    : public eos_base_t<ideal_gas_eos_t> 
{
    using error_type = eos_err_t;  
    public:

    ideal_gas_eos_t() = default ; 

    ideal_gas_eos_t(double temp_atm, double c2p_epsmax)
     : eos_base_t<ideal_gas_eos_t>(
        /*rhomax*/  1e100,
        /*rhomin*/  1e-100, 
        /*tmax*/    1e100,
        /*tmin*/    0.0,
        /*yemax*/   1e100,
        /*yemin*/   0.0,
        /*mass*/    1.0,
        /*eps_min*/ 0.0,
        /*eps_max*/ c2p_epsmax, 
        /*h_min*/   1.+1e-15,
        /*s_max*/   1e100,
        /*t_atm*/   temp_atm,
        /*ye_atm*/  0,
        /*atm_is_beta*/ true,
        /*nonsense*/ false
     )
    {
        gamma_m1 = grace::get_param<double>("eos","ideal_gas_eos","gamma") - 1. ; 
        k0       = grace::get_param<double>("eos","ideal_gas_eos","kappa_id")   ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press__eps_rho_ye_impl(double& eps, double& rho, double& ye, error_type& err) const 
    {
        limit_eps(eps,err) ; 
        return press__eps_rho(eps,rho) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press_temp__eps_rho_ye_impl(double& temp, double& eps, double& rho, double& ye, error_type& err) const
    {
        limit_eps(eps,err) ; 
        temp = temp__eps(eps) ; 
        return press__eps_rho(eps,rho) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press__temp_rho_ye_impl(double& temp, double& rho, double& ye, error_type& err) const 
    {
        limit_temp(temp, err) ; 
        return rho * temp;  
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps__temp_rho_ye_impl(double& temp, double& rho, double& ye, error_type& err) const 
    {
        limit_temp(temp, err) ; 
        return eps__temp(temp) ; 
    }

    // initial data interface
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_cold__rho_impl(double& rho, error_type& err) const 
    {
        return k0 * Kokkos::pow(rho, 1+gamma_m1) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    rho__press_cold_impl(double& press_cold, error_type& err) const 
    {
        // note that if k0 is zero this will be nan,
        // deal with it
        return Kokkos::pow(press_cold/k0, 1./(gamma_m1+1.)) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    energy_cold__press_cold_impl(double& press_cold, error_type& err) const 
    {
        double rho = rho__press_cold_impl(press_cold,err) ; 
        double eps = press_cold / (rho*gamma_m1) ; 
        return eps * ( 1 + rho ) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    rho__energy_cold_impl(double& e_cold, error_type& err) const 
    {
        // need a lil rootfind 
        double gloc = gamma_m1 + 1 ; 
        double kloc = k0 ; 
        auto const froot = [&] (double rho) {
            double p = kloc * Kokkos::pow(rho,gloc) ; 
            double e = rho * ( 1 + kloc * Kokkos::pow(rho,gloc-1) /(gloc-1) ) ; 
            return e - e_cold ; 
        } ; 

        return utils::brent(froot,0, e_cold, 1e-14) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_cold__rho_impl(double& rho, error_type& err) const 
    {
        double press_cold = press_cold__rho_impl(rho,err ) ; 
        return press_cold / (rho*gamma_m1) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    ye_cold__rho_impl(double const& rho, error_type& err) const {
        return 0. ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    ye_cold__press_impl(double const& press, error_type& err) const {
        return 0. ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    temp_cold__rho_impl(double& rho, error_type& err) const 
    {
        return 0. ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    entropy_cold__rho_impl(double& rho, error_type& err) const 
    {
        return  entropy__eps_rho(ent_min, rho) ; 
    }

    // back to normal API 

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_range__rho_ye( double& eps_min, double& eps_max
                     , double &rho, double &ye, error_type &err) const 
    {
        eps_min = 0.0 ; 
        eps_max = this->c2p_eps_max ; 
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    entropy_range__rho_ye( double& s_min, double& s_max
                         , double &rho, double &ye, error_type &err) const 
    { 
        s_min = entropy__eps_rho(ent_min,rho) ; 
        s_max = entropy__eps_rho(this->c2p_eps_max,rho) ; 
    }

    // the "get everything" functions 
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press_h_csnd2__eps_rho_ye_impl( double &h, double &csnd2, double &eps
                                  , double &rho, double &ye
                                  , error_type &err) const
    {
        limit_eps(eps,err); 
        const double press  = press__eps_rho(eps,rho) ; 
        h = h__eps_press_rho(eps,press,rho); 
        csnd2 = csnd2__eps_h(eps,h) ; 
        return press ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press_h_csnd2__temp_rho_ye_impl( double &h, double &csnd2, double &temp
                                   , double &rho, double &ye
                                   , error_type &err) const
    {
        limit_temp(temp,err)                   ; 
        double eps = eps__temp(temp)           ; 
        double press = press__eps_rho(eps,rho) ; 
        h = h__eps_press_rho(eps,press,rho)    ;
        csnd2 = csnd2__eps_h(eps,h) ; 
        return press ;  
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye_impl( double &eps, double &csnd2, double &temp
                                     , double &rho, double &ye
                                     , error_type& err) const 
    {
        limit_temp(temp,err)                   ; 
        eps = eps__temp(temp)           ; 
        double press = press__eps_rho(eps,rho) ; 
        double h = h__eps_press_rho(eps,press,rho)    ;
        csnd2 = csnd2__eps_h(eps,h) ; 
        return press ;   
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye_impl( double& h, double& csnd2, double& temp 
                                               , double& entropy, double& eps 
                                               , double& rho, double& ye 
                                               , error_type& err ) const
    {
        limit_eps(eps,err) ; 
        double press = press__eps_rho(eps,rho) ; 
        h = h__eps_press_rho(eps,press,rho) ; 
        csnd2 = csnd2__eps_h(eps,h) ; 
        temp = temp__eps(eps) ; 
        entropy = entropy__eps_rho(eps,rho) ; 
        return press ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_csnd2_entropy__temp_rho_ye_impl( double& csnd2, double& entropy, double& temp 
                                       , double& rho, double& ye 
                                       , error_type& err ) const 
    {
        limit_temp(temp,err)                   ; 
        double eps = eps__temp(temp)           ; 
        double press = press__eps_rho(eps,rho) ; 
        double h = h__eps_press_rho(eps,press,rho)    ;
        csnd2 = csnd2__eps_h(eps,h) ; 
        entropy = entropy__eps_rho(eps,rho) ; 
        return eps ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_eps_csnd2_entropy__temp_rho_ye_impl( double& eps, double& csnd2, double& entropy, double& temp 
                                       , double& rho, double& ye 
                                       , error_type& err ) const 
    {
        limit_temp(temp,err)                   ; 
        eps = eps__temp(temp)           ; 
        double press = press__eps_rho(eps,rho) ; 
        double h = h__eps_press_rho(eps,press,rho)    ;
        csnd2 = csnd2__eps_h(eps,h) ; 
        entropy = entropy__eps_rho(eps,rho) ; 
        return press ;  
    }

    // entropy inversion 
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2_temp_eps__entropy_rho_ye_impl( double& h, double& csnd2, double& temp
                                               , double& eps, double& entropy, double& rho 
                                               , double& ye, error_type& err) const
    {
        limit_entropy(entropy,rho,err) ; 
        eps = eps__entropy_rho(entropy,rho) ; 
        double press = press__eps_rho(eps,rho) ; 
        h     = h__eps_press_rho(eps,press,rho) ; 
        temp  = temp__eps(eps);
        csnd2 = csnd2__eps_h(eps,h) ; 
        return press ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_h_csnd2_temp_entropy__press_rho_ye_impl( double& h, double& csnd2, double& temp
                                               , double& entropy, double& press, double& rho 
                                               , double& ye, error_type& err) const 
    {
        double eps = press/(gamma_m1*rho)  ; 
        h       = h__eps_press_rho(eps,press,rho) ; 
        csnd2   = csnd2__eps_h(eps,h) ; 
        entropy = entropy__eps_rho(eps,rho) ; 
        temp    = temp__eps(eps) ; 
        return eps; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_impl( 
        double &mup, double &mun, double &Xa, double &Xh, double &Xn, double &Xp
      , double &Abar, double &Zbar, double &temp, double &rho, double &ye
      , error_type &err) const
    {
        mup = 0. ; 
        mun = 0. ;
        Xa  = 0. ; 
        Xh  = 0. ; 
        Xn  = 1. ; 
        Xp  = 0. ; 
        Abar = 1. ; 
        Zbar = 0. ; 
        return 0. ; 
    }

    private:
    //==
    double gamma_m1 ; //!< Gamma - 1 
    double k0       ; //!< Adiabat for initial data 
    double ent_min  ; //!< Low bound for s 
    //==

    void KOKKOS_INLINE_FUNCTION 
    limit_eps(double& eps, error_type& err) const 
    {
        if ( eps > this->c2p_eps_max ) {
            err.set(EOS_ERROR_T::EOS_EPS_TOO_HIGH) ; 
            eps = this->c2p_eps_max ; 
        }
    }

    void KOKKOS_INLINE_FUNCTION 
    limit_temp(double& temp, error_type& err) const 
    {
        double tempmax = temp__eps(this->c2p_eps_max) ; 
        if ( temp < 0 ) {
            temp = 0.0 ; 
            err.set(EOS_ERROR_T::EOS_TEMPERATURE_TOO_LOW) ; 
        }
        if ( temp > tempmax ) {
            err.set(EOS_ERROR_T::EOS_TEMPERATURE_TOO_HIGH) ; 
            temp = tempmax ; 
        }
    }

    void KOKKOS_INLINE_FUNCTION
    limit_entropy(double& s, double rho, error_type& err) const 
    {
        double smin = entropy__eps_rho(ent_min, rho) ; 
        double smax = entropy__eps_rho(this->c2p_eps_max, rho) ; 
        if ( s < smin ) {
            s = smin ; 
            err.set(EOS_ERROR_T::EOS_ENTROPY_TOO_LOW) ; 
        }
        if ( s > smax ) {
            s = smax ; 
            err.set(EOS_ERROR_T::EOS_ENTROPY_TOO_HIGH) ; 
        }
    }

    double KOKKOS_INLINE_FUNCTION 
    entropy__eps_rho(double eps, double rho) const 
    {
        return Kokkos::log(eps * pow(rho, -gamma_m1)) / (gamma_m1);
    }

    double KOKKOS_INLINE_FUNCTION 
    eps__entropy_rho(double entropy, double rho) const 
    {
        return Kokkos::exp(gamma_m1 * entropy) * pow(rho, gamma_m1);
    }

    double KOKKOS_INLINE_FUNCTION 
    temp__eps(double eps) const { return eps * gamma_m1 ; } 

    double KOKKOS_INLINE_FUNCTION 
    eps__temp(double eps) const { return eps / gamma_m1 ; } 

    double KOKKOS_INLINE_FUNCTION
    h__eps_press_rho(double eps, double p, double rho) const 
    {
        return 1 + eps + p/rho; 
    }

    double KOKKOS_INLINE_FUNCTION
    csnd2__eps_h(double eps, double h) const 
    {
        return (gamma_m1+1) * gamma_m1 * eps / h ; 
    }

    double KOKKOS_INLINE_FUNCTION 
    press__eps_rho(double eps, double rho) const { return gamma_m1 * eps * rho ; }
} ; 

}

#endif 