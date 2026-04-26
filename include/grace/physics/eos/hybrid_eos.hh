/**
 * @file hybrid_eos.hh
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

#ifndef GRACE_PHYSICS_HYBRID_EOS_HH
#define GRACE_PHYSICS_HYBRID_EOS_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/physics/eos/eos_base.hh>

#include <Kokkos_Core.hpp>

namespace grace {
/**
 * @brief Concrete EOS type corresponding to 
 *        a cold EOS with a ideal gas thermal 
 *        extension.
 * \ingroup eos
 * @tparam cold_eos_t Type of cold EOS. 
 * The methods of this class are explicit implementations
 * of public methods from <code>eos_base_t</code>.
 */
template < typename cold_eos_t >
class hybrid_eos_t
    : public eos_base_t<hybrid_eos_t<cold_eos_t>> 
{
    using error_type = eos_err_t ;  
 public:
    
    hybrid_eos_t() = default ; 

    hybrid_eos_t( cold_eos_t _cold_eos 
                , double _gamma_th_m1 
                , double _entropy_min
                , double _h_min
                , double baryon_mass
                , double c2p_eps_max 
                , double c2p_temp_atm )
     : eos_base_t<hybrid_eos_t<cold_eos_t>>{ _cold_eos.eos_rhomax, _cold_eos.eos_rhomin
                                           , 1e99, 0
                                           , 1e99, 0
                                           , baryon_mass
                                           , 0
                                           , c2p_eps_max
                                           , _h_min
                                           , 1.e99
                                           , c2p_temp_atm
                                           , 0.0
                                           , false }
     , gamma_th_m1(_gamma_th_m1)
     , entropy_min(_entropy_min)
     , cold_eos(_cold_eos)
    {}

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press__eps_rho_ye_impl(double& eps, double& rho, double& ye, error_type& err) const 
    {
        check_rho_range(rho, err) ; 
        double eps_cold ;
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold,rho) ; 
        check_eps_range(eps,eps_cold,rho,err) ;
        return press_cold + (eps-eps_cold) * rho * gamma_th_m1 ;  
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press_temp__eps_rho_ye_impl(double& temp, double& eps, double& rho, double& ye, error_type& err) const
    {
        check_rho_range(rho, err) ; 
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold,rho) ; 
        check_eps_range(eps,eps_cold,rho,err) ;

        double eps_th = eps-eps_cold ; 
        temp = temp__eps_th(eps_th) ;
        return press_cold + eps_th * rho * gamma_th_m1 ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press__temp_rho_ye_impl(double& temp, double& rho, double& ye, error_type& err) const 
    {
        check_rho_range(rho, err)   ; 
        check_temp_range(temp, err) ; 
        double eps_cold;
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        return press_cold + rho * temp;  
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps__temp_rho_ye_impl(double& temp, double& rho, double& ye, error_type& err) const 
    {
        check_rho_range(rho, err)   ; 
        check_temp_range(temp, err) ; 
        double eps_cold;
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        temp = math::max(0,temp);
        return eps_cold + eps_th__temp(temp) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_cold__rho_impl(double& rho, error_type& err) const 
    {
        check_rho_range(rho, err)   ; 
        double eps_cold ; 
        return cold_eos.press_cold_eps_cold__rho(eps_cold,rho) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    energy_cold__press_cold_impl(double& press_cold, error_type& err) const 
    {
        //check_press_cold_range(press_cold,err)   ;
        return cold_eos.energy_cold__press_cold(press_cold) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    rho__press_cold_impl(double& press_cold, error_type& err) const 
    {
        //check_press_cold_range(press_cold,err)   ;
        return cold_eos.rho__press_cold(press_cold) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    rho__energy_cold_impl(double& e_cold, error_type& err) const 
    {
        return cold_eos.rho__energy_cold(e_cold) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_cold__rho_impl(double& rho, error_type& err) const 
    {
        check_rho_range(rho, err) ; 
        double eps_cold ; 
        auto dummy = cold_eos.press_cold_eps_cold__rho(eps_cold,rho) ; 
        return eps_cold ;
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
        return 0. ; 
    }


    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_range__rho_ye( double& eps_min, double& eps_max
                     , double &rho, double &ye, error_type &err) const 
    {
        check_rho_range(rho,err) ; 
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        eps_min = math::min(eps_cold,this->c2p_eps_max) ; 
        eps_max = this->c2p_eps_max ; 
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    entropy_range__rho_ye( double& s_min, double& s_max
                         , double &rho, double &ye, error_type &err) const 
    {
        check_rho_range(rho,err) ; 
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        auto eps_th_max = this->c2p_eps_max - eps_cold ;
        double ethmin_guarded = entropy_min ;  
        s_min = entropy__eps_th_rho(ethmin_guarded,rho) ; 
        s_max = entropy__eps_th_rho(eps_th_max,rho) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press_h_csnd2__eps_rho_ye_impl( double &h, double &csnd2, double &eps
                                  , double &rho, double &ye
                                  , error_type &err) const
    {
        check_rho_range(rho,err) ; 
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        check_eps_range(eps,eps_cold,rho,err) ;
        const double eps_th = eps-eps_cold ; 
        const double press  = press_cold + eps_th * rho * gamma_th_m1 ; 
        h = 1. + eps + press/rho ; 
        csnd2 = (cold_eos.dpress_cold_drho__rho(rho) + gamma_th_m1 * (gamma_th_m1+1) * eps_th) / h; 
        return press ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    press_h_csnd2__temp_rho_ye_impl( double &h, double &csnd2, double &temp
                                   , double &rho, double &ye
                                   , error_type &err) const
    {
        check_rho_range(rho,err) ; 
        check_temp_range(temp, err) ; 
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        const double eps_th = eps_th__temp(temp) ; 
        const double press  = press_cold + eps_th * rho * gamma_th_m1 ; 
        h = 1. + eps_cold + eps_th + press/rho ; 
        csnd2 = (cold_eos.dpress_cold_drho__rho(rho) + gamma_th_m1 * (gamma_th_m1+1) * eps_th) / h; 
        return press ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye_impl( double &eps, double &csnd2, double &temp
                                     , double &rho, double &ye
                                     , error_type& err) const 
    {
        check_rho_range(rho,err) ; 
        check_temp_range(temp, err) ; 
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        const double eps_th = eps_th__temp(temp) ; 
        const double press  = press_cold + eps_th * rho * gamma_th_m1 ; 
        eps = eps_cold + eps_th ;
        double const h = 1. + eps + press/rho ; 
        csnd2 = (cold_eos.dpress_cold_drho__rho(rho) + gamma_th_m1 * (gamma_th_m1+1) * eps_th) / h; 
        return press ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye_impl( double& h, double& csnd2, double& temp 
                                               , double& entropy, double& eps 
                                               , double& rho, double& ye 
                                               , error_type& err ) const
    {
        check_rho_range(rho,err) ; 
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        check_eps_range(eps,eps_cold,rho,err) ;
        double eps_th = eps-eps_cold ;
        double const press  = press_cold + eps_th * rho * gamma_th_m1 ; 
        h = 1. + eps + press / rho ; 
        csnd2 = (cold_eos.dpress_cold_drho__rho(rho) + gamma_th_m1 * (gamma_th_m1+1) * eps_th) / h;
        entropy = entropy__eps_th_rho(eps_th,rho) ; 
        temp = temp__eps_th(eps_th) ; 
        return press ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_csnd2_entropy__temp_rho_ye_impl( double& csnd2, double& entropy, double& temp 
                                       , double& rho, double& ye 
                                       , error_type& err ) const 
    {
        check_rho_range(rho, err)   ; 
        check_temp_range(temp, err) ;
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        double eps_th = eps_th__temp(temp) ; 
        const double press  = press_cold + eps_th * rho * gamma_th_m1 ;
        const double eps = eps_th + eps_cold ; 
        const double h = 1. + eps + press / rho ; 
        csnd2 = (cold_eos.dpress_cold_drho__rho(rho) + gamma_th_m1 * (gamma_th_m1+1) * eps_th) / h;
        entropy = entropy__eps_th_rho(eps_th,rho) ; 
        return eps ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_eps_csnd2_entropy__temp_rho_ye_impl( double& eps, double& csnd2, double& entropy, double& temp 
                                       , double& rho, double& ye 
                                       , error_type& err ) const 
    {
        check_rho_range(rho, err)   ; 
        check_temp_range(temp, err) ;
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        double eps_th = eps_th__temp(temp) ; 
        const double press  = press_cold + eps_th * rho * gamma_th_m1 ;
        eps = eps_th + eps_cold ; 
        const double h = 1. + eps + press / rho ; 
        csnd2 = (cold_eos.dpress_cold_drho__rho(rho) + gamma_th_m1 * (gamma_th_m1+1) * eps_th) / h;
        entropy = entropy__eps_th_rho(eps_th,rho) ; 
        return press ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2_temp_eps__entropy_rho_ye_impl( double& h, double& csnd2, double& temp
                                               , double& eps, double& entropy, double& rho 
                                               , double& ye, error_type& err) const
    {
        check_rho_range(rho, err)   ; 
        check_entropy_range(entropy, rho, err) ;
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        double eps_th = exp(gamma_th_m1 * entropy) * pow(rho, gamma_th_m1);
        const double press  = press_cold + eps_th * rho * gamma_th_m1 ; 
        temp = temp__eps_th(eps_th);
        eps = eps_cold + eps_th ; 
        h   = 1. + eps + press/rho ; 
        csnd2 = (cold_eos.dpress_cold_drho__rho(rho) + gamma_th_m1 * (gamma_th_m1+1) * eps_th) / h;
        return press ;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_h_csnd2_temp_entropy__press_rho_ye_impl( double& h, double& csnd2, double& temp
                                               , double& entropy, double& press, double& rho 
                                               , double& ye, error_type& err) const 
    {
        check_rho_range(rho, err)   ; 
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        press = math::max(press,press_cold) ; 
        double eps_th = eps_th__press_press_cold_rho(press,press_cold,rho) ; 
        double const eps = eps_th + eps_cold ; 
        h   = 1. + eps + press/rho ; 
        csnd2 = (cold_eos.dpress_cold_drho__rho(rho) + gamma_th_m1 * (gamma_th_m1+1) * eps_th) / h;
        entropy = entropy__eps_th_rho(eps_th,rho) ; 

        temp = temp__eps_th(eps_th) ; 
        return eps ; 
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

    double gamma_th_m1 ; 
    double entropy_min ; 

    cold_eos_t cold_eos ;


    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    eps_th__eps_rho(double& eps, double& rho, error_type& err) const {
        double eps_cold ; 
        auto press_cold = cold_eos.press_cold_eps_cold__rho(eps_cold, rho);
        eps = math::max(eps,eps_cold) ; 
        return eps-eps_cold ;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    eps_th__press_press_cold_rho( double& press, double& press_cold, double& rho) const 
    {
        return fmax(0, (press-press_cold) / (rho*gamma_th_m1)) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    entropy__eps_th_rho( double& eps_th, double& rho) const 
    {
        const auto eps_thL = std::max(eps_th, entropy_min);
        const double entropy = log(eps_thL * pow(rho, -gamma_th_m1)) / (gamma_th_m1);
        return entropy;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    temp__eps_th( double& eps_th) const 
    {
        return fmax(0.0, eps_th * gamma_th_m1 ) ;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    eps_th__temp( double& temp) const 
    {
        return fmax(0.0, temp / gamma_th_m1 ) ;
    }

    void KOKKOS_INLINE_FUNCTION 
    check_eps_range(double& eps, double eps_cold, double rho, error_type& err) const {
        double epsmin = math::min(eps_cold,this->c2p_eps_max) ; 
        double epsmax = this->c2p_eps_max ; 
        if ( eps < epsmin ) {
            err.set(EOS_ERROR_T::EOS_EPS_TOO_LOW) ; 
            eps = epsmin ; 
        } 
        if ( eps > epsmax ) {
            err.set(EOS_ERROR_T::EOS_EPS_TOO_HIGH) ; 
            eps = epsmax ;
        }
    }

    void KOKKOS_INLINE_FUNCTION 
    check_entropy_range(double& ent, double rho, error_type& err) const {
        double entmin, entmax;
        double dummy{0} ; 
        entropy_range__rho_ye(entmin, entmax, rho,dummy,err) ; 
        if ( ent < entmin ) {
            err.set(EOS_ERROR_T::EOS_ENTROPY_TOO_LOW) ; 
            ent = entmin ; 
        } 
        if ( ent > entmax ) {
            err.set(EOS_ERROR_T::EOS_ENTROPY_TOO_HIGH) ; 
            ent = entmax ;
        }
    }

    void KOKKOS_INLINE_FUNCTION
    check_rho_range(double& rho, error_type& err) const {
        if ( rho < this->eos_rhomin ) {
            err.set(EOS_ERROR_T::EOS_RHO_TOO_LOW) ; 
            rho = this->eos_rhomin ; 
        }
        if ( rho > this->eos_rhomax) {
            err.set(EOS_ERROR_T::EOS_RHO_TOO_HIGH) ; 
            rho = this->eos_rhomax ; 
        }
    }

    void KOKKOS_INLINE_FUNCTION
    check_press_cold_range(double& press, error_type& err) const {
        double dummy ; 
        double rhomin{this->eos_rhomin}, rhomax{this->eos_rhomax}; 
        double pmin = cold_eos.press_cold_eps_cold__rho(dummy, rhomin) ; 
        double pmax = cold_eos.press_cold_eps_cold__rho(dummy, rhomax) ; 
        if ( press < pmin ) {
            err.set(EOS_ERROR_T::EOS_PRESS_TOO_LOW) ; 
            press = pmin ; 
        }
        if ( press > pmax ) {
            err.set(EOS_ERROR_T::EOS_PRESS_TOO_HIGH) ; 
            press = pmax ; 
        }
    }

    void KOKKOS_INLINE_FUNCTION
    check_temp_range(double& temp, error_type& err) const {
        if ( temp < this->eos_tempmin ) {
            err.set(EOS_ERROR_T::EOS_TEMPERATURE_TOO_LOW) ; 
            temp = this->eos_tempmin ; 
        }
        if ( temp > this->eos_tempmax) {
            err.set(EOS_ERROR_T::EOS_TEMPERATURE_TOO_HIGH) ; 
            temp = this->eos_tempmax ; 
        }
    }
    
    
} ; 

} /* namespace grace */

#endif /* GRACE_PHYSICS_HYBRID_EOS_HH */
