/**
 * @file eos_base.hh
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

#ifndef GRACE_PHYSICS_EOS_BASE_HH
#define GRACE_PHYSICS_EOS_BASE_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>

namespace grace {

enum EOS_ERROR_T {
    EOS_NO_ERROR=0,
    EOS_RHO_TOO_HIGH,
    EOS_RHO_TOO_LOW,
    EOS_NUM_ERRORS
} ; 

/**
 * @brief Base class for eos handling.
 * \ingroup eos
 * @tparam eos_impl_t Concrete eos type.
 * Equation of state classes derive from this 
 * interface class through CRTP inheritance. The 
 * minimal eos interface that needs to be implemented 
 * corresponds with the set of public methods of 
 * this class.
 */
template< typename eos_impl_t >
class eos_base_t {
    using error_type = unsigned int; 
 public:
    /**
     * @brief Default ctor.
     * 
     */
    eos_base_t() = default ; 
    /**
     * @brief Constructor
     * 
     * @param _energy_shift Table energy shift.
     * @param _eos_rhomax   Max rest-mass density.
     * @param _eos_rhomin   Min rest-mass density.
     * @param _eos_tempmax 
     * @param _eos_tempmin 
     * @param _eos_yemax 
     * @param _eos_yemin 
     * @param _baryon_mass 
     * @param _c2p_ye_atm 
     * @param _c2p_rho_atm 
     * @param _c2p_temp_atm 
     * @param _c2p_eps_atm 
     * @param _c2p_eps_min 
     * @param _c2p_eps_max 
     * @param _c2p_h_min 
     * @param _c2p_h_max 
     * @param _atm_is_beta_eq 
     * @param _extend_table_high 
     */
    eos_base_t( double _energy_shift, double _eos_rhomax, double _eos_rhomin
              , double _eos_tempmax, double _eos_tempmin
              , double _eos_yemax, double _eos_yemin
              , double _baryon_mass, double _c2p_ye_atm
              , double _c2p_rho_atm, double _c2p_temp_atm 
              , double _c2p_eps_atm, double _c2p_eps_min
              , double _c2p_eps_max, double _c2p_h_min
              , double _c2p_h_max, bool _atm_is_beta_eq
              , bool _extend_table_high )
     : energy_shift(_energy_shift), eos_rhomax(_eos_rhomax), eos_rhomin(_eos_rhomin)
     , eos_tempmax(_eos_tempmax), eos_tempmin(_eos_tempmin)
     , eos_yemax(_eos_yemax), eos_yemin(_eos_yemin)
     , baryon_mass(_baryon_mass), c2p_ye_atm(_c2p_ye_atm)
     , c2p_rho_atm(_c2p_rho_atm), c2p_temp_atm(_c2p_temp_atm)
     , c2p_eps_atm(_c2p_eps_atm), c2p_eps_min(_c2p_eps_min), c2p_eps_max(_c2p_eps_max)
     , c2p_h_min(_c2p_h_min), c2p_h_max(_c2p_h_max)
     , atm_is_beta_eq(_atm_is_beta_eq), extend_table_high(_extend_table_high)
    {}

    double GRACE_HOST_DEVICE
    press__eps_rho_ye(double& eps, double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press__eps_rho_ye_impl(eps,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    press_temp__eps_rho_ye(double& temp, double& eps, double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_temp__eps_rho_ye_impl(temp,eps,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE 
    press__temp_rho_ye(double& temp, double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press__temp_rho_ye_impl(temp,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE 
    eps__temp_rho_ye(double& temp, double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps__temp_rho_ye_impl(temp,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    press_cold__rho_ye(double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_cold__rho_ye_impl(rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    eps_cold__rho_ye(double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps_cold__rho_ye_impl(rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    temp_cold__rho_ye(double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->temp_cold__rho_ye_impl(rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE 
    eps__press_temp_rho_ye(double& press, double& temp, double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps__press_temp_rho_ye_impl(press,temp,rho,ye,err) ; 
    }

    void GRACE_HOST_DEVICE
    eps_range__rho_ye(double& eps_min, double& eps_max, double &rho, double &ye, error_type &err) const 
    {
        static_cast<eos_impl_t const*>(this)->eps_range__rho_ye_impl(eps_min,eps_max,rho,ye,err) ; 
    }

    void GRACE_HOST_DEVICE
    entropy_range__rho_ye(double& s_min, double& s_max, double &rho, double &ye, error_type &err) const 
    {
        static_cast<eos_impl_t const*>(this)->entropy_range__rho_ye_impl(s_min,s_max,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2__eps_rho_ye( double &h, double &csnd2, double &eps
                             , double &rho, double &ye
                             , error_type &err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_h_csnd2__eps_rho_ye_impl(h,csnd2,eps,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2__temp_rho_ye( double &h, double &csnd2, double &temp
                              , double &rho, double &ye
                              , error_type &err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_h_csnd2__temp_rho_ye_impl(h,csnd2,temp,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    eps_h_csnd2__press_rho_ye( double &h, double &csnd2, double &press
                             , double &rho, double &ye
                             , error_type &err) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps_h_csnd2__press_rho_ye_impl(h,csnd2,press,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye( double &eps, double &csnd2, double &temp
                                , double &rho, double &ye
                                , error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_eps_csnd2__temp_rho_ye_impl(eps,csnd2,temp,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye( double& h, double& csnd2, double& temp 
                                          , double& entropy, double& eps 
                                          , double& rho, double& ye 
                                          , error_type& err ) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_h_csnd2_temp_entropy__eps_rho_ye_impl(h,csnd2,temp,entropy,eps,rho,ye,err);
    }

    double GRACE_HOST_DEVICE
    eps_csnd2_entropy__temp_rho_ye( double& csnd2, double& entropy, double& temp 
                                  , double& rho, double& ye 
                                  , error_type& err ) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps_csnd2_entropy__temp_rho_ye_impl(csnd2,entropy,temp,rho,ye,err);
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_eps__entropy_rho_ye( double& h, double& csnd2, double& temp
                                          , double& eps, double& entropy, double& rho 
                                          , double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_h_csnd2_temp_eps__entropy_rho_ye_impl(h,csnd2,temp,eps,entropy,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    eps_h_csnd2_temp_entropy__press_rho_ye( double& h, double& csnd2, double& temp
                                          , double& entropy, double& press, double& rho 
                                          , double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps_h_csnd2_temp_entropy__press_rho_ye_impl(h,csnd2,temp,entropy,press,rho,ye,err) ; 
    }

    double GRACE_HOST_DEVICE
    press_eps_ye__beta_eq__rho_temp( double& eps, double& ye
                                   , double& rho, double& temp
                                   , error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_eps_ye__beta_eq__rho_temp_impl(eps,ye,rho,temp,err) ; 
    }

    double GRACE_HOST_DEVICE
    mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye( 
        double &mup, double &mun, double &Xa, double &Xh, double &Xn, double &Xp
      , double &Abar, double &Zbar, double &temp, double &rho, double &ye
      , error_type &err) const 
    {
        return static_cast<eos_impl_t const*>(this)->mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_impl(
            mup,mun,Xa,Xh,Xn,Xp,Abar,Zbar,temp,rho,ye,err
            ) ; 
    }

 protected:
    static constexpr bool has_ye = eos_impl_t::has_ye ; 

    double energy_shift ; 

    double eos_rhomax, eos_rhomin ;
    double eos_tempmax, eos_tempmin ;
    double eos_yemax, eos_yemin ; 

    double baryon_mass ;

    double c2p_ye_atm    ; 
    double c2p_rho_atm   ; 
    double c2p_temp_atm  ; 
    double c2p_eps_atm   ; 
    double c2p_eps_min   ; 
    double c2p_eps_max   ; 
    double c2p_h_min     ; 
    double c2p_h_max     ; 

    bool atm_is_beta_eq    ; 
    bool extend_table_high ; 

} ;

}

#endif /* GRACE_PHYSICS_EOS_BASE_HH */