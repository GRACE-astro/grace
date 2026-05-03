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
#include <grace/utils/bitset.hh>

namespace grace {
/**
 * @brief EOS errors.
 */
enum EOS_ERROR_T {
    EOS_RHO_TOO_HIGH=0,
    EOS_RHO_TOO_LOW,
    EOS_EPS_TOO_HIGH,
    EOS_EPS_TOO_LOW,
    EOS_YE_TOO_LOW,
    EOS_YE_TOO_HIGH,
    EOS_TEMPERATURE_TOO_LOW,
    EOS_TEMPERATURE_TOO_HIGH,
    EOS_ENTROPY_TOO_LOW,
    EOS_ENTROPY_TOO_HIGH,
    EOS_PRESS_TOO_LOW,
    EOS_PRESS_TOO_HIGH,
    EOS_NUM_ERRORS
} ; 

using eos_err_t = bitset_t<EOS_NUM_ERRORS> ; 

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
    //! Error codes are GPU-friendly.
    using error_type = eos_err_t; 
 public:
    /**
     * @brief Default ctor.
     */
    eos_base_t() = default ; 
    /**
     * @brief Constructor.
     * 
     * @param _energy_shift Table energy shift.
     * @param _eos_rhomax   Max rest-mass density.
     * @param _eos_rhomin   Min rest-mass density.
     * @param _eos_tempmax  Max temperature.
     * @param _eos_tempmin  Min temperature.
     * @param _eos_yemax    Max electron fraction.
     * @param _eos_yemin    Min electron fraction.
     * @param _baryon_mass  Baryonic mass.
     * @param _c2p_ye_atm   Atmosphere Y_e.
     * @param _c2p_rho_atm  Atmosphere rest-mass density.
     * @param _c2p_temp_atm Atmosphere temperature.
     * @param _c2p_eps_atm  Atmosphere specific internal energy.
     * @param _c2p_eps_min  Minimum specific internal energy.
     * @param _c2p_eps_max  Maximum specific internal energy.
     * @param _c2p_h_min    Minimum specific enthalpy.
     * @param _c2p_h_max    Maximum specicif enthalpy.
     * @param _atm_is_beta_eq  Is atmosphere in beta-equilibrium? 
     * @param _extend_table_high  Extend table for high rest-mass density? 
     */
    eos_base_t( double _eos_rhomax, double _eos_rhomin
              , double _eos_tempmax, double _eos_tempmin
              , double _eos_yemax, double _eos_yemin
              , double _baryon_mass, double _c2p_eps_min
              , double _c2p_eps_max, double _c2p_h_min
              , double _c2p_h_max, double _c2p_temp_atm, double _c2p_ye_atm, bool _atm_is_beta_eq
              , bool _extend_table_high )
     : eos_rhomax(_eos_rhomax), eos_rhomin(_eos_rhomin)
     , eos_tempmax(_eos_tempmax), eos_tempmin(_eos_tempmin)
     , eos_yemax(_eos_yemax), eos_yemin(_eos_yemin)
     , baryon_mass(_baryon_mass), c2p_eps_min(_c2p_eps_min), c2p_eps_max(_c2p_eps_max)
     , c2p_h_min(_c2p_h_min), c2p_h_max(_c2p_h_max), c2p_temp_atm(_c2p_temp_atm), c2p_ye_atm(_c2p_ye_atm)
     , atm_is_beta_eq(_atm_is_beta_eq), extend_table_high(_extend_table_high)
    {}
    /**
     * @brief Get pressure given eps rho and ye.
     * 
     * @param eps Specific internal energy.
     * @param rho Rest-mass density.
     * @param ye  Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press__eps_rho_ye(double& eps, double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press__eps_rho_ye_impl(eps,rho,ye,err) ; 
    }
    /**
     * @brief Get pressure and temperature given eps rho and ye.
     * 
     * @param temp Temperature.
     * @param eps Specific internal energy.
     * @param rho Rest-mass density.
     * @param ye  Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_temp__eps_rho_ye(double& temp, double& eps, double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_temp__eps_rho_ye_impl(temp,eps,rho,ye,err) ; 
    }
    /**
     * @brief Get pressure given temperature rho and ye.
     * 
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye  Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE 
    press__temp_rho_ye(double& temp, double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press__temp_rho_ye_impl(temp,rho,ye,err) ; 
    }
    /**
     * @brief Get eps given temperature rho and ye.
     * 
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The specific internal energy.
     */
    double GRACE_HOST_DEVICE 
    eps__temp_rho_ye(double& temp, double& rho, double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps__temp_rho_ye_impl(temp,rho,ye,err) ; 
    }
    /**
     * @brief Range of epsilon given rho and ye.
     * 
     * @param eps_min Minimum epsilon.
     * @param eps_max Maximum epsilon.
     * @param rho     Rest-mass density.
     * @param ye      Electron fraction.
     * @param err     Error code.
     */
    void GRACE_HOST_DEVICE
    eps_range__rho_ye(double& eps_min, double& eps_max, double &rho, double &ye, error_type &err) const 
    {
        static_cast<eos_impl_t const*>(this)->eps_range__rho_ye_impl(eps_min,eps_max,rho,ye,err) ; 
    }
    /**
     * @brief Range of entropy given rho and ye.
     * 
     * @param eps_min Minimum entropy.
     * @param eps_max Maximum entropy.
     * @param rho     Rest-mass density.
     * @param ye      Electron fraction.
     * @param err     Error code.
     */
    void GRACE_HOST_DEVICE
    entropy_range__rho_ye(double& s_min, double& s_max, double &rho, double &ye, error_type &err) const 
    {
        static_cast<eos_impl_t const*>(this)->entropy_range__rho_ye_impl(s_min,s_max,rho,ye,err) ; 
    }
    /**
     * @brief Pressure, specific enthalpy and 
     *        square sound speed given eps, 
     *        rho and ye.
     * 
     * @param h Specific enthalpy.
     * @param csnd2 Sound speed squared.
     * @param eps Specific internal energy.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_h_csnd2__eps_rho_ye( double &h, double &csnd2, double &eps
                             , double &rho, double &ye
                             , error_type &err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_h_csnd2__eps_rho_ye_impl(h,csnd2,eps,rho,ye,err) ; 
    }
    /**
     * @brief Pressure, specific enthalpy and 
     *        square sound speed given temp, 
     *        rho and ye.
     * 
     * @param h Specific enthalpy.
     * @param csnd2 Sound speed squared.
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_h_csnd2__temp_rho_ye( double &h, double &csnd2, double &temp
                              , double &rho, double &ye
                              , error_type &err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_h_csnd2__temp_rho_ye_impl(h,csnd2,temp,rho,ye,err) ; 
    }
    /**
     * @brief Pressure, specific internal energy
     *        and square sound speed given temperature,
     *        rho and ye.
     * 
     * @param eps Specific internal energy.
     * @param csnd2 Square sound speed.
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye( double &eps, double &csnd2, double &temp
                                , double &rho, double &ye
                                , error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_eps_csnd2__temp_rho_ye_impl(eps,csnd2,temp,rho,ye,err) ; 
    }
    /**
     * @brief Pressure, specific enthalpy,
     *        square sound speed, temperature
     *        and entropy given epsilon,
     *        rho and ye.
     * 
     * @param h Specific enthalpy.
     * @param csnd2 Square sound speed.
     * @param temp Temperature.
     * @param entropy Entropy per baryon.
     * @param eps Specific internal energy.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye( double& h, double& csnd2, double& temp 
                                          , double& entropy, double& eps 
                                          , double& rho, double& ye 
                                          , error_type& err ) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_h_csnd2_temp_entropy__eps_rho_ye_impl(h,csnd2,temp,entropy,eps,rho,ye,err);
    }
    /**
     * @brief Epsilon, square sound speed and entropy
     *        given temperature, rho and ye.
     * 
     * @param csnd2 Square sound speed.
     * @param entropy Entropy per baryon.
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The specific internal energy.
     */
    double GRACE_HOST_DEVICE
    eps_csnd2_entropy__temp_rho_ye( double& csnd2, double& entropy, double& temp 
                                  , double& rho, double& ye 
                                  , error_type& err ) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps_csnd2_entropy__temp_rho_ye_impl(csnd2,entropy,temp,rho,ye,err);
    }
    /**
     * @brief Pressure, epsilon, square sound speed and entropy
     *        given temperature, rho and ye.
     * 
     * @param eps Specific internal energy.
     * @param csnd2 Square sound speed.
     * @param entropy Entropy per baryon.
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The specific internal energy.
     */
    double GRACE_HOST_DEVICE
    press_eps_csnd2_entropy__temp_rho_ye( double& eps, double& csnd2, double& entropy, double& temp 
                                  , double& rho, double& ye 
                                  , error_type& err ) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_eps_csnd2_entropy__temp_rho_ye_impl(eps,csnd2,entropy,temp,rho,ye,err);
    }
    /**
     * @brief Pressure, specific enthalpy, square sound speed, 
     *        temperature and specific internal energy given
     *        entropy, rho and ye.
     * 
     * @param h Specific enthalpy.
     * @param csnd2 Square sound speed.
     * @param temp Temperature.
     * @param eps Specific internal energy.
     * @param entropy Entropy per baryon.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure.
     */
    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_eps__entropy_rho_ye( double& h, double& csnd2, double& temp
                                          , double& eps, double& entropy, double& rho 
                                          , double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_h_csnd2_temp_eps__entropy_rho_ye_impl(h,csnd2,temp,eps,entropy,rho,ye,err) ; 
    }
    /**
     * @brief Specific internal energy, specific enthalpy,
     *        square sound speed, temperature and entropy 
     *        given pressure, rho and ye.
     * 
     * @param h Specific enthalpy.
     * @param csnd2 Square sound speed.
     * @param temp Temperature.
     * @param entropy Entropy per baryon.
     * @param press Pressure.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The specific internal energy.
     */
    double GRACE_HOST_DEVICE
    eps_h_csnd2_temp_entropy__press_rho_ye( double& h, double& csnd2, double& temp
                                          , double& entropy, double& press, double& rho 
                                          , double& ye, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps_h_csnd2_temp_entropy__press_rho_ye_impl(h,csnd2,temp,entropy,press,rho,ye,err) ; 
    }
    
    /**
     * @brief Get cold pressure given rho and ye.
     * 
     * @param rho Rest-mass density.
     * @param err Error code.
     * @return double The pressure at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    press_cold__rho(double& rho, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->press_cold__rho_impl(rho,err) ; 
    }
    /**
     * @brief Get cold energy density given press and ye.
     * 
     * @param press Pressure.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    rho__press_cold(double& press_cold, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->rho__press_cold_impl(press_cold,err) ; 
    }
    /**
     * @brief Get cold energy density given press and ye.
     * 
     * @param press Pressure.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    rho__energy_cold(double& e_cold, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->rho__energy_cold_impl(e_cold,err) ; 
    }
    /**
     * @brief Get cold energy density given press and ye.
     * 
     * @param press Pressure.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The pressure at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    energy_cold__press_cold(double& press_cold, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->energy_cold__press_cold_impl(press_cold,err) ; 
    }
    /**
     * @brief Cold specific internal energy given rho and ye.
     * 
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double The specific internal energy at \f$T=0\f$
     */
    double GRACE_HOST_DEVICE
    eps_cold__rho(double& rho, error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->eps_cold__rho_impl(rho,err) ; 
    }
    /**
     * @brief Electron fraction on cold table given rest mass dens
     * 
     * @param rho Rest-mass density.
     * @param err Error code.
     * @return double The electron fraction
     */
    double GRACE_HOST_DEVICE
    ye_cold__rho(double& rho,  error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->ye_cold__rho_impl(rho,err) ; 
    }
    /**
     * @brief Electron fraction on cold table given pressure
     * 
     * @param press Cold pressure.
     * @param err Error code.
     * @return double The electron fraction
     */
    double GRACE_HOST_DEVICE
    ye_cold__press(double& press,  error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->ye_cold__press_impl(press,err) ; 
    }
    /**
     * @brief Temperature of cold slice given rest mass dens
     * 
     * @param rho Rest-mass density.
     * @param err Error code.
     * @return double The temperature 
     */
    double GRACE_HOST_DEVICE
    temp_cold__rho(double& rho,  error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->temp_cold__rho_impl(rho,err) ;  
    }
    /**
     * @brief Temperature of cold slice given rest mass dens
     * 
     * @param rho Rest-mass density.
     * @param err Error code.
     * @return double The temperature 
     */
    double GRACE_HOST_DEVICE
    entropy_cold__rho(double& rho,  error_type& err) const 
    {
        return static_cast<eos_impl_t const*>(this)->entropy_cold__rho_impl(rho,err) ;  
    }
    /**
     * @brief Electron, proton, neutron chemical potentials, 
     *        alpha particles, hydrogen, neutron and proton fractions,
     *        average mass number and average charge number given  
     *        temperature, rho and ye.
     * 
     * @param mup Proton chemical potential.
     * @param mun Neutron chemical potential.
     * @param Xa \f$\alpha\f$ particle fraction.
     * @param Xh H fraction.
     * @param Xn Neutron fraction.
     * @param Xp Proton fraction.
     * @param Abar Average mass number.
     * @param Zbar Average proton number.
     * @param temp Temperature.
     * @param rho Rest-mass density.
     * @param ye Electron fraction.
     * @param err Error code.
     * @return double Electron chemical potential.
     */
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
    /**
     * @brief Get maximum allowed epsilon, might differ from table bound.
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    get_c2p_ye_max() const { return eos_yemax ; }
    /**
     * @brief Get maximum allowed epsilon, might differ from table bound.
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    get_c2p_ye_min() const { return eos_yemin ; }
    /**
     * @brief Get maximum allowed epsilon, might differ from table bound.
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    get_c2p_eps_max() const { return c2p_eps_max ; } 
    /**
     * @brief Get the atmosphere electron fraction.
     * 
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    temp_atmosphere() const { return c2p_temp_atm ; }
    /**
     * @brief Get the atmosphere electron fraction.
     * 
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    ye_atmosphere() const { return c2p_ye_atm ; }
    /**
     * @brief Get the minimum of the enthalpy. Only needed for c2p
     * 
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    enthalpy_minimum() const { return c2p_h_min ; }
     /**
     * @brief Get the EOS density minimum. Only needed for c2p
     * 
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    density_minimum() const { return eos_rhomin ; }
     /**
     * @brief Get the EOS density maximum. Only needed for c2p
     * 
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    density_maximum() const { return eos_rhomax ; }
 protected:
    //! Does this EOS depend on ye?
    static constexpr bool has_ye = eos_impl_t::has_ye ; 
    //! Maximum and minimum rest-mass densities.
    double eos_rhomax, eos_rhomin ;
    //! Maximum and minimum temperatures.
    double eos_tempmax, eos_tempmin ;
    //! Maximum and minimum electron fractions.
    double eos_yemax, eos_yemin ; 
    //! Baryon mass.
    double baryon_mass ;
    //! Minimum specific internal energy.
    double c2p_eps_min   ; 
    //! Maximum specific internal energy.
    double c2p_eps_max   ; 
    //! Minimum specific enthalpy
    double c2p_h_min     ; 
    //! Maximum specific enthalpy
    double c2p_h_max     ; 
    //! Atmosphere temperature 
    double c2p_temp_atm    ;
    //! Atmosphere ye 
    double c2p_ye_atm    ;
    //! Is the atmosphere beta-equilibrated?
    bool atm_is_beta_eq    ; 
    //! Is the table extended at high rho? 
    bool extend_table_high ; 
} ;

}

#endif /* GRACE_PHYSICS_EOS_BASE_HH */
