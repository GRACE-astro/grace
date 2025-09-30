/**
 * @file tabulated_eos.hh
 * @author Khalil Pierre (khalil3.14erre@gmail.com"
 * @brief 
 * @date 2025-02-03
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

#ifndef GRACE_PHYSICS_EOS_TABULATED_EOS_HH
#define GRACE_PHYSICS_EOS_TABULATED_EOS_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/utils/interpolators.hh> 
#include <grace/utils/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>
#include <bitset>

namespace grace { 
/**
 * @brief 
 *        
 * \ingroup eos
 * 
 */
class tabulated_eos_t : public eos_base_t<tabulated_eos_t>
{

  using error_type = unsigned int; 

 public:

    enum EV {
      PRESS = 0,
      EPS,
      S,
      CS2,
      MUE,
      MUP,
      MUN,
      XA,
      XH,
      XN,
      XP,
      ABAR,
      ZBAR,
      NUM_VARS
    };

    //For a consistent dimension odering
    enum dim {
      rho = 0,
      temp,
      yes,
      num_dim
    };

    tabulated_eos_t() = default;

    tabulated_eos_t(
      Kokkos::View<double****, grace::default_space> alltables,
      Kokkos::View<double*, grace::default_space> logrho,
      Kokkos::View<double*, grace::default_space> logtemp,
      Kokkos::View<double*, grace::default_space> yes,
      Kokkos::View<double***, grace::default_space> epstable, //TODO!! This does not seem to be needed
      Kokkos::View<double [dim::num_dim], grace::default_space> coord_spacing,
      Kokkos::View<double [dim::num_dim], grace::default_space> inverse_coord_spacing,
      double c2p_ye_atm,
      double c2p_rho_atm,
      double c2p_temp_atm,
      double c2p_eps_atm,
      double c2p_eps_min,
      double c2p_eps_max,
      double c2p_h_min,
      double c2p_h_max,
      double c2p_press_max, //TODO!! This does not seem to be needed
      double eos_rhomax,
      double eos_rhomin,
      double eos_tempmax,
      double eos_tempmin,
      double eos_yemax,
      double eos_yemin,
      double baryon_mass,
      double energy_shift,
      bool atm_is_beta_eq,
      bool extend_table_high)
    : eos_base_t<tabulated_eos_t>{energy_shift, eos_rhomax, eos_rhomin
                                , eos_tempmax, eos_tempmin, eos_yemax
                                , eos_yemin, baryon_mass
                                , c2p_ye_atm, c2p_rho_atm, c2p_temp_atm
                                , c2p_eps_atm, c2p_eps_min, c2p_eps_max
                                , c2p_h_min, c2p_h_max, atm_is_beta_eq
                                , extend_table_high}

    , _alltables(alltables), _logrho(logrho), _logtemp(logtemp), _yes(yes)
    , _epstable(epstable), _coord_spacing(coord_spacing)
    , _inverse_coord_spacing(inverse_coord_spacing), _eos_rhomax(eos_rhomax)
    , _eos_rhomin(eos_rhomin), _eos_tempmax(eos_tempmax), _eos_tempmin(eos_tempmin)
    , _eos_yemax(eos_yemax), _eos_yemin(eos_yemin), _energy_shift(energy_shift)
    {GRACE_INFO( "Table extents {}, {}, {}, {}", alltables.extent(0) , alltables.extent(1) , alltables.extent(2) , alltables.extent(3) ) ;}


  private:

    Kokkos::View<double****, grace::default_space> _alltables ; 
    Kokkos::View<double***, grace::default_space> _epstable;
    Kokkos::View<double*, grace::default_space> _logrho, _logtemp, _yes ;



    //Get nearest table index of input value xin
    // int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    // find_index_uniform(const Kokkos::View<double*>& x, const double& idx, const double& xin) const {
    //   //Shift xin by lowest value
    //   const double xL = xin - x(0) ;

    //   //Kokkos::printf("xL = %f \n", xL) ;

    //   //If xin is smaller then lowest value 0 index is returned
    //   if (xL <= 0) return 0 ;

    //   int index = static_cast<int>(xL * idx) ;

    //   //Kokkos::printf("index = %d \n", index) ;

    //   //Returns second to last index if calculated index is out of bounds
    //   if (index > x.extent(0) - 2) return x.extent(0) - 2 ;

    //   //Kokkos::printf("Index %d \n", index) ;

    //   return index ;
    // }

    //GPU friendly implementation of find_index_uniform
    int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    find_index_uniform(const Kokkos::View<double*>& x, const double& idx, const double& xin) const {
        // Get table boundaries
        const double x_min = x(0);
        const double x_max = x(x.extent(0) - 1);
        const double max_index = static_cast<double>(x.extent(0) - 2);
        
        // FIRST: Clamp input to table bounds to prevent extreme values
        const double clamped_xin = min(max(xin, x_min), x_max);
        
        // SECOND: Calculate safe fractional index
        const double fractional_index = (clamped_xin - x_min) * idx;
        
        // THIRD: Clamp to valid index range
        const double clamped_index = min(max(fractional_index, 0.0), max_index);
        
        return static_cast<int>(clamped_index);
    }
    

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate(const double* x, const double* y, const double &xrho, const double &xtemp, const double &xye) const {

      double const c00 =  utils::detail::linterp1d(x[3*0+0],x[3*1+0], y[0],y[1], xrho) ; 
      double const c10 =  utils::detail::linterp1d(x[3*2+0],x[3*3+0], y[2],y[3], xrho) ; 
      double const c0  =  utils::detail::linterp1d(x[3*0+1],x[3*2+1], c00 , c10, xtemp) ; 
      double const c01 =  utils::detail::linterp1d(x[3*4+0],x[3*5+0], y[4],y[5], xrho) ; 
      double const c11 =  utils::detail::linterp1d(x[3*6+0],x[3*7+0], y[6],y[7], xrho) ; 
      double const c1  =  utils::detail::linterp1d(x[3*4+1],x[3*6+1], c01 , c11, xtemp) ;
      
      
      return utils::detail::linterp1d(x[3*0+2], x[3*4+2], c0, c1, xye) ;  

    }

    //Interpolates one of the 3D tables, the table integer specifices which table, and xrho/temp/ye the interpolation position
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate_table(const int& table, const double &xrho, const double &xtemp, const double &xye) const {
      int irho, itemp, iye;
      
      //Find 3D tables corresponding to int table
      auto table_3D = Kokkos::subview(_alltables, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), table);

      // Kokkos::printf("logrho = %d, logtemp = %d, ye = %d \n", _logrho.extent(0), _logtemp.extent(0), _yes.extent(0)) ;
      // Kokkos::printf("lrho0 = %f, ltemp0 = %f, ye0 = %f \n", _logrho(0), _logtemp(0), _yes(0)) ;


      //Calculate index position of rho, temp, ye 
      irho = find_index_uniform(_logrho, _inverse_coord_spacing(dim::rho), xrho);
      itemp = find_index_uniform(_logtemp, _inverse_coord_spacing(dim::temp), xtemp);
      iye = find_index_uniform(_yes, _inverse_coord_spacing(dim::yes), xye);

      //Kokkos::printf("irho = %d, itemp = %d, iye = %d \n", irho, itemp, iye) ;

      //Need to create stencil for interploation, these are the dimensions
      size_t constexpr stencil_size = 2UL; 
      size_t constexpr npoints = stencil_size * stencil_size * stencil_size;

      //Create arrays for stencil
      double table_coordinates[3 * npoints];
      double table_values[npoints];

      //Parametric coordinates are used to calulate stencil indicies and values
      int parametric_coordinates[3 * npoints];

      //Loop to calculate parametric variables
      for ( int is=0; is<8; ++is){
        
        parametric_coordinates[3*is + 0UL] = (is%2) ; 
        parametric_coordinates[3*is + 1UL] = (is/2)%2 ; 
        parametric_coordinates[3*is + 2UL] = (is/2)/2 ;
      
      }

      //Flatened array is used to store all stencil indicies
      for ( int is=0; is < npoints; ++is){
        //Convert indices to physical coordinates
        int i_idx = irho + parametric_coordinates[3*is + 0];
        int j_idx = itemp + parametric_coordinates[3*is + 1];
        int k_idx = iye + parametric_coordinates[3*is + 2];

        //calulates coordinates for stencil
        table_coordinates[3*is + 0] = _logrho(i_idx);
        table_coordinates[3*is + 1] = _logtemp(j_idx);
        table_coordinates[3*is + 2] = _yes(k_idx);

        table_values[is] = table_3D(i_idx, j_idx, k_idx);
                                
      }
    
      return interpolate(table_coordinates, table_values, xrho, xtemp, xye);
    }

    //TODO!! Need to talk to Carlo to see what errors make sense to abort simulation on
    //Might need to split into warning and error message feature
    error_type GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    checkbounds(double &xrho, double &xtemp, double &xye, bool check_temp = true) const {
    
    error_type error_code;

    if (xrho > _eos_rhomax && !extend_table_high) {
      // This happens only inside the AH.
      // (assuming that the table has a sane range)
      error_code = EOS_ERROR_T::EOS_RHO_TOO_HIGH;
      xrho = _eos_rhomax;

    } else if  (xrho < _eos_rhomin) {
      // Might happen in the atmosphere and will be caught later. This point is
      // most likely also incorrect and should be reset.
      xrho = _eos_rhomin;
      error_code = EOS_ERROR_T::EOS_RHO_TOO_LOW;
    }

    if (xye > _eos_yemax) {
      // This is a serious issue and should not happen.
      // Maybe we should even abort the simulation here
      //TODO! Talk to Carlo about if error message should be returned here
      xye = _eos_yemax;
      error_code = EOS_ERROR_T::EOS_YE_TOO_HIGH;
    
    } else if (xye < _eos_yemin) {
      // Ok, this can be fixed
      xye = _eos_yemin;
      error_code = EOS_ERROR_T::EOS_YE_TOO_LOW;
    }
    
    if (check_temp) {
      if (xtemp > _eos_tempmax) {
        // This should never happen...
        // But removing energy should be fine, so what about
        xtemp = _eos_tempmax;
        error_code = EOS_ERROR_T::TEMP_TOO_HIGH;
      
      } else if (xtemp < _eos_tempmin) {
        // Might happen in the atmosphere, let's reset
        xtemp = _eos_tempmin;
        error_code = EOS_ERROR_T::TEMP_TOO_LOW;
      }
    }
    return error_code;

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    find_logtemp_from_eps(const double &xrho, double &eps, const double &xye) const {

      auto const leps = log(eps + _energy_shift);

      //Find eps range
      auto const epsmin = interpolate_table(EV::EPS, xrho, _logtemp(0), xye);

      auto const epsmax = interpolate_table(EV::EPS, xrho, _logtemp(_logtemp.size() - 1), xye);

      if (leps <= epsmin) {
        eps = exp(epsmin) - _energy_shift;
        return _logtemp(0);
      }
      if (leps >= epsmax) {
        eps = exp(epsmax) - _energy_shift;
        return _logtemp(_logtemp.size() - 1);
      }

      
      //TODO! Ask Carlo if a Kokkos Lambda would be more appropriate here
      auto const func = [&] (double &lt) {
        auto const vars = interpolate_table(EV::EPS, xrho, lt, xye);
        return leps - vars;
      } ;

      return utils::brent(func, _logtemp(0), _logtemp(_logtemp.size() - 1), 1.e-14); 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    find_logtemp_from_entropy(const double &xrho, double &entropy, const double &xye) const {
      
      const double min_entropy = interpolate_table(EV::S, xrho, _logtemp(0), xye);
      const double max_entropy = interpolate_table(EV::S, xrho, _logtemp(_logtemp.size() - 1), xye);

      if (entropy <= min_entropy){
        entropy = min_entropy;
        return _logtemp(0);
      }
      if (entropy >= max_entropy){
        entropy = max_entropy;
        return _logtemp(_logtemp.size() - 1);
      }

      auto const func = [&] (double &lt) {
        auto const vars = interpolate_table(EV::S, xrho, lt, xye);
        return entropy - vars;
      };

      return utils::brent(func, _logtemp(0), _logtemp(_logtemp.size() - 1), 1.e-14);

    }

    
  public:

    Kokkos::View<double [dim::num_dim], grace::default_space> _coord_spacing;
    Kokkos::View<double [dim::num_dim], grace::default_space> _inverse_coord_spacing;
    double _eos_rhomax, _eos_rhomin, _eos_tempmax, _eos_tempmin;
    double _eos_yemax, _eos_yemin, _energy_shift;

    //---------------------------------------For testing---------------------------------------//
    //-----------------------------------------------------------------------------------------//
    int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    _find_index_uniform(const Kokkos::View<double*>& _x, const double& _idx, double _xin) const {
    
      return find_index_uniform(_x, _idx, _xin);

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    _interpolate_table(const int& _table, double &_xrho, double &_xtemp, double &_xye) const {

      return interpolate_table(_table, _xrho, _xtemp, _xye);
    }

    Kokkos::View<double*, grace::default_space> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_logrho() const {
     
      return _logrho;
    } 

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_logrho(int x) const {
      
      return _logrho(x);
    } 

    Kokkos::View<double*, grace::default_space> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_logtemp() const {
      
      return _logtemp;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_logtemp(int x) const {
      
      return _logtemp(x);
    }

    Kokkos::View<double*, grace::default_space> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_yes() const {
      
      return _yes;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_yes(int x) const {
      
      return _yes(x);
    }

    auto GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_table(int table_id) const {

      return Kokkos::subview(_alltables, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), table_id);

    }

    //-----------------------------------------------------------------------------------------//
    
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press__eps_rho_ye_impl(double &eps, double &rho, double &ye, error_type &err) const {

      double temp_tmp = 0;
      // Check if rho and Y_e lie inside the table, otherwise abort!
      err = checkbounds(rho, temp_tmp, ye, false);

      const double lrho = log(rho);
      
      const double ltemp = find_logtemp_from_eps(lrho, eps, ye);

      const double vars = interpolate_table(EV::PRESS, lrho, ltemp, ye);

      return exp(vars);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_temp__eps_rho_ye_impl(double &temp, double &eps, double &rho, double &ye, error_type &err) const {
      // Check if rho and Y_e lie inside the table, otherwise abort!
      err = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = find_logtemp_from_eps(lrho, eps, ye);

      temp = exp(ltemp);

      const double vars = interpolate_table(EV::PRESS, lrho, ltemp, ye);

      return exp(vars);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press__temp_rho_ye_impl(double &temp, double &rho, double &ye, error_type &err) const {
      err = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = log(temp);

      const double vars = interpolate_table(EV::PRESS, lrho, ltemp, ye);

      return exp(vars);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps__temp_rho_ye_impl(double &temp, double &rho, double &ye, error_type &err) const {
      err = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = log(temp);

      const double vars = interpolate_table(EV::EPS, lrho, ltemp, ye);

      return exp(vars) - _energy_shift;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_cold__rho_ye_impl(double &rho, double &ye, error_type &err) const {
      double temp_tmp = 0;
      err = checkbounds(rho, temp_tmp, ye, false);
      
      const double lrho = log(rho);

      //Note this is a little different then margherita implementation
      double const vars = interpolate_table(EV::PRESS, lrho, _logtemp(0), ye);

      return exp(vars);

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_cold__rho_ye_impl(double &rho, double &ye, error_type &err) const {
      
      double temp_tmp = 0;
      err = checkbounds(rho, temp_tmp, ye, false);

      const double lrho = log(rho);

      const double vars = interpolate_table(EV::EPS, lrho, _logtemp(0), ye);

      return exp(vars) - _energy_shift;
    }

    //TODO!! this function seems unnecessary see if it is needed
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    temp_cold__rho_ye_impl(const double &rho, const double &ye, error_type &err) const {

      return exp(_logtemp(0));
    }

    //TODO! Ask about this function
    //This function can be called for a hybrid EOS but not tabulated 
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps__press_temp_rho_ye_impl(const double &press, double &temp, double &rho, double &ye, error_type &err) const {

      // ERROR("This routine should not be used. There is no monotonicity condition "
      //       "to enforce a succesfull inversion from eps(press). So you better "
      //       "rewrite your code to not require this call...");

      return 0;
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_range__rho_ye_impl(double& eps_min, double& eps_max, double &rho, double &ye, error_type &err) const {

      double temp_tmp = 0;
      err = checkbounds(rho, temp_tmp, ye, false);

      const double lrho = log(rho);

      eps_min = interpolate_table(EV::EPS, lrho, _logtemp(0), ye) - _energy_shift;
      eps_max = interpolate_table(EV::EPS, lrho, _logtemp(_logtemp.size() - 1), ye) - _energy_shift;
    
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    entropy_range__rho_ye_impl(double& s_min, double& s_max, double &rho, double &ye, error_type &err) const {

      double temp_tmp = 0;
      err = checkbounds(rho, temp_tmp, ye, false);

      const double lrho = log(rho);

      s_min = interpolate_table(EV::S, lrho, _logtemp(0), ye);
      s_max = interpolate_table(EV::S, lrho, _logtemp(_logtemp.size() - 1), ye);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2__eps_rho_ye_impl(double &h, double &csnd2, double &eps, double &rho, double &ye, error_type &err) const {

      double temp_tmp = 0;
      err = checkbounds(rho, temp_tmp, ye, false);

      const double lrho = log(rho);
      const double ltemp = find_logtemp_from_eps(lrho, eps, ye);

      //Calculate and assign specific enthalpy
      const double vars_press = interpolate_table(EV::PRESS, lrho, ltemp, ye);

      const double press = exp(vars_press);

      h = 1. + eps + press / rho;
      
      //Assign value for sound speed
      csnd2 = interpolate_table(EV::CS2, lrho, ltemp, ye);

      return press;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2__temp_rho_ye_impl(double &h, double &csnd2, double &temp, double &rho, double &ye, error_type &err) const {

      err = checkbounds(rho, temp, ye);

      const double ltemp = log(temp);
      const double lrho = log(rho);

      const double vars_press = interpolate_table(EV::PRESS, lrho, ltemp, ye);
      const double vars_eps = interpolate_table(EV::EPS, lrho, ltemp, ye);

      const double press = exp(vars_press);
      const double eps = exp(vars_eps) - _energy_shift;

      h = 1. + eps + press / rho;

      csnd2 = interpolate_table(EV::CS2, lrho, ltemp, ye);

      return press;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_h_csnd2__press_rho_ye_impl(double &h, double &csnd2, double &press, double &rho, double &ye, error_type &err) const {
      
      // ERROR("This routine should not be used. There is no monotonicity condition "
      //       "to enforce a succesfull inversion from eps(press). So you better "
      //       "rewrite your code to not require this call...");

    return 0;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye_impl(double &eps, double &csnd2, double &temp, double &rho, double &ye , error_type &err) const {
    
      err = checkbounds(rho, temp, ye);

      const double ltemp = Kokkos::log(temp);
      const double lrho = Kokkos::log(rho);

      const double vars_press = interpolate_table(EV::PRESS, lrho, ltemp, ye);
      const double vars_eps = interpolate_table(EV::EPS, lrho, ltemp, ye);

      const double press = exp(vars_press);
      
      eps = exp(vars_eps) - _energy_shift;

      csnd2 = interpolate_table(EV::CS2, lrho, ltemp, ye);

      return press;

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye_impl( double& h, double& csnd2, double& temp, double& entropy, double& eps , double& rho
                                          , double& ye , error_type &err ) const {

      err = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = find_logtemp_from_eps(lrho, eps, ye);

      const double vars_press = interpolate_table(EV::PRESS, lrho, ltemp, ye);

      const double press = exp(vars_press);
      
      temp = exp(ltemp);

      h = 1. + eps + press / rho;

      entropy = interpolate_table(EV::S, lrho, ltemp, ye);
      csnd2 = interpolate_table(EV::CS2, lrho, ltemp, ye);

      return press;
    
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_csnd2_entropy__temp_rho_ye_impl( double& csnd2, double& entropy, double& temp, double& rho, double& ye, error_type &err ) const {

      err = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = log(temp);

      //TODO! This line and line bellow included in the margherita implementation but seems redundent
      //const double vars_press = interpolate_table(EV::PRESS, lrho, ltemp, ye);
      
      const double vars_eps = interpolate_table(EV::EPS, lrho, ltemp, ye);

      const double eps = exp(vars_eps) - _energy_shift;

      //TODO! This is included in the margherita implementation but seems redundent
      //const double h = 1. eps + exp(vars_press) / rho;

      csnd2 = interpolate_table(EV::CS2, lrho, ltemp, ye);
      entropy = interpolate_table(EV::S, lrho, ltemp, ye);

      return eps;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2_temp_eps__entropy_rho_ye_impl ( double& h, double& csnd2, double& temp, double& eps, double& entropy
                                                , double& rho, double& ye, error_type &err) const {
    
      err = checkbounds(rho, temp, ye);

      const double lrho = log(rho); 
      const double ltemp = find_logtemp_from_entropy(lrho, entropy, ye);

      const double vars_press = interpolate_table(EV::PRESS, lrho, ltemp, ye);
      const double vars_eps = interpolate_table(EV::EPS, lrho, ltemp, ye);

      const double press = exp(vars_press);
      eps = exp(vars_eps) - _energy_shift;
      temp = exp(ltemp);

      h = 1. + eps + press / rho;

      csnd2 = interpolate_table(EV::CS2, lrho, ltemp, ye);

      return press;

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_h_csnd2_temp_entropy__press_rho_ye_impl( double& h, double& csnd2, double&temp, double& entropy, double& press, double& rho
                                          , double& ye, error_type& err) const {
      
      // ERROR("This routine should not be used. There is no monotonicity condition "
      //         "to enforce a succesfull inversion from eps(press). So you better "
      //         "rewrite your code to not require this call...");

      return 0;
                                    
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_eps_ye__beta_eq__rho_temp_impl(double &eps, double &ye, double &rho, double &temp, error_type& err) const{
      
      err = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = log(temp);

      // Beta equilibrium requires that \mu_n =  \mu_p +\mu_e -\mu_nu
      // Since we assume that \mu_nu should be negligible we demand
      // \mu_n-\mu_p-\mu_e = \mu_hat =0

      auto const func = [&](double &ye) {
        const double vars_mue = interpolate_table(EV::MUE, lrho, ltemp, ye);
        const double vars_mup = interpolate_table(EV::MUP, lrho, ltemp, ye);
        const double vars_mun = interpolate_table(EV::MUN, lrho, ltemp, ye);

        return vars_mue + vars_mup - vars_mun;
      };

      ye = utils::brent(func, _yes(0), _yes(_yes.size() - 1), 1.e-14);

      const double vars_press = interpolate_table(EV::PRESS, lrho, ltemp, ye);
      const double vars_eps = interpolate_table(EV::EPS, lrho, ltemp, ye);

      eps = exp(vars_eps) - _energy_shift;

      return exp(vars_press);

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_impl( 
        double &mup, double &mun, double &Xa, double &Xh, double &Xn, double &Xp
      , double &Abar, double &Zbar, double &temp, double &rho, double &ye
      , error_type &err) const {

      err = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = log(temp);

      const double MUE = interpolate_table(EV::MUE, lrho, ltemp, ye);
      mup = interpolate_table(EV::MUP, lrho, ltemp, ye);
      mun = interpolate_table(EV::MUN, lrho, ltemp, ye);
      Xa = interpolate_table(EV::XA, lrho, ltemp, ye);
      Xh = interpolate_table(EV::XH, lrho, ltemp, ye);
      Xn = interpolate_table(EV::XN, lrho, ltemp, ye);
      Xp = interpolate_table(EV::XP, lrho, ltemp, ye);
      Abar = interpolate_table(EV::ABAR, lrho, ltemp, ye);
      Zbar = interpolate_table(EV::ZBAR, lrho, ltemp, ye);

      return MUE;

      }


} ;


} 

#endif /* GRACE_PHYSICS_EOS_TABULATED_EOS_HH */