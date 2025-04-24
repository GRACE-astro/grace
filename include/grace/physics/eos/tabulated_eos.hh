/**
 * @file piecewise_polytropic_eos.hh
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


#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/utils/interpolators.hh> //TODO! Should this be in the grace_utils header
#include <grace/utils/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/physics/eos/physical_constants.hh>
#include <hdf5.h>

#include <Kokkos_Core.hpp>
#include <bitset>

namespace grace { 
/**
 * @brief 
 *        
 * \ingroup eos
 * 
 */
class tabulated_eos_t
{
  //TODO! This is different from how Carlo handels errors for the polytopic_eos
  typedef std::bitset<EOS_ERROR_T::EOS_NUM_ERRORS> error_type_array;

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
      Kokkos::View<double***, grace::default_space> epstable,
      Kokkos::View<double [dim::num_dim], grace::default_space> coord_spacing,
      Kokkos::View<double [dim::num_dim], grace::default_space> inverse_coord_spacing,
      double c2p_eps_min,
      double c2p_h_min,
      double c2p_h_max,
      double c2p_press_max,
      double eos_rhomax,
      double eos_rhomin,
      double eos_tempmax,
      double eos_tempmin,
      double eos_yemax,
      double eos_yemin,
      double energy_shift)
    : _alltables(alltables), _logrho(logrho), _logtemp(logtemp), _yes(yes)
    , _epstable(epstable), _coord_spacing(coord_spacing)
    , _inverse_coord_spacing(inverse_coord_spacing), _c2p_eps_min(c2p_eps_min)
    , _c2p_h_min(c2p_h_min), _c2p_h_max(c2p_h_max), _c2p_press_max(c2p_press_max)
    , _eos_rhomax(eos_rhomax), _eos_rhomin(eos_rhomin), _eos_tempmax(eos_tempmax)
    , _eos_tempmin(eos_tempmin), _eos_yemax(eos_yemax), _eos_yemin(eos_yemin)
    , _energy_shift(energy_shift)
    {}

    bool extend_table_high = false;


  private:

    Kokkos::View<double****, grace::default_space> _alltables ; 
    Kokkos::View<double***, grace::default_space> _epstable;
    Kokkos::View<double*, grace::default_space> _logrho, _logtemp, _yes ;
    Kokkos::View<double [dim::num_dim], grace::default_space> _coord_spacing;
    Kokkos::View<double [dim::num_dim], grace::default_space> _inverse_coord_spacing;
    double _c2p_eps_min, _c2p_h_min, _c2p_h_max, _c2p_press_max, _eos_rhomax;
    double _eos_rhomin, _eos_tempmax, _eos_tempmin, _eos_yemax, _eos_yemin;
    double _energy_shift;


    //Get nearest table index of input value xin
    template <typename T>
    int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    find_index_uniform(const Kokkos::View<double*>& x, const double& idx, T xin) const {
      //As grid spacing is uniform, inverse grid spacing can be used to calulate index
      
      //Shift xin by lowest value
      const double xL = xin - x(0);

      //If xin is smaller then lowest value 0 index is returned
      if (xL <= 0) return 0;

      int index = static_cast<int>(xL * idx);

      //Returns second to last index if calculated index is out of bounds
      if (index > x.size() - 2) return x.size() - 2;

      return index;
    }

    //Interpolates one of the 3D tables, the table integer specifices which table, and xrho/temp/ye the interpolation position
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate_table(const int& table, const double &xrho, const double &xtemp, const double &xye) const {
      int irho, itemp, iye;
      
      //Find 3D tables corresponding to int table
      auto table_3D = Kokkos::subview(_alltables, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, table);

      //Calculate index position of rho, temp, ye 
      irho = find_index_uniform(_logrho, _inverse_coord_spacing(dim::rho), _coord_spacing(dim::rho));
      itemp = find_index_uniform(_logtemp, _inverse_coord_spacing(dim::temp), _coord_spacing(dim::temp));
      iye = find_index_uniform(_yes, _inverse_coord_spacing(dim::yes), _coord_spacing(dim::yes));

      //Need to create stencil for interploation, these are the dimensions
      size_t constexpr stencil = utils::linear_interp_t<3>::stencil_size; 
      size_t constexpr npoints = stencil * stencil * stencil;

      //Create arrays for stencil
      double table_coordinates[3 * npoints];
      double table_value[npoints];

      //Parametric coordinates are used to calulate stencil indicies and values
      int parametric_coordinates[3 * npoints];
      utils::linear_interp_t<3>::get_parametric_coordinates(parametric_coordinates); 

      //Flatened array is used to store all stencil indicies
      for ( int is=0; is < npoints; ++is){
        //calulates coordinates for stencil
        table_coordinates[3*is + 0] = irho + parametric_coordinates[3*is + 0];
        table_coordinates[3*is + 1] = itemp + parametric_coordinates[3*is + 1];
        table_coordinates[3*is + 2] = iye + parametric_coordinates[3*is + 2];
        
        //calculate values for stencil
        table_value[is] = table_3D(irho + parametric_coordinates[3*is + 0], 
                                   itemp + parametric_coordinates[3*is + 1],
                                   iye + parametric_coordinates[3*is + 2]);
                                
      }
      
      //TODO! Currently passing the stencil like this generates warnings
      //sets up interploator object with stencil
      auto interpolator = utils::linear_interp_t<3>(table_coordinates, table_value); 

      //Interpolates using input values xrho/temp/ye
      return interpolator.interpolate(xrho, xtemp, xye);
    }

    error_type_array GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    checkbounds(double &xrho, double &xtemp, double &xye, bool check_temp = true) const {
    
    error_type_array error_codes;

    if (xrho > _eos_rhomax && !extend_table_high) {
      // This happens only inside the AH.
      // (assuming that the table has a sane range)
      error_codes[EOS_ERROR_T::EOS_RHO_TOO_HIGH] = true;
      xrho = _eos_rhomax;

    } else if  (xrho < _eos_rhomin) {
      // Might happen in the atmosphere and will be caught later. This point is
      // most likely also incorrect and should be reset.
      xrho = _eos_rhomin;
      error_codes[EOS_ERROR_T::EOS_RHO_TOO_LOW] = true;
    }

    if (xye > _eos_yemax) {
      // This is a serious issue and should not happen.
      // Maybe we should even abort the simulation here
      //TODO! Talk to Carlo about if error message should be returned here
      xye = _eos_yemax;
      error_codes[EOS_ERROR_T::EOS_YE_TOO_HIGH] = true;
    
    } else if (xye < _eos_yemin) {
      // Ok, this can be fixed
      xye = _eos_yemin;
      error_codes[EOS_ERROR_T::EOS_YE_TOO_LOW] = true;
    }
    
    if (check_temp) {
      if (xtemp > _eos_tempmax) {
        // This should never happen...
        // But removing energy should be fine, so what about
        xtemp = _eos_tempmax;
        error_codes[EOS_ERROR_T::TEMP_TOO_HIGH] = true;
      
      } else if (xtemp < _eos_tempmin) {
        // Might happen in the atmosphere, let's reset
        xtemp = _eos_tempmin;
        error_codes[EOS_ERROR_T::TEMP_TOO_LOW] = true;
      }
    }
    return error_codes;

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

      //TODO! Look into notes to remind myself why this is done
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
    return_logrho() const {
     
      return _logrho;
    } 

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    return_logrho(int x) const {
      
      return _logrho(x);
    } 

    Kokkos::View<double*, grace::default_space> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    return_logtemp() const {
      
      return _logtemp;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    return_logtemp(int x) const {
      
      return _logtemp(x);
    }

    Kokkos::View<double*, grace::default_space> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    return_yes() const {
      
      return _yes;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    return_yes(int x) const {
      
      return _yes(x);
    }

    //-----------------------------------------------------------------------------------------//
    
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press__eps_rho_ye(double &eps, double &rho, double &ye, error_type_array &error) const {

      double temp_tmp = 0;
      // Check if rho and Y_e lie inside the table, otherwise abort!
      error = checkbounds(rho, temp_tmp, ye, false);

      const double lrho = log(rho);
      
      const double ltemp = find_logtemp_from_eps(lrho, eps, ye);

      const double vars = interpolate_table(EV::PRESS, lrho, ltemp, ye);

      return exp(vars);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_temp__eps_rho_ye(double &temp, double &eps, double &rho, double &ye, error_type_array &error) const {
      // Check if rho and Y_e lie inside the table, otherwise abort!
      error = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = find_logtemp_from_eps(lrho, eps, ye);

      temp = exp(ltemp);

      const double vars = interpolate_table(EV::PRESS, lrho, ltemp, ye);

      return exp(vars);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press__temp_rho_ye(double &temp, double &rho, double &ye, error_type_array &error) const {
      error = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = log(temp);

      const double vars = interpolate_table(EV::PRESS, lrho, ltemp, ye);

      return exp(vars);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps__temp_rho_ye(double &temp, double &rho, double &ye, error_type_array &error) const {
      error = checkbounds(rho, temp, ye);

      const double lrho = log(rho);
      const double ltemp = log(temp);

      const double vars = interpolate_table(EV::EPS, lrho, ltemp, ye);

      return exp(vars) - _energy_shift;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_cold__rho_ye(double &rho, double &ye, error_type_array &error) const {
      double temp_tmp = 0;
      error = checkbounds(rho, temp_tmp, ye, false);
      
      const double lrho = log(rho);

      //Note this is a little different then margherita implementation
      double const vars = interpolate_table(EV::PRESS, lrho, _logtemp(0), ye);

      return exp(vars);

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_cold__rho_ye(double &rho, double &ye, error_type_array &error) const {
      
      double temp_tmp = 0;
      error = checkbounds(rho, temp_tmp, ye, false);

      const double lrho = log(rho);

      const double vars = interpolate_table(EV::EPS, lrho, _logtemp(0), ye);

      return exp(vars) - _energy_shift;
    }

    //TODO!! this function seems unnecessary see if it is needed
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    temp_cold__rho_ye(const double &rho, const double &ye, error_type_array &error) const {

      return exp(_logtemp(0));
    }

    //TODO! Ask about this function
    //This function can be called for a hybrid EOS but not tabulated 
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps__press_temp_rho_ye(const double &press, double &temp, double &rho, double &ye, error_type_array &error) const {

      ERROR("This routine should not be used. There is no monotonicity condition "
            "to enforce a succesfull inversion from eps(press). So you better "
            "rewrite your code to not require this call...");

      return 0;
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_range__rho_ye(double& eps_min, double& eps_max, double &rho, double &ye, error_type_array &error) const {

      double temp_tmp = 0;
      error = checkbounds(rho, temp_tmp, ye, false);

      const double lrho = log(rho);

      eps_min = interpolate_table(EV::EPS, lrho, _logtemp(0), ye) - _energy_shift;
      eps_max = interpolate_table(EV::EPS, lrho, _logtemp(_logtemp.size() - 1), ye) - _energy_shift;
    
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    entropy_range__rho_ye(double& s_min, double& s_max, double &rho, double &ye, error_type_array &error) const {

      double temp_tmp = 0;
      error = checkbounds(rho, temp_tmp, ye, false);

      const double lrho = log(rho);

      s_min = interpolate_table(EV::S, lrho, _logtemp(0), ye);
      s_max = interpolate_table(EV::S, lrho, _logtemp(_logtemp.size() - 1), ye);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2__eps_rho_ye(double &h, double &csnd2, double &eps, double &rho, double &ye, error_type_array &error) const {

      double temp_tmp = 0;
      error = checkbounds(rho, temp_tmp, ye, false);

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
    press_h_csnd2__temp_rho_ye(double &h, double &csnd2, double &temp, double &rho, double &ye, error_type_array &error) const {

      error = checkbounds(rho, temp, ye);

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
    eps_h_csnd2__press_rho_ye(double &h, double &csnd2, double &press, double &rho, double &ye, error_type_array &error) const {
      
      ERROR("This routine should not be used. There is no monotonicity condition "
            "to enforce a succesfull inversion from eps(press). So you better "
            "rewrite your code to not require this call...");

    return 0;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye(double &eps, double &csnd2, double &temp, double &rho, double &ye , error_type_array &error) const {
    
      error = checkbounds(rho, temp, ye);

      const double ltemp = log(temp);
      const double lrho = log(rho);
      
      const double vars_press = interpolate_table(EV::PRESS, lrho, ltemp, ye);
      const double vars_eps = interpolate_table(EV::EPS, lrho, ltemp, ye);

      const double press = exp(vars_press);
      
      eps = exp(vars_eps) - _energy_shift;

      csnd2 = interpolate_table(EV::CS2, lrho, ltemp, ye);

      return press;

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye( double& h, double& csnd2, double& temp, double& entropy, double& eps , double& rho
                                          , double& ye , error_type_array &error ) const {

      error = checkbounds(rho, temp, ye);

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
    eps_csnd2_entropy__temp_rho_ye( double& csnd2, double& entropy, double& temp, double& rho, double& ye, error_type_array &error ) const {

      error = checkbounds(rho, temp, ye);

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
    press_h_csnd2_temp_eps__entropy_rho_ye  ( double& h, double& csnd2, double& temp, double& eps, double& entropy
                                            , double& rho, double& ye, error_type_array &error) const {
    
      error = checkbounds(rho, temp, ye);

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
    eps_h_csnd2_temp_entropy__press_rho_ye( double& h, double& csnd2, double&temp, double& entropy, double& press, double& rho
                                          , double& ye, error_type_array& err) const {
      
      ERROR("This routine should not be used. There is no monotonicity condition "
              "to enforce a succesfull inversion from eps(press). So you better "
              "rewrite your code to not require this call...");

      return 0;
                                    
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_eps_ye__beta_eq__rho_temp(double &eps, double &ye, double &rho, double &temp, error_type_array& error) const{
      
      error = checkbounds(rho, temp, ye);

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
    mue_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye( 
        double &mup, double &mun, double &Xa, double &Xh, double &Xn, double &Xp
      , double &Abar, double &Zbar, double &temp, double &rho, double &ye
      , error_type_array &error) const {

      error = checkbounds(rho, temp, ye);

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


// Catch HDF5 errors
#define HDF5_ERROR(fn_call)                                          \
  do {                                                               \
    int _error_code = fn_call;                                       \
    if (_error_code < 0) {                                           \
      printf(    "HDF5 call '%s' returned error code %d", #fn_call,  \
                  _error_code);                                      \
    }                                                                \
  } while (0)

//Check to see if file is readable
static inline int file_is_readable(const char *filename) {
  FILE *fp = NULL;
  fp = fopen(filename, "r");
  if (fp != NULL) {
    fclose(fp);
    return 1;
  }
  return 0;
}
                               

//Following two functions are used to read in a lot of variables in the same way
//The first reads the meta date of a group in our case we are interested in the number of points
static inline void READ_ATTR_HDF5_COMPOSE(hid_t GROUP, const char *NAME, void * VAR, hid_t TYPE) {
  hid_t dataset;
  HDF5_ERROR(dataset = H5Aopen(GROUP, NAME, H5P_DEFAULT));            
  HDF5_ERROR(H5Aread(dataset, TYPE, VAR));                            
  HDF5_ERROR(H5Aclose(dataset));
}

//The second function reads in values from the HDF5 data set
//A memory buffer is passed into var for the data to be read into
static inline void READ_EOS_HDF5_COMPOSE(hid_t GROUP, const char *NAME, void * VAR, hid_t TYPE, hid_t MEM) {
  hid_t dataset;                                                      
  HDF5_ERROR(dataset = H5Dopen2(GROUP, NAME, H5P_DEFAULT));         
  HDF5_ERROR(H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR)); 
  HDF5_ERROR(H5Dclose(dataset));    
}


static tabulated_eos_t setup_tabulated_eos_compose(const char *nuceos_table_name) {
//static int setup_tabulated_eos_compose(const char *nuceos_table_name) {
  
  using namespace physical_constants;

  constexpr size_t NTABLES = tabulated_eos_t::EV::NUM_VARS;

  GRACE_INFO("*******************************");
  GRACE_INFO("Reading COMPOSE nuc_eos table file:");
  GRACE_INFO("{}", nuceos_table_name);
  GRACE_INFO("*******************************");

  hid_t file;

  if (!file_is_readable(nuceos_table_name)){
      ERROR("Could not read nuceos_table_name " << nuceos_table_name);
  }

  //HDF5 file is opened
  HDF5_ERROR(file = H5Fopen(nuceos_table_name, H5F_ACC_RDONLY, H5P_DEFAULT));
  
  hid_t parameters;

  //Parameter group is opened
  //From the parameter group the dimensions of the tables can be read 
  HDF5_ERROR(parameters = H5Gopen(file, "/Parameters", H5P_DEFAULT));

  //Table dimensions will be stored in these variables
  int nrho, ntemp, nye;

  // Read size of tables
  READ_ATTR_HDF5_COMPOSE(parameters,"pointsnb", &nrho, H5T_NATIVE_INT);
  READ_ATTR_HDF5_COMPOSE(parameters,"pointst", &ntemp, H5T_NATIVE_INT);
  READ_ATTR_HDF5_COMPOSE(parameters,"pointsyq", &nye, H5T_NATIVE_INT);

  //Will be exported at end of function, is not used within function scope
  std::array<size_t, 3> num_points = {size_t(nrho), size_t(ntemp), size_t(nye)};    

  // Allocate memory for tables
  double *logrho = new double[nrho];
  double *logtemp = new double[ntemp];
  double *yes = new double[nye];

  // Read values of denisty, tempreature and electron fraction respectivley
  READ_EOS_HDF5_COMPOSE(parameters,"nb", logrho, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_EOS_HDF5_COMPOSE(parameters,"t", logtemp, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_EOS_HDF5_COMPOSE(parameters,"yq", yes, H5T_NATIVE_DOUBLE, H5S_ALL);

  //Density, temperatur and electron fraction make up the basis of the grid
  //Now we load in the data that correspond to the values at each table point
  //We start with the thermal tables

  hid_t thermo_id;
  HDF5_ERROR(thermo_id = H5Gopen(file, "/Thermo_qty", H5P_DEFAULT));
  
  //We need to find the number of thermal tables in the HDF5 file
  int nthermo;
  READ_ATTR_HDF5_COMPOSE(thermo_id,"pointsqty", &nthermo, H5T_NATIVE_INT);

  // Read thermo index array
  int *thermo_index = new int[nthermo];
  READ_EOS_HDF5_COMPOSE(thermo_id,"index_thermo", thermo_index, H5T_NATIVE_INT, H5S_ALL);

  // Allocate memory and read table
  double *thermo_table = new double[nthermo * nrho * ntemp * nye];
  READ_EOS_HDF5_COMPOSE(thermo_id,"thermo", thermo_table, H5T_NATIVE_DOUBLE, H5S_ALL);

  // Now read compositions!

  // number of available particle information
  int ncomp = 0;
  hid_t comp_id;

  //Turns off some HDF5 error messages
  int status_e = H5Eset_auto(H5E_DEFAULT, NULL, NULL);
  int status_comp = H5Gget_objinfo(file, "/Composition_pairs", 0, nullptr);

  //
  if(status_comp == 0){
    HDF5_ERROR(comp_id = H5Gopen(file, "/Composition_pairs", H5P_DEFAULT));
    READ_ATTR_HDF5_COMPOSE(comp_id, "pointspairs", &ncomp, H5T_NATIVE_INT);
  }

  int *index_yi = nullptr;
  double *yi_table = nullptr;

  if(ncomp > 0){

    // index identifying particle type
    index_yi = new int[ncomp];
    READ_EOS_HDF5_COMPOSE(comp_id,"index_yi", index_yi, H5T_NATIVE_INT, H5S_ALL);

    // Read composition
    yi_table = new double[ncomp * nrho * ntemp * nye];
    READ_EOS_HDF5_COMPOSE(comp_id,"yi", yi_table, H5T_NATIVE_DOUBLE, H5S_ALL);
  }

  // Read average charge and mass numbers
  int nav=0;
  double *zav_table = nullptr;
  double *yav_table = nullptr;
  double *aav_table = nullptr;

  int status_av = H5Gget_objinfo(file, "Composition_quadruples", 0, nullptr);

  hid_t av_id;

  if(status_av ==0){
    HDF5_ERROR(av_id = H5Gopen(file, "/Composition_quadruples", H5P_DEFAULT));
    READ_ATTR_HDF5_COMPOSE(av_id, "pointsav", &nav, H5T_NATIVE_INT);
  }

  if(nav >0){
    //If nav is not equal to 1 the code will terminate 
    assert(nav == 1 &&
	   "nav != 1 in this table, so there is none or more than "
	   "one definition of an average nucleus."
	   "Please check and generalize accordingly.");

    // Read average tables
    zav_table = new double[nrho * ntemp * nye];
    yav_table = new double[nrho * ntemp * nye];
    aav_table = new double[nrho * ntemp * nye];
    READ_EOS_HDF5_COMPOSE(av_id, "zav", zav_table, H5T_NATIVE_DOUBLE, H5S_ALL);
    READ_EOS_HDF5_COMPOSE(av_id, "yav", yav_table, H5T_NATIVE_DOUBLE, H5S_ALL);
    READ_EOS_HDF5_COMPOSE(av_id, "aav", aav_table, H5T_NATIVE_DOUBLE, H5S_ALL);
  }

  HDF5_ERROR(H5Fclose(file));

  // Need to sort the thermo indices to match the tabulated_eos_t ordering

  //Compose associates table variables with specific numerical values
  constexpr size_t PRESS_C = 1;
  constexpr size_t S_C = 2;
  constexpr size_t MUN_C = 3;
  constexpr size_t MUP_C = 4;
  constexpr size_t MUE_C = 5;
  constexpr size_t EPS_C = 7;
  constexpr size_t CS2_C = 12;


  //Lambda function to go through the thermo_index array and 
  //finds array location of quiered index
  auto const find_index = [&](size_t const &index) {
    for (int i = 0; i < nthermo; ++i) {
      if (thermo_index[i] == index) return i;
    }
    assert(!"Could not find index of all required quantities. This should not "
            "happen.");
    return -1;
  };

  // IMPORTANT: The order here needs to match EV from tabulated_eos_t object!
  //Array here contains location of variables in the thermo_index array
  int thermo_index_conv[7]{find_index(PRESS_C), find_index(EPS_C),
                           find_index(S_C),     find_index(CS2_C),
                           find_index(MUE_C),   find_index(MUP_C),
                           find_index(MUN_C)};


  //Want to copy table data to the all table array with correct ordering 

  //Allocate memory for the all table array, good point to introduce kokkos views

  //Create Kokkos views to pass data too
  //TODO! What is the best odering for access patterns
  Kokkos::View<double****, grace::default_space> alltables("AllTables", nrho, ntemp, nye, NTABLES); 
  Kokkos::View<double *, grace::default_space> logrhoview("LogRhoView", nrho);
  Kokkos::View<double *, grace::default_space> logtempview("LogTempView", ntemp);
  Kokkos::View<double *, grace::default_space> yesview("yesView", nye);


  auto h_alltables = Kokkos::create_mirror_view(alltables); 
  auto h_logrhoview = Kokkos::create_mirror_view(logrhoview); 
  auto h_logtempview = Kokkos::create_mirror_view(logtempview); 
  auto h_yesview = Kokkos::create_mirror_view(yesview);

  //Allocate data to kokkos views and convert units/convert logs to natural log
  for (int i = 0; i < nrho; i++) h_logrhoview(i) = log(logrho[i] * baryon_mass * cm3_to_fm3 * densCGS_to_CU);
  for (int i = 0; i < ntemp; i++) h_logtempview(i) = log(logtemp[i]);
  for (int i = 0; i < nye; i++) h_yesview(i) = yes[i];

  //Every element of the thermal table is itterated through. The old
  //index is saved and the index required for the GRACE odering is calculated
  //Data is then transfered from the thermal tables to the all tables 
  for (int iv = tabulated_eos_t::EV::PRESS; iv <= tabulated_eos_t::EV::MUN; iv++)
    for (int k = 0; k < nye; k++)
      for (int j = 0; j < ntemp; j++)
        for (int i = 0; i < nrho; i++) {
          auto const iv_thermo = thermo_index_conv[iv];
          int indold = i + nrho * (j + ntemp * (k + nye * iv_thermo));
          h_alltables(i, j, k, iv) = thermo_table[indold];
        }

  //Lambda function to work out index_yi location of table identifier ID
  auto const find_index_yi = [&](size_t const &index) {
    for (int i = 0; i < ncomp; ++i) {
      if (index_yi[i] == index) return i;
    }
    assert(!"Could not find index of all required quantities. This should not "
            "happen.");
    return -1;
  };


  //A similar method as above is used to fix average compositions!
  for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
      for (int i = 0; i < nrho; i++) {
        int indold = i + nrho * (j + ntemp * k);
        int indnew = NTABLES * (i + nrho * (j + ntemp * k));

	      if(nav >0){
	        // ABAR
          h_alltables(i, j, k, tabulated_eos_t::EV::ABAR) = aav_table[indold];
	        // ZBAR
          h_alltables(i, j, k, tabulated_eos_t::EV::ZBAR) = zav_table[indold];
	        // Xh
          h_alltables(i, j, k, tabulated_eos_t::EV::XH) = aav_table[indold] * yav_table[indold];
	      }
	
        //Here the identifier ID is hard coded 
        if(ncomp>0){
	        // Xn
          h_alltables(i, j, k, tabulated_eos_t::EV::XN) =
            yi_table[indold + nrho * nye * ntemp * find_index_yi(10)];
	        // Xp
          h_alltables(i, j, k, tabulated_eos_t::EV::XP) =
            yi_table[indold + nrho * nye * ntemp * find_index_yi(11)];
	        // Xa
          h_alltables(i, j, k, tabulated_eos_t::EV::XA) = 
            4. * yi_table[indold + nrho * nye * ntemp * find_index_yi(4002)];
	      }
  }

  //Free memory
  delete[] thermo_index;
  delete[] thermo_table;
  delete[] logrho;
  delete[] logtemp;
  delete[] yes;

  if(index_yi != nullptr) delete[] index_yi;
  if(yi_table != nullptr) delete[] yi_table;

  if(zav_table != nullptr) delete[] zav_table;
  if(yav_table != nullptr) delete[] yav_table;
  if(aav_table != nullptr) delete[] aav_table;

  //Allocate memory for linear energy density table
  //linear scale is used to extrapolate negative energy densities
  //double *epstable;
  Kokkos::View<double *** , grace::default_space> epstable("linear_energy_table", nrho, ntemp, nye);
  auto h_epstable = Kokkos::create_mirror_view(epstable); 
  

  double c2p_eps_min = 1.e99;
  double c2p_h_min = 1.e99;
  double c2p_h_max = 0.;
  double c2p_press_max = 0.;

  double energy_shift = 0;


  //Get eps_min and convert units
  for (int k = 0; k < nye; k++)
    for (int j = 0; j < ntemp; j++)
      for (int i = 0; i < nrho; i++) {
        double pressL, epsL, rhoL;

        c2p_eps_min = math::min(c2p_eps_min, h_alltables(i, j, k, tabulated_eos_t::EV::EPS));

        
        { //pressure
        h_alltables(i, j, k, tabulated_eos_t::EV::PRESS) = 
          log(h_alltables(i, j, k, tabulated_eos_t::EV::PRESS) * MeV_to_erg * cm3_to_fm3 * pressCGS_to_CU);

        pressL = exp(h_alltables(i, j, k, tabulated_eos_t::EV::PRESS));
        c2p_press_max = math::max(c2p_press_max, pressL);
        }

        { //eps
        if (c2p_eps_min < 0) {
          energy_shift = -2.0 * c2p_eps_min;
          h_alltables(i, j, k, tabulated_eos_t::EV::EPS) += energy_shift;
        }
        
        h_epstable(i, j, k) = h_alltables(i, j, k, tabulated_eos_t::EV::EPS);
        h_alltables(i, j, k, tabulated_eos_t::EV::EPS) = log(h_alltables(i, j, k, tabulated_eos_t::EV::EPS));
        epsL = h_epstable(i, j, k) - energy_shift;
        }

        { //cs2
        if (h_alltables(i, j, k, tabulated_eos_t::EV::CS2) < 0) h_alltables(i, j, k, tabulated_eos_t::EV::CS2) = 0;
        h_alltables(i, j, k, tabulated_eos_t::EV::CS2) = 
          math::min(0.9999999, h_alltables(i, j, k, tabulated_eos_t::EV::CS2));
        }

        { //chemical potential
        auto const mu_q = h_alltables(i, j, k, tabulated_eos_t::EV::MUP);
        auto const mu_b = h_alltables(i, j, k, tabulated_eos_t::EV::MUN);
        
        h_alltables(i, j, k, tabulated_eos_t::EV::MUP) += mu_b;
        h_alltables(i, j, k, tabulated_eos_t::EV::MUE) -= mu_q;

        }

        rhoL = exp(h_logrhoview(i));
        const double hL = 1. + epsL + pressL / rhoL;
        c2p_h_min = math::min(c2p_h_min, hL);
        c2p_h_max = math::max(c2p_h_max, hL);

      }


  //Table bounds
  //!TODO Talk to Carlo about GPU accessability of variables 
  double eos_rhomax = exp(h_logrhoview(nrho - 1));
  double eos_rhomin = exp(h_logrhoview(0));

  double eos_tempmax = exp(h_logtempview[ntemp - 1]);
  double eos_tempmin = exp(h_logtempview[0]);

  double eos_yemax = h_yesview[nye - 1];
  double eos_yemin = h_yesview[0];

  //Calculate coordinate spacing

  Kokkos::View<double [tabulated_eos_t::dim::num_dim], grace::default_space> coord_spacing;
  Kokkos::View<double [tabulated_eos_t::dim::num_dim], grace::default_space> inverse_coord_spacing;

  auto h_coord_spacing = Kokkos::create_mirror_view(coord_spacing); 
  auto h_inverse_coord_spacing = Kokkos::create_mirror_view(inverse_coord_spacing);

  h_coord_spacing(tabulated_eos_t::dim::rho) = h_logrhoview(0) - h_logrhoview(1);
  h_coord_spacing(tabulated_eos_t::dim::temp) = h_logtempview(0) - h_logtempview(1);
  h_coord_spacing(tabulated_eos_t::dim::yes) = h_yesview(0) - h_yesview(1);

  h_inverse_coord_spacing(tabulated_eos_t::dim::rho) = 1. / h_coord_spacing(tabulated_eos_t::dim::rho);
  h_inverse_coord_spacing(tabulated_eos_t::dim::temp) = 1. / h_coord_spacing(tabulated_eos_t::dim::temp);
  h_inverse_coord_spacing(tabulated_eos_t::dim::yes) = 1. / h_coord_spacing(tabulated_eos_t::dim::yes);


  GRACE_INFO("*******************************");
  GRACE_INFO("Tabulated data read, transfering to GPU");
  GRACE_INFO("*******************************");

  //Copy data from host to device
  Kokkos::deep_copy(alltables, h_alltables);
  Kokkos::deep_copy(logrhoview, h_logrhoview);
  Kokkos::deep_copy(logtempview, h_logtempview); 
  Kokkos::deep_copy(yesview, h_yesview);
  Kokkos::deep_copy(epstable, h_epstable);
  Kokkos::deep_copy(coord_spacing, h_coord_spacing);
  Kokkos::deep_copy(inverse_coord_spacing, h_inverse_coord_spacing);


  return tabulated_eos_t{
      alltables 
    , logrhoview
    , logtempview
    , yesview
    , epstable
    , coord_spacing
    , inverse_coord_spacing
    , c2p_eps_min
    , c2p_h_min
    , c2p_h_max
    , c2p_press_max
    , eos_rhomax
    , eos_rhomin
    , eos_tempmax
    , eos_tempmin
    , eos_yemax
    , eos_yemin
    , energy_shift};
    

} 

} 