/**
 * @file linear_mhd_waves.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-04-29
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

#ifndef GRACE_PHYSICS_ID_LINEAR_MHD_WAVES_HH
#define GRACE_PHYSICS_ID_LINEAR_MHD_WAVES_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
// #include <grace/utils/device/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

//**************************************************************************************************
namespace grace {

namespace linear_mhd_utils{

    enum WAVE_TYPE{
        CONTACT, 
        SLOW_MAGNETOSONIC,
        ALFVEN,
        FAST_MAGNETOSONIC,
        N_WAVE_TYPES
    };

    enum WAVE_DIRECTION{
        RIGHT, 
        LEFT,
        STANDING
    };
}



//**************************************************************************************************
/**
 * \defgroup initial_data Initial Data
 * 
 */
//**************************************************************************************************
/**
 * @brief Linear MHD waves initial data kernel
 * \ingroup initial_data
 * @tparam eos_t type of equation of state
 * @note this kernel has to be adjused if magnetic field initialization method and storage location changes in the future (e.g. vec pot);
 * We base the setup on: 
 * https://ui.adsabs.harvard.edu/abs/2008JCoPh.227.4123G/abstract
 * and 
 * https://robertcaddy.com/posts/mhd-test-problems/
 * @warning This MHD ID kernel will have to be adapted in the future if vector potential is used instead 
 * 
 */
template < typename eos_t >
struct linear_mhd_wave_id_t {

    using WAVE_TYPE = grace::linear_mhd_utils::WAVE_TYPE;
    using WAVE_DIRECTION = grace::linear_mhd_utils::WAVE_DIRECTION;
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //!< Type of state vector
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Construct a new shocktube id kernel object
     * 
     * @param eos Equation of state
     * @param pcoords Physical coordinate array
     * @param wave_type Type of the MHD wave
     * @param ampl amplitude of the wave
     * @param wavelength wavelength of the wave
     * @param wave_movement direction of the wave (left, right or standing)
    
     */
     linear_mhd_wave_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , const WAVE_TYPE wave_type
        , const double ampl
        , const double wavelength
        , const WAVE_DIRECTION wave_movement
        )
        : _eos(eos)
        , _pcoords(pcoords)
        , _wave_type(wave_type)
        , _ampl(ampl)
        , _wavelength(wavelength)
        , _wave_movement(wave_movement)
    {} 
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Obtain initial data at a point
     * 
     * @param i x cell index 
     * @param j y cell index 
     * @param k z cell index
     * @param q Quadrant index
     * @return grmhd_id_t Initial data at this point
     */
    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {

        grmhd_id_t id ; 

        const double x = _pcoords(VEC(i,j,k),0,q);
        const double y = _pcoords(VEC(i,j,k),1,q);
        const double z = _pcoords(VEC(i,j,k),2,q);
   
        // set Minkowski metric 
        id.betax = 0; id.betay=0; id.betaz = 0; 
        id.alp = 1 ; 
        id.gxx = 1; id.gyy = 1; id.gzz = 1;
        id.gxy = 0; id.gxz = 0; id.gyz = 0 ;
        id.kxx = 0; id.kyy = 0; id.kzz = 0 ;
        id.kxy = 0; id.kxz =0 ; id.kyz = 0 ; 

        // set the Lagrange multiplier to zero
        #ifdef GRACE_ENABLE_B_FIELD_GLM
        id.phi_glm = 0.;
        #endif 

        auto gamma = _eos.get_gammas();
        // printf("linear mhd wave: gamma: %f\n", gamma(0));
        // auto gamma = _eos.get_eos().get_gammas();
        // if(gamma.size()!=1)  GRACE_INFO("Linear MHD waves can only be run with a single-piece polytrope!");


        // initialize the constant state
        // chosen in such a way that the individual waves can manifest clearly as perturbations of the background
        // and wave speeds (v_C, v_A, v_SM, v_FM) are well-separated 
        id.rho   = 1.0;
        id.press = 1/gamma(0);
        id.vx = 0.0 ;
        id.vy = 0.0 ;
        id.vz = 0.0 ;
        id.bx = 1.0 ;  // as usual, the kernel needs to be adapted if vector potential is used instead 
        id.by = 3./2. ; 
        id.bz = 0.0 ; 

        
        double epsilon = 1.0 ;
        if(_wave_movement == WAVE_DIRECTION::LEFT) epsilon = -1.0;

        // set the particular wave type
        if(_wave_type == WAVE_TYPE::FAST_MAGNETOSONIC){
            double const prefactor = 1.0 / 2.0 / Kokkos::sqrt(5.); 
            id.rho   += prefactor * 2.;
            id.press += prefactor * 4 * epsilon;
            id.vx    +=-prefactor * 2 * epsilon ;
            id.vy    += 0.0 ;
            id.vz    += 0.0;
            id.bx    += prefactor*4.0;
            id.by    += 0.0;
            id.bz    += prefactor*9.0;
        }
        else if(_wave_type == WAVE_TYPE::ALFVEN){
            id.rho   += 0.;
            id.press += 0 ;
            id.vx    += 0 ;
            id.vy    +=-epsilon * 1.0 ;
            id.vz    += 0.0;
            id.bx    += 0.0;
            id.by    += 1.0;
            id.bz    += 0.0;
        }
        else if(_wave_type == WAVE_TYPE::SLOW_MAGNETOSONIC){
            double const prefactor = 1.0 / 2.0 / Kokkos::sqrt(5.); 
            id.rho   += prefactor * 4.;
            id.press += prefactor * 2 * epsilon;
            id.vx    += prefactor * 4 * epsilon ;
            id.vy    += 0.0 ;
            id.vz    += 0.0;
            id.bx    += prefactor*(-2.0);
            id.by    += 0.0;
            id.bz    += prefactor*3.0;
        }
        else if(_wave_type ==  WAVE_TYPE::CONTACT){
            id.vx = 1.0; // overwriting for the contact wave
            double const prefactor = 1.0 / 2.0 ; 
            id.rho   += prefactor * 2.;
            id.press += prefactor * 2;
            id.vx    += 0.0;
            id.vy    += 0.0;
            id.vz    += 0.0;
            id.bx    += 0.0;
            id.by    += 0.0;
            id.bz    += 1.0;
        }
        else {
            // some communication?
        }
       

        unsigned int err ; 
        id.ye  = _eos.ye_beta_eq__press_cold(id.press, err);
        return std::move(id) ; 
    }
    //**************************************************************************************************

    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    WAVE_TYPE _wave_type ;                    
    WAVE_DIRECTION _wave_movement ;                    
    double _wavelength ;
    double _ampl ; 

    //**************************************************************************************************
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
//**************************************************************************************************
#endif 