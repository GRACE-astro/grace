/**
 * @file orszag_tang_vortex.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-05-0-5
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

#ifndef GRACE_PHYSICS_ID_ORSZAG_TANG_VORTEX_MHD_HH
#define GRACE_PHYSICS_ID_ORSZAG_TANG_VORTEX_MHD_HH

#include <grace_config.h>

#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

//**************************************************************************************************
namespace grace {
//**************************************************************************************************
/**
 * \defgroup initial_data Initial Data
 * 
 */
//**************************************************************************************************
/**
 * @brief Orszag-Tang Vortex MHD test - a classical 2D testbed for inspecting the divB violations - MHD initial data kernel
*
 * \ingroup initial_data
 * @tparam eos_t type of equation of state
 * @note this kernel has to be checked for and adjusted if needed 
 * should the magnetic field initialization method/location changes in the future (e.g. vec pot)
 * @note only in 2D/3D problems can divergence cleaning performance of the GLM method's be judged 
 *       in 1D (and flat spacetime), the evolution of phi_glm is trivial
 */
template < typename eos_t >
struct orszag_tang_vortex_mhd_id_t {
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //!< Type of state vector
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Construct a new orszag_tang_vortex_mhd id kernel object
     * 
     * @param eos Equation of state
     * @param pcoords Physical coordinate array
     * @param rho density
     * @param press pressure
     */
     orszag_tang_vortex_mhd_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rho
        , double press
        )
        : _eos(eos)
        , _pcoords(pcoords)
        , _rho(rho), _press(press)
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
   
        
        id.rho = _rho;
        id.press = _press; 
        id.vx = -0.99 * Kokkos::sin(y);
        id.vy =  0.99 * Kokkos::sin(x);
        id.vz = 0.;
        id.bx = -Kokkos::sin(y);
        id.by =  Kokkos::sin(2.*x);
        id.bz = 0.; 
        
    
        // set the Minkowski metric 
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

        unsigned int err ; 
        id.ye  = _eos.ye_beta_eq__press_cold(id.press, err);
        return std::move(id) ; 
    }
    //**************************************************************************************************

    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rho, _press;                    //!< Left and right states  
    
    //**************************************************************************************************
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
//**************************************************************************************************
#endif /* GRACE_PHYSICS_ID_ORSZAG_TANG_VORTEX_MHD_HH */