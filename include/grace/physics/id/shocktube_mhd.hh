/**
 * @file shocktube_mhd.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-04-17
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

#ifndef GRACE_PHYSICS_ID_SHOCKTUBE_MHD_HH
#define GRACE_PHYSICS_ID_SHOCKTUBE_MHD_HH

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
//**************************************************************************************************
/**
 * \defgroup initial_data Initial Data
 * 
 */
//**************************************************************************************************
/**
 * @brief Shocktube MHD initial data kernel
 * \ingroup initial_data
 * @tparam eos_t type of equation of state
 * @note this kernel has to be checked for and adjusted if needed 
 * should the magnetic field initialization method/location changes in the future (e.g. vec pot)
 * 
 */
template < typename eos_t >
struct shocktube_mhd_id_t {
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //!< Type of state vector
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Construct a new shocktube id kernel object
     * 
     * @param eos Equation of state
     * @param pcoords Physical coordinate array
     * @param rhoL Left density
     * @param rhoR Right density
     * @param pL Left pressure
     * @param pR Right pressure
     * @param vxL Left velocity in the x direction
     * @param vyL Left velocity in the y direction
     * @param vzL Left velocity in the z direction
     * @param vxR Right velocity in the x direction
     * @param vyR Right velocity in the y direction
     * @param vzR Right velocity in the z direction
     * @param bxL Left magnetic field in the x direction
     * @param byL Left magnetic field in the y direction
     * @param bzL Left magnetic field in the z direction
     * @param bxR Right magnetic field in the x direction
     * @param byR Right magnetic field in the y direction
     * @param bzR Right magnetic field in the z direction
     */
    shocktube_mhd_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rhoL, double rhoR 
        , double pL  , double pR
        , double vxL, double vyL, double vzL
        , double vxR, double vyR, double vzR
        , double bxL, double byL, double bzL
        , double bxR, double byR, double bzR
        )
        : _eos(eos)
        , _pcoords(pcoords)
        , _rhoL(rhoL), _rhoR(rhoR)
        , _pL(pL), _pR(pR)
        , _vxL(vxL), _vyL(vyL), _vzL(vzL)
        , _vxR(vxR), _vyR(vyR), _vzR(vzR)
        , _bxL(bxL), _byL(byL), _bzL(bzL)
        , _bxR(bxR), _byR(byR), _bzR(bzR)
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

        if( _pcoords(VEC(i,j,k),0,q) <= 0 ) {
            id.rho     = _rhoL ; 
            id.press   = _pL   ; 
            id.vx = _vxL ;
            id.vy = _vyL ;
            id.vz = _vzL ;
            id.bx = _bxL ;
            id.by = _byL ;
            id.bz = _bzL ; 
        } else {
            id.rho     = _rhoR ; 
            id.press   = _pR   ;
            id.vx = _vxR ;
            id.vy = _vyR ;
            id.vz = _vzR ;
            id.bx = _bxR ;
            id.by = _byR ;
            id.bz = _bzR ; 
        }

        // if vector potential is used, we set its values in the so-called symmetric gauge:
        // A_i = 0.5 * (B^j x^k - B^k x^j) where (i,j,k) are (0,1,2) and even permutations 
        id.ax = 0.5 * (id.by * z - id.bz * y); 
        id.ay = 0.5 * (id.bz * x - id.bx * z); 
        id.az = 0.5 * (id.bx * y - id.by * x); 
        // note that this^ is only possible for piece-wise constant magnetic field 
        id.phi_em = 0.0;

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
    double _rhoL, _rhoR, _pL, _pR;                    //!< Left and right states  
    double _vxL, _vyL,_vzL, _vxR, _vyR,_vzR ;          //!< Left and right state velocities  
    double _bxL, _byL,_bzL, _bxR, _byR,_bzR ;          //!< Left and right state magnetic field   
    
    //**************************************************************************************************
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
//**************************************************************************************************
#endif /* GRACE_PHYSICS_ID_SHOCKTUBE_MHD_HH */