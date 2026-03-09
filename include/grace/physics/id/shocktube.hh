/**
 * @file shocktube.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-07-22
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

#ifndef GRACE_PHYSICS_ID_SHOCKTUBE_HH
#define GRACE_PHYSICS_ID_SHOCKTUBE_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

namespace grace {


template < typename eos_t >
struct shocktube_id_t {
    using state_t = grace::var_array_t ; 
    
    shocktube_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rhoL, double rhoR 
        , double pL  , double pR 
        , double BxL, double BxR
        , double ByL, double ByR
        , double BzL, double BzR )
        : _eos(eos)
        , _pcoords(pcoords), _rhoL(rhoL)
        , _rhoR(rhoR), _pL(pL), _pR(pR)
        , _BxL(BxL), _BxR(BxR)
        , _ByL(ByL), _ByR(ByR)
        , _BzL(BzL), _BzR(BzR)
    {} 

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE GRACE_DEVICE_EXTERNAL_LINKAGE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 
        if( _pcoords(VEC(i,j,k),0,q) <= 0 ) {
            id.rho     = _rhoL ; 
            id.press   = _pL   ; 
            id.bx = _BxL ;
            id.by = _ByL ;
            id.bz = _BzL ;
        } else {
            id.rho     = _rhoR ; 
            id.press   = _pR   ;
            id.bx = _BxR ;
            id.by = _ByR ;
            id.bz = _BzR ;
        }

        id.vx = 0; id.vy = 0.; id.vz = 0.;
        id.betax = 0; id.betay=0; id.betaz = 0; 
        id.alp = 1 ; 
        id.gxx = 1; id.gyy = 1; id.gzz = 1;
        id.gxy = 0; id.gxz = 0; id.gyz = 0 ;
        id.kxx = 0; id.kyy = 0; id.kzz = 0 ;
        id.kxy = 0; id.kxz =0 ; id.kyz = 0 ; 
        eos_err_t err ; 
        id.ye  = _eos.ye_cold__press(id.press, err);
        double h, csnd2;
        id.eps = _eos.eps_h_csnd2_temp_entropy__press_rho_ye(
            h, csnd2, id.temp, id.entropy, id.press, id.rho, id.ye, err
        ) ; 
        return std::move(id) ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rhoL, _rhoR, _pL, _pR, _BxL, _BxR, _ByL, _ByR, _BzL, _BzR;          //!< Left and right states  
} ; 


}

#endif 