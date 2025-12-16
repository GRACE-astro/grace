/**
 * @file cloud.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-12-8
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

#ifndef GRACE_PHYSICS_CLOUD_ID_HH
#define GRACE_PHYSICS_CLOUD_ID_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

namespace grace {


template < typename eos_t >
struct cloud_id_t {
    using state_t = grace::var_array_t ; 
    
    cloud_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rho0, double T0, double r0, double p)
        : _eos(eos)
        , _pcoords(pcoords)
        , _rho0(rho0)
        , _T0(T0)
        , _r0(r0)
        , _p(p)
    {} 

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 
        double r = sqrt(
            SQR(_pcoords(i,j,k,0,q)) +
            SQR(_pcoords(i,j,k,1,q)) +
            SQR(_pcoords(i,j,k,2,q)) 
        ) ; 
        id.rho = _rho0 ; 

        if ( r <= _r0 ) {
            id.rho = _rho0 ; 
        } else {
            id.rho =  _rho0 * pow(r/_r0, _p) ;
        }

        if ( r<=_r0/2) {
            id.press = _T0 * id.rho ; 
        } else {
            id.press = _T0 * pow(r/(0.5*_r0), _p) * id.rho ; 
        }
        id.bx = id.by = id.bz = 0.; 
        id.vx = 0; id.vy = 0.; id.vz = 0.;
        id.betax = 0; id.betay=0; id.betaz = 0; 
        id.alp = 1 ; 
        id.gxx = 1; id.gyy = 1; id.gzz = 1;
        id.gxy = 0; id.gxz = 0; id.gyz = 0 ;
        id.kxx = 0; id.kyy = 0; id.kzz = 0 ;
        id.kxy = 0; id.kxz =0 ; id.kyz = 0 ; 
        unsigned int err ; 
        id.ye  = _eos.ye_beta_eq__press_cold(id.press, err);
        return std::move(id) ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rho0, _T0, _r0, _p ; 
} ; 


}

#endif /* GRACE_PHYSICS_CLOUD_ID_HH */