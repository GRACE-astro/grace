/**
 * @file blastwave.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-01-07
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
#ifndef GRACE_PHYSICS_ID_BLASTWAVE_HH
#define GRACE_PHYSICS_ID_BLASTWAVE_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

template < typename eos_t >
struct blastwave_id_t {
    using state_t = grace::var_array_t ; 

    blastwave_id_t(
        eos_t eos,
        grace::coord_array_t<GRACE_NSPACEDIM> pcoords
    ) : _eos(eos), _pcoords(pcoords)
    {    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 
        double rcyl = sqrt(
            SQR(_pcoords(VEC(i,j,k),0,q)) +
            SQR(_pcoords(VEC(i,j,k),1,q)) 
        ) ; 

        if (rcyl<0.5) {
            id.press = 2.5 ; 
        } else {
            id.press = 0.1 ; 
        }

        id.rho = 1.0 ; 

        id.vx = id.vy = id.vz = 0.0 ; 
        id.bx = id.by = id.bz = 0.0 ; 

        id.alp = 1 ; 
        id.gxx = 1; id.gyy = 1; id.gzz = 1;
        id.gxy = 0; id.gxz = 0; id.gyz = 0 ;
        id.kxx = 0; id.kyy = 0; id.kzz = 0 ;
        id.kxy = 0; id.kxz =0 ; id.kyz = 0 ; 

        unsigned int err ; 
        id.ye  = _eos.ye_cold__press(id.press, err);

        return id ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
} ; 

#endif /* GRACE_PHYSICS_ID_BLASTWAVE_HH */