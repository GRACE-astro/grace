/**
 * @file gauge_wave.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-11-25
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
#ifndef GRACE_PHYSICS_ID_GAUGE_WAVE_HH
#define GRACE_PHYSICS_ID_GAUGE_WAVE_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>

namespace grace {

template < typename eos_t >
struct gauge_wave_id_t {
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; 
    
    gauge_wave_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double A, double d)
        : _eos(eos)
        , _pcoords(pcoords), _rhoL(rhoL)
        , _A(A), _d(d)
    {} 

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        double const x = _pcoords(VEC(i,j,k),0,q);
        grmhd_id_t id ; 
        id.rho   = 0 ; 
        id.press = 0 ;
        id.ye    = 0 ;
        id.vx = id.vy = id.vz = 0;

        id.betax = id.betay = id.betaz = 0 ; 
        
        double const H0 = 1. + _A * Kokkos::sin(2*M_PI*x/_d) ; 

        id.gxx = H0 ;
        id.gyy = 1. ;
        id.gzz = 1. ; 
        id.gxy = id.gxz = id.gyz = 0 ; 

        id.alp = Kokkos::sqrt(H0)

        id.kxy = id.kxz = id.kyz = id.kyy = id.kzz = 0 ;
        id.kxx = - _A * M_PI * Kokkos::cos(2*M_PI*x/_d) / ( id.alp * _d );  
        return std::move(id) ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _A, _d ;                                   //!< Left and right states  
} ; 


}

#endif 