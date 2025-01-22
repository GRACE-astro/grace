/**
 * @file minkowski.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-01-21
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

#ifndef GRACE_PHYSICS_ID_MINKOWSKI_HH
#define GRACE_PHYSICS_ID_MINKOWSKI_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>

//**************************************************************************************************
namespace grace {
//**************************************************************************************************
//**************************************************************************************************
/**
 * @brief Initial data for trivial BSSN checks
 * \ingroup initial_data
 * @tparam eos_t Type of equation of state
 */
//**************************************************************************************************
template < typename eos_t >
struct minkowski_id_t {
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //! Type of state array
    //**************************************************************************************************
    /**
     * @brief Construct a new minkowski_id kernel
     * 
     * @param eos Equation of state
     * @param pcoords physical coordinates
     */
    minkowski_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords)
        : _eos(eos)
        , _pcoords(pcoords)
    {} 
    //**************************************************************************************************
    //**************************************************************************************************
    /**
     * @brief Return minkowski initial data at a point
     * 
     * @param i x cell index
     * @param j y cell index
     * @param k z cell index
     * @param q quadrant index
     * 
     * @return grmhd_id_t Initial data objects for minkowski ID test
     */
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
        
        id.gxx = 1. ;
        id.gyy = 1. ;
        id.gzz = 1. ; 
        id.gxy = id.gxz = id.gyz = 0.0 ; 

        id.alp = 1. ; 
        
        id.kxx = id.kxy = id.kxz = id.kyy = id.kyz = id.kzz = 0 ;


        return std::move(id) ; 
    }
    //**************************************************************************************************

    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    //**************************************************************************************************  
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
//**************************************************************************************************
#endif /* GRACE_PHYSICS_ID_MINKOWSKI_HH */