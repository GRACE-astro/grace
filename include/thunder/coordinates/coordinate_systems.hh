/**
 * @file coordinate_systems.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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

#ifndef THUNDER_AMR_COORDINATES_SYSTEMS_HH 
#define THUNDER_AMR_COORDINATES_SYSTEMS_HH

#include <thunder_config.h>

#include <thunder/coordinates/cartesian_coordinate_systems.hh>
#include <thunder/coordinates/spherical_coordinate_systems.hh>

namespace thunder { 

#ifdef THUNDER_CARTESIAN_COORDINATES 
using coordinate_system = utils::singleton_holder<cartesian_coordinate_system_impl_t> ; 
using device_coordinate_system = cartesian_device_coordinate_system_impl_t ; 
#elif defined(THUNDER_SPHERICAL_COORDINATES)
using coordinate_system = utils::singleton_holder<spherical_coordinate_system_impl_t> ; 
using device_coordinate_system = spherical_device_coordinate_system_impl_t ; 
#endif 
/*****************************************************************************************/
/*                          Global coordinate utilities                                  */
/*****************************************************************************************/
/**
 * @brief Get physical coordinates of a point inside a local cell, given 
 *        the quadrant and spatial indices.
 * 
 * @param itree Tree index 
 * @param lcoords Logical coordinates \f$[0,1]^{d}\f$ of the points inside the tree
 * 
 */
std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
get_physical_coordinates(
      int const itree 
    , std::array<double,THUNDER_NSPACEDIM> const& logical_coords
) ;

/**
 * @brief Get physical coordinates of a point inside a local cell, given 
 *        the quadrant and spatial indices.
 * 
 * @param ijk Spatial indices of cell inside the quadrant
 * @param nq  Local quadrant index  
 * @param cell_coordinates Logical coordinates \f$[0,1]^{d}\f$ of the points inside the cell
 * @param include_gzs Include ghostzones in the calculation
 * 
 * NB: Including ghostzones means that \f$ijk={0,0(,0)}\f$ has negative logical coordinates,
 *     and the logical coordinates origin lies at \f$ijk={ngz,ngz(,ngz)}\f$.
 */
std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
get_physical_coordinates(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , size_t nq 
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates={VEC(0.5,0.5,0.5)}
    , bool include_gzs = false
) ; 

/**
 * @brief Get tree-logical coordinates of a point given tree index and physical coordinates.
 * 
 * @param ijk Spatial indices of cell inside the quadrant
 * @param nq  Local quadrant index  
 * @param local_coords Logical coordinates \f$[0,1]^{d}\f$ of the points inside the cell
 * @param include_gzs Include ghostzones in the calculation
 * 
 * NB: Including ghostzones means that \f$ijk={0,0(,0)}\f$ has negative logical coordinates,
 *     and the logical coordinates origin lies at \f$ijk={ngz,ngz(,ngz)}\f$.
 */
std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
get_logical_coordinates(
      int itree
    , std::array<double,THUNDER_NSPACEDIM> const& physical_coordinates      
) ; 

} /* namespace thunder::amr */

#endif /* THUNDER_AMR_COORDINATES_SYSTEMS_HH */