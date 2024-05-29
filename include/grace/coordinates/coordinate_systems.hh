/**
 * @file coordinate_systems.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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

#ifndef GRACE_AMR_COORDINATES_SYSTEMS_HH 
#define GRACE_AMR_COORDINATES_SYSTEMS_HH

#include <grace_config.h>

#include <grace/coordinates/cartesian_coordinate_systems.hh>
#include <grace/coordinates/spherical_coordinate_systems.hh>

namespace grace { 
/**
 * \defgroup grace_coordinates Coordinate utilities for GRACE
 */
/*****************************************************************************************/
/*****************************************************************************************/
/*                          Coordinate system                                            */
/*****************************************************************************************/
/*****************************************************************************************/
/**
 * @brief The coordinate system is a singleton that holds access to all coordinate
 *        related functions in grace
*/
#ifdef GRACE_CARTESIAN_COORDINATES 
/*****************************************************************************************/
using coordinate_system = utils::singleton_holder<cartesian_coordinate_system_impl_t> ; 
using device_coordinate_system = cartesian_device_coordinate_system_impl_t ; 
/*****************************************************************************************/
#elif defined(GRACE_SPHERICAL_COORDINATES)
/*****************************************************************************************/
using coordinate_system = utils::singleton_holder<spherical_coordinate_system_impl_t> ; 
using device_coordinate_system = spherical_device_coordinate_system_impl_t ; 
/*****************************************************************************************/
#endif 
/*****************************************************************************************/
/*****************************************************************************************/
/*                          Global coordinate utilities                                  */
/*****************************************************************************************/
/*****************************************************************************************/
/**
 * @brief Get physical coordinates of a point inside a local cell, given 
 *        the quadrant and spatial indices.
 * 
 * \ingroup grace_coordinates
 * 
 * @param itree Tree index 
 * @param lcoords Logical coordinates \f$[0,1]^{d}\f$ of the points inside the tree
 * 
 */
std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
get_physical_coordinates(
      int const itree 
    , std::array<double,GRACE_NSPACEDIM> const& logical_coords
) ;
/*****************************************************************************************/
/**
 * @brief Get physical coordinates of a point inside a local cell, given 
 *        the quadrant and spatial indices.
 * 
 * @param ijk Spatial indices of cell inside the quadrant
 * @param nq  Local quadrant index  
 * @param cell_coordinates Logical coordinates \f$[0,1]^{d}\f$ of the points inside the cell
 * @param include_gzs Include ghostzones in the calculation
 * 
 * \ingroup grace_coordinates
 * 
 * NB: Including ghostzones means that \f$ijk={0,0(,0)}\f$ has negative logical coordinates,
 *     and the logical coordinates origin lies at \f$ijk={ngz,ngz(,ngz)}\f$.
 */
std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
get_physical_coordinates(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
    , size_t nq 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates={VEC(0.5,0.5,0.5)}
    , bool include_gzs = false
) ; 
/*****************************************************************************************/
/**
 * @brief Get tree-logical coordinates of a point given tree index and physical coordinates.
 * 
 * @param ijk Spatial indices of cell inside the quadrant
 * @param nq  Local quadrant index  
 * @param local_coords Logical coordinates \f$[0,1]^{d}\f$ of the points inside the cell
 * @param include_gzs Include ghostzones in the calculation
 * 
 * \ingroup grace_coordinates
 * 
 * NB: Including ghostzones means that \f$ijk={0,0(,0)}\f$ has negative logical coordinates,
 *     and the logical coordinates origin lies at \f$ijk={ngz,ngz(,ngz)}\f$.
 */
std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
get_logical_coordinates(
      int itree
    , std::array<double,GRACE_NSPACEDIM> const& physical_coordinates      
) ; 
/*****************************************************************************************/
} /* namespace grace::amr */

#endif /* GRACE_AMR_COORDINATES_SYSTEMS_HH */