/**
 * @file coordinate_systems.cpp
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

#include <grace/amr/grace_amr.hh> 
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/errors/error.hh> 

#include <array> 

namespace grace { 

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
get_physical_coordinates(
      int const itree 
    , std::array<double,GRACE_NSPACEDIM> const& logical_coords) {
        return coordinate_system::get().get_physical_coordinates(itree,logical_coords) ; 
}

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
get_physical_coordinates(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
    , size_t nq 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates
    , bool include_gzs)
{
    return coordinate_system::get().get_physical_coordinates(
        ijk, nq, cell_coordinates, include_gzs
    ) ; 
} 

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
get_logical_coordinates(
      int itree
    , std::array<double,GRACE_NSPACEDIM> const& physical_coordinates) 
{
    return coordinate_system::get().get_logical_coordinates(
        itree, physical_coordinates
    ) ; 
}

}