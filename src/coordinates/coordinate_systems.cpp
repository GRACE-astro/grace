/**
 * @file coordinate_systems.cpp
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

#include <thunder/amr/thunder_amr.hh> 
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/errors/error.hh> 

#include <array> 

namespace thunder { 

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
get_physical_coordinates(
      int const itree 
    , std::array<double,THUNDER_NSPACEDIM> const& logical_coords) {
        return coordinate_system::get().get_physical_coordinates(itree,logical_coords) ; 
}

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
get_physical_coordinates(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , size_t nq 
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates
    , bool include_gzs)
{
    return coordinate_system::get().get_physical_coordinates(
        ijk, nq, cell_coordinates, include_gzs
    ) ; 
} 

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
get_logical_coordinates(
      int itree
    , std::array<double,THUNDER_NSPACEDIM> const& physical_coordinates) 
{
    return coordinate_system::get().get_logical_coordinates(
        itree, physical_coordinates
    ) ; 
}

}