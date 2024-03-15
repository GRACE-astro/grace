/**
 * @file amr_functions.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief This file contains free functions that are used throughtout the 
 *        code to access amr related data or trigger amr related actions.
 * @version 0.1
 * @date 2024-02-29
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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

 #ifndef AMR_FUNCTIONS_HH
 #define AMR_FUNCTIONS_HH 

#include <thunder_config.h>

#include <thunder/utils/inline.h> 
#include <thunder/utils/device.h> 

#include <thunder/data_structures/macros.hh>
#include <thunder/amr/quadrant.hh>
#include <thunder/amr/tree.hh>
#include <thunder/amr/forest.hh> 

#include <thunder/config/config_parser.hh>

#include <array>
#include <tuple>
#include <cstdlib>

namespace thunder { namespace amr {

/**
 * @brief Get the number of grid cells per quadrant 
 *        in each direction. 
 * 
 * @return a tuple containing the number of grid cells per quadrant 
 *         in each direction.
 */
decltype(auto) 
get_quadrant_extents()
{
    auto& config = thunder::config_parser::get() ; 
    auto const nx = config["amr"]["npoints_block_x"].as<size_t>() ; 
    auto const ny = config["amr"]["npoints_block_y"].as<size_t>() ; 
    auto const nz = config["amr"]["npoints_block_x"].as<size_t>() ; 
    return std::make_tuple(nx,ny,nz) ;  
}

/**
 * @brief Get the number of ghost cells. 
 * 
 * @return number of ghost cells. 
 */
int 
get_n_ghosts()
{
    auto& config = thunder::config_parser::get() ; 
    return config["amr"]["n_ghostzones"].as<int>() ; 
}

} } /* thunder::amr */ 

 #endif 