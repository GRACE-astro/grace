/**
 * @file evolve.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-13
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

#ifndef THUNDER_EVOLVE_HH
#define THUNDER_EVOLVE_HH

#include <thunder_config.h>

#include <thunder/data_structures/variable_properties.hh>

namespace thunder {

void evolve() ; 

void advance_substep( double const t, double const dt, double const dtfact 
                    , thunder::var_array_t<THUNDER_NSPACEDIM>& state 
                    , thunder::var_array_t<THUNDER_NSPACEDIM>& state_p 
                    , thunder::var_array_t<THUNDER_NSPACEDIM>& aux 
                    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>& cvol
                    , thunder::staggered_coordinate_arrays_t& surfs_and_edges ) ; 

}

#endif /* THUNDER_EVOLVE_HH */