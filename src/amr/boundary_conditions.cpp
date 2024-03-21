/**
 * @file boundary_conditions.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-21
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

#include <Kokkos_Core.hpp>

#include <thunder/amr/thunder_amr.hh>
#include <thunder/system/thunder_system.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/data_structures/macros.hh>
#include <thunder/data_structures/memory_defaults.hh>
#include <thunder/data_structures/variable_indices.hh>
#include <thunder/data_structures/variables.hh>
#include <thunder/config/config_parser.hh>

#include <thunder/amr/bc_helpers.hh>

namespace thunder { namespace amr {

void apply_boundary_conditions() {
    using namespace thunder ;

}

}} /* namespace thunder::amr */