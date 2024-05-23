/**
 * @file reductors.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-22
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
#ifndef THUNDER_UTILS_REDUCTORS_HH
#define THUNDER_UTILS_REDUCTORS_HH
#include <thunder_config.h>
#include <thunder/utils/math.hh>
#include <thunder/utils/inline.hh>
#include <thunder/utils/device.h>
#include <thunder/data_structures/variable_properties.hh>

#include <Kokkos_Core.hpp>

namespace utils {

struct thunder_min_reduction_t      {} ; 
struct thunder_max_reduction_t      {} ; 
struct thunder_integral_reduction_t {} ;  
struct thunder_norm2_reduction_t    {} ;
struct thunder_sum_reduction_t      {} ; 




}

#endif /* THUNDER_UTILS_REDUCTORS_HH */