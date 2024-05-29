/**
 * @file reductors.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-22
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
#ifndef GRACE_UTILS_REDUCTORS_HH
#define GRACE_UTILS_REDUCTORS_HH
#include <grace_config.h>
#include <grace/utils/math.hh>
#include <grace/utils/inline.hh>
#include <grace/utils/device.h>
#include <grace/data_structures/variable_properties.hh>

#include <Kokkos_Core.hpp>

namespace utils {

struct grace_min_reduction_t      {} ; 
struct grace_max_reduction_t      {} ; 
struct grace_integral_reduction_t {} ;  
struct grace_norm2_reduction_t    {} ;
struct grace_sum_reduction_t      {} ; 




}

#endif /* GRACE_UTILS_REDUCTORS_HH */