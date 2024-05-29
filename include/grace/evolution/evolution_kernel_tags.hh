/**
 * @file evolution_kernel_tags.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-13
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef GRACE_EVOLUTION_EVOLUTION_KERNEL_TAGS_HH 
#define GRACE_EVOLUTION_EVOLUTION_KERNEL_TAGS_HH

namespace grace {

struct x_flux_computation_kernel_t {};

struct y_flux_computation_kernel_t {};

struct z_flux_computation_kernel_t {};


struct sources_computation_kernel_t {}; 

struct auxiliaries_computation_kernel_t {} ;

struct eigenspeed_kernel_t {} ; 

}

#endif /* GRACE_EVOLUTION_EVOLUTION_KERNEL_TAGS_HH */