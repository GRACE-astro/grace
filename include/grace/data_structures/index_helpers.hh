/**
 * @file index_helpers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-08-20
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#ifndef GRACE_DATA_STRUCTURES_INDEX_HELPERS_HH
#define GRACE_DATA_STRUCTURES_INDEX_HELPERS_HH

#include <grace_config.h>

namespace grace {

enum vec_index_types_t {
    UP = 0,
    DOWN = 1 
} ;

enum two_tens_index_types_t {
    UPUP = 0,
    UPDOWN = 1,
    DOWNDOWN = 2
} ; 

}

#endif 