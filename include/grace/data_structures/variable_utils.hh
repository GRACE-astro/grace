/**
 * @file variable_utils.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-22
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

#ifndef GRACE_DATA_STRUCTURES_VARIABLE_UTILS_HH
#define GRACE_DATA_STRUCTURES_VARIABLE_UTILS_HH

#include <string> 
#include <cstdlib> 
#include <grace/data_structures/variable_indices.hh>

namespace grace { namespace variables {

int 
get_n_evolved() ; 

int 
get_n_hrsc() ; 

int 
get_n_auxiliary() ; 

std::string
get_bc_type( int64_t var_idx, size_t const& var_type = grace::variables::EVOLVED)  ;

std::string
get_var_name(int64_t var_idx, bool is_aux)   ; 

std::vector<std::size_t>
get_vector_state_variables_indices() ; 

std::vector<std::size_t>
get_tensor_state_variables_indices() ; 

std::vector<std::size_t>
get_vector_aux_variables_indices() ; 

std::vector<std::size_t>
get_tensor_aux_variables_indices() ; 


} } /* namespace grace::variables */

#endif /* GRACE_DATA_STRUCTURES_VARIABLE_UTILS_HH */