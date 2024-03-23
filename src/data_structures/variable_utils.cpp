/**
 * @file variable_utils.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-23
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

#include <thunder/data_structures/variable_indices.hh>
#include <thunder/data_structures/variable_properties.hh>
#include <thunder/data_structures/variable_utils.hh>
#include <thunder/errors/assert.hh>
namespace thunder { namespace variables  {

int 
get_n_evolved()
{
    return detail::num_evolved ;
} 

std::string 
get_bc_type(int64_t varidx)
{
    ASSERT_DBG( varidx < detail::_var_bc_types.size(), 
              "Requested variable " << varidx << " does not have registered BCs."); 
    return detail::_var_bc_types[varidx]  ; 
}

std::vector<int> 
get_vector_var_indices()
{
    std::vector<int> indices ; 
    int idx =0 ;
    for( auto const& props: detail::_varprops){
        if(props.second.is_vector)
            indices.push_back(idx) ; 
        idx++ ; 
    }
    return indices  ;
}

std::vector<int> 
get_tensor_var_indices() 
{
    std::vector<int> indices ; 
    int idx =0 ;
    for( auto const& props: detail::_varprops){
        if(props.second.is_tensor)
            indices.push_back(idx) ; 
        idx++ ; 
    }
    return indices  ;
}

}}