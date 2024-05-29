/**
 * @file variable_utils.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-23
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

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/errors/assert.hh>
#include <grace/errors/error.hh> 

namespace grace { namespace variables  {

int 
get_n_evolved()
{
    return detail::num_evolved ;
} 

int 
get_n_hrsc()
{
    return detail::num_fluxes ;
} 

int 
get_n_auxiliary()
{
    return detail::num_auxiliary ;
} 


std::string 
get_bc_type(int64_t varidx, size_t const& var_type)
{
    if ( var_type == EVOLVED ) {
        ASSERT_DBG( varidx < detail::_var_bc_types.size(), 
                "Requested variable " << varidx << " does not have registered BCs."); 
        return detail::_var_bc_types[varidx]  ;
    } else if ( var_type == AUXILIARY ) {
        ASSERT_DBG( varidx < detail::_aux_bc_types.size(), 
                "Requested auxiliary variable " << varidx << " does not have registered BCs."); 
        return detail::_aux_bc_types[varidx]  ;
    } else if ( var_type == FACE_STAGGERED ) {
        ASSERT_DBG( varidx < detail::_face_vars_bc_types.size(), 
                "Requested face-staggered variable " << varidx << " does not have registered BCs."); 
        return detail::_face_vars_bc_types[varidx]  ;
    } else if ( var_type == FACE_STAGGERED_AUXILIARY) {
        ASSERT_DBG( varidx < detail::_face_aux_bc_types.size(), 
                "Requested face-staggered auxiliary variable " << varidx << " does not have registered BCs."); 
        return detail::_face_aux_bc_types[varidx]  ;
    } else if ( var_type == EDGE_STAGGERED) {
        ASSERT_DBG( varidx < detail::_edge_vars_bc_types.size(), 
                "Requested edge-staggered variable " << varidx << " does not have registered BCs."); 
        return detail::_edge_vars_bc_types[varidx]  ;
    } else if ( var_type == EDGE_STAGGERED_AUXILIARY) {
        ASSERT_DBG( varidx < detail::_edge_aux_bc_types.size(), 
                "Requested edge-staggered auxiliary variable " << varidx << " does not have registered BCs."); 
        return detail::_edge_aux_bc_types[varidx]  ;
    } else if ( var_type == CORNER_STAGGERED) {
        ASSERT_DBG( varidx < detail::_corner_vars_bc_types.size(), 
                "Requested corner-staggered variable " << varidx << " does not have registered BCs."); 
        return detail::_corner_vars_bc_types[varidx]  ;
    } else if ( var_type == CORNER_STAGGERED_AUXILIARY) {
        ASSERT_DBG( varidx < detail::_corner_aux_bc_types.size(), 
                "Requested corner-staggered auxiliary variable " << varidx << " does not have registered BCs."); 
        return detail::_corner_aux_bc_types[varidx]  ;
    } else {
        ERROR("Unrecognized variable type in get_bc_type.") ; 
    }
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