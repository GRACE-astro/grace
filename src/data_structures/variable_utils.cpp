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

bool var_exists(std::string const& vname) {
    auto it = detail::_varprops.find(vname);
    if (it == detail::_varprops.end()) {
        return false ;
    }
    return true ;
}

grace::variable_properties_t<GRACE_NSPACEDIM>&
get_variable_properties(std::string const& vname, int& err)
{   
    // Check if the variable exists in the map to avoid undefined behavior.
    auto it = detail::_varprops.find(vname);
    if (it == detail::_varprops.end()) {
        err = -1 ; 
        return detail::_varprops.begin()->second; 
    }
    err = 0 ;
    return it->second; 
}

int 
get_n_evolved_face_staggered()
{
    return detail::num_face_staggered_vars ;
} 

int 
get_n_evolved_edge_staggered()
{
    return detail::num_edge_staggered_vars ;
} 

int 
get_n_evolved_corner_staggered()
{
    return detail::num_corner_staggered_vars ;
} 

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
get_bc_type(int64_t varidx, var_staggering_t const& staggering)
{
    if ( staggering == var_staggering_t::CELL_CENTER ) {
        ASSERT_DBG( varidx < detail::_var_bc_types.size(), 
                "Requested variable " << varidx << " does not have registered BCs."); 
        return detail::_var_bc_types[varidx]  ;
    } else if ( staggering == var_staggering_t::FACE ) {
        ASSERT_DBG( varidx < detail::_face_vars_bc_types.size(), 
                "Requested face-staggered variable " << varidx << " does not have registered BCs."); 
        return detail::_face_vars_bc_types[varidx]  ;
    } else if ( staggering == var_staggering_t::EDGE ) {
        ASSERT_DBG( varidx < detail::_edge_vars_bc_types.size(), 
                "Requested edge-staggered variable " << varidx << " does not have registered BCs."); 
        return detail::_edge_vars_bc_types[varidx]  ;
    } else if ( staggering == var_staggering_t::CORNER) {
        ASSERT_DBG( varidx < detail::_corner_vars_bc_types.size(), 
                "Requested corner-staggered variable " << varidx << " does not have registered BCs."); 
        return detail::_corner_vars_bc_types[varidx]  ;
    } else {
        ERROR("Unrecognized variable staggering in get_bc_type.") ; 
    }
}

std::string get_var_name(int64_t var_idx, bool is_aux) {
    return is_aux ? detail::_auxnames[var_idx]
                  : detail::_varnames[var_idx] ; 
}

std::vector<std::size_t>
get_vector_state_variables_indices() {
    std::vector<std::size_t> indices ; 
    for( int i=0; i<detail::_varnames.size(); ++i){
        auto const& props = detail::_varprops[detail::_varnames[i]] ;
        if(props.is_vector) {
            indices.push_back(i); 
            i += 2; 
        }
    }
    return std::move(indices)  ;
}

std::vector<std::size_t>
get_tensor_state_variables_indices() {
    std::vector<std::size_t> indices ; 
    for( int i=0; i<detail::_varnames.size(); ++i){
        auto const& props = detail::_varprops[detail::_varnames[i]] ;
        if(props.is_tensor) {
            indices.push_back(i); 
            i += 5; 
        }
    }
    return std::move(indices)  ;
}

std::vector<std::size_t>
get_vector_aux_variables_indices() {
    std::vector<std::size_t> indices ; 
    for( int i=0; i<detail::_auxnames.size(); ++i){
        auto const& props = detail::_varprops[detail::_auxnames[i]] ;
        if(props.is_vector) {
            indices.push_back(i); 
            i += 2; 
        }
    }
    return std::move(indices)  ;
}

std::vector<std::size_t>
get_tensor_aux_variables_indices() {
    std::vector<std::size_t> indices ; 
    for( int i=0; i<detail::_auxnames.size(); ++i){
        auto const& props = detail::_varprops[detail::_auxnames[i]] ;
        if(props.is_tensor) {
            indices.push_back(i); 
            i += 5; 
        }
    }
    return std::move(indices)  ;
}


}} /* namespace grace::variables */