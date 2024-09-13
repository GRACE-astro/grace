/**
* @file variables.cpp
* @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
* @brief 
* @version 0.1
* @date 2024-03-07
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

#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_indices.hh>

#include <grace/config/config_parser.hh>
#include <grace/amr/forest.hh>

#include <grace/errors/assert.hh>

#include <vector>
#include <algorithm>
#include <string> 

namespace grace 
{

size_t get_variable_index_ext(std::string const& name, int& staggering, bool is_aux) {
    auto ivar = get_variable_index(name,is_aux) ; 
    if ( ivar >= 0 ) {
        staggering = -1 ; 
        return ivar ; 
    }
    return get_staggered_variable_index(name, staggering, is_aux) ; 
}

size_t get_staggered_variable_index( std::string const& name, int& staggering, bool is_aux)
{
    using namespace grace::variables::detail ; 
    if( !is_aux ) {
        auto itf = std::find(_face_staggered_varnames.begin(), _face_staggered_varnames.end());
        if (itf != _face_staggered_varnames.end())
        {   
            staggering = 0 ; 
            return std::distance(_face_staggered_varnames.begin(),itf) ; 
        } 
        auto itc = std::find(_corner_staggered_varnames.begin(), _corner_staggered_varnames.end());
        if (itc != _corner_staggered_varnames.end())
        {   
            staggering = 1 ; 
            return std::distance(_corner_staggered_varnames.begin(),itc) ; 
        } 
        auto ite = std::find(_edge_staggered_varnames.begin(), _edge_staggered_varnames.end());
        if (ite != _edge_staggered_varnames.end())
        {   
            staggering = 2 ; 
            return std::distance(_edge_staggered_varnames.begin(),ite) ; 
        } 
    } else {
        auto itf = std::find(_face_staggered_aux.begin(), _face_staggered_aux.end());
        if (itf != _face_staggered_aux.end())
        {   
            staggering = 0 ; 
            return std::distance(_face_staggered_aux.begin(),itf) ; 
        } 
        auto itc = std::find(_corner_staggered_aux.begin(), _corner_staggered_aux.end());
        if (itc != _corner_staggered_aux.end())
        {   
            staggering = 1 ; 
            return std::distance(_corner_staggered_aux.begin(),itc) ; 
        } 
        auto ite = std::find(_edge_staggered_aux.begin(), _edge_staggered_aux.end());
        if (ite != _edge_staggered_aux.end())
        {   
            staggering = 2 ; 
            return std::distance(_edge_staggered_aux.begin(),ite) ; 
        }
    }
    return -1; 
}

size_t get_variable_index(std::string const& name, bool is_aux)
{
    using namespace grace::variables::detail ; 
    /* first check if it's a state variable */
    if( !is_aux ) {
        auto it = std::find(_varnames.begin(), _varnames.end(), name);
        if (it != _varnames.end())
        {
            return std::distance(_varnames.begin(),it) ; 
        } 
    } else {
        auto it = std::find(_auxnames.begin(), _auxnames.end(), name); 
        if (it != _auxnames.end())
        {
            return std::distance(_auxnames.begin(),it) ; 
        } 
    }
    return -1; 
}

variable_list_impl_t::variable_list_impl_t() 
    : _coords("coordinates", 0,0)
    , _coords_ispacing("inverse_grid_spacing", 0,0)
    , _coords_spacing("grid_spacing", 0,0)
    , _cell_volumes("cell_volumes",VEC(0,0,0),0)
    , _state("state", VEC(0,0,0),0,0)
    , _state_p("scratch_state", VEC(0,0,0),0,0)
    , _halo("halo", VEC(0,0,0),0,0)
    , _aux("auxiliaries", VEC(0,0,0),0,0)
    , _staggered_coords()
    , _staggered_vars() 
    , _staggered_vars_p() 
    , _staggered_aux() 
{
    using namespace grace; 
    /* Get param parser and forest object */
    auto& params = config_parser::get() ; 
    auto& forest = amr::forest::get()   ; 
    /* Read parameters from config file: */
    /* 1) Grid quadrant (octant) dimensions */
    size_t nx {params["amr"]["npoints_block_x"].as<size_t>()} ; 
    size_t ny {params["amr"]["npoints_block_y"].as<size_t>()} ; 
    size_t nz {params["amr"]["npoints_block_z"].as<size_t>()} ; 
    /* 2) Number of ghostzones for evolved vars */
    size_t ngz { params["amr"]["n_ghostzones"].as<size_t>() } ;  
    /* register all variables known to GRACE */
    variables::register_variables() ;
    /* allocate memory for states */ 
    size_t nq          = forest.local_num_quadrants() ;
    Kokkos::realloc( _coords
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ;
    Kokkos::realloc( _coords_ispacing
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ;
    Kokkos::realloc( _coords_spacing
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ;
    Kokkos::realloc( _cell_volumes
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , nq 
                   ) ;
    Kokkos::realloc( _state
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , variables::detail::num_evolved
                   , nq 
                   ) ;
    Kokkos::realloc( _state_p
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , variables::detail::num_evolved
                   , nq 
                   ) ;
    Kokkos::realloc( _aux
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , variables::detail::num_auxiliary
                   , nq 
                   ) ;
    
    _staggered_coords.realloc(VEC(nx,ny,nz),ngz,nq) ; 
    _staggered_vars.realloc( VEC(nx,ny,nz),ngz,nq
                           , variables::detail::num_face_staggered_vars
                           , variables::detail::num_edge_staggered_vars 
                           , variables::detail::num_corner_staggered_vars) ;
    _staggered_vars_p.realloc( VEC(nx,ny,nz),ngz,nq
                             , variables::detail::num_face_staggered_vars
                             , variables::detail::num_edge_staggered_vars 
                             , variables::detail::num_corner_staggered_vars) ;
    _staggered_aux.realloc( VEC(nx,ny,nz),ngz,nq
                          , variables::detail::num_face_staggered_aux
                          , variables::detail::num_edge_staggered_aux
                          , variables::detail::num_corner_staggered_aux) ;
    
    ASSERT(variables::detail::_varnames.size() == variables::detail::num_evolved, 
    "Num evolved is " << variables::detail::num_evolved << " but varnames.size() is " << variables::detail::_varnames.size() ) ; 

    /* all done */
}

} /* namespace grace */