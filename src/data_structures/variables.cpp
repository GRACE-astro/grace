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
#include <grace/data_structures/variable_utils.hh>

#include <grace/config/config_parser.hh>
#include <grace/amr/forest.hh>

#include <grace/errors/assert.hh>

#include <vector>
#include <algorithm>
#include <string> 

namespace grace 
{

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
    ASSERT_DBG(0, 
    "In get_variable_index, variable "
    << name << " not found.") ; 
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
    , _fluxes()
    , _emf()
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
    int nvars_hrsc = variables::get_n_hrsc() ;
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
    Kokkos::realloc( _fluxes
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , nvars_hrsc 
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ;
    Kokkos::realloc( _vbar
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , 4 // v^i v^j c_p c_m
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ; 
    Kokkos::realloc( _emf
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , GRACE_NSPACEDIM 
                   , nq ) ; 
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
    // allocate staging buffers for timestepper 
    auto tstepper = get_param<std::string>("evolution","time_stepper") ; 
    int n_staging_bufs{0} ;  
    if ( tstepper == "rk3" or tstepper == "imex222") { 
        n_staging_bufs = 1 ; 
    }
    _staging_buffer.reserve(n_staging_bufs) ; 
    _stag_staging_buffer.reserve(n_staging_bufs) ;
    for( int i=0; i<n_staging_bufs; ++i) {
        _staging_buffer.emplace_back("staging_buffer", nx+2*ngz,ny+2*ngz,nz+2*ngz,variables::detail::num_evolved,nq) ; 
        _stag_staging_buffer.emplace_back(
            nx,ny,nz,ngz,nq,
            variables::detail::num_face_staggered_vars,
            variables::detail::num_edge_staggered_vars,
            variables::detail::num_corner_staggered_vars
        ); 
    }
    /* all done */
}


} /* namespace grace */