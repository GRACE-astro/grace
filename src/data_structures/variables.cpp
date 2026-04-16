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
#include <grace/errors/warn.hh>
#include <grace/system/print.hh>

#include <vector>
#include <algorithm>
#include <string> 

namespace grace 
{

int get_variable_index(std::string const& name, bool is_aux)
{
    using namespace grace::variables::detail ; 
    /* first check if it's a state variable */
    if( !is_aux ) {
        auto it = std::find(_varnames.begin(), _varnames.end(), name);
        if (it != _varnames.end())
        {
            return std::distance(_varnames.begin(),it) ; 
        } 
        GRACE_WARN("Variable index could not be retrieved for {}", name) ; 
    } else {
        auto it = std::find(_auxnames.begin(), _auxnames.end(), name); 
        if (it != _auxnames.end())
        {
            return std::distance(_auxnames.begin(),it) ; 
        } 
        GRACE_WARN("Auxiliary variable index could not be retrieved for {}", name) ;
    }
    
    ASSERT_DBG(0, 
    "In get_variable_index, variable "
    << name << " not found.") ; 
    return -1; 
}

variable_list_impl_t::variable_list_impl_t() 
    : _coords_ispacing("inverse_grid_spacing", 0,0)
    , _coords_spacing("grid_spacing", 0,0)
    , _state("state", VEC(0,0,0),0,0)
    , _state_p("scratch_state", VEC(0,0,0),0,0)
    , _aux("auxiliaries", VEC(0,0,0),0,0)
    , _staggered_vars()
    , _staggered_vars_p()
    , _fluxes()
    , _emf()
    #ifdef GRACE_ENABLE_Z4C_METRIC
    , _z4c_curv_scratch("z4c_curv_scratch", VEC(0,0,0),0,0)
    #endif
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
    Kokkos::realloc( _coords_ispacing
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ;
    Kokkos::realloc( _coords_spacing
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ;
    Kokkos::realloc( _state
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , N_EVOL_VARS
                   , nq 
                   ) ;
    Kokkos::realloc( _state_p
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , N_EVOL_VARS
                   , nq 
                   ) ;
    Kokkos::realloc( _aux
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , N_AUX_VARS
                   , nq 
                   ) ;  
    Kokkos::realloc( _fluxes
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , nvars_hrsc 
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ;
    #ifdef GRACE_GRMHD_USE_GS
    Kokkos::realloc( _Ecenter
                   , VEC( nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , 3 // E^x E^y E^z
                   , nq 
                   ) ; 
    Kokkos::realloc( _Eface
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , 2 // E^i E^j
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ; 
    #else 
    Kokkos::realloc( _vbar
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , 4 // v^i v^j c_p c_m
                   , GRACE_NSPACEDIM
                   , nq 
                   ) ; 
    #endif 
    Kokkos::realloc( _emf
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , GRACE_NSPACEDIM 
                   , nq ) ; 
    _staggered_vars.realloc( VEC(nx,ny,nz),ngz,nq
                           , N_FC_X
                           , N_EC_YZ
                           , N_VC) ;
    _staggered_vars_p.realloc( VEC(nx,ny,nz),ngz,nq
                             , N_FC_X
                             , N_EC_YZ
                             , N_VC) ;
    
    ASSERT(variables::detail::_varnames.size() == N_EVOL_VARS, 
    "Num evolved is " << N_EVOL_VARS << " but varnames.size() is " << variables::detail::_varnames.size() ) ; 
    // allocate staging buffers for timestepper 
    auto tstepper = get_param<std::string>("evolution","time_stepper") ; 
    int n_staging_bufs{0} ;  
    if ( tstepper == "rk3" or tstepper == "imex222" or tstepper=="rk4") { 
        n_staging_bufs = 1 ; 
    }
    _staging_buffer.reserve(n_staging_bufs) ; 
    _stag_staging_buffer.reserve(n_staging_bufs) ;
    for( int i=0; i<n_staging_bufs; ++i) {
        _staging_buffer.emplace_back("staging_buffer", nx+2*ngz,ny+2*ngz,nz+2*ngz,N_EVOL_VARS,nq) ; 
        _stag_staging_buffer.emplace_back(
            nx,ny,nz,ngz,nq,
            N_FC_X,
            N_EC_YZ,
            N_VC
        ); 
    }
    /* all done */
}

void variable_list_impl_t::resize_aux_staging_and_flux_buffers(int nq_new) 
{
    DECLARE_GRID_EXTENTS;
    int nvars_hrsc = variables::get_n_hrsc() ;
    Kokkos::realloc( _aux
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , N_AUX_VARS
                   , nq_new
                   ) ;  
    Kokkos::realloc( _fluxes
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , nvars_hrsc 
                   , GRACE_NSPACEDIM
                   , nq_new 
                   ) ;
    #ifdef GRACE_GRMHD_USE_GS
    Kokkos::realloc( _Ecenter
                   , VEC( nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , 3 // E^x E^y E^z
                   , nq_new 
                   ) ; 
    Kokkos::realloc( _Eface
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , 2 // E^i E^j
                   , GRACE_NSPACEDIM
                   , nq_new 
                   ) ; 
    #else 
    Kokkos::realloc( _vbar
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , 4 // v^i v^j c_p c_m
                   , GRACE_NSPACEDIM
                   , nq_new 
                   ) ; 
    #endif 
    Kokkos::realloc( _emf
                   , VEC( nx + 1 + 2*ngz,ny + 1 + 2*ngz,nz + 1 + 2*ngz)
                   , GRACE_NSPACEDIM
                   , nq_new ) ;

    #ifdef GRACE_ENABLE_Z4C_METRIC
    Kokkos::realloc( _z4c_curv_scratch
                   , VEC(nx + 2*ngz, ny + 2*ngz, nz + 2*ngz)
                   , N_Z4C_CURV_SCRATCH
                   , nq_new
                   ) ;
    #endif

    for( int ibuf=0; ibuf<_staging_buffer.size(); ++ibuf) {
        Kokkos::realloc(
            _staging_buffer[ibuf]
            , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
            , N_EVOL_VARS
            , nq_new 
        ) ; 
        _stag_staging_buffer[ibuf].realloc(
            nx,ny,nz,ngz,nq_new,
            N_FC_X,
            N_EC_YZ,
            N_VC
        ) ; 
    }
}


} /* namespace grace */