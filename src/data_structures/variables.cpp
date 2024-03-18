/**
* @file variables.cpp
* @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
* @brief 
* @version 0.1
* @date 2024-03-07
* 
* @copyright This file is part of Thunder.
* Thunder is an evolution framework that uses Finite Difference
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

#include <thunder/data_structures/variables.hh>
#include <thunder/data_structures/variable_indices.hh>

#include <thunder/config/config_parser.hh>
#include <thunder/amr/forest.hh>

#include <thunder/errors/assert.hh>

#include <vector>
#include <algorithm>
#include <string> 

namespace thunder 
{

size_t get_variable_index(std::string const& name)
{
    using namespace thunder::variables::detail ; 
    /* first check if it's a state variable */
    auto it = std::find(_varnames.begin(), _varnames.end(), name);
    if (it != _varnames.end())
    {
        return std::distance(_varnames.begin(),it) ; 
    } 
    it = std::find(_auxnames.begin(), _auxnames.end(), name); 
    if (it != _auxnames.end())
    {
        return std::distance(_varnames.begin(),it) ; 
    } 
    ASSERT_DBG(0, 
    "In get_variable_index, variable "
    << name << " not found.") ; 
    return -1; 
}

variable_list_impl_t::variable_list_impl_t() 
    : _coords("coordinates", VEC(0,0,0), 0,0)
    , _state("state", VEC(0,0,0),0,0)
    , _state_p("scratch_state", VEC(0,0,0),0,0)
    , _aux("auxiliaries", VEC(0,0,0),0,0)
{
    using namespace thunder; 
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
    /* register all variables known to Thunder */
    variables::register_variables() ;
    /* allocate memory for states */ 
    size_t nq          = forest.local_num_quadrants() ;
    Kokkos::realloc( _coords
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , nq 
                   , THUNDER_NSPACEDIM
                   ) ;
    Kokkos::realloc( _state
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , nq 
                   , variables::detail::num_evolved
                   ) ;
    Kokkos::realloc( _state_p
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , nq 
                   , variables::detail::num_evolved
                   ) ;
    Kokkos::realloc( _aux
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , nq 
                   , variables::detail::num_auxiliary
                   ) ;
    /* all done */}

} /* namespace thunder */