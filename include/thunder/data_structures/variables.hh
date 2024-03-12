/**
 * @file variables.hh
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

#ifndef THUNDER_DATA_STRUCTURES_VARIABLES_HH
#define THUNDER_DATA_STRUCTURES_VARIABLES_HH

#include <Kokkos_Core.hpp>

#include <thunder_config.h>

#include<code_modules.h>
#include<thunder/data_structures/variable_properties.hh>
#include<thunder/data_structures/variable_indices.hh>
#include<thunder/data_structures/macros.hh>

#include<thunder/utils/inline.h>
#include<thunder/utils/singleton_holder.hh> 
#include<thunder/utils/creation_policies.hh>
#include<thunder/utils/lifetime_tracker.hh>

namespace thunder { 

template< size_t ndim = THUNDER_NSPACEDIM > 
using var_array_t = variable_properties_t<ndim>::view_t ;  

template< size_t ndim = THUNDER_NSPACEDIM > 
using coord_array_t = coord_array_impl_t<ndim>::view_t ; 

/**
 * @brief Create additional state. Allocates memory on device.
 * 
 * @tparam ndim Number of space dimensions.
 *
 * @param src State used to copy data and layout from.
 * @param initialize Copy data from source? 
 * 
 * Memory is released as soon as the caller's scope is exited.
 */
template< size_t ndim=THUNDER_NSPACEDIM>
static var_array_t<ndim> create_state(var_array_t<ndim> const& src, bool initialize=true) ; 


template< size_t ndim = THUNDER_NSPACEDIM > 
class variable_list_impl_t
{

public: 
    
    THUNDER_ALWAYS_INLINE var_array_t<ndim>  
    getaux() { return _aux ; }

    THUNDER_ALWAYS_INLINE var_array_t<ndim>  
    getstate() { return _state ; }

    THUNDER_ALWAYS_INLINE var_array_t<ndim> 
    getscratch(int tl) { return _state_p ; }


private: 

    variable_list_impl_t() ; 

    ~variable_list_impl_t() = default; 

    coord_array_t<ndim>  _coords  ;  //!< Gridpoint coordinates    
    var_array_t<ndim> _state   ;     //!< State variables 
    var_array_t<ndim> _state_p ;     //!< Second timelevel, allocated at all times 
    var_array_t<ndim> _aux     ;     //!< Auxiliary variables 

    friend class utils::singleton_holder<variable_list_impl_t, memory::default_create> ; //!< Give access 
    friend class memory::new_delete_creator<variable_list_impl_t, memory::new_delete_allocator> ; //!< Give access 

    static constexpr size_t longevity = THUNDER_VARIABLES ; 

} ; 

template<size_t ndim>
variable_list_impl_t<ndim>::variable_list_impl_t() 
    : _coords("coordinates", VEC(0,0,0), 0)
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
                   , nq ) ;
    Kokkos::realloc( _state
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , variables::detail::num_evolved
                   , nq ) ;
    Kokkos::realloc( _state_p
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , variables::detail::num_evolved
                   , nq ) ;
    Kokkos::realloc( _aux
                   , VEC(nx + 2*ngz,ny + 2*ngz,nz + 2*ngz)
                   , variables::detail::num_auxiliary
                   , nq ) ;
    /* all done */
}

using variable_list = utils::singleton_holder<variable_list_impl_t<THUNDER_NSPACEDIM> > ; 

} /* thunder */

#endif /* THUNDER_DATA_STRUCTURES_VARIABLES_HH */ 