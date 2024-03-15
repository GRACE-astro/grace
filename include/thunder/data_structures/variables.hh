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
//*****************************************************************************************************
/**
 * @brief Create additional state. Allocates memory on device.
 * 
 * \ingroup variables
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

//*****************************************************************************************************
/**
 * @brief Implementation of the variable list type.
 * 
 * \ingroup variables
 * 
 * This class is wrapped with a <code>singleton_holder</code>
 * and is used as a utility to collect all variables and coordinates
 * and ensure they have the appropriate lifetime during Thunder's execution.
 */
class variable_list_impl_t
{

public:
    //*****************************************************************************************************
    /**
     * @brief Get cell coordinates.
     * 
     * @return Cell coordinates. 
     */
    THUNDER_ALWAYS_INLINE coord_array_t<THUNDER_NSPACEDIM> 
    getcoords() { return _coords ; } 
    //*****************************************************************************************************
    /**
     * @brief Get the auxiliary variables.
     * 
     * @return The auxiliary variables. 
     */
    THUNDER_ALWAYS_INLINE var_array_t<THUNDER_NSPACEDIM>  
    getaux() { return _aux ; }
    //*****************************************************************************************************
    /**
     * @brief Get state vector
     * 
     * @return The state vector, containing all evolved variables
     *         on all local cells.  
     */
    THUNDER_ALWAYS_INLINE var_array_t<THUNDER_NSPACEDIM>  
    getstate() { return _state ; }
    //*****************************************************************************************************
    /**
     * @brief Get the scratch state vector 
     * 
     * @return The scratch state vector, used during time 
     *         evolution to hold the previous state. 
     */
    THUNDER_ALWAYS_INLINE var_array_t<THUNDER_NSPACEDIM> 
    getscratch() { return _state_p ; }
    //*****************************************************************************************************

private: 
    //*****************************************************************************************************
    /**
     * @brief (Never) construct a new <code>variable_list_impl_t</code> object
     * 
     */
    variable_list_impl_t() ; 
    //*****************************************************************************************************
    /**
     * @brief (Never) destroy the <code>variable_list_impl_t</code> object
     * 
     */
    ~variable_list_impl_t() = default; 
    //*****************************************************************************************************
    //******** Member variables ***************************************************************************
    //*****************************************************************************************************
    coord_array_t<THUNDER_NSPACEDIM>  _coords  ;  //!< Gridpoint coordinates    
    var_array_t<THUNDER_NSPACEDIM> _state   ;     //!< State variables 
    var_array_t<THUNDER_NSPACEDIM> _state_p ;     //!< Second timelevel, allocated at all times 
    var_array_t<THUNDER_NSPACEDIM> _aux     ;     //!< Auxiliary variables 
    //*****************************************************************************************************
    friend class utils::singleton_holder<variable_list_impl_t, memory::default_create> ; //!< Give access 
    friend class memory::new_delete_creator<variable_list_impl_t, memory::new_delete_allocator> ; //!< Give access 
    //*****************************************************************************************************
    static constexpr size_t longevity = THUNDER_VARIABLES ; //!< Schedule destruction at appropriate time.
    //*****************************************************************************************************
} ; 
//*****************************************************************************************************
/**
 * @brief Proxy holding all variables within Thunder. Only a unique instance 
 *        of this class exists at runtime and its reference can be obtained 
 *        via the <code>get()</code> static method. See references on 
 *        <code>singleton_holder</code> for details.
 * \ingroup variables
 */
using variable_list = utils::singleton_holder<variable_list_impl_t > ; 
//*****************************************************************************************************
//*****************************************************************************************************

} /* namespace thunder */

#endif /* THUNDER_DATA_STRUCTURES_VARIABLES_HH */ 