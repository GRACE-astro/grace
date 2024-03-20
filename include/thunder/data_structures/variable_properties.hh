/**
 * @file variable_properties.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-12
 * 
 * @copyright This file is part of MagMA.
 * MagMA is an evolution framework that uses Discontinuous Galerkin
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

#ifndef THUNDER_DATA_STRUCTURES_VARIABLE_PROPERTIES_HH
#define THUNDER_DATA_STRUCTURES_VARIABLE_PROPERTIES_HH

#include <thunder_config.h>
#include <Kokkos_Core.hpp> 

#include <thunder/data_structures/memory_defaults.hh>

namespace thunder {
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond thunder_detail
 * \ingroup variables
 */
template< size_t ndim > 
struct variable_properties_t 
{ } ; 
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond thunder_detail
 *  
 * \ingroup variables
 */
template<> 
struct variable_properties_t<2>
{
    using view_t = Kokkos::View<double ****, DefaultSpace> ; 
    std::array<bool, 2> staggering; 
    bool has_gz ; 
    bool is_vector ;  

    std::string name ; 
} ; 
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond thunder_detail
 * \ingroup variables
 */
template<> 
struct variable_properties_t<3>
{
    using view_t = Kokkos::View<double *****, DefaultSpace> ; 
    std::array<bool, 3> staggering; 
    bool has_gz ; 
    bool is_vector;
    
    std::string name ;
} ; 
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond thunder_detail
 * \ingroup variables
 */
template< size_t ndim >
struct coord_array_impl_t {} ;
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond thunder_detail
 * \ingroup variables
 */
template<> 
struct coord_array_impl_t<2> { using view_t = Kokkos::View<double ****, DefaultSpace>; };
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond thunder_detail
 * \ingroup variables
 */
template<> 
struct coord_array_impl_t<3> { using view_t = Kokkos::View<double *****, DefaultSpace>; } ;
//*****************************************************************************************************
/**
 * @brief Proxy for variable <code>View</code> type in Thunder
 * \ingroup variables
 * @tparam ndim Number of spatial dimension
 */
template< size_t ndim = THUNDER_NSPACEDIM > 
using var_array_t = variable_properties_t<ndim>::view_t ; 
//***************************************************************************************************** 
/**
 * @brief Proxy for coordinate <code>View</code> type in Thunder
 * \ingroup variables
 * @tparam ndim Number of spatial dimension
 */
template< size_t ndim = THUNDER_NSPACEDIM > 
using coord_array_t = coord_array_impl_t<ndim>::view_t ; 
//*****************************************************************************************************
//*****************************************************************************************************
} /* namespace thunder */

#endif /* THUNDER_DATA_STRUCTURES_VARIABLE_PROPERTIES_HH */