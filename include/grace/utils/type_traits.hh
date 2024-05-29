/**
 * @file type_traits.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief This file contains meta-programming utilities.
 * @version 0.1
 * @date 2023-03-13
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
#ifndef GRACE_UTILS_TYPE_TRAITS_HH
#define GRACE_UTILS_TYPE_TRAITS_HH

#include <type_traits>

namespace meta {
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief type-trait utils for meta-programming.
 * Used to express erroneous type in SFINAE-like contexts.
 */
//**************************************************************************************************
struct no_such_type {} ;
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief type-trait utility that determines whether a type 
 *        is valid. All types are valid except for
 *        <code>no_such_type</code>.
 * @tparam T type to check.
 */
//**************************************************************************************************
template<typename T>
struct is_valid : std::true_type {};
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief specialization of <code>is_valid</code>.
 */
//**************************************************************************************************
template<> struct is_valid<no_such_type> : std::false_type {} ; 
//**************************************************************************************************

//! \cond grace_detail
//**************************************************************************************************
/**
 * @brief Value (bool) of <code>is_valid<T></code> result.
 */
//**************************************************************************************************
template<typename T>
bool is_valid_v = is_valid<T>::value ; 
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Type of <code>is_valid<T></code> result.
 */
//**************************************************************************************************
template<typename T>
using is_valid_t = typename is_valid<T>::type ; 
//**************************************************************************************************
//! \endcond

//**************************************************************************************************
}
#endif 