/**
 * @file sc_wrappers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Define some utility classes/functions to wrap around sc clunky syntax.
 * @version 0.1
 * @date 2024-02-29
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

#ifndef GRACE_UTILS_SC_WRAPPERS_HH 
#define GRACE_UTILS_SC_WRAPPERS_HH 

#include <sc.h>
#include <sc_containers.h>

#include <grace/errors/assert.hh>
#include <grace/utils/iterator.hh>
#include <grace/utils/make_string.hh> 
#include <grace/utils/type_name.hh> 

namespace grace { 

/**
 * @brief View of an sc_array
 * \ingroup utils
 * @tparam T Data type contained in the array 
 * 
 * This is a non-owning view of an <code>sc_array_t</code>.
 * It is templated for the contained type to get around the 
 * type erasure of sc_arrays which makes them quite annoying to 
 * handle. Also defines some basic accessors which should always
 * exist for container types.  
 */
template< typename T > // Data type 
class sc_array_view_t 
{
    private:

        using index_type = std::size_t ; //!< Type of index 
        using value_type          = T  ; //!< Type of values 
        using pointer             = T* ; //!< Type of pointer to values 
        using reference           = T& ; //!< Type of reference to values 
        using iterator_type       = utils::iterator<value_type>       ; //!< Iterator type 
        using const_iterator_type = utils::const_iterator<value_type> ; //!< Const iterator type 

        sc_array_t * array_ ; //!< Array being viewed. Not owned! 

    public: 
        /**
         * @brief Default ctor.
         */
        sc_array_view_t( sc_array_t * arr ) : array_(arr) {} ;
        /**
         * @brief No copy ctor.
         */
        sc_array_view_t( sc_array_view_t const& ) = delete   ; 
        /**
         * @brief Move ctor
         */
        sc_array_view_t( sc_array_view_t && rhs ) = default  ; 
        /**
         * @brief Dtor
         */
        ~sc_array_view_t() = default ; 

        /**
         * @brief Data access operator.
         * 
         * @param i Index (checked for bounds in debug)
         * @return Ref Reference to item in the array.
         */
        reference operator[] (index_type const& i ){
            ASSERT_DBG( i < array_->elem_count
                      , "Out of bounds access detected." ) ; 
            return reinterpret_cast<pointer>(array_->array)[i] ; 
        }
        /**
         * @brief Const data access operator.
         * 
         * @param i Index (checked for bounds in debug)
         * @return Type Item in the array. 
         */
        value_type operator[] (index_type const& i ) const {
            ASSERT_DBG( i < array_->elem_count
                      , "Out of bounds access detected.") ;
            return reinterpret_cast<pointer>(array_->array)[i] ; 
        }
        /**
         * @brief Size of the array. 
         * 
         * @return index_t The number of elements in the array.
         */
        index_type size() const {
            return array_->elem_count; 
        }
        /**
         * @brief Return iterator pointing at the beginning of the array.
         */
        iterator_type begin() 
        { 
            return iterator_type(
                reinterpret_cast<pointer>(array_->array)
                ) ; 
        }
        /**
         * @brief Return iterator pointing at the end of the array.
         */
        iterator_type end() 
        { 
            return iterator_type(
                reinterpret_cast<pointer>(array_->array) + array_->elem_count 
                ) ; 
        }
        /**
         * @brief Return const iterator pointing at the beginning of the array.
         */
        const_iterator_type cbegin() 
        const { 
            return const_iterator_type(
                reinterpret_cast<pointer>(array_->array)
            ) ; 
        }
        /**
         * @brief Return const iterator pointing at the end of the array.
         */
        const_iterator_type cend() 
        const { 
            return const_iterator_type(
                reinterpret_cast<pointer>(array_->array) + array_->elem_count 
            ) ; 
        }
        
} ; 

} /* grace */

 #endif /* GRACE_AMR_SC_WRAPPERS_HH */ 