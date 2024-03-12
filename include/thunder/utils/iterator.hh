/**
 * @file iterator.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-02-29
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

 #ifndef THUNDER_UTILS_ITERATOR_HH 
 #define THUNDER_UTILS_ITERATOR_HH

#include <cstdlib> 
#include <iterator> 

namespace utils { 

/**
 * @brief Generic iterator class 
 * 
 * @tparam value_type Type of value in the container
 * @tparam difference_type Type of difference between iterators 
 */
template< typename value_type 
        , typename difference_type = std::ptrdiff_t > 
class iterator 
{
    
    public:
        using pointer    = value_type* ;                      //!< Type of pointer 
        using reference  = value_type& ;                      //!< Type of reference
        using iterator_category = std::forward_iterator_tag ; //!< Tag 
        /**
         * @brief Construct a new iterator object
         */
        iterator( pointer ptr ) : _m(ptr) {} ; 
        /**
         * @brief Copy constructor 
         */
        iterator( const iterator& other ) = default ; 
        /**
         * @brief Destroy the iterator object
         */
        ~iterator() = default ; 
        /**
         * @brief Increment operator 
         */
        iterator& operator++() { 
            ++ _m ; 
            return *this ; 
        }
        /**
         * @brief Forward increment operator 
         */
        iterator operator++(int)
        {
            iterator ret = (*this) ; 
            ++(*this) ; 
            return ret ; 
        }
        /**
         * @brief Equality operator 
         */
        bool operator == (iterator other)
        {
            return other._m == _m ; 
        }
        /**
         * @brief Inequality operator 
         */
        bool operator != (iterator other)
        {
            return other._m != _m ; 
        }
        /**
         * @brief Dereference operator 
         */
        reference operator*() const { return *_m ; }

        private:
            pointer _m{nullptr} ; //!< Data being pointed to
} ; 

/**
 * @brief Specialization of generic iterator for const data.
 * 
 * @tparam value_type Type of value in the container
 * @tparam difference_type Type of difference between iterators 
 */
template< typename value_type 
        , typename difference_type = std::ptrdiff_t > 
class const_iterator 
 : public iterator<value_type, difference_type> 
{
    public:
        using pointer    = const value_type* ;                //!< Type of pointer 
        using reference  = const value_type& ;                //!< Type of reference
        using iterator_category = std::forward_iterator_tag ; //!< Tag  
        /**
         * @brief Dereference constant iterator. 
         */
        value_type operator*() const { return *(this->_m) ; }
} ; 

}  /* utils */


 #endif /* THUNDER_UTILS_ITERATOR_HH */ 