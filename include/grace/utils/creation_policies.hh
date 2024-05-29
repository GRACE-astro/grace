/**
 * @file creators.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-03-10
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
#ifndef GRACE_MEMORY_CREATORS_HH
#define GRACE_MEMORY_CREATORS_HH

#include <algorithm>

#include <grace/utils/allocators.hh>

namespace memory {
//**************************************************************************************************
/**
 * @brief defines new_delete creator and its instantiations
 * \ingroup memory 
 * @tparam T type of the object to be created 
 * @tparam allocator memory allocator to be used 
 */
//**************************************************************************************************
template <typename T ,
          typename allocator > 
class new_delete_creator 
{
    //**************************************************************************************************
public:
    //**************************************************************************************************
    /**
     * @brief create an object of type T using placement <code> ::operator new </code>.
     * 
     * @tparam ArgT parameter pack to be passed to the constructor.
     * @param args arguments forwarded to the constructor.
     * @returns T* pointer to dynamically allocated object. 
     */
    //**************************************************************************************************
    template<typename ... ArgT>
    static inline __attribute__((always_inline))
    T* create(ArgT ... args) {
        void* buffer = allocator::allocate(sizeof(T), alignof(T)) ;
        return new(buffer) T(std::forward<ArgT>(args)...) ;
    } ;
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief destroy object created by this class
     */
    static inline __attribute__((always_inline))
    void destroy(T* ptr)
    {
        ptr->~T();
        allocator::deallocate(ptr, sizeof(T), alignof(T)) ;
    }
    //**************************************************************************************************

    //**************************************************************************************************s
} ;
/**
 * @brief Alias for creator class using global
 *        <code>malloc</code> and <code>free</code>.
 * \ingroup memory
 * @tparam T type being created.
 */
template<typename T>
using default_create = new_delete_creator<T, new_delete_allocator> ;

}

 #endif 
