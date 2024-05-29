/**
 * @file grace/memory/new_delete_allocator.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Bring <code> std::malloc </code> and <code> std::free </code> into grace's allocator system
 * @version 0.1
 * @date 2023-03-10
 * 
 * Simple allocator for Host memory, used to abstract dynamic allocation policies
 * from singleton code. This system does not support memory allocation on the Device
 * and should only be used for data structures meant to live on the Host such as 
 * the grid hierarchy and its connectivity.
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
#ifndef GRACE_NEW_DELETE_ALLOCATOR_HH
#define GRACE_NEW_DELETE_ALLOCATOR_HH

#include <cstdlib>
#include <memory_resource> 
#include <iostream> 

namespace memory 
{
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Thin wrapper around <code>malloc</code>, <code>realloc</code> and <code>free</code>.
 */
//**************************************************************************************************
class new_delete_allocator 
{   
    public:
        //**************************************************************************************************
        /**
         * @brief Allocate memory using global allocator.
         * 
         * @param size size (in bytes) of requested allocation.
         * @param alignment aligment.
         * @return void* pointer to allocated memory
         */
        //**************************************************************************************************
        [[nodiscard]] static inline void*
        allocate(std::size_t size,
                 std::size_t alignment ) 
        {
            return std::aligned_alloc(alignment, size) ; 
        } 
        //**************************************************************************************************

        //**************************************************************************************************
        /**
         * @brief deallocate pointer allocated by this class.
         * 
         * @param ptr pointer to bre free'd
         * @param size size (in bytes) of memory owned by <code> ptr </code>.
         * @param alignment aligment of <code> ptr </code>.
         */ 
        static inline void deallocate( void* ptr, 
                                       std::size_t size,
                                       std::size_t alignment ) noexcept 
        {
            std::free(ptr) ; 
        } 
        //**************************************************************************************************
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
#endif 