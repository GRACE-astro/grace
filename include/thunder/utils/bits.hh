/**
 * @file bits.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2023-03-23
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
#ifndef THUNDER_UTILS_BITS_HH
#define THUNDER_UTILS_BITS_HH

#include <thunder/utils/inline.h>

#include <cstdlib>
#include <limits.h>

namespace utils {
/**
 * @brief 
 * 
 * @tparam T 
 * @param ptr 
 * @param n 
 * @return THUNDER_ALWAYS_INLINE 
 */
template< typename T >
THUNDER_ALWAYS_INLINE bool nth_bit(const T* ptr, const size_t n)
{
    static constexpr size_t WORDSIZE = sizeof(unsigned char) ; 
    return *( reinterpret_cast<const unsigned char*>(ptr) + n/CHAR_BIT ) & (1 << (n % CHAR_BIT));
}

template< typename T>
THUNDER_ALWAYS_INLINE void bit_set(T& k, const size_t n)
{
    k |= 1UL << n ; 
}

template< typename T>
THUNDER_ALWAYS_INLINE void bit_clear(T& k, const size_t n)
{
    k &= ~(1UL << n) ; 
}

template< typename T>
THUNDER_ALWAYS_INLINE void bit_toggle(T& k, const size_t n)
{
    k ^= 1UL << n ; 
}

template< typename T>
THUNDER_ALWAYS_INLINE bool bit_check(T& k, const size_t n)
{
    return (k >> n) & 1U;
}

template< typename T>
THUNDER_ALWAYS_INLINE void bit_set_to(T& k, bool x, const size_t n)
{
    k ^= (-x ^ k) & (1UL << n); 
}

} // namespace utils
#endif /* THUNDER_UTILS_BITS_HH */
