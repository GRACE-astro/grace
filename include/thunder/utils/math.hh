/**
 * @file math.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-03-13
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
#ifndef THUNDER_MATH_UTILS_HH
#define THUNDER_MATH_UTILS_HH

#include <cstdlib>
#include <cmath>

namespace math
{
  namespace detail {
  //! \cond thunder_detail 
  /**
   * @brief Compute integer powers.
   * \ingroup utils
   * @tparam T type of argument 
   * @tparam N power
   * Implemented as a struct to allow for partial specialization.
   */
  template< typename T, size_t N>
  struct int_pow_impl
  {
    static inline __attribute__((always_inline))
    T get(T const& x)
    {
      if constexpr ( N==0 )
        return static_cast<T>(1.); 
      else
        return x * int_pow_impl<T,N-1>::get(x) ; 
    } ; 
  } ; 
  //! \endcond
  } // namespace detail

  /**
   * @brief Compute integer powers.
   * 
   * @tparam T type of argument 
   * @tparam N power
   * @param x value to be exponentiated
   * @return T x to the power of N
   */
  template<size_t N, typename T> 
  static inline __attribute__((always_inline))
  T int_pow(T const& x )
  {
    return detail::int_pow_impl<T,N>::get(x) ; 
  } ;

  /**
   * @brief compute absolute value (type agnostic).
   * @tparam T type of parameter
   * @param x value whose absolute value we want
   * @return T absolute value of x
   */
  template < typename T >
  static inline __attribute__((always_inline))
  T abs ( T const & x ) {
    return ( x > static_cast<T>(0) ) ? x : -x ;
  }
  
}

#endif 
