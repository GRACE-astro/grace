/**
 * @file math.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
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
#ifndef GRACE_UTILS_MATH_HH
#define GRACE_UTILS_MATH_HH

#include <grace_config.h>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <cstdlib>
#include <cmath> 

#ifdef __CUDACC__
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif


namespace math
{
namespace detail {
//! \cond grace_detail 
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
  static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
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
 * \ingroup utils
 * 
 * @tparam T type of argument 
 * @tparam N power
 * @param x value to be exponentiated
 * @return T x to the power of N
 */
template<size_t N, typename T> 
static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
T int_pow(T const& x )
{
  return detail::int_pow_impl<T,N>::get(x) ; 
} ;

/**
 * @brief compute absolute value (type agnostic).
 * \ingroup utils
 * @tparam T type of parameter
 * @param x value whose absolute value we want
 * @return T absolute value of x
 */
template < typename T >
static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
T abs ( T const & x ) {
  return ( x > static_cast<T>(0) ) ? x : -x ;
}

/**
 * @brief Signum
 * \ingroup utils
 * @tparam T Type of input
 * @param val Value whose sign is sought
 * @return int The signum of val.
 */
template< typename T > 
int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
sgn(T const& val) {
    return (static_cast<T>(0)<val) - (static_cast<T>(0)>val) ; 
}
namespace detail {
/**
 * @brief Specialization of max function for doubles.
 * \ingroup utils
 * \cond grace_detail
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
dmax(double const& A, double const& B){
  return (B<A) ? A : B ; 
}
/**
 * @brief Specialization of min function for doubles.
 * \ingroup utils
 * \cond grace_detail
 */
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
dmin(double const& A, double const& B){
  return (B<A) ? B : A ; 
}
}
/**
 * @brief Maximum between two numeric values.
 * \ingroup utils
 * 
 * @tparam T Type of the numeric values
 * @param A Value A.
 * @param B Value B. 
 * @return T The max between A and B.
 */
template< typename T > 
T GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
max(T const& A, T const& B) {
  return (A>B) ? A : B ; 
}
/**
 * @brief Maximum of a set of floating point values.
 * \ingroup utils
 * 
 * @tparam Arg_t Type(s) of other arguments.
 * @param A Value A.
 * @param B Value B.
 * @param args Other arguments.
 * @return double The max between A,B and other arguments.
 * NB: This function \b will downcast any <code>long double</code> 
 *     passed into it.
 */
template < typename ... Arg_t > 
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
max(double const& A, double const& B, Arg_t && ... args ) {
  if constexpr ( sizeof...(Arg_t) == 0 ) {
    return detail::dmax(A,B) ; 
  } else {
    return detail::dmax(A, max(B,args...)) ; 
  }
}
/**
 * @brief Minimum between two numeric values.
 * \ingroup utils
 * 
 * @tparam T Type of the numeric values
 * @param A Value A.
 * @param B Value B. 
 * @return T The min between A and B.
 */
template< typename T > 
T GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
min(T const& A, T const& B) {
  return (A<B) ? A : B ; 
}
/**
 * @brief Minimum of a set of floating point values.
 * \ingroup utils
 * 
 * @tparam Arg_t Type(s) of other arguments.
 * @param A Value A.
 * @param B Value B.
 * @param args Other arguments.
 * @return double The min between A,B and other arguments.
 * NB: This function \b will downcast any <code>long double</code> 
 *     passed into it.
 */
template < typename ... Arg_t > 
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
min(double const& A, double const& B, Arg_t && ... args ) {
  if constexpr ( sizeof...(Arg_t) == 0 ) {
    return detail::dmin(A,B) ; 
  } else {
    return detail::dmin(A, max(B,args...)) ; 
  }
}
/**
 * @brief Clamp a value between a max and a min.
 * \ingroup utils
 * 
 * @tparam T Type of the value.
 * @param val The value being clamped.
 * @param vmin The minimum allowed value for <code>val</code>.
 * @param vmax The maximum allowed value for <code>val</code>.
 * @return T The clamped value.
 */
template< typename T>
T GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
clamp(T const& val, T const& vmin, T const& vmax)
{
  return max(vmin, min(vmax, val)) ; 
}

template< typename T >
int GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
floor_int(T const& val)
{
  #ifdef __CUDA_ARCH__
    return static_cast<int>(floor(val));
  #elif defined(__HIP_DEVICE_COMPILE__)
    return static_cast<int>(floor(val));
  #else
    return static_cast<int>(std::floor(val));
  #endif
}

}

#endif /* GRACE_UTILS_MATH_HH */
