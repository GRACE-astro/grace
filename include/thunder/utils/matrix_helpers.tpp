/**
 * @file matrix_helpers.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-04-14
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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
#ifndef THUNDER_UTILS_MATRIX_HELPERS_TPP 
#define THUNDER_UTILS_MATRIX_HELPERS_TPP

#include <thunder_config.h>

#include <thunder/utils/inline.h>
#include <thunder/utils/device.h>

#include <array> 
#include <cstdlib> 

namespace utils {

/**
 * @brief Return the NxN identity matrix.
 * \ingroup utils
 * 
 * @tparam N Number of rows and columns of the matrix.
 * @return std::array<double, N*N> The identity matrix.
 */
template< size_t N > 
std::array<double, N*N> THUNDER_ALWAYS_INLINE 
identity_matrix() {
    std::array<double,N*N> id{0.}; 
    for(int i=0; i<N;++i) id[N*i+i] = 1. ; 
    return std::move(id); 
}

namespace detail {
/**
 * @brief Helper struct for small matrix determinant computation.
 * \ingroup utils
 * \cond thunder_detail 
 * @tparam N Size of the matrix. So far supported up to 3.
 */
template<size_t N>
struct det_impl_t {
    double THUNDER_ALWAYS_INLINE 
    operator() (std::array<double, N*N> const&) ; 
} ;
/**
 * @brief Helper struct for small matrix determinant computation.
 * \ingroup utils
 * \cond thunder_detail 
 * @tparam N Size of the matrix. So far supported up to 3.
 */
template<>
struct det_impl_t<1UL> {
    double THUNDER_ALWAYS_INLINE 
    operator() (std::array<double, 1UL> const& A) {
        return A[0] ; 
    }; 
} ;
/**
 * @brief Helper struct for small matrix determinant computation.
 * \ingroup utils
 * \cond thunder_detail 
 * @tparam N Size of the matrix. So far supported up to 3.
 */
template<>
struct det_impl_t<2UL> {
    double THUNDER_ALWAYS_INLINE 
    operator() (std::array<double, 4UL> const& A) {
        static constexpr int N = 2 ;
        return A[0 + N*0] * A[1 + N*1] - A[0 + N*1] * A[1 + N*0] ; 
    }; 
} ;
/**
 * @brief Helper struct for small matrix determinant computation.
 * \ingroup utils
 * \cond thunder_detail 
 * @tparam N Size of the matrix. So far supported up to 3.
 */
template<>
struct det_impl_t<3UL> {
    double THUNDER_ALWAYS_INLINE 
    operator() (std::array<double, 9UL> const& A) {
        static constexpr int N = 3 ; 
        return A[0 + 0*N] * det_impl_t<2UL>{}({A[1 + 1*N], A[2 + 1*N],
                                               A[1 + 2*N], A[2 + 2*N]}) 
             - A[1 + 0*N] * det_impl_t<2UL>{}({A[0 + 1*N], A[2 + 1*N],
                                               A[0 + 2*N], A[2 + 2*N]}) 
             + A[2 + 0*N] * det_impl_t<2UL>{}({A[0 + 1*N], A[1 + 1*N],
                                               A[0 + 2*N], A[1 + 2*N]}) ;
    }; 
} ;

}
/**
 * @brief Compute the determinant of a small square matrix.
 * \ingroup utils
 * @tparam N Number of rows and columns of the matrix (must be <=3)
 * @param A  Matrix. 
 * @return double The determinant of the matrix.
 */
template< size_t N > 
double THUNDER_ALWAYS_INLINE 
det(std::array<double,N*N> const& A) {
    return detail::det_impl_t<N>{}(A) ; 
}

template< size_t N = THUNDER_NSPACEDIM>
int THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE
delta(int i, int j) { return i==j ; }

} /* namespace utils */

#endif /* THUNDER_UTILS_MATRIX_HELPERS_TPP */