/**
 * @file puncture_tracker.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-01-27
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
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
#ifndef GRACE_UTILS_REDUCTIONS_HH
#define GRACE_UTILS_REDUCTIONS_HH

#include <Kokkos_Core.hpp>

namespace grace {

template< typename T, int N >
struct array_sum_t {
    T data[N] ; 

    KOKKOS_INLINE_FUNCTION 
    array_sum_t() {
        for( int i=0; i<N; ++i ) data[i] = 0.0 ; 
    }

    KOKKOS_INLINE_FUNCTION 
    array_sum_t( const array_sum_t & other ) {
        for( int i=0; i<N; ++i ) data[i] = other.data[i] ; 
    }

    KOKKOS_INLINE_FUNCTION
    array_sum_t& operator += (const array_sum_t& other) {
        for( int i=0; i<N; ++i ) data[i] += other.data[i] ; 
        return *this ;
    }
} ;

struct block_dist_t {
    double min_d0, min_d1 ;  // closest cell to each CO

    KOKKOS_INLINE_FUNCTION
    block_dist_t() : 
        min_d0(DBL_MAX), min_d1(DBL_MAX) {}

    KOKKOS_INLINE_FUNCTION
    block_dist_t( block_dist_t const& other) = default ; 

    KOKKOS_INLINE_FUNCTION
    block_dist_t& operator+=(const block_dist_t& other) {
        min_d0 = Kokkos::fmin(min_d0, other.min_d0) ;
        min_d1 = Kokkos::fmin(min_d1, other.min_d1) ;
        return *this ;
    }
} ;

template< typename T >
using pair_sum_t = array_sum_t<T,2> ; 

template< typename T >
using quad_sum_t = array_sum_t<T,4> ; 

}

namespace Kokkos {
    template<> 
    struct reduction_identity< grace::block_dist_t > {
        KOKKOS_FORCEINLINE_FUNCTION static grace::block_dist_t sum() {
            return grace::block_dist_t() ; 
        }
    };
}
#define INSTANTIATE_CUSTOM_REDUCED_TYPE(T)\
namespace Kokkos {\
template<> \
struct reduction_identity< grace::pair_sum_t<T> > {\
    KOKKOS_FORCEINLINE_FUNCTION static grace::pair_sum_t<T> sum() {\
        return grace::pair_sum_t<T>() ; \
    }\
};\
template<> \
struct reduction_identity< grace::array_sum_t<T,8> > {\
    KOKKOS_FORCEINLINE_FUNCTION static grace::array_sum_t<T,8> sum() {\
        return grace::array_sum_t<T,8>() ; \
    }\
};\
template<> \
struct reduction_identity< grace::array_sum_t<T,3> > {\
    KOKKOS_FORCEINLINE_FUNCTION static grace::array_sum_t<T,3> sum() {\
        return grace::array_sum_t<T,3>() ; \
    }\
};\
template<> \
struct reduction_identity< grace::array_sum_t<T,4> > {\
    KOKKOS_FORCEINLINE_FUNCTION static grace::array_sum_t<T,4> sum() {\
        return grace::array_sum_t<T,4>() ; \
    }\
};\
template<> \
struct reduction_identity< grace::array_sum_t<T,6> > {\
    KOKKOS_FORCEINLINE_FUNCTION static grace::array_sum_t<T,6> sum() {\
        return grace::array_sum_t<T,6>() ; \
    }\
};\
}

INSTANTIATE_CUSTOM_REDUCED_TYPE(double)
INSTANTIATE_CUSTOM_REDUCED_TYPE(int)
INSTANTIATE_CUSTOM_REDUCED_TYPE(size_t)

#undef INSTANTIATE_CUSTOM_REDUCED_TYPE

#endif /* GRACE_UTILS_REDUCTIONS_HH */