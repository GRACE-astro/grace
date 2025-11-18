/**
 * @file reduction_utils.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-17
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

#ifndef GRACE_UTILS_REDUCTION_UTILS_HH
#define GRACE_UTILS_REDUCTION_UTILS_HH

#include <Kokkos_Core.hpp>

namespace grace {

template< class ScalarType, int N >
struct array_type {
     ScalarType the_array[N];
  
     KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
     array_type() { 
       for (int i = 0; i < N; i++ ) { the_array[i] = 0; }
     }
     KOKKOS_INLINE_FUNCTION   // Copy Constructor
     array_type(const array_type & rhs) { 
        for (int i = 0; i < N; i++ ){
           the_array[i] = rhs.the_array[i];
        }
     }
     KOKKOS_INLINE_FUNCTION   // add operator
     array_type& operator += (const array_type& src) {
       for ( int i = 0; i < N; i++ ) {
          the_array[i]+=src.the_array[i];
       }
       return *this;
     }
};

}

#define SPECIALIZE_KOKKOS_TEMPLATE(n)\
template<>\
struct reduction_identity< grace::array_type<double,n> > {\
    KOKKOS_FORCEINLINE_FUNCTION static grace::array_type<double,n> sum() {\
        return grace::array_type<double,n>();\
    }\
}

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
SPECIALIZE_KOKKOS_TEMPLATE(1);
SPECIALIZE_KOKKOS_TEMPLATE(2);
SPECIALIZE_KOKKOS_TEMPLATE(3);
SPECIALIZE_KOKKOS_TEMPLATE(4);
SPECIALIZE_KOKKOS_TEMPLATE(5);
SPECIALIZE_KOKKOS_TEMPLATE(6);
SPECIALIZE_KOKKOS_TEMPLATE(7);
SPECIALIZE_KOKKOS_TEMPLATE(8);
SPECIALIZE_KOKKOS_TEMPLATE(9);
SPECIALIZE_KOKKOS_TEMPLATE(10);
}

#undef SPECIALIZE_KOKKOS_TEMPLATE



#endif 