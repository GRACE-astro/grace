/**
 * @file index_helper.hh
 * @author  (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-07-08
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#ifndef GRACE_AMR_INDEX_HELPERS_HH
#define GRACE_AMR_INDEX_HELPERS_HH

#include <grace_config.h>

#include <grace/utils/device/device.h>
#include <grace/utils/inline.h>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/amr/p4est_headers.hh>

#include <Kokkos_Core.hpp>

#include <array>

namespace grace {
#ifdef GRACE_3D
template< typename View_t >
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
fill_transfer_matrix_spherical(View_t& transfer_matrix)
{
    #define FILL_MATRIX(a,b,a00,a01,a02,a10,a11,a12,a20,a21,a22)\
    transfer_matrix(a,b,0,0) = a00 ; \
    transfer_matrix(a,b,0,1) = a01 ; \
    transfer_matrix(a,b,0,2) = a02 ; \
    transfer_matrix(a,b,1,0) = a10 ; \
    transfer_matrix(a,b,1,1) = a11 ; \
    transfer_matrix(a,b,1,2) = a12 ; \
    transfer_matrix(a,b,2,0) = a20 ; \
    transfer_matrix(a,b,2,1) = a21 ; \
    transfer_matrix(a,b,2,2) = a22 

    // All to identity...
    for( int i=0; i<2*P4EST_FACES + 1;++i){
        for( int j=0; j<2*P4EST_FACES + 1;++j){
            FILL_MATRIX(i,j, 1,0,0,0,1,0,0,0,1); 
        }
    }
    // ... and then set the nontrivial ones by hand
    // 1 0 and 0 1
    FILL_MATRIX(0,1, -1,0,0,0,0,1,0,1,0);
    FILL_MATRIX(1,0, -1,0,0,0,0,1,0,1,0);
    // 0 2 
    FILL_MATRIX(0,2, 1,0,0,0,1,0,0,0,1);
    FILL_MATRIX(2,0, 1,0,0,0,1,0,0,0,1);    
    // 0 3
    FILL_MATRIX(0,3, 0,1,0,-1,0,0,0,0,1);
    FILL_MATRIX(3,0, 0,-1,0,1,0,0,0,0,1);
    // 0 4 
    FILL_MATRIX(0,4, 0,0,1,1,0,0,0,1,0);
    FILL_MATRIX(4,0, 0,1,0,0,0,1,1,0,0);
    // 0 5 
    FILL_MATRIX(0,5, 0,0,1,0,1,0,-1,0,0);
    FILL_MATRIX(5,0, 0,0,-1,0,1,0,1,0,0);
    // 0 6 
    FILL_MATRIX(0,6, 0,1,0,0,0,1,1,0,0);
    FILL_MATRIX(6,0, 0,0,1,1,0,0,0,1,0);
    #undef FILL_MATRIX
}
#else 
template< typename View_t >
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
fill_transfer_matrix_spherical(View_t& transfer_matrix)
{
    #define FILL_MATRIX(a,b,a00,a01,a10,a11)\
    transfer_matrix(a,b,0,0) = a00 ; \
    transfer_matrix(a,b,0,1) = a01 ; \
    transfer_matrix(a,b,1,0) = a10 ; \
    transfer_matrix(a,b,1,1) = a11 

    // All to identity...
    for( int i=0; i<2*P4EST_FACES + 1+1;++i){
        for( int j=0; j<2*P4EST_FACES + 1+1;++j){
            FILL_MATRIX(i,j, 1,0,0,0,1,0,0,0,1); 
        }
    }

    FILL_MATRIX(0,1, -1,0,0,1) ; 
    FILL_MATRIX(1,0  -1,0,0,1) ; 

    FILL_MATRIX(0,2, 1,0,0,1) ; 
    FILL_MATRIX(2,0  1,0,0,1) ; 

    FILL_MATRIX(0,3, 0,1,-1,0) ; 
    FILL_MATRIX(3,0  0,-1,1,0) ; 

    FILL_MATRIX(0,4, 0,1,1,0)  ; 
    FILL_MATRIX(4,0  0,1,1,0) ; 

    FILL_MATRIX(1,3, 1,0,0,-1) ;
    FILL_MATRIX(3,1, 1,0,0,-1) ;

    FILL_MATRIX(1,4, 1,0,0,1) ;
    FILL_MATRIX(4,1, 1,0,0,1) ;

    FILL_MATRIX(2,3, 1,0,0,1) ;
    FILL_MATRIX(3,2, 1,0,0,1) ;

    #undef FILL_MATRIX
}
#endif 

struct index_helper_t {

using index_t = std::array<int64_t, GRACE_NSPACEDIM>;

GRACE_HOST_DEVICE 
index_helper() 
{
    #ifdef GRACE_SPHERICAL_COORDINATES 
    int const ntrees = 2*P4EST_FACES + 1 ; 
    index_transfer_matrix = Kokkos::View<int **[GACE_NSPACEDIM][GACE_NSPACEDIM], grace::default_space>{
        "index_transfer", ntrees,ntrees
    } ; 
    fill_transfer_matrix_spherical(index_transfer_matrix) ; 
    #endif 
}

index_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
operator() (index_t const& ijk, int64_t tid_a, int64_t tid_b, index_t const& ncells)
const {
    if( tid_a == tid_b ) {
        return ijk ; 
    }

    auto transfer = Kokkos::subview(index_transfer_matrix,tid_a,tid_b, Kokkos::ALL(),Kokkos::ALL()) ;

    index_t lmn{VEC(0,0,0)} ;

    #pragma unroll GRACE_NSPACEDIM*GRACE_NSPACEDIM
    for( int i=0; i<GRACE_NSPACEDIM;++i) for( int j=0; j<GRACE_NSPACEDIM; ++j) {
        lmn[i] += (transfer(i,j) < 0) ? ( ncells[j] + transfer(i,j) * ijk[j] )
                                      : ( transfer(i,j) * ijk[j]             ) ; 
    }
    return std::move(lmn) ; 
}

Kokkos::View<int **[GACE_NSPACEDIM][GACE_NSPACEDIM], grace::default_space>
    index_transfer_matrix ; 

} ; 

}

#endif 