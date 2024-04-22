/**
 * @file rotation_matrices.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
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

#ifndef THUNDER_AMR_ROTATION_MATRICES_HH 
#define THUNDER_AMR_ROTATION_MATRICES_HH

#include <thunder/errors/assert.hh>

namespace thunder { 
namespace detail {
namespace {

#ifdef THUNDER_3D 
static constexpr std::array<double,9> a0 = {-1,0,0,
                                            0,0,1,
                                            0,1,0} ;
static constexpr std::array<double,9> a1 = {1,0,0,
                                            0,1,0,
                                            0,0,1} ;
static constexpr std::array<double,9> a2 = {0,1,0,
                                            -1,0,0,
                                            0,0,1} ;
static constexpr std::array<double,9> a3 = {0,0,1,
                                            1,0,0,
                                            0,1,0} ;
static constexpr std::array<double,9> a4 = {0,0,1,
                                            0,1,0,
                                            -1,0,0} ;
static constexpr std::array<double,9> a5 = {0,1,0,
                                            0,0,1,
                                            1,0,0} ;
                                            
static constexpr std::array<double,9> a0i = {-1,0,0,
                                            0,0,1,
                                            0,1,0} ;
static constexpr std::array<double,9> a1i = {1,0,0,
                                            0,1,0,
                                            0,0,1} ;
static constexpr std::array<double,9> a2i = {0,-1,0,
                                            1,0,0,
                                            0,0,1} ;
static constexpr std::array<double,9> a3i = {0,1,0,
                                            0,0,1,
                                            1,0,0} ;
static constexpr std::array<double,9> a4i = {0,0,-1,
                                            0,1,0,
                                            1,0,0} ;
static constexpr std::array<double,9> a5i = {0,0,1,
                                            1,0,0,
                                            0,1,0} ;
static constexpr std::array<std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>,P4EST_FACES> rotation_matrices 
    = { a0,a1,a2,a3,a4,a5 } ;
static constexpr std::array<std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>,P4EST_FACES> inv_rotation_matrices 
    = { a0i,a1i,a2i,a3i,a4i,a5i } ;
#else 
static constexpr std::array<double,4> a0 = {-1,0,0,1} ;
static constexpr std::array<double,4> a1 = {1,0,0,1} ;
static constexpr std::array<double,4> a2 = {0,1,-1,0} ;
static constexpr std::array<double,4> a3 = {0,1,1,0} ;
static constexpr std::array<std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>,P4EST_FACES> rotation_matrices 
    = { a0,a1,a2,a3 } ;
static constexpr std::array<double,4> a0i = {-1,0,0,1} ;
static constexpr std::array<double,4> a1i = {1,0,0,1} ;
static constexpr std::array<double,4> a2i = {0,-1,1,0} ;
static constexpr std::array<double,4> a3i = {0,1,1,0} ;
static constexpr std::array<std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>,P4EST_FACES> inv_rotation_matrices 
    = { a0i,a1i,a2i,a3i } ;
#endif 

}

/**
 * @brief Apply a discrete rotation to a vector.
 * 
 * \ingroup thunder_coordinates 
 * \cond thunder_detail
 * @param _x 
 * @param dir 
 * @param inverse 
 * @return std::array<double,THUNDER_NSPACEDIM> 
 */
static std::array<double,THUNDER_NSPACEDIM> THUNDER_ALWAYS_INLINE 
apply_discrete_rotation(std::array<double,THUNDER_NSPACEDIM>const& _x, unsigned dir, bool inverse=false) 
{
    ASSERT_DBG(dir < P4EST_FACES, "Got rotation direction " << dir << " out of bounds.") ; 
    std::array<double, THUNDER_NSPACEDIM> _y {VEC(0,0,0)}; 
    auto const& matrices = inverse ? inv_rotation_matrices : rotation_matrices ; 
    #pragma unroll
    for(int idir=0; idir<THUNDER_NSPACEDIM; ++idir){
        for( int jdir=0; jdir<THUNDER_NSPACEDIM; ++jdir){
            _y[idir] += matrices[dir][idir*THUNDER_NSPACEDIM+jdir] * _x[jdir] ;
        }
    }
    return std::move(_y) ; 
}


}} /* namespace thunder::detail */

#endif /* THUNDER_AMR_ROTATION_MATRICES_HH */