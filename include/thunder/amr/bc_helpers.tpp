/**
 * @file bc_helpers.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-21
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

#ifndef THUNDER_AMR_BC_HELPERS_TPP
#define THUNDER_AMR_BC_HELPERS_TPP

namespace thunder { namespace amr {

template< typename ViewAT
        , typename ViewBT >
void exchange_interior_boundary(
    ViewAT& viewA,
    ViewBT& viewB,
    size_t faceA,
    size_t faceB,
    size_t iqA,
    size_t iqB,
    VEC(nx,ny,nz),
    size_t ngz ) 
{   
    auto& a_interior = Kokkos::subview(
        viewA 
      , VEC( Kokkos::pair(ngz,2*ngz)
           , Kokkos::pair(ngz,2*ngz)
           , Kokkos::pair(ngz,2*ngz) )
      , iqA
      , Kokkos::ALL()
    ) ; 
    auto& b_interior = Kokkos::subview(
        viewA 
      , VEC( Kokkos::pair(ngz,2*ngz)
           , Kokkos::pair(ngz,2*ngz)
           , Kokkos::pair(ngz,2*ngz) )
      , iqB
      , Kokkos::ALL()
    ) ; 

    auto& a_exterior = Kokkos::subview(
        viewA 
      , VEC( Kokkos::pair(ngz,2*ngz)
           , Kokkos::pair(ngz,2*ngz)
           , Kokkos::pair(ngz,2*ngz) )
      , iqA
      , Kokkos::ALL()
    ) ; 
    auto& b_exterior = Kokkos::subview(
        viewA 
      , VEC( Kokkos::pair(ngz,2*ngz)
           , Kokkos::pair(ngz,2*ngz)
           , Kokkos::pair(ngz,2*ngz) )
      , iqB
      , Kokkos::ALL()
    ) ;

}

}}

#endif /* THUNDER_AMR_BC_HELPERS_TPP */