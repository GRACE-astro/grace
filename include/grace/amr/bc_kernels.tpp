/**
 * @file bc_kernels.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-23
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
#ifndef GRACE_AMR_BC_KERNELS_TPP 
#define GRACE_AMR_BC_KERNELS_TPP

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/data_structures/macros.hh>

namespace grace { namespace amr {
template< size_t order >
struct extrap_bc_t 
{
      template< typename ViewT >
      static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply(
            ViewT& u,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) ; 
} ; 

template<>
struct extrap_bc_t<0>
{
      template< typename ViewT >
      static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply(
            ViewT& u,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) 
      {
            u(VEC(i,j,k), q) = u(VEC(i-dx,j-dy,k-dz),q) ; 
      }; 
} ; 

template<>
struct extrap_bc_t<3>
{
      template< typename ViewT >
      static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply(
            ViewT& u,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) 
      {
            u(VEC(i,j,k), q) =( 4*u(VEC(i-dx,j-dy,k-dz),q) 
                              - 6*u(VEC(i-2*dx,j-2*dy,k-2*dz),q) 
                              + 4*u(VEC(i-3*dx,j-3*dy,k-3*dz),q) 
                              -   u(VEC(i-4*dx,j-4*dy,k-4*dz),q) ); 
      }; 
} ;

/**
 * @brief Apply outgoing boundary conditions.
 * \ingroup amr
 */
using outgoing_bc_t = extrap_bc_t<0> ; 


}}

#endif /* GRACE_AMR_BC_KERNELS_TPP */