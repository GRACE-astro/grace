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

namespace grace { namespace amr {
template< size_t order >
struct extrap_bc_t 
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply(
            ViewT& dst, ViewT& src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) ; 
} ; 

template<>
struct extrap_bc_t<0>
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply(
            ViewT& dst, ViewT& src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) 
      {
            dst(VEC(i,j,k), q) = src(VEC(i-dx,j-dy,k-dz),q) ; 
      }; 
} ; 

template<>
struct extrap_bc_t<3>
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply(
            ViewT& dst, ViewT& src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) 
      {
            dst(VEC(i,j,k), q) =( 4*src(VEC(i-dx,j-dy,k-dz),q) 
                              - 6*src(VEC(i-2*dx,j-2*dy,k-2*dz),q) 
                              + 4*src(VEC(i-3*dx,j-3*dy,k-3*dz),q) 
                              -   src(VEC(i-4*dx,j-4*dy,k-4*dz),q) ); 
      }; 
} ;

/**
 * @brief Apply outgoing boundary conditions.
 * \ingroup amr
 */
using outgoing_bc_t = extrap_bc_t<0> ; 


struct sommerfeld_bc_t {

      grace::scalar_array_t<GRACE_NSPACEDIM> idx ; 
      grace::coord_array_t<GRACE_NSPACEDIM>  pcoords ; 
      double dt, dtfact, f0, v0

      sommerfeld_bc_t(
            grace::scalar_array_t<GRACE_NSPACEDIM>  _idx, 
            grace::coord_array_t<GRACE_NSPACEDIM>  _pcoords,
            double _dt,
            double _dtfact,
            double _f0,
            double _v0
      )
       : idx(_idx), _pcoords(pcoords), dt(_dt), dtfact(_dtfact), f0(_f0), v0(_v0)
      {}

      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply(
            ViewT& dst, ViewT& src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q, 
      )  
      {
            double derivx, derivy, derivz ; 
            
            double const xi = pcoords(VEC(i,j,k),0,q) ; 
            double const yi = pcoords(VEC(i,j,k),1,q) ; 
            double const zi = pcoords(VEC(i,j,k),2,q) ; 

            double const ri = Kokkos::sqrt(xi*xi + yi*yi + zi*zi) ; 
            double const rinv = 1./ri ; 

            double const vx = v0 * xi * rinv ; 
            double const vy = v0 * yi * rinv ; 
            double const vz = v0 * zi * rinv ; 

            if ( dx == 0 ) {
                  derivx = (src(VEC(i+1,j,k),q) - src(VEC(i-1,j,k),q) ) * idx(0,q) * 0.5 ; 
            } else {
                  derivx = dx * 0.5 * (
                        3 * src(VEC(i,j,k),q) - 4 * src(VEC(i-dx,j,k),q) + src(VEC(i-2*dx,j,k),q)  
                  ) * idx(0,q)  ; 
            }

            if ( dy == 0 ) {
                  derivy = (src(VEC(i,j+1,k),q) - src(VEC(i,j-1,k),q) ) * idx(1,q) * 0.5 ; 
            } else {
                  derivy = dy * 0.5 * (
                        3 * src(VEC(i,j,k),q) - 4 * src(VEC(i,j-dy,k),q) + src(VEC(i,j-2*dy,k),q)  
                  ) * idx(1,q)  ; 
            }

            if ( dz == 0 ) {
                  derivz = (src(VEC(i,j,k+1),q) - src(VEC(i,j,k-1),q) ) * idx(2,q) * 0.5 ; 
            } else {
                  derivz = dz * 0.5 * (
                        3 * src(VEC(i,j,k),q) - 4 * src(VEC(i,j,k-dz),q) + src(VEC(i,j,k-2*dz),q)  
                  ) * idx(2,q)  ; 
            }

            dst(VEC(i,j,k),q) = src(VEC(i,j,k),q) + dt * dtfact * (
                  - vx*derivx - vy*derivy - vz*derivz - v0 * (src(VEC(i,j,k),q)-f0)*rinv 
            ) ; 

      }
} ; 

}}

#endif /* GRACE_AMR_BC_KERNELS_TPP */