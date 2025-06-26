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
#include <grace/utils/numerics/fd_utils.hh>
namespace grace { namespace amr {
template< size_t order >
struct extrap_bc_t 
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            ViewT dst, ViewT src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) const ; 
} ; 

template< size_t order >
struct extrap_bc_t_fallback 
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            ViewT dst, ViewT src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) const ; 
} ; 

template< size_t order >
struct extrap_bc_diagonal_t
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            ViewT dst, ViewT src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) const ; 
} ; 

template< size_t order >
struct avg_bc_t 
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            ViewT dst, ViewT src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) const ; 
} ;

template< size_t order >
struct extrap_bc_hybrid_t
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            ViewT dst, ViewT src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) const ; 
} ; 
// --------------------------
template<>
struct extrap_bc_t<0>
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            ViewT dst, ViewT src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) const
      {
            dst(VEC(i,j,k), q) = src(VEC(i-dx,j-dy,k-dz),q) ; 
      }; 
} ; 

template<>
struct extrap_bc_t<3>
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            ViewT dst, ViewT src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) const
      {
            dst(VEC(i,j,k), q) = (
                4 * src(VEC(i-dx,j-dy,k-dz),q) 
              - 6 * src(VEC(i-2*dx,j-2*dy,k-2*dz),q) 
              + 4 * src(VEC(i-3*dx,j-3*dy,k-3*dz),q) 
              -     src(VEC(i-4*dx,j-4*dy,k-4*dz),q)
            );
      }
} ;

template<>
struct extrap_bc_t_fallback<3>
{
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            ViewT dst, ViewT src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q 
      ) const
      {
        int const dir_sum = abs(dx) + abs(dy) + abs(dz);

        if (dir_sum == 1) {
            // Pure face direction: safe to use full 3rd order stencil
            dst(VEC(i,j,k), q) = (
                4 * src(VEC(i-dx,j-dy,k-dz),q) 
              - 6 * src(VEC(i-2*dx,j-2*dy,k-2*dz),q) 
              + 4 * src(VEC(i-3*dx,j-3*dy,k-3*dz),q) 
              -     src(VEC(i-4*dx,j-4*dy,k-4*dz),q)
            );
        } else if (dir_sum == 2) {
            // Edge: use 1st order extrapolation to avoid diagonally invalid paths
            dst(VEC(i,j,k), q) = src(VEC(i-dx,j-dy,k-dz), q);
        } else if (dir_sum == 3) {
            // Corner: fallback to simple copy
            dst(VEC(i,j,k), q) = src(VEC(i-dx,j-dy,k-dz), q);
        } else {
            // Should not happen, but fallback safely
            dst(VEC(i,j,k), q) = src(VEC(i,j,k), q);
        }
    }
} ;


template<>
struct extrap_bc_diagonal_t<3>{

    template< typename ViewT >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    apply (
        ViewT dst,
        ViewT src,
        VEC( int i, int j, int k),
        VEC( int8_t dx, int8_t dy, int8_t dz), 
        int64_t q
    ) const {

        // Compute shifted stencil positions
        int i1 = i - dx,     j1 = j - dy,     k1 = k - dz;
        int i2 = i - 2*dx,   j2 = j - 2*dy,   k2 = k - 2*dz;
        int i3 = i - 3*dx,   j3 = j - 3*dy,   k3 = k - 3*dz;
        int i4 = i - 4*dx,   j4 = j - 4*dy,   k4 = k - 4*dz;

        // Perform 3rd order extrapolation along the multi-axis direction vector (dx, dy, dz)
        dst(VEC(i,j,k), q) = (
              4.0 * src(VEC(i1,j1,k1), q)
            - 6.0 * src(VEC(i2,j2,k2), q)
            + 4.0 * src(VEC(i3,j3,k3), q)
            -       src(VEC(i4,j4,k4), q));
    }
};

template<>
struct avg_bc_t<3>{

    template< typename ViewT >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    apply (
        ViewT dst,
        ViewT src,
        VEC( int i, int j, int k),
        VEC( int8_t dx, int8_t dy, int8_t dz), 
        int64_t q
    ) const {
        int count = 0;
        double value = 0.0;

        // Accumulate neighboring directions
        if (dx != 0) {
            value += src(VEC(i - dx, j, k), q);
            count++;
        }
        if (dy != 0) {
            value += src(VEC(i, j - dy, k), q);
            count++;
        }
        if (dz != 0) {
            value += src(VEC(i, j, k - dz), q);
            count++;
        }

        // Average
        if (count > 0) {
            dst(VEC(i,j,k), q) = value / count;
        } else {
            // Pure face: fallback to copy
            dst(VEC(i,j,k), q) = src(VEC(i - dx, j - dy, k - dz), q);
        }
    }
} ;

template<>
struct extrap_bc_hybrid_t<3>
{
    template< typename ViewT >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    apply (
        ViewT dst,
        ViewT src,
        VEC( int i, int j, int k),
        VEC( int8_t dx, int8_t dy, int8_t dz),
        int64_t q
    ) const
    {
        int dir_sum = abs(dx) + abs(dy) + abs(dz);

        if (dir_sum == 1) {
            // Face: full 3rd-order diagonal extrapolation is fine
            dst(VEC(i,j,k), q) = (
                4.0 * src(VEC(i - dx, j - dy, k - dz), q)
              - 6.0 * src(VEC(i - 2*dx, j - 2*dy, k - 2*dz), q)
              + 4.0 * src(VEC(i - 3*dx, j - 3*dy, k - 3*dz), q)
              -       src(VEC(i - 4*dx, j - 4*dy, k - 4*dz), q)
            );
        } else if (dir_sum == 2) {
            // Edge: fallback to 1st-order diagonal
            dst(VEC(i,j,k), q) = src(VEC(i - dx, j - dy, k - dz), q);
        } else if (dir_sum == 3) {
            // Corner: average face neighbors
            int count = 0;
            double val = 0.0;

            if (dx != 0) { val += src(VEC(i - dx, j, k), q); count++; }
            if (dy != 0) { val += src(VEC(i, j - dy, k), q); count++; }
            if (dz != 0) { val += src(VEC(i, j, k - dz), q); count++; }

            dst(VEC(i,j,k), q) = (count > 0) ? val / count : src(VEC(i,j,k), q);
        } else {
            // Safety net
            dst(VEC(i,j,k), q) = src(VEC(i,j,k), q);
        }
    }
};

/**
 * @brief Apply outgoing boundary conditions.
 * \ingroup amr
 */
using outgoing_bc_t = extrap_bc_t<0> ; 

using extrap_bc_Chosen_t = avg_bc_t<3>; //extrap_bc_t<3> ; 


struct sommerfeld_bc_t {

      grace::scalar_array_t<GRACE_NSPACEDIM> idx ; 
      grace::coord_array_t<GRACE_NSPACEDIM>  pcoords ; 
      double dt, dtfact, f0, v0 ; 
      int VEC(nx,ny,nz),ngz ; 

      sommerfeld_bc_t(
            grace::scalar_array_t<GRACE_NSPACEDIM>  _idx, 
            grace::coord_array_t<GRACE_NSPACEDIM>  _pcoords,
            double _dt,
            double _dtfact,
            double _f0,
            double _v0,
            VEC(int _nx, int _ny, int _nz), int _ngz
      )
       : idx(_idx), 
         pcoords(_pcoords), 
         dt(_dt), 
         dtfact(_dtfact), 
         f0(_f0), v0(_v0),
         VEC(nx(_nx), ny(_ny), nz(_nz)),
         ngz(_ngz)
      {}

 
      template< typename ViewT >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            ViewT dst, ViewT src,
            VEC( int i, int j, int k),
            VEC( int8_t dx, int8_t dy, int8_t dz), 
            int64_t q
      )  const 
      {
            double derivx, derivy, derivz ; 
            
            double const xi = pcoords(VEC(i,j,k),0,q) ; 
            double const yi = pcoords(VEC(i,j,k),1,q) ; 
            double const zi = pcoords(VEC(i,j,k),2,q) ; 

            double const ri = Kokkos::sqrt(xi*xi + yi*yi + zi*zi) ; 
            double const rinv = 1./ri ; 
            //double rinv = (ri > 1e-12) ? 1. / ri : 0.0;

            double const vx = v0 * xi * rinv ; 
            double const vy = v0 * yi * rinv ; 
            double const vz = v0 * zi * rinv ; 

            auto var = Kokkos::subview(src, VEC(Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()), q) ; 
            auto const fd_der_x = [&] ()
            {

                  return 0.5 * (var(VEC(i+1,j,k)) - var(VEC(i-1,j,k))) * idx(0,q) ; 
                  //return grace::detail::fd_der_bnd_check_recursive<1,1,0>::template doit<2>(var, VEC(i,j,k),VEC(nx,ny,nz),ngz)* idx(0,q);
            } ; 

            auto const fd_der_y = [&] ()
            {
                  return 0.5 * (var(VEC(i,j+1,k)) - var(VEC(i,j-1,k))) * idx(1,q) ; 
                  //return grace::detail::fd_der_bnd_check_recursive<1,1,1>::template doit<2>(var, VEC(i,j,k),VEC(nx,ny,nz),ngz)* idx(1,q); 
            } ; 

            auto const fd_der_z = [&] ()
            {
                  return 0.5 * (var(VEC(i,j,k+1)) - var(VEC(i,j,k-1))) * idx(2,q) ; 
                  //return grace::detail::fd_der_bnd_check_recursive<1,1,2>::template doit<2>(var, VEC(i,j,k),VEC(nx,ny,nz),ngz)* idx(2,q);
            } ; 

            if ( dx == 0 ) {
                  derivx = fd_der_x() ; 
            } else {
                  derivx = dx * 0.5 * (
                        3 * src(VEC(i,j,k),q) - 4 * src(VEC(i-dx,j,k),q) + src(VEC(i-2*dx,j,k),q)  
                  ) * idx(0,q)  ; 
            }

            if ( dy == 0 ) {
                  derivy = fd_der_y() ; 
            } else {
                  derivy = dy * 0.5 * (
                        3 * src(VEC(i,j,k),q) - 4 * src(VEC(i,j-dy,k),q) + src(VEC(i,j-2*dy,k),q)  
                  ) * idx(1,q)  ; 
            }

            if ( dz == 0 ) {
                  derivz = fd_der_z() ; 
            } else {
                  derivz = dz * 0.5 * (
                        3 * src(VEC(i,j,k),q) - 4 * src(VEC(i,j,k-dz),q) + src(VEC(i,j,k-2*dz),q)  
                  ) * idx(2,q)  ; 
            }
            /*
            if (i == 0 && j == 0 && k == 0) {
                  printf("[SOMMERFELD DEBUG] q=%ld (i,j,k)=(%d,%d,%d) x=(%.3e,%.3e,%.3e) |r|=%.3e "
                        "v=(%.3e,%.3e,%.3e) deriv=(%.3e,%.3e,%.3e) phi=%.3e res=%.3e update=%.3e\n",
                        q, i, j, k,
                        xi, yi, zi, ri,
                        vx, vy, vz,
                        derivx, derivy, derivz,
                        src(VEC(i,j,k),q),
                        (src(VEC(i,j,k),q) - f0) * rinv,
                        dt * dtfact * (
                        - vx * derivx - vy * derivy - vz * derivz
                        - v0 * (src(VEC(i,j,k),q) - f0) * rinv));
            }
            */

            dst(VEC(i,j,k),q) += dt * dtfact * (
                  - vx*derivx - vy*derivy - vz*derivz - v0 * (src(VEC(i,j,k),q)-f0)*rinv 
            ) ; 

      }
};

}}

#endif /* GRACE_AMR_BC_KERNELS_TPP */