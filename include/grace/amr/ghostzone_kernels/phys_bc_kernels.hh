/**
 * @file phys_bc_kernels.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-09-05
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
#ifndef GRACE_AMR_PHYS_BC_KERNELS_HH
#define GRACE_AMR_PHYS_BC_KERNELS_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/fd_utils.hh>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

template< size_t order >
struct extrap_bc_t 
{
      template< typename src_view_t, typename dst_view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            dst_view_t dst, src_view_t src,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
      ) const ; 
} ; 

template<>
struct extrap_bc_t<0>
{
      template< typename src_view_t, typename dst_view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            dst_view_t dst, src_view_t src,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
      ) const
      {
            dst(VEC(i,j,k)) = src(VEC(i-dx,j-dy,k-dz)) ; 
      }; 
} ; 


template<>
struct extrap_bc_t<3>
{
      template< typename src_view_t, typename dst_view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            dst_view_t dst, src_view_t src,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
        ) const
      {
            dst(VEC(i,j,k)) =( 4*src(VEC(i-dx,j-dy,k-dz)) 
                              - 6*src(VEC(i-2*dx,j-2*dy,k-2*dz)) 
                              + 4*src(VEC(i-3*dx,j-3*dy,k-3*dz)) 
                              -   src(VEC(i-4*dx,j-4*dy,k-4*dz)) ); 
      }; 
} ;

/**
 * @brief Apply outgoing boundary conditions.
 * \ingroup amr
 */
using outflow_bc_t = extrap_bc_t<0> ;

template< typename src_view_t, typename dst_view_t >
struct face_phys_bc_k {

    readonly_view_t<std::size_t> qid ;
    readonly_view_t<bc_t> var_bcs ; 
    readonly_view_t<uint8_t> face    ;

    src_view_t src_data ; 
    dst_view_t dst_data ; 

    outflow_bc_t outflow_kernel ;
    extrap_bc_t<3> extrap_kernel ; 

    std::size_t VEC(nx,ny,nz), ngz ; 


    face_phys_bc_k(
        dst_view_t _dst, src_view_t _src, 
        Kokkos::View<size_t*> _qid, Kokkos::View<bc_t*> _var_bcs, 
        Kokkos::View<uint8_t*> _face, VEC(size_t _nx, size_t _ny, size_t _nz), size_t _ngz 
    ) : qid(_qid), var_bcs(_var_bcs), face(_face), src_data(_src), dst_data(_dst), 
        VEC(nx(_nx), ny(_ny), nz(_nz)), ngz(_ngz)
    {
        outflow_kernel = outflow_bc_t{} ; extrap_kernel = extrap_bc_t<3>{} ;
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (
        std::size_t ig, std::size_t j, std::size_t k, std::size_t iv, std::size_t iq 
    ) const 
    {
        auto face_id = face(iq) ; 
        auto quad_id = qid(iq)  ; 
        auto bc_kind = var_bcs(iv) ; 
        auto faceb2 = face_id / 2 ; 

        static int8_t const dir[] = {
            static_cast<int8_t>((face_id == 0) ? -1 : (face_id == 1 ? +1 : 0)),
            static_cast<int8_t>((face_id == 2) ? -1 : (face_id == 3 ? +1 : 0)),
            static_cast<int8_t>((face_id == 4) ? -1 : (face_id == 5 ? +1 : 0))
        };
        
        // Compute sweep range in normal direction
        size_t lmin, lmax, idir;
        auto compute_bounds = [](int8_t face, size_t ngz, size_t n, size_t& lmin, size_t& lmax, size_t& idir) {
            if (face % 2 == 0) {  // negative side
                lmin = ngz - 1;
                lmax = -1;
                idir = -1;
            } else {              // positive side
                lmin = n + ngz;
                lmax = n + 2 * ngz;
                idir = +1;
            }
        };

        compute_bounds(face_id, ngz, (faceb2 == 0 ? nx : (faceb2 == 1 ? ny : nz)), lmin, lmax, idir);

        auto src_sv = Kokkos::subview(
            src_data, 
            VEC(Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
            iv, quad_id 
        ) ; 

        auto dst_sv = Kokkos::subview(
            dst_data, 
            VEC(Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
            iv, quad_id 
        ) ;
 
        for (size_t ig = lmin; ig != lmax; ig += idir) {
            // Reconstruct full 3D index depending on face orientation
            auto const I = (faceb2 == 0) ? ig : j;
            auto const J = (faceb2 == 1) ? ig : (faceb2 == 0 ? j : k);
            auto const K = (faceb2 == 2) ? ig : k;

            switch (bc_kind) {
                case BC_OUTFLOW:
                    outflow_kernel.template apply<decltype(dst_sv), decltype(src_sv)>(
                        dst_sv, src_sv, VEC(I, J, K), VEC(dir[0], dir[1], dir[2]));
                    break;

                case BC_LAGRANGE_EXTRAP:
                    extrap_kernel.template apply<decltype(dst_sv), decltype(src_sv)>(
                        dst_sv, src_sv, VEC(I, J, K), VEC(dir[0], dir[1], dir[2]));
                    break;
                case BC_NONE:
                    break;
                default:
                    // fallback or assert
                    break;
            }
        }
    }




} ; 

}} /* namespace grace::amr */

#endif /* GRACE_AMR_PHYS_BC_KERNELS_HH */