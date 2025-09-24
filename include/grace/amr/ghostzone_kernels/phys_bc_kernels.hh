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
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
      ) const ; 
} ; 

template<>
struct extrap_bc_t<0>
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
      ) const
      {
            view(VEC(i,j,k)) = view(VEC(i-dx,j-dy,k-dz)) ; 
      }; 
} ; 


template<>
struct extrap_bc_t<3>
{
      template< typename view_t >
      void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
      apply (
            view_t view,
            VEC( size_t i, size_t j, size_t k),
            VEC( int8_t dx, int8_t dy, int8_t dz)
        ) const
      {
            view(VEC(i,j,k)) =( 4*view(VEC(i-dx,j-dy,k-dz)) 
                              - 6*view(VEC(i-2*dx,j-2*dy,k-2*dz)) 
                              + 4*view(VEC(i-3*dx,j-3*dy,k-3*dz)) 
                              -   view(VEC(i-4*dx,j-4*dy,k-4*dz)) ); 
      }; 
} ;

/**
 * @brief Apply outgoing boundary conditions.
 * \ingroup amr
 */
using outflow_bc_t = extrap_bc_t<0> ;


template< element_kind_t elem_kind, element_kind_t bc_kind, typename view_t >
struct phys_bc_op {

    readonly_view_t<std::size_t> qid   ;
    readonly_view_t<uint8_t> eid       ;
    readonly_twod_view_t<int8_t,3> dir ;
    readonly_view_t<bc_t> var_bcs      ; 
    
    view_t data ; 

    outflow_bc_t outflow_kernel ;
    extrap_bc_t<3> extrap_kernel ; 

    // only one view involved, if nx needs to be 
    // halved, just do it here 
    size_t n, ngz ; 

    void set_data_ptr(view_alias_t alias) {
        data = alias.get() ; 
    }


    phys_bc_op(
        view_t _data ,
        Kokkos::View<size_t*> _qid, Kokkos::View<uint8_t*> _eid, 
        Kokkos::View<int8_t*[3]> _dir, Kokkos::View<bc_t*> _var_bcs, 
         VEC(size_t _nx, size_t _ny, size_t _nz), size_t _ngz, bool _is_cbuf
    ) : qid(_qid),  eid(_eid), dir(_dir), var_bcs(_var_bcs), data(_data), 
        n(_nx), ngz(_ngz)
    {
        outflow_kernel = outflow_bc_t{} ; extrap_kernel = extrap_bc_t<3>{} ;
    }
    
    KOKKOS_INLINE_FUNCTION 
    void compute_zero_dir(size_t& lmin, size_t& lmax, size_t& idir, uint8_t dir_idx, uint8_t eid) const {
        idir = +1;
        if constexpr (elem_kind == element_kind_t::CORNER) 
        {
            lmin = ((eid>>dir_idx) & 1) ? n + ngz : 0 ; 

            lmax = lmin + ngz;
        } else if constexpr ((elem_kind == element_kind_t::EDGE) && (bc_kind == element_kind_t::FACE)) { 
            // supercalifragilistichespiralidoso 
            
            if(eid/4 == dir_idx) { 
                // along-edge → full sweep
                lmin = ngz; lmax = n + ngz; idir = +1;
            } else {
                // perpendicular → ghost only
                if(eid < 4) {          // X-axis edges
                    lmin = ((eid>>((dir_idx+1)%2))&1) ? n + ngz : 0;
                } else if(eid < 8) {   // Y-axis edges
                    lmin = ((eid>>(dir_idx/2))&1) ? n + ngz : 0;
                } else {               // Z-axis edges
                    lmin = ((eid>>dir_idx)&1) ? n + ngz : 0;
                }
                lmax = lmin + ngz;
            }       

        } else {
            lmin = ngz; lmax = n + ngz ;
        }
    }
    
    KOKKOS_INLINE_FUNCTION 
    void compute_bounds_impl(int8_t dir, size_t& lmin, size_t& lmax, size_t& idir, uint8_t idx, uint8_t eid) const
    {
      if (dir < 0) {
        lmin = ngz - 1; lmax = -1; idir = -1;
      } else if (dir > 0) {
        lmin = n + ngz; lmax = n + 2 * ngz; idir = +1;
      } else {
        // anche se ti sembra che abbia un suono spaventoso 
        compute_zero_dir(lmin,lmax,idir,idx,eid) ;  
      }
    };

    KOKKOS_INLINE_FUNCTION 
    void compute_bounds(const int8_t dir[3], size_t lmin[3], size_t lmax[3], size_t idir[3], uint8_t eid) const
    {
        #pragma unroll 
        for( int ii=0; ii<3; ++ii) compute_bounds_impl(dir[ii],lmin[ii],lmax[ii],idir[ii],ii,eid) ; 
    };

    KOKKOS_INLINE_FUNCTION
    void operator() (
        std::size_t k, std::size_t iv, std::size_t iq 
    ) const 
    {
        auto _eid = eid(iq) ; 
        auto _qid = qid(iq)  ; 
        auto bc_kind = var_bcs(iv) ;         
        
        int8_t _dir[] = {dir(iq,0), dir(iq,1), dir(iq,2)} ; 
        // se lo dici forte avrà un successo strepitoso 
        size_t lmin[3], lmax[3], idir[3];
        compute_bounds(_dir, lmin, lmax, idir, _eid);

        auto sv = Kokkos::subview(
            data, 
            VEC(Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL()),
            iv, _qid 
        ) ; 

        
        // loop not unrollable 
        for (int ig = lmin[0]; ig != lmax[0]; ig += idir[0])
        for (int jg = lmin[1]; jg != lmax[1]; jg += idir[1])
        for (int kg = lmin[2]; kg != lmax[2]; kg += idir[2]) {
            switch (bc_kind) {
                case BC_OUTFLOW:
                    outflow_kernel.template apply<decltype(sv)>(
                        sv, VEC(ig,jg,kg), VEC(_dir[0], _dir[1], _dir[2]));
                    break;

                case BC_LAGRANGE_EXTRAP:
                    extrap_kernel.template apply<decltype(sv)>(
                        sv, VEC(ig,jg,kg), VEC(_dir[0], _dir[1], _dir[2]));
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
// supercalifragilistichespiralidoso! 
#endif /* GRACE_AMR_PHYS_BC_KERNELS_HH */