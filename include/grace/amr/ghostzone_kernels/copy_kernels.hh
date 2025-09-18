/**
 * @file copy_kernels.hh
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
#ifndef GRACE_AMR_BC_COPY_GHOSTZONES_HH
#define GRACE_AMR_BC_COPY_GHOSTZONES_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {


template< 
    element_kind_t elem_kind,
    bool do_reciprocal_copy,
    typename ViewA_t,
    typename ViewB_t 
>
struct copy_k {

    ViewA_t src_view ; 
    ViewB_t dest_view; 
    readonly_view_t<std::size_t> src_qid, dest_qid ; 
    readonly_view_t<uint8_t> src_element_view, dest_element_view; 

    index_transformer_t transf ; 

    copy_k(
        ViewB_t _src_view,
        ViewA_t _dest_view,
        Kokkos::View<size_t*> _src_qid, Kokkos::View<size_t*> _dest_qid,
        Kokkos::View<uint8_t*> _src_elem, Kokkos::View<uint8_t*> _dest_elem, 
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz
    ) : src_view(_src_view)
      , dest_view(_dest_view)
      , src_qid(_src_qid)
      , dest_qid(_dest_qid)
      , src_element_view(_src_elem)
      , dest_element_view(_dest_elem)
      , transf(VEC(_nx,_ny,_nz),_ngz)
    {}


    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {
        auto const ie_src  = src_element_view(iq) ; 
        auto const ie_dest = dest_element_view(iq) ; 

        auto const src_q  = src_qid(iq)  ; 
        auto const dest_q = dest_qid(iq) ;

        std::size_t VEC(i_a,j_a,k_a), VEC(i_b,j_b,k_b) ; 
        transf.compute_indices<elem_kind,true>(
            ig, VECD(j, k), i_a, j_a, k_a, (int) ie_src
        ) ; 
        transf.compute_indices<elem_kind,false>(
            ig, VECD(j, k), i_b, j_b, k_b, ie_dest
        ) ; 
        dest_view(
            VEC(i_b,j_b,k_b), ivar, dest_q 
        ) = src_view(VEC(i_a,j_a,k_a), ivar, src_q) ;
        if constexpr( do_reciprocal_copy ){
            
            transf.compute_indices<elem_kind,true>(
            ig, VECD(j, k), i_a, j_a, k_a, ie_dest 
            ) ; 
            transf.compute_indices<elem_kind,false>(
                ig, VECD(j, k), i_b, j_b, k_b, ie_src
            ) ;
            src_view(
                VEC(i_b,j_b,k_b), ivar, src_q 
            ) = dest_view(VEC(i_a,j_a,k_a), ivar, dest_q) ;
        }
    }
} ; 

KOKKOS_INLINE_FUNCTION
int edge_to_face_dir(int face, int edge) {
    int edge_axis = edge / 4;    // 0=x,1=y,2=z
    int normal    = face / 2;    // 0=x,1=y,2=z

    // Collect the two tangential axes
    int t0 = (normal + 1) % 3;
    int t1 = (normal + 2) % 3;

    // Sort so j = min, k = max
    int j_axis = t0 < t1 ? t0 : t1;
    int k_axis = t0 < t1 ? t1 : t0;

    // Now map edge axis to j/k
    return (edge_axis == j_axis) ? 0 : 1;
}


template<
     element_kind_t view_elem_kind,
    element_kind_t cbuf_elem_kind,
>
struct cbuf_to_view_offsets {
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const ; 
} ; 

template<>
struct cbuf_to_view_offsets<element_kind_t::FACE,element_kind_t::FACE>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const {
        j = nx / 2 * ( (ichild<<0) & 1 ) ; 
        k = nx / 2 * ( (ichild<<1) & 1 ) ; 
    }; 
} ; 

template<>
struct cbuf_to_view_offsets<element_kind_t::EDGE,element_kind_t::EDGE>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const {
        j = 0 ; 
        k = nx / 2 * ( (ichild<<0) & 1 ) ; 
    }; 
} ; 

template<>
struct cbuf_to_view_offsets<element_kind_t::CORNER,element_kind_t::CORNER>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const {
        j = 0 ; 
        k = 0 ; 
    }; 
} ;

template<>
struct cbuf_to_view_offsets<element_kind_t::FACE,element_kind_t::CORNER>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const {
        j = nx / 2 - ( (ichild<<0) & 1 ) * ngz ; 
        k = nx / 2 - ( (ichild<<1) & 1 ) * ngz ; 
    }; 
} ; 

template<>
struct cbuf_to_view_offsets<element_kind_t::FACE, element_kind_t::EDGE>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const 
    {
        // Here: iedge=0 -> edge along j-direction
        //       iedge=1 -> edge along k-direction
        auto const iedge = edge_to_face_dir(ie_v,ie_c) ; 
        
        if ( iedge == 0 ) {
            j = ( ((ichild>>0) & 1) ? nx/2 : 0 )  ; 
            k = nx / 2 -  ngz * ((ichild>>1) & 1) ; 
        } else {
            k = ( ((ichild>>0) & 1) ? nx/2 : 0 )  ; 
            j = nx / 2 -  ngz * ((ichild>>1) & 1) ; 
        }
    }
};

template<>
struct cbuf_to_view_offsets<element_kind_t::EDGE, element_kind_t::CORNER>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const 
    {
        j = 0 ; 
        k = nx / 2 -  ngz * ichild ; 
    }
};

template< 
    element_kind_t view_elem_kind,
    element_kind_t cbuf_elem_kind,
    typename view_t,
    typename cbuf_t 
>
struct cbuf_copy_k {

    view_t view ; 
    cbuf_t cbuf ; 
    readonly_view_t<std::size_t> view_qid, cbuf_qid ; 
    readonly_view_t<uint8_t> elem_view, cbuf_elem_view, view_ic; 

    index_transformer_t transf ; 

    copy_k(
        view_t _view,
        cbuf_t _cbuf,
        Kokkos::View<size_t*> _view_qid, 
        Kokkos::View<size_t*> _cbuf_qid,
        Kokkos::View<uint8_t*> _elem_view, 
        Kokkos::View<uint8_t*> _cbuf_elem_view,
        Kokkos::View<uint8_t*> _ic_view, 
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz
    ) : view(_view)
      , cbuf(_cbuf)
      , elem_view(_view_qid)
      , cbuf_elem_view(_cbuf_qid)
      , view_ic(_ic_view)
      , elem_view(_elem_view)
      , cbuf_elem_view(_cbuf_elem_view)
      , transf(VEC(_nx,_ny,_nz),_ngz)
    {}


    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {
        auto const ie_view  = elem_view(iq) ; 
        auto const ie_cbuf = cbuf_elem_view(iq) ; 

        auto const view_q  = view_qid(iq)  ; 
        auto const cbuf_q = cbuf_qid(iq) ;

        // we need to offset into the coarse quad 
        ichild = view_ic(iq) ; 
        size_t j_off{0}, k_off{0} ; 
        cbuf_to_view_offsets<view_elem_kind,cbuf_elem_kind>::get(
            j_off, k_off, transf.nx, transf.ngz, ichild, ie_view, ie_cbuf 
        ) ; 

        std::size_t VEC(i_a,j_a,k_a), VEC(i_b,j_b,k_b) ; 
        // copy into cbuf's gzs
        transf.compute_indices<view_elem_kind,true>(
        ig, VECD(j + j_off, k + k_off), i_a, j_a, k_a, ie_view 
        ) ; 
        transf.compute_indices<cbuf_elem_kind,false>(
            ig, VECD(j, k), 
            i_b, j_b, k_b, ie_cbuf, /* halved ncells */ true 
        ) ;
        cbuf(
            VEC(i_b,j_b,k_b), ivar, cbuf_q 
        ) = view(VEC(i_a,j_a,k_a), ivar, view_q) ;
        
        if constexpr ( view_elem_kind == cbuf_elem_kind ) {
            // copy into normal view's gzs
            // only if same elem kind (face into face etc.)
            transf.compute_indices<cbuf_elem_kind,true>(
                ig, VECD(j, k), 
                i_a, j_a, k_a, ie_cbuf, /* halved ncells */ true 
            ) ; 
            transf.compute_indices<view_elem_kind,false>(
                ig, VECD(j + j_off, k + k_off), i_b, j_b, k_b, ie_view
            ) ; 
            view(
                VEC(i_b,j_b,k_b), ivar, view_q 
            ) = cbuf(VEC(i_a,j_a,k_a), ivar, cbuf_q) ;
        }
    }
} ; 
    
}} /* namespace grace::amr */


#endif /* GRACE_AMR_BC_COPY_GHOSTZONES_HH */