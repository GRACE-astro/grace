/**
 * @file pack_unpack_kernels.hh
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
#ifndef GRACE_AMR_PACK_UNPACK_KERNELS_HH
#define GRACE_AMR_PACK_UNPACK_KERNELS_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/ghost_array.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

template< 
    element_kind_t elem_kind,
    typename view_t
>
struct pack_op {
    view_t src_view ; 
    ghost_array_t dest_view; 

    readonly_view_t<std::size_t> src_qid, dest_qid ; 
    readonly_view_t<uint8_t> src_elem_view ;
    
    std::size_t rank ; 

    index_transformer_t transf ; 

    void set_data_ptr(view_alias_t alias) {
        src_view = alias.get() ; 
    }

    pack_op(
        view_t _src_view,
        ghost_array_t _dest_view,
        Kokkos::View<size_t*> _src_qid, 
        Kokkos::View<size_t*> _dest_qid,
        Kokkos::View<uint8_t*> _src_elem,  
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank 
    ) : src_view(_src_view)
      , dest_view(_dest_view)
      , src_qid(_src_qid)
      , dest_qid(_dest_qid)
      , src_elem_view(_src_elem)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz)
    { } 


    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {
        auto const ie  = src_elem_view(iq) ; 

        auto const src_q  = src_qid(iq)  ; 
        auto const dest_q = dest_qid(iq) ;

        std::size_t VEC(i_a,j_a,k_a) ; 
        // phys indices 
        transf.compute_indices<elem_kind,true>(
            ig, VECD(j, k), i_a, j_a, k_a, ie
        ) ; 
        
        dest_view.at_interface<elem_kind>(ig,j,k,ivar,dest_q,rank) = 
            src_view(VEC(i_a,j_a,k_a), ivar, src_q) ;
        
    }
 
} ; 

template< 
    element_kind_t elem_kind,
    typename view_t
>
struct unpack_op {
    
    ghost_array_t src_view ; 
    view_t dest_view; 

    readonly_view_t<std::size_t> src_qid, dest_qid ; 
    readonly_view_t<uint8_t> dest_face_view ;
    
    std::size_t rank ; 

    index_transformer_t transf ; 

    void set_data_ptr(view_alias_t alias) {
        dest_view = alias.get() ; 
    }

    unpack_op(
        ghost_array_t _src_view,
        view_t _dest_view,
        Kokkos::View<size_t*> _src_qid, 
        Kokkos::View<size_t*> _dest_qid,
        Kokkos::View<uint8_t*> _dest_face,  
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank 
    ) : src_view(_src_view)
      , dest_view(_dest_view)
      , src_qid(_src_qid)
      , dest_qid(_dest_qid)
      , dest_face_view(_dest_face)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz)
    { } 


    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {
        auto const dest_face  = dest_face_view(iq) ; 

        auto const src_q  = src_qid(iq)  ; 
        auto const dest_q = dest_qid(iq) ;


        std::size_t VEC(i_a,j_a,k_a) ; 
        transf.compute_indices<elem_kind,false>(
            ig, VECD(j, k), i_a, j_a, k_a, dest_face 
        ) ; 

        dest_view(VEC(i_a,j_a,k_a), ivar, dest_q) = src_view.at_interface<elem_kind>(ig,j,k,ivar,src_q,rank) ; 
    }

} ; 

// pack into the coarse buffer ghosts
// from the coarse buffer 
template< 
    element_kind_t elem_kind,
    typename view_t
>
struct pack_to_cbuf_op {
    view_t cbuf ; 
    ghost_array_t ghost_view; 

    readonly_view_t<std::size_t> cbuf_qid, ghost_qid ; 
    readonly_view_t<uint8_t> src_elem_view ;
    
    std::size_t rank ; 

    index_transformer_t transf ; 

    pack_to_cbuf_op(
        view_t _view,
        ghost_array_t _ghost_view,
        Kokkos::View<size_t*> _cbuf_qid, 
        Kokkos::View<size_t*> _ghost_qid,
        Kokkos::View<uint8_t*> _src_elem,  
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank 
    ) : cbuf(_view)
      , ghost_view(_ghost_view)
      , cbuf_qid(_cbuf_qid)
      , ghost_qid(_ghost_qid)
      , src_elem_view(_src_elem)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz)
    { } 


    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {
        auto const ie  = src_elem_view(iq) ; 

        auto const cbuf_q  = cbuf_qid(iq)  ; 
        auto const ghost_q = ghost_qid(iq) ; 

        std::size_t VEC(i_a,j_a,k_a) ; 
        // phys indices 
        transf.compute_indices<elem_kind,true>(
            ig, VECD(j, k), i_a, j_a, k_a, ie, /*half ncells*/ true 
        ) ; 
        
        ghost_view.at_cbuf<elem_kind>(ig,j,k,ivar,ghost_q,rank) = 
            cbuf(VEC(i_a,j_a,k_a), ivar, cbuf_q) ;
        
    }

} ;

// unpack from cbuf into a regular 
// quadrant. We need to offset depending 
// on the child-id of the source 
template< 
    element_kind_t elem_kind,
    typename view_t
>
struct unpack_from_cbuf_op {
    view_t view ; 
    ghost_array_t ghost_view; 

    readonly_view_t<std::size_t> qid, ghost_qid ; 
    readonly_view_t<uint8_t> dst_elem_view, ichild_view ;
    
    std::size_t rank ; 

    index_transformer_t transf ; 

    unpack_from_cbuf_op(
        ghost_array_t _ghost_view,
        view_t _view,
        Kokkos::View<size_t*> _ghost_qid,
        Kokkos::View<size_t*> _qid, 
        Kokkos::View<uint8_t*> _dst_elem,  
        Kokkos::View<uint8_t*> _ichild,
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank 
    ) : view(_view)
      , ghost_view(_ghost_view)
      , qid(_qid)
      , ghost_qid(_ghost_qid)
      , dst_elem_view(_dst_elem)
      , ichild_view(_ichild)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz)
    { } 


    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {
        auto const ie  = dst_elem_view(iq) ; 

        auto const view_q  = qid(iq)  ; 
        auto const ghost_q = ghost_qid(iq) ;


        std::size_t VEC(i_a,j_a,k_a) ;

        auto ichild = ichild_view(iq) ; 
        size_t j_off{0UL}, k_off{0UL} ; 
        cbuf_to_view_offsets<elem_kind>::get(
            j_off,k_off, transf.nx, ichild 
        ) ; 

        // ghost indices 
        transf.compute_indices<elem_kind,false>(
            ig, VECD(j+j_off, k+k_off), i_a, j_a, k_a, ie, /*half ncells*/ false 
        ) ; 
        
        view(VEC(i_a,j_a,k_a), ivar, view_q) = 
            ghost_view.at_cbuf<elem_kind>(ig,j,k,ivar,ghost_q,rank) ;
        
    }

// TODO figure out if we want to schedule different pack / unpack kernels
// depending if the source / destination is a cbuffer. We could also store 
// two pointers inside this function and switch iteration to iteration.. 

} ;
    
// unpack to cbuf from a regular 
// quadrant. We need to offset depending 
// on the child-id of the source 
template< 
    element_kind_t elem_kind,
    typename view_t
>
struct unpack_to_cbuf_op {
    view_t cbuf ; 
    ghost_array_t ghost_view; 

    readonly_view_t<std::size_t> cbuf_qid, ghost_qid ; 
    readonly_view_t<uint8_t> dst_elem_view, ichild_view ;
    
    std::size_t rank ; 

    index_transformer_t transf ; 

    unpack_to_cbuf_op(
        ghost_array_t _ghost_view,
        view_t _view,
        Kokkos::View<size_t*> _ghost_qid,
        Kokkos::View<size_t*> _qid, 
        Kokkos::View<uint8_t*> _dst_elem,  
        Kokkos::View<uint8_t*> _ichild,
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank 
    ) : cbuf(_view)
      , ghost_view(_ghost_view)
      , cbuf_qid(_qid)
      , ghost_qid(_ghost_qid)
      , dst_elem_view(_dst_elem)
      , ichild_view(_ichild)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz)
    { }

    // here the loop will run to nx + ngz in any non-ghost direction.
    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {

        auto const ie  = dst_elem_view(iq) ; 

        auto const cbuf_q  = cbuf_qid(iq)  ; 
        auto const ghost_q = ghost_qid(iq) ;

        auto ichild = ichild_view(iq) ;
        size_t j_off{0UL}, k_off{0UL} ; 
        view_to_cbuf_offsets<elem_kind>::get(
            j_off,k_off, transf.nx, transf.ngz, ichild 
        ) ;

        std::size_t VEC(i_a,j_a,k_a) ;
        // ghost indices (in cbuf)
        transf.compute_indices<elem_kind,false>(
            ig, VECD(j, k), i_a, j_a, k_a, ie, /*half ncells*/ true 
        ) ; 

        cbuf(VEC(i_a,j_a,k_a), ivar, cbuf_q) = 
            ghost_view.at_interface<elem_kind>(ig,j+j_off,k+k_off,ivar,ghost_q,rank) ;

    }

} ; 

}} /* namespace grace::amr */

#endif /* GRACE_AMR_PACK_UNPACK_KERNELS_HH */