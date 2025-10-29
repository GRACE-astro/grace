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

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        src_view = alias.get<stag>() ; 
    }

    pack_op(
        view_t _src_view,
        ghost_array_t _dest_view,
        Kokkos::View<size_t*> _src_qid, 
        Kokkos::View<size_t*> _dest_qid,
        Kokkos::View<uint8_t*> _src_elem,  
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank,
        grace::var_staggering_t stag
    ) : src_view(_src_view)
      , dest_view(_dest_view)
      , src_qid(_src_qid)
      , dest_qid(_dest_qid)
      , src_elem_view(_src_elem)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz, stag)
    { } 
    // here we pick phys indices
    KOKKOS_INLINE_FUNCTION
    bool in_range(size_t i, size_t j, size_t k, int8_t ie) const {
        
        size_t nx  = transf.nx + transf.sx ; 
        size_t ny  = transf.ny + transf.sy ; 
        size_t nz  = transf.nz + transf.sz ;
        size_t ngz = transf.ngz ;
        if constexpr ( elem_kind == FACE ) {
            const int axis = ie / 2;
            if ( axis == 0 ) { // across X - face
                return (j >= ngz and j<ny+ngz) and ( k>=ngz and k<nz+ngz) ; 
            } else if ( axis == 1 ) {
                return (i>=ngz and i<nx+ngz) and (k>=ngz and k<nz+ngz) ;
            } else {
                return (i>=ngz and i<nx+ngz) and (j>=ngz and j<ny+ngz) ; 
            }
        } else if constexpr ( elem_kind == EDGE ) {
            if ( ie < 4 ) {
                int side1 = ((ie>>0)&1) ; 
                int side2 = ((ie>>1)&1) ; 
                bool gz_in_range1 = side1 ? j < ny + ngz : j < 2*ngz + transf.sy ;
                bool gz_in_range2 = side2 ? k < nz + ngz : k < 2*ngz + transf.sz ;
                return (i>=ngz and i<nx+ngz) and gz_in_range1 and gz_in_range2 ; 
            } else if ( ie < 8 ) {
                int side1 = ((ie>>0)&1) ; 
                int side2 = ((ie>>1)&1) ; 
                bool gz_in_range1 = side1 ? i < nx + ngz : i < 2*ngz + transf.sx ;
                bool gz_in_range2 = side2 ? k < nz + ngz : k < 2*ngz + transf.sz ;
                return (j>=ngz and j<ny+ngz) and gz_in_range1 and gz_in_range2 ;
            } else {
                int side1 = ((ie>>0)&1) ; 
                int side2 = ((ie>>1)&1) ; 
                bool gz_in_range1 = side1 ? i < nx + ngz : i < 2*ngz + transf.sx ;
                bool gz_in_range2 = side2 ? j < ny + ngz : j < 2*ngz + transf.sy ;
                return (k>=ngz and k<nz+ngz) and gz_in_range1 and gz_in_range2 ;
            }
        } else {
            int side1 = ((ie>>0)&1) ; 
            int side2 = ((ie>>1)&1) ; 
            int side3 = ((ie>>2)&1) ; 
            bool gz_in_range1 = side1 ? i < nx + ngz : i < 2*ngz + transf.sx ;
            bool gz_in_range2 = side2 ? j < ny + ngz : j < 2*ngz + transf.sy ;
            bool gz_in_range3 = side3 ? k < nz + ngz : k < 2*ngz + transf.sz ;
            return gz_in_range1 and gz_in_range2 and gz_in_range3 ;
        }        
    }

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
        if ( in_range(i_a,j_a,k_a,ie) ) {
            //if ( transf.sx or transf.sy or transf.sz )
            //    printf("Pack: stag (%zu,%zu,%zu) i %zu j %zu k %zu q %zu ie %d value %f\n", transf.sx,transf.sy,transf.sz,i_a,j_a,k_a,src_q,ie, src_view(VEC(i_a,j_a,k_a), ivar, src_q)) ;
            dest_view.at_interface<elem_kind>(ig,j,k,ivar,dest_q,rank) = 
                src_view(VEC(i_a,j_a,k_a), ivar, src_q) ;
        }
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
    readonly_view_t<uint8_t> dest_element_view ;
    
    std::size_t rank ; 

    index_transformer_t transf ; 

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        dest_view = alias.get<stag>() ; 
    }

    KOKKOS_INLINE_FUNCTION
    bool in_range(size_t i, size_t j, size_t k, int8_t ie) const {
        
        size_t nx  = transf.nx + transf.sx ; 
        size_t ny  = transf.ny + transf.sy ; 
        size_t nz  = transf.nz + transf.sz ;
        size_t ngz = transf.ngz ;
        if constexpr ( elem_kind == FACE ) {
            const int axis = ie / 2;
            if ( axis == 0 ) { // across X - face
                return (j >= ngz and j<ny+ngz) and ( k>=ngz and k<nz+ngz) ; 
            } else if ( axis == 1 ) {
                return (i>=ngz and i<nx+ngz) and (k>=ngz and k<nz+ngz) ;
            } else {
                return (i>=ngz and i<nx+ngz) and (j>=ngz and j<ny+ngz) ; 
            }
        } else if constexpr ( elem_kind == EDGE ) {
            if ( ie < 4 ) {
                int side1 = ((ie>>0)&1) ; 
                int side2 = ((ie>>1)&1) ; 
                bool gz_in_range1 = side1 ? j < ny + 2 * ngz : j < ngz + transf.sy ;
                bool gz_in_range2 = side2 ? k < nz + 2 * ngz : k < ngz + transf.sz ;
                return (i>=ngz and i<nx+ngz) and gz_in_range1 and gz_in_range2 ; 
            } else if ( ie < 8 ) {
                int side1 = ((ie>>0)&1) ; 
                int side2 = ((ie>>1)&1) ; 
                bool gz_in_range1 = side1 ? i < nx + 2 * ngz : i < ngz + transf.sx ;
                bool gz_in_range2 = side2 ? k < nz + 2 * ngz : k < ngz + transf.sz ;
                return (j>=ngz and j<ny+ngz) and gz_in_range1 and gz_in_range2 ;
            } else {
                int side1 = ((ie>>0)&1) ; 
                int side2 = ((ie>>1)&1) ; 
                bool gz_in_range1 = side1 ? i < nx + 2 * ngz : i < ngz + transf.sx ;
                bool gz_in_range2 = side2 ? j < ny + 2 * ngz : j < ngz + transf.sy ;
                return (k>=ngz and k<nz+ngz) and gz_in_range1 and gz_in_range2 ;
            }
        }  else {
            int side1 = ((ie>>0)&1) ; 
            int side2 = ((ie>>1)&1) ; 
            int side3 = ((ie>>2)&1) ; 
            bool gz_in_range1 = side1 ? i < nx + 2 * ngz : i < ngz + transf.sx ;
            bool gz_in_range2 = side2 ? j < ny + 2 * ngz : j < ngz + transf.sy ;
            bool gz_in_range3 = side3 ? k < nz + 2 * ngz : k < ngz + transf.sz ;
            return gz_in_range1 and gz_in_range2 and gz_in_range3 ;
        }        
    }

    unpack_op(
        ghost_array_t _src_view,
        view_t _dest_view,
        Kokkos::View<size_t*> _src_qid, 
        Kokkos::View<size_t*> _dest_qid,
        Kokkos::View<uint8_t*> _dest_ie,  
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank,
        grace::var_staggering_t stag
    ) : src_view(_src_view)
      , dest_view(_dest_view)
      , src_qid(_src_qid)
      , dest_qid(_dest_qid)
      , dest_element_view(_dest_ie)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz, stag)
    { } 


    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {
        auto const ie  = dest_element_view(iq) ; 

        auto const src_q  = src_qid(iq)  ; 
        auto const dest_q = dest_qid(iq) ;


        std::size_t VEC(i_a,j_a,k_a) ; 
        transf.compute_indices<elem_kind,false>(
            ig, VECD(j, k), i_a, j_a, k_a, ie 
        ) ; 
        if ( in_range(i_a,j_a,k_a,ie) ) {   
            //if ( transf.sx or transf.sy or transf.sz )
            //    printf("Unpack: stag (%zu,%zu,%zu) i %zu j %zu k %zu q %zu ie %d value %f\n", transf.sx,transf.sy,transf.sz,i_a,j_a,k_a,dest_q,ie, src_view.at_interface<elem_kind>(ig,j,k,ivar,src_q,rank)) ;  
            dest_view(VEC(i_a,j_a,k_a), ivar, dest_q) = src_view.at_interface<elem_kind>(ig,j,k,ivar,src_q,rank) ; 
        }
        
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

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {}

    pack_to_cbuf_op(
        view_t _view,
        ghost_array_t _ghost_view,
        Kokkos::View<size_t*> _cbuf_qid, 
        Kokkos::View<size_t*> _ghost_qid,
        Kokkos::View<uint8_t*> _src_elem,  
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank, var_staggering_t stag
    ) : cbuf(_view)
      , ghost_view(_ghost_view)
      , cbuf_qid(_cbuf_qid)
      , ghost_qid(_ghost_qid)
      , src_elem_view(_src_elem)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz,stag)
    { } 

    KOKKOS_INLINE_FUNCTION
    bool in_range(size_t i, size_t j, size_t k, int8_t ie) const {
        
        size_t nx  = transf.nx/2 + transf.sx ; 
        size_t ny  = transf.ny/2 + transf.sy ; 
        size_t nz  = transf.nz/2 + transf.sz ;
        size_t ngz = transf.ngz ;
        if constexpr ( elem_kind == FACE ) {
            const int axis = ie / 2;
            if ( axis == 0 ) { // across X - face
                return (j >= ngz and j<ny+ngz) and ( k>=ngz and k<nz+ngz) ; 
            } else if ( axis == 1 ) {
                return (i>=ngz and i<nx+ngz) and (k>=ngz and k<nz+ngz) ;
            } else {
                return (i>=ngz and i<nx+ngz) and (j>=ngz and j<ny+ngz) ; 
            }
        } else if constexpr ( elem_kind == EDGE ) {
            if ( ie < 4 ) {
                int side1 = ((ie>>0)&1) ; 
                int side2 = ((ie>>1)&1) ; 
                bool gz_in_range1 = side1 ? j < ny + ngz : j < 2*ngz + transf.sy ;
                bool gz_in_range2 = side2 ? k < nz + ngz : k < 2*ngz + transf.sz ;
                return (i>=ngz and i<nx+ngz) and gz_in_range1 and gz_in_range2 ; 
            } else if ( ie < 8 ) {
                int side1 = ((ie>>0)&1) ; 
                int side2 = ((ie>>1)&1) ; 
                bool gz_in_range1 = side1 ? i < nx + ngz : i < 2*ngz + transf.sx ;
                bool gz_in_range2 = side2 ? k < nz + ngz : k < 2*ngz + transf.sz ;
                return (j>=ngz and j<ny+ngz) and gz_in_range1 and gz_in_range2 ;
            } else {
                int side1 = ((ie>>0)&1) ; 
                int side2 = ((ie>>1)&1) ; 
                bool gz_in_range1 = side1 ? i < nx + ngz : i < 2*ngz + transf.sx ;
                bool gz_in_range2 = side2 ? j < ny + ngz : j < 2*ngz + transf.sy ;
                return (k>=ngz and k<nz+ngz) and gz_in_range1 and gz_in_range2 ;
            }
        } else {
            int side1 = ((ie>>0)&1) ; 
            int side2 = ((ie>>1)&1) ; 
            int side3 = ((ie>>2)&1) ; 
            bool gz_in_range1 = side1 ? i < nx + ngz : i < 2*ngz + transf.sx ;
            bool gz_in_range2 = side2 ? j < ny + ngz : j < 2*ngz + transf.sy ;
            bool gz_in_range3 = side3 ? k < nz + ngz : k < 2*ngz + transf.sz ;
            return gz_in_range1 and gz_in_range2 and gz_in_range3 ;
        }             
    }

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
        if ( in_range(i_a,j_a,k_a,ie))
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

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        view = alias.get<stag>() ; 
    }

    unpack_from_cbuf_op(
        ghost_array_t _ghost_view,
        view_t _view,
        Kokkos::View<size_t*> _ghost_qid,
        Kokkos::View<size_t*> _qid, 
        Kokkos::View<uint8_t*> _dst_elem,  
        Kokkos::View<uint8_t*> _ichild,
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank, var_staggering_t stag
    ) : view(_view)
      , ghost_view(_ghost_view)
      , qid(_qid)
      , ghost_qid(_ghost_qid)
      , dst_elem_view(_dst_elem)
      , ichild_view(_ichild)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz,stag)
    { } 

    KOKKOS_INLINE_FUNCTION
    bool in_range(size_t i, size_t j, size_t k, int8_t ie, int8_t ic) const {
        size_t sx = transf.sx ; 
        size_t sy = transf.sy ; 
        size_t sz = transf.sz ; 
        size_t nx  = transf.nx ; 
        size_t ny  = transf.ny ; 
        size_t nz  = transf.nz ;
        size_t ngz = transf.ngz ;
        if constexpr ( elem_kind == FACE ) { // for faces gz loop is ngz
            const int axis = ie / 2;

            const int ioff = (ic>>0)&1 ; 
            const int joff = (ic>>1)&1 ; 
            // if upper child : 0 to n + ngz
            // else: ngz to n + 2 ngz 
            const int lbi = ioff ? ngz + transf.nx/2 : ngz ; 
            const int lbj = joff ? ngz + transf.nx/2 : ngz ; 
            
            if ( axis == 0 ) { // across X - face
                const int ubi = ioff ?  ny+sy+ngz : ny/2+sy+ngz ;
                const int ubj = joff ?  nz+sz+ngz : nz/2+sz+ngz ;  
                return (j >= lbi and j<ubi) and ( k>=lbj and k<ubj) ; 
            } else if ( axis == 1 ) {
                const int ubi = ioff ?  nx+sx+ngz : nx/2+sx+ngz ;
                const int ubj = joff ?  nz+sz+ngz : nz/2+sz+ngz ;
                return (i>=lbi and i<ubi) and (k>=lbj and k<ubj) ;
            } else {
                const int ubi = ioff ?  nx+sx+ngz : nx/2+sx+ngz ;
                const int ubj = joff ?  ny+sy+ngz : ny/2+sy+ngz ;
                return (i>=lbi and i<ubi) and (j>=lbj and j<ubj) ; 
            }
        } else if constexpr ( elem_kind == EDGE ) { // for edges gz loop is gz + 1 
            int side1 = ((ie>>0)&1) ; 
            int side2 = ((ie>>1)&1) ; 
            int off = ic ; 
            
            if ( ie < 4 ) {
                const int lb = off ? ngz + nx / 2 : ngz ; 
                const int ub = off ? nx+sx+ngz : nx/2+sx+ngz ; 
                bool gz_in_range1 = side1 ? j < ny + sy + 2 * ngz : j < ngz + transf.sy ;
                bool gz_in_range2 = side2 ? k < nz + sz + 2 * ngz : k < ngz + transf.sz ;
                return (i>=lb and i<ub) and gz_in_range1 and gz_in_range2 ; 
            } else if ( ie < 8 ) {
                const int lb = off ? ngz + ny / 2 : ngz ; 
                const int ub = off ? ny+sy+ngz : ny/2+sy+ngz ; 
                bool gz_in_range1 = side1 ? i < nx + sx + 2 * ngz : i < ngz + transf.sx ;
                bool gz_in_range2 = side2 ? k < nz + sz + 2 * ngz : k < ngz + transf.sz ;
                return (j>=lb and j<ub) and gz_in_range1 and gz_in_range2 ; 
            } else {
                const int lb = off ? ngz + nz / 2 : ngz ; 
                const int ub = off ? nz+sz+ngz : nz/2+sz+ngz ; 
                bool gz_in_range1 = side1 ? i < nx + sx + 2 * ngz : i < ngz + transf.sx ;
                bool gz_in_range2 = side2 ? j < ny + sy + 2 * ngz : j < ngz + transf.sy ;
                return (k>=lb and k<ub) and gz_in_range1 and gz_in_range2 ; 
            }
        } else {
            int side1 = ((ie>>0)&1) ; 
            int side2 = ((ie>>1)&1) ; 
            int side3 = ((ie>>2)&1) ; 
            bool gz_in_range1 = side1 ? i < nx + sx + 2 * ngz : i < ngz + transf.sx ;
            bool gz_in_range2 = side2 ? j < ny + sy + 2 * ngz : j < ngz + transf.sy ;
            bool gz_in_range3 = side3 ? k < nz + sz + 2 * ngz : k < ngz + transf.sz ;
            return gz_in_range1 and gz_in_range2 and gz_in_range3 ;
        }        
    }

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
        if ( in_range(i_a,j_a,k_a,ie,ichild))
            view(VEC(i_a,j_a,k_a), ivar, view_q) = 
                ghost_view.at_cbuf<elem_kind>(ig,j,k,ivar,ghost_q,rank) ;
        
    }

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

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {}

    unpack_to_cbuf_op(
        ghost_array_t _ghost_view,
        view_t _view,
        Kokkos::View<size_t*> _ghost_qid,
        Kokkos::View<size_t*> _qid, 
        Kokkos::View<uint8_t*> _dst_elem,  
        Kokkos::View<uint8_t*> _ichild,
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars, std::size_t _rank, var_staggering_t stag
    ) : cbuf(_view)
      , ghost_view(_ghost_view)
      , cbuf_qid(_qid)
      , ghost_qid(_ghost_qid)
      , dst_elem_view(_dst_elem)
      , ichild_view(_ichild)
      , rank(_rank)
      , transf(VEC(_nx,_ny,_nz),_ngz,stag)
    { }

    // range check:
    // All non-gz loops are extended by 1 to accomodate staggerings
    // independent of the element (face/edge) orientation. This means 
    // we need to check to avoid out-of-bounds accesses. In this function
    // since we fill the virtual edges and corners adjacent to faces 
    // we need to accomodate for that too.
    KOKKOS_INLINE_FUNCTION
    bool in_range(size_t i, size_t j, size_t k, int8_t ie, int8_t ic /*child idx*/) const {
        size_t sx = transf.sx ; 
        size_t sy = transf.sy ; 
        size_t sz = transf.sz ; 
        size_t nx  = transf.nx/2 ; 
        size_t ny  = transf.ny/2 ; 
        size_t nz  = transf.nz/2 ;
        size_t ngz = transf.ngz ;
        if constexpr ( elem_kind == FACE ) {
            const int axis = ie / 2;
            const int ioff = (ic>>0)&1 ; 
            const int joff = (ic>>1)&1 ; 
            // if upper child : 0 to n + ngz
            // else: ngz to n + 2 ngz 
            const int lbi = ioff ? 0 : ngz ; 
            const int lbj = joff ? 0 : ngz ; 

            if ( axis == 0 ) { // across X - face
                size_t const ubi = ioff ? ny + ngz + sy : ny + 2 * ngz + sy ; 
                size_t const ubj = joff ? nz + ngz + sz : nz + 2 * ngz + sz ; 
                return (j >= lbi and j<ubi) and ( k>=lbj and k<ubj) ; 
            } else if ( axis == 1 ) {
                size_t const ubi = ioff ? nx + ngz + sx : nx + 2 * ngz + sx ; 
                size_t const ubj = joff ? nz + ngz + sz : nz + 2 * ngz + sz ;
                return (i>=lbi and i<ubi) and (k>=lbj and k<ubj) ;
            } else {
                size_t const ubi = ioff ? nx + ngz + sx : nx + 2 * ngz + sx ; 
                size_t const ubj = joff ? ny + ngz + sy : ny + 2 * ngz + sy ; 
                return (i>=lbi and i<ubi) and (j>=lbj and j<ubj) ; 
            }
        } else if constexpr ( elem_kind == EDGE ) {
            // if upper child : 0 to n + ngz
            // else: ngz to n + 2 ngz 
            const int lbi = ic ? 0 : ngz ; 
            // ghostzone checks 
            const int side1 = (ie>>0)&1 ;
            const int side2 = (ie>>1)&1 ; 
            if ( ie < 4 ) {
                size_t const ubi = ic ? nx + ngz + sx : nx + 2 * ngz + sx ; 
                bool gz_in_range1 = side1 ? j < ny + sy + 2 * ngz : j < ngz + transf.sy ;
                bool gz_in_range2 = side2 ? k < nz + sz + 2 * ngz : k < ngz + transf.sz ;
                return (i>=lbi and i<ubi) and gz_in_range1 and gz_in_range2 ; 
            } else if ( ie < 8 ) {
                size_t const ubi = ic ? ny + ngz + sy : ny + 2 * ngz + sy ; 
                bool gz_in_range1 = side1 ? i < nx + sx + 2 * ngz : i < ngz + transf.sx ;
                bool gz_in_range2 = side2 ? k < nz + sz + 2 * ngz : k < ngz + transf.sz ;
                return (j>=lbi and j<ubi) and gz_in_range1 and gz_in_range2 ;
            } else {
                size_t const ubi = ic ? nz + ngz + sz : nz + 2 * ngz + sz ; 
                bool gz_in_range1 = side1 ? i < nx + sx + 2 * ngz : i < ngz + transf.sx ;
                bool gz_in_range2 = side2 ? j < ny + sy + 2 * ngz : j < ngz + transf.sy ;
                return (k>=lbi and k<ubi) and gz_in_range1 and gz_in_range2 ;
            }
        } else {
            // ghostzone checks 
            int side1 = ((ie>>0)&1) ; 
            int side2 = ((ie>>1)&1) ; 
            int side3 = ((ie>>2)&1) ; 
            bool gz_in_range1 = side1 ? i < nx + sx + 2 * ngz : i < ngz + transf.sx ;
            bool gz_in_range2 = side2 ? j < ny + sy + 2 * ngz : j < ngz + transf.sy ;
            bool gz_in_range3 = side3 ? k < nz + sz + 2 * ngz : k < ngz + transf.sz ;  
            return gz_in_range1 and gz_in_range2 and gz_in_range3 ; 
        }
    }

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
        int  j_off_c{0}, k_off_c{0} ; 
        view_to_cbuf_offsets<elem_kind>::get(
            j_off,k_off,j_off_c,k_off_c, transf.nx, transf.ngz, ichild 
        ) ;

        std::size_t VEC(i_a,j_a,k_a) ;
        // ghost indices (in cbuf)
        transf.compute_indices<elem_kind,false>(
            ig, VECD(j + j_off_c, k + k_off_c), i_a, j_a, k_a, ie, /*half ncells*/ true 
        ) ; 
        if( in_range(i_a,j_a,k_a,ie,ichild))
            cbuf(VEC(i_a,j_a,k_a), ivar, cbuf_q) = 
                ghost_view.at_interface<elem_kind>(ig,j+j_off,k+k_off,ivar,ghost_q,rank) ;

    }

} ; 

}} /* namespace grace::amr */

#endif /* GRACE_AMR_PACK_UNPACK_KERNELS_HH */