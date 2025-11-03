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
    typename view_t
>
struct copy_op {

    view_t view ; 
    readonly_view_t<std::size_t> src_qid, dest_qid ; 
    readonly_view_t<uint8_t> src_element_view, dest_element_view; 

    index_transformer_t transf ; 

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        view = alias.get<stag>() ; 
    }

    copy_op(
        view_t _view,
        Kokkos::View<size_t*> _src_qid, Kokkos::View<size_t*> _dest_qid,
        Kokkos::View<uint8_t*> _src_elem, Kokkos::View<uint8_t*> _dest_elem, 
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz), grace::var_staggering_t stag,
        std::size_t _ngz
    ) : view(_view)
      , src_qid(_src_qid)
      , dest_qid(_dest_qid)
      , src_element_view(_src_elem)
      , dest_element_view(_dest_elem)
      , transf(VEC(_nx,_ny,_nz),_ngz,stag)
    {}

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
        if ( in_range(i_b,j_b,k_b, ie_dest) ) {
            view(
                VEC(i_b,j_b,k_b), ivar, dest_q 
            ) = view(VEC(i_a,j_a,k_a), ivar, src_q) ;
        }
        

    }
} ; 

// this is a copy operation normal view -> cbuf 
template< 
    element_kind_t elem_kind,
    typename view_t,
    typename cbuf_t 
>
struct copy_to_cbuf_op {

    view_t view ; 
    cbuf_t cbuf ; 
    
    readonly_view_t<std::size_t> view_qid
                               , cbuf_qid ; 
    
    readonly_view_t<uint8_t> elem_view
                           , cbuf_elem_view
                           , view_ic; 

    index_transformer_t transf ; 


    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        view = alias.get<stag>() ; 
    }

    copy_to_cbuf_op(
        view_t _view,
        cbuf_t _cbuf,
        Kokkos::View<size_t*> _view_qid, 
        Kokkos::View<size_t*> _cbuf_qid,
        Kokkos::View<uint8_t*> _elem_view, 
        Kokkos::View<uint8_t*> _cbuf_elem_view,
        Kokkos::View<uint8_t*> _ic_view, 
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, var_staggering_t stag
    ) : view(_view)
      , cbuf(_cbuf)
      , view_qid(_view_qid)
      , cbuf_qid(_cbuf_qid)
      , view_ic(_ic_view)
      , elem_view(_elem_view)
      , cbuf_elem_view(_cbuf_elem_view)
      , transf(VEC(_nx,_ny,_nz),_ngz, stag)
    {}

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

    // the loop(s) in non-gz directions are extended by ngz
    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {
        auto const ie_view  = elem_view(iq) ; 
        auto const ie_cbuf = cbuf_elem_view(iq) ; 

        auto const view_q  = view_qid(iq)  ; 
        auto const cbuf_q = cbuf_qid(iq) ;

        // we need to offset into the coarse quad, 
        // accounting for the extra ngz in the loop
        auto const ichild = view_ic(iq) ; 
        size_t j_off{0UL}, k_off{0UL} ; 
        int j_off_c{0}, k_off_c{0} ; 
        view_to_cbuf_offsets<elem_kind>::get(
            j_off,k_off,j_off_c,k_off_c, transf.nx, transf.ngz, ichild 
        ) ;


        std::size_t VEC(i_a,j_a,k_a), VEC(i_b,j_b,k_b) ; 
        // copy into cbuf's gzs

        // physical indices, offset 
        transf.compute_indices<elem_kind,true>(
        ig, VECD(j + j_off, k + k_off), i_a, j_a, k_a, ie_view, false
        ) ; 
        // gz indices, no offset 
        transf.compute_indices<elem_kind,false>(
            ig, VECD(j + j_off_c, k + k_off_c), 
            i_b, j_b, k_b, ie_cbuf, /* halved ncells */ true 
        ) ;
        // We need to check the cbuf indices here
        if ( in_range(i_b,j_b,k_b,ie_cbuf,ichild)) 
            cbuf(
                VEC(i_b,j_b,k_b), ivar, cbuf_q 
            ) = view(VEC(i_a,j_a,k_a), ivar, view_q) ;
        if ( cbuf_q == 16 and i_b==10 and j_b == 10 and k_b == 8 and ivar==0 ) {
                printf("Here from fill kernel: val %f \n", cbuf(i_b,j_b,k_b,ivar,cbuf_q));
            } 
        
    }
} ; 

// this is a copy operation cbuf -> normal view
/**
 * @brief Copy data from a coarse buffer into the ghostzones of a coarse quadrant.
 * @ingroup amr 
 * @tparam elem_kind Element kind.
 * @tparam view_t Type of normal view.
 * @tparam cbuf_t Type of coarse buffer.
 */
template< 
    element_kind_t elem_kind,
    typename view_t,
    typename cbuf_t 
>
struct copy_from_cbuf_op {

    view_t view ; //!< Data view.
    cbuf_t cbuf ; //!< Coarse buffers view.
    
    //! View and coarse buffers quad-id.
    readonly_view_t<std::size_t> view_qid
                               , cbuf_qid ; 
    //! Element-id in view and coarse buffers, child index in the view.
    readonly_view_t<uint8_t> elem_view
                           , cbuf_elem_view
                           , view_ic; 
    //! Index transformer.
    index_transformer_t transf ; 
    /**
     * @brief Set the data ptr 
     * 
     * @param alias The alias that needs to be swapped to.
     */
    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        view = alias.get<stag>() ; 
    }

    copy_from_cbuf_op(
        cbuf_t _cbuf,
        view_t _view,        
        Kokkos::View<size_t*> _view_qid, 
        Kokkos::View<size_t*> _cbuf_qid,
        Kokkos::View<uint8_t*> _elem_view, 
        Kokkos::View<uint8_t*> _cbuf_elem_view,
        Kokkos::View<uint8_t*> _ic_view, 
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, var_staggering_t stag 
    ) : view(_view)
      , cbuf(_cbuf)
      , view_qid(_view_qid)
      , cbuf_qid(_cbuf_qid)
      , view_ic(_ic_view)
      , elem_view(_elem_view)
      , cbuf_elem_view(_cbuf_elem_view)
      , transf(VEC(_nx,_ny,_nz),_ngz,stag)
    {}

    // range check:
    // All non-gz loops are extended by 1 so we need to check 
    // range.
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
            // if upper child : n/2+ngz to n + ngz
            // else: n/2+ngz to n + 2 ngz 
            if ( axis == 0 ) { // across X - face
                const int lbi = ioff ? ngz + ny/2 + sy : ngz ; 
                const int lbj = joff ? ngz + nz/2 + sz : ngz ; 
                const int ubi = ioff ? ngz + ny + sy   : ngz+ny/2+sy ;
                const int ubj = joff ? ngz + nz + sz   : ngz+nz/2+sz ;  
                return (j >= lbi and j<ubi) and ( k>=lbj and k<ubj) ; 
            } else if ( axis == 1 ) {
                const int lbi = ioff ? ngz + nx/2 + sx : ngz ; 
                const int lbj = joff ? ngz + nz/2 + sz : ngz ; 
                const int ubi = ioff ? ngz + nx + sx   : nx/2+sx+ngz ;
                const int ubj = joff ? ngz + nz + sz   : nz/2+sz+ngz ;
                return (i>=lbi and i<ubi) and (k>=lbj and k<ubj) ;
            } else {
                const int lbi = ioff ? ngz + nx/2 + sx : ngz ; 
                const int lbj = joff ? ngz + ny/2 + sy : ngz ; 
                const int ubi = ioff ? ngz + nx + sx   : nx/2+sx+ngz ;
                const int ubj = joff ? ngz + ny + sy   : ny/2+sy+ngz ;
                return (i>=lbi and i<ubi) and (j>=lbj and j<ubj) ; 
            }
        } else if constexpr ( elem_kind == EDGE ) { // for edges gz loop is gz + 1 
            int side1 = ((ie>>0)&1) ; 
            int side2 = ((ie>>1)&1) ; 
            int off = ic ; 
            
            if ( ie < 4 ) {
                const int lb = off ? ngz + nx/2 + sx : ngz ; 
                const int ub = off ? ngz + nx + sx   : nx/2+sx+ngz ; 
                bool gz_in_range1 = side1 ? j < ny + sy + 2 * ngz : j < ngz + sy ;
                bool gz_in_range2 = side2 ? k < nz + sz + 2 * ngz : k < ngz + sz ;
                return (i>=lb and i<ub) and gz_in_range1 and gz_in_range2 ; 
            } else if ( ie < 8 ) {
                const int lb = off ? ngz + ny/2 + sy : ngz ; 
                const int ub = off ? ngz + ny + sy   : ny/2+sy+ngz ; 
                bool gz_in_range1 = side1 ?  i < nx + sx + 2 * ngz : i < ngz + sx ;//here i >= nx + sx + ngz and
                bool gz_in_range2 = side2 ?  k < nz + sz + 2 * ngz : k < ngz + sz ;//k >= nz + sz + ngz and
                return (j>=lb and j<ub) and gz_in_range1 and gz_in_range2 ; 
            } else {
                const int lb = off ? ngz + nz/2 + sz : ngz ; 
                const int ub = off ? ngz + nz + sz   : nz/2+sz+ngz ; 
                bool gz_in_range1 = side1 ? i < nx + sx + 2 * ngz : i < ngz + sx ;
                bool gz_in_range2 = side2 ? j < ny + sy + 2 * ngz : j < ngz + sy ;
                return (k>=lb and k<ub) and gz_in_range1 and gz_in_range2 ; 
            }
        } else {
            int side1 = ((ie>>0)&1) ; 
            int side2 = ((ie>>1)&1) ; 
            int side3 = ((ie>>2)&1) ; 
            bool gz_in_range1 = side1 ? i < nx + sx + 2 * ngz : i < ngz + sx ;
            bool gz_in_range2 = side2 ? j < ny + sy + 2 * ngz : j < ngz + sy ;
            bool gz_in_range3 = side3 ? k < nz + sz + 2 * ngz : k < ngz + sz ;
            return gz_in_range1 and gz_in_range2 and gz_in_range3 ;
        }
    }

    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) const 
    {
        auto const ie_view  = elem_view(iq) ; 
        auto const ie_cbuf = cbuf_elem_view(iq) ; 

        auto const view_q  = view_qid(iq)  ; 
        auto const cbuf_q = cbuf_qid(iq) ;

        // we need to offset into the coarse quad, 
        // accounting for the extra ngz in the loop
        auto const ichild = view_ic(iq) ; 
        size_t j_off{0UL}, k_off{0UL} ; 
        cbuf_to_view_offsets<elem_kind>::get(
            j_off,k_off, transf.nx, ichild 
        ) ;


        std::size_t VEC(i_a,j_a,k_a), VEC(i_b,j_b,k_b) ; 
        // copy into view's gzs 
        transf.compute_indices<elem_kind,true>(
            ig, VECD(j, k), 
            i_a, j_a, k_a, ie_cbuf, /* halved ncells */ true 
        ) ; 
        transf.compute_indices<elem_kind,false>(
            ig, VECD(j + j_off, k + k_off), i_b, j_b, k_b, ie_view,  /* halved ncells */ false
        ) ; 
        // check view indices here
        if ( in_range(i_b,j_b,k_b,ie_view,ichild) )
            view(
                VEC(i_b,j_b,k_b), ivar, view_q 
            ) = cbuf(VEC(i_a,j_a,k_a), ivar, cbuf_q) ;
        
    }
} ; 
    
}} /* namespace grace::amr */


#endif /* GRACE_AMR_BC_COPY_GHOSTZONES_HH */