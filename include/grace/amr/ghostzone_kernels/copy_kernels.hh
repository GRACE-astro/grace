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
                return (i>=ngz and i<nx+ngz) ; 
            } else if ( ie < 8 ) {
                return (j>=ngz and j<ny+ngz) ;
            } else {
                return (k>=ngz and k<nz+ngz) ;
            }
        } 
        return true ; 
        
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
        if ( in_range(i_a,j_a,k_a, ie_src) ) {
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
        std::size_t _ngz
    ) : view(_view)
      , cbuf(_cbuf)
      , view_qid(_view_qid)
      , cbuf_qid(_cbuf_qid)
      , view_ic(_ic_view)
      , elem_view(_elem_view)
      , cbuf_elem_view(_cbuf_elem_view)
      , transf(VEC(_nx,_ny,_nz),_ngz)
    {}

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
        cbuf(
            VEC(i_b,j_b,k_b), ivar, cbuf_q 
        ) = view(VEC(i_a,j_a,k_a), ivar, view_q) ;
        
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
        std::size_t _ngz
    ) : view(_view)
      , cbuf(_cbuf)
      , view_qid(_view_qid)
      , cbuf_qid(_cbuf_qid)
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
        view(
            VEC(i_b,j_b,k_b), ivar, view_q 
        ) = cbuf(VEC(i_a,j_a,k_a), ivar, cbuf_q) ;
        
    }
} ; 
    
}} /* namespace grace::amr */


#endif /* GRACE_AMR_BC_COPY_GHOSTZONES_HH */