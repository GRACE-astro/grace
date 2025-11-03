/**
 * @file restrict_kernels.hh
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

#ifndef GRACE_AMR_BC_RESTRICT_HH
#define GRACE_AMR_BC_RESTRICT_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

template< typename view_t > 
struct restrict_op {

    view_t src_view, dest_view ; 
    readonly_view_t<size_t> src_q, dest_q ; 
    size_t ngz ;

    restrict_op(
        view_t _src_view, view_t _dest_view,
        Kokkos::View<size_t*> _src_q, 
        Kokkos::View<size_t*> _dest_q, 
        size_t _ngz
    ) : src_view(_src_view)
      , dest_view(_dest_view)
      , src_q(_src_q)
      , dest_q(_dest_q)
      , ngz(_ngz)
    {}

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        src_view = alias.get<stag>() ; 
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (size_t i, size_t j, size_t k, size_t iv, size_t iq) const 
    {
        auto src_qid = src_q(iq) ; 
        auto dst_qid = dest_q(iq) ; 

        dest_view(i+ngz,j+ngz,k+ngz,iv,dst_qid) = 0.125 * (
            src_view(2*i+ngz  ,2*j+ngz  ,2*k+ngz  ,iv,src_qid) + 
            src_view(2*i+ngz+1,2*j+ngz  ,2*k+ngz  ,iv,src_qid) + 
            src_view(2*i+ngz  ,2*j+ngz+1,2*k+ngz  ,iv,src_qid) + 
            src_view(2*i+ngz  ,2*j+ngz  ,2*k+ngz+1,iv,src_qid) + 
            src_view(2*i+ngz+1,2*j+ngz+1,2*k+ngz  ,iv,src_qid) +
            src_view(2*i+ngz+1,2*j+ngz  ,2*k+ngz+1,iv,src_qid) +  
            src_view(2*i+ngz  ,2*j+ngz+1,2*k+ngz+1,iv,src_qid) +
            src_view(2*i+ngz+1,2*j+ngz+1,2*k+ngz+1,iv,src_qid)  
        ) ; 
    }


} ; 

// NOTE: ghostzone restrict fills the data on the face 
// of the cbuf. So here we would not need to in principle.
// However we still do since copy_to_cbuf does not. Anyway 
// it should not matter as this data is never used in practice.
// in conclusion --> stagger loop in one direction! 
// Since we separate staggerings here we can just specialize 
// the loop to have the right dimensions and we do **NOT** need 
// a range check ( remember these are interior loops ).
template< int stag_dir, typename view_t > 
struct div_free_restrict_op {

    view_t src_view, dest_view ; 
    readonly_view_t<size_t> src_q, dest_q ; 
    size_t ngz ;

    div_free_restrict_op(
        view_t _src_view, view_t _dest_view,
        Kokkos::View<size_t*> _src_q, 
        Kokkos::View<size_t*> _dest_q, 
        size_t _ngz
    ) : src_view(_src_view)
      , dest_view(_dest_view)
      , src_q(_src_q)
      , dest_q(_dest_q)
      , ngz(_ngz)
    {}

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        src_view = alias.get<stag>() ; 
    }
    // here it is assumed that the loop is staggered
    KOKKOS_INLINE_FUNCTION
    void operator() (size_t i, size_t j, size_t k, size_t iv, size_t iq) const 
    {
        auto src_qid = src_q(iq) ; 
        auto dst_qid = dest_q(iq) ; 

        // we need to add up the 4 values 
        // coming from within the coarse 
        // face
        double val = 0 ; 
         for( int ii=0; ii<=(stag_dir!=0); ++ii) {
            for( int jj=0; jj<=(stag_dir!=1); ++jj){
                for(int kk=0; kk<=(stag_dir!=2); ++kk){
                    val += src_view(2*i+ngz+ii,2*j+ngz+jj,2*k+ngz+kk,iv,src_qid) ; 
                }
            }
        }

        dest_view(i+ngz,j+ngz,k+ngz,iv,dst_qid) = 0.25 * val ; 
    }


} ; 

template<element_kind_t elem_kind>
struct ghost_restrict_tag_t {} ; 
using ghost_restrict_face_tag   = ghost_restrict_tag_t<FACE>   ; 
using ghost_restrict_edge_tag   = ghost_restrict_tag_t<EDGE>   ; 
using ghost_restrict_corner_tag = ghost_restrict_tag_t<CORNER> ; 

/**
 * @brief Restrict inside ghostzones.
 * 
 * @tparam view_t Type of data array.
 */
template<  typename view_t > 
struct ghost_restrict_op {

    view_t data, cbuf ; //!< Data and coarse buffer arrays.
    readonly_view_t<size_t> qid, cbuf_id ; //!< Data and coarse buffer quad-ids 
    readonly_view_t<uint8_t> elem_id ; //!< Element ids

    prolong_index_transformer_t transf ;  //!< Index transformations
    
    ghost_restrict_op(
        view_t _data, view_t _cbuf,
        Kokkos::View<size_t*> _qid, 
        Kokkos::View<size_t*> _cbuf_id,
        Kokkos::View<uint8_t*> _eid,  
        size_t n, size_t _ngz
    ) : data(_data)
      , cbuf(_cbuf)
      , qid(_qid)
      , cbuf_id(_cbuf_id)
      , elem_id(_eid)
      , transf(n,_ngz, STAG_CENTER)
    {}

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        data = alias.get<stag>() ; 
    }
    // runs to n/2 n/2 
    KOKKOS_INLINE_FUNCTION
    void operator() (ghost_restrict_face_tag,  size_t j, size_t k, size_t iv, size_t iq) const 
    {
        auto q_id = qid(iq) ; 
        auto c_id = cbuf_id(iq) ; 

        auto e_id = elem_id(iq) ;

        int s[3] ; 
        transf.get_stencil<FACE>(s, e_id) ; 

        // loop in the ghostzones, only ngz/2
        for( int i=0; i<transf.g/2; ++i) {
            size_t i_c, j_c, k_c ; 
            transf.compute_indices<FACE>(
                i,j,k, i_c,j_c,k_c, e_id, true  
            ) ; 
            size_t i_f, j_f, k_f ; 
            transf.compute_indices<FACE>(
                2*i,2*j,2*k, i_f,j_f,k_f, e_id, false  
            ) ; 

            cbuf(i_c,j_c,k_c,iv,c_id) = 0.125 * (
                data(i_f     ,j_f     ,k_f     ,iv,q_id) +
                data(i_f+s[0],j_f     ,k_f     ,iv,q_id) +
                data(i_f     ,j_f+s[1],k_f     ,iv,q_id) +
                data(i_f     ,j_f     ,k_f+s[2],iv,q_id) +
                data(i_f+s[0],j_f+s[1],k_f     ,iv,q_id) +
                data(i_f+s[0],j_f     ,k_f+s[2],iv,q_id) +
                data(i_f     ,j_f+s[1],k_f+s[2],iv,q_id) +
                data(i_f+s[0],j_f+s[1],k_f+s[2],iv,q_id) 
            ) ;  
        }
         
    }
    #if 1
    // runs to ng/2 n/2 
    KOKKOS_INLINE_FUNCTION
    void operator() (ghost_restrict_edge_tag, size_t k, size_t iv, size_t iq) const 
    {
        auto q_id = qid(iq) ; 
        auto c_id = cbuf_id(iq) ; 

        auto e_id = elem_id(iq) ;

        int s[3] ; 
        transf.get_stencil<EDGE>(s, e_id) ; 
        // only ngz/2
        for( int j=0; j<transf.g/2; ++j) 
        for( int i=0; i<transf.g/2; ++i) {
            size_t i_c, j_c, k_c ; 
            transf.compute_indices<EDGE>(
                i,j,k, i_c,j_c,k_c, e_id, true  
            ) ; 
            size_t i_f, j_f, k_f ; 
            transf.compute_indices<EDGE>(
                2*i,2*j,2*k, i_f,j_f,k_f, e_id, false  
            ) ; 

            cbuf(i_c,j_c,k_c,iv,c_id) = 0.125 * (
                data(i_f     ,j_f     ,k_f     ,iv,q_id) +
                data(i_f+s[0],j_f     ,k_f     ,iv,q_id) +
                data(i_f     ,j_f+s[1],k_f     ,iv,q_id) +
                data(i_f     ,j_f     ,k_f+s[2],iv,q_id) +
                data(i_f+s[0],j_f+s[1],k_f     ,iv,q_id) +
                data(i_f+s[0],j_f     ,k_f+s[2],iv,q_id) +
                data(i_f     ,j_f+s[1],k_f+s[2],iv,q_id) +
                data(i_f+s[0],j_f+s[1],k_f+s[2],iv,q_id) 
            ) ;  
        }
         
    }
    #endif 
    // runs to ng/2 n/2 
    KOKKOS_INLINE_FUNCTION
    void operator() (ghost_restrict_corner_tag, size_t iv, size_t iq) const 
    {
        auto q_id = qid(iq) ; 
        auto c_id = cbuf_id(iq) ; 

        auto e_id = elem_id(iq) ;

        int s[3] ; 
        transf.get_stencil<CORNER>(s, e_id) ; 
        
        // only ngz/2
        for( int k=0; k<transf.g/2; ++k) 
        for( int j=0; j<transf.g/2; ++j) 
        for( int i=0; i<transf.g/2; ++i)  {
            size_t i_c, j_c, k_c ; 
            transf.compute_indices<CORNER>(
                i,j,k, i_c,j_c,k_c, e_id, true  
            ) ; 
            size_t i_f, j_f, k_f ; 
            transf.compute_indices<CORNER>(
                2*i,2*j,2*k, i_f,j_f,k_f, e_id, false  
            ) ; 

            cbuf(i_c,j_c,k_c,iv,c_id) = 0.125 * (
                data(i_f     ,j_f     ,k_f     ,iv,q_id) +
                data(i_f+s[0],j_f     ,k_f     ,iv,q_id) +
                data(i_f     ,j_f+s[1],k_f     ,iv,q_id) +
                data(i_f     ,j_f     ,k_f+s[2],iv,q_id) +
                data(i_f+s[0],j_f+s[1],k_f     ,iv,q_id) +
                data(i_f+s[0],j_f     ,k_f+s[2],iv,q_id) +
                data(i_f     ,j_f+s[1],k_f+s[2],iv,q_id) +
                data(i_f+s[0],j_f+s[1],k_f+s[2],iv,q_id) 
            ) ; 
        }
         
    }

} ; 

/**
 * @brief Restrict inside ghostzones.
 * 
 * @tparam view_t Type of data array.
 */
template<  int stag_dir, typename view_t > 
struct div_free_ghost_restrict_op {

    view_t data, cbuf ; //!< Data and coarse buffer arrays.
    readonly_view_t<size_t> qid, cbuf_id ; //!< Data and coarse buffer quad-ids 
    readonly_view_t<uint8_t> elem_id ; //!< Element ids

    prolong_index_transformer_t transf ;  //!< Index transformations
    
    div_free_ghost_restrict_op(
        view_t _data, view_t _cbuf,
        Kokkos::View<size_t*> _qid, 
        Kokkos::View<size_t*> _cbuf_id,
        Kokkos::View<uint8_t*> _eid,  
        size_t n, size_t _ngz, var_staggering_t stag
    ) : data(_data)
      , cbuf(_cbuf)
      , qid(_qid)
      , cbuf_id(_cbuf_id)
      , elem_id(_eid)
      , transf(n,_ngz,stag)
    {}

    // range check: all the non-gz loops are extended by 
    // 1 since they need to apply to all face/edge orientation.
    // we need to check if we are in range before writing
    // NOTE: the following function assumes that transf has full
    // n and not n/2.. therefore it must be compared with i_f... not i_c...
    template< element_kind_t elem_kind >
    KOKKOS_INLINE_FUNCTION
    bool in_range(size_t i, size_t j, size_t k, int8_t ie) const {
        size_t sx = (stag_dir==0); 
        size_t sy = (stag_dir==1);
        size_t sz = (stag_dir==2);
        size_t nx  = transf.n/2 ; 
        size_t ny  = transf.n/2 ; 
        size_t nz  = transf.n/2 ;
        size_t ngz = transf.g ;
        if constexpr ( elem_kind == FACE ) {
            const int axis = ie / 2;
            const int side = ie % 2; 
            if ( axis == 0 ) { // across X - face
                bool gz_in_range = side ? i < nx + ngz + ngz/2 + sx : i < ngz+sx ; 
                return (j >= ngz and j<ny+sy+ngz) and ( k>=ngz and k<nz+sz+ngz) and gz_in_range ; 
            } else if ( axis == 1 ) {
                bool gz_in_range = side ? j < ny + ngz + ngz/2 + sy : j < ngz+sy ; 
                return (i>=ngz and i<nx+sx+ngz) and (k>=ngz and k<nz+sz+ngz) and gz_in_range ;
            } else {
                bool gz_in_range = side ? k < nz + ngz + ngz/2 + sz : k < ngz+sz ; 
                return (i>=ngz and i<nx+sx+ngz) and (j>=ngz and j<ny+sy+ngz) and gz_in_range ; 
            }
        } else if constexpr ( elem_kind == EDGE ) {
            const int side1 = (ie>>0)&1 ;
            const int side2 = (ie>>1)&1 ; 
            if ( ie < 4 ) {
                bool gz_in_range1 = side1 ? j < ny + ngz + ngz/2 + sy : j < ngz+sy ;
                bool gz_in_range2 = side2 ? k < nz + ngz + ngz/2 + sz : k < ngz+sz ; 
                return (i>=ngz and i<nx+sx+ngz) and gz_in_range1 and gz_in_range2 ; 
            } else if ( ie < 8 ) {
                bool gz_in_range1 = side1 ? i < nx + ngz + ngz/2 + sx : i < ngz+sx ; 
                bool gz_in_range2 = side2 ? k < nz + ngz + ngz/2 + sz : k < ngz+sz ;
                return (j>=ngz and j<ny+sy+ngz) and gz_in_range1 and gz_in_range2 ; 
            } else {
                bool gz_in_range1 = side1 ? i < nx + ngz + ngz/2 + sx : i < ngz+sx ; 
                bool gz_in_range2 = side2 ? j < ny + ngz + ngz/2 + sy : j < ngz+sy ;
                return (k>=ngz and k<nz+sz+ngz) and gz_in_range1 and gz_in_range2 ;
            }
        } else {
            const int side1 = (ie>>0)&1 ;
            const int side2 = (ie>>1)&1 ; 
            const int side3 = (ie>>2)&1 ; 
            bool gz_in_range1 = side1 ? i < nx + ngz + ngz/2 + sx : i < ngz+sx ; 
            bool gz_in_range2 = side2 ? j < ny + ngz + ngz/2 + sy : j < ngz+sy ;
            bool gz_in_range3 = side3 ? k < nz + ngz + ngz/2 + sz : k < ngz+sz ;
            return gz_in_range1 and gz_in_range2 and gz_in_range3 ; 
        }
        
    }

    // Note that the loops in the GZ for fields 
    // whose staggering aligns with the loop 
    // direction are extended by 1. this is needed
    // since the limited slopes in the prolongation 
    // have a stencil of +- 1 in each direction 
    // orthogonal to the staggering. For this reason 
    // in the prolongation index transformer the staggering 
    // acts in the opposite way as the normal index transf. 
    // Instead of shifting up in the upper gz it shifts down
    // in the lower ones. 

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        data = alias.get<stag>() ; 
    }
    // runs to n/2 n/2 
    KOKKOS_INLINE_FUNCTION
    void operator() (ghost_restrict_face_tag,  size_t j, size_t k, size_t iv, size_t iq) const 
    {
        auto q_id = qid(iq) ; 
        auto c_id = cbuf_id(iq) ; 

        auto e_id = elem_id(iq) ;
        auto _data = data ;
        // this provides the stencil sign 
        int s[3] ; 
        transf.get_stencil<FACE>(s, e_id) ; 

        s[0] = stag_dir == 0 ? 0 : s[0] ; 
        s[1] = stag_dir == 1 ? 0 : s[1] ; 
        s[2] = stag_dir == 2 ? 0 : s[2] ; 

        


        auto const compute_restricted_val = [=] (size_t i_f, size_t j_f, size_t k_f)
        {
            double val = 0 ; 
            for( int ii=0; ii<=(stag_dir!=0); ++ ii ) {
                for( int jj=0; jj<=(stag_dir!=1); ++jj ) {
                    for(int kk=0; kk<=(stag_dir!=2); ++kk) {
                        val += _data(i_f+ii*s[0],j_f+jj*s[1],k_f+kk*s[2],iv,q_id) ; 
                    }
                }
            }
            return 0.25 * val ;
        } ; 
        // loop in the ghostzones, only ngz/2 ( + 1 )
        for( int i=0; i<transf.g/2+1; ++i) {
            size_t i_c, j_c, k_c ; 
            transf.compute_indices<FACE>(
                i,j,k, i_c,j_c,k_c, e_id, true  
            ) ; 
            size_t i_f, j_f, k_f ; 
            transf.compute_indices<FACE>(
                2*i,2*j,2*k, i_f,j_f,k_f, e_id, false  
            ) ; 
            if ( in_range<FACE>(i_c,j_c,k_c,e_id) ) 
                cbuf(i_c,j_c,k_c,iv,c_id) = compute_restricted_val(i_f,j_f,k_f) ; 
        }
         
    }
    #if 1
    // runs to ng/2 n/2 
    KOKKOS_INLINE_FUNCTION
    void operator() (ghost_restrict_edge_tag, size_t k, size_t iv, size_t iq) const 
    {
        auto q_id = qid(iq) ; 
        auto c_id = cbuf_id(iq) ; 

        auto e_id = elem_id(iq) ;
        auto _data = data ;
        int s[3] ; 
        transf.get_stencil<EDGE>(s, e_id) ; 
        s[0] = stag_dir == 0 ? 0 : s[0] ; 
        s[1] = stag_dir == 1 ? 0 : s[1] ; 
        s[2] = stag_dir == 2 ? 0 : s[2] ; 

        auto const compute_restricted_val = [=] (size_t i_f, size_t j_f, size_t k_f)
        {
            double val = 0 ; 
            for( int ii=0; ii<=(stag_dir!=0); ++ ii ) {
                for( int jj=0; jj<=(stag_dir!=1); ++jj ) {
                    for(int kk=0; kk<=(stag_dir!=2); ++kk) {
                        val += _data(i_f+ii*s[0],j_f+jj*s[1],k_f+kk*s[2],iv,q_id) ; 
                    }
                }
            }
            return 0.25 * val ;
        } ; 

        // only ngz/2
        for( int j=0; j<transf.g/2+1; ++j) 
        for( int i=0; i<transf.g/2+1; ++i) {
            size_t i_c, j_c, k_c ; 
            transf.compute_indices<EDGE>(
                i,j,k, i_c,j_c,k_c, e_id, true  
            ) ; 
            size_t i_f, j_f, k_f ; 
            transf.compute_indices<EDGE>(
                2*i,2*j,2*k, i_f,j_f,k_f, e_id, false  
            ) ; 
            if ( in_range<EDGE>(i_c,j_c,k_c,e_id) ) 
                cbuf(i_c,j_c,k_c,iv,c_id) = compute_restricted_val(i_f,j_f,k_f) ; 
        }
         
    }
    #endif 
    // runs to ng/2 n/2 
    KOKKOS_INLINE_FUNCTION
    void operator() (ghost_restrict_corner_tag, size_t iv, size_t iq) const 
    {
        auto q_id = qid(iq) ; 
        auto c_id = cbuf_id(iq) ; 

        auto e_id = elem_id(iq) ;
        auto _data = data ; 

        int s[3] ; 
        transf.get_stencil<CORNER>(s, e_id) ; 
        s[0] = stag_dir == 0 ? 0 : s[0] ; 
        s[1] = stag_dir == 1 ? 0 : s[1] ; 
        s[2] = stag_dir == 2 ? 0 : s[2] ; 
        
        auto const compute_restricted_val = [=] (size_t i_f, size_t j_f, size_t k_f)
        {
            double val = 0 ; 
            for( int ii=0; ii<=(stag_dir!=0); ++ ii ) {
                for( int jj=0; jj<=(stag_dir!=1); ++jj ) {
                    for(int kk=0; kk<=(stag_dir!=2); ++kk) {
                        val += _data(i_f+ii*s[0],j_f+jj*s[1],k_f+kk*s[2],iv,q_id) ; 
                    }
                }
            }
            return 0.25 * val ;
        } ; 

        // only ngz/2
        for( int k=0; k<transf.g/2+1; ++k) 
        for( int j=0; j<transf.g/2+1; ++j) 
        for( int i=0; i<transf.g/2+1; ++i)  {
            size_t i_c, j_c, k_c ; 
            transf.compute_indices<CORNER>(
                i,j,k, i_c,j_c,k_c, e_id, true  
            ) ; 
            size_t i_f, j_f, k_f ; 
            transf.compute_indices<CORNER>(
                2*i,2*j,2*k, i_f,j_f,k_f, e_id, false  
            ) ; 
            // corner: check range
            if ( in_range<CORNER>(i_c,j_c,k_c,e_id))
                cbuf(i_c,j_c,k_c,iv,c_id) = compute_restricted_val(i_f,j_f,k_f) ; 
        }
         
    }

} ; 

}} /* namespace grace::amr */

#endif /* GRACE_AMR_BC_RESTRICT_HH */