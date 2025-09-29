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

    void set_data_ptr(view_alias_t alias) {
        src_view = alias.get() ; 
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
      , transf(n,_ngz)
    {}

    void set_data_ptr(view_alias_t alias) {
        data = alias.get() ; 
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

}} /* namespace grace::amr */

#endif /* GRACE_AMR_BC_RESTRICT_HH */