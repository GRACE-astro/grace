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

#ifndef GRACE_AMR_BC_PROLONG_HH
#define GRACE_AMR_BC_PROLONG_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {


template< typename interpolator, element_kind_t elem_kind, typename view_t > 
struct prolong_op {

    view_t view, cbuf ; 
    readonly_view_t<size_t> view_qid, cbuf_qid ; 
    readonly_view_t<uint8_t> eid ; 

    prolong_index_transformer_t transf; 

    template< var_staggering_t stag >
    void set_data_ptr(view_alias_t alias) 
    {
        view = alias.get<stag>() ; 
    }

    prolong_op(
        view_t _view, view_t _cbuf, 
        Kokkos::View<size_t*> _view_qid,
        Kokkos::View<size_t*> _cbuf_qid,
        Kokkos::View<uint8_t*> _eid,
        size_t n, size_t ngz 
    ) : view(_view), cbuf(_cbuf)
      , view_qid(_view_qid)
      , cbuf_qid(_cbuf_qid)
      , eid(_eid)
      , transf(n,ngz)
    {} 

    // this loop goes full nx 
    KOKKOS_INLINE_FUNCTION
    void operator() (size_t i, size_t j, size_t k, size_t iv, size_t iq) const 
    {
        auto qid = view_qid(iq) ; 
        auto cid = cbuf_qid(iq) ; 
        auto e_id = eid(iq) ; 
        
        

        // transform
        size_t i_c,j_c,k_c ; 
        transf.compute_indices<elem_kind>(
            i/2,j/2,k/2, i_c,j_c,k_c, e_id, true /* half nx */
        ) ; 

        size_t i_f,j_f,k_f ; 
        transf.compute_indices<elem_kind>(
            i,j,k, i_f,j_f,k_f, e_id, false 
        ) ;

        int signs[3] ;
        transf.get_signs<elem_kind>(
            i,j,k, signs, e_id 
        ) ; 

        view(VEC(i_f,j_f,k_f),iv,qid) = interpolator::interpolate(
                                            VEC(i_c,j_c,k_c),
                                            cid,iv, 
                                            VEC(signs[0],signs[1],signs[2]),
                                            cbuf
                                        ) ; 
        
    }

} ; 

} }

#endif 