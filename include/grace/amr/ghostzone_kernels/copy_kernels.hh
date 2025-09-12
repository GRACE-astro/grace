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

#include <grace/utils/device.hh>
#include <grace/utils/inline.hh>

#include <index_helpers.hh>
#include <type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

template< 
    bool do_reciprocal_copy
    typename ViewA_t,
    typename ViewB_t 
>
struct face_copy_k {

    ViewA_t src_view ; 
    ViewB_t dest_view; 
    readonly_view_t<std::size_t> src_qid, dest_qid ; 
    readonly_view_t<uint8_t> src_face_view, dest_face_view; 

    std::size_t VEC(nx, ny, nz), ngz; 

    face_index_transformer_t transf ; 

    face_copy_k(
        ViewB_t _src_view,
        ViewA_t _dest_view,
        quad_view_t _src_qid, quad_view_t _dest_qid,
        face_view_t _src_face, face_view_t _dest_face, 
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz, std::size_t _nvars
    ) : src_view(_src_view)
      , dest_view(_dest_view)
      , src_qid(_src_qid)
      , dest_qid(_dest_qid)
      , src_face_view(_src_face)
      , dest_face_view(_dest_face)
      , VEC(nx(_nx), ny(_ny), nz(_nz))
      , ngz(_ngz)
      , transf(VEC(nx,ny,nz),ngz)
    {}


    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) {
        auto const src_face  = src_face_view(iq) ; 
        auto const dest_face = dest_face_view(iq) ; 

        auto const src_q  = src_qid(iq)  ; 
        auto const dest_q = dest_qid(iq) ;


        std::size_t VEC(i_a,j_a,k_a), VEC(i_b,j_b,k_b) ; 
        transf.compute_phys_indices<true>(
            ig, VECD(j, k), i_a, j_a, k_a, src_face
        ) ; 
        transf.compute_ghost_indices<true>(
            ig, VECD(j, k), i_a, j_a, k_a, src_face
        ) ; 

        dest_view(
            VEC(i_b,j_b,k_b), ivar, dest_q 
        ) = src_view(VEC(i_a,j_a,k_a), ivar, src_q) ;
        
        if constexpr( do_reciprocal_copy ){
            
            transf.compute_phys_indices<true>(
            ig, VECD(j, k), i_a, j_a, k_a, dest_face 
            ) ; 
            transf.compute_ghost_indices<true>(
                ig, VECD(j, k), i_b, j_b, k_b, src_face
            ) ; 
            src_view(
                VEC(i_b,j_b,k_b), ivar, dest_q 
            ) = dest_view(VEC(i_a,j_a,k_a), ivar, src_q) ;
        }
    }
} ; 
    
}} /* namespace grace::amr */


#endif /* GRACE_AMR_BC_COPY_GHOSTZONES_HH */