/**
 * @file bc_copy_ghostzones.hh
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
    simple_face_info_t info; 

    std::size_t VEC(nx, ny, nz), ngz; 

    face_copy_k(
        ViewB_t _dest_view,
        ViewA_t _src_view,
        Kokkos::View<simple_face_info_t*> _info,
        VEC( std::size_t _nx, std::size_t _ny, std::size_t _nz),
        std::size_t _ngz 
    ) : src_view(_src_view)
      , dest_view(_dest_view)
      , info(_info)
      , VEC(nx(_nx), ny(_ny), nz(_nz))
      , ngz(_ngz)
    {}

    KOKKOS_INLINE_FUNCTION
    void compute_face_offset_p(std::size_t ig, std::size_t j, std::size_t k,
                               std::size_t& i_out, std::size_t& j_out, std::size_t& k_out, int face) {
        // _p stands for physical, this returns the points on the inside of the domain
        switch (face) {
            case 0:
            i_out = ngz + ig ; 
            j_out = j + ngz  ; 
            k_out = k + ngz  ;
            break ; 
            case 1:
            i_out = nx + ig ; 
            j_out = j + ngz  ; 
            k_out = k + ngz  ;
            break ; 
            case 2:
            i_out = j + ngz ; 
            j_out = ngz + ig ; 
            k_out = k + ngz  ;
            break ; 
            case 3:
            i_out = j + ngz ; 
            j_out = ny + ig ;
            k_out = k + ngz  ; 
            break ; 
            case 4:
            i_out = j + ngz ; 
            j_out = k + ngz ; 
            k_out = ngz + ig ; 
            break ; 
            case 5:
            i_out = j + ngz ; 
            j_a = k + ngz ; 
            k_out = nz + ig ;
            break ; 
            default: 
            break ; 
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_face_offset_g(std::size_t ig, std::size_t j, std::size_t k,
                               std::size_t& i_out, std::size_t& j_out, std::size_t& k_out, int face) {
        // _g stands for ghost, this returns the points on the outside of the domain
        switch (face) {
            case 0:
            i_out = ig ; 
            j_out = j + ngz  ; 
            k_out = k + ngz  ; 
            break ; 
            case 1:
            i_out = nx + ngz + ig ; 
            j_out = j + ngz  ; 
            k_out = k + ngz  ;
            break ; 
            case 2:
            i_out = j + ngz ; 
            j_out = ig ; 
            k_out = k + ngz  ;
            break ; 
            case 3:
            i_out = j + ngz ; 
            j_out = ny + ngz + ig ;
            k_out = k + ngz  ;
            break ; 
            case 4:
            i_out = j + ngz ; 
            j_out = k + ngz ; 
            k_out = ig ;
            break ; 
            case 5:
            i_out = j + ngz ; 
            j_a   = k + ngz ; 
            k_out = nz + ngz + ig ;
            break ; 
            default:
            break ; 
        }
    }


    KOKKOS_INLINE_FUNCTION 
    void operator() (
        std::size_t ig, VECD(std::size_t j, std::size_t k), size_t ivar, size_t iq
    ) {
        auto const src_face  = info.face_a_d(iq) ; 
        auto const dest_face = info.face_b_d(iq) ; 

        auto const src_q  = info.qid_a_d(iq)  ; 
        auto const dest_q = info.qid_b_d(iq) ;


        std::size_t VEC(i_a,j_a,k_a), VEC(i_b,j_b,k_b) ; 
        compute_face_offset_p(
            ig, VECD(j, k), i_a, j_a, k_a, src_face
        ) ; 
        compute_face_offset_g(
            ig, VECD(j, k), i_b, j_b, k_b, dest_face 
        )

        dest_view(
            VEC(i_b,j_b,k_b), ivar, dest_q 
        ) = src_view(VEC(i_a,j_a,k_a), ivar, src_q) ;
        
        if constexpr( do_reciprocal_copy ){
            compute_face_offset_p(
            ig, VECD(j, k), i_a, j_a, k_a, dest_face 
            ) ; 
            compute_face_offset_g(
                ig, VECD(j, k), i_b, j_b, k_b, src_face
            )
            src_view(
                VEC(i_b,j_b,k_b), ivar, dest_q 
            ) = dest_view(VEC(i_a,j_a,k_a), ivar, src_q) ;
        }
    }
} ; 
    
}} /* namespace grace::amr */


#endif /* GRACE_AMR_BC_COPY_GHOSTZONES_HH */