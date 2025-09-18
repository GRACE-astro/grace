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

#ifndef GRACE_AMR_BC_COPY_GHOSTZONES_HH
#define GRACE_AMR_BC_COPY_GHOSTZONES_HH 

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

template< element_kind_t elem_kind, typename view_t > 
struct restrict_k {

    view_t src_view, dest_view ; 
    readonly_view_t<size_t> src_q, dest_q ; 
    size_t ngz ;

    KOKKOS_INLINE_FUNCTION
    void operator() (size_t i, size_t j, size_t k, size_t iv, size_t iq)
    {
        auto src_qid = src_q(iq) ; 
        auto dst_qid = dst_q(iq) ; 

        dest_view(i+ngz,j+ngz,k+ngz,iv,dst_qid) = (
            src_view(2*i+ngz,2*j+ngz,2*k+ngz) + 
            src_view(2*i+ngz+1,2*j+ngz,2*k+ngz) + 
            src_view(2*i+ngz,2*j+ngz+1,2*k+ngz) + 
            src_view(2*i+ngz,2*j+ngz,2*k+ngz+1) + 
            src_view(2*i+ngz+1,2*j+ngz+1,2*k+ngz) +
            src_view(2*i+ngz+1,2*j+ngz,2*k+ngz+1) +  
            src_view(2*i+ngz,2*j+ngz+1,2*k+ngz+1) +
            src_view(2*i+ngz+1,2*j+ngz+1,2*k+ngz+1)  
        ) / 8 ; 
    }

} ; 

}} /* namespace grace::amr */

#endif /* GRACE_AMR_BC_COPY_GHOSTZONES_HH */