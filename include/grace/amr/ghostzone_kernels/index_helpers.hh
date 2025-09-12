/**
 * @file index_helpers.hh
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
#ifndef GRACE_AMR_INDEX_HELPERS_HH
#define GRACE_AMR_INDEX_HELPERS_HH


namespace grace { namespace amr {

struct face_index_transformer_t 
{
    std::size_t VEC(nx,ny,nz), _ngz ; 

    face_index_transformer_t(
        VEC(size_t _nx,size_t _ny, size_t _nz), ngz
    )
    : VEC(nx(_nx),ny(_ny),nz(_nz)), _ngz(ngz)
    {}

    template< bool offset_by_ngz >
    KOKKOS_INLINE_FUNCTION
    compute_phys_indices(
        std::size_t ig, std::size_t j, std::size_t k,
        std::size_t& i_out, std::size_t& j_out, std::size_t& k_out, int face ) const 
    {
        std::size_t ngz = offset_by_ngz ? ngz : 0 ; 
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
            j_out = k + ngz ; 
            k_out = nz + ig ;
            break ; 
            default: 
            break ; 
        }
    }

    KOKKOS_INLINE_FUNCTION
    void compute_ghost_indices(
        std::size_t ig, std::size_t j, std::size_t k,
        std::size_t& i_out, std::size_t& j_out, std::size_t& k_out, int face) const 
    {
        // _g stands for ghost, this returns the points on the outside of the domain
        std::size_t ngz = offset_by_ngz ? ngz : 0 ; 
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
} ; 


}} /* namespace grace::amr */
#endif /* GRACE_AMR_INDEX_HELPERS_HH */