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

#include <grace_config.h>

#include <grace/amr/amr_functions.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

KOKKOS_INLINE_FUNCTION 
void compute_phys_indices_face(
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, std::size_t const& g, 
    std::size_t const& ig, std::size_t const& j, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int face
)
{
    const int axis = face / 2;   // 0 = x, 1 = y, 2 = z
    const int side = face % 2;   // 0 = low, 1 = high

    if (axis == 0) { // X-faces
        i_out = side ? nx + ig : g + ig;
        j_out = g + j;
        k_out = g + k;
    } else if (axis == 1) { // Y-faces
        i_out = g + j;
        j_out = side ? ny + ig : g + ig;
        k_out = g + k;
    } else { // Z-faces
        i_out = g + j;
        j_out = g + k;
        k_out = side ? nz + ig : g + ig;
    }
}

KOKKOS_INLINE_FUNCTION 
void compute_ghost_indices_face(
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, std::size_t const& g, 
    std::size_t const& ig, std::size_t const& j, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int face
)
{
    const int axis = face / 2;
    const int side = face % 2;

    if (axis == 0) { // X-faces
        i_out = side ? nx + g + ig : ig;
        j_out = g + j;
        k_out = g + k;
    } else if (axis == 1) { // Y-faces
        i_out = g + j;
        j_out = side ? ny + g + ig : ig;
        k_out = g + k;
    } else { // Z-faces
        i_out = g + j;
        j_out = g + k;
        k_out = side ? nz + g + ig : ig;
    }
}

KOKKOS_INLINE_FUNCTION 
void compute_phys_indices_edge(
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, std::size_t const& g, 
    std::size_t const& ig, std::size_t const& jg, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int edge
)
{

    if (edge < 4) {
        // X-axis edges
        int y_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        i_out = g + k;                          // varies
        j_out = y_off ? ny + ig - 1 : g + ig;   // fixed
        k_out = z_off ? nz + jg - 1 : g + jg;   // fixed
    }
    else if (edge < 8) {
        // Y-axis edges
        int x_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        i_out = x_off ? nx + ig - 1 : g + ig;   // fixed
        j_out = g + k;                          // varies
        k_out = z_off ? nz + jg - 1 : g + jg;   // fixed
    }
    else {
        // Z-axis edges
        int x_off = (edge >> 0) & 1;
        int y_off = (edge >> 1) & 1;
        i_out = x_off ? nx + ig - 1 : g + ig;   // fixed
        j_out = y_off ? ny + jg - 1 : g + jg;   // fixed
        k_out = g + k;                          // varies
    }
}
KOKKOS_INLINE_FUNCTION 
void compute_ghost_indices_edge(
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, std::size_t const& g, 
    std::size_t const& ig, std::size_t const& jg, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int edge
)
{

    if (edge < 4) {
        // X-axis edges
        int y_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        i_out = g + k;                      // varies
        j_out = y_off ? ny + g + ig : ig;   // fixed
        k_out = z_off ? nz + g + jg : jg;   // fixed
    }
    else if (edge < 8) {
        // Y-axis edges
        int x_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        i_out = x_off ? nx + g + ig : ig;   // fixed
        j_out = g + k;                      // varies
        k_out = z_off ? nz + g + jg : jg;   // fixed
    }
    else {
        // Z-axis edges
        int x_off = (edge >> 0) & 1;
        int y_off = (edge >> 1) & 1;
        i_out = x_off ? nx + g + ig : ig;   // fixed
        j_out = y_off ? ny + g + jg : jg;   // fixed
        k_out = g + k;                      // varies
    }
}

KOKKOS_INLINE_FUNCTION 
void compute_phys_indices_corner(
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, std::size_t const& g, 
    std::size_t const& ig, std::size_t const& j, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int corner
)
{
    int x_off = (corner) & 1 ; 
    int y_off = (corner >> 1) & 1 ; 
    int z_off = (corner >> 2) & 1 ; 

    i_out = x_off ? nx + i - 1 : g + i ; 
    j_out = y_off ? ny + j - 1 : g + j ; 
    k_out = z_off ? nz + k - 1 : g + k ; 
}

KOKKOS_INLINE_FUNCTION 
void compute_ghost_indices_corner(
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, std::size_t const& g, 
    std::size_t const& ig, std::size_t const& j, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int corner
)
{
    int x_off = (corner) & 1 ; 
    int y_off = (corner >> 1) & 1 ; 
    int z_off = (corner >> 2) & 1 ; 

    i_out = x_off ? nx + g + i :  i ; 
    j_out = y_off ? ny + g + j :  j ; 
    k_out = z_off ? nz + g + k :  k ; 
}

enum element_kind_t : uint8_t {
    FACE, EDGE, CORNER 
} ; 

struct index_transformer_t {
    
    index_transformer_t(std::size_t _nx, std::size_t _ny,
                             std::size_t _nz, std::size_t _ngz)
        : nx(_nx), ny(_ny), nz(_nz), ngz(_ngz) 
    {}
    

    std::size_t nx, ny, nz, ngz;

    index_transformer_t(std::size_t _nx, std::size_t _ny,
                        std::size_t _nz, std::size_t _ngz)
        : nx(_nx), ny(_ny), nz(_nz), ngz(_ngz) {}
    

    // Unified entry point
    template< element_kind_t elem_kind 
            , bool is_phys >
    KOKKOS_INLINE_FUNCTION
    void compute_indices(std::size_t ig, std::size_t j, std::size_t k,
                         std::size_t& i_out, std::size_t& j_out,
                         std::size_t& k_out, int ielem) const
    {

        if constexpr ( elem_kind == element_kind_t::FACE ) {
            if constexpr ( is_phys ) {
                compute_phys_indices_face(nx,ny,nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            } else {
                compute_ghost_indices_face(nx,ny,nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            }
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            if constexpr ( is_phys ) {
                compute_phys_indices_edge(nx,ny,nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            } else {
                compute_ghost_indices_edge(nx,ny,nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            }
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            if constexpr ( is_phys ) {
                compute_phys_indices_corner(nx,ny,nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            } else {
                compute_ghost_indices_corner(nx,ny,nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            }
        }
    }


} ; 

}} /* namespace grace::amr */
#endif /* GRACE_AMR_INDEX_HELPERS_HH */