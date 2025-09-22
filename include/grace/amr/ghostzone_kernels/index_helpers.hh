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

enum element_kind_t : uint8_t {
    FACE=0, EDGE=1, CORNER=2
} ; 

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
    std::size_t const& i, std::size_t const& j, std::size_t const& k,
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
    std::size_t const& i, std::size_t const& j, std::size_t const& k,
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

struct index_transformer_t {
    

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
                         std::size_t& k_out, int ielem, bool half_ncells=false) const
    {
        size_t _nx = half_ncells ? nx / 2 : nx ; 
        size_t _ny = half_ncells ? ny / 2 : ny ; 
        size_t _nz = half_ncells ? nz / 2 : nz ; 
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            if constexpr ( is_phys ) {
                compute_phys_indices_face(_nx,_ny,_nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            } else {
                compute_ghost_indices_face(_nx,_ny,_nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            }
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            if constexpr ( is_phys ) {
                compute_phys_indices_edge(_nx,_ny,_nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            } else {
                compute_ghost_indices_edge(_nx,_ny,_nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            }
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            if constexpr ( is_phys ) {
                compute_phys_indices_corner(_nx,_ny,_nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            } else {
                compute_ghost_indices_corner(_nx,_ny,_nz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            }
        }
    }


} ; 


KOKKOS_INLINE_FUNCTION
int edge_to_face_dir(int face, int edge) {
    int edge_axis = edge / 4;    // 0=x,1=y,2=z
    int normal    = face / 2;    // 0=x,1=y,2=z

    // Collect the two tangential axes
    int t0 = (normal + 1) % 3;
    int t1 = (normal + 2) % 3;

    // Sort so j = min, k = max
    int j_axis = t0 < t1 ? t0 : t1;
    int k_axis = t0 < t1 ? t1 : t0;

    // Now map edge axis to j/k
    return (edge_axis == j_axis) ? 0 : 1;
}


template<
     element_kind_t elem_kind
>
struct cbuf_to_view_offsets {
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, uint8_t ichild ) const ; 
} ; 

// FACE -> FACE  

template<>
struct cbuf_to_view_offsets<element_kind_t::FACE>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, uint8_t ichild) const {
        j = nx / 2 * ( (ichild>>0) & 1 ) ; 
        k = nx / 2 * ( (ichild>>1) & 1 ) ; 
    }; 
} ; 

// EDGE -> EDGE  

template<>
struct cbuf_to_view_offsets<element_kind_t::EDGE>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, uint8_t ichild) const {
        j = 0 ; 
        k = nx / 2 * ( (ichild>>0) & 1 ) ; 
    }; 
} ; 

// CORNER -> CORNER 

template<>
struct cbuf_to_view_offsets<element_kind_t::CORNER>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, uint8_t ichild) const {
        j = 0 ; 
        k = 0 ; 
    }; 
} ;




template<
     element_kind_t elem_kind
>
struct view_to_cbuf_offsets {
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, uint8_t ichild ) const ; 
} ; 

template<> 
struct view_to_cbuf_offsets<element_kind_t::FACE> {
    KOKKOS_INLINE_FUNCTION 
    static void get(
        size_t& j, size_t& k,
        size_t nx, size_t ngz, uint8_t ichild ) const 
    {
        j = (nx / 2 - ngz)* ( (ichild>>0) & 1 ) ; 
        k = (nx / 2 - ngz)* ( (ichild>>1) & 1 ) ; 
    }

} ;

template<> 
struct view_to_cbuf_offsets<element_kind_t::EDGE> {
    KOKKOS_INLINE_FUNCTION 
    static void get(
        size_t& j, size_t& k,
        size_t nx, size_t ngz, uint8_t ichild ) const 
    {
        j = 0 ; 
        k = (nx / 2-ngz) * ( (ichild>>0) & 1 ) ; 
    }

} ;

template<> 
struct view_to_cbuf_offsets<element_kind_t::CORNER> {
    KOKKOS_INLINE_FUNCTION 
    static void get(
        size_t& j, size_t& k,
        size_t nx, size_t ngz, uint8_t ichild ) const 
    {
        j = 0 ; 
        k = 0 ; 
    }

} ;


#if 0
// EDGE -> FACE 

template<>
struct cbuf_to_view_offsets<element_kind_t::FACE, element_kind_t::EDGE>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const 
    {
        // Here: iedge=0 -> edge along j-direction
        //       iedge=1 -> edge along k-direction
        auto const iedge = edge_to_face_dir(ie_v,ie_c) ; 
        
        if ( iedge == 0 ) {
            j = ( ((ichild>>0) & 1) ? nx/2 : 0 )  ; 
            k = nx / 2 -  ngz * ((ichild>>1) & 1) ; 
        } else {
            k = ( ((ichild>>0) & 1) ? nx/2 : 0 )  ; 
            j = nx / 2 -  ngz * ((ichild>>1) & 1) ; 
        }
    }
};

// CORNER -> FACE 

template<>
struct cbuf_to_view_offsets<element_kind_t::FACE,element_kind_t::CORNER>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const {
        j = nx / 2 - ( (ichild<<0) & 1 ) * ngz ; 
        k = nx / 2 - ( (ichild<<1) & 1 ) * ngz ; 
    }; 
} ; 

// CORNER -> EDGE 

template<>
struct cbuf_to_view_offsets<element_kind_t::EDGE, element_kind_t::CORNER>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, size_t ngz, 
        uint8_t ichild, 
        uint8_t ie_v=0, uint8_t ie_c=0) const 
    {
        j = 0 ; 
        k = nx / 2 -  ngz * ichild ; 
    }
};

#endif 
}} /* namespace grace::amr */
#endif /* GRACE_AMR_INDEX_HELPERS_HH */