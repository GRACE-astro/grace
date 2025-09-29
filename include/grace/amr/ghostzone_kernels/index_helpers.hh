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

template< amr::element_kind_t elem_kind >
Kokkos::Array<int64_t, 5> get_iter_range(size_t ngz,size_t _nx, size_t nv, size_t nq,  bool offset=false) {
    int64_t const nx = offset ? static_cast<int64_t>(_nx + ngz) : static_cast<int64_t>( _nx) ; 
    if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
        return Kokkos::Array<int64_t, 5>{static_cast<int64_t>(ngz),nx,nx,static_cast<int64_t>(nv),static_cast<int64_t>(nq)} ; 
    } else if constexpr  ( elem_kind == amr::element_kind_t::EDGE ) {
        return Kokkos::Array<int64_t, 5>{static_cast<int64_t>(ngz),static_cast<int64_t>(ngz),nx,static_cast<int64_t>(nv),static_cast<int64_t>(nq)} ; 
    } else {
        return Kokkos::Array<int64_t, 5>{static_cast<int64_t>(ngz),static_cast<int64_t>(ngz),static_cast<int64_t>(ngz),static_cast<int64_t>(nv),static_cast<int64_t>(nq)} ; 
    }
}



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
        j_out = y_off ? ny + ig : g + ig;   // fixed
        k_out = z_off ? nz + jg : g + jg;   // fixed
    }
    else if (edge < 8) {
        // Y-axis edges
        int x_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        i_out = x_off ? nx + ig : g + ig;   // fixed
        j_out = g + k;                          // varies
        k_out = z_off ? nz + jg : g + jg;   // fixed
    }
    else {
        // Z-axis edges
        int x_off = (edge >> 0) & 1;
        int y_off = (edge >> 1) & 1;
        i_out = x_off ? nx + ig : g + ig;   // fixed
        j_out = y_off ? ny + jg : g + jg;   // fixed
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

    i_out = x_off ? nx + i  : g + i ; 
    j_out = y_off ? ny + j  : g + j ; 
    k_out = z_off ? nz + k  : g + k ; 
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
void _get_signs_face(
    std::size_t const& ig, std::size_t const& j, std::size_t const& k,
    int face, int signs[3]
) 
{
    const int axis = face / 2;
    const int side = face % 2;

    if (axis == 0) { // X-faces
        signs[0] = (ig%2==1) - (ig%2==0) ; 
        signs[1] = (j%2==1) - (j%2==0) ;
        signs[2] = (k%2==1) - (k%2==0) ;
        signs[0] *= side ? +1 : -1 ; 
    } else if (axis == 1) { // Y-faces
        signs[1] = (ig%2==1) - (ig%2==0) ; 
        signs[0] = (j%2==1) - (j%2==0) ;
        signs[2] = (k%2==1) - (k%2==0) ;
        signs[1] *= side ? +1 : -1 ; 
    } else { // Z-faces
        signs[2] = (ig%2==1) - (ig%2==0) ; 
        signs[0] = (j%2==1) - (j%2==0) ;
        signs[1] = (k%2==1) - (k%2==0) ;
        signs[2] *= side ? +1 : -1 ; 
    }
}
KOKKOS_INLINE_FUNCTION
void _get_stencil_face(int stencil[3], int8_t face) {
    const int axis = face / 2;
    const int side = face % 2;

    if (axis == 0) { 
        // X-faces
        stencil[0] = side ? +1 : -1 ; 
        stencil[1] = stencil[2] = + 1; 
    } else if (axis==1) {
        // Y-faces
        stencil[1] = side ? +1 : -1 ; 
        stencil[0] = stencil[2] = + 1; 
    } else {
        // Z-faces
        stencil[2] = side ? +1 : -1 ; 
        stencil[0] = stencil[1] = + 1; 
    }
}


KOKKOS_INLINE_FUNCTION 
void _compute_ghost_indices_face_invert(
    std::size_t const& n, std::size_t const& g, 
    std::size_t const& ig, std::size_t const& j, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int face
)
{
    const int axis = face / 2;
    const int side = face % 2;

    if (axis == 0) { // X-faces
        i_out = side ? n + g + ig : g - ig - 1 ;
        j_out = g + j;
        k_out = g + k;
    } else if (axis == 1) { // Y-faces
        i_out = g + j;
        j_out = side ? n + g + ig : g - ig - 1 ;
        k_out = g + k;
    } else { // Z-faces
        i_out = g + j;
        j_out = g + k;
        k_out = side ? n + g + ig : g - ig - 1 ;
    }
}

KOKKOS_INLINE_FUNCTION
void _get_signs_edge(
    std::size_t const& i, std::size_t const& j, std::size_t const& k,
    int edge, int signs[3]
) 
{
    if (edge < 4) {
        // X-axis edges
        int y_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        signs[0] = (k%2==1) - (k%2==0) ; 
        signs[1] = (i%2==1) - (i%2==0) ;
        signs[2] = (j%2==1) - (j%2==0) ;

        signs[1] *= y_off ? +1 : -1 ; 
        signs[2] *= z_off ? +1 : -1 ; 
    } else if ( edge < 8 ) {
        // Y-axis edges
        int x_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        signs[0] = (i%2==1) - (i%2==0) ; 
        signs[1] = (k%2==1) - (k%2==0) ;
        signs[2] = (j%2==1) - (j%2==0) ;

        signs[0] *= x_off ? +1 : -1 ; 
        signs[2] *= z_off ? +1 : -1 ; 
    } else {
        // Z-axis edges
        int x_off = (edge >> 0) & 1;
        int y_off = (edge >> 1) & 1;
        signs[0] = (i%2==1) - (i%2==0) ; 
        signs[1] = (j%2==1) - (j%2==0) ;
        signs[2] = (k%2==1) - (k%2==0) ;

        signs[0] *= x_off ? +1 : -1 ; 
        signs[1] *= y_off ? +1 : -1 ; 
    }
}
KOKKOS_INLINE_FUNCTION
void _get_stencil_edge(int stencil[3], int8_t edge) {

    if (edge < 4) {
        // X-axis edges
        int y_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        stencil[0] = +1 ; 
        stencil[1] = y_off ? +1 : -1 ; 
        stencil[2] = z_off ? +1 : -1 ;
    } else if (edge < 8) {
        // Y-axis edges
        int x_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        stencil[0] = x_off ? +1 : -1 ; 
        stencil[1] = +1 ; 
        stencil[2] = z_off ? +1 : -1 ;
    } else {
        // Z-axis edges
        int x_off = (edge >> 0) & 1;
        int y_off = (edge >> 1) & 1;
        stencil[0] = x_off ? +1 : -1 ; 
        stencil[1] = y_off ? +1 : -1 ; 
        stencil[2] = +1 ; 
    }
}

KOKKOS_INLINE_FUNCTION 
void _compute_ghost_indices_edge_invert(
    std::size_t const& n, std::size_t const& g, 
    std::size_t const& ig, std::size_t const& jg, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int edge
)
{

    if (edge < 4) {
        // X-axis edges
        int y_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        i_out = g + k;                              // varies
        j_out = y_off ? n + g + ig : g - ig - 1 ;   // fixed
        k_out = z_off ? n + g + jg : g - jg - 1 ;   // fixed
    } else if (edge < 8) {
        // Y-axis edges
        int x_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        i_out = x_off ? n + g + ig : g - ig - 1 ;   // fixed
        j_out = g + k;                              // varies
        k_out = z_off ? n + g + jg : g - jg - 1 ;   // fixed
    } else {
        // Z-axis edges
        int x_off = (edge >> 0) & 1;
        int y_off = (edge >> 1) & 1;
        i_out = x_off ? n + g + ig : g - ig - 1 ;   // fixed
        j_out = y_off ? n + g + jg : g - jg - 1 ;   // fixed
        k_out = g + k;                              // varies
    }
}

KOKKOS_INLINE_FUNCTION
void _get_signs_corner(
    std::size_t const& i, std::size_t const& j, std::size_t const& k,
    int corner, int signs[3]
) 
{
    int x_off = (corner) & 1 ; 
    int y_off = (corner >> 1) & 1 ; 
    int z_off = (corner >> 2) & 1 ; 

    signs[0] = (i%2==1) - (i%2==0) ; 
    signs[1] = (j%2==1) - (j%2==0) ;
    signs[2] = (k%2==1) - (k%2==0) ;
    signs[0] *= x_off ? +1 : -1 ; 
    signs[1] *= y_off ? +1 : -1 ; 
    signs[2] *= z_off ? +1 : -1 ; 
}
KOKKOS_INLINE_FUNCTION
void _get_stencil_corner(int stencil[3], int8_t corner) {

    int x_off = (corner) & 1 ; 
    int y_off = (corner >> 1) & 1 ; 
    int z_off = (corner >> 2) & 1 ; 

    stencil[0] = x_off ? +1 : -1 ; 
    stencil[1] = y_off ? +1 : -1 ; 
    stencil[2] = z_off ? +1 : -1 ; 
}

KOKKOS_INLINE_FUNCTION 
void _compute_ghost_indices_corner_invert(
    std::size_t const& n, std::size_t const& g, 
    std::size_t const& i, std::size_t const& j, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int corner
)
{
    int x_off = (corner) & 1 ; 
    int y_off = (corner >> 1) & 1 ; 
    int z_off = (corner >> 2) & 1 ; 

    i_out = x_off ? n + g + i :  g - i - 1 ; 
    j_out = y_off ? n + g + j :  g - j - 1 ; 
    k_out = z_off ? n + g + k :  g - k - 1 ; 
}

struct prolong_index_transformer_t {
    std::size_t n, g;
    prolong_index_transformer_t(std::size_t _n,std::size_t  _ngz)
        : n(_n), g(_ngz) {}

    
    template< element_kind_t elem_kind >
    KOKKOS_INLINE_FUNCTION
    void compute_indices(std::size_t ig, std::size_t j, std::size_t k,
                         std::size_t& i_out, std::size_t& j_out,
                         std::size_t& k_out, int ielem, bool half_ncells=false) const
    {
        size_t _n = half_ncells ? n / 2 : n ; 
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            _compute_ghost_indices_face_invert(_n,g,ig,j,k,i_out,j_out,k_out,ielem);
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            _compute_ghost_indices_edge_invert(_n,g,ig,j,k,i_out,j_out,k_out,ielem);
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            _compute_ghost_indices_corner_invert(_n,g,ig,j,k,i_out,j_out,k_out,ielem);
        }
    }

    
    template< element_kind_t elem_kind >
    KOKKOS_INLINE_FUNCTION
    void get_signs( std::size_t ig, std::size_t j, std::size_t k,
                    int signs[3], int ielem ) const 
    {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            _get_signs_face(ig,j,k,ielem,signs) ; 
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            _get_signs_edge(ig,j,k,ielem,signs) ; 
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            _get_signs_corner(ig,j,k,ielem,signs) ; 
        }
    }

    template< element_kind_t elem_kind >
    KOKKOS_INLINE_FUNCTION
    void get_stencil( int stencil[3], int8_t ielem ) const 
    {   
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            _get_stencil_face(stencil,ielem) ; 
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            _get_stencil_edge(stencil,ielem) ; 
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            _get_stencil_corner(stencil,ielem) ; 
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
        size_t nx, uint8_t ichild )  ; 
} ; 

// FACE -> FACE  

template<>
struct cbuf_to_view_offsets<element_kind_t::FACE>
{
    KOKKOS_INLINE_FUNCTION
    static void get(
        size_t& j, size_t& k, 
        size_t nx, uint8_t ichild)  {
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
        size_t nx, uint8_t ichild)  {
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
        size_t nx, uint8_t ichild)  {
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
        size_t nx, uint8_t ichild )  ; 
} ; 

template<> 
struct view_to_cbuf_offsets<element_kind_t::FACE> {
    KOKKOS_INLINE_FUNCTION 
    static void get(
        size_t& j, size_t& k,
        size_t& j_c, size_t& k_c,
        size_t nx, size_t ngz, uint8_t ichild )  
    {
        j = (nx / 2 - ngz)* ( (ichild>>0) & 1 ) ; 
        k = (nx / 2 - ngz)* ( (ichild>>1) & 1 ) ; 
        j_c = (- ngz)* ( (ichild>>0) & 1 ) ; 
        k_c = (- ngz)* ( (ichild>>1) & 1 ) ; 
    }

} ;

template<> 
struct view_to_cbuf_offsets<element_kind_t::EDGE> {
    KOKKOS_INLINE_FUNCTION 
    static void get(
        size_t& j, size_t& k,
        size_t& j_c, size_t& k_c,
        size_t nx, size_t ngz, uint8_t ichild )  
    {
        j = 0 ; 
        k = (nx / 2 - ngz) * ( (ichild>>0) & 1 ) ;
        j_c = 0 ; 
        k_c = (- ngz)* ( (ichild>>0) & 1 ) ;  
    }

} ;

template<> 
struct view_to_cbuf_offsets<element_kind_t::CORNER> {
    KOKKOS_INLINE_FUNCTION 
    static void get(
        size_t& j, size_t& k,
        size_t& j_c, size_t& k_c,
        size_t nx, size_t ngz, uint8_t ichild )  
    {
        j = 0 ; j_c = 0 ; 
        k = 0 ; k_c = 0 ; 
    }

} ;

namespace detail {
constexpr std::array<std::array<int8_t,4>,P4EST_FACES> f2e = 
{{
    {{8,10,4,6}}, //0
    {{9,11,5,7}}, //1
    {{8,9,0,2}}, //2
    {{10,11,1,3}}, //3
    {{4,5,0,1}}, //4
    {{6,7,2,3}} //5 
}}; 
constexpr std::array<std::array<int8_t,2>,P4EST_FACES/2> face_axes = 
{{
    {{1,2}}, {{0,2}}, {{0,1}}
}} ;

inline constexpr std::array<std::array<uint8_t,2>,12> e2f = 
{{
    {{4,2}}, {{4,3}}, {{5,2}}, {{5,3}}, {{4,0}}, {{4,1}}, {{5,0}}, {{5,1}}, {{2,0}}, {{2,1}}, {{3,0}}, {{3,1}}
}}  ;
inline constexpr std::array<std::array<uint8_t,2>,12> e2c = 
{{
    {{0,1}}, {{2,3}}, {{4,5}}, {{6,7}}, {{0,2}}, {{1,3}}, {{4,6}}, {{5,7}}, {{0,4}}, {{1,5}}, {{2,6}}, {{3,7}}
}}  ;

inline constexpr std::array<std::array<uint8_t,3>,P4EST_CHILDREN> c2e = 
{{
    {{0,4,8}},  //0
    {{1,5,9}},  //1
    {{1,4,10}}, //2
    {{5,1,11}}, //3
    {{2,6,8}},  //4 
    {{2,7,9}},  //5
    {{3,6,10}}, //6
    {{3,7,11}}  //7
} } ;

} /*namespace detail*/

}} /* namespace grace::amr */
#endif /* GRACE_AMR_INDEX_HELPERS_HH */