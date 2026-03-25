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
#include <grace/data_structures/variable_properties.hh>

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
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, 
    std::size_t const& sx, std::size_t const& sy, std::size_t const& sz, std::size_t const& g, 
    std::size_t const& ig, std::size_t const& j, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int face
)
{
    const int axis = face / 2;   // 0 = x, 1 = y, 2 = z
    const int side = face % 2;   // 0 = low, 1 = high

    if (axis == 0) { // X-faces
        i_out = side ? nx + ig : g + sx + ig;
        j_out = g + j;
        k_out = g + k;
    } else if (axis == 1) { // Y-faces
        i_out = g + j;
        j_out = side ? ny + ig : g + sy + ig;
        k_out = g + k;
    } else { // Z-faces
        i_out = g + j;
        j_out = g + k;
        k_out = side ? nz + ig : g + sz + ig;
    }
}

KOKKOS_INLINE_FUNCTION 
void compute_ghost_indices_face(
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, 
    std::size_t const& sx, std::size_t const& sy, std::size_t const& sz, std::size_t const& g, std::size_t const& g_off,
    std::size_t const& ig, std::size_t const& j, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int face
)
{
    const int axis = face / 2;
    const int side = face % 2;

    if (axis == 0) { // X-faces
        i_out = side ? nx + sx + g + ig : g_off + ig;
        j_out = g + j;
        k_out = g + k;
    } else if (axis == 1) { // Y-faces
        i_out = g + j;
        j_out = side ? ny + sy + g + ig : g_off + ig;
        k_out = g + k;
    } else { // Z-faces
        i_out = g + j;
        j_out = g + k;
        k_out = side ? nz + sz + g + ig : g_off + ig;
    }
}

KOKKOS_INLINE_FUNCTION
size_t _get_first_iter_index_face(
    int idir, int8_t ielem, size_t n, size_t g
) 
{
    size_t first_indices[3]  ;
    const int axis = ielem / 2;
    const int side = ielem % 2;

    if ( axis == 0 ) {
        first_indices[0] = side ? n + g : 0 ;
        first_indices[1] = g ; 
        first_indices[2] = g ; 
    } else if ( axis == 1 ) {
        first_indices[1] = side ? n + g : 0 ;
        first_indices[0] = g ; 
        first_indices[2] = g ; 
    } else {
        first_indices[2] = side ? n + g : 0 ;
        first_indices[0] = g ; 
        first_indices[1] = g ; 
    }
    return first_indices[idir] ; 
}

// this index is always the one of the face 
// center before the interior of the quadrant.
// we need to return this since this is where 
// an additional face at i+1/2 needs to be filled 
// by prolong.
KOKKOS_INLINE_FUNCTION
size_t _get_last_iter_index_face(
    int idir, int8_t ielem, size_t n, size_t g
) 
{
    size_t last_indices[3]  ;
    const int axis = ielem / 2;
    const int side = ielem % 2;

    if ( axis == 0 ) {
        last_indices[0] = side ? n + 2*g-2 : g-2 ;
        last_indices[1] = n+g-2 ; 
        last_indices[2] = n+g-2 ; 
    } else if ( axis == 1 ) {
        last_indices[1] = side ? n + 2*g-2 : g-2 ;
        last_indices[0] = n+g-2 ; 
        last_indices[2] = n+g-2 ; 
    } else {
        last_indices[2] = side ? n + 2*g-2 : g-2 ;
        last_indices[0] = n+g-2 ; 
        last_indices[1] = n+g-2 ; 
    }
    return last_indices[idir] ; 
}

KOKKOS_INLINE_FUNCTION 
void compute_phys_indices_edge(
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, 
    std::size_t const& sx, std::size_t const& sy, std::size_t const& sz, std::size_t const& g,
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
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, 
    std::size_t const& sx, std::size_t const& sy, std::size_t const& sz, std::size_t const& g, std::size_t const& g_off,
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
        j_out = y_off ? ny + g + ig : ig + g_off ;   // fixed
        k_out = z_off ? nz + g + jg : jg + g_off ;   // fixed
    }
    else if (edge < 8) {
        // Y-axis edges
        int x_off = (edge >> 0) & 1;
        int z_off = (edge >> 1) & 1;
        i_out = x_off ? nx + g + ig : ig + g_off ;   // fixed
        j_out = g + k;                      // varies
        k_out = z_off ? nz + g + jg : jg + g_off ;   // fixed
    }
    else {
        // Z-axis edges
        int x_off = (edge >> 0) & 1;
        int y_off = (edge >> 1) & 1;
        i_out = x_off ? nx + g + ig : ig + g_off ;   // fixed
        j_out = y_off ? ny + g + jg : jg + g_off ;   // fixed
        k_out = g + k;                      // varies
    }
}

KOKKOS_INLINE_FUNCTION
size_t _get_first_iter_index_edge(
    int idir, int8_t ielem, size_t n, size_t g
) 
{
    size_t first_indices[3]  ;
    
    if ( ielem < 4 ) {
        // X-axis edges
        int y_off = (ielem >> 0) & 1;
        int z_off = (ielem >> 1) & 1;
        first_indices[0] = g ; 
        first_indices[1] = y_off ? n + g : 0 ; 
        first_indices[2] = z_off ? n + g : 0 ; 
    } else if ( ielem < 8 ) {
        // Y-axis edges
        int x_off = (ielem >> 0) & 1;
        int z_off = (ielem >> 1) & 1;
        first_indices[1] = g ; 
        first_indices[0] = x_off ? n + g : 0 ; 
        first_indices[2] = z_off ? n + g : 0 ; 
    } else {
        // Z-axis edges
        int x_off = (ielem >> 0) & 1;
        int y_off = (ielem >> 1) & 1;
        first_indices[2] = g ; 
        first_indices[0] = x_off ? n + g : 0 ; 
        first_indices[1] = y_off ? n + g : 0 ; 
    }
    return first_indices[idir] ; 
}


KOKKOS_INLINE_FUNCTION
size_t _get_last_iter_index_edge(
    int idir, int8_t ielem, size_t n, size_t g
) 
{
    size_t last_indices[3]  ;
    
    if ( ielem < 4 ) {
        // X-axis edges
        int y_off = (ielem >> 0) & 1;
        int z_off = (ielem >> 1) & 1;
        last_indices[0] = n + g - 2 ; 
        last_indices[1] = y_off ? n + 2*g -2 : g-2 ;
        last_indices[2] = z_off ? n + 2*g -2 : g-2 ;
    } else if ( ielem < 8 ) {
        // Y-axis edges
        int x_off = (ielem >> 0) & 1;
        int z_off = (ielem >> 1) & 1;
        last_indices[1] = n + g - 2 ; 
        last_indices[0] = x_off ? n + 2*g -2 : g-2 ;
        last_indices[2] = z_off ? n + 2*g -2 : g-2 ;
    } else {
        // Z-axis edges
        int x_off = (ielem >> 0) & 1;
        int y_off = (ielem >> 1) & 1;
        last_indices[2] = n + g - 2 ; 
        last_indices[0] = x_off ? n + 2*g -2 : g-2 ;
        last_indices[1] = y_off ? n + 2*g -2 : g-2 ;
    }
    return last_indices[idir] ; 
}

KOKKOS_INLINE_FUNCTION 
void compute_phys_indices_corner(
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, 
    std::size_t const& sx, std::size_t const& sy, std::size_t const& sz, std::size_t const& g,
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
    std::size_t const& nx, std::size_t const& ny, std::size_t const& nz, 
    std::size_t const& sx, std::size_t const& sy, std::size_t const& sz, std::size_t const& g, std::size_t const& g_off,
    std::size_t const& i, std::size_t const& j, std::size_t const& k,
    std::size_t& i_out, std::size_t& j_out,
    std::size_t& k_out, int corner
)
{
    int x_off = (corner) & 1 ; 
    int y_off = (corner >> 1) & 1 ; 
    int z_off = (corner >> 2) & 1 ; 

    i_out = x_off ? nx + g + i :  i + g_off ; 
    j_out = y_off ? ny + g + j :  j + g_off ; 
    k_out = z_off ? nz + g + k :  k + g_off ; 
}

KOKKOS_INLINE_FUNCTION
size_t _get_first_iter_index_corner(
    int idir, int8_t ielem, size_t n, size_t g
) 
{
    int x_off = (ielem) & 1 ; 
    int y_off = (ielem >> 1) & 1 ; 
    int z_off = (ielem >> 2) & 1 ; 

    size_t first_indices[] = {
        x_off ? n + g : 0,
        y_off ? n + g : 0,
        z_off ? n + g : 0 
    } ; 
    return first_indices[idir] ; 
}


KOKKOS_INLINE_FUNCTION
size_t _get_last_iter_index_corner(
    int idir, int8_t ielem, size_t n, size_t g
) 
{
    int x_off = (ielem) & 1 ; 
    int y_off = (ielem >> 1) & 1 ; 
    int z_off = (ielem >> 2) & 1 ; 

    size_t last_indices[] = {
        x_off ? n + 2*g - 2 : g-2,
        y_off ? n + 2*g - 2 : g-2,
        z_off ? n + 2*g - 2 : g-2
    } ; 
    return last_indices[idir] ; 
}

struct index_transformer_t {
    

    std::size_t nx, ny, nz, ngz;
    std::size_t sx, sy, sz ;

    index_transformer_t(std::size_t _nx, std::size_t _ny,
                        std::size_t _nz, std::size_t _ngz, grace::var_staggering_t stag)
        : nx(_nx), ny(_ny), nz(_nz), ngz(_ngz) {
            auto s = get_index_staggerings(stag) ;
            sx = s[0] ; sy = s[1] ; sz = s[2] ; 
        }
    
    index_transformer_t(std::size_t _nx, std::size_t _ny,
                        std::size_t _nz, std::size_t _ngz)
        : nx(_nx), ny(_ny), nz(_nz), ngz(_ngz) {
            auto s = get_index_staggerings(grace::STAG_CENTER) ;
            sx = s[0] ; sy = s[1] ; sz = s[2] ; 
        }
    // Unified entry point
    template< element_kind_t elem_kind 
            , bool is_phys >
    KOKKOS_INLINE_FUNCTION
    void compute_indices(std::size_t ig, std::size_t j, std::size_t k,
                         std::size_t& i_out, std::size_t& j_out,
                         std::size_t& k_out, int ielem, bool half_ncells=false, bool off_gz=false) const
    {
        size_t _nx = half_ncells ? nx / 2 : nx ; 
        size_t _ny = half_ncells ? ny / 2 : ny ; 
        size_t _nz = half_ncells ? nz / 2 : nz ; 
        size_t _g_off = half_ncells and off_gz ? ngz / 2 : 0 ; 
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            if constexpr ( is_phys ) {
                compute_phys_indices_face(_nx,_ny,_nz,sx,sy,sz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            } else {
                compute_ghost_indices_face(_nx,_ny,_nz,sx,sy,sz,ngz,_g_off,ig,j,k,i_out,j_out,k_out,ielem);
            }
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            if constexpr ( is_phys ) {
                compute_phys_indices_edge(_nx,_ny,_nz,sx,sy,sz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            } else {
                compute_ghost_indices_edge(_nx,_ny,_nz,sx,sy,sz,ngz,_g_off,ig,j,k,i_out,j_out,k_out,ielem);
            }
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            if constexpr ( is_phys ) {
                compute_phys_indices_corner(_nx,_ny,_nz,sx,sy,sz,ngz,ig,j,k,i_out,j_out,k_out,ielem);
            } else {
                compute_ghost_indices_corner(_nx,_ny,_nz,sx,sy,sz,ngz,_g_off,ig,j,k,i_out,j_out,k_out,ielem);
            }
        }
    }

    template< element_kind_t elem_kind >
    KOKKOS_INLINE_FUNCTION
    size_t last_index(int idir, int8_t ielem, bool half_ncells=false) const {
        size_t _n = half_ncells ? nx / 2 : nx ; 
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            return _get_last_iter_index_face(idir,ielem,_n,ngz) ; 
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            return _get_last_iter_index_edge(idir,ielem,_n,ngz) ; 
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            return _get_last_iter_index_corner(idir,ielem,_n,ngz) ; 
        }
    }

    template< element_kind_t elem_kind >
    KOKKOS_INLINE_FUNCTION
    size_t first_index(int idir, int8_t ielem, bool half_ncells=false ) const {
        size_t _n = half_ncells ? nx / 2 : nx ; 
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            return _get_first_iter_index_face(idir,ielem,_n,ngz) ; 
        } else if constexpr ( elem_kind == element_kind_t::EDGE ) {
            return _get_first_iter_index_edge(idir,ielem,_n,ngz) ; 
        } else if constexpr ( elem_kind == element_kind_t::CORNER ) {
            return _get_first_iter_index_corner(idir,ielem,_n,ngz) ; 
        }
    }


} ; 

/*************************************************************************************************************/
/*************************************************************************************************************/


KOKKOS_INLINE_FUNCTION
int edge_to_face_dir(int face, int edge) {
    int edge_axis = edge / 4;    // 0=x,1=y,2=z
    int normal    = face / 2;    // 0=x,1=y,2=z

    // Collect the two tangential axes
    int t0 = (normal + 1) % 3;
    int t1 = (normal + 2) % 3;

    // Sort so j = min, k = max
    int j_axis = t0 < t1 ? t0 : t1;

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
        int& j_c, int& k_c,
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
        int& j_c, int& k_c,
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
        int& j_c, int& k_c,
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

constexpr std::array<std::array<int8_t,4>,P4EST_FACES> f2c = 
{{
    {{0,2,4,6}}, //0
    {{1,3,5,7}}, //1
    {{0,1,4,5}}, //2
    {{2,3,6,7}}, //3
    {{0,1,2,3}}, //4
    {{4,5,6,7}} //5 
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
    {{0,5,9}},  //1
    {{1,4,10}}, //2
    {{1,5,11}}, //3
    {{2,6,8}},  //4 
    {{2,7,9}},  //5
    {{3,6,10}}, //6
    {{3,7,11}}  //7
} } ;

} /*namespace detail*/

} /* namespace grace::amr */

namespace detail {
//! Unsafe function, use with care
static void get_face_prolong_dependencies(int eid, int * faces, int * edges, int * corners)
{
    static const int other_dirs[3][2] = {
        {1,2}, {0,2}, {0,1}
    } ; 
    static const int other_dirs_2[3][3] = {
        {-1,2,1}, {2,-1,0}, {1,0,-1}
    } ;

    int8_t fdir  = eid/2 ; 
    int8_t fside = eid%2 ; 

    int facts[3] = {1,2,4} ;

    int icorner = 0 ;
    for( int iside=0; iside<2; ++iside) {
        for( int jside=0; jside<2; ++jside) {
            int idir = other_dirs[fdir][0]; 
            int jdir = other_dirs[fdir][1];
            corners[icorner++] = facts[fdir] * fside + facts[idir]*iside + facts[jdir] * jside ; 
        }
    }
    int iface=0;
    for( int jside=0; jside<2; ++jside) {
        for( int idx_d=0; idx_d<2; ++idx_d) {
            int idir = other_dirs[fdir][idx_d] ; 
            faces[iface++] = 2*idir + jside ; 
        }
    }
    int iedge = 0 ; 
    for( int iside=0; iside<2; ++iside) {
        for( int jside=0; jside<2; ++jside) {
            edges[iedge++] = 4*fdir + iside + 2*jside;
        }
    }

    for( int idx_d=0; idx_d<2; ++idx_d){
        for ( int iside=0; iside<2; ++iside) {
            int idir = other_dirs[fdir][idx_d] ; 
            int ff = fdir<other_dirs_2[fdir][idir] ? 1 : 2 ; 
            int fi = fdir<other_dirs_2[fdir][idir] ? 2 : 1 ; 
            edges[iedge++] = 4 * idir + ff * fside + fi*iside ; 
        }
    }
}

static void get_edge_prolong_dependencies(int eid, int * faces, int * edges, int * corners)
{
    static const int other_dirs[3][2] = {
        {1,2}, {0,2}, {0,1}
    } ; 
    static const int other_dirs_2[3][3] = {
        {-1,2,1}, {2,-1,0}, {1,0,-1}
    } ;

    int edir = eid/4;
    int iside = (eid>>0)&1;
    int jside = (eid>>1)&1;
    int sides[2] = {iside,jside};

    int facts[3] = {1,2,4};

    int icorner=0;
    for( int kside=0; kside<2; ++kside) {
        int idir = other_dirs[edir][0];
        int jdir = other_dirs[edir][1];
        corners[icorner++] = facts[edir]*kside + facts[idir]*iside + facts[jdir]*jside ; 
    }

    int iface=0;
    for( int idx_d=0; idx_d<2; ++idx_d) {
        int idir = other_dirs[edir][idx_d];
        faces[iface++] = 2*idir + sides[idx_d] ;
    }
    for( int fside=0; fside<2; ++fside) {
        faces[iface++] = 2*edir + fside ; 
    }

    int iedge=0;
    for( int idx_d=0; idx_d<2; ++idx_d) {
        for( int kside=0; kside<2; ++kside) {
            int jdir = other_dirs[edir][idx_d]  ; 
            int kdir = other_dirs_2[edir][jdir] ;
            int sidx = kdir<jdir ? 0 : 1 ;
            int fk = edir < kdir ? 1 : 2 ; 
            int fj = edir < kdir ? 2 : 1 ; 
            edges[iedge++] = 4 * jdir + fk * kside + fj * sides[sidx] ;  
        }
    }
}

static void get_corner_prolong_dependencies(int eid, int * faces, int * edges, int * corners)
{
    static const int other_dirs[3][2] = {
        {1,2}, {0,2}, {0,1}
    } ; 

    int iside = (eid>>0)&1 ;
    int jside = (eid>>1)&1 ; 
    int kside = (eid>>2)&1 ; 
    int sides[3] = {iside,jside,kside} ; 

    int iface=0;
    for( int idir=0; idir<3; ++idir){
        faces[iface++] = 2 * idir + sides[idir] ; 
    }

    int iedge=0; 
    for( int idir=0; idir<3; ++idir) {
        edges[iedge++] = 4 * idir + sides[other_dirs[idir][0]] + 2*sides[other_dirs[idir][1]];
    }
}

}

} /* namespace grace */
#endif /* GRACE_AMR_INDEX_HELPERS_HH */