/**
 * @file variable_properties.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-12
 * 
 * @copyright This file is part of MagMA.
 * MagMA is an evolution framework that uses Discontinuous Galerkin
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

#ifndef GRACE_DATA_STRUCTURES_VARIABLE_PROPERTIES_HH
#define GRACE_DATA_STRUCTURES_VARIABLE_PROPERTIES_HH

#include <grace_config.h>
#include <Kokkos_Core.hpp> 

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/macros.hh>

namespace grace {

enum bc_t: uint8_t {BC_OUTFLOW=0, BC_LAGRANGE_EXTRAP, BC_NONE} ; 

enum var_staggering_t : uint8_t {
    STAG_CENTER=0, STAG_FACEX, STAG_FACEY, STAG_EDGEXY, STAG_FACEZ, STAG_EDGEXZ, STAG_EDGEYZ, STAG_CORNER, N_VAR_STAGGERINGS
} ; 


static inline 
std::array<int,3> get_index_staggerings(grace::var_staggering_t stag) {
    return std::array<int,3>({stag&1, (stag>>1)&1,(stag>>2)&1});
}
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 * \ingroup variables
 */
template< size_t ndim > 
struct variable_properties_t 
{ } ; 
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 *  
 * \ingroup variables
 */
template<> 
struct variable_properties_t<2>
{
    var_staggering_t staggering; 
    bool is_evolved ; 
    bool is_vector ;  
    bool is_tensor ; 
    int8_t comp_num ; 
    bc_t bc_type ; 
    size_t index ;

    std::string name ; 
} ; 
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 * \ingroup variables
 */
template<> 
struct variable_properties_t<3>
{
    var_staggering_t staggering; 
    bool is_evolved ; 
    bool is_vector ;  
    bool is_tensor ; 
    int8_t comp_num ; 
    bc_t bc_type ; 
    size_t index ;

    std::string name ; 
} ; 
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 * \ingroup variables
 */
template< size_t ndim >
struct coord_array_impl_t {} ;
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 * \ingroup variables
 */
template<> 
struct coord_array_impl_t<2> { using view_t = Kokkos::View<double ****, Kokkos::LayoutLeft, default_space>; };
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 * \ingroup variables
 */
template<> 
struct coord_array_impl_t<3> { using view_t = Kokkos::View<double *****, Kokkos::LayoutLeft, default_space>; } ;
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 * \ingroup variables
 */
template< size_t ndim >
struct cell_vol_array_impl_t {} ;
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 * \ingroup variables
 */
template<> 
struct cell_vol_array_impl_t<2> { using view_t = Kokkos::View<double ***, Kokkos::LayoutLeft, default_space>; };
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 * \ingroup variables
 */
template<> 
struct cell_vol_array_impl_t<3> { using view_t = Kokkos::View<double ****, Kokkos::LayoutLeft, default_space>; } ;
//*****************************************************************************************************
/**
 * @brief Helper class 
 * \cond grace_detail
 * \ingroup variables
 */
template< size_t ndim >
struct scalar_array_impl_t { using view_t = Kokkos::View<double **, Kokkos::LayoutLeft, default_space>; } ;
//*****************************************************************************************************
/**
 * @brief Proxy for variable <code>View</code> type in GRACE
 * \ingroup variables
 * @tparam ndim Number of spatial dimension
 */
using var_array_t = Kokkos::View<double EXPR(*,*,*)**, Kokkos::LayoutLeft, default_space> ;  ; 
//***************************************************************************************************** 
/**
 * @brief Proxy for coordinate <code>View</code> type in GRACE
 * \ingroup variables
 * @tparam ndim Number of spatial dimension
 */
template< size_t ndim = GRACE_NSPACEDIM > 
using coord_array_t = coord_array_impl_t<ndim>::view_t ; 
//*****************************************************************************************************
/**
 * @brief Proxy for scalar <code>View</code> type in GRACE
 * \ingroup variables
 * @tparam ndim Number of spatial dimension
 */
template< size_t ndim = GRACE_NSPACEDIM > 
using scalar_array_t = scalar_array_impl_t<ndim>::view_t ; 
//*****************************************************************************************************
/**
 * @brief Proxy for volume cell <code>View</code> type in GRACE
 * \ingroup variables
 * @tparam ndim Number of spatial dimension
 */
template< size_t ndim = GRACE_NSPACEDIM > 
using cell_vol_array_t = cell_vol_array_impl_t<ndim>::view_t ; 
//*****************************************************************************************************
/**
 * @brief Proxy for flux <code>View</code> type in GRACE
 * \ingroup variables
 */
using flux_array_t = Kokkos::View<double EXPR(*,*,*) ***, Kokkos::LayoutLeft, default_space> ; 
/**
 * @brief Proxy for jacobian matrix <code>View</code> type in GRACE
 * \ingroup variables
 */
using jacobian_array_t = Kokkos::View<double EXPR(*,*,*) ***, Kokkos::LayoutLeft, default_space> ; 
/*****************************************************************************************************/
/*****************************************************************************************************/
/*                               STAGGERED FIELDS UTILS                                              */
/*****************************************************************************************************/
/*****************************************************************************************************/
struct staggered_coordinate_arrays_t {
    cell_vol_array_t<GRACE_NSPACEDIM> cell_face_surfaces_x, cell_face_surfaces_y, cell_face_surfaces_z ;
    #ifdef GRACE_3D  
    cell_vol_array_t<GRACE_NSPACEDIM> cell_edge_lengths_xy, cell_edge_lengths_xz, cell_edge_lengths_yz ;
    #endif 
    staggered_coordinate_arrays_t()
        : VEC( cell_face_surfaces_x("Cell face surfaces X", VEC(0,0,0),0)
             , cell_face_surfaces_y("Cell face surfaces Y", VEC(0,0,0),0)
             , cell_face_surfaces_z("Cell face surfaces Z", VEC(0,0,0),0) ) 
        #ifdef GRACE_3D 
        , VEC( cell_edge_lengths_xy("Cell edge lengths XY", VEC(0,0,0),0) 
             , cell_edge_lengths_xz("Cell edge lengths XZ", VEC(0,0,0),0) 
             , cell_edge_lengths_yz("Cell edge lengths YZ", VEC(0,0,0),0) )
        #endif 
    {} 

    staggered_coordinate_arrays_t(VEC(int nx, int ny, int nz), int ngz, int nq)
        : VEC( cell_face_surfaces_x("Cell face surfaces X", VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nq)
             , cell_face_surfaces_y("Cell face surfaces Y", VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nq)
             , cell_face_surfaces_z("Cell face surfaces Z", VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nq) )
        #ifdef GRACE_3D  
        , VEC( cell_edge_lengths_xy("Cell edge lengths XY", VEC(nx+2*ngz+1,ny+2*ngz+1,nz+2*ngz),nq) 
             , cell_edge_lengths_xz("Cell edge lengths XZ", VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz+1),nq) 
             , cell_edge_lengths_yz("Cell edge lengths YZ", VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz+1),nq) )
        #endif 
    {}

    void realloc(VEC(int nx, int ny, int nz), int ngz, int nq) {
        EXPR(
        Kokkos::realloc(cell_face_surfaces_x, VEC(nx + 2*ngz + 1, ny + 2*ngz, nz+2*ngz), nq) ;,
        Kokkos::realloc(cell_face_surfaces_y, VEC(nx + 2*ngz, ny + 2*ngz + 1, nz+2*ngz), nq) ;,
        Kokkos::realloc(cell_face_surfaces_z , VEC(nx + 2*ngz, ny + 2*ngz, nz+2*ngz + 1), nq) ;
        ) 
        #ifdef GRACE_3D 
        Kokkos::realloc(cell_edge_lengths_xy, VEC(nx + 2*ngz + 1, ny + 2*ngz + 1, nz+2*ngz), nq ) ;
        Kokkos::realloc(cell_edge_lengths_xz, VEC(nx + 2*ngz + 1, ny + 2*ngz, nz+2*ngz + 1), nq ) ;
        Kokkos::realloc(cell_edge_lengths_yz , VEC(nx + 2*ngz, ny + 2*ngz+1, nz+2*ngz + 1), nq ) ;
        #endif 
    }
} ;   
struct staggered_variable_arrays_t {
    var_array_t face_staggered_fields_x, face_staggered_fields_y, face_staggered_fields_z ; 
    #ifdef GRACE_3D 
    var_array_t edge_staggered_fields_xy, edge_staggered_fields_xz, edge_staggered_fields_yz ; 
    #endif 
    var_array_t corner_staggered_fields ;

    staggered_variable_arrays_t() 
      : VEC( face_staggered_fields_x("x-face staggered variables", VEC(0,0,0),0,0)
           , face_staggered_fields_y("y-face staggered variables", VEC(0,0,0),0,0)
           , face_staggered_fields_z("z-face staggered variables", VEC(0,0,0),0,0) )
    #ifdef GRACE_3D 
      , VEC( edge_staggered_fields_xy("xy-face staggered variables", VEC(0,0,0),0,0)
           , edge_staggered_fields_xz("xz-face staggered variables", VEC(0,0,0),0,0)
           , edge_staggered_fields_yz("yz-face staggered variables", VEC(0,0,0),0,0) )
    #endif 
      , corner_staggered_fields("corner staggered variables", VEC(0,0,0),0,0) 
    {} 

    void 
    realloc(VEC(int nx, int ny, int nz), int ngz, int nq, int nvars_face, int nvars_edge, int nvars_corner) {
        EXPR(
        Kokkos::realloc(face_staggered_fields_x, VEC(nx + 2*ngz + 1, ny + 2*ngz, nz+2*ngz), nvars_face, nq) ;
        ,
        Kokkos::realloc(face_staggered_fields_y, VEC(nx + 2*ngz, ny + 2*ngz + 1, nz+2*ngz), nvars_face, nq) ;
        ,
        Kokkos::realloc(face_staggered_fields_z, VEC(nx + 2*ngz, ny + 2*ngz, nz+2*ngz + 1), nvars_face, nq) ;
        )
        #ifdef GRACE_3D 
        Kokkos::realloc(edge_staggered_fields_xy, VEC(nx + 2*ngz + 1, ny + 2*ngz + 1, nz+2*ngz), nvars_edge, nq) ;
        Kokkos::realloc(edge_staggered_fields_xz, VEC(nx + 2*ngz + 1, ny + 2*ngz, nz+2*ngz + 1), nvars_edge, nq) ;
        Kokkos::realloc(edge_staggered_fields_yz, VEC(nx + 2*ngz, ny + 2*ngz + 1, nz+2*ngz + 1), nvars_edge, nq) ;
        #endif 
        Kokkos::realloc(corner_staggered_fields, VEC(nx + 2*ngz+1, ny + 2*ngz+1, nz+2*ngz+1), nvars_corner, nq) ;
    }
} ; 

static inline void deep_copy(
    staggered_variable_arrays_t& dest, 
    staggered_variable_arrays_t& src 
) {
    Kokkos::deep_copy(dest.face_staggered_fields_x, src.face_staggered_fields_x) ; 
    Kokkos::deep_copy(dest.face_staggered_fields_y, src.face_staggered_fields_y) ; 
    Kokkos::deep_copy(dest.face_staggered_fields_z, src.face_staggered_fields_z) ;

    Kokkos::deep_copy(dest.edge_staggered_fields_xy, src.edge_staggered_fields_xy) ; 
    Kokkos::deep_copy(dest.edge_staggered_fields_xz, src.edge_staggered_fields_xz) ; 
    Kokkos::deep_copy(dest.edge_staggered_fields_yz, src.edge_staggered_fields_yz) ;

    Kokkos::deep_copy(dest.corner_staggered_fields, src.corner_staggered_fields) ;
}
/*****************************************************************************************************/
/*****************************************************************************************************/
} /* namespace grace */

#endif /* GRACE_DATA_STRUCTURES_VARIABLE_PROPERTIES_HH */