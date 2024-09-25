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
/*****************************************************************************************************/
/*****************************************************************************************************/
#ifdef GRACE_3D
enum var_staggering_t {
    CELL_CENTER=0,
    FACE=1,
    EDGE=2,
    CORNER=3
} ; 
#else 
enum var_staggering_t {
    CELL_CENTER=0,
    FACE=1,
    CORNER=2
} ; 
#endif 
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
    using view_t = Kokkos::View<double ****, Kokkos::LayoutLeft, default_space> ; 
    var_staggering_t staggering; 
    bool is_evolved ; 
    bool is_vector ;  
    bool is_tensor ; 
    int8_t comp_num ; 
    std::string bc_type ; 
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
    using view_t = Kokkos::View<double *****, Kokkos::LayoutLeft, default_space> ; 
    var_staggering_t staggering; 
    bool is_evolved ; 
    bool is_vector;
    bool is_tensor; 
    int8_t comp_num ;
    std::string bc_type ; 
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
template< size_t ndim = GRACE_NSPACEDIM > 
using var_array_t = variable_properties_t<ndim>::view_t ; 
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
    var_array_t<GRACE_NSPACEDIM> face_staggered_fields_x, face_staggered_fields_y, face_staggered_fields_z ; 
    #ifdef GRACE_3D 
    var_array_t<GRACE_NSPACEDIM> edge_staggered_fields_xy, edge_staggered_fields_xz, edge_staggered_fields_yz ; 
    #endif 
    var_array_t<GRACE_NSPACEDIM> corner_staggered_fields ;

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

    void 
    deep_copy(staggered_variable_arrays_t const& other)
    {
        Kokkos::deep_copy(face_staggered_fields_x,other.face_staggered_fields_x) ; 
        Kokkos::deep_copy(face_staggered_fields_y,other.face_staggered_fields_y) ; 
        Kokkos::deep_copy(face_staggered_fields_z,other.face_staggered_fields_z) ; 
        #ifdef GRACE_3D
        Kokkos::deep_copy(edge_staggered_fields_xy,other.edge_staggered_fields_xy) ; 
        Kokkos::deep_copy(edge_staggered_fields_xz,other.edge_staggered_fields_xz) ; 
        Kokkos::deep_copy(edge_staggered_fields_yz,other.edge_staggered_fields_yz) ; 
        #endif 
        Kokkos::deep_copy(corner_staggered_fields,other.corner_staggered_fields) ; 
    }

    template< typename ... IndexT > 
    deep_copy_subview(staggered_variable_arrays_t const& other, IndexT const& ... indices) ; 
    {
        using execspace = typename face_staggered_fields_x::ExecutionSpace ;
        auto sview_fx       = Kokkos::subview(face_staggered_fields_x, indices...) ; 
        auto sview_fx_other = Kokkos::subview(other.face_staggered_fields_x, indices...) ; 
        Kokkos::deep_copy(execspace{}, sview_fx,sview_fx_other) ; 
        auto sview_fy       = Kokkos::subview(face_staggered_fields_y, indices...) ; 
        auto sview_fy_other = Kokkos::subview(other.face_staggered_fields_y, indices...) ; 
        Kokkos::deep_copy(execspace{}, sview_fy,sview_fy_other) ;  
        auto sview_fz       = Kokkos::subview(face_staggered_fields_z, indices...) ; 
        auto sview_fz_other = Kokkos::subview(other.face_staggered_fields_z, indices...) ; 
        Kokkos::deep_copy(execspace{}, sview_fz,sview_fz_other) ;  
        #ifdef GRACE_3D
        auto sview_exy       = Kokkos::subview(edge_staggered_fields_xy, indices...) ; 
        auto sview_exy_other = Kokkos::subview(other.edge_staggered_fields_xy, indices...) ; 
        Kokkos::deep_copy(execspace{}, sview_exy,sview_exy_other) ; 
        auto sview_exz       = Kokkos::subview(edge_staggered_fields_xz, indices...) ; 
        auto sview_exz_other = Kokkos::subview(other.edge_staggered_fields_xz, indices...) ; 
        Kokkos::deep_copy(execspace{}, sview_exz,sview_exz_other) ; 
        auto sview_eyz       = Kokkos::subview(edge_staggered_fields_yz, indices...) ; 
        auto sview_eyz_other = Kokkos::subview(other.edge_staggered_fields_yz, indices...) ; 
        Kokkos::deep_copy(execspace{}, sview_eyz,sview_eyz_other) ;
        #endif 
        auto sview_c       = Kokkos::subview(corner_staggered_fields, indices...) ; 
        auto sview_c_other = Kokkos::subview(other.corner_staggered_fields, indices...) ; 
        Kokkos::deep_copy(execspace{},sview_c,sview_c_other) ;
        Kokkos::fence() ; 
    }

    template< typename ... IndexT >
    decltype(auto) subview(IndexT ... indices)
    {
        return staggered_variable_arrays_subview_t{
            Kokkos::subview(face_staggered_fields_x, indices...) , 
            Kokkos::subview(face_staggered_fields_y, indices...) , 
            Kokkos::subview(face_staggered_fields_z, indices...) , 
            #ifdef GRACE_3D 
            Kokkos::subview(edge_staggered_fields_xy, indices...) , 
            Kokkos::subview(edge_staggered_fields_xz, indices...) , 
            Kokkos::subview(edge_staggered_fields_yz, indices...) , 
            #endif 
            Kokkos::subview(corner_staggered_fields, indices...) 
        } ; 
    }
   
} ; 

template < typename SViewT >
struct staggered_variable_arrays_subview_t {
    SViewT face_staggered_fields_x, face_staggered_fields_y, face_staggered_fields_z ; 
    #ifdef GRACE_3D 
    SViewT edge_staggered_fields_xy, edge_staggered_fields_xz, edge_staggered_fields_yz ; 
    #endif 
    SViewT corner_staggered_fields ;

    staggered_variable_arrays_subview_t(
        SViewT _fx, SViewT _fy, SViewT _fz,
        #ifdef GRACE_3D
        SViewT _exy, SViewT _exz, SViewT _eyz,
        #endif 
        SViewT _c
    )
    : face_staggered_fields_x(_fx), face_staggered_fields_y(_fy), face_staggered_fields_z(_fz),
    #ifdef GRACE_3D 
      edge_staggered_fields_xy(_exy), edge_staggered_fields_xz(_exz), edge_staggered_fields_yz(_eyz),
    #endif
      corner_staggered_fields(_c)
    {} ;

    template < typename SViewT_B >
    void deep_copy(staggered_variable_arrays_subview_t<SViewT_B> const& other )
    {
        Kokkos::deep_copy(face_staggered_fields_x,other.face_staggered_fields_x) ; 
        Kokkos::deep_copy(face_staggered_fields_y,other.face_staggered_fields_y) ; 
        Kokkos::deep_copy(face_staggered_fields_z,other.face_staggered_fields_z) ; 
        #ifdef GRACE_3D
        Kokkos::deep_copy(edge_staggered_fields_xy,other.edge_staggered_fields_xy) ; 
        Kokkos::deep_copy(edge_staggered_fields_xz,other.edge_staggered_fields_xz) ; 
        Kokkos::deep_copy(edge_staggered_fields_yz,other.edge_staggered_fields_yz) ; 
        #endif 
        Kokkos::deep_copy(corner_staggered_fields,other.corner_staggered_fields) ;
    }

    template < typename SViewT_B >
    void deep_copy_async(staggered_variable_arrays_subview_t<SViewT_B> const& other )
    {
        using execspace = typename face_staggered_fields_x::ExecutionSpace ;
        Kokkos::deep_copy(execspace{}, face_staggered_fields_x,other.face_staggered_fields_x) ; 
        Kokkos::deep_copy(execspace{}, face_staggered_fields_y,other.face_staggered_fields_y) ; 
        Kokkos::deep_copy(execspace{}, face_staggered_fields_z,other.face_staggered_fields_z) ; 
        #ifdef GRACE_3D
        Kokkos::deep_copy(execspace{}, edge_staggered_fields_xy,other.edge_staggered_fields_xy) ; 
        Kokkos::deep_copy(execspace{}, edge_staggered_fields_xz,other.edge_staggered_fields_xz) ; 
        Kokkos::deep_copy(execspace{}, edge_staggered_fields_yz,other.edge_staggered_fields_yz) ; 
        #endif 
        Kokkos::deep_copy(execspace{}, corner_staggered_fields,other.corner_staggered_fields) ;
    }

} ; 

/*****************************************************************************************************/
/*****************************************************************************************************/
} /* namespace grace */

#endif /* GRACE_DATA_STRUCTURES_VARIABLE_PROPERTIES_HH */