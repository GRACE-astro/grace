/**
 * @file bc_helpers.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Index fiesta.
 * @date 2024-03-21
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

#include <grace/amr/bc_copy_ghostzones.hh>
#include <grace/amr/prolongation_kernels.tpp> 
#include <grace/amr/restriction_kernels.tpp> 
#include <grace/utils/prolongation.hh>
#include <grace/utils/limiters.hh> 
#include <grace/utils/restriction.hh>
#include <grace/amr/boundary_conditions.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/bc_helpers.tpp>
#include <grace/amr/grace_amr.hh> 
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/interpolators.hh>
#include <grace/data_structures/grace_data_structures.hh>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

namespace grace { namespace amr { 

void copy_interior_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& vars
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::staggered_variable_arrays_t& staggered_state
    , grace::staggered_variable_arrays_t& staggered_halo
    , Kokkos::vector<simple_face_info_t>& interior_faces
    , Kokkos::vector<simple_corner_info_t>& interior_corners
    #ifdef GRACE_3D
    , Kokkos::vector<simple_edge_info_t>& interior_edges
    #endif 
)
{
    /******************************************************/
    /*                CELL CENTERS                        */
    /******************************************************/
    copy_interior_ghostzones_cell_centers(
          vars
        , halo 
        , interior_faces 
        , interior_corners 
        #ifdef GRACE_3D 
        , interior_edges 
        #endif 
    ) ; 
    /******************************************************/
    /*                CELL CORNERS                        */
    /******************************************************/
    copy_interior_ghostzones_corners(
          staggered_state.corner_staggered_fields
        , staggered_halo.corner_staggered_fields
        , interior_faces 
        , interior_corners 
        #ifdef GRACE_3D 
        , interior_edges 
        #endif 
    ) ;
}

void copy_interior_ghostzones_cell_centers(
      grace::var_array_t<GRACE_NSPACEDIM>& vars
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , Kokkos::vector<simple_face_info_t>& interior_faces
    , Kokkos::vector<simple_corner_info_t>& interior_corners
    #ifdef GRACE_3D
    , Kokkos::vector<simple_edge_info_t>& interior_edges
    #endif 
)
{
    using namespace grace; 
    using namespace Kokkos ; 

    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int64_t ngz = amr::get_n_ghosts() ;
    int64_t nq  = amr::get_local_num_quadrants() ;
    int nvars  = variables::get_n_evolved()      ;
    size_t const n_faces = interior_faces.size()   ;
    size_t const n_corners = interior_corners.size()   ;
    #ifdef GRACE_3D
    size_t const n_edges = interior_edges.size()   ;
    #endif 
    auto& d_face_info = interior_faces.d_view    ; 
    auto& d_corner_info = interior_corners.d_view    ; 
    #ifdef GRACE_3D
    auto& d_edge_info = interior_edges.d_view    ; 
    #endif
    if( EXPR( n_faces == 0, and n_corners == 0, and n_edges==0) ) {
        return ; 
    }
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy(
            {0,VECD(0,0), 0,0},
            {ngz, VECD(static_cast<long>(nx),static_cast<long>(ny)), static_cast<long>(nvars), static_cast<long>(n_faces)}
        ) ;
    parallel_for(GRACE_EXECUTION_TAG("AMR", "copy_interior_ghostzones_across_faces")
                , policy 
                , KOKKOS_LAMBDA(const size_t& ig, VECD(const size_t& j, const size_t& k), const size_t& ivar, const size_t& iface)
        {
            /* Get information about quadrants sharing the face */
            int polarity     =  d_face_info(iface).has_polarity_flip ;
            int is_ghost     =  d_face_info(iface).is_ghost          ; 
            int which_face_a =  d_face_info(iface).which_face_a      ; 
            int which_face_b =  d_face_info(iface).which_face_b      ; 
            int tid_a        =  d_face_info(iface).which_tree_a      ;
            int tid_b        =  d_face_info(iface).which_tree_b      ;
            int64_t qid_a    =  d_face_info(iface).qid_a             ;
            int64_t qid_b    =  d_face_info(iface).qid_b             ; 

            /* Get correct array to read from / write to */
            auto& view_a = vars ; 
            auto& view_b = (is_ghost) ? halo : vars ; 
            
            auto const index_mapping = [&] (
                VEC(size_t const ig, size_t const j, size_t const k),
                int const fa, int const fb, size_t ijk[GRACE_NSPACEDIM], size_t lmn[GRACE_NSPACEDIM]) 
            {
                static constexpr int FIRST_ALONG_FACE = 0 ;
                static constexpr int SECOND_ALONG_FACE = 1 ;
                static constexpr int POSITIVE_FACE = 2 ;
                static constexpr int NEGATIVE_FACE = 3 ;
                static int face_direction[3][6] = {
                    {NEGATIVE_FACE, POSITIVE_FACE, FIRST_ALONG_FACE, FIRST_ALONG_FACE, FIRST_ALONG_FACE, FIRST_ALONG_FACE},
                    {FIRST_ALONG_FACE, FIRST_ALONG_FACE, NEGATIVE_FACE, POSITIVE_FACE, SECOND_ALONG_FACE, SECOND_ALONG_FACE},
                    {SECOND_ALONG_FACE,SECOND_ALONG_FACE,SECOND_ALONG_FACE,SECOND_ALONG_FACE,NEGATIVE_FACE,POSITIVE_FACE}
                }; 
                int xa = face_direction[0][fa] ; 
                int ya = face_direction[1][fa] ;
                int za = face_direction[2][fa] ; 
                EXPR(
                ijk[0] = (xa==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (xa==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((xa==NEGATIVE_FACE) ? ig : (nx+ngz+ig)) ) ;,
                ijk[1] = (ya==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (ya==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((ya==NEGATIVE_FACE) ? ig : (ny+ngz+ig)) ) ;,
                ijk[2] = (za==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (za==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((za==NEGATIVE_FACE) ? ig : (nz+ngz+ig)) ) ;
                )
                int xb = face_direction[0][fb] ; 
                int yb = face_direction[1][fb] ;
                int zb = face_direction[2][fb] ; 
                EXPR(
                lmn[0] = (xb==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (xb==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((xb==NEGATIVE_FACE) ? (ngz+ig) : (nx+ig)) ) ;, 
                lmn[1] = (yb==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (yb==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((yb==NEGATIVE_FACE) ? (ngz+ig) : (ny+ig)) ) ;,
                lmn[2] = (zb==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (zb==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((zb==NEGATIVE_FACE) ? (ngz+ig) : (nz+ig)) ) ;
                )
            } ; 
            size_t ijk_a[GRACE_NSPACEDIM], ijk_b[GRACE_NSPACEDIM] ; 
            index_mapping(VEC(ig,j,k), which_face_a, which_face_b, ijk_a, ijk_b) ; 
            view_a(VEC(ijk_a[0],ijk_a[1],ijk_a[2]),ivar,qid_a) =  view_b(VEC(ijk_b[0],ijk_b[1],ijk_b[2]),ivar,qid_b) ; 
            if( ! is_ghost ) {
                index_mapping(VEC(ig,j,k), which_face_b, which_face_a, ijk_b, ijk_a) ;
                view_b(VEC(ijk_b[0],ijk_b[1],ijk_b[2]),ivar,qid_b) =  view_a(VEC(ijk_a[0],ijk_a[1],ijk_a[2]),ivar,qid_a) ;
            }
        });
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy_corner(
            {0,VECD(0,0), 0,0},
            {ngz, VECD(ngz, ngz), static_cast<long>(nvars), static_cast<long>(n_corners)}
        ) ;
    parallel_for(GRACE_EXECUTION_TAG("AMR", "copy_interior_ghostzones_across_corners")
                , policy_corner 
                , KOKKOS_LAMBDA(const size_t& ig, VECD(const size_t& jg, const size_t& kg), const size_t& ivar, const size_t& icorner)
        {
            /* Get information about quadrants sharing the corner */
            int is_ghost     =  d_corner_info(icorner).is_ghost          ; 
            int which_corner_a =  d_corner_info(icorner).which_corner_a      ; 
            int which_corner_b =  d_corner_info(icorner).which_corner_b      ; 
            int tid_a        =  d_corner_info(icorner).which_tree_a      ;
            int tid_b        =  d_corner_info(icorner).which_tree_b      ;
            int64_t qid_a    =  d_corner_info(icorner).qid_a             ;
            int64_t qid_b    =  d_corner_info(icorner).qid_b             ; 
            /* Helper lambda that maps corner indices in and out of ghostzones. */
            auto const index_mapping = [&] ( int const ii, 
                                             int const jj, 
                                             int const kk, 
                                             int const ca, 
                                             int const cb, 
                                             int ijk[GRACE_NSPACEDIM], 
                                             int lmn[GRACE_NSPACEDIM] ) 
            {
                int x = (ca >> 0) & 1;  
                int y = (ca >> 1) & 1;  
                int z = (ca >> 2) & 1;  
                EXPR(
                ijk[0] = (x == 0) ? ii : (nx + ngz + ii);,
                ijk[1] = (y == 0) ? jj : (ny + ngz + jj);,
                ijk[2] = (z == 0) ? kk : (nz + ngz + kk);)
                x = (cb >> 0) & 1;  
                y = (cb >> 1) & 1;  
                z = (cb >> 2) & 1;
                EXPR(
                lmn[0] = (x == 0) ? (ngz + ii) : (nx + ii);,
                lmn[1] = (y == 0) ? (ngz + jj) : (ny + jj);,
                lmn[2] = (z == 0) ? (ngz + kk) : (nz + kk);)
            } ; 
            /* Get correct array to read from / write to */
            auto& view_a = vars ; 
            auto& view_b = (is_ghost) ? halo : vars ; 
            int ijk_a [GRACE_NSPACEDIM], ijk_b [GRACE_NSPACEDIM] ; 
            index_mapping(ig,jg,kg,which_corner_a,which_corner_b, ijk_a,ijk_b) ; 
            view_a(VEC(ijk_a[0],ijk_a[1],ijk_a[2]),ivar,qid_a) =  view_b(VEC(ijk_b[0],ijk_b[1],ijk_b[2]),ivar,qid_b) ;    
            if( ! is_ghost ) {
                index_mapping(ig,jg,kg,which_corner_b,which_corner_a, ijk_b,ijk_a) ;
                view_b(VEC(ijk_b[0],ijk_b[1],ijk_b[2]),ivar,qid_b) =  view_a(VEC(ijk_a[0],ijk_a[1],ijk_a[2]),ivar,qid_a) ;
            }
        });
    #ifdef GRACE_3D
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy_edge(
            {0,0,0, 0,0},
            {ngz, ngz, static_cast<long>(nx), static_cast<long>(nvars), static_cast<long>(n_edges)}
        ) ;
    parallel_for(GRACE_EXECUTION_TAG("AMR", "copy_interior_ghostzones_across_edges")
                , policy_edge 
                , KOKKOS_LAMBDA(const size_t& ig, const size_t& jg, const size_t& k, const size_t& ivar, const size_t& iedge)
        {
            /* Get information about quadrants sharing the edge */
            int is_ghost     =  d_edge_info(iedge).is_ghost          ; 
            int which_edge_a =  d_edge_info(iedge).which_edge_a      ; 
            int which_edge_b =  d_edge_info(iedge).which_edge_b      ; 
            int tid_a        =  d_edge_info(iedge).which_tree_a      ;
            int tid_b        =  d_edge_info(iedge).which_tree_b      ;
            int64_t qid_a    =  d_edge_info(iedge).qid_a             ;
            int64_t qid_b    =  d_edge_info(iedge).qid_b             ; 
            /* Helper lambda that maps edge indices in and out of ghostzones. */
            auto const index_mapping = [&] ( int const iig, 
                                             int const jjg, 
                                             int const kk, 
                                             int const ea, 
                                             int const eb, 
                                             int ijk[GRACE_NSPACEDIM], 
                                             int lmn[GRACE_NSPACEDIM] ) 
            {
                static constexpr const int ALONG_EDGE = -1 ;
                static constexpr const int NEGATIVE_EDGE = 0 ;
                static constexpr const int POSITIVE_EDGE = 1 ;
                static const int edge_directions[3][12] = {
                    {ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, 
                    NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE,
                     NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE},  // x directions
                    {NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, 
                    ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE,
                    NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE}, // y directions
                    {NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE, 
                    NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE, 
                    ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE} // z directions
                };   
                
                // Extract directional values for edges ea and eb
                int x_ea = edge_directions[0][ea];
                int y_ea = edge_directions[1][ea];
                int z_ea = edge_directions[2][ea];

                // Map indices for ijk based on edge ea
                ijk[0] = (x_ea == ALONG_EDGE) ? (kk+ngz) : (x_ea == NEGATIVE_EDGE ? iig : (nx + ngz + iig));
                ijk[1] = (y_ea == ALONG_EDGE) 
                            ? (kk+ngz) 
                            : (x_ea == ALONG_EDGE 
                                ? (y_ea == NEGATIVE_EDGE ? iig : (ny + ngz + iig)) 
                                : (y_ea == NEGATIVE_EDGE ? jjg : (ny + ngz + jjg)));
                ijk[2] = (z_ea == ALONG_EDGE) ? (kk+ngz) : (z_ea == NEGATIVE_EDGE ? jjg : (nz + ngz + jjg));

                int x_eb = edge_directions[0][eb];
                int y_eb = edge_directions[1][eb];
                int z_eb = edge_directions[2][eb];

                // Map indices for lmn based on edge eb
                lmn[0] = (x_eb == ALONG_EDGE) ? (kk+ngz) : (x_eb == NEGATIVE_EDGE ? (ngz + iig) : (nx + iig));
                lmn[1] = (y_eb == ALONG_EDGE) 
                            ? (kk+ngz)
                            : (x_eb == ALONG_EDGE 
                                ? (y_eb == NEGATIVE_EDGE ? (ngz + iig) : (ny + iig)) 
                                : (y_eb == NEGATIVE_EDGE ? (ngz + jjg) : (ny + jjg)));
                lmn[2] = (z_eb == ALONG_EDGE) ? (kk+ngz) : (z_eb == NEGATIVE_EDGE ? (ngz + jjg) : (nz + jjg));
            } ; 
            /* Get correct array to read from / write to */
            auto& view_a = vars ; 
            auto& view_b = (is_ghost) ? halo : vars ; 
            int ijk_a [GRACE_NSPACEDIM], ijk_b [GRACE_NSPACEDIM] ; 
            index_mapping(ig,jg,k,which_edge_a,which_edge_b, ijk_a,ijk_b) ; 
            view_a(VEC(ijk_a[0],ijk_a[1],ijk_a[2]),ivar,qid_a) =  view_b(VEC(ijk_b[0],ijk_b[1],ijk_b[2]),ivar,qid_b) ; 
            if( ! is_ghost ) {
                index_mapping(ig,jg,k,which_edge_b,which_edge_a, ijk_b,ijk_a) ;
                view_b(VEC(ijk_b[0],ijk_b[1],ijk_b[2]),ivar,qid_b) =  view_a(VEC(ijk_a[0],ijk_a[1],ijk_a[2]),ivar,qid_a) ;
            }
        }) ;
    #endif 
}

void copy_interior_ghostzones_corners(
      grace::var_array_t<GRACE_NSPACEDIM>& vars
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , Kokkos::vector<simple_face_info_t>& interior_faces
    , Kokkos::vector<simple_corner_info_t>& interior_corners
    #ifdef GRACE_3D
    , Kokkos::vector<simple_edge_info_t>& interior_edges
    #endif 
)
{
    using namespace grace; 
    using namespace Kokkos ; 

    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int64_t ngz = amr::get_n_ghosts() ;
    int64_t nq  = amr::get_local_num_quadrants() ;
    int nvars  = variables::get_n_evolved_corner_staggered()      ; 
    size_t const n_faces = interior_faces.size()   ;
    size_t const n_corners = interior_corners.size()   ;
    #ifdef GRACE_3D
    size_t const n_edges = interior_edges.size()   ;
    #endif 
    auto& d_face_info = interior_faces.d_view    ; 
    auto& d_corner_info = interior_corners.d_view    ; 
    #ifdef GRACE_3D
    auto& d_edge_info = interior_edges.d_view    ; 
    #endif
    if( (EXPR( n_faces == 0, and n_corners == 0, and n_edges==0)) or nvars==0 ) {
        return ; 
    }
    #if 1
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy(
            {0,VECD(0,0), 0,0},
            {ngz, VECD(static_cast<long>(nx),static_cast<long>(ny)), static_cast<long>(nvars), static_cast<long>(n_faces)}
        ) ;
    parallel_for(GRACE_EXECUTION_TAG("AMR", "copy_interior_ghostzones_across_faces_corner_staggering")
                , policy 
                , KOKKOS_LAMBDA(const size_t& ig, VECD(const size_t& j, const size_t& k), const size_t& ivar, const size_t& iface)
        {
            /* Get information about quadrants sharing the face */
            int polarity     =  d_face_info(iface).has_polarity_flip ;
            int is_ghost     =  d_face_info(iface).is_ghost          ; 
            int which_face_a =  d_face_info(iface).which_face_a      ; 
            int which_face_b =  d_face_info(iface).which_face_b      ; 
            int tid_a        =  d_face_info(iface).which_tree_a      ;
            int tid_b        =  d_face_info(iface).which_tree_b      ;
            int64_t qid_a    =  d_face_info(iface).qid_a             ;
            int64_t qid_b    =  d_face_info(iface).qid_b             ; 

            /* Get correct array to read from / write to */
            auto& view_a = vars ; 
            auto& view_b = (is_ghost) ? halo : vars ; 
            
            auto const index_mapping = [&] (
                VEC(size_t const ig, size_t const j, size_t const k),
                int const fa, int const fb, size_t ijk[GRACE_NSPACEDIM], size_t lmn[GRACE_NSPACEDIM]) 
            {
                static constexpr int FIRST_ALONG_FACE = 0 ;
                static constexpr int SECOND_ALONG_FACE = 1 ;
                static constexpr int POSITIVE_FACE = 2 ;
                static constexpr int NEGATIVE_FACE = 3 ;
                static int face_direction[3][6] = {
                    {NEGATIVE_FACE, POSITIVE_FACE, FIRST_ALONG_FACE, FIRST_ALONG_FACE, FIRST_ALONG_FACE, FIRST_ALONG_FACE},
                    {FIRST_ALONG_FACE, FIRST_ALONG_FACE, NEGATIVE_FACE, POSITIVE_FACE, SECOND_ALONG_FACE, SECOND_ALONG_FACE},
                    {SECOND_ALONG_FACE,SECOND_ALONG_FACE,SECOND_ALONG_FACE,SECOND_ALONG_FACE,NEGATIVE_FACE,POSITIVE_FACE}
                }; 
                int xa = face_direction[0][fa] ; 
                int ya = face_direction[1][fa] ;
                int za = face_direction[2][fa] ; 
                EXPR(
                ijk[0] = (xa==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (xa==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((xa==NEGATIVE_FACE) ? ig : (nx+ngz+ig)) ) ;,
                ijk[1] = (ya==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (ya==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((ya==NEGATIVE_FACE) ? ig : (ny+ngz+ig)) ) ;,
                ijk[2] = (za==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (za==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((za==NEGATIVE_FACE) ? ig : (nz+ngz+ig)) ) ;
                )
                int xb = face_direction[0][fb] ; 
                int yb = face_direction[1][fb] ;
                int zb = face_direction[2][fb] ; 
                EXPR(
                lmn[0] = (xb==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (xb==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((xb==NEGATIVE_FACE) ? (ngz+ig) : (nx+ig)) ) ;, 
                lmn[1] = (yb==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (yb==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((yb==NEGATIVE_FACE) ? (ngz+ig) : (ny+ig)) ) ;,
                lmn[2] = (zb==FIRST_ALONG_FACE) 
                         ? (j+ngz) 
                         : ( (zb==SECOND_ALONG_FACE) 
                             ? (k+ngz) 
                             : ((zb==NEGATIVE_FACE) ? (ngz+ig) : (nz+ig)) ) ;
                )
            } ; 
            size_t ijk_a[GRACE_NSPACEDIM], ijk_b[GRACE_NSPACEDIM] ; 
            index_mapping(VEC(ig,j,k), which_face_a, which_face_b, ijk_a, ijk_b) ; 
            #pragma unroll P4EST_CHILDREN
            for( int icc=0; icc<P4EST_CHILDREN; ++icc ) {
                int ix = (icc >> 0) & 1;  
                int iy = (icc >> 1) & 1;  
                int iz = (icc >> 2) & 1;
                view_a(VEC(ijk_a[0]+ix,ijk_a[1]+iy,ijk_a[2]+iz),ivar,qid_a) =
                    view_b(VEC(ijk_b[0]+ix,ijk_b[1]+iy,ijk_b[2]+iz),ivar,qid_b) ;
            }
            if( ! is_ghost ) {
                index_mapping(VEC(ig,j,k), which_face_b, which_face_a, ijk_b, ijk_a) ;
                #pragma unroll P4EST_CHILDREN
                for( int icc=0; icc<P4EST_CHILDREN; ++icc ) {
                    int ix = (icc >> 0) & 1;  
                    int iy = (icc >> 1) & 1;  
                    int iz = (icc >> 2) & 1;
                    view_b(VEC(ijk_b[0]+ix,ijk_b[1]+iy,ijk_b[2]+iz),ivar,qid_b) =
                        view_a(VEC(ijk_a[0]+ix,ijk_a[1]+iy,ijk_a[2]+iz),ivar,qid_a) ;
                }
            }
        });
    #endif
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy_corner(
            {0,VECD(0,0), 0,0},
            {ngz, VECD(ngz, ngz), static_cast<long>(nvars), static_cast<long>(n_corners)}
        ) ;
    parallel_for(GRACE_EXECUTION_TAG("AMR", "copy_interior_ghostzones_across_corners_corner_staggered")
                , policy_corner 
                , KOKKOS_LAMBDA(const size_t& ig, VECD(const size_t& jg, const size_t& kg), const size_t& ivar, const size_t& icorner)
        {
            /* Get information about quadrants sharing the corner */
            int is_ghost     =  d_corner_info(icorner).is_ghost          ; 
            int which_corner_a =  d_corner_info(icorner).which_corner_a      ; 
            int which_corner_b =  d_corner_info(icorner).which_corner_b      ; 
            int tid_a        =  d_corner_info(icorner).which_tree_a      ;
            int tid_b        =  d_corner_info(icorner).which_tree_b      ;
            int64_t qid_a    =  d_corner_info(icorner).qid_a             ;
            int64_t qid_b    =  d_corner_info(icorner).qid_b             ; 
            /* Helper lambda that maps corner indices in and out of ghostzones. */
            auto const index_mapping = [&] ( int const ii, 
                                             int const jj, 
                                             int const kk, 
                                             int const ca, 
                                             int const cb, 
                                             int ijk[GRACE_NSPACEDIM], 
                                             int lmn[GRACE_NSPACEDIM] ) 
            {
                int x = (ca >> 0) & 1;  
                int y = (ca >> 1) & 1;  
                int z = (ca >> 2) & 1;  
                EXPR(
                ijk[0] = (x == 0) ? ii : (nx + ngz + ii);,
                ijk[1] = (y == 0) ? jj : (ny + ngz + jj);,
                ijk[2] = (z == 0) ? kk : (nz + ngz + kk);)
                x = (cb >> 0) & 1;  
                y = (cb >> 1) & 1;  
                z = (cb >> 2) & 1;
                EXPR(
                lmn[0] = (x == 0) ? (ngz + ii) : (nx + ii);,
                lmn[1] = (y == 0) ? (ngz + jj) : (ny + jj);,
                lmn[2] = (z == 0) ? (ngz + kk) : (nz + kk);)
            } ; 
            /* Get correct array to read from / write to */
            auto& view_a = vars ; 
            auto& view_b = (is_ghost) ? halo : vars ; 
            int ijk_a [GRACE_NSPACEDIM], ijk_b [GRACE_NSPACEDIM] ; 
            index_mapping(ig,jg,kg, which_corner_a,which_corner_b, ijk_a,ijk_b) ; 
            #pragma unroll P4EST_CHILDREN
            for( int icc=0; icc<P4EST_CHILDREN; ++icc ) {
                int ix = (icc >> 0) & 1;  
                int iy = (icc >> 1) & 1;  
                int iz = (icc >> 2) & 1;
                view_a(VEC(ijk_a[0]+ix,ijk_a[1]+iy,ijk_a[2]+iz),ivar,qid_a) =
                    view_b(VEC(ijk_b[0]+ix,ijk_b[1]+iy,ijk_b[2]+iz),ivar,qid_b) ;
            }
            if( ! is_ghost ) {
                index_mapping(ig,jg,kg, which_corner_b,which_corner_a, ijk_b,ijk_a) ;
                #pragma unroll P4EST_CHILDREN
                for( int icc=0; icc<P4EST_CHILDREN; ++icc ) {
                    int ix = (icc >> 0) & 1;  
                    int iy = (icc >> 1) & 1;  
                    int iz = (icc >> 2) & 1;
                    view_b(VEC(ijk_b[0]+ix,ijk_b[1]+iy,ijk_b[2]+iz),ivar,qid_b) =
                        view_a(VEC(ijk_a[0]+ix,ijk_a[1]+iy,ijk_a[2]+iz),ivar,qid_a) ;
                }
            }
        });
    #ifdef GRACE_3D
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy_edge(
            {0,0,0, 0,0},
            {ngz, ngz, static_cast<long>(nx), static_cast<long>(nvars), static_cast<long>(n_edges)}
        ) ;
    parallel_for(GRACE_EXECUTION_TAG("AMR", "copy_interior_ghostzones_across_edges")
                , policy_edge 
                , KOKKOS_LAMBDA(const size_t& ig, const size_t& jg, const size_t& k, const size_t& ivar, const size_t& iedge)
        {
            /* Get information about quadrants sharing the edge */
            int is_ghost     =  d_edge_info(iedge).is_ghost          ; 
            int which_edge_a =  d_edge_info(iedge).which_edge_a      ; 
            int which_edge_b =  d_edge_info(iedge).which_edge_b      ; 
            int tid_a        =  d_edge_info(iedge).which_tree_a      ;
            int tid_b        =  d_edge_info(iedge).which_tree_b      ;
            int64_t qid_a    =  d_edge_info(iedge).qid_a             ;
            int64_t qid_b    =  d_edge_info(iedge).qid_b             ; 
            /* Helper lambda that maps edge indices in and out of ghostzones. */
            auto const index_mapping = [&] ( int const iig, 
                                             int const jjg, 
                                             int const kk, 
                                             int const ea, 
                                             int const eb, 
                                             int ijk[GRACE_NSPACEDIM], 
                                             int lmn[GRACE_NSPACEDIM] ) 
            {
                static constexpr const int ALONG_EDGE = -1 ;
                static constexpr const int NEGATIVE_EDGE = 0 ;
                static constexpr const int POSITIVE_EDGE = 1 ;
                static const int edge_directions[3][12] = {
                    {ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, 
                    NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE,
                     NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE},  // x directions
                    {NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, 
                    ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE,
                    NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE}, // y directions
                    {NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE, 
                    NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE, 
                    ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE} // z directions
                };   
                
                // Extract directional values for edges ea and eb
                int x_ea = edge_directions[0][ea];
                int y_ea = edge_directions[1][ea];
                int z_ea = edge_directions[2][ea];

                // Map indices for ijk based on edge ea
                ijk[0] = (x_ea == ALONG_EDGE) ? (kk+ngz) : (x_ea == NEGATIVE_EDGE ? iig : (nx + ngz + iig));
                ijk[1] = (y_ea == ALONG_EDGE) 
                            ? (kk+ngz) 
                            : (x_ea == ALONG_EDGE 
                                ? (y_ea == NEGATIVE_EDGE ? iig : (ny + ngz + iig)) 
                                : (y_ea == NEGATIVE_EDGE ? jjg : (ny + ngz + jjg)));
                ijk[2] = (z_ea == ALONG_EDGE) ? (kk+ngz) : (z_ea == NEGATIVE_EDGE ? jjg : (nz + ngz + jjg));

                int x_eb = edge_directions[0][eb];
                int y_eb = edge_directions[1][eb];
                int z_eb = edge_directions[2][eb];

                // Map indices for lmn based on edge eb
                lmn[0] = (x_eb == ALONG_EDGE) ? (kk+ngz) : (x_eb == NEGATIVE_EDGE ? (ngz + iig) : (nx + iig));
                lmn[1] = (y_eb == ALONG_EDGE) 
                            ? (kk+ngz)
                            : (x_eb == ALONG_EDGE 
                                ? (y_eb == NEGATIVE_EDGE ? (ngz + iig) : (ny + iig)) 
                                : (y_eb == NEGATIVE_EDGE ? (ngz + jjg) : (ny + jjg)));
                lmn[2] = (z_eb == ALONG_EDGE) ? (kk+ngz) : (z_eb == NEGATIVE_EDGE ? (ngz + jjg) : (nz + jjg));
            } ; 
            /* Get correct array to read from / write to */
            auto& view_a = vars ; 
            auto& view_b = (is_ghost) ? halo : vars ; 
            int ijk_a [GRACE_NSPACEDIM], ijk_b [GRACE_NSPACEDIM] ; 
            index_mapping(ig,jg,k,which_edge_a,which_edge_b, ijk_a,ijk_b) ; 
            #pragma unroll P4EST_CHILDREN
            for( int icc=0; icc<P4EST_CHILDREN; ++icc ) {
                int ix = (icc >> 0) & 1;  
                int iy = (icc >> 1) & 1;  
                int iz = (icc >> 2) & 1;
                view_a(VEC(ijk_a[0]+ix,ijk_a[1]+iy,ijk_a[2]+iz),ivar,qid_a) =
                    view_b(VEC(ijk_b[0]+ix,ijk_b[1]+iy,ijk_b[2]+iz),ivar,qid_b) ;
            }
            if( ! is_ghost ) {
                index_mapping(ig,jg,k,which_edge_b,which_edge_a, ijk_b,ijk_a) ;
                #pragma unroll P4EST_CHILDREN
                for( int icc=0; icc<P4EST_CHILDREN; ++icc ) {
                    int ix = (icc >> 0) & 1;  
                    int iy = (icc >> 1) & 1;  
                    int iz = (icc >> 2) & 1;
                    view_b(VEC(ijk_b[0]+ix,ijk_b[1]+iy,ijk_b[2]+iz),ivar,qid_b) =
                        view_a(VEC(ijk_a[0]+ix,ijk_a[1]+iy,ijk_a[2]+iz),ivar,qid_a) ;
                }
            }
        }) ;
    #endif 
}


}} /* namespace grace::amr */
