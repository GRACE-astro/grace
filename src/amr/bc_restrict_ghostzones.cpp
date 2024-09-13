/**
 * @file bc_restrict_ghostzones.cpp
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

#include <grace/amr/bc_restrict_ghostzones.hh>
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

void restrict_hanging_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::staggered_variable_arrays_t& staggered_state
    , grace::staggered_variable_arrays_t& staggered_halo 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& halo_vols 
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
    , Kokkos::vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>& hanging_edges
    #endif 
)
{
    /******************************************************/
    /*                CELL CENTERS                        */
    /******************************************************/
    restrict_hanging_ghostzones_cell_centers(
          state
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
    restrict_hanging_ghostzones_corners(
          staggered_state.corner_staggered_fields
        , staggered_halo.corner_staggered_fields
        , interior_faces 
        , interior_corners 
        #ifdef GRACE_3D 
        , interior_edges 
        #endif 
    ) ;
}

void restrict_hanging_ghostzones_cell_centers(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& halo_vols 
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
    , Kokkos::vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>& hanging_edges
    #endif 
) 
{
    using namespace grace ;
    using namespace Kokkos  ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ;
    int64_t nq  = amr::get_local_num_quadrants() ;
    int const nvars = variables::get_n_evolved() ;
    auto const n_faces = hanging_faces.size()   ;
    auto const n_corners = hanging_corners.size()   ;
    #ifdef GRACE_3D
    auto const n_edges = hanging_edges.size()   ;
    #endif 
    if( EXPR(n_faces == 0, and n_corners == 0, and n_edges == 0) ) {
        return ; 
    }
    auto& d_face_info = hanging_faces.d_view    ; 
    auto& d_corner_info = hanging_corners.d_view    ; 
    #ifdef GRACE_3D 
    auto& d_edge_info = hanging_edges.d_view    ; 
    #endif 
    constexpr const int n_neighbors = PICK_D(2,4) ;  

    utils::vol_average_restrictor_t restriction_kernel ; 
    /*************************************************/
    /* Kernel:                                       */
    /* Restrict data onto coarse quadrants from fine */
    /* neighboring quadrants.                        */
    /*************************************************/
    GRACE_VERBOSE("Initiating restriction on {} hanging interior faces.", n_faces) ;
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy(
            {0,VECD(0,0), 0,0},
            {ngz, VECD(static_cast<long>(nx),static_cast<long>(ny)), static_cast<long>(nvars), static_cast<long>(n_faces)}
        ) ;
    parallel_for(GRACE_EXECUTION_TAG("AMR", "restrict_hanging_faces")
                , policy 
                , KOKKOS_LAMBDA(const size_t& ig, VECD(const size_t& j, const size_t& k), const size_t& ivar, const size_t& iface)
        {
            int polarity     =  d_face_info(iface).has_polarity_flip         ; 
            int8_t is_ghost_coarse   =  d_face_info(iface).is_ghost_coarse   ; 
            int64_t qid_coarse       =  d_face_info(iface).qid_coarse        ;
            int8_t which_face_coarse =  d_face_info(iface).which_face_coarse    ; 
            int8_t which_face_fine   =  d_face_info(iface).which_face_fine      ; 
            int tid_coarse = d_face_info(iface).which_tree_coarse ; 
            int tid_fine   = d_face_info(iface).which_tree_fine   ;
            int8_t is_ghost_fine[P4EST_CHILDREN/2] ; 
            int64_t qid_fine[P4EST_CHILDREN/2] ;
            for( int ii=0; ii<P4EST_CHILDREN/2; ++ ii){
                is_ghost_fine[ii] = d_face_info(iface).is_ghost_fine[ii] ;
                qid_fine[ii]      = d_face_info(iface).qid_fine[ii]      ; 
            } 
            int64_t const ng = (which_face_fine/2==0) * nx + ((which_face_fine/2==1) * ny) + ((which_face_fine/2==2) * nz) ;
            int64_t const n1 = (which_face_fine/2==0) * ny + ((which_face_fine/2==1) * nx) + ((which_face_fine/2==2) * nx) ;
            int64_t const n2 = (which_face_fine/2==0) * nz + ((which_face_fine/2==1) * nz) + ((which_face_fine/2==2) * ny) ;
            #ifdef GRACE_SPHERICAL_COORDINATES
            index_helper_t mapper{} ; 
            #endif 
            if( ! is_ghost_coarse )
            {
            
                const int8_t ichild = EXPRD(
                        math::floor_int((2*j)/n1)
                    , + math::floor_int((2*k)/n2) * 2
                    ) ; 
                int64_t qid_b   = qid_fine[ichild] ; 
                auto& fine_view = is_ghost_fine[ichild] ? halo : state ; 
                auto& fine_vol  = is_ghost_fine[ichild] ? halo_vols : vols ; 
                /* Compute indices of cell to be filled */
                EXPR( 
                int const i_c = EXPR((which_face_coarse==0) * ig 
                                + (which_face_coarse==1) * (nx+ngz+ig),
                                + (which_face_coarse/2==1) * (j+ngz), 
                                + (which_face_coarse/2==2) * (j+ngz)) ;,
                int const j_c = EXPR((which_face_coarse==2) * ig
                                + (which_face_coarse==3) * (ny+ngz+ig), 
                                + (which_face_coarse/2==0) * (j+ngz), 
                                + (which_face_coarse/2==2) * (k+ngz));  ,
                int const k_c = (which_face_coarse==4) * ig 
                                + (which_face_coarse==5) * (nz+ngz+ig)  
                                + (which_face_coarse/2!=2) * (k+ngz) ;
                )  
                #ifdef GRACE_SPHERICAL_COORDINATES
                int64_t const VEC( Ig{ (2*ig)%ng }, I1{ (2*j)%n1 + ngz }, I2{ (2*k)%n2 + ngz } ) ; 
                EXPR(
                    int i_f = 
                        (which_face_fine == 0) * ( ngz + Ig      )
                    +   (which_face_fine == 1) * ( ng - ngz + Ig )
                    +   (which_face_fine/2 == 1) * I1 
                    +   (which_face_fine/2 == 2) * I1 ;, 
                    int j_f = EXPR(
                        (which_face_fine == 2) * ( ngz + Ig      )
                    +   (which_face_fine == 3) * ( ng - ngz + Ig ),
                    +   (which_face_fine/2 == 0) * I1, 
                    +   (which_face_fine/2 == 2) * I2) ;, 
                    int k_f = 
                        (which_face_fine == 4) * ( ngz + Ig      )
                    +   (which_face_fine == 5) * ( ng - ngz + Ig )
                    +   (which_face_fine/2 == 0) * I2 
                    +   (which_face_fine/2 == 1) * I2 ;
                )
                // HERE WE ASSUME N1==N2==N3
                lmn = mapper({VEC(i_f,j_f,k_f)}, tid_coearse, tid_fine, {VEC(ng,n1,n2)}) ; 
                i_f = lmn[0]; j_f = lmn[1]; k_f = lmn[2];
                #else 
                size_t const VEC( Ig{ (2*ig)%ng }, I1{ (2*j)%n1 + ngz }, I2{ (2*k)%n2 + ngz } ) ; 
                EXPR(
                    const int i_f = 
                        (which_face_fine == 0) * ( ngz + Ig      )
                    +   (which_face_fine == 1) * ( ng - ngz + Ig )
                    +   (which_face_fine/2 == 1) * I1 
                    +   (which_face_fine/2 == 2) * I1 ;, 
                    const int j_f = EXPR(
                        (which_face_fine == 2) * ( ngz + Ig      )
                    +   (which_face_fine == 3) * ( ng - ngz + Ig ),
                    +   (which_face_fine/2 == 0) * I1, 
                    +   (which_face_fine/2 == 2) * I2) ;, 
                    const int k_f = 
                        (which_face_fine == 4) * ( ngz + Ig      )
                    +   (which_face_fine == 5) * ( ng - ngz + Ig )
                    +   (which_face_fine/2 == 0) * I2 
                    +   (which_face_fine/2 == 1) * I2 ;
                )
                #endif 
                /* Call restriction operator on fine data */ 
                state(VEC(i_c,j_c,k_c),ivar,qid_coarse) = 
                    utils::vol_average_restrictor_t::apply(VEC(i_f,j_f,k_f), fine_view, fine_vol, qid_b, ivar) ;  
                
            }
        }
    )   ;
    GRACE_VERBOSE("Initiating restriction on {} hanging interior corners.", n_corners) ; 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        corner_policy(
            {0,VECD(0,0), 0,0},
            {VEC(ngz,ngz,ngz), static_cast<long>(nvars), static_cast<long>(n_corners)}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("AMR", "restrict_hanging_corners")
                , corner_policy 
                , KOKKOS_LAMBDA(VEC(int const ig, int const jg, int const kg), int const ivar, int const icorner)
        {
            /* Collect the necessary information                                */
            int8_t is_ghost_coarse   =  d_corner_info(icorner).is_ghost_coarse       ; 
            int64_t qid_coarse       =  d_corner_info(icorner).qid_coarse            ;
            int8_t which_corner_coarse =  d_corner_info(icorner).which_corner_coarse ; 
            int8_t which_corner_fine   =  d_corner_info(icorner).which_corner_fine   ; 
            int tid_coarse           =  d_corner_info(icorner).which_tree_coarse     ; 
            int tid_fine             =  d_corner_info(icorner).which_tree_fine       ;
            int8_t is_ghost_fine     =  d_corner_info(icorner).is_ghost_fine         ;  
            int64_t qid_fine         =  d_corner_info(icorner).qid_fine              ;
            
            if ( ! is_ghost_coarse ) {
                auto& fine_view = is_ghost_fine ? halo : state ; 
                #ifndef GRACE_CARTESIAN_COORDINATES
                auto& fine_vols = is_ghost_fine[ichild] ? halo_vols : vols ; 
                #endif 
                /* Utility to map the coarse index into the fine quadrant */
                auto const index_mapping = [&] (
                    VEC(int const ig, int const jg, int const kg),
                    int const ca, int const cb, int IJK[GRACE_NSPACEDIM], int ijk[GRACE_NSPACEDIM]
                )
                {
                    int x = (ca >> 0) & 1;  
                    int y = (ca >> 1) & 1;  
                    int z = (ca >> 2) & 1;
                    EXPR(
                    IJK[0] = (x==0) ? ig : (nx + ngz + ig) ;, 
                    IJK[1] = (y==0) ? jg : (ny + ngz + jg) ;,
                    IJK[2] = (z==0) ? kg : (nz + ngz + kg) ;
                    )
                    x = (cb >> 0) & 1;  
                    y = (cb >> 1) & 1;  
                    z = (cb >> 2) & 1;
                    EXPR(
                    ijk[0] = (x==0) ? (ngz+2*ig) : (nx - ngz + 2*ig) ;, 
                    ijk[1] = (y==0) ? (ngz+2*jg) : (ny - ngz + 2*jg) ;,
                    ijk[2] = (z==0) ? (ngz+2*kg) : (nz - ngz + 2*kg) ;
                    )
                } ; 

                int ijk_f[GRACE_NSPACEDIM], ijk_c[GRACE_NSPACEDIM] ; 
                index_mapping(VEC(ig,jg,kg), which_corner_coarse, which_corner_fine, ijk_c, ijk_f ) ; 

                /* Call the resstriction operator on the correct indices */
                #ifndef GRACE_CARTESIAN_COORDINATES
                state(VEC(ijk_c[0],ijk_c[1],ijk_c[2]),ivar,qid_coarse) = 
                    utils::vol_average_restrictor_t::apply(VEC(ijk_f[0],ijk_f[1],ijk_f[2]), fine_view, fine_vols, qid_fine, ivar);
                #else 
                state(VEC(ijk_c[0],ijk_c[1],ijk_c[2]),ivar,qid_coarse) = 
                    utils::vol_average_restrictor_t::apply(VEC(ijk_f[0],ijk_f[1],ijk_f[2]), fine_view, qid_fine, ivar); 
                #endif  
            } 
            
        } 
    ) ; 
    GRACE_VERBOSE("Initiating restriction on {} hanging interior edges.", n_edges) ; 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy_edge(
            {0,0,0, 0,0},
            {ngz, ngz, static_cast<long>(nx), static_cast<long>(nvars), static_cast<long>(n_edges)}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("AMR", "restrict_hanging_edges")
                , policy_edge 
                , KOKKOS_LAMBDA(const size_t& ig, const size_t& jg, const size_t& k, const size_t& ivar, const size_t& iedge)
        {

            /* Collect the necessary information                                */
            int8_t is_ghost_coarse   =  d_edge_info(iedge).is_ghost_coarse       ; 
            int64_t qid_coarse       =  d_edge_info(iedge).qid_coarse            ;
            int8_t which_edge_coarse =  d_edge_info(iedge).which_edge_coarse     ; 
            int8_t which_edge_fine   =  d_edge_info(iedge).which_edge_fine       ; 
            int tid_coarse           =  d_edge_info(iedge).which_tree_coarse     ; 
            int tid_fine             =  d_edge_info(iedge).which_tree_fine       ;

            int8_t is_ghost_fine[2]  = {
                d_edge_info(iedge).is_ghost_fine[0],
                d_edge_info(iedge).is_ghost_fine[1]
            } ; 
            int64_t qid_fine[2]      =  {
                d_edge_info(iedge).qid_fine[0],
                d_edge_info(iedge).qid_fine[1]
            } ;

            if ( ! is_ghost_coarse ) {
                const int ichild = math::floor_int( (2*k) / nx )    ;  // was int8 
                const int64_t qid_child = d_edge_info(iedge).qid_fine[ichild]             ;
                auto& fine_view = is_ghost_fine[ichild] ? halo : state ; 
                #ifndef GRACE_CARTESIAN_COORDINATES
                auto& fine_vols = is_ghost_fine[ichild] ? halo_vols : vols ; 
                #endif
                /* Utility to map the coarse index into the fine quadrant */
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
                    lmn[0] = (x_eb == ALONG_EDGE) ? ((2*kk)%nx+ngz) : (x_eb == NEGATIVE_EDGE ? (ngz + 2*iig) : (nx - ngz + 2*iig));
                    lmn[1] = (y_eb == ALONG_EDGE) 
                                ? ((2*kk)%ny+ngz)
                                : (x_eb == ALONG_EDGE 
                                    ? (y_eb == NEGATIVE_EDGE ? (ngz + 2*iig) : (ny - ngz + 2*iig)) 
                                    : (y_eb == NEGATIVE_EDGE ? (ngz + 2*jjg) : (ny - ngz + 2*jjg)));
                    lmn[2] = (z_eb == ALONG_EDGE) ? ((2*kk)%nz+ngz) : (z_eb == NEGATIVE_EDGE ? (ngz + 2*jjg) : (nz - ngz + 2*jjg));
                } ;
                int ijk_c[GRACE_NSPACEDIM], ijk_f[GRACE_NSPACEDIM];
                index_mapping(ig,jg,k, which_edge_coarse, which_edge_fine, ijk_c, ijk_f) ; 

                /* Call the resstriction operator on the correct indices */
                #ifndef GRACE_CARTESIAN_COORDINATES
                state(VEC(ijk_c[0],ijk_c[1],ijk_c[2]),ivar,qid_coarse) = 
                    utils::vol_average_restrictor_t::apply(VEC(ijk_f[0],ijk_f[1],ijk_f[2]), fine_view, fine_vols, qid_child, ivar); 
                #else 
                state(VEC(ijk_c[0],ijk_c[1],ijk_c[2]),ivar,qid_coarse) = 
                    utils::vol_average_restrictor_t::apply(VEC(ijk_f[0],ijk_f[1],ijk_f[2]), fine_view, qid_child, ivar); 
                #endif 
            }
        }
    ) ; 

}

void restrict_hanging_ghostzones_corners(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
    , Kokkos::vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>& hanging_edges
    #endif 
) 
{
    using namespace grace ;
    using namespace Kokkos  ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ;
    int64_t nq  = amr::get_local_num_quadrants() ;
    int const nvars = variables::get_n_evolved_corner_staggered() ;
    auto const n_faces = hanging_faces.size()   ;
    auto const n_corners = hanging_corners.size()   ;
    #ifdef GRACE_3D
    auto const n_edges = hanging_edges.size()   ;
    #endif 
    if( EXPR(n_faces == 0, and n_corners == 0, and n_edges == 0) ) {
        return ; 
    }
    auto& d_face_info = hanging_faces.d_view    ; 
    auto& d_corner_info = hanging_corners.d_view    ; 
    #ifdef GRACE_3D 
    auto& d_edge_info = hanging_edges.d_view    ; 
    #endif 
    constexpr const int n_neighbors = PICK_D(2,4) ;  

    /*************************************************/
    /* Kernel:                                       */
    /* Restrict data onto coarse quadrants from fine */
    /* neighboring quadrants.                        */
    /*************************************************/
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy(
            {0,VECD(0,0), 0,0},
            {ngz+1, VECD(static_cast<long>(nx+1),static_cast<long>(ny+1)), static_cast<long>(nvars), static_cast<long>(n_faces)}
        ) ;
    parallel_for(GRACE_EXECUTION_TAG("AMR", "restrict_hanging_faces_corner_staggered")
                , policy 
                , KOKKOS_LAMBDA(const size_t& ig, VECD(const size_t& j, const size_t& k), const size_t& ivar, const size_t& iface)
        {
            int polarity     =  d_face_info(iface).has_polarity_flip         ; 
            int8_t is_ghost_coarse   =  d_face_info(iface).is_ghost_coarse   ; 
            int64_t qid_coarse       =  d_face_info(iface).qid_coarse        ;
            int8_t which_face_coarse =  d_face_info(iface).which_face_coarse    ; 
            int8_t which_face_fine   =  d_face_info(iface).which_face_fine      ; 
            int tid_coarse = d_face_info(iface).which_tree_coarse ; 
            int tid_fine   = d_face_info(iface).which_tree_fine   ;
            int8_t is_ghost_fine[P4EST_CHILDREN/2] ; 
            int64_t qid_fine[P4EST_CHILDREN/2] ;
            for( int ii=0; ii<P4EST_CHILDREN/2; ++ ii){
                is_ghost_fine[ii] = d_face_info(iface).is_ghost_fine[ii] ;
                qid_fine[ii]      = d_face_info(iface).qid_fine[ii]      ; 
            } 
            int64_t const ng = (which_face_fine/2==0) * nx + ((which_face_fine/2==1) * ny) + ((which_face_fine/2==2) * nz) ;
            int64_t const n1 = (which_face_fine/2==0) * ny + ((which_face_fine/2==1) * nx) + ((which_face_fine/2==2) * nx) ;
            int64_t const n2 = (which_face_fine/2==0) * nz + ((which_face_fine/2==1) * nz) + ((which_face_fine/2==2) * ny) ;
            #ifdef GRACE_SPHERICAL_COORDINATES
            index_helper_t mapper{} ; 
            #endif 
            if( ! is_ghost_coarse )
            {
            
                const int8_t ichild = EXPRD(
                        math::floor_int((2*j)/n1)
                    , + math::floor_int((2*k)/n2) * 2
                    ) ; 
                int64_t qid_b   = qid_fine[ichild] ; 
                auto& fine_view = is_ghost_fine[ichild] ? halo : state ; 
                /* Compute indices of cell to be filled */
                /* NB: here we are restricting corner-staggered vars which */
                /* means that the extent of the grid is nx+1, ny+1, nz+1.  */
                /* However since the coarse and fine data coincides on the */
                /* edges of the physical grid here we replace the coarse   */
                /* entry with the corresponding fine one. This implies that*/
                /* whenever we are dealing with the upper face nx/y/z won't*/
                /* be replaced by nx/y/z+1 (this is also why the loop goes */
                /* to ngz+1).                                              */
                EXPR( 
                int const i_c = EXPR((which_face_coarse==0) * ig 
                                + (which_face_coarse==1) * (nx+ngz+ig),
                                + (which_face_coarse/2==1) * (j+ngz), 
                                + (which_face_coarse/2==2) * (j+ngz)) ;,
                int const j_c = EXPR((which_face_coarse==2) * ig
                                + (which_face_coarse==3) * (ny+ngz+ig), 
                                + (which_face_coarse/2==0) * (j+ngz), 
                                + (which_face_coarse/2==2) * (k+ngz));  ,
                int const k_c = (which_face_coarse==4) * ig 
                                + (which_face_coarse==5) * (nz+ngz+ig)  
                                + (which_face_coarse/2!=2) * (k+ngz) ;
                )  
                /* For what concerns the directions within the face */
                /* since the last point of a fine child and the     */
                /* first point along the same axis of the neighbor  */
                /* coincide, it does not matter which one we pick   */
                size_t const VEC( Ig{ (2*ig)%ng }, I1{ (2*j)%n1 + ngz }, I2{ (2*k)%n2 + ngz } ) ; 
                /* Same concept as above, nx/y/z does not need a +1 */
                EXPR(
                    const int i_f = 
                        (which_face_fine == 0) * ( ngz + Ig      )
                    +   (which_face_fine == 1) * ( ng - ngz + Ig )
                    +   (which_face_fine/2 == 1) * I1 
                    +   (which_face_fine/2 == 2) * I1 ;, 
                    const int j_f = EXPR(
                        (which_face_fine == 2) * ( ngz + Ig      )
                    +   (which_face_fine == 3) * ( ng - ngz + Ig ),
                    +   (which_face_fine/2 == 0) * I1, 
                    +   (which_face_fine/2 == 2) * I2) ;, 
                    const int k_f = 
                        (which_face_fine == 4) * ( ngz + Ig      )
                    +   (which_face_fine == 5) * ( ng - ngz + Ig )
                    +   (which_face_fine/2 == 0) * I2 
                    +   (which_face_fine/2 == 1) * I2 ;
                )
                #endif 
                /* Call restriction operator on fine data */ 
                state(VEC(i_c,j_c,k_c),ivar,qid_coarse) = fine_view(VEC(i_f,j_f,k_f, ivar, qid_b)) ;  
            }
        }
    )   ;
    GRACE_VERBOSE("Initiating restriction on {} hanging interior corners.", n_corners) ; 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        corner_policy(
            {0,VECD(0,0), 0,0},
            {VEC(ngz+1,ngz+1,ngz+1), static_cast<long>(nvars), static_cast<long>(n_corners)}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("AMR", "restrict_hanging_corners")
                , corner_policy 
                , KOKKOS_LAMBDA(VEC(int const ig, int const jg, int const kg), int const ivar, int const icorner)
        {
            /* Collect the necessary information                                */
            int8_t is_ghost_coarse   =  d_corner_info(icorner).is_ghost_coarse       ; 
            int64_t qid_coarse       =  d_corner_info(icorner).qid_coarse            ;
            int8_t which_corner_coarse =  d_corner_info(icorner).which_corner_coarse ; 
            int8_t which_corner_fine   =  d_corner_info(icorner).which_corner_fine   ; 
            int tid_coarse           =  d_corner_info(icorner).which_tree_coarse     ; 
            int tid_fine             =  d_corner_info(icorner).which_tree_fine       ;
            int8_t is_ghost_fine     =  d_corner_info(icorner).is_ghost_fine         ;  
            int64_t qid_fine         =  d_corner_info(icorner).qid_fine              ;
            
            if ( ! is_ghost_coarse ) {
                auto& fine_view = is_ghost_fine ? halo : state ; 
                /* Utility to map the coarse index into the fine quadrant */
                auto const index_mapping = [&] (
                    VEC(int const ig, int const jg, int const kg),
                    int const ca, int const cb, int IJK[GRACE_NSPACEDIM], int ijk[GRACE_NSPACEDIM]
                )
                {
                    int x = (ca >> 0) & 1;  
                    int y = (ca >> 1) & 1;  
                    int z = (ca >> 2) & 1;
                    EXPR(
                    IJK[0] = (x==0) ? ig : (nx + ngz + ig) ;, 
                    IJK[1] = (y==0) ? jg : (ny + ngz + jg) ;,
                    IJK[2] = (z==0) ? kg : (nz + ngz + kg) ;
                    )
                    x = (cb >> 0) & 1;  
                    y = (cb >> 1) & 1;  
                    z = (cb >> 2) & 1;
                    EXPR(
                    ijk[0] = (x==0) ? (ngz+2*ig) : (nx - ngz + 2*ig) ;, 
                    ijk[1] = (y==0) ? (ngz+2*jg) : (ny - ngz + 2*jg) ;,
                    ijk[2] = (z==0) ? (ngz+2*kg) : (nz - ngz + 2*kg) ;
                    )
                } ; 

                int ijk_f[GRACE_NSPACEDIM], ijk_c[GRACE_NSPACEDIM] ; 
                index_mapping(VEC(ig,jg,kg), which_corner_coarse, which_corner_fine, ijk_c, ijk_f ) ; 

                state(VEC(ijk_c[0],ijk_c[1],ijk_c[2]),ivar,qid_coarse) = 
                    fine_view(VEC(ijk_f[0],ijk_f[1],ijk_f[2]), ivar, qid_fine) ; 
            } 
            
        } 
    ) ; 
    GRACE_VERBOSE("Initiating restriction on {} hanging interior edges.", n_edges) ; 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy_edge(
            {0,0,0, 0,0},
            {ngz+1, ngz+1, static_cast<long>(nx+1), static_cast<long>(nvars), static_cast<long>(n_edges)}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("AMR", "restrict_hanging_edges")
                , policy_edge 
                , KOKKOS_LAMBDA(const size_t& ig, const size_t& jg, const size_t& k, const size_t& ivar, const size_t& iedge)
        {

            /* Collect the necessary information                                */
            int8_t is_ghost_coarse   =  d_edge_info(iedge).is_ghost_coarse       ; 
            int64_t qid_coarse       =  d_edge_info(iedge).qid_coarse            ;
            int8_t which_edge_coarse =  d_edge_info(iedge).which_edge_coarse     ; 
            int8_t which_edge_fine   =  d_edge_info(iedge).which_edge_fine       ; 
            int tid_coarse           =  d_edge_info(iedge).which_tree_coarse     ; 
            int tid_fine             =  d_edge_info(iedge).which_tree_fine       ;

            int8_t is_ghost_fine[2]  = {
                d_edge_info(iedge).is_ghost_fine[0],
                d_edge_info(iedge).is_ghost_fine[1]
            } ; 
            int64_t qid_fine[2]      =  {
                d_edge_info(iedge).qid_fine[0],
                d_edge_info(iedge).qid_fine[1]
            } ;

            if ( ! is_ghost_coarse ) {
                const int ichild = math::floor_int( (2*k) / nx )    ;  // was int8 
                const int64_t qid_child = d_edge_info(iedge).qid_fine[ichild]             ;
                auto& fine_view = is_ghost_fine[ichild] ? halo : state ; 
                /* Utility to map the coarse index into the fine quadrant */
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
                    lmn[0] = (x_eb == ALONG_EDGE) ? ((2*kk)%nx+ngz) : (x_eb == NEGATIVE_EDGE ? (ngz + 2*iig) : (nx - ngz + 2*iig));
                    lmn[1] = (y_eb == ALONG_EDGE) 
                                ? ((2*kk)%ny+ngz)
                                : (x_eb == ALONG_EDGE 
                                    ? (y_eb == NEGATIVE_EDGE ? (ngz + 2*iig) : (ny - ngz + 2*iig)) 
                                    : (y_eb == NEGATIVE_EDGE ? (ngz + 2*jjg) : (ny - ngz + 2*jjg)));
                    lmn[2] = (z_eb == ALONG_EDGE) ? ((2*kk)%nz+ngz) : (z_eb == NEGATIVE_EDGE ? (ngz + 2*jjg) : (nz - ngz + 2*jjg));
                } ;
                int ijk_c[GRACE_NSPACEDIM], ijk_f[GRACE_NSPACEDIM];
                index_mapping(ig,jg,k, which_edge_coarse, which_edge_fine, ijk_c, ijk_f) ; 

                /* Call the resstriction operator on the correct indices */
                state(VEC(ijk_c[0],ijk_c[1],ijk_c[2]),ivar,qid_coarse) = 
                    fine_view(VEC(ijk_f[0],ijk_f[1],ijk_f[2]), ivar, qid_child) ; 
            }
        }
    ) ; 

}

}} /* namespace grace::amr */