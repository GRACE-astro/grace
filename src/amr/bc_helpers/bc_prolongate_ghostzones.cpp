/**
 * @file bc_prolongate_ghostzones.cpp
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

#include <grace/amr/bc_prolongate_ghostzones.hh>
#include <grace/amr/prolongation_kernels.tpp> 
#include <grace/amr/restriction_kernels.tpp> 
#include <grace/config/config_parser.hh>
#include <grace/utils/numerics/prolongation.hh>
#include <grace/utils/numerics/limiters.hh> 
#include <grace/utils/numerics/prolongation.hh>
#include <grace/utils/numerics/math.hh>
#include <grace/amr/boundary_conditions.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/bc_helpers.tpp>
#include <grace/amr/grace_amr.hh> 
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/numerics/interpolators.hh>
#include <grace/data_structures/grace_data_structures.hh>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

#include <string>

namespace grace { namespace amr {
/**************************************************************************************************/
void prolongate_hanging_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo
    , grace::staggered_variable_arrays_t& staggered_state
    , grace::staggered_variable_arrays_t& staggered_halo
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& halo_vols 
    , grace::device_vector<hanging_face_info_t>& hanging_faces
    , grace::device_vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , grace::device_vector<hanging_edge_info_t>& hanging_edges
    #endif 
)
{
    auto const cell_center_limiter = grace::get_param<std::string>("amr", "prolongation_limiter_type") ; 

    if( cell_center_limiter == "minmod" ) {
        prolongate_hanging_ghostzones_cell_centers<utils::linear_prolongator_t<grace::minmod>>(
                  state
                , halo
                , vols
                , halo_vols
                , hanging_faces
                , hanging_corners
                #ifdef GRACE_3D
                , hanging_edges
                #endif 
        ) ; 
    } else if ( cell_center_limiter == "monotonized-central") {
        prolongate_hanging_ghostzones_cell_centers<utils::linear_prolongator_t<grace::MCbeta>>(
                  state
                , halo
                , vols
                , halo_vols
                , hanging_faces
                , hanging_corners
                #ifdef GRACE_3D
                , hanging_edges
                #endif 
        ) ;
    } else {
        ERROR("Unsupported limiter in ghost-zone prolongation for cell-centered vars.") ;
    }

    auto const corner_interp_order = grace::get_param<int>("amr", "prolongation_order") ; 
    auto& corner_state = staggered_state.corner_staggered_fields ; 
    auto& corner_halo  = staggered_halo.corner_staggered_fields  ; 

    if (corner_interp_order == 2) {
        prolongate_hanging_ghostzones_corners<utils::lagrange_prolongator_t<2>>(
              corner_state
            , corner_halo 
            , hanging_faces
            , hanging_corners
            #ifdef GRACE_3D
            , hanging_edges
            #endif 
        ) ; 
    } else if ( corner_interp_order == 4 ) {
        prolongate_hanging_ghostzones_corners<utils::lagrange_prolongator_t<4>>(
              corner_state
            , corner_halo 
            , hanging_faces
            , hanging_corners
            #ifdef GRACE_3D
            , hanging_edges
            #endif 
        ) ; 
    }  

}
/**************************************************************************************************/
/**************************************************************************************************/
template< typename InterpT > 
void prolongate_hanging_ghostzones_cell_centers(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& halo_vols 
    , grace::device_vector<hanging_face_info_t>& hanging_faces
    , grace::device_vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , grace::device_vector<hanging_edge_info_t>& hanging_edges
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
    
    if( (EXPR(n_faces == 0, and n_corners == 0, and n_edges == 0)) ){
        return ; 
    }
    auto& d_face_info = hanging_faces.d_view    ; 
    auto& d_corner_info = hanging_corners.d_view    ; 
    #ifdef GRACE_3D 
    auto& d_edge_info = hanging_edges.d_view    ; 
    #endif 

    constexpr const int n_neighbors = PICK_D(2,4) ;  

    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
    policy(
        {0,VECD(0,0), 0,0},
        {ngz, VECD(static_cast<long>(nx+2*ngz),static_cast<long>(ny+2*ngz)), static_cast<long>(nvars), static_cast<long>(n_faces)}
    ) ;
    
    /*************************************************/
    /* Kernel:                                       */
    /* Prolongate data onto fine quadrants ghost     */
    /* zones from coarse neighboring quadrants.      */
    /*************************************************/
    parallel_for
    (             GRACE_EXECUTION_TAG("AMR","prolongate_hanging_faces")
                , policy 
                , KOKKOS_LAMBDA( const size_t& ig, VECD(const size_t& j, const size_t& k), const size_t& ivar, const size_t& iface )
        {
            int polarity     =  d_face_info(iface).has_polarity_flip         ; 
            int8_t is_ghost_coarse   =  d_face_info(iface).is_ghost_coarse   ; 
            int64_t iq_coarse        =  d_face_info(iface).qid_coarse        ;
            int8_t which_face_coarse =  d_face_info(iface).which_face_coarse    ; 
            int8_t which_face_fine   =  d_face_info(iface).which_face_fine      ; 
            int tid_coarse           =  d_face_info(iface).which_tree_coarse    ; 
            int tid_fine             =  d_face_info(iface).which_tree_fine      ; 
            int8_t is_ghost_fine[P4EST_CHILDREN/2] ; 
            int64_t qid_fine[P4EST_CHILDREN/2] ;
            for( int ii=0; ii<P4EST_CHILDREN/2; ++ ii){
                is_ghost_fine[ii] = d_face_info(iface).is_ghost_fine[ii] ;
                qid_fine[ii]      = d_face_info(iface).qid_fine[ii]      ; 
            }  
            
            int64_t n1 = (which_face_fine/2==0) * ny + ((which_face_fine/2==1) * nx) + ((which_face_fine/2==2) * nx) ;
            int64_t n2 = (which_face_fine/2==0) * nz + ((which_face_fine/2==1) * nz) + ((which_face_fine/2==2) * ny) ;

            auto& coarse_view   = is_ghost_coarse ? halo : state ; 

            #ifndef GRACE_CARTESIAN_COORDINATES
            index_helper_t mapper{} ;
            #endif 
            
            /* First we compute the indices of the point */
            /* we are calculating.                       */
            EXPR( 
            int const i_f = EXPR((which_face_fine==0) * ig 
                        + (which_face_fine==1) * (nx+ngz+ig),
                        + (which_face_fine/2==1) * (j), 
                        + (which_face_fine/2==2) * (j)) ;,
            int const j_f = EXPR((which_face_fine==2) * ig
                        + (which_face_fine==3) * (ny+ngz+ig), 
                        + (which_face_fine/2==0) * (j), 
                        + (which_face_fine/2==2) * (k));  ,
            int const k_f = (which_face_fine==4) * ig 
                        + (which_face_fine==5) * (nz+ngz+ig)  
                        + (which_face_fine/2!=2) * (k) ;
            )
            /* Then we loop over all child quadrants */
            /* and call the prolongation kernel.     */
            #pragma unroll 4
            for( int ichild=0; ichild<GRACE_FACE_CHILDREN; ++ichild) {
                if( is_ghost_fine[ichild] ) continue ; 
                
                int64_t iq_fine = qid_fine[ichild] ; 
                /* 
                * First we need to find the physical index 
                * in the coarse quadrant closest 
                * to the requested ghost index in the fine
                * quadrant. 
                */ 
                EXPRD( 
                int const iquad_1 = ichild % 2 ;, 
                int const iquad_2 = static_cast<int>(math::floor_int(ichild/2))%2;
                )
                int const VECD( I1{ ngz - math::floor_int(ngz/2) + math::floor_int( (iquad_1*n1 + j)/ 2) }
                              , I2{ ngz - math::floor_int(ngz/2) + math::floor_int( (iquad_2*n2 + k)/ 2) } ) ; 
                #ifdef GRACE_CARTESIAN_COORDINATES
                EXPR(
                int const i_c = 
                          (which_face_coarse == 0) * (ngz + math::floor_int( ig / 2 ) )
                        + (which_face_coarse == 1) * (nx + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) )
                        + (which_face_coarse/2!=0) * I1 ;,

                int const j_c = EXPR(
                          (which_face_coarse == 2) * (ngz + math::floor_int( ig / 2 ) )
                        + (which_face_coarse == 3) * (ny + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) ),
                        + (which_face_coarse/2==0) * I1, 
                        + (which_face_coarse/2==2) * I2 );,

                int const k_c =  
                          (which_face_coarse == 4) * (ngz + math::floor_int( ig / 2 ) )
                        + (which_face_coarse == 5) * (nz + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) )
                        + (which_face_coarse/2!=2) * I2 ;
                )
                #else 
                EXPR(
                int i_c = 
                        (which_face_coarse == 0) * (ngz + math::floor_int( ig / 2 ) )
                        + (which_face_coarse == 1) * (nx + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) )
                        + (which_face_coarse/2!=0) * I1 ;,

                int j_c = EXPR(
                        (which_face_coarse == 2) * (ngz + math::floor_int( ig / 2 ) )
                        + (which_face_coarse == 3) * (ny + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) ),
                        + (which_face_coarse/2==0) * I1, 
                        + (which_face_coarse/2==2) * I2 );,

                int k_c =  
                        (which_face_coarse == 4) * (ngz + math::floor_int( ig / 2 ) )
                        + (which_face_coarse == 5) * (nz + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) )
                        + (which_face_coarse/2!=2) * I2 ;
                )
                auto const lmn = mapper({VEC(i_c,j_c,k_c)}, tid_fine,tid_coarse, {ng,n1,n2}) ; 
                i_c = lmn[0]; j_c = lmn[1] ; k_c = lmn[2] ;
                #endif  
                /* Get signs for slope corrections */
                EXPR(  
                int const sign_x = (which_face_coarse == 0) * (
                                        (!polarity) * ( (ig%2==1) - (ig%2==0) )
                                    + (polarity) *  ( ((ig)%2==0) - ((ig)%2==1) )
                                    )
                                    + (which_face_coarse == 1) * (
                                        (!polarity) * ( (ig%2==1) - (ig%2==0) )
                                    + (polarity) *  ( (ig%2==0) - (ig%2==1) )
                                    )
                                    + (which_face_coarse/2 != 0) * (
                                    (j % 2==1) - (j % 2==0)
                                    ) ;,
                int const sign_y = EXPR((which_face_coarse == 2) * (
                                        (!polarity) * ( (ig%2==1) - (ig%2==0) )
                                    + (polarity) *  ( (ig%2==0) - (ig%2==1) )
                                    )
                                    + (which_face_coarse == 3) * (
                                        (!polarity) * ( (ig%2==1) - (ig%2==0) )
                                    + (polarity) *  ( (ig%2==0) - (ig%2==1) )
                                    ),
                                    + (which_face_coarse/2 == 0) * (
                                    (j % 2==1) - (j % 2==0)
                                    ),
                                    + (which_face_coarse/2 == 2) * (
                                    (k % 2==1) - (k % 2==0)
                                    ) );, 
                int const sign_z = (which_face_coarse == 4) * (
                                        (!polarity) * ( (ig%2==1) - (ig%2==0) )
                                    + (polarity) *  ( (ig%2==0) - (ig%2==1) )
                                    )
                                    + (which_face_coarse == 5) * (
                                        (!polarity) * ( (ig%2==1) - (ig%2==0) )
                                    + (polarity) *  ( (ig%2==0) - (ig%2==1) )
                                    )
                                    + (which_face_coarse/2 != 2) * (
                                    (k % 2==1) - (k % 2==0)
                                    ) ; )
                #ifndef GRACE_CARTESIAN_COORDINATES
                state(VEC(i_f,j_f,k_f), ivar, iq_fine)
                    = InterpT::interpolate( VEC(i_f,j_f,k_f)
                                            , VEC(i_c,j_c,k_c)
                                            , iq_fine, iq_coarse, ngz, ivar
                                            , VEC(sign_x, sign_y, sign_z)
                                            , coarse_view 
                                            , vols ) ;
                #else 
                state(VEC(i_f,j_f,k_f), ivar, iq_fine)
                    = InterpT::interpolate( VEC(i_f,j_f,k_f)
                                            , VEC(i_c,j_c,k_c)
                                            , iq_fine, iq_coarse, ngz, ivar
                                            , VEC(sign_x, sign_y, sign_z)
                                            , coarse_view  ) ;
                #endif 
            }  
        }
    )   ; /* end of loop over faces */
}
/**************************************************************************************************/
template< typename InterpT > 
void prolongate_hanging_ghostzones_corners(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::device_vector<hanging_face_info_t>& hanging_faces
    , grace::device_vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , grace::device_vector<hanging_edge_info_t>& hanging_edges
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
    
    if( (EXPR(n_faces == 0, and n_corners == 0, and n_edges == 0)) or nvars==0 ){
        return ; 
    }
    auto& d_face_info = hanging_faces.d_view    ; 
    auto& d_corner_info = hanging_corners.d_view    ; 
    #ifdef GRACE_3D 
    auto& d_edge_info = hanging_edges.d_view    ; 
    #endif 

    constexpr const int n_neighbors = PICK_D(2,4) ;  

    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>, IndexType<int>> 
    policy(
        {0,VECD(0,0), 0,0},
        { ngz/2
        , VECD(static_cast<int>(nx/2+ngz),static_cast<int>(ny/2+ngz))
        , static_cast<int>(nvars)
        , static_cast<int>(n_faces) }
    ) ;
    
    /*************************************************/
    /* Kernel:                                       */
    /* Prolongate data onto fine quadrants ghost     */
    /* zones from coarse neighboring quadrants.      */
    /*************************************************/
    parallel_for
    (             GRACE_EXECUTION_TAG("AMR","prolongate_hanging_faces_corner_vars")
                , policy 
                , KOKKOS_LAMBDA( const size_t& ig, VECD(const size_t& j, const size_t& k), 
                                 const size_t& ivar, const size_t& iface )
        {
            int polarity     =  d_face_info(iface).has_polarity_flip         ; 
            int8_t is_ghost_coarse   =  d_face_info(iface).is_ghost_coarse   ; 
            int64_t iq_coarse        =  d_face_info(iface).qid_coarse        ;
            int8_t which_face_coarse =  d_face_info(iface).which_face_coarse    ; 
            int8_t which_face_fine   =  d_face_info(iface).which_face_fine      ; 
            int tid_coarse           =  d_face_info(iface).which_tree_coarse    ; 
            int tid_fine             =  d_face_info(iface).which_tree_fine      ; 
            for ( int ichild =0 ; ichild < P4EST_HALF; ++ichild )
            {
                /***********************************************/
                /* Decide loop offset in directions orthogonal */
                /* to ghostzones being filled                  */
                /***********************************************/
                int8_t const cx = ((ichild >> 0U) & 1U);
                int8_t const cy = ((ichild >> 1U) & 1U);

                int const off_x = cx ? (nx/2+ngz/2) : ngz/2 ;
                int const off_y = cy ? (ny/2+ngz/2) : ngz/2 ; 

                int const foff_x = 0 ; 
                int const foff_y = 0 ; 
                 
                int64_t qid_b     = d_face_info(iface).qid_fine[ichild] ; 
                bool is_ghost_fine = d_face_info(iface).is_ghost_fine[ichild] ; 
                if ( !is_ghost_fine ) {
                    auto& cview = is_ghost_coarse ? halo : state ; 
                    /* First we compute the indices of the point */
                    /* we are calculating.                       */
                    EXPR( 
                    int const i_c = EXPR((which_face_coarse==0) * (ngz+ig) 
                                + (which_face_coarse==1) * (nx+ig),
                                + (which_face_coarse/2==1) * (off_x + j), 
                                + (which_face_coarse/2==2) * (off_x + j)) ;,
                    int const j_c = EXPR((which_face_coarse==2) * (ngz+ig)
                                + (which_face_coarse==3) * (ny+ig), 
                                + (which_face_coarse/2==0) * (off_x + j), 
                                + (which_face_coarse/2==2) * (off_y + k));  ,
                    int const k_c  = (which_face_coarse==4) * (ngz+ig) 
                                + (which_face_coarse==5) * (nz+ig)  
                                + (which_face_coarse/2!=2) * (off_y + k) ;
                    )
                    EXPR(
                    int const i_f =  EXPR((which_face_fine==0) * (2*ig) 
                                + (which_face_fine==1) * (nx+ngz+2*ig),
                                + (which_face_fine/2==1) * ((2*j) + foff_x), 
                                + (which_face_fine/2==2) * ((2*j) + foff_x)) ;,
                    int const j_f =  EXPR((which_face_fine==2) * (2*ig) 
                                + (which_face_fine==3) * (ny+ngz+2*ig),
                                + (which_face_fine/2==0) * ((2*j) + foff_x), 
                                + (which_face_fine/2==2) * ((2*k) + foff_y)) ;,
                    int const k_f =  (which_face_fine==4) * (2*ig) 
                                + (which_face_fine==5) * (nx+ngz+2*ig)
                                + (which_face_fine/2!=2) * ((2*k) + foff_y) ;
                    )
                    auto fine_view = subview(state, VEC(ALL(),ALL(),ALL()), ivar, qid_b) ; 
                    auto coarse_view = subview(cview, VEC(ALL(),ALL(),ALL()), ivar, iq_coarse) ;
                    /* Fill the fine state */
                    InterpT::interpolate(VEC(i_f,j_f,k_f), VEC(i_c,j_c,k_c), coarse_view, fine_view) ; 
                }
            }
            
        }
    )   ; /* end of loop over faces */
    #if 1 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>, IndexType<int>> 
        corner_policy(
            {0,VECD(0,0), 0,0},
            { VEC(math::floor_int(ngz/2),math::floor_int(ngz/2),math::floor_int(ngz/2))
            , static_cast<int>(nvars), static_cast<int>(n_corners)}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("AMR", "prolongate_hanging_corners")
                , corner_policy 
                , KOKKOS_LAMBDA(VEC(int const ig, int const jg, int const kg), int const ivar, int const icorner)
        {
            /* Collect the necessary information                                */
            int8_t is_ghost_coarse   =  d_corner_info(icorner).is_ghost_coarse       ; 
            int64_t iq_coarse        =  d_corner_info(icorner).qid_coarse            ;
            int8_t which_corner_coarse =  d_corner_info(icorner).which_corner_coarse ; 
            int8_t which_corner_fine   =  d_corner_info(icorner).which_corner_fine   ; 
            int8_t is_ghost_fine     =  d_corner_info(icorner).is_ghost_fine         ;  
            int64_t iq_fine          =  d_corner_info(icorner).qid_fine              ;
            /* Get the correct view to index into for coarse data               */
            auto& cview   = is_ghost_coarse ? halo : state ; 
            if( ! is_ghost_fine ) {
                /* Utility to map the coarse index into the fine quadrant */
                auto const index_mapping = [=] (
                    VEC(int const ig, int const jg, int const kg),
                    int const cf, int const cc, int ijk_f[GRACE_NSPACEDIM], int ijk_c[GRACE_NSPACEDIM]
                )
                {
                    int x = (cc >> 0) & 1;  
                    int y = (cc >> 1) & 1;  
                    int z = (cc >> 2) & 1;
                    EXPR(
                    ijk_c[0] = (x==0) ? (ngz+ig) : (nx + ig) ;, 
                    ijk_c[1] = (y==0) ? (ngz+jg) : (ny + jg) ;,
                    ijk_c[2] = (z==0) ? (ngz+kg) : (nz + kg) ;
                    )
                    x = (cf >> 0) & 1;  
                    y = (cf >> 1) & 1;  
                    z = (cf >> 2) & 1;
                    EXPR(
                    ijk_f[0] = (x==0) ? 2*ig
                                    : (nx+ngz+2*ig) ;, 
                    ijk_f[1] = (y==0) ? 2*jg 
                                    : (ny+ngz+2*jg) ;,
                    ijk_f[2] = (z==0) ? 2*kg 
                                    : (nz+ngz+2*kg) ;
                    )
                } ;
                int ijk_c[GRACE_NSPACEDIM], ijk_f[GRACE_NSPACEDIM] ; 
                index_mapping(VEC(ig,jg,kg), which_corner_fine, which_corner_coarse, ijk_f, ijk_c) ; 
                auto fine_view = subview(state, VEC(ALL(),ALL(),ALL()), ivar, iq_fine) ; 
                auto coarse_view = subview(cview, VEC(ALL(),ALL(),ALL()), ivar, iq_coarse) ;
                /* Fill the fine state */
                InterpT::interpolate(VEC(ijk_f[0],ijk_f[1],ijk_f[2]), VEC(ijk_c[0],ijk_c[1], ijk_c[2]), coarse_view, fine_view) ; 
            }
        }
    ); /* End of loop over corner neighbors */

    /* Loop over edge neighbors             */
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        edge_policy(
            {0,0,0, 0,0},
            {ngz/2, ngz/2, static_cast<int>(nx/2+ngz), static_cast<long>(nvars), static_cast<long>(n_edges)}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("AMR", "prolongate_hanging_edges")
                , edge_policy 
                , KOKKOS_LAMBDA(const size_t& ig, const size_t& jg, const size_t& k, const size_t& ivar, const size_t& iedge)
        {
            /* Collect the necessary information                                */
            int8_t is_ghost_coarse   =  d_edge_info(iedge).is_ghost_coarse       ; 
            int64_t iq_coarse        =  d_edge_info(iedge).qid_coarse            ;
            int8_t which_edge_coarse =  d_edge_info(iedge).which_edge_coarse     ; 
            int8_t which_edge_fine   =  d_edge_info(iedge).which_edge_fine       ; 
            int tid_coarse           =  d_edge_info(iedge).which_tree_coarse     ; 
            int tid_fine             =  d_edge_info(iedge).which_tree_fine       ;
             
            for(int ichild=0; ichild<2; ++ichild){

                int const off  = ichild ? (nx/2+ngz/2) : ngz/2 ; 
                int const foff = 0 ; 

                int8_t is_ghost_fine     =  d_edge_info(iedge).is_ghost_fine[ichild] ;
                int64_t qid_fine         =  d_edge_info(iedge).qid_fine[ichild]      ;

                auto& cview = is_ghost_coarse ? halo : state     ;
                if( !is_ghost_fine ){
                    auto const fine_index_mapping = [=] ( int const ig, int const jg, int const k, 
                                            int const ec, int ijk_c[GRACE_NSPACEDIM],
                                            int const ef, int ijk_f[GRACE_NSPACEDIM]) 
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
                        int x_ea = edge_directions[0][ec];
                        int y_ea = edge_directions[1][ec];
                        int z_ea = edge_directions[2][ec];

                        // Map indices for ijk based on edge ea
                        ijk_c[0] = (x_ea == ALONG_EDGE) ? (k+off) : (x_ea == NEGATIVE_EDGE ? (ngz+ig) : (nx + ig));
                        ijk_c[1] = (y_ea == ALONG_EDGE) 
                                    ? (k+off) 
                                    : (x_ea == ALONG_EDGE 
                                        ? (y_ea == NEGATIVE_EDGE ? (ngz+ig) : (ny + ig)) 
                                        : (y_ea == NEGATIVE_EDGE ? (ngz+jg) : (ny + jg)));
                        ijk_c[2] = (z_ea == ALONG_EDGE) ? (k+off) : (z_ea == NEGATIVE_EDGE ? (ngz+jg) : (nz + jg));

                        int x_eb = edge_directions[0][ef];
                        int y_eb = edge_directions[1][ef];
                        int z_eb = edge_directions[2][ef];

                        // Map indices for ijk based on edge ea
                        ijk_f[0] = (x_eb == ALONG_EDGE) ? ((2*k)+foff) : (x_eb == NEGATIVE_EDGE ? (2*ig)  : (nx+ngz+2*ig));
                        ijk_f[1] = (y_eb == ALONG_EDGE) 
                                    ? ((2*k)+foff) 
                                    : (x_eb == ALONG_EDGE 
                                        ? (y_eb == NEGATIVE_EDGE ? (2*ig) : (ny+ngz+2*ig)) 
                                        : (y_eb == NEGATIVE_EDGE ? (2*jg) : (ny+ngz+2*jg)));
                        ijk_f[2] = (z_eb == ALONG_EDGE) ? ((2*k)+foff) : (z_eb == NEGATIVE_EDGE ? (2*jg) : (nz+ngz+2*jg));
                    } ; 

                    /* Find fine cell index                                               */
                    int ijk_f[GRACE_NSPACEDIM], ijk_c[GRACE_NSPACEDIM] ;
                    fine_index_mapping( ig,jg,k, which_edge_coarse, ijk_c, which_edge_fine, ijk_f ) ; 
                    auto fine_view = subview(state, VEC(ALL(),ALL(),ALL()), ivar, qid_fine) ; 
                    auto coarse_view = subview(cview, VEC(ALL(),ALL(),ALL()), ivar, iq_coarse) ;
                    /* Fill the fine state */
                    InterpT::interpolate(VEC(ijk_f[0],ijk_f[1],ijk_f[2]), VEC(ijk_c[0],ijk_c[1], ijk_c[2]), coarse_view, fine_view) ; 
                }
            }
        }
    ) ; /* End of loop over edge neighbors */
    #endif 

    /******************************************************/
    /* Now we need to take care of corners and edges that */
    /* are not part of the "grid" according to p4est.     */
    /* That is, corners/edges that are not shared by all  */
    /* quadrants touching them.                           */
    /******************************************************/
    
}
/**************************************************************************************************/
/*                                  Instantiate templates                                         */
/**************************************************************************************************/
template void 
prolongate_hanging_ghostzones_cell_centers<utils::linear_prolongator_t<grace::minmod>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&  
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , grace::device_vector<hanging_face_info_t>& 
    , grace::device_vector<hanging_corner_info_t>&
    #ifdef GRACE_3D 
    , grace::device_vector<hanging_edge_info_t>&
    #endif
) ; 
/**************************************************************************************************/
template void 
prolongate_hanging_ghostzones_cell_centers<utils::linear_prolongator_t<grace::MCbeta>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&   
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , grace::device_vector<hanging_face_info_t>& 
    , grace::device_vector<hanging_corner_info_t>&
    #ifdef GRACE_3D 
    , grace::device_vector<hanging_edge_info_t>&
    #endif
) ; 
/**************************************************************************************************/
template void 
prolongate_hanging_ghostzones_corners<utils::lagrange_prolongator_t<2>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&   
    , grace::device_vector<hanging_face_info_t>& 
    , grace::device_vector<hanging_corner_info_t>&
    #ifdef GRACE_3D 
    , grace::device_vector<hanging_edge_info_t>&
    #endif
) ;
/**************************************************************************************************/ 
template void 
prolongate_hanging_ghostzones_corners<utils::lagrange_prolongator_t<4>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&   
    , grace::device_vector<hanging_face_info_t>& 
    , grace::device_vector<hanging_corner_info_t>&
    #ifdef GRACE_3D 
    , grace::device_vector<hanging_edge_info_t>&
    #endif
) ; 
/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
}} /* namespace grace::amr */
