/**
 * @file bc_helpers.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Index fiesta.
 * @date 2024-03-21
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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

#include <thunder/amr/bc_helpers.hh>
#include <thunder/amr/prolongation_kernels.tpp> 
#include <thunder/amr/restriction_kernels.tpp> 
#include <thunder/utils/prolongation.hh>
#include <thunder/utils/limiters.hh> 
#include <thunder/amr/boundary_conditions.hh>
#include <thunder/amr/p4est_headers.hh>
#include <thunder/amr/bc_helpers.tpp>
#include <thunder/amr/thunder_amr.hh> 
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/utils/interpolators.hh>
#include <thunder/data_structures/thunder_data_structures.hh>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>


namespace thunder { namespace amr {

void thunder_iterate_faces( p4est_iter_face_info_t * info 
                          , void* user_data  )
{
    using namespace thunder; 
    auto face_info = reinterpret_cast<thunder_face_info_t*>(user_data) ; 
    sc_array_view_t<p4est_iter_face_side_t> sides{
        &(info->sides)
    } ; 
    auto& physical_boundary_info = face_info->phys_boundary_info ; 
    auto& simple_info    = face_info->simple_interior_info       ;
    auto& hanging_info   = face_info->hanging_interior_info      ;
    /**************************************************/
    /* This means we are at a physical boundary       */
    /* we store the index in user_info and return     */
    /* since physical boundary conditions are handled */
    /* separately.                                    */
    /**************************************************/
    if( sides.size() == 1 ) { 
        size_t offset = amr::get_local_quadrants_offset(sides[0].treeid);
        if( sides[0].is_hanging ) {
            physical_boundary_info.push_back(
                (offset+sides[0].is.hanging.quadid[0]) * P4EST_FACES + sides[0].face
            ); 
            physical_boundary_info.push_back(
                (offset+sides[0].is.hanging.quadid[1]) * P4EST_FACES + sides[0].face
            ); 
        } else {
            physical_boundary_info.push_back(
                (offset+sides[0].is.full.quadid) * P4EST_FACES + sides[0].face
            ); 
        }
        return ; 
    }
    /***************************************************/
    /* Now we are left with two possibilities:         */
    /* 1) The face crosses process boundaries          */
    /* 2) Both sides of the face are local             */
    /* In both cases we need to check whether we cross */
    /* a tree boundary.                                */
    /***************************************************/
    int8_t const orientation = info->orientation ; 
    ASSERT( orientation == 0
          , "Twisted grid topologies not yet implemented"
            " in ghost exchange." ) ; 
    /* if we cross a tree boundary, check polarity */
    int polarity_flip{0}  ;
    if( info->tree_boundary )
    {
        polarity_flip = 
            amr::trees_have_opposite_polarity(
                sides[0].treeid,
                sides[0].face
            ) ; 
    }
    int   side1_hanging{sides[0].is_hanging}
        , side2_hanging{sides[1].is_hanging}  ; 

    if( side2_hanging )
    {
        hanging_face_info_t this_face_info{} ; 
        this_face_info.has_polarity_flip = polarity_flip    ;
        this_face_info.which_tree_a = sides[0].treeid ; 
        this_face_info.which_tree_b = sides[1].treeid ; 
        this_face_info.which_face_a = sides[0].face          ;
        this_face_info.is_ghost_a   = sides[0].is.full.is_ghost ;
        this_face_info.qid_a        = sides[0].is.full.quadid ;
        if( !this_face_info.is_ghost_a ){
            this_face_info.qid_a += get_local_quadrants_offset(sides[0].treeid); 
        }
        this_face_info.level_a      = static_cast<int>(sides[0].is.full.quad->level) ;
        this_face_info.which_face_b = sides[1].face          ;
        this_face_info.is_ghost_b   = sides[1].is.hanging.is_ghost[0]   ; 
        this_face_info.level_b      = static_cast<int>(sides[1].is.hanging.quad[0]->level) ;
        this_face_info.is_ghost_c   = sides[1].is.hanging.is_ghost[1]   ; 
        this_face_info.qid_b        = sides[1].is.hanging.quadid[0]     ; 
        if( !this_face_info.is_ghost_b ){
            this_face_info.qid_b += get_local_quadrants_offset(sides[1].treeid); 
        }
        this_face_info.qid_c        = sides[1].is.hanging.quadid[1]  ;
        if( !this_face_info.is_ghost_c ){
            this_face_info.qid_c += get_local_quadrants_offset(sides[1].treeid); 
        }
        #ifdef THUNDER_3D
        this_face_info.is_ghost_d   = sides[1].is.hanging.is_ghost[2]   ; 
        this_face_info.is_ghost_e   = sides[1].is.hanging.is_ghost[3]   ; 
        this_face_info.qid_d        = sides[1].is.hanging.quadid[2]     ;  
        if( !this_face_info.is_ghost_d ){
            this_face_info.qid_d += get_local_quadrants_offset(sides[1].treeid); 
        }
        this_face_info.qid_e        = sides[1].is.hanging.quadid[3]     ;
        if( !this_face_info.is_ghost_e ){
            this_face_info.qid_d += get_local_quadrants_offset(sides[1].treeid); 
        }
        #endif 
        hanging_info.push_back(this_face_info) ; 
    } else if(side1_hanging) {
        hanging_face_info_t this_face_info{} ; 
        this_face_info.which_tree_a = sides[1].treeid ; 
        this_face_info.which_tree_b = sides[0].treeid ; 
        this_face_info.has_polarity_flip = polarity_flip    ;
        this_face_info.which_face_a = sides[1].face          ;
        this_face_info.is_ghost_a   = sides[1].is.full.is_ghost ;
        this_face_info.level_a      = static_cast<int>(sides[1].is.full.quad->level) ;
        this_face_info.qid_a        = sides[1].is.full.quadid  ;  
        if( !this_face_info.is_ghost_a ){
            this_face_info.qid_a += get_local_quadrants_offset(sides[1].treeid); 
        } 
        this_face_info.which_face_b = sides[0].face          ;
        this_face_info.is_ghost_b   = sides[0].is.hanging.is_ghost[0]   ;
        this_face_info.level_b      = static_cast<int>(sides[0].is.hanging.quad[0]->level) ;  
        this_face_info.is_ghost_c   = sides[0].is.hanging.is_ghost[1]   ; 
        this_face_info.qid_b        = sides[0].is.hanging.quadid[0] ; 
        if( !this_face_info.is_ghost_b ){
            this_face_info.qid_b += get_local_quadrants_offset(sides[0].treeid); 
        }
        this_face_info.qid_c        = sides[0].is.hanging.quadid[1] ; 
        if( !this_face_info.is_ghost_c ){
            this_face_info.qid_c += get_local_quadrants_offset(sides[0].treeid); 
        }
        #ifdef THUNDER_3D
        this_face_info.is_ghost_d   = sides[0].is.hanging.is_ghost[2]   ; 
        this_face_info.is_ghost_e   = sides[0].is.hanging.is_ghost[3]   ; 
        this_face_info.qid_d        = sides[0].is.hanging.quadid[2]  ;
        if( !this_face_info.is_ghost_d ){
            this_face_info.qid_d += get_local_quadrants_offset(sides[0].treeid); 
        }
        this_face_info.qid_e        = sides[0].is.hanging.quadid[3];  
        if( !this_face_info.is_ghost_e ){
            this_face_info.qid_e += get_local_quadrants_offset(sides[0].treeid); 
        }
        #endif 
        hanging_info.push_back(this_face_info) ;
    } else {
        simple_face_info_t this_face_info {} ;
        this_face_info.has_polarity_flip = polarity_flip    ;
        if( sides[0].is.full.is_ghost) {
            auto offset = get_local_quadrants_offset(sides[1].treeid);
            this_face_info.is_ghost = 1 ; 
            this_face_info.which_face_a = sides[1].face ; 
            this_face_info.which_face_b = sides[0].face ;
            this_face_info.which_tree_a = sides[1].treeid ; 
            this_face_info.which_tree_b = sides[0].treeid ; 
            this_face_info.qid_a = sides[1].is.full.quadid 
                + offset ;
            this_face_info.qid_b = sides[0].is.full.quadid ;
        } else if(sides[1].is.full.is_ghost){
            auto offset = get_local_quadrants_offset(sides[0].treeid);
            this_face_info.is_ghost = 1 ; 
            this_face_info.which_face_a = sides[0].face ; 
            this_face_info.which_face_b = sides[1].face ;
            this_face_info.which_tree_a = sides[0].treeid ; 
            this_face_info.which_tree_b = sides[1].treeid ; 
            this_face_info.qid_a = sides[0].is.full.quadid 
                + offset ;
            this_face_info.qid_b = sides[1].is.full.quadid ; 
        } else {
            auto offset = get_local_quadrants_offset(sides[0].treeid);
            this_face_info.is_ghost = 0 ; 
            this_face_info.which_face_a = sides[0].face ; 
            this_face_info.which_face_b = sides[1].face ;
            this_face_info.which_tree_a = sides[0].treeid ; 
            this_face_info.which_tree_b = sides[1].treeid ; 
            this_face_info.qid_a = sides[0].is.full.quadid 
            + offset ;
            offset = get_local_quadrants_offset(sides[1].treeid);
            this_face_info.qid_b = sides[1].is.full.quadid
            + offset ;
        }
        simple_info.push_back(this_face_info) ; 
    }

}


void copy_interior_ghostzones(
      thunder::var_array_t<THUNDER_NSPACEDIM>& vars
    , thunder::var_array_t<THUNDER_NSPACEDIM>& halo 
    , Kokkos::vector<simple_face_info_t>& interior_faces
)
{
    using namespace thunder; 
    using namespace Kokkos ; 

    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int64_t ngz = amr::get_n_ghosts() ;
    int64_t nq  = amr::get_local_num_quadrants() ;
    int nvars  = variables::get_n_evolved()      ;
    auto const n_faces = interior_faces.size()   ;
    auto& d_face_info = interior_faces.d_view    ; 

    TeamPolicy<default_execution_space> 
        policy( n_faces, AUTO() ) ; 
    using member_t = decltype(policy)::member_type ;

    parallel_for( THUNDER_EXECUTION_TAG("AMR","copy_interior_ghostzones")
                , policy 
                , KOKKOS_LAMBDA( const member_t& team )
        {
            /* Get information about quadrants sharing the face */
            int polarity     =  d_face_info(team.league_rank()).has_polarity_flip ;
            int is_ghost     =  d_face_info(team.league_rank()).is_ghost          ; 
            int which_face_a =  d_face_info(team.league_rank()).which_face_a      ; 
            int which_face_b =  d_face_info(team.league_rank()).which_face_b      ; 
            int64_t qid_a    =  d_face_info(team.league_rank()).qid_a             ;
            int64_t qid_b    =  d_face_info(team.league_rank()).qid_b             ; 
            /* Get extents in direction(s) orthogonal to the face */
            int64_t n1 = (which_face_a/2==0) * ny 
                       + ((which_face_a/2==1) * nx) 
                       + ((which_face_a/2==2) * nx) ;
            int64_t n2 = (which_face_a/2==0) * nz   
                       + ((which_face_a/2==1) * nz) 
                       + ((which_face_a/2==2) * ny) ;
            /* Get correct array to read from / write to */
            auto& view_a = vars ; 
            auto& view_b = (is_ghost) ? halo : vars ;  

            TeamThreadMDRange<Rank<THUNDER_NSPACEDIM>,member_t>
                team_range( team, VECD(n1,n2), nvars) ; 
            parallel_for( team_range
                        , KOKKOS_LAMBDA(VECD(int& j, int& k), int& ivar)
                    {
                    
                    for( int ig=0; ig<ngz; ++ig){
                        int i_a = EXPR((which_face_a==0) * 
                                    ( (!polarity)*ig 
                                    + (polarity)*(ngz-1-ig) ) 
                                    + (which_face_a==1) * 
                                    ( (!polarity)*(nx+ngz+ig) 
                                    + (polarity)*(nx+2*ngz-1-ig) ),
                                    + (which_face_a/2==1) * (j+ngz), 
                                    + (which_face_a/2==2) * (j+ngz)) ;

                        int j_a = EXPR((which_face_a==2) * 
                                ( (!polarity)*ig 
                                + (polarity)*(ngz-1-ig) )
                                + (which_face_a==3) * 
                                ( (!polarity)*(ny+ngz+ig)  
                                + (polarity)*(ny+2*ngz-1-ig) ), 
                                + (which_face_a/2==0) * (j+ngz), 
                                + (which_face_a/2==2) * (k+ngz));  
                        
                        int i_b = EXPR((which_face_b==0)*(ngz+ig) 
                                + (which_face_b==1)*(nx+ig), 
                                + (which_face_b/2==1) * (j+ngz),
                                + (which_face_b/2==2) * (j+ngz)) ;
                        
                        int j_b = EXPR((which_face_b==2)*(ngz+ig) 
                                + (which_face_b==3)*(ny+ig),
                                + (which_face_b/2==0) * (j+ngz),
                                + (which_face_b/2==2) * (k+ngz)) ; 

                        #ifdef THUNDER_3D
                        int k_a = (which_face_a==4) * 
                                ( (!polarity)*ig 
                                + (polarity)*(ngz-1-ig) )
                                + (which_face_a==5) * 
                                ( (!polarity)*(nz+ngz+ig)  
                                + (polarity)*(nz+2*ngz-1-ig) )
                                + (which_face_a/2!=2) * (k+ngz) ;

                        int k_b = (which_face_b==4)*(ngz+ig)
                                + (which_face_b==5)*(nz+ig)
                                + (which_face_b/2!=2) * (k + ngz) ;
                        #endif 

                        view_a(VEC(i_a,j_a,k_a),ivar,qid_a) =  view_b(VEC(i_b,j_b,k_b),ivar,qid_b) ; 

                        i_b = EXPR((which_face_b==0) * 
                                ( (!polarity)*ig 
                                + (polarity)*(ngz-1-ig) )
                                + (which_face_b==1) * 
                                ( (!polarity)*(nx+ngz+ig)  
                                + (polarity)*(nx+2*ngz-1-ig) ),
                                + (which_face_b/2==1) * (j+ngz), 
                                + (which_face_b/2==2) * (j+ngz)) ;

                        j_b = EXPR((which_face_b==2) * 
                                ( (!polarity)*ig 
                                + (polarity)*(ngz-1-ig) )
                                + (which_face_b==3) * 
                                ( (!polarity)*(ny+ngz+ig)  
                                + (polarity)*(ny+2*ngz-1-ig) ), 
                                + (which_face_b/2==0) * (j+ngz), 
                                + (which_face_b/2==2) * (k+ngz)) ;  
                        
                        i_a = EXPR((which_face_a==0)*(ngz+ig) 
                                + (which_face_a==1)*(nx+ig), 
                                + (which_face_a/2==1) * (j+ngz),
                                + (which_face_a/2==2) * (j+ngz)) ;
                        
                        j_a = EXPR((which_face_a==2)*(ngz+ig) 
                                + (which_face_a==3)*(ny+ig),
                                + (which_face_a/2==0) * (j+ngz),
                                + (which_face_a/2==2) * (k+ngz)) ; 

                        #ifdef THUNDER_3D
                        k_b =     (which_face_b==4) * 
                                ( (!polarity)*ig 
                                + (polarity)*(ngz-1-ig) )
                                + (which_face_b==5) * 
                                ( (!polarity)*(nz+ngz+ig)  
                                + (polarity)*(nz+2*ngz-1-ig) )
                                + (which_face_b/2!=2) * (k+ngz) ;

                        k_a =     (which_face_a==4)*(ngz+ig)
                                + (which_face_a==5)*(nz+ig)
                                + (which_face_a/2!=2) * (k + ngz) ;
                        #endif  
                        
                        view_b(VEC(i_b,j_b,k_b),ivar,qid_b) =  view_a(VEC(i_a,j_a,k_a),ivar,qid_a) ;
                    }
                    } );

        }
    )   ;  
}

template< typename InterpT > 
void interp_hanging_ghostzones(
      thunder::var_array_t<THUNDER_NSPACEDIM>& state
    , thunder::var_array_t<THUNDER_NSPACEDIM>& halo 
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>& coords 
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>& halo_coords 
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>& vols 
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>& halo_vols 
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
) 
{
    using namespace thunder ;
    using namespace Kokkos  ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ;
    int64_t nq  = amr::get_local_num_quadrants() ;
    int const nvars = variables::get_n_evolved() ;
    auto const n_faces = hanging_faces.size()   ;
    auto& d_face_info = hanging_faces.d_view    ; 

    constexpr const int n_neighbors = PICK_D(2,4) ;  

    TeamPolicy<default_execution_space> 
        policy( n_faces, AUTO() ) ; 
    using member_t = decltype(policy)::member_type ;

    ghostzone_restrictor_t<decltype(state),decltype(vols)> restriction_kernel{
        ngz, state, halo, vols, halo_vols 
    } ; 

    /*************************************************/
    /* Kernel 1:                                     */
    /* Restrict data onto coarse quadrants from fine */
    /* neighboring quadrants.                        */
    /*************************************************/
    parallel_for
    (             THUNDER_EXECUTION_TAG("AMR","restrict_hanging_faces")
                , policy 
                , KOKKOS_LAMBDA( const member_t& team )
        {
            int polarity     =  d_face_info(team.league_rank()).has_polarity_flip ; 
            int is_ghost_a   =  d_face_info(team.league_rank()).is_ghost_a   ; 
            int is_ghost_b   =  d_face_info(team.league_rank()).is_ghost_b   ; 
            int is_ghost_c   =  d_face_info(team.league_rank()).is_ghost_a   ; 
            int which_face_a =  d_face_info(team.league_rank()).which_face_a ; 
            int which_face_b =  d_face_info(team.league_rank()).which_face_b ; 
            int64_t qid_a    =  d_face_info(team.league_rank()).qid_a        ;
            int64_t qid_b    =  d_face_info(team.league_rank()).qid_b        ;
            int64_t qid_c    =  d_face_info(team.league_rank()).qid_c        ; 
            #ifdef THUNDER_3D 
            int is_ghost_d   =  d_face_info(team.league_rank()).is_ghost_d   ; 
            int is_ghost_e   =  d_face_info(team.league_rank()).is_ghost_e   ;
            int64_t qid_d    =  d_face_info(team.league_rank()).qid_d        ;
            int64_t qid_e    =  d_face_info(team.league_rank()).qid_e        ;
            #endif 
            
            int64_t ng = (which_face_b/2==0) * nx + ((which_face_b/2==1) * ny) + ((which_face_b/2==2) * nz) ;
            int64_t n1 = (which_face_b/2==0) * ny + ((which_face_b/2==1) * nx) + ((which_face_b/2==2) * nx) ;
            int64_t n2 = (which_face_b/2==0) * nz + ((which_face_b/2==1) * nz) + ((which_face_b/2==2) * ny) ;
            auto& view_a = is_ghost_a ? halo : state ; 
            int fine_view_is_ghost[] = {
                    (is_ghost_b) ,
                    (is_ghost_c) ,
                    #ifdef THUNDER_3D 
                    (is_ghost_d) ,
                    (is_ghost_e) 
                    #endif
            } ; 

            int64_t fine_iqs[THUNDER_FACE_CHILDREN]  =
                {
                    qid_b, qid_c
                    #ifdef THUNDER_3D 
                    ,qid_d, qid_e
                    #endif
                } ; 

            TeamThreadMDRange<Rank<THUNDER_NSPACEDIM+1>,member_t>
                team_range( team, VEC(ngz, n1,n2), nvars) ; 
            parallel_for( team_range
                        , KOKKOS_LAMBDA(VEC(int& ig, int& j, int& k), int& ivar)
                    { 
                        /* Compute indices of cell to be filled */
                        EXPR( 
                        int const i0 = 
                            (which_face_a == 0) * (ig)
                        +   (which_face_a == 1) * (nx + ngz + ig)
                        +   (which_face_a/2!=0) * (j+ngz); ,
                        int const j0 = EXPR(
                            (which_face_a == 2) * (ig)
                        +   (which_face_a == 3) * (ny + ngz + ig),
                        +   (which_face_a/2==0) * (j+ngz), 
                        +   (which_face_a/2==2) * (k+ngz)); ,
                        int const k0 = 
                            (which_face_a == 4) * (ig)
                        +   (which_face_a == 5) * (nz + ngz + ig)
                        +   (which_face_a/2!=2) * (k+ngz) ; 
                        )  
                        /* Call restriction operator on fine data */ 
                        view_a(VEC(i0,j0,k0),ivar,qid_a) = 
                            restriction_kernel(
                                VEC(ig,j,k)
                            ,   VEC(ng,n1,n2)
                            ,   fine_iqs
                            ,   ivar
                            ,   which_face_b
                            ,   polarity
                            ,   fine_view_is_ghost 
                            ) ; 
                    } );

        }
    )   ;
    ghostzone_prolongator_t<InterpT, decltype(state), decltype(coords), decltype(vols)>
        prolongation_kernel{
              VEC(nx,ny,nz), ngz 
            , state, halo 
            , vols, halo_vols 
            , coords, halo_coords
        } ; 
    /*************************************************/
    /* Kernel 2:                                     */
    /* Prolongate data onto fine quadrants ghost     */
    /* zones from coarse neighboring quadrants.      */
    /*************************************************/
    parallel_for
    (             THUNDER_EXECUTION_TAG("AMR","prolongate_hanging_faces")
                , policy 
                , KOKKOS_LAMBDA( const member_t& team )
        {
            int polarity     =  d_face_info(team.league_rank()).has_polarity_flip ; 
            int is_ghost_a   =  d_face_info(team.league_rank()).is_ghost_a   ; 
            int is_ghost_b   =  d_face_info(team.league_rank()).is_ghost_b   ; 
            int is_ghost_c   =  d_face_info(team.league_rank()).is_ghost_a   ; 
            int which_face_coarse =  d_face_info(team.league_rank()).which_face_a ; 
            int which_face_fine   =  d_face_info(team.league_rank()).which_face_b ; 
            int64_t qid_a    =  d_face_info(team.league_rank()).qid_a        ;
            int64_t qid_b    =  d_face_info(team.league_rank()).qid_b        ;
            int64_t qid_c    =  d_face_info(team.league_rank()).qid_c        ; 
            #ifdef THUNDER_3D 
            int is_ghost_d   =  d_face_info(team.league_rank()).is_ghost_d   ; 
            int is_ghost_e   =  d_face_info(team.league_rank()).is_ghost_e   ;
            int64_t qid_d    =  d_face_info(team.league_rank()).qid_d        ;
            int64_t qid_e    =  d_face_info(team.league_rank()).qid_e        ;
            #endif 

            int fine_view_is_ghost[] = {
                    (is_ghost_b) ,
                    (is_ghost_c) ,
                    #ifdef THUNDER_3D 
                    (is_ghost_d) ,
                    (is_ghost_e) 
                    #endif
            } ; 

            int64_t fine_iqs[]  =
                {
                    qid_b, qid_c
                    #ifdef THUNDER_3D 
                    ,qid_d, qid_e
                    #endif
                } ; 
            /* We store the quadrant level for efficiency and to avoid passing         */
            /* around two different cell spacing arrays.                               */
            EXPR(
            double const dx_fine =  1./(1L<<d_face_info(team.league_rank()).level_b)/nx;,
            double const dy_fine =  1./(1L<<d_face_info(team.league_rank()).level_b)/ny;,
            double const dz_fine =  1./(1L<<d_face_info(team.league_rank()).level_b)/nz; ) 
            
            int64_t n1 = (which_face_fine/2==0) * ny + ((which_face_fine/2==1) * nx) + ((which_face_fine/2==2) * nx) ;
            int64_t n2 = (which_face_fine/2==0) * nz + ((which_face_fine/2==1) * nz) + ((which_face_fine/2==2) * ny) ;

            TeamThreadMDRange<Rank<THUNDER_NSPACEDIM+1>,member_t>
                team_range( team, VEC(ngz, n1,n2), nvars) ; 
            parallel_for( team_range
                        , KOKKOS_LAMBDA(VEC(int& ig, int& j, int& k), int& ivar)
                    { 
                        /* First we compute the indices of the point */
                        /* we are calculating.                       */
                        EXPR(
                        int const i_f = 
                                (which_face_fine==0) * ig 
                          +     (which_face_fine==1) * (nx + ngz + ig)
                          +     (which_face_fine/2!=0) * (j + ngz) ;,  
                        int const j_f = EXPR(
                                (which_face_fine==2) * ig 
                          +     (which_face_fine==3) * (ny + ngz + ig),
                          +     (which_face_fine/2==0) * (j + ngz), 
                          +     (which_face_fine/2==2) * (k + ngz) );, 
                        int const k_f = 
                                (which_face_fine==4) * ig 
                          +     (which_face_fine==5) * (nz + ngz + ig)
                          +     (which_face_fine/2!=2) * (k + ngz) ; 
                        )
                        /* Then we loop over all child quadrants */
                        /* and call the prolongation kernel.     */
                        #pragma unroll 4
                        for( int ichild=0; ichild<n_neighbors; ++ichild) {
                            int64_t iq_fine = fine_iqs[ichild] ; 
                            auto& fine_view = fine_view_is_ghost[ichild] 
                                        ? halo 
                                        : state ; 
                            fine_view(VEC(i_f,j_f,k_f), ivar, iq_fine)
                                = prolongation_kernel( VEC(ig,j,k),VEC(i_f,j_f,k_f)
                                                     , VEC(dx_fine,dy_fine,dz_fine)
                                                     , qid_a, iq_fine
                                                     , ivar, ichild 
                                                     , polarity
                                                     , which_face_coarse 
                                                     , which_face_fine 
                                                     , is_ghost_a 
                                                     , fine_view_is_ghost[ichild]  ) ; 
                        }  
                    } );

        }
    )   ;
}

template void 
interp_hanging_ghostzones<utils::linear_prolongator_t<thunder::minmod>>(
      thunder::var_array_t<THUNDER_NSPACEDIM>& 
    , thunder::var_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
) ; 
template void 
interp_hanging_ghostzones<utils::linear_prolongator_t<thunder::MCbeta>>(
      thunder::var_array_t<THUNDER_NSPACEDIM>& 
    , thunder::var_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
) ; 

}} /* namespace thunder::amr */