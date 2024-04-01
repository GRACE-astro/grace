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

namespace detail {
/**
 * @brief Assemble interpolation stencil in a quadrant/tree boundary aware fashion.
 * 
 * NB: This function is not meant to be called by user code.
 * 
 */
template< typename ViewT > 
void THUNDER_HOST_DEVICE THUNDER_ALWAYS_INLINE 
get_interpolation_stencil( double * x_interp, double * y_interp, int nstencil 
                         , int which_tree_a
                         , int which_tree_b
                         , int which_face_a 
                         , int which_face_b 
                         , neigbor_info_t const & neighbor_info
                         , double qcoords_a[THUNDER_NSPACEDIM]
                         , double qcoords_b[THUNDER_NSPACEDIM]
                         , double cidx_a[THUNDER_NSPACEDIM]
                         , double cidx_b[THUNDER_NSPACEDIM]
                         , ViewT& view_b 
                         , int64_t qid_b 
                         , int ivar )
{
    int const npoints = EXPR(nstencil, *nstencil, *nstencil) ; 

    /* Stencil layout is C style by convention (right alignment) */
    /* Also by convention, the first direction is the            */
    /* one orthogonal to the quadrant face                       */
    /* whose ghost zones are being filled                        */
    int offset = Kokkos::floor((nstencil-1)/2) ; 
    for(int i=0; i<nstencil; ++i) {
        for(int j=0; j<nstencil; ++j) {
            for(int k=0; k<nstencil; ++k) {

            }
        } 
    }
}

}


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
    Kokkos::vector<simple_face_info_t>& interior_faces
)
{
    using namespace thunder; 
    using namespace Kokkos ; 

    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int64_t ngz = amr::get_n_ghosts() ;
    int64_t nq  = amr::get_local_num_quadrants() ;
    auto& vars = variable_list::get().getstate() ; 
    auto& halo = variable_list::get().gethalo()  ;
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
void interp_simple_ghostzones(
      Kokkos::vector<simple_face_info_t>& interior_faces
    , Kokkos::View<neighbor_info_t>& neighbor_info 
)
{
    using namespace thunder; 
    using namespace Kokkos ; 

    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int64_t ngz = amr::get_n_ghosts() ;
    int64_t nq  = amr::get_local_num_quadrants() ;
    auto& vars = variable_list::get().getstate() ; 
    auto& halo = variable_list::get().gethalo()  ;
    int nvars  = variables::get_n_evolved()      ;
    auto const n_faces = interior_faces.size()   ;
    auto& d_face_info = interior_faces.d_view    ; 

    auto& coord_system = coordinate_system::get() ; 
    auto device_coord_system = coord_system.get_device_coord_system() ; 

    TeamPolicy<default_execution_space> 
        policy( n_faces, AUTO() ) ; 
    using member_t = decltype(policy)::member_type ;

    parallel_for( THUNDER_EXECUTION_TAG("AMR","interp_simple_interior_ghostzones")
                , policy 
                , KOKKOS_LAMBDA( const member_t& team )
        {
            int polarity     =  d_face_info(team.league_rank()).has_polarity_flip ;
            int is_ghost     =  d_face_info(team.league_rank()).is_ghost          ; 
            int which_face_a =  d_face_info(team.league_rank()).which_face_a      ; 
            int which_face_b =  d_face_info(team.league_rank()).which_face_b      ;
            int which_tree_a =  d_face_info(team.league_rank()).which_tree_a      ;
            int which_tree_b =  d_face_info(team.league_rank()).which_tree_b      ;
            int64_t qid_a    =  d_face_info(team.league_rank()).qid_a             ;
            int64_t qid_b    =  d_face_info(team.league_rank()).qid_b             ; 
            
            int64_t n1 = (which_face_a/2==0) * ny + ((which_face_a/2==1) * nx) + ((which_face_a/2==2) * nx) ;
            int64_t n2 = (which_face_a/2==0) * nz + ((which_face_a/2==1) * nz) + ((which_face_a/2==2) * ny) ;
            
            auto& view_a = vars ; 
            auto& view_b = (is_ghost) ? halo : vars ; 

            auto& coords_a = coords ; 
            auto& coords_b = (is_ghost) ? halo_coords : coords ; 

            auto& invspacing_a = invspacing ; 
            auto& invspacing_b =  (is_ghost) ? halo_invspacing : invspacing ; 
            /* Collect coordinates */
            double qcoords_a[THUNDER_NSPACEDIM] =
            {VEC(
                coords_a(0,qid_a), coords_a(1,qid_a), coords_a(2,qid_a)
            )} ; 
            double qcoords_b[THUNDER_NSPACEDIM] =
            {VEC(
                coords_b(0,qid_b), coords_a(1,qid_b), coords_a(2,qid_b)
            )} ;

            double cidx_a[THUNDER_NSPACEDIM] =
            {VEC(
                invspacing_a(0,qid_a), invspacing_a(1,qid_a), invspacing_a(2,qid_a)
            )} ; 
            double cidx_b[THUNDER_NSPACEDIM] =
            {VEC(
                invspacing_b(0,qid_b), invspacing_b(1,qid_b), invspacing_b(2,qid_b)
            )} ;

            TeamThreadMDRange<Rank<THUNDER_NSPACEDIM>,member_t>
                team_range( team, VECD(n1,n2), nvars) ; 
            parallel_for( team_range
                        , KOKKOS_LAMBDA(VECD(int& j, int& k), int& ivar)

                    {
                        /*************************************************/
                        /* We handle these cases separately since        */
                        /* cross-tree boundaries require significant     */
                        /* extra work with respect to simple interior    */
                        /* boundaries. Anyhow, all threads in a team     */
                        /* will take the same branch and it should be    */
                        /* *essentially* free.                           */
                        /*************************************************/
                        if( which_tree_a == which_tree_b ){
                            for( int ig=0; ig<ngz; ++ig){
                                EXPR(
                                int i_a = EXPR(
                                        (which_face_a==0)   * ig 
                                        + (which_face_a==1)   * (nx+ngz+ig),
                                        + (which_face_a/2==1) * (j+ngz), 
                                        + (which_face_a/2==2) * (j+ngz));,
                                int j_a = EXPR(
                                        (which_face_a==2)   * ig 
                                        + (which_face_a==3)   * (nx+ngz+ig),
                                        + (which_face_a/2==0) * (j+ngz), 
                                        + (which_face_a/2==2) * (k+ngz));,
                                int k_a = EXPR(
                                        (which_face_a==4)   * ig 
                                        + (which_face_a==5)   * (nx+ngz+ig),
                                        + (which_face_a/2==0) * (k+ngz), 
                                        + (which_face_a/2==1) * (k+ngz));
                                )
                                double lcoords[THUNDER_NSPACEDIM] = 
                                {VEC(
                                    coords_a(0,qid_a) + (i_a-ngz) * invspacing_a(0,qid_a),
                                    coords_a(1,qid_a) + (j_a-ngz) * invspacing_a(1,qid_a),
                                    coords_a(2,qid_a) + (k_a-ngz) * invspacing_a(2,qid_a)
                                )};
                                EXPR(
                                int i_b = (lcoords_b[0] - coords_b(0,qid_b)) * invspacing_b(0,qid_b);,
                                int j_b = (lcoords_b[1] - coords_b(1,qid_b)) * invspacing_b(1,qid_b);,
                                int k_b = (lcoords_b[2] - coords_b(2,qid_b)) * invspacing_b(2,qid_b);
                                )
                                view_a(VEC(i_a,j_a,k_a),ivar,qid_a) =  view_b(VEC(i_b,j_b,k_b),ivar,qid_b) ;
                                EXPR(
                                i_b = EXPR(
                                          (which_face_b==0)   * ig 
                                        + (which_face_b==1)   * (nx+ngz+ig),
                                        + (which_face_b/2==1) * (j+ngz), 
                                        + (which_face_b/2==2) * (j+ngz));,
                                j_b = EXPR(
                                          (which_face_b==2)   * ig 
                                        + (which_face_b==3)   * (nx+ngz+ig),
                                        + (which_face_b/2==0) * (j+ngz), 
                                        + (which_face_b/2==2) * (k+ngz));,
                                k_b = EXPR(
                                          (which_face_b==4)   * ig 
                                        + (which_face_b==5)   * (nx+ngz+ig),
                                        + (which_face_b/2==0) * (k+ngz), 
                                        + (which_face_b/2==1) * (k+ngz));
                                )
                                {EXPR(
                                lcoords[0] = coords_b(0,qid_b) + (i_b-ngz) * invspacing_b(0,qid_b);,
                                lcoords[1] = coords_b(1,qid_b) + (j_b-ngz) * invspacing_b(1,qid_b);,
                                lcoords[2] = coords_b(2,qid_b) + (k_b-ngz) * invspacing_b(2,qid_b);
                                )};
                                EXPR(
                                i_a = (lcoords[0] - coords(0,qid_b)) * invspacing_b(0,qid_b);,
                                j_a = (lcoords[1] - coords(1,qid_b)) * invspacing_b(1,qid_b);,
                                k_a = (lcoords[2] - coords(2,qid_b)) * invspacing_b(2,qid_b);
                                )
                                view_b(VEC(i_b,j_b,k_b),ivar,qid_b) =  view_a(VEC(i_a,j_a,k_a),ivar,qid_a) ;
                            }
                        } else {
                            for( int ig=0; ig<ngz; ++ig){
                                
                                EXPR(
                                int i_a = EXPR(
                                          (which_face_a==0)   * ig 
                                        + (which_face_a==1)   * (nx+ngz+ig),
                                        + (which_face_a/2==1) * (j+ngz), 
                                        + (which_face_a/2==2) * (j+ngz));,
                                int j_a = EXPR(
                                          (which_face_a==2)   * ig 
                                        + (which_face_a==3)   * (nx+ngz+ig),
                                        + (which_face_a/2==0) * (j+ngz), 
                                        + (which_face_a/2==2) * (k+ngz));,
                                int k_a = EXPR(
                                          (which_face_a==4)   * ig 
                                        + (which_face_a==5)   * (nx+ngz+ig),
                                        + (which_face_a/2==0) * (k+ngz), 
                                        + (which_face_a/2==1) * (k+ngz));
                                )

                                double lcoords_a[THUNDER_NSPACEDIM] = 
                                {VEC(
                                    qcoords_a[0] + (i_a-ngz) * cidx_a[0],
                                    qcoords_a[1] + (j_a-ngz) * cidx_a[1],
                                    qcoords_a[2] + (k_a-ngz) * cidx_a[2]
                                )};
                                double lcoords_b[THUNDER_NSPACEDIM] ; 
                                device_coords.transfer_coordinates( which_tree_a, which_tree_b
                                                                  , which_face_a, which_face_b
                                                                  , lcoords_a, lcoords_b ); 
                                /*************************************************/
                                /* Find indices of closest point (from below)    */
                                /* in source tree                                */
                                /*************************************************/
                                EXPR(
                                int i_b = (lcoords_b[0] - qcoords_b[0]) * cidx_b[0];,
                                int j_b = (lcoords_b[1] - qcoords_b[1]) * cidx_b[1];,
                                int k_b = (lcoords_b[2] - qcoords_b[2]) * cidx_b[2];
                                )
                                /*************************************************/
                                /* Assemble a stencil for interpolation.         */
                                /* Here a few considerations are necessary:      */
                                /* 1) Depending on the logic coordinate spacing  */
                                /*    of each tree, the first point in the ghost */
                                /*    zones might be closer to the quadrant      */
                                /*    boudary than any valid point. Since ghosts */
                                /*    are obviously invalid at this stage, in    */
                                /*    that case we need to fill the first point  */
                                /*    in the stencil with data from quadrant A.  */
                                /* 2) Analogously, for multi-point stencils, to  */
                                /*    avoid the intrinsic inaccuracies of one-   */
                                /*    sided interpolation, we gather the points  */
                                /*    we need from quadrant A.                   */
                                /*************************************************/

                                size_t constexpr n_stencil = InterpT::stencil_size ; 
                                size_t constexpr n_points  = EXPR(n_stencil, *n_stencil, *n_stencil) ; 

                                double x_interp[n_points * THUNDER_NSPACEDIM] ; 
                                double y_interp[n_points * THUNDER_NSPACEDIM] ; 
                                
                                detail::get_interpolation_stencil( x_interp,y_interp
                                                                 , which_tree_a
                                                                 , which_tree_b
                                                                 , which_face_a 
                                                                 , which_face_b 
                                                                 , neighbor_info
                                                                 , qcoords_a 
                                                                 , qcoords_b
                                                                 , cidx_a 
                                                                 , cidx_b 
                                                                 , view_b 
                                                                 , qid_b 
                                                                 , ivar ) ; 

                                InterpT interpolator(x_interp,y_interp) ; 

                                view_a(VEC(i_a,j_a,k_a),ivar,qid_a) = interpolator.interpolate(  lcoords_a[0]
                                                                                               , lcoords_a[1]
                                                                                               , lcoords_a[2] ) ;

                                if( !is_ghost ){
                                    /* Fill ghost zones of view_b */
                                }
                                

                            } /* loop over points in gzs */
                        }
                    } );
        }
    )   ;  
}

template< typename InterpT > 
void interp_hanging_ghostzones(
    Kokkos::vector<hanging_face_info_t>& hanging_faces
) 
{
    using namespace thunder ;
    using namespace Kokkos  ; 

    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int64_t ngz = amr::get_n_ghosts() ;
    int64_t nq  = amr::get_local_num_quadrants() ;
    auto& vars = variable_list::get().getstate() ; 
    auto& halo = variable_list::get().gethalo()  ; 
    int const nvars = variables::get_n_evolved() ;
    auto const n_faces = hanging_faces.size()   ;
    auto& d_face_info = hanging_faces.d_view    ; 

    constexpr const int n_neighbors = PICK_D(2,4) ;  

    TeamPolicy<default_execution_space> 
        policy( n_faces, AUTO() ) ; 
    using member_t = decltype(policy)::member_type ;

    parallel_for
    (             THUNDER_EXECUTION_TAG("AMR","prolongate_restrict_hanging_faces")
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
            
            int64_t n1 = (which_face_a/2==0) * ny + ((which_face_a/2==1) * nx) + ((which_face_a/2==2) * nx) ;
            int64_t n2 = (which_face_a/2==0) * nz + ((which_face_a/2==1) * nz) + ((which_face_a/2==2) * ny) ;
            
            auto& view_a = (is_ghost_a) ? halo : vars ;
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

            
            double const dx_parent =  1./(1L<<d_face_info(team.league_rank()).level_a)/nx;
            double const dy_parent =  1./(1L<<d_face_info(team.league_rank()).level_a)/ny;
            double const dz_parent =  1./(1L<<d_face_info(team.league_rank()).level_a)/nz;
            
            double const dx_child =  1./(1L<<d_face_info(team.league_rank()).level_b)/nx;
            double const dy_child =  1./(1L<<d_face_info(team.league_rank()).level_b)/ny;
            double const dz_child =  1./(1L<<d_face_info(team.league_rank()).level_b)/nz;
            

            TeamThreadMDRange<Rank<THUNDER_NSPACEDIM>,member_t>
                team_range( team, VEC(ngz, n1,n2)) ; 
            parallel_for( team_range
                        , KOKKOS_LAMBDA(VEC(int& ig, int& j, int& k))
                    {
                        int fine_idx = 
                            (2*j)/n1 
                            #ifdef THUNDER_3D 
                            + (2*k)/n2 * 2 
                            #endif 
                        ;
                        /***********************************************/
                        /* Coordinate spacings orthogonal and parallel */ 
                        /* to face in parent quadrant.                 */
                        /***********************************************/
                        double dx0 = EXPR( (which_face_a/2==0) * dx_parent,
                                   + (which_face_a/2==1) * dy_parent, 
                                   + (which_face_a/2==2) * dz_parent ) ;
                        double dx1 = EXPR( (which_face_a/2==0) * dy_parent,
                                   + (which_face_a/2==1) * dx_parent, 
                                   + (which_face_a/2==2) * dx_parent ) ;
                        double dx2 = EXPR( (which_face_a/2==0) * dz_parent,
                                   + (which_face_a/2==1) * dz_parent, 
                                   + (which_face_a/2==2) * dy_parent ) ; 
                        /***********************************************/
                        /* indices in coarse quadrant                  */
                        /* these are in ghost-zone                     */
                        /***********************************************/
                        int i_a = EXPR((which_face_a==0) * 
                                  ( (!polarity)*ig 
                                  + (polarity)*(ngz-1-ig) )
                                + (which_face_a==1) * 
                                  ( (!polarity)*(nx+ig)  
                                  + (polarity)*(nx+ngz-1-ig) ),
                                + (which_face_a/2==1) * (j+ngz), 
                                + (which_face_a/2==2) * (j+ngz)) ;

                        int j_a = EXPR((which_face_a==2) * 
                                  ( (!polarity)*ig 
                                  + (polarity)*(ngz-1-ig) )
                                + (which_face_a==3) * 
                                  ( (!polarity)*(ny+ig)  
                                  + (polarity)*(ny+ngz-1-ig) ), 
                                + (which_face_a/2==0) * (j+ngz), 
                                + (which_face_a/2==2) * (k+ngz)) ;  
                        /***********************************************/
                        /* Interpolation coordinates (defined in fine) */
                        /* grid's coordinate system                    */
                        /***********************************************/
                        EXPR( 
                        double x0 = EXPR(((which_face_b==0) * (ig + 0.5)      
                                  +  (which_face_b==0) * (nx - ig - 0.5)) * dx0,
                                  + (which_face_b/2==1) * ((2 * j)%nx)    * dx1, 
                                  + (which_face_b/2==2) * ((2 * j)%nx)    * dx1);,
                        double y0 = EXPR(((which_face_b==2) * (ig + 0.5) 
                                  +  (which_face_b==3) * (ny - ig - 0.5)) * dx0,
                                  + (which_face_b/2==0) * ((2 * j)%ny)    * dx1, 
                                  + (which_face_b/2==2) * ((2 * k)%ny)    * dx2);,
                        double z0 = EXPR(((which_face_b==4) * (ig + 0.5) 
                                  +  (which_face_b==5) * (nz - ig - 0.5)) * dx0, 
                                  + (which_face_b/2==0) * ((2 * k)%nz)    * dx2, 
                                  + (which_face_b/2==1) * ((2 * k)%nz)    * dx2);
                        )
                        /***********************************************/
                        /* indices in fine quadrant                    */
                        /* these are within the physical grid          */
                        /***********************************************/
                        int i_b = (which_face_b==0) * (ngz+Kokkos::floor((x0-dx_child/2)/dx_child)-1)
                                + (which_face_b!=0) * (Kokkos::floor(x0/dx_child) + ngz);
                        
                        int j_b = (which_face_b==2) * (ngz+Kokkos::floor((y0-dy_child/2)/dy_child)-1)
                                + (which_face_b!=2) * (Kokkos::floor(y0/dy_child) + ngz);
                        /***********************************************/
                        #ifdef THUNDER_3D
                        int k_a = (which_face_a==4) * 
                                  ( (!polarity)*ig 
                                  + (polarity)*(ngz-1-ig) )
                                + (which_face_a==5) * 
                                  ( (!polarity)*(nz+ig)  
                                  + (polarity)*(nz+ngz-1-ig) )
                                + (which_face_a/2!=2) * (k+ngz) ;

                        int k_b = (which_face_b==4) * (ngz+Kokkos::floor((z0-dz_child/2)/dz_child)-1)
                                + (which_face_b!=4) * (Kokkos::floor(z0/dz_child) + ngz);
                        /***********************************************/
                        #endif 
                        /* fine variable view and quadrant index       */
                        auto& view_2 = (fine_view_is_ghost[fine_idx]) ? halo : vars ; 
                        auto iq_fine = fine_iqs[fine_idx]   ;
                        size_t constexpr stencil = InterpT::stencil_size ; 
                        size_t constexpr npoints = stencil*stencil       ;
                        int x_param[THUNDER_NSPACEDIM*npoints]           ; 
                        InterpT::get_parametric_coordinates(x_param )    ; 
                        double x[THUNDER_NSPACEDIM*npoints]              ;
                        double y[THUNDER_NSPACEDIM*npoints]              ; 
                        int smin = Kokkos::floor(stencil/2) - 1          ;
                        for(size_t is=0; is<npoints;++is)
                        {
                            EXPR(
                            x[THUNDER_NSPACEDIM*is + 0]  
                                = (i_b + 0.5 - smin - ngz + x_param[THUNDER_NSPACEDIM*is+0]) * dx_child;,

                            x[THUNDER_NSPACEDIM*is + 1]  
                                = (j_b + 0.5 - smin - ngz + x_param[THUNDER_NSPACEDIM*is+1]) * dy_child;,

                            x[THUNDER_NSPACEDIM*is + 2]  
                                = ( k_b + 0.5 - smin - ngz + x_param[THUNDER_NSPACEDIM*is+2]) * dz_child;
                            ); 
                        }
                        for(int ivar=0; ivar<nvars; ++ivar){
                            for(size_t is=0; is<npoints;++is)
                            {
                                y[is] = view_2 (
                                    VEC( i_b - smin + x_param[THUNDER_NSPACEDIM*is+0]
                                       , j_b - smin + x_param[THUNDER_NSPACEDIM*is+1]
                                       , k_b - smin + x_param[THUNDER_NSPACEDIM*is+2] )
                                    , ivar, iq_fine ) ; 
                            }
                            InterpT interpolator(x,y);
                            view_a(VEC(i_a,j_a,k_a),ivar,qid_a) = interpolator.interpolate(VEC(x0,y0,z0)) ; 
                        }
                        /******************************************************/
                        /* Now we play the same game, this time interpolating */
                        /* from coarse to fine                                */
                        /******************************************************/
                        /* Fine quadrant (ghost) indices */
                        i_b = EXPR((which_face_b==0) * 
                                  ( (!polarity)*ig 
                                  + (polarity)*(ngz-1-ig) )
                                + (which_face_b==1) * 
                                  ( (!polarity)*(nx+ig)  
                                  + (polarity)*(nx+ngz-1-ig) ),
                                + (which_face_b/2==1) * (j+ngz),     
                                + (which_face_b/2==2) * (j+ngz));    
                        
                        j_b = EXPR((which_face_b==2) * 
                                  ( (!polarity)*ig 
                                  + (polarity)*(ngz-1-ig) )
                                + (which_face_b==3) * 
                                  ( (!polarity)*(ny+ig)  
                                  + (polarity)*(ny+ngz-1-ig) ), 
                                + (which_face_b/2==0) * (j+ngz), 
                                + (which_face_b/2==2) * (k+ngz)); 
                        #ifdef THUNDER_3D 
                        k_b =     (which_face_b==4) * 
                                  ( (!polarity)*ig 
                                  + (polarity)*(ngz-1-ig) )
                                + (which_face_b==5) * 
                                  ( (!polarity)*(nz+ig)  
                                  + (polarity)*(nz+ngz-1-ig) ) 
                                + (which_face_b/2==0) * (k+ngz) 
                                + (which_face_b/2==1) * (k+ngz); 
                        #endif 
                        /* Right handed cordinates across face (0) */
                        /* and orthogonal to it (1,2)              */
                        dx0 = EXPR( (which_face_b/2==0) * dx_child,
                              + (which_face_b/2==1) * dy_child, 
                              + (which_face_b/2==2) * dz_child ) ;
                        dx1 = EXPR( (which_face_b/2==0) * dy_child,
                              + (which_face_b/2==1) * dx_child, 
                              + (which_face_b/2==2) * dx_child ) ;
                        dx2 = EXPR( (which_face_b/2==0) * dz_child,
                              + (which_face_b/2==1) * dz_child, 
                              + (which_face_b/2==2) * dy_child ) ;
                        for( int ifine = 0; ifine <n_neighbors ; ++ifine)
                        {
                            int const iquad_1 = ifine % 2 ;
                            #ifdef THUNDER_3D 
                            int const iquad_2 = static_cast<int>(Kokkos::floor(ifine/2));
                            #endif 
                            /* coordinates of the interpolated point wrt the coarse grid */
                            EXPR( 
                            x0 = EXPR(((which_face_a==0) * (ig + 0.5)      
                                 +  (which_face_a==0) * (nx - ig - 0.5)) * dx0,
                                 + (which_face_a/2==1) * (iquad_1 * nx + j + 0.5) * dx1, 
                                 + (which_face_a/2==2) * (iquad_1 * nx + j + 0.5) * dx1);,
                            y0 = EXPR(((which_face_a==2) * (ig + 0.5) 
                                 +  (which_face_a==3) * (ny - ig - 0.5)) * dx0,
                                 + (which_face_a/2==0) * (iquad_1 * ny + j + 0.5) * dx1, 
                                 + (which_face_a/2==2) * (iquad_2 * ny + k + 0.5) * dx2);,
                            z0 = EXPR(((which_face_a==4) * (ig + 0.5) 
                                 +  (which_face_a==5) * (nz - ig - 0.5)) * dx0, 
                                 + (which_face_a/2==0) * (iquad_2 * nz + k + 0.5) * dx2, 
                                 + (which_face_a/2==1) * (iquad_2 * nz + k + 0.5) * dx2);
                            )
                            /* coarse (physical) indices   */
                            /* TODO: this can be optimized */
                            i_a =     (which_face_a==0)   * (ngz+Kokkos::floor((x0+0.5*dx_parent)/2)-1)
                                    + (which_face_a==1)   * (Kokkos::floor(x0/dx_parent) + ngz)
                                    + (which_face_a/2==1) * (Kokkos::floor(x0/dx_parent) + ngz) 
                                    + (which_face_a/2==2) * (Kokkos::floor(x0/dx_parent) + ngz);

                            j_a =     (which_face_a==2)   * (ngz+Kokkos::floor((y0+0.5*dy_parent)/2)-1)
                                    + (which_face_a==3)   * (Kokkos::floor(y0/dy_parent) + ngz)
                                    + (which_face_a/2==0) * (Kokkos::floor(y0/dy_parent) + ngz) 
                                    + (which_face_a/2==2) * (Kokkos::floor(y0/dy_parent) + ngz);

                            #ifdef THUNDER_3D              
                            k_a =     (which_face_a==4)   * (ngz+Kokkos::floor((z0+0.5*dz_parent)/2)-1)
                                    + (which_face_a==5)   * (Kokkos::floor(z0/dz_parent) + ngz)
                                    + (which_face_a/2==0) * (Kokkos::floor(z0/dz_parent) + ngz) 
                                    + (which_face_a/2==1) * (Kokkos::floor(z0/dz_parent) + ngz); 
                            #endif 
                            auto& view_3  = (fine_view_is_ghost[ifine]) ? halo : vars ; 
                            iq_fine = fine_iqs[ifine]   ; 
                            for(size_t is=0; is<npoints;++is)
                            {
                                EXPR(
                                x[THUNDER_NSPACEDIM*is + 0]  
                                    = (i_a + 0.5 - smin - ngz + x_param[THUNDER_NSPACEDIM*is+0]) * dx_parent;,

                                x[THUNDER_NSPACEDIM*is + 1]  
                                    = (j_a + 0.5 - smin - ngz + x_param[THUNDER_NSPACEDIM*is+1]) * dy_parent;,

                                x[THUNDER_NSPACEDIM*is + 2]  
                                    = (k_a + 0.5 - smin - ngz + x_param[THUNDER_NSPACEDIM*is+2]) * dz_parent;
                                ); 
                            }
                            
                            
                            for(int ivar=0; ivar<nvars; ++ivar){
                                for(size_t is=0; is<npoints;++is)
                                {
                                    y[is] = view_3 (
                                        VEC( i_a - smin + x_param[THUNDER_NSPACEDIM*is+0]
                                           , j_a - smin + x_param[THUNDER_NSPACEDIM*is+1]
                                           , k_a - smin + x_param[THUNDER_NSPACEDIM*is+2] )
                                        , ivar, qid_a ) ; 
                                }
                                InterpT interpolator(x,y);
                                view_a(VEC(i_b,j_b,k_b),ivar,iq_fine) = interpolator.interpolate(VEC(x0,y0,z0)) ; 
                            }
                        }

                    } );

        }
    )   ;
}

template void 
interp_hanging_ghostzones<utils::linear_interp_t<THUNDER_NSPACEDIM>>(Kokkos::vector<hanging_face_info_t>&) ; 

}} /* namespace thunder::amr */