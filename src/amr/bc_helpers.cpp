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

#include <grace/amr/bc_helpers.hh>
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

void grace_iterate_faces( p4est_iter_face_info_t * info 
                          , void* user_data  )
{
    using namespace grace; 
    auto face_info = reinterpret_cast<grace_face_info_t*>(user_data) ; 
    sc_array_view_t<p4est_iter_face_side_t> sides{
        &(info->sides)
    } ; 
    auto& physical_boundary_info = face_info->phys_boundary_info ; 
    auto& simple_info    = face_info->simple_interior_info       ;
    auto& hanging_info   = face_info->hanging_interior_info      ;
    auto& coarse_hanging_info = face_info->coarse_hanging_quads_info ; 
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
            #ifdef GRACE_3D 
            physical_boundary_info.push_back(
                (offset+sides[0].is.hanging.quadid[2]) * P4EST_FACES + sides[0].face
            ); 
            physical_boundary_info.push_back(
                (offset+sides[0].is.hanging.quadid[3]) * P4EST_FACES + sides[0].face
            ); 
            #endif 
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

    if( sides[1].is_hanging )
    {
        hanging_face_info_t this_face_info{} ; 
        this_face_info.has_polarity_flip = polarity_flip ; 
        this_face_info.level_coarse = static_cast<int>(sides[0].is.full.quad->level) ;
        this_face_info.level_fine   = this_face_info.level_coarse + 1 ;
        this_face_info.which_face_coarse = sides[0].face ;
        this_face_info.which_face_fine   = sides[1].face ; 
        this_face_info.is_ghost_coarse   = sides[0].is.full.is_ghost ; 
        this_face_info.qid_coarse        = sides[0].is.full.quadid   
            + ( this_face_info.is_ghost_coarse ? 0 : get_local_quadrants_offset(sides[0].treeid)  ) ; 
        int any_fine_ghost{0} ; 
        for(int ii=0; ii<P4EST_CHILDREN/2; ++ii) {
            this_face_info.is_ghost_fine[ii] = sides[1].is.hanging.is_ghost[ii] ; 
            any_fine_ghost += this_face_info.is_ghost_fine[ii] ; 
            this_face_info.qid_fine[ii] = sides[1].is.hanging.quadid[ii]
                + ( this_face_info.is_ghost_fine[ii] ? 0 : get_local_quadrants_offset(sides[1].treeid)  ) ; 
            ASSERT_DBG(this_face_info.is_ghost_fine[ii] == 0 or this_face_info.is_ghost_fine[ii] == 1
                      , "Is ghost fine neither true or false.") ; 
            ASSERT_DBG(this_face_info.qid_fine[ii]>=0, "Negative quadrant id in grace iter faces.") ; 
        }
        if( this_face_info.is_ghost_coarse ) { 
            auto halos = info->ghost_layer ; 
            coarse_hanging_info.rcv_quadid.push_back( this_face_info.qid_coarse ) ;
            for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                size_t first_halo  = halos->proc_offsets[iproc]   ; 
                size_t last_halo   = halos->proc_offsets[iproc+1] ;
                if( this_face_info.qid_coarse >= first_halo and this_face_info.qid_coarse < last_halo ) {
                    coarse_hanging_info.rcv_procid.push_back( iproc ) ; 
                }
            }
            face_info->n_hanging_ghost_faces ++ ;  
        } else if (  any_fine_ghost ) {
            auto halos = info->ghost_layer ; 
            sc_array_view_t<p4est_quadrant_t>  mirror_quads { &(halos->mirrors) } ; 
            coarse_hanging_info.snd_quadid.push_back(this_face_info.qid_coarse) ; 
            std::set<int> _snd_procid;
            for( int ii=0; ii<P4EST_CHILDREN/2; ++ii){
                if( this_face_info.is_ghost_fine[ii] ) {
                    for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                        size_t first_halo  = halos->proc_offsets[iproc]   ; 
                        size_t last_halo   = halos->proc_offsets[iproc+1] ;
                        if( this_face_info.qid_fine[ii] >= first_halo and this_face_info.qid_fine[ii] < last_halo ) {
                            _snd_procid.insert( iproc ) ; 
                        }
                    }
                }
            }
            coarse_hanging_info.snd_procid.push_back(_snd_procid) ; 
            face_info->n_hanging_ghost_faces ++ ;
        }
        hanging_info.push_back(this_face_info) ; 
    } else if(sides[0].is_hanging) {
        hanging_face_info_t this_face_info{} ; 
        this_face_info.has_polarity_flip = polarity_flip ; 
        this_face_info.level_coarse = static_cast<int>(sides[1].is.full.quad->level) ;
        this_face_info.level_fine   = this_face_info.level_coarse + 1 ;
        this_face_info.which_face_coarse = sides[1].face ;
        this_face_info.which_face_fine   = sides[0].face ; 
        this_face_info.is_ghost_coarse   = sides[1].is.full.is_ghost ; 
        this_face_info.qid_coarse        = sides[1].is.full.quadid   
            + ( this_face_info.is_ghost_coarse ? 0 : get_local_quadrants_offset(sides[1].treeid)  ) ; 
        int any_fine_ghost{0} ; 
        for(int ii=0; ii<P4EST_CHILDREN/2; ++ii) {
            this_face_info.is_ghost_fine[ii] = sides[0].is.hanging.is_ghost[ii] ; 
            any_fine_ghost += this_face_info.is_ghost_fine[ii] ; 
            this_face_info.qid_fine[ii] = sides[0].is.hanging.quadid[ii]
                + ( this_face_info.is_ghost_fine[ii] ? 0 : get_local_quadrants_offset(sides[0].treeid)  ) ; 
            ASSERT_DBG(this_face_info.is_ghost_fine[ii] == 0 or this_face_info.is_ghost_fine[ii] == 1
                      , "Is ghost fine neither true or false.") ; 
            ASSERT_DBG(this_face_info.qid_fine[ii]>=0, "Negative quadrant id in grace iter faces.") ; 
        }
        if( this_face_info.is_ghost_coarse ) { 
            auto halos = info->ghost_layer ; 
            coarse_hanging_info.rcv_quadid.push_back( this_face_info.qid_coarse ) ;
            for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                size_t first_halo  = halos->proc_offsets[iproc]   ; 
                size_t last_halo   = halos->proc_offsets[iproc+1] ;
                if( this_face_info.qid_coarse >= first_halo and this_face_info.qid_coarse < last_halo ) {
                    coarse_hanging_info.rcv_procid.push_back( iproc ) ; 
                }
            }
            face_info->n_hanging_ghost_faces ++ ;  
        } else if (  any_fine_ghost ) {
            auto halos = info->ghost_layer ; 
            sc_array_view_t<p4est_quadrant_t>  mirror_quads { &(halos->mirrors) } ; 
            coarse_hanging_info.snd_quadid.push_back(this_face_info.qid_coarse) ; 
            std::set<int> _snd_procid;
            for( int ii=0; ii<P4EST_CHILDREN/2; ++ii){
                if( this_face_info.is_ghost_fine[ii] ) {
                    for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                        size_t first_halo  = halos->proc_offsets[iproc]   ; 
                        size_t last_halo   = halos->proc_offsets[iproc+1] ;
                        if( this_face_info.qid_fine[ii] >= first_halo and this_face_info.qid_fine[ii] < last_halo ) {
                            _snd_procid.insert( iproc ) ; 
                        }
                    }
                }
            }
            coarse_hanging_info.snd_procid.push_back(_snd_procid) ; 
            face_info->n_hanging_ghost_faces ++ ;
        }
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
            face_info->n_simple_ghost_faces ++ ; 
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
            face_info->n_simple_ghost_faces ++ ; 
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
      grace::var_array_t<GRACE_NSPACEDIM>& vars
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , Kokkos::vector<simple_face_info_t>& interior_faces
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
    auto& d_face_info = interior_faces.d_view    ; 
    if( n_faces == 0 ) {
        return ; 
    }
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy(
            {0,VECD(0,0), 0,0},
            {ngz, VECD(static_cast<long>(nx),static_cast<long>(ny)), static_cast<long>(nvars), static_cast<long>(n_faces)}
        ) ;
    parallel_for(GRACE_EXECUTION_TAG("AMR", "copy_interior_ghostzones")
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
            int i_a = EXPR((which_face_a==0) *ig 
                        + (which_face_a==1) * (nx+ngz+ig),
                        + (which_face_a/2==1) * (j+ngz), 
                        + (which_face_a/2==2) * (j+ngz)) ;

            int j_a = EXPR((which_face_a==2) * ig 
                    + (which_face_a==3) * (ny+ngz+ig), 
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

            #ifdef GRACE_3D
            int k_a = (which_face_a==4) * ig 
                    + (which_face_a==5) *  (nz+ngz+ig)
                    + (which_face_a/2!=2) * (k+ngz) ;

            int k_b = (which_face_b==4)*(ngz+ig)
                    + (which_face_b==5)*(nz+ig)
                    + (which_face_b/2!=2) * (k + ngz) ;
            #endif 

            view_a(VEC(i_a,j_a,k_a),ivar,qid_a) =  view_b(VEC(i_b,j_b,k_b),ivar,qid_b) ; 
            
            if( ! is_ghost ) {
                i_b = EXPR((which_face_b==0) * ig 
                        + (which_face_b==1) * (nx+ngz+ig),
                        + (which_face_b/2==1) * (j+ngz), 
                        + (which_face_b/2==2) * (j+ngz)) ;

                j_b = EXPR((which_face_b==2) * ig
                        + (which_face_b==3) * (ny+ngz+ig), 
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

                #ifdef GRACE_3D
                k_b =     (which_face_b==4) *ig 
                        + (which_face_b==5) * (nz+ngz+ig)
                        + (which_face_b/2!=2) * (k+ngz) ;

                k_a =     (which_face_a==4)*(ngz+ig)
                        + (which_face_a==5)*(nz+ig)
                        + (which_face_a/2!=2) * (k + ngz) ;
                #endif  
                #ifndef GRACE_CARTESIAN_COORDINATES
                // TODO HERE  WE ASSUME Nx==Ny==Nz
                auto const lmn = mapper({VEC(i_b,j_b,k_b)}, tid_a, tid_b, {ng,n1,n2}) ;
                i_b = lmn[0]; j_b = lmn[1]; k_b = lmn[2] ; 
                #endif 
                view_b(VEC(i_b,j_b,k_b),ivar,qid_b) =  view_a(VEC(i_a,j_a,k_a),ivar,qid_a) ;
            }
        });
    #if 0 
    TeamPolicy<default_execution_space> 
        policy( n_faces, AUTO() ) ; 
    using member_t = decltype(policy)::member_type ;

    parallel_for( GRACE_EXECUTION_TAG("AMR","copy_interior_ghostzones")
                , policy 
                , KOKKOS_LAMBDA( const member_t& team )
        {
            /* Get information about quadrants sharing the face */
            int polarity     =  d_face_info(team.league_rank()).has_polarity_flip ;
            int is_ghost     =  d_face_info(team.league_rank()).is_ghost          ; 
            int which_face_a =  d_face_info(team.league_rank()).which_face_a      ; 
            int which_face_b =  d_face_info(team.league_rank()).which_face_b      ; 
            int tid_a        =  d_face_info(team.league_rank()).which_tree_a      ;
            int tid_b        =  d_face_info(team.league_rank()).which_tree_b      ;
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
            #ifndef GRACE_CARTESIAN_COORDINATES
            index_helper_t mapper{} ; 
            #endif 
            TeamThreadMDRange<Rank<GRACE_NSPACEDIM+1>,member_t>
                team_range( team, ngz, VECD(n1,n2), nvars) ; 
            parallel_for( team_range
                        , KOKKOS_LAMBDA(int& ig, VECD(int& j, int& k), int& ivar)
                    {
                    int i_a = EXPR((which_face_a==0) *ig 
                                + (which_face_a==1) * (nx+ngz+ig),
                                + (which_face_a/2==1) * (j+ngz), 
                                + (which_face_a/2==2) * (j+ngz)) ;

                    int j_a = EXPR((which_face_a==2) * ig 
                            + (which_face_a==3) * (ny+ngz+ig), 
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

                    #ifdef GRACE_3D
                    int k_a = (which_face_a==4) * ig 
                            + (which_face_a==5) *  (nz+ngz+ig)
                            + (which_face_a/2!=2) * (k+ngz) ;

                    int k_b = (which_face_b==4)*(ngz+ig)
                            + (which_face_b==5)*(nz+ig)
                            + (which_face_b/2!=2) * (k + ngz) ;
                    #endif 
                    #ifndef GRACE_CARTESIAN_COORDINATES
                    // TODO HERE  WE ASSUME Nx==Ny==Nz
                    auto const lmn = mapper({VEC(i_a,j_a,k_a)}, tid_b, tid_a, {ng,n1,n2}) ;
                    i_a = lmn[0]; j_a = lmn[1]; k_a = lmn[2] ; 
                    #endif 

                    view_a(VEC(i_a,j_a,k_a),ivar,qid_a) =  view_b(VEC(i_b,j_b,k_b),ivar,qid_b) ; 
                    
                    if( ! is_ghost ) {
                        i_b = EXPR((which_face_b==0) * ig 
                                + (which_face_b==1) * (nx+ngz+ig),
                                + (which_face_b/2==1) * (j+ngz), 
                                + (which_face_b/2==2) * (j+ngz)) ;

                        j_b = EXPR((which_face_b==2) * ig
                                + (which_face_b==3) * (ny+ngz+ig), 
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

                        #ifdef GRACE_3D
                        k_b =     (which_face_b==4) *ig 
                                + (which_face_b==5) * (nz+ngz+ig)
                                + (which_face_b/2!=2) * (k+ngz) ;

                        k_a =     (which_face_a==4)*(ngz+ig)
                                + (which_face_a==5)*(nz+ig)
                                + (which_face_a/2!=2) * (k + ngz) ;
                        #endif  
                        #ifndef GRACE_CARTESIAN_COORDINATES
                        // TODO HERE  WE ASSUME Nx==Ny==Nz
                        auto const lmn = mapper({VEC(i_b,j_b,k_b)}, tid_a, tid_b, {ng,n1,n2}) ;
                        i_b = lmn[0]; j_b = lmn[1]; k_b = lmn[2] ; 
                        #endif 
                        view_b(VEC(i_b,j_b,k_b),ivar,qid_b) =  view_a(VEC(i_a,j_a,k_a),ivar,qid_a) ;
                    }
                    } );
        }
    )   ;  
    #endif 
}

void restrict_hanging_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& halo_vols 
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
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
    if( n_faces == 0 ) {
        return ; 
    }
    auto& d_face_info = hanging_faces.d_view    ; 

    constexpr const int n_neighbors = PICK_D(2,4) ;  

    TeamPolicy<default_execution_space> 
        policy( n_faces, AUTO() ) ; 
    using member_t = decltype(policy)::member_type ;
    utils::vol_average_restictor_t restriction_kernel ; 
    /*************************************************/
    /* Kernel:                                       */
    /* Restrict data onto coarse quadrants from fine */
    /* neighboring quadrants.                        */
    /*************************************************/
    parallel_for
    (             GRACE_EXECUTION_TAG("AMR","restrict_hanging_faces")
                , policy 
                , KOKKOS_LAMBDA( const member_t& team )
        {
            int polarity     =  d_face_info(team.league_rank()).has_polarity_flip         ; 
            int8_t is_ghost_coarse   =  d_face_info(team.league_rank()).is_ghost_coarse   ; 
            int64_t qid_coarse       =  d_face_info(team.league_rank()).qid_coarse        ;
            int8_t which_face_coarse =  d_face_info(team.league_rank()).which_face_coarse    ; 
            int8_t which_face_fine   =  d_face_info(team.league_rank()).which_face_fine      ; 
            int tid_coarse = d_face_info(team.league_rank()).which_tree_coarse ; 
            int tid_fine   = d_face_info(team.league_rank()).which_tree_fine   ;
            int8_t is_ghost_fine[P4EST_CHILDREN/2] ; 
            int64_t qid_fine[P4EST_CHILDREN/2] ;
            for( int ii=0; ii<P4EST_CHILDREN/2; ++ ii){
                is_ghost_fine[ii] = d_face_info(team.league_rank()).is_ghost_fine[ii] ;
                qid_fine[ii]      = d_face_info(team.league_rank()).qid_fine[ii]      ; 
            } 
            int64_t const ng = (which_face_fine/2==0) * nx + ((which_face_fine/2==1) * ny) + ((which_face_fine/2==2) * nz) ;
            int64_t const n1 = (which_face_fine/2==0) * ny + ((which_face_fine/2==1) * nx) + ((which_face_fine/2==2) * nx) ;
            int64_t const n2 = (which_face_fine/2==0) * nz + ((which_face_fine/2==1) * nz) + ((which_face_fine/2==2) * ny) ;
            #ifdef GRACE_SPHERICAL_COORDINATES
            index_helper_t mapper{} ; 
            #endif 
            if( ! is_ghost_coarse )
            {
            TeamThreadMDRange<Rank<GRACE_NSPACEDIM>,member_t>
                team_range( team, VECD(n1,n2), nvars) ; 
            parallel_for( team_range
                        , KOKKOS_LAMBDA(VECD(int& j, int& k), int& ivar)
                    { 
                        const int8_t ichild = EXPRD(
                                math::floor_int((2*j)/n1)
                            , + math::floor_int((2*k)/n2) * 2
                            ) ; 
                        int64_t qid_b   = qid_fine[ichild] ; 
                        auto& fine_view = is_ghost_fine[ichild] ? halo : state ; 
                        auto& fine_vol  = is_ghost_fine[ichild] ? halo_vols : vols ; 
                        for(int ig=0; ig<ngz; ++ig){
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
                            int64_t const VEC( Ig{ (2*ig)%ng }, I1{ (2*j)%n1 + ngz }, I2{ (2*k)%n2 + ngz } ) ; 
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
                                utils::vol_average_restictor_t::apply(VEC(i_f,j_f,k_f), fine_view, fine_vol, qid_b, ivar) ;  
                        }
                    } );
            }
        }
        
    )   ;

}

template< typename InterpT > 
void prolongate_hanging_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& halo_vols 
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
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
    auto& d_face_info = hanging_faces.d_view    ; 
    if( n_faces == 0 ) {
        return ; 
    }
    constexpr const int n_neighbors = PICK_D(2,4) ;  

    TeamPolicy<default_execution_space> 
        policy( n_faces, AUTO() ) ; 
    using member_t = decltype(policy)::member_type ;
    
    /*************************************************/
    /* Kernel:                                       */
    /* Prolongate data onto fine quadrants ghost     */
    /* zones from coarse neighboring quadrants.      */
    /*************************************************/
    parallel_for
    (             GRACE_EXECUTION_TAG("AMR","prolongate_hanging_faces")
                , policy 
                , KOKKOS_LAMBDA( const member_t& team )
        {
            int polarity     =  d_face_info(team.league_rank()).has_polarity_flip         ; 
            int8_t is_ghost_coarse   =  d_face_info(team.league_rank()).is_ghost_coarse   ; 
            int64_t iq_coarse        =  d_face_info(team.league_rank()).qid_coarse        ;
            int8_t which_face_coarse =  d_face_info(team.league_rank()).which_face_coarse    ; 
            int8_t which_face_fine   =  d_face_info(team.league_rank()).which_face_fine      ; 
            int tid_coarse           =  d_face_info(team.league_rank()).which_tree_coarse    ; 
            int tid_fine             =  d_face_info(team.league_rank()).which_tree_fine      ; 
            int8_t is_ghost_fine[P4EST_CHILDREN/2] ; 
            int64_t qid_fine[P4EST_CHILDREN/2] ;
            for( int ii=0; ii<P4EST_CHILDREN/2; ++ ii){
                is_ghost_fine[ii] = d_face_info(team.league_rank()).is_ghost_fine[ii] ;
                qid_fine[ii]      = d_face_info(team.league_rank()).qid_fine[ii]      ; 

            }  
            
            int64_t n1 = (which_face_fine/2==0) * ny + ((which_face_fine/2==1) * nx) + ((which_face_fine/2==2) * nx) ;
            int64_t n2 = (which_face_fine/2==0) * nz + ((which_face_fine/2==1) * nz) + ((which_face_fine/2==2) * ny) ;

            auto& coarse_view   = is_ghost_coarse ? halo : state ; 

            #ifndef GRACE_CARTESIAN_COORDINATES
            index_helper_t mapper{} ;
            #endif 
            TeamThreadMDRange<Rank<GRACE_NSPACEDIM+1>,member_t>
                team_range( team, VEC(ngz, n1,n2), nvars) ; 
            parallel_for( team_range
                        , KOKKOS_LAMBDA(VEC(int& ig, int& j, int& k), int& ivar)
                    { 
                        /* First we compute the indices of the point */
                        /* we are calculating.                       */
                        EXPR( 
                        int const i_f = EXPR((which_face_fine==0) * ig 
                                    + (which_face_fine==1) * (nx+ngz+ig),
                                    + (which_face_fine/2==1) * (j+ngz), 
                                    + (which_face_fine/2==2) * (j+ngz)) ;,
                        int const j_f = EXPR((which_face_fine==2) * ig
                                    + (which_face_fine==3) * (ny+ngz+ig), 
                                    + (which_face_fine/2==0) * (j+ngz), 
                                    + (which_face_fine/2==2) * (k+ngz));  ,
                        int const k_f = (which_face_fine==4) * ig 
                                    + (which_face_fine==5) * (nz+ngz+ig)  
                                    + (which_face_fine/2!=2) * (k+ngz) ;
                        )
                        /* Then we loop over all child quadrants */
                        /* and call the prolongation kernel.     */
                        #pragma unroll 4
                        for( int ichild=0; ichild<GRACE_FACE_CHILDREN; ++ichild) {
                            if( is_ghost_fine[ichild] ) continue ; 
                            /* WARNING! No relative orientation assumed here */
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
                            int const VECD( I1{ math::floor_int((iquad_1 * n1 + j ) / 2) + ngz }
                                          , I2{ math::floor_int((iquad_2 * n2 + k ) / 2) + ngz } ) ; 
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
                            state(VEC(i_f,j_f,k_f), ivar, iq_fine)
                                = InterpT::interpolate( VEC(i_f,j_f,k_f)
                                                      , VEC(i_c,j_c,k_c)
                                                      , iq_fine, iq_coarse, ngz, ivar
                                                      , VEC(sign_x, sign_y, sign_z)
                                                      , coarse_view 
                                                      , vols ) ; 
                        }  
                    } );

        }
    )   ;

}

template void 
prolongate_hanging_ghostzones<utils::linear_prolongator_t<grace::minmod>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&  
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
) ; 
template void 
prolongate_hanging_ghostzones<utils::linear_prolongator_t<grace::MCbeta>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&   
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
) ; 

}} /* namespace grace::amr */
