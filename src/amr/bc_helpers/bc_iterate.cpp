/**
 * @file bc_iterate.cpp
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

#include <grace/amr/bc_iterate.hh>
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
    auto& face_info = reinterpret_cast<grace_neighbor_info_t*>(user_data)->face_info ; 
    sc_array_view_t<p4est_iter_face_side_t> sides{
        &(info->sides)
    } ; 
    auto& simple_info    = face_info.simple_interior_info       ;
    auto& hanging_info   = face_info.hanging_interior_info      ;
    auto& physical_boundary_info = face_info.phys_boundary_info ;
    /*************************************************************/
    /* This is a vector storing all the unique hanging faces     */
    /* in the local forest. It does so in the format             */
    /* hanging_faces_info[ P4EST_FACES * iquad + iface ] = 0 / 1 */
    /*************************************************************/
    auto& coarse_hanging_info = 
        reinterpret_cast<grace_neighbor_info_t*>(user_data)->coarse_hanging_quads_info ;
    /**************************************************/
    /* This means we are at a physical boundary       */
    /* we store the index in user_info and return     */
    /* since physical boundary conditions are handled */
    /* separately.                                    */
    /**************************************************/
    if( sides.size() == 1 ) { 
        size_t offset = amr::get_local_quadrants_offset(sides[0].treeid);
        auto const fill_bc_info = [&] (int64_t qid, int8_t iface, grace_phys_bc_info_t& info)
        {
            info.qid = offset+qid ;
            info.dir_x = (iface == 0 ) ? -1 
                                       : (iface==1 ? +1 : 0) ; 
            info.dir_y = (iface == 2 ) ? -1 
                                       : (iface==3 ? +1 : 0) ;
            info.dir_z = (iface == 4 ) ? -1 
                                       : (iface==5 ? +1 : 0) ;
            info.face = iface ; 

        } ; 
        if( sides[0].is_hanging ) {
            for( int ic=0; ic<P4EST_HALF; ++ic) {
                grace_phys_bc_info_t this_face_info{} ; 
                fill_bc_info(sides[0].is.hanging.quadid[ic], sides[0].face, this_face_info) ; 
                physical_boundary_info.push_back(this_face_info) ; 
            }
        } else {
            grace_phys_bc_info_t this_face_info{} ; 
            fill_bc_info(sides[0].is.full.quadid, sides[0].face, this_face_info) ;
            physical_boundary_info.push_back(this_face_info); 
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
    /* If the first side is hanging */
    if( sides[1].is_hanging )
    {
        /* Collect quadrant information */
        hanging_face_info_t this_face_info{} ; 
        this_face_info.has_polarity_flip = polarity_flip ; 
        this_face_info.level_coarse = static_cast<int>(sides[0].is.full.quad->level) ;
        this_face_info.level_fine   = this_face_info.level_coarse + 1 ;
        this_face_info.which_face_coarse = sides[0].face ;
        this_face_info.which_face_fine   = sides[1].face ; 
        this_face_info.is_ghost_coarse   = sides[0].is.full.is_ghost ; 
        this_face_info.qid_coarse        = sides[0].is.full.quadid   
            + ( this_face_info.is_ghost_coarse ? 0 : get_local_quadrants_offset(sides[0].treeid)  ) ; 
        
        /* Collect fine quadrant information */
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
            /* If the coarse quadrant is not local, we add the coarse quad */
            /* to the list of those that have to be received after the     */               
            /* restriction.                                                */
            auto halos = info->ghost_layer ; 
            coarse_hanging_info.rcv_quadid.push_back( this_face_info.qid_coarse ) ;
            for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                size_t first_halo  = halos->proc_offsets[iproc]   ; 
                size_t last_halo   = halos->proc_offsets[iproc+1] ;
                if( this_face_info.qid_coarse >= first_halo and this_face_info.qid_coarse < last_halo ) {
                    coarse_hanging_info.rcv_procid.push_back( iproc ) ; 
                }
            }
            face_info.n_hanging_ghost_faces ++ ;
        } else if (  any_fine_ghost ) {
            /* If any of the fine quadrants is not local, we add the coarse */
            /* quadrant to the list of those that have to be received after */
            /* restriction.                                                 */
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
            face_info.n_hanging_ghost_faces ++ ;
        }
        GRACE_VERBOSE("Hanging face, quadid_coarse {} face_coarse {}, quadid_fine [ {} {} {} {} ] face fine {}",
                     static_cast<int>(this_face_info.qid_coarse), static_cast<int>(this_face_info.which_face_coarse), 
                     static_cast<int>(this_face_info.qid_fine[0]), static_cast<int>(this_face_info.qid_fine[1]),
                     static_cast<int>(this_face_info.qid_fine[2]), static_cast<int>(this_face_info.qid_fine[3]),  static_cast<int>(this_face_info.which_face_fine)) ;
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
            /* If the coarse quadrant is not local, we add the coarse quad */
            /* to the list of those that have to be received after the     */               
            /* restriction.                                                */
            auto halos = info->ghost_layer ; 
            coarse_hanging_info.rcv_quadid.push_back( this_face_info.qid_coarse ) ;
            for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                size_t first_halo  = halos->proc_offsets[iproc]   ; 
                size_t last_halo   = halos->proc_offsets[iproc+1] ;
                if( this_face_info.qid_coarse >= first_halo and this_face_info.qid_coarse < last_halo ) {
                    coarse_hanging_info.rcv_procid.push_back( iproc ) ; 
                }
            }
            face_info.n_hanging_ghost_faces ++ ;  
        } else if (  any_fine_ghost ) {
            /* If any of the fine quadrants is not local, we add the coarse */
            /* quadrant to the list of those that have to be received after */
            /* restriction.                                                 */
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
            face_info.n_hanging_ghost_faces ++ ;
        }
        GRACE_VERBOSE("Hanging face, quadid_coarse {} face_coarse {}, quadid_fine [ {} {} {} {} ] face fine {}",
                     static_cast<int>(this_face_info.qid_coarse), static_cast<int>(this_face_info.which_face_coarse), 
                     static_cast<int>(this_face_info.qid_fine[0]), static_cast<int>(this_face_info.qid_fine[1]),
                     static_cast<int>(this_face_info.qid_fine[2]), static_cast<int>(this_face_info.qid_fine[3]),  static_cast<int>(this_face_info.which_face_fine)) ;
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
            face_info.n_simple_ghost_faces ++ ; 
            GRACE_TRACE("Ghost face qid_a {} qid_b {}", this_face_info.qid_a, this_face_info.qid_b) ; 
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
            face_info.n_simple_ghost_faces ++ ; 
            GRACE_TRACE("Ghost face qid_a {} qid_b {}", this_face_info.qid_a, this_face_info.qid_b) ; 
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


void grace_iterate_edges( p8est_iter_edge_info_t * info 
                        , void* user_data  )
{
    using namespace grace; 
    auto& edge_info = reinterpret_cast<grace_neighbor_info_t*>(user_data)->edge_info ;  
    sc_array_view_t<p8est_iter_edge_side_t> sides{
        &(info->sides)
    } ; 
    /* Physical boundaries are handled by face neighbors no need to store it here. */
    auto& simple_info            = edge_info.simple_interior_info      ;
    auto& hanging_info           = edge_info.hanging_interior_info     ;
    auto& physical_boundary_info = edge_info.phys_boundary_info     ; 
    auto& coarse_hanging_info = 
        reinterpret_cast<grace_neighbor_info_t*>(user_data)->coarse_hanging_quads_info ; 
    /**************************************************/
    /* This means we are at a physical boundary       */
    /* we store the index in user_info and return     */
    /* since physical boundary conditions are handled */
    /* separately.                                    */
    /**************************************************/
    edge_info.n_edges_total ++ ; 
    if( sides.size() < 4 ) { 
        /*************************************************************/
        /* This section of code performs the following operations    */
        /* For each side of the edge (meaning each quadrant touching */
        /* this edge):                                               */
        /*   i) Check if the quadrant(s) is(are) local or ghost      */
        /*  ii) If local, checks whether any of the faces touching   */
        /*      the edge are hanging.                                */
        /* iii) If hanging, store qid and edgeid info in the hanging */
        /*      phys_bc vector, else in the simple_phys_bc vector    */
        /*************************************************************/
        auto nsides = sides.size() ; 
        auto fill_bc_info_edge = [&] (int64_t const qid, int8_t edge, int64_t offset, grace_phys_bc_info_t& info)
        {
            /* here -1 corresponds to a negative edge, 0 to a along and +1 to a positive*/
            static constexpr const int ALONG_EDGE    =  0 ;
            static constexpr const int NEGATIVE_EDGE = -1 ;
            static constexpr const int POSITIVE_EDGE = +1 ;
            static constexpr const int8_t edges_directions[3][P8EST_EDGES] = {
                {ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, 
                    NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE,
                    NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE},  // x directions
                {NEGATIVE_EDGE, POSITIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, 
                    ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE,
                    NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE}, // y directions
                {NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE, 
                    NEGATIVE_EDGE, NEGATIVE_EDGE, POSITIVE_EDGE, POSITIVE_EDGE, 
                    ALONG_EDGE, ALONG_EDGE, ALONG_EDGE, ALONG_EDGE} // z directions
            } ; 
            info.qid = qid + offset ; 
            info.dir_x = edges_directions[0][edge] ; 
            info.dir_y = edges_directions[1][edge] ;
            info.dir_z = edges_directions[2][edge] ; 
        } ; 
        /* This means that we are on the grid edge */
        /* We store this information as simple     */
        /* phys boundary info                      */
        if ( nsides == 1 ) {
            int64_t const offset = amr::get_local_quadrants_offset(sides[0].treeid);
            if( sides[0].is_hanging ) {
                for( int ic=0; ic<2; ++ic) {
                    if( sides[0].is.hanging.is_ghost[ic])
                        continue ;
                    grace_phys_bc_info_t this_edge_info {} ;
                    fill_bc_info_edge(sides[0].is.hanging.quadid[ic], sides[0].edge, offset, this_edge_info) ; 
                    physical_boundary_info.push_back(this_edge_info) ;
                }
            } else {
                grace_phys_bc_info_t this_edge_info {} ;
                fill_bc_info_edge(sides[0].is.full.quadid, sides[0].edge, offset, this_edge_info) ; 
                physical_boundary_info.push_back(this_edge_info) ;
            }
        }
        edge_info.n_exterior_edges ++ ; 
        return ; 
    }
    ASSERT(  sides.size() == 4
           , "Something went wrong in iter_edges, sides.size() != 4.") ; 
    /***************************************************/
    /* Group quadrants in 2 pairs of true edge         */
    /* neighbors by checking which of them share a face*/
    /***************************************************/
    int side_idx_pairs[2][2] ; 
    auto is_edge_neighbor = [&](int i, int j) {
        for (int iff = 0; iff < 2; ++iff) {
            for (int jff = 0; jff < 2; ++jff) {
                if (sides[i].faces[iff] == sides[j].faces[jff] ) {
                    return false;
                }
            }
        }
        return true;
    };
    std::vector<int> found(4, 0);
    int ip = 0;
    for(int in=0; in<4; ++in){
        if (found[in]) continue;
        for (int jn = in + 1; jn < 4; ++jn) { 
            if (found[jn]) continue;
            if( is_edge_neighbor(in,jn) ) {
                side_idx_pairs[ip][0] = in ;
                side_idx_pairs[ip][1] = jn ;
                found[in] = 1 ; 
                found[jn] = 1 ;  
                ip++ ; 
                break ;
            }
        }
    } 
    /***************************************************/
    /* Now we have two pairs of "sides" of the edge    */
    /* that do not share a face. We will store info    */
    /* about these two "sides" (whether they are full) */
    /* or hanging, the quadid's involved on either side*/
    /* etc. in the following.                          */
    /***************************************************/
    for( int i=0 ; i<2; ++i) {
        auto const& sidea = sides[side_idx_pairs[i][0]] ; 
        auto const& sideb = sides[side_idx_pairs[i][1]] ; 
        if ( sidea.is_hanging and sideb.is_hanging) {
            /* This tricky situation arises where an edge */
            /* faces 1 coarse quadrant and 3 fine ones.   */
            /* in the following we assume that the z-order*/
            /* is the same on either side of this edge.   */
            #pragma unroll 2
            for( int ic=0; ic<2; ++ic) {
                simple_edge_info_t this_edge_info {} ; 
                if( sidea.is.hanging.is_ghost[ic]) {
                    auto offset = get_local_quadrants_offset(sideb.treeid);
                    this_edge_info.is_ghost     = 1 ; 
                    this_edge_info.which_edge_a = sideb.edge ; 
                    this_edge_info.which_edge_b = sidea.edge ;
                    this_edge_info.which_tree_a = sideb.treeid ; 
                    this_edge_info.which_tree_b = sidea.treeid ; 
                    this_edge_info.qid_a = sideb.is.hanging.quadid[ic]
                        + offset ;
                    this_edge_info.qid_b = sidea.is.hanging.quadid[ic] ;
                    edge_info.n_simple_ghost_edges ++ ; 
                    
                } else if(sideb.is.hanging.is_ghost[ic]){
                    auto offset = get_local_quadrants_offset(sidea.treeid);
                    this_edge_info.is_ghost     = 1 ; 
                    this_edge_info.which_edge_a = sidea.edge ; 
                    this_edge_info.which_edge_b = sideb.edge ;
                    this_edge_info.which_tree_a = sidea.treeid ; 
                    this_edge_info.which_tree_b = sideb.treeid ; 
                    this_edge_info.qid_a = sidea.is.hanging.quadid[ic] 
                        + offset ;
                    this_edge_info.qid_b = sideb.is.hanging.quadid[ic] ;
                    edge_info.n_simple_ghost_edges ++ ; 
                    
                } else {
                    auto offset = get_local_quadrants_offset(sidea.treeid);
                    this_edge_info.is_ghost = 0 ; 
                    this_edge_info.which_edge_a = sidea.edge ; 
                    this_edge_info.which_edge_b = sideb.edge ;
                    this_edge_info.which_tree_a = sidea.treeid ; 
                    this_edge_info.which_tree_b = sideb.treeid ; 
                    this_edge_info.qid_a = sidea.is.hanging.quadid[ic]
                        + offset ;
                    offset = get_local_quadrants_offset(sideb.treeid);
                    this_edge_info.qid_b = sideb.is.hanging.quadid[ic]
                        + offset ;
                    
                }
                simple_info.push_back(this_edge_info) ;
                GRACE_VERBOSE("Double simple edge, quadid_a {} edge_a {}, quadid_b {}  edge_b {}",
                     this_edge_info.qid_a, this_edge_info.which_edge_a, 
                     this_edge_info.qid_b, this_edge_info.which_edge_b) ;
            }
             
        } else if ( sidea.is_hanging ) {
            if ( sideb.is.full.quadid < 0 or sideb.is.full.quadid > amr::get_local_num_quadrants() ) {
                GRACE_WARN("A: qid < 0! edge_code {} id {} is_hanging {} is_ghost {} ", sideb.edge, sideb.is.full.quadid, sideb.is_hanging,  sideb.is.full.is_ghost) ; 
            }
            hanging_edge_info_t this_edge_info {} ;  
            this_edge_info.level_coarse = static_cast<int>(sideb.is.full.quad->level) ; 
            this_edge_info.level_fine   = this_edge_info.level_coarse + 1;
            this_edge_info.which_edge_coarse = sideb.edge ; 
            this_edge_info.which_edge_fine   = sidea.edge ; 
            this_edge_info.is_ghost_coarse   = sideb.is.full.is_ghost ; 
            this_edge_info.qid_coarse        = sideb.is.full.quadid
                + (this_edge_info.is_ghost_coarse ? 0 : get_local_quadrants_offset(sideb.treeid)) ; 
            int any_fine_ghost{0} ; 
            for( int ii=0; ii<2; ++ii) {
                this_edge_info.is_ghost_fine[ii] = sidea.is.hanging.is_ghost[ii] ; 
                this_edge_info.qid_fine[ii] = sidea.is.hanging.quadid[ii]
                    + (this_edge_info.is_ghost_fine[ii] ? 0 : get_local_quadrants_offset(sidea.treeid)) ;
                any_fine_ghost += this_edge_info.is_ghost_fine[ii] ;
            } 
            GRACE_VERBOSE("Hanging edge (A), quadid_coarse {} edge_coarse {}, quadid_fine [ {} {} ] is_ghost_fine [ {} {} ] edge fine {}",
                     this_edge_info.qid_coarse, this_edge_info.which_edge_coarse, 
                     this_edge_info.qid_fine[0], this_edge_info.qid_fine[1], this_edge_info.is_ghost_fine[0], this_edge_info.is_ghost_fine[1], this_edge_info.which_edge_fine) ; 
            hanging_info.push_back(this_edge_info) ; 
        } else if (sideb.is_hanging) {
            hanging_edge_info_t this_edge_info {} ;  
            if ( sidea.is.full.quadid < 0 or sidea.is.full.quadid > amr::get_local_num_quadrants() ) {
                GRACE_WARN("B qid < 0! edge_code {} id {} is_hanging {} is_ghost {} ", sidea.edge, sidea.is.full.quadid, sidea.is_hanging, sidea.is.full.is_ghost) ; 
            }
            this_edge_info.level_coarse = static_cast<int>(sidea.is.full.quad->level) ; 
            this_edge_info.level_fine   = this_edge_info.level_coarse + 1;
            this_edge_info.which_edge_coarse = sidea.edge ; 
            this_edge_info.which_edge_fine   = sideb.edge ; 
            this_edge_info.is_ghost_coarse   = sidea.is.full.is_ghost ; 
            this_edge_info.qid_coarse        = sidea.is.full.quadid
                + (this_edge_info.is_ghost_coarse ? 0 : get_local_quadrants_offset(sidea.treeid)) ; 
            int any_fine_ghost{0} ;
            for( int ii=0; ii<2; ++ii) {
                this_edge_info.is_ghost_fine[ii] = sideb.is.hanging.is_ghost[ii] ; 
                this_edge_info.qid_fine[ii] = sideb.is.hanging.quadid[ii]
                    + (this_edge_info.is_ghost_fine[ii] ? 0 : get_local_quadrants_offset(sideb.treeid)) ; 
                any_fine_ghost += this_edge_info.is_ghost_fine[ii];
            }
            GRACE_VERBOSE("Hanging edge (B), quadid_coarse {} edge_coarse {}, quadid_fine [ {} {} ] is_ghost_fine [ {} {} ] edge fine {}",
                     this_edge_info.qid_coarse, this_edge_info.which_edge_coarse, 
                     this_edge_info.qid_fine[0], this_edge_info.qid_fine[1], 
                     this_edge_info.is_ghost_fine[0], this_edge_info.is_ghost_fine[1], 
                     this_edge_info.which_edge_fine) ; 
            hanging_info.push_back(this_edge_info) ; 
        } else {
            simple_edge_info_t this_edge_info {} ; 
            if( sidea.is.full.is_ghost and not sideb.is.full.is_ghost) {
                auto offset = get_local_quadrants_offset(sideb.treeid);
                this_edge_info.is_ghost     = 1 ; 
                this_edge_info.which_edge_a = sideb.edge ; 
                this_edge_info.which_edge_b = sidea.edge ;
                this_edge_info.which_tree_a = sideb.treeid ; 
                this_edge_info.which_tree_b = sidea.treeid ; 
                this_edge_info.qid_a = sideb.is.full.quadid 
                    + offset ;
                this_edge_info.qid_b = sidea.is.full.quadid ;
                edge_info.n_simple_ghost_edges ++ ; 
                GRACE_TRACE("Ghost edge A qid_a {} qid_b {}", this_edge_info.qid_a, this_edge_info.qid_b) ; 
            } else if(sideb.is.full.is_ghost and not sidea.is.full.is_ghost){
                auto offset = get_local_quadrants_offset(sidea.treeid);
                this_edge_info.is_ghost     = 1 ; 
                this_edge_info.which_edge_a = sidea.edge ; 
                this_edge_info.which_edge_b = sideb.edge ;
                this_edge_info.which_tree_a = sidea.treeid ; 
                this_edge_info.which_tree_b = sideb.treeid ; 
                this_edge_info.qid_a = sidea.is.full.quadid 
                    + offset ;
                this_edge_info.qid_b = sideb.is.full.quadid ;
                edge_info.n_simple_ghost_edges ++ ; 
                GRACE_TRACE("Ghost edge B qid_a {} qid_b {}", this_edge_info.qid_a, this_edge_info.qid_b) ; 
            } else {
                auto offset = get_local_quadrants_offset(sidea.treeid);
                this_edge_info.is_ghost = 0 ; 
                this_edge_info.which_edge_a = sidea.edge ; 
                this_edge_info.which_edge_b = sideb.edge ;
                this_edge_info.which_tree_a = sidea.treeid ; 
                this_edge_info.which_tree_b = sideb.treeid ; 
                this_edge_info.qid_a = sidea.is.full.quadid 
                    + offset ;
                offset = get_local_quadrants_offset(sideb.treeid);
                this_edge_info.qid_b = sideb.is.full.quadid
                    + offset ;
            }
            simple_info.push_back(this_edge_info) ; 
        }
    }

}


void grace_iterate_corners( p4est_iter_corner_info_t * info 
                          , void* user_data  )
{
    using namespace grace; 
    auto& corner_info = reinterpret_cast<grace_neighbor_info_t*>(user_data)->corner_info ;  
    sc_array_view_t<p4est_iter_corner_side_t> sides{
        &(info->sides)
    } ; 
    /* Physical boundaries are handled by face neighbors no need to store it here. */
    auto& simple_info         = corner_info.simple_interior_info         ;
    auto& hanging_info        = corner_info.hanging_interior_info        ;
    #if 1
    auto& physical_boundary_info = corner_info.phys_boundary_info     ; 
    #endif 
    auto& coarse_hanging_info = 
        reinterpret_cast<grace_neighbor_info_t*>(user_data)->coarse_hanging_quads_info ; 
    /**************************************************/
    /* This means we are at a physical boundary       */
    /* we store the index in user_info and return     */
    /* since physical boundary conditions are handled */
    /* separately.                                    */
    /**************************************************/
    if( sides.size() < P4EST_CHILDREN ) { 
        /**********************************************************************/
        /* First we determine if any faces or edges sharing this corner are   */
        /* hanging. Reasons to check this:                                    */
        /*   i) Hanging edges indicate that boundary conditions (BCs) depend  */
        /*      on ghost-zones filled by prolongation at the end. If no edges */
        /*      are hanging, BCs must be filled before the second coarse halo */
        /*      transfer to ensure all ghost-zones are populated.             */
        /*  ii) Hanging faces signify an interior face. Additionally, there   */
        /*      will be another corner connected to this face that is not a   */
        /*      corner for the neighboring quadrant. This requires additional */
        /*      information about that corner.                                */
        /* Once determined, if the quadrant is local, store its info in the   */
        /* appropriate vector: (P4EST_CHILDREN * qid + icorner). Use          */
        /* simple_bc_info for case i) and hanging_bc_info for case ii).       */
        /**********************************************************************/
        auto nsides = sides.size() ;
        auto fill_bc_info = [&] (int64_t qid, int8_t corner, grace_phys_bc_info_t& info)
        {
            info.qid = qid ; 
            info.dir_x = (((corner >> 0) & 1) == 0) ? -1 : +1 ; 
            info.dir_y = (((corner >> 1) & 1) == 0) ? -1 : +1 ; 
            info.dir_z = (((corner >> 2) & 1) == 0) ? -1 : +1 ;
        } ; 
        /* This means that we are on the grid edge */
        if ( nsides == 1 ) {
            int64_t qid = sides[0].quadid 
                + amr::get_local_quadrants_offset(sides[0].treeid);
            int8_t corner = sides[0].corner ; 
            grace_phys_bc_info_t info {} ; 
            fill_bc_info(qid,corner,info) ; 
            physical_boundary_info.push_back(
                info
            ) ; 
        }        
        // Since this was a physical boundary we return now.
        return ; 
    }
    ASSERT(  sides.size() == P4EST_CHILDREN
           , "Something went wrong in iter_corners, sides.size() != P4EST_CHILDREN.") ; 
    /***************************************************/
    /* Group quadrants in 4 pairs of true corner       */
    /* neighbors by checking which of them share a face*/
    /* or an edge.                                     */
    /***************************************************/
    int side_idx_pairs[4][2] ;
    auto is_corner_neighbor = [&](int i, int j) {
        for (int iff = 0; iff < 3; ++iff) {
            for (int jff = 0; jff < 3; ++jff) {
                if (sides[i].faces[iff] == sides[j].faces[jff] ||
                    sides[i].edges[iff] == sides[j].edges[jff]) {
                    return false;
                }
            }
        }
        return true;
    };
    std::vector<bool> found(8, false);
    int ip = 0;
    
    for(int in=0; in<8; ++in){
        if (found[in]) continue;
        for (int jn = in + 1; jn < 8; ++jn) { 
            if (found[jn]) continue;
            if( is_corner_neighbor(in,jn) and not (in==jn)) {
                side_idx_pairs[ip][0] = in ;
                side_idx_pairs[ip][1] = jn ;
                found[in] = true ; 
                found[jn] = true ;  
                ip++ ; 
                break ;
            }
        }
    } 
    
    /***************************************************/
    /* Now we have two pairs of "sides" of the corner    */
    /* that do not share a face. We will store info    */
    /* about these two "sides" (whether they are full) */
    /* or hanging, the quadid's involved on either side*/
    /* etc. in the following.                          */
    /***************************************************/
    for( int i=0 ; i<4; ++i) {
        auto const& sidea = sides[side_idx_pairs[i][0]] ; 
        auto const& sideb = sides[side_idx_pairs[i][1]] ; 
        int level_a = static_cast<int>(sidea.quad->level) ; 
        int level_b = static_cast<int>(sideb.quad->level) ; 
        if ( level_a > level_b ) {
            hanging_corner_info_t this_corner_info {} ;  
            this_corner_info.level_coarse = level_b ; 
            this_corner_info.level_fine   = this_corner_info.level_coarse + 1;
            this_corner_info.which_corner_coarse = sideb.corner ; 
            this_corner_info.which_corner_fine   = sidea.corner ; 
            this_corner_info.is_ghost_coarse   = sideb.is_ghost ; 
            this_corner_info.qid_coarse        = sideb.quadid
                + (this_corner_info.is_ghost_coarse ? 0 : get_local_quadrants_offset(sideb.treeid)) ; 
            this_corner_info.is_ghost_fine = sidea.is_ghost ; 
            this_corner_info.qid_fine = sidea.quadid
                + (this_corner_info.is_ghost_fine ? 0 : get_local_quadrants_offset(sidea.treeid)) ; 
            hanging_info.push_back(this_corner_info) ; 
            GRACE_VERBOSE("Hanging corner, quadid_coarse {} corner_coarse {}, quadid_fine {} corner fine {}",
                     this_corner_info.qid_coarse, this_corner_info.which_corner_coarse, this_corner_info.qid_fine, this_corner_info.which_corner_fine) ; 
        } else if (level_a < level_b) {
            hanging_corner_info_t this_corner_info {} ;  
            this_corner_info.level_coarse = level_a ; 
            this_corner_info.level_fine   = this_corner_info.level_coarse + 1;
            this_corner_info.which_corner_coarse = sidea.corner ; 
            this_corner_info.which_corner_fine   = sideb.corner ; 
            this_corner_info.is_ghost_coarse   = sidea.is_ghost ; 
            this_corner_info.qid_coarse        = sidea.quadid
                + (this_corner_info.is_ghost_coarse ? 0 : get_local_quadrants_offset(sidea.treeid)) ; 
            this_corner_info.is_ghost_fine = sideb.is_ghost ; 
            this_corner_info.qid_fine = sideb.quadid
                + (this_corner_info.is_ghost_fine ? 0 : get_local_quadrants_offset(sideb.treeid)) ;
            GRACE_VERBOSE("Hanging corner, quadid_coarse {} corner_coarse {}, quadid_fine {} corner fine {}",
                     this_corner_info.qid_coarse, this_corner_info.which_corner_coarse, this_corner_info.qid_fine, this_corner_info.which_corner_fine) ; 
            hanging_info.push_back(this_corner_info) ; 
        } else {
            simple_corner_info_t this_corner_info {} ; 
            if( sidea.is_ghost and not sideb.is_ghost) {
                auto offset = get_local_quadrants_offset(sideb.treeid);
                this_corner_info.is_ghost     = 1 ; 
                this_corner_info.which_corner_a = sideb.corner ; 
                this_corner_info.which_corner_b = sidea.corner ;
                this_corner_info.which_tree_a = sideb.treeid ; 
                this_corner_info.which_tree_b = sidea.treeid ; 
                this_corner_info.qid_a = sideb.quadid 
                    + offset ;
                this_corner_info.qid_b = sidea.quadid ;
                corner_info.n_simple_ghost_corners ++ ;
                GRACE_TRACE("Ghost corner A qid_a {} qid_b {}", this_corner_info.qid_a, this_corner_info.qid_b) ;  
            } else if(sideb.is_ghost and not sidea.is_ghost){
                auto offset = get_local_quadrants_offset(sidea.treeid);
                this_corner_info.is_ghost     = 1 ; 
                this_corner_info.which_corner_a = sidea.corner ; 
                this_corner_info.which_corner_b = sideb.corner ;
                this_corner_info.which_tree_a = sidea.treeid ; 
                this_corner_info.which_tree_b = sideb.treeid ; 
                this_corner_info.qid_a = sidea.quadid 
                    + offset ;
                this_corner_info.qid_b = sideb.quadid ;
                corner_info.n_simple_ghost_corners ++ ; 
                GRACE_TRACE("Ghost corner B qid_a {} qid_b {}", this_corner_info.qid_a, this_corner_info.qid_b) ;  
            } else {
                auto offset = get_local_quadrants_offset(sidea.treeid);
                this_corner_info.is_ghost = 0 ; 
                this_corner_info.which_corner_a = sidea.corner ; 
                this_corner_info.which_corner_b = sideb.corner ;
                this_corner_info.which_tree_a = sidea.treeid ; 
                this_corner_info.which_tree_b = sideb.treeid ; 
                this_corner_info.qid_a = sidea.quadid 
                + offset ;
                offset = get_local_quadrants_offset(sideb.treeid);
                this_corner_info.qid_b = sideb.quadid
                + offset ;
            }
            simple_info.push_back(this_corner_info) ; 
        }
    }

}


}} /* namespace grace::amr */