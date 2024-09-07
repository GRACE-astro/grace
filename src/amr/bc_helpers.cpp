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
    auto& face_info = reinterpret_cast<grace_neighbor_info_t*>(user_data)->face_info ; 
    sc_array_view_t<p4est_iter_face_side_t> sides{
        &(info->sides)
    } ; 
    auto& physical_boundary_info = face_info.phys_boundary_info ; 
    auto& simple_info    = face_info.simple_interior_info       ;
    auto& hanging_info   = face_info.hanging_interior_info      ;
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
    auto& simple_info         = edge_info.simple_interior_info      ;
    auto& hanging_info        = edge_info.hanging_interior_info     ;
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
        edge_info.n_exterior_edges ++ ; 
        return ; 
    }
    ASSERT(  sides.size() == 4
           , "Something went wrong in iter_edges, sides.size() != 4.") ; 
    /***************************************************/
    /* Group quadrants in 2 pairs of true edge         */
    /* neighbors by checking which of them share a face*/
    /***************************************************/
    //! TODO maybe we want these in z-order ?
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
            #if 0 
            if( this_edge_info.is_ghost_coarse ) {
                /* If the coarse quadrant is not local, we add the coarse quad */
                /* to the list of those that have to be received after the     */               
                /* restriction.                                                */ 
                auto halos = info->ghost_layer ; 
                coarse_hanging_info.rcv_quadid.push_back( this_edge_info.qid_coarse ) ;
                for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                    size_t first_halo  = halos->proc_offsets[iproc]   ; 
                    size_t last_halo   = halos->proc_offsets[iproc+1] ;
                    if( this_edge_info.qid_coarse >= first_halo and this_edge_info.qid_coarse < last_halo ) {
                        coarse_hanging_info.rcv_procid.push_back( iproc ) ; 
                    }
                }
                edge_info.n_hanging_ghost_edges ++ ;
            } else if (  any_fine_ghost ) {
                /* If any of the fine quadrants is not local, we add the coarse */
                /* quadrant to the list of those that have to be received after */
                /* restriction.                                                 */
                auto halos = info->ghost_layer ; 
                sc_array_view_t<p4est_quadrant_t>  mirror_quads { &(halos->mirrors) } ; 
                coarse_hanging_info.snd_quadid.push_back(this_edge_info.qid_coarse) ; 
                std::set<int> _snd_procid;
                for( int ii=0; ii<2; ++ii){
                    if( this_edge_info.is_ghost_fine[ii] ) {
                        for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                            size_t first_halo  = halos->proc_offsets[iproc]   ; 
                            size_t last_halo   = halos->proc_offsets[iproc+1] ;
                            if( this_edge_info.qid_fine[ii] >= first_halo and this_edge_info.qid_fine[ii] < last_halo ) {
                                _snd_procid.insert( iproc ) ; 
                            }
                        }
                    }
                }
                coarse_hanging_info.snd_procid.push_back(_snd_procid) ; 
                edge_info.n_hanging_ghost_edges ++ ;
            }
            #endif 
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
            #if 0 
            if( this_edge_info.is_ghost_coarse ) {
                /* If the coarse quadrant is not local, we add the coarse quad */
                /* to the list of those that have to be received after the     */               
                /* restriction.                                                */ 
                auto halos = info->ghost_layer ; 
                coarse_hanging_info.rcv_quadid.push_back( this_edge_info.qid_coarse ) ;
                for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                    size_t first_halo  = halos->proc_offsets[iproc]   ; 
                    size_t last_halo   = halos->proc_offsets[iproc+1] ;
                    if( this_edge_info.qid_coarse >= first_halo and this_edge_info.qid_coarse < last_halo ) {
                        coarse_hanging_info.rcv_procid.push_back( iproc ) ; 
                    }
                }
                edge_info.n_hanging_ghost_edges ++ ;
            } else if (  any_fine_ghost ) {
                /* If any of the fine quadrants is not local, we add the coarse */
                /* quadrant to the list of those that have to be received after */
                /* restriction.                                                 */
                auto halos = info->ghost_layer ; 
                sc_array_view_t<p4est_quadrant_t>  mirror_quads { &(halos->mirrors) } ; 
                coarse_hanging_info.snd_quadid.push_back(this_edge_info.qid_coarse) ; 
                std::set<int> _snd_procid;
                for( int ii=0; ii<2; ++ii){
                    if( this_edge_info.is_ghost_fine[ii] ) {
                        for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                            size_t first_halo  = halos->proc_offsets[iproc]   ; 
                            size_t last_halo   = halos->proc_offsets[iproc+1] ;
                            if( this_edge_info.qid_fine[ii] >= first_halo and this_edge_info.qid_fine[ii] < last_halo ) {
                                _snd_procid.insert( iproc ) ; 
                            }
                        }
                    }
                }
                coarse_hanging_info.snd_procid.push_back(_snd_procid) ; 
                edge_info.n_hanging_ghost_edges ++ ;
            }
            #endif 
            GRACE_VERBOSE("Hanging edge (B), quadid_coarse {} edge_coarse {}, quadid_fine [ {} {} ] is_ghost_fine [ {} {} ] edge fine {}",
                     this_edge_info.qid_coarse, this_edge_info.which_edge_coarse, 
                     this_edge_info.qid_fine[0], this_edge_info.qid_fine[1], this_edge_info.is_ghost_fine[0], this_edge_info.is_ghost_fine[1], this_edge_info.which_edge_fine) ; 
            hanging_info.push_back(this_edge_info) ; 
        } else {
            simple_edge_info_t this_edge_info {} ; 
            if( sidea.is.full.is_ghost) {
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
            } else if(sideb.is.full.is_ghost){
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
    auto& simple_info         = corner_info.simple_interior_info      ;
    auto& hanging_info        = corner_info.hanging_interior_info     ;
    auto& coarse_hanging_info = 
        reinterpret_cast<grace_neighbor_info_t*>(user_data)->coarse_hanging_quads_info ; 
    /**************************************************/
    /* This means we are at a physical boundary       */
    /* we store the index in user_info and return     */
    /* since physical boundary conditions are handled */
    /* separately.                                    */
    /**************************************************/
    if( sides.size() < P4EST_CHILDREN ) { 
        return ; 
    }
    ASSERT(  sides.size() == P4EST_CHILDREN
           , "Something went wrong in iter_corners, sides.size() != P4EST_CHILDREN.") ; 
    /***************************************************/
    /* Group quadrants in 4 pairs of true corner       */
    /* neighbors by checking which of them share a face*/
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
            #if 0
            if( this_corner_info.is_ghost_coarse ) {
                /* If the coarse quadrant is not local, we add the coarse quad */
                /* to the list of those that have to be received after the     */               
                /* restriction.                                                */ 
                auto halos = info->ghost_layer ; 
                coarse_hanging_info.rcv_quadid.push_back( this_corner_info.qid_coarse ) ;
                for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                    size_t first_halo  = halos->proc_offsets[iproc]   ; 
                    size_t last_halo   = halos->proc_offsets[iproc+1] ;
                    if( this_corner_info.qid_coarse >= first_halo and this_corner_info.qid_coarse < last_halo ) {
                        coarse_hanging_info.rcv_procid.push_back( iproc ) ; 
                    }
                }
                corner_info.n_hanging_ghost_corners ++ ;
            } else if (  this_corner_info.is_ghost_fine ) {
                /* If any of the fine quadrants is not local, we add the coarse */
                /* quadrant to the list of those that have to be received after */
                /* restriction.                                                 */
                auto halos = info->ghost_layer ; 
                sc_array_view_t<p4est_quadrant_t>  mirror_quads { &(halos->mirrors) } ; 
                coarse_hanging_info.snd_quadid.push_back(this_corner_info.qid_coarse) ; 
                std::set<int> _snd_procid;
                for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                    size_t first_halo  = halos->proc_offsets[iproc]   ; 
                    size_t last_halo   = halos->proc_offsets[iproc+1] ;
                    if( this_corner_info.qid_fine >= first_halo and this_corner_info.qid_fine < last_halo ) {
                        _snd_procid.insert( iproc ) ; 
                    }
                }
                coarse_hanging_info.snd_procid.push_back(_snd_procid) ; 
                corner_info.n_hanging_ghost_corners ++ ;
            } 
            #endif 
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
            #if 0 
            if( this_corner_info.is_ghost_coarse ) {
                /* If the coarse quadrant is not local, we add the coarse quad */
                /* to the list of those that have to be received after the     */               
                /* restriction.                                                */ 
                auto halos = info->ghost_layer ; 
                coarse_hanging_info.rcv_quadid.push_back( this_corner_info.qid_coarse ) ;
                for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                    size_t first_halo  = halos->proc_offsets[iproc]   ; 
                    size_t last_halo   = halos->proc_offsets[iproc+1] ;
                    if( this_corner_info.qid_coarse >= first_halo and this_corner_info.qid_coarse < last_halo ) {
                        coarse_hanging_info.rcv_procid.push_back( iproc ) ; 
                    }
                }
                corner_info.n_hanging_ghost_corners ++ ;
            } else if (  this_corner_info.is_ghost_fine ) {
                /* If any of the fine quadrants is not local, we add the coarse */
                /* quadrant to the list of those that have to be received after */
                /* restriction.                                                 */
                auto halos = info->ghost_layer ; 
                sc_array_view_t<p4est_quadrant_t>  mirror_quads { &(halos->mirrors) } ; 
                coarse_hanging_info.snd_quadid.push_back(this_corner_info.qid_coarse) ; 
                std::set<int> _snd_procid;
                for( int iproc = 0; iproc<parallel::mpi_comm_size(); ++iproc ) {
                    size_t first_halo  = halos->proc_offsets[iproc]   ; 
                    size_t last_halo   = halos->proc_offsets[iproc+1] ;
                    if( this_corner_info.qid_fine >= first_halo and this_corner_info.qid_fine < last_halo ) {
                        _snd_procid.insert( iproc ) ; 
                    }
                }
                coarse_hanging_info.snd_procid.push_back(_snd_procid) ; 
                corner_info.n_hanging_ghost_corners ++ ;
            } 
            #endif
            GRACE_VERBOSE("Hanging corner, quadid_coarse {} corner_coarse {}, quadid_fine {} corner fine {}",
                     this_corner_info.qid_coarse, this_corner_info.which_corner_coarse, this_corner_info.qid_fine, this_corner_info.which_corner_fine) ; 
            hanging_info.push_back(this_corner_info) ; 
        } else {
            simple_corner_info_t this_corner_info {} ; 
            if( sidea.is_ghost) {
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
            } else if(sideb.is_ghost){
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


void copy_interior_ghostzones(
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

void restrict_hanging_ghostzones(
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

template< typename InterpT > 
void prolongate_hanging_ghostzones(
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
    
    if( EXPR(n_faces == 0, and n_corners == 0, and n_edges == 0) ){
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
    #if 0 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        corner_policy(
            {0,VECD(0,0), 0,0},
            {VEC(ngz,ngz,ngz), static_cast<long>(nvars), static_cast<long>(n_corners)}
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
            int tid_coarse           =  d_corner_info(icorner).which_tree_coarse     ; 
            int tid_fine             =  d_corner_info(icorner).which_tree_fine       ;
            int8_t is_ghost_fine     =  d_corner_info(icorner).is_ghost_fine         ;  
            int64_t iq_fine          =  d_corner_info(icorner).qid_fine              ;
            /* Get the correct view to index into for coarse data               */
            auto& coarse_view   = is_ghost_coarse ? halo : state ; 

            /* Utility to map the coarse index into the fine quadrant */
            auto const index_mapping(
                VEC(int const ig, int const jg, int const kg),
                int const ca, int const cb, int ijk[GRACE_NSPACEDIM], int IJK[GRACE_NSPACEDIM], int sign[GRACE_NSPACEDIM]
            )
            {
                int x = (ca >> 0) & 1;  
                int y = (ca >> 1) & 1;  
                int z = (ca >> 2) & 1;
                EXPR(
                ijk[0] = (x==0) ? ig : (nx + ngz + ig) ;, 
                ijk[1] = (y==0) ? jg : (ny + ngz + jg) ;,
                ijk[2] = (z==0) ? kg : (nz + ngz + kg) ;,
                )
                x = (cb >> 0) & 1;  
                y = (cb >> 1) & 1;  
                z = (cb >> 2) & 1;
                EXPR(
                IJK[0] = (x==0) ? (ngz+math::floor_int( ig / 2 )) 
                                : (nx + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 )) ;, 
                IJK[1] = (y==0) ? (ngz+math::floor_int( jg / 2 )) 
                                : (ny + ngz - 1 - math::floor_int( (ngz - 1 - jg) / 2 )) ;,
                IJK[2] = (z==0) ? (ngz+math::floor_int( kg / 2 )) 
                                : (nz + ngz - 1 - math::floor_int( (ngz - 1 - kg) / 2 )) ;,
                )
                EXPR(
                sign[0] = (ig%2==1) - (ig%2==0) ;,
                sign[1] = (jg%2==1) - (jg%2==0) ;,
                sign[2] = (kg%2==1) - (kg%2==0) ;
                )
            } ;
            int ijk_c[GRACE_NSPACEDIM], ijk_f[GRACE_NSPACEDIM], sign[GRACE_NSPACEDIM] ; 
            index_mapping(VEC(ig,jg,kg), which_corner_fine, which_corner_coarse, ijk_f, ijk_c, sign) ; 

            /* Fill the fine state */
            #ifndef GRACE_CARTESIAN_COORDINATES     
            state(VEC(i_f,j_f,k_f), ivar, iq_fine)
                                = InterpT::interpolate( VEC(ijk_f[0],ijk_f[1],ijk_f[2])
                                                      , VEC(ijk_c[0],ijk_c[1],ijk_c[2])
                                                      , iq_fine, iq_coarse, ngz, ivar
                                                      , VEC(sign[0], sign[1], sign[2])
                                                      , coarse_view 
                                                      , vols ) ; 
            #else 
            state(VEC(i_f,j_f,k_f), ivar, iq_fine)
                                = InterpT::interpolate( VEC(ijk_f[0],ijk_f[1],ijk_f[2])
                                                      , VEC(ijk_c[0],ijk_c[1],ijk_c[2])
                                                      , iq_fine, iq_coarse, ngz, ivar
                                                      , VEC(sign[0], sign[1], sign[2])
                                                      , coarse_view ) ; 
            #endif 

        }
    ); /* End of loop over corner neighbors */

    /* Loop over edge neighbors             */
    MDRangePolicy<Rank<GRACE_NSPACEDIM+2>> 
        policy_edge(
            {0,0,0, 0,0},
            {ngz, ngz, static_cast<long>(nx), static_cast<long>(nvars), static_cast<long>(n_edges)}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("AMR", "prolongate_hanging_edges")
                , corner_policy 
                , KOKKOS_LAMBDA(const size_t& ig, const size_t& jg, const size_t& k, const size_t& ivar, const size_t& iedge)
        {
            /* Collect the necessary information                                */
            int8_t is_ghost_coarse   =  d_edge_info(iedge).is_ghost_coarse       ; 
            int64_t iq_coarse        =  d_edge_info(iedge).qid_coarse            ;
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

            auto& coarse_view = is_ghost_coarse ? halo : state     ;

            auto const fine_index_mapping( int const ig, int const jg, int const k, 
                                           int const ef, int ijk[GRACE_NSPACEDIM] ) {
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
                int x_ea = edge_directions[0][ef];
                int y_ea = edge_directions[1][ef];
                int z_ea = edge_directions[2][ef];

                // Map indices for ijk based on edge ea
                ijk[0] = (x_ea == ALONG_EDGE) ? (k+ngz) : (x_ea == NEGATIVE_EDGE ? ig : (nx + ngz + ig));
                ijk[1] = (y_ea == ALONG_EDGE) 
                            ? (k+ngz) 
                            : (x_ea == ALONG_EDGE 
                                ? (y_ea == NEGATIVE_EDGE ? ig : (ny + ngz + ig)) 
                                : (y_ea == NEGATIVE_EDGE ? jg : (ny + ngz + jg)));
                ijk[2] = (z_ea == ALONG_EDGE) ? (k+ngz) : (z_ea == NEGATIVE_EDGE ? jg : (nz + ngz + jg));
            }

            /* Find fine cell index                                               */
            int ijk_f[GRACE_NSPACEDIM] ;
            fine_index_mapping( ig,jg,k, which_edge_fine, ijk_f ) ; 

            /* Loop over the two fine quadrants whose ghost-cells we need to fill */
            #pragma unroll 2
            for( int ichild=0; ichild<2; ++ichild ) {
                if( is_ghost_fine[ichild] ) 
                    continue ; 
                const int64_t iq_fine = qid_fine[ichild]             ;
                 
                /* Utility to map the coarse index into the fine quadrant */
                auto const coarse_index_mapping = [&] 
                ( int const ig, int const jg, int const k, 
                  int const ec, int lmn[GRACE_NSPACEDIM], int sign[GRACE_NSPACEDIM] ) 
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
                    
                    int x_eb = edge_directions[0][ec];
                    int y_eb = edge_directions[1][ec];
                    int z_eb = edge_directions[2][ec];
 
                    int const K = math::floor( ( ichild * nx + k ) / 2 ); 

                    // Map indices for lmn based on edge eb
                    lmn[0] = (x_eb == ALONG_EDGE) ? (K+ngz) : (x_eb == NEGATIVE_EDGE ? (ngz + math::floor_int( ig / 2 ) ) 
                                                                                     : (nx + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) ));
                    lmn[1] = (y_eb == ALONG_EDGE) 
                                ? (K+ngz)
                                : (x_eb == ALONG_EDGE 
                                    ? (y_eb == NEGATIVE_EDGE ? (ngz + math::floor_int( ig / 2 ) ) 
                                                             : (nx + ngz - 1 - math::floor_int( (ngz - 1 - ig) / 2 ) )) 
                                    : (y_eb == NEGATIVE_EDGE ? (ngz + math::floor_int( jg / 2 ) )
                                                             : (ny + ngz - 1 - math::floor_int( (ngz - 1 - jg) / 2 ) )) );
                    lmn[2] = (z_eb == ALONG_EDGE) ? (K+ngz) : (z_eb == NEGATIVE_EDGE ? (ngz + math::floor_int( jg / 2 ) )
                                                                                     : (nz + ngz - 1 - math::floor_int( (ngz - 1 - jg) / 2 ) ));
                    // Figure out the sign to use in the prolongator 
                    sign[0] = (x_eb == ALONG_EDGE) ? ( (k%2==1) - (k%2==0) ) : ( (ig%2==1) - (ig%2==0) ) ; 
                    sign[1] = (y_eb == ALONG_EDGE) ? ( (k%2==1) - (k%2==0) ) 
                                                   : ( (x_eb == ALONG_EDGE)  ? ( (ig%2==1) - (ig%2==0) )
                                                                             : ( (jg%2==1) - (jg%2==0) ) ); 
                    sign[2] = (z_eb == ALONG_EDGE) ? ( (k%2==1) - (k%2==0) ) : ( (jg%2==1) - (jg%2==0) ) ; 
                } ; 
                
                int ijk_c[GRACE_NSPACEDIM], sign[GRACE_NSPACEDIM] ; 
                coarse_index_mapping(ig,jg,k, which_edge_fine, ijk_f, sign) ; 

                /* Fill the fine state */
                #ifndef GRACE_CARTESIAN_COORDINATES     
                state(VEC(i_f,j_f,k_f), ivar, iq_fine)
                                    = InterpT::interpolate( VEC(ijk_f[0],ijk_f[1],ijk_f[2])
                                                        , VEC(ijk_c[0],ijk_c[1],ijk_c[2])
                                                        , iq_fine, iq_coarse, ngz, ivar
                                                        , VEC(sign[0], sign[1], sign[2])
                                                        , coarse_view 
                                                        , vols ) ; 
                #else 
                state(VEC(i_f,j_f,k_f), ivar, iq_fine)
                                    = InterpT::interpolate( VEC(ijk_f[0],ijk_f[1],ijk_f[2])
                                                        , VEC(ijk_c[0],ijk_c[1],ijk_c[2])
                                                        , iq_fine, iq_coarse, ngz, ivar
                                                        , VEC(sign[0], sign[1], sign[2])
                                                        , coarse_view ) ; 
                #endif 

            }
        }
    ) ; /* End of loop over edge neighbors */
    #endif 

}

template void 
prolongate_hanging_ghostzones<utils::linear_prolongator_t<grace::minmod>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&  
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
    , Kokkos::vector<hanging_corner_info_t>&
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>&
    #endif
) ; 
template void 
prolongate_hanging_ghostzones<utils::linear_prolongator_t<grace::MCbeta>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&   
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
    , Kokkos::vector<hanging_corner_info_t>&
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>&
    #endif
) ; 

}} /* namespace grace::amr */
