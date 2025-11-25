/**
 * @file iterate_faces.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Index fiesta.
 * @date 2025-09-05
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
#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/errors/assert.hh>
#include <grace/system/print.hh>

#include <grace/amr/p4est_headers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/forest.hh>

#include <grace/data_structures/memory_defaults.hh>

#include <grace/utils/sc_wrappers.hh>

#include <grace/amr/amr_ghosts.hh>


#include <vector>

namespace grace {

inline void fill_full_face_desc(
    face_descriptor_t& desc,
    int8_t this_face,
    int8_t other_face, 
    size_t qid,
    bool is_remote,
    p4est_topidx_t treeid,
    p4est_quadrant_t const* quad
)
{
    desc.face = other_face ; 
    desc.kind = interface_kind_t::INTERNAL ; 
    desc.data.full.is_remote = is_remote ;
    desc.data.full.quad_id = qid ; 
    desc.data.full.task_id.fill(UNSET_TASK_ID);
    if ( is_remote ) {
        desc.data.full.owner_rank =
            p4est_comm_find_owner(grace::amr::forest::get().get(), treeid, quad, 0) ; 
    }
}

inline void fill_hanging_face_desc(
    face_descriptor_t& desc,
    int8_t this_face,
    int8_t other_face, 
    int8_t islot,
    size_t qid,
    bool is_remote,
    p4est_topidx_t treeid,
    p4est_quadrant_t const* quad
)
{
    desc.face = other_face ;
    desc.kind = interface_kind_t::INTERNAL ; 
    desc.data.hanging.quad_id[islot] = qid;
    desc.data.hanging.is_remote[islot] = is_remote;
    desc.data.hanging.task_id[islot].fill(UNSET_TASK_ID);
    if (is_remote) {
        desc.data.hanging.owner_rank[islot] =
            p4est_comm_find_owner(grace::amr::forest::get().get(), treeid, quad, 0);
    }
}

static void register_face(
    p4est_iter_face_side_t const& s0,
    p4est_iter_face_side_t const& s1,
    std::vector<quad_neighbors_descriptor_t>& neighbors
)
{
    
    int8_t f = s0.face ;
    if ( s0.is_hanging ) {
        for( int iq=0; iq<P4EST_CHILDREN/2; ++iq){
            if (s0.is.hanging.is_ghost[iq]) continue ; 
            auto offset = grace::amr::get_local_quadrants_offset(s0.treeid);
            auto const qid = s0.is.hanging.quadid[iq] + offset ;
            auto& desc = neighbors[qid].faces[f] ; 
            neighbors[qid].n_registered_faces ++ ; 
            auto other_offset = s1.is.full.is_ghost ? 0 : grace::amr::get_local_quadrants_offset(s1.treeid) ; 
            desc.level_diff = level_diff_t::COARSER ; 
            desc.child_id = iq ; 
            fill_full_face_desc(
                desc, f, s1.face, 
                s1.is.full.quadid + other_offset, 
                s1.is.full.is_ghost, s1.treeid, 
                s1.is.full.quad
            );
        }
    } else {
        if ( s0.is.full.is_ghost ) return ; // don't register if remote 
        auto offset = grace::amr::get_local_quadrants_offset(s0.treeid);
        auto const qid = s0.is.full.quadid + offset ;
        auto& desc = neighbors[qid].faces[f] ; 
        neighbors[qid].n_registered_faces ++ ; 
        if ( s1.is_hanging ) {
            for( int iq=0; iq<P4EST_CHILDREN/2; ++iq) {
                auto const other_offset = s1.is.hanging.is_ghost[iq] ? 0 :  grace::amr::get_local_quadrants_offset(s1.treeid) ; 
                desc.level_diff = level_diff_t::FINER ; 
                fill_hanging_face_desc(
                    desc, f, s1.face,
                    iq, s1.is.hanging.quadid[iq] + other_offset, 
                    s1.is.hanging.is_ghost[iq], s1.treeid, s1.is.hanging.quad[iq]
                ) ; 
            }
        } else {
            auto const other_offset = s1.is.full.is_ghost ? 0 :  grace::amr::get_local_quadrants_offset(s1.treeid) ; 
            desc.level_diff = level_diff_t::SAME ; 
            fill_full_face_desc(
                desc, f, s1.face, 
                s1.is.full.quadid + other_offset, 
                s1.is.full.is_ghost, s1.treeid, 
                s1.is.full.quad
            );
        }
    }
}

void register_refluxing_face(
    p4est_iter_face_side_t const& coarse_side,
    p4est_iter_face_side_t const& fine_side,
    p4est_iter_data_t* info 
) 
{
    hanging_face_reflux_desc_t desc {} ; 
    desc.coarse_is_remote = coarse_side.is.full.is_ghost ; 
    desc.coarse_qid = coarse_side.is.full.is_ghost 
                    ? coarse_side.is.full.quadid 
                    : coarse_side.is.full.quadid + grace::amr::get_local_quadrants_offset(coarse_side.treeid) ; 
    desc.coarse_face_id = coarse_side.face ; 
    desc.coarse_owner_rank = p4est_comm_find_owner(grace::amr::forest::get().get(), coarse_side.treeid, coarse_side.is.full.quad, 0);
    desc.fine_face_id = fine_side.face ; 
    for( int ic=0; ic<P4EST_CHILDREN/2; ++ic) {
        // local / remote 
        desc.fine_is_remote[ic] = fine_side.is.hanging.is_ghost[ic] ; 
        desc.fine_qid[ic] = fine_side.is.hanging.is_ghost[ic] 
                            ? fine_side.is.hanging.quadid[ic] 
                            : fine_side.is.hanging.quadid[ic] + grace::amr::get_local_quadrants_offset(fine_side.treeid) ; 
        desc.fine_owner_rank[ic] = p4est_comm_find_owner(grace::amr::forest::get().get(), fine_side.treeid, fine_side.is.hanging.quad[ic], 0);
    } 
    info->reflux_faces->push_back(desc) ; 
}

void register_refluxing_edges(
    p4est_iter_face_side_t const& coarse_side,
    p4est_iter_face_side_t const& fine_side,
    p4est_iter_data_t* info 
) 
{
    auto fdir = coarse_side.face / 2 ; 
    std::array<std::array<int,2>,3> other_dirs {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 

    std::array<std::array<int,2>,6> face_coarse_edges {{
        {{4,8}}, // 0
        {{5,9}}, // 1
        {{0,8}}, // 2
        {{1,10}}, // 3
        {{0,4}}, // 4
        {{2,6}} // 5
    }} ;
     
    auto const get_fine_edge_id = [&] (int8_t idir, int8_t iside) -> int8_t  {
        return grace::amr::detail::f2e[fine_side.face][2*!idir+!iside]; 
    } ;
    for( int idir=0; idir<2; ++idir) {
        int edge_dir = other_dirs[fdir][idir] ; 
        // ... 
        hanging_edge_reflux_desc_t desc {} ; 
        desc.n_fine = 2 ; 
        desc.n_coarse = 1 ;
        desc.n_sides = 3 ; // only three sides here 
        desc.fine_sides[0] = 1 ;
        desc.fine_sides[1] = 2 ;
        desc.coarse_sides[0] = 0 ; 

        auto& csd = desc.sides[0] ; // by convention we put the coarse side at 0 
        csd.is_fine = false ; 
        csd.octants.coarse.is_remote = coarse_side.is.full.is_ghost ;
        csd.octants.coarse.quad_id = coarse_side.is.full.is_ghost 
                                ? coarse_side.is.full.quadid 
                                : coarse_side.is.full.quadid + grace::amr::get_local_quadrants_offset(coarse_side.treeid) ; 
        csd.octants.coarse.owner_rank = p4est_comm_find_owner(grace::amr::forest::get().get(), coarse_side.treeid, coarse_side.is.full.quad, 0);

        csd.edge_id = face_coarse_edges[fdir][idir] ; 

        // now these are used inside the kernel writing 
        // data back to the emf in the coarse buffer, to 
        // decide whether the emf index should be shifted
        // by nx/2.
        // In that kernel, i and j represent the two 
        // directions orthogonal to the EMF in z-order.
        // Therefore here we want to be consistent with that.
        // In particular, one of these dirs will always be fdir,
        // and the other one (the one we want to shift) is the 
        // direction orthogonal to fdir and idir. The code below
        // implements that:
        // if fdir > other_dirs[fdir][!idir] --> we need to shift i in EMF kernel
        // if fdir < other_dirs[fdir][!idir] --> we need to shift j in EMF kernel
        // equivalently, we can just compare fdir to the orthogonal directions of 
        // edge dir. The one (i or j) which is fdir does **not** need shifting
        csd.off_i = (fdir == other_dirs[edge_dir][1]) ; 
        csd.off_j = (fdir == other_dirs[edge_dir][0]) ; 

        // each edge has 2 sides that are fine 
        // here we loop over them in the direction
        // of increasing coordinate across the edge 
        // and within the face
        for( int iside=0; iside<2; ++iside) {
            // fetch desc
            auto& fsd = desc.sides[1+iside] ; 
            // edge id is opposite ours 
            fsd.edge_id = get_fine_edge_id(idir,iside) ; 
            fsd.off_i = fsd.off_j = 0 ; 
            fsd.is_fine = true; 
            for ( int ichild2=0; ichild2<2; ++ichild2) {
                int ichild = idir == 0 ? ichild2 + 2 * iside 
                                       : iside   + 2 * ichild2 ; 

                fsd.octants.fine.is_remote[ichild2] = fine_side.is.hanging.is_ghost[ichild] ; 
                fsd.octants.fine.quad_id[ichild2] = fine_side.is.hanging.is_ghost[ichild]
                                                  ? fine_side.is.hanging.quadid[ichild]
                                                  : fine_side.is.hanging.quadid[ichild] + grace::amr::get_local_quadrants_offset(fine_side.treeid) ; 
                fsd.octants.fine.owner_rank[ichild2] = p4est_comm_find_owner(grace::amr::forest::get().get(), fine_side.treeid, fine_side.is.hanging.quad[ichild], 0);
            }   
        }
        info->reflux_edges->push_back(desc) ; 
    }  // loop over edge dir within face 
}


void grace_iterate_faces(
    p4est_iter_face_info_t * info,
    void* user_data 
) 
{
    auto iter_data = reinterpret_cast<p4est_iter_data_t*>(user_data) ; 
    auto ghosts = iter_data->ghost_layer ; 
    sc_array_view_t<p4est_iter_face_side_t> sides{
        &(info->sides)
    } ;
    
    auto const& s0 = sides[0] ; 
    /* Grid boundary case first */
    if (sides.size() == 1) {
        auto offset = amr::get_local_quadrants_offset(s0.treeid) ; 
        auto& desc = ghosts->at(s0.is.full.quadid + offset); 
        uint8_t f = s0.face ;
        auto& face = desc.faces[f];
        face.kind = interface_kind_t::PHYS ;
        face.data.phys.dir[0] = face.data.phys.dir[1] = face.data.phys.dir[2] = 0;
        face.data.phys.dir[static_cast<size_t>(f/2)] = f%2 ? +1 : -1 ;
        face.data.phys.type = amr::element_kind_t::FACE ; 
        face.data.phys.in_cbuf = false ;
        face.data.phys.task_id.fill(UNSET_TASK_ID);
        desc.n_registered_faces ++ ; 
        return ; 
    }
    auto const& s1 = sides[1] ; 
    register_face(s0,s1,*ghosts) ; 
    register_face(s1,s0,*ghosts) ; 

    if (s0.is_hanging) {
        // register for reflux 
        // here:
        // if both are local we register to local array,
        // if s0 is local we register to send array,
        // if s1 is local we register to receive array 
        register_refluxing_face(s1,s0,iter_data) ; 
        // register the virtual edges too
        //register_refluxing_edges(s1,s0,iter_data) ;
    } else if (s1.is_hanging) {
        register_refluxing_face(s0,s1,iter_data) ; 
        // register the virtual edges too 
        //register_refluxing_edges(s0,s1,iter_data) ; 
    }
}

} /* namespace grace */