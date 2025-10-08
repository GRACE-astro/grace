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
    auto offset = grace::amr::get_local_quadrants_offset(s0.treeid);
    int8_t f = s0.face ;
    if ( s0.is_hanging ) {
        for( int iq=0; iq<P4EST_CHILDREN/2; ++iq){
            if (s0.is.hanging.is_ghost[iq]) continue ; 
            
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

void grace_iterate_faces(
    p4est_iter_face_info_t * info,
    void* user_data 
) 
{
    auto ghosts = reinterpret_cast<std::vector<quad_neighbors_descriptor_t>*>(user_data) ; 
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
}

} /* namespace grace */