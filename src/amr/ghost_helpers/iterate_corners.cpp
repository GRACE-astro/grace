/**
 * @file iterate_corners.cpp
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
#include <grace/utils/sc_wrappers.hh>

#include <grace/errors/assert.hh>
#include <grace/system/print.hh>

#include <grace/amr/p4est_headers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/forest.hh>

#include <grace/data_structures/memory_defaults.hh>

#include <grace/amr/amr_ghosts.hh>


#include <vector>

namespace grace {

static void register_physical_boundary_corner(
    grace::sc_array_view_t<p4est_iter_corner_side_t> const& sides, 
    std::vector<quad_neighbors_descriptor_t>& neighbors
)
{
    size_t const nsides = sides.size() ; 

    // first order of business, figure out directions 
    std::array<int8_t, 3> dir ;

    auto const get_dir = [&] (int off) -> int8_t {
        return off ? static_cast<int8_t>(+1) : static_cast<int8_t>(-1) ; 
    } ; 

    if ( nsides == 1 ) {
        auto const& side = sides[0] ; 
        if (side.is_ghost) return ; 

        auto const offset = grace::amr::get_local_quadrants_offset(side.treeid); 
        // not hanging not ghost 
        auto qid = side.quadid +  offset ; 
        auto& corner = neighbors[qid].corners[side.corner] ; 
        corner.kind = interface_kind_t::PHYS ;
        corner.phys.dir[0] = get_dir((side.corner>>0)&1)  ; 
        corner.phys.dir[1] = get_dir((side.corner>>1)&1) ; 
        corner.phys.dir[2] = get_dir((side.corner>>2)&1) ; 
        corner.phys.in_cbuf = false ;
        corner.filled = true ; 
        corner.phys.type = amr::element_kind_t::CORNER ; 
        corner.phys.task_id.fill(UNSET_TASK_ID); 
    } else if (nsides==2) {
        // we are on a grid edge 
        // we need to identify the edge direction 
        int off [3][2] = {
            {(sides[0].corner >> 0)&1, (sides[1].corner >> 0)&1},
            {(sides[0].corner >> 1)&1, (sides[1].corner >> 1)&1},
            {(sides[0].corner >> 2)&1, (sides[1].corner >> 2)&1},
        } ; 

        // offsets 
        auto const o1 = grace::amr::get_local_quadrants_offset(sides[0].treeid); 
        auto const o2 = grace::amr::get_local_quadrants_offset(sides[1].treeid); 
        // quad_ids 
        auto qid1 = sides[0].quadid +  o1 ; 
        auto qid2 = sides[1].quadid +  o2 ; 
        // not hanging not ghost 
        if ( !sides[0].is_ghost ) {
            auto& c1 = neighbors[qid1].corners[sides[0].corner] ; 
            c1.kind = interface_kind_t::PHYS ;
            c1.filled = true ; 
            c1.phys.dir[0] = (off[0][0] == off[0][1]) ? get_dir(off[0][0]) : 0 ; 
            c1.phys.dir[1] = (off[1][0] == off[1][1]) ? get_dir(off[1][0]) : 0 ; 
            c1.phys.dir[2] = (off[2][0] == off[2][1]) ? get_dir(off[2][0]) : 0 ; 
            c1.phys.in_cbuf =false ;
            c1.phys.type = amr::element_kind_t::EDGE ; 
            c1.phys.task_id.fill(UNSET_TASK_ID);
        }
        if ( !sides[1].is_ghost ) {
            auto& c2 = neighbors[qid2].corners[sides[1].corner] ;
            // set kind  
            c2.kind = interface_kind_t::PHYS ;
            // set filled 
            c2.filled = true ; 
            // grid normal 
            c2.phys.dir[0] = (off[0][0] == off[0][1]) ? get_dir(off[0][0]) : 0 ; 
            c2.phys.dir[1] = (off[1][0] == off[1][1]) ? get_dir(off[1][0]) : 0 ; 
            c2.phys.dir[2] = (off[2][0] == off[2][1]) ? get_dir(off[2][0]) : 0 ; 
            // we can check here that one and only one is 0 
            c2.phys.type = amr::element_kind_t::EDGE ;
            c2.phys.in_cbuf = false ;
            c2.phys.task_id.fill(UNSET_TASK_ID);
        }
    } else if (nsides==4) {
        int off[3][4] = {
            {(sides[0].corner >> 0)&1, (sides[1].corner >> 0)&1,
            (sides[2].corner >> 0)&1, (sides[3].corner >> 0)&1},
            {(sides[0].corner >> 1)&1, (sides[1].corner >> 1)&1,
            (sides[2].corner >> 1)&1, (sides[3].corner >> 1)&1},
            {(sides[0].corner >> 2)&1, (sides[1].corner >> 2)&1,
            (sides[2].corner >> 2)&1, (sides[3].corner >> 2)&1},
        };
        for( int iside=0; iside<4; ++iside) {
            if (sides[iside].is_ghost) continue ; 
            // offsets 
            auto const o = grace::amr::get_local_quadrants_offset(sides[iside].treeid);
            // quad_ids 
            auto qid = sides[iside].quadid +  o ;
            // not hanging not ghost 
            auto& c = neighbors[qid].corners[sides[iside].corner] ; 
            // set kind 
            c.kind = interface_kind_t::PHYS ;
            // set filled 
            c.filled = true ; 
            // corner.phys.dir = {x, y, z}
            for(int d=0; d<3; ++d){
                // if all 4 offsets along this axis are the same, take +1/-1
                c.phys.dir[d] = (off[d][0]==off[d][1] && off[d][1]==off[d][2] && off[d][2]==off[d][3])
                                    ? get_dir(off[d][0])
                                    : 0; 
            }
            c.phys.type = amr::element_kind_t::FACE ; 
            c.phys.in_cbuf = false ;
            c.phys.task_id.fill(UNSET_TASK_ID);
        }
    } else {
        ERROR("Unexpected number of side " << sides.size() ) ; 
    }
    
    // neighbors[qid].n_registered_corners ++ ; 
    
}


static void register_corner(
    p4est_iter_corner_side_t const& s0,
    p4est_iter_corner_side_t const& s1,
    std::vector<quad_neighbors_descriptor_t>& neighbors)
{
    if (s0.is_ghost) return; // we only register if local

    auto const offset = grace::amr::get_local_quadrants_offset(s0.treeid);
    auto const qid = s0.quadid + offset;
    auto const c   = s0.corner;
    auto& desc     = neighbors[qid].corners[c];

    neighbors[qid].n_registered_corners++;

    auto const l0 = static_cast<int>(s0.quad->level);
    auto const l1 = static_cast<int>(s1.quad->level);

    auto const other_offset = s1.is_ghost ? 0 : grace::amr::get_local_quadrants_offset(s1.treeid);
    desc.filled = true ; 
    desc.kind = interface_kind_t::INTERNAL;
    desc.data.quad_id = s1.quadid + other_offset;
    desc.data.is_remote = s1.is_ghost;
    desc.data.task_id.fill(UNSET_TASK_ID) ;
    if (s1.is_ghost) {
        desc.data.owner_rank =
            p4est_comm_find_owner(grace::amr::forest::get().get(), s1.treeid, s1.quad, 0);
    }

    if (l0 > l1) {
        desc.level_diff = level_diff_t::COARSER;
    } else if (l1 > l0) {
        desc.level_diff = level_diff_t::FINER;
    } else {
        desc.level_diff = level_diff_t::SAME;
    }

    desc.corner = s1.corner ; 
}

void grace_iterate_corners(p4est_iter_corner_info_t* info, void* user_data)
{
    auto ghosts = reinterpret_cast<std::vector<quad_neighbors_descriptor_t>*>(user_data);
    sc_array_view_t<p4est_iter_corner_side_t> sides{&(info->sides)};
    
    if (sides.size() < P4EST_CHILDREN) {
        register_physical_boundary_corner(sides, *ghosts);
        return; 
    }

    // Build opposite pairs
    static constexpr int opposite_corner[8] = {7,6,5,4,3,2,1,0};
    auto is_corner_neighbor = [&](int i, int j) {
        return opposite_corner[sides[i].corner] == sides[j].corner;
    };

    std::vector<std::array<int,2>> pairs;
    std::vector<bool> found(sides.size(), false);

    for (int in = 0; in < static_cast<int>(sides.size()); ++in) {
        if (found[in]) continue;
        for (int jn = in + 1; jn < static_cast<int>(sides.size()); ++jn) {
            if (found[jn]) continue;
            if (is_corner_neighbor(in, jn)) {
                pairs.push_back({in, jn});
                found[in] = found[jn] = true;
                break;
            }
        }
    }

    // Register both directions
    for (auto const& [i0, i1] : pairs) {
        register_corner(sides[i0], sides[i1], *ghosts);
        register_corner(sides[i1], sides[i0], *ghosts);
    }
}




} /* namespace grace */