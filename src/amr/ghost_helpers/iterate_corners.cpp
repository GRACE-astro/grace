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
    p4est_iter_corner_side_t const& side, 
    std::vector<quad_neighbors_descriptor_t>& neighbors
)
{
    auto const offset = grace::amr::get_local_quadrants_offset(side.treeid); 
    
    if (side.is_ghost) return ; 
    // not hanging not ghost 
    auto qid = side.quadid +  offset ; 
    neighbors[qid].corners[side.corner].kind = interface_kind_t::PHYS ;
    neighbors[qid].n_registered_corners ++ ; 
    
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

    desc.kind = interface_kind_t::INTERNAL;
    desc.data.quad_id = s1.quadid + other_offset;
    desc.data.is_remote = s1.is_ghost;

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
}

void grace_iterate_corners(p4est_iter_corner_info_t* info, void* user_data)
{
    auto ghosts = reinterpret_cast<std::vector<quad_neighbors_descriptor_t>*>(user_data);
    sc_array_view_t<p4est_iter_corner_side_t> sides{&(info->sides)};

    if (sides.size() < P4EST_CHILDREN) {
        for (auto const& side : sides) {
            register_physical_boundary_corner(side, *ghosts);
        }
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