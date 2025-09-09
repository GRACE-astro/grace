/**
 * @file amr_ghosts.cpp
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

#include <grace/config/config_parser.hh>
#include <grace/errors/assert.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <vector>

#include <grace/amr/amr_ghosts.hh>
#include <grace/amr/bc_helpers.tpp>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/p4est_headers.hh>

namespace grace {

void register_simple_face(
    p4est_iter_face_side_t const& quad,
    p4est_iter_face_side_t const& neighbor,
    std::vector<quad_neighbors_descriptor_t>& neighbors) 
{
    auto qid = quad.is.full.quadid + amr::get_local_quadrants_offset(quad.treeid) ; 

    int8_t const f = quad.face ;
    auto& desc = neighbors[qid].faces[f] ; 

    
    desc.level_diff = 0 ; 
    desc.kind = interface_kind_t::INTERNAL ; 
    
    desc.data.full.is_remote = neighbor.is.full.is_ghost ; 
    auto offset = neighbor.is.full.is_ghost  ? 0 : amr::get_local_quadrants_offset(neighbor.treeid) ;
    desc.data.full.quad_id = neighbor.is.full.quadid + offset ;
    // TODO figure out where the rank is  
    if ( desc.data.full.is_remote ) {
        #if 0 
        // I believe that neighbor.is.full.quad is a pointer to the below 
        sc_array_view_t<p4est_quadrant_t> halo_quads{
            &(halo->ghosts)
        } ;
        auto rank = halo_quads[neighbor.is.full.quadid].piggy1.owner_rank ; 
        #endif 

        desc.data.full.owner_rank = neighbor.is.full.quad->p.piggy1.owner_rank ;
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
    auto const& s0 = sides[0] ; auto const& s1 = sides[1] ; 

    /* Grid boundary case first */
    if (sides.size() == 1) {
        auto offset = amr::get_local_quadrants_offset(s0.treeid) ; 
        auto& desc = ghosts->at(s0.is.full.quadid + offset); 
        uint8_t f = s0.face ;
        desc.faces[f].kind = interface_kind_t::PHYS ; 
        return ; 
    }

    if ( s0.is_hanging ) {

    } else if ( s1.is_hanging ) {

    } else {
        if ( s0.is.full.is_ghost ) {
            register_simple_face(s1, s0, *ghosts) ; 
        } else if (s1.is.full.is_ghost ) {
            register_simple_face(s0, s1, *ghosts) ; 
        } else {
            register_simple_face(s0, s1, *ghosts) ; 
            register_simple_face(s1, s0, *ghosts) ; 
        }
    }
}

void amr_ghosts_impl_t::update() {

    // Destroy old ghost layer if present
    if (p4est_ghost_layer) {
        p4est_ghost_destroy(p4est_ghost_layer);
    }

    // Rebuild ghost layer from scratch
    p4est_ghost_layer = p4est_ghost_new(grace::amr::forest::get().get(), P4EST_CONNECT_FULL);

    // Clear and re-alloc the ghost_layer to the correct size
    auto nq = amr::get_local_num_quadrants() ; 
    ghost_layer.clear() ; 
    ghost_layer.resize(nq) ; 

    // Register neighbor faces into ghost_layer
    p4est_iterate(grace::amr::forest::get().get(),      /*forest*/
                  p4est_ghost_layer,                    /*ghost layer*/
                  static_cast<void*>(&ghost_layer),     /*user data*/
                  nullptr,                              /*volume*/
                  &grace_iterate_faces,                 /*face*/
                  nullptr,                              /*edge*/
                  nullptr );                            /*corner*/
}

} /* namespace grace */