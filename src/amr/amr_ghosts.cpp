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

namespace grace {

grace::face_descriptor_t register_phys_boundary_face(
    grace::sc_array_view_t<p4est_iter_face_side_t> const& sides 
) 
{
    face_descriptor_t this_face ; 
    this_face.kind = PHYS  ; 
    this_face.loc  = LOCAL ;

    auto offset = amr::get_local_quadrants_offset(sides[0].treeid);
    this_face.qid_a = offset + sides[0].is.full.quadid ; 

    return this_face ;

}

grace::face_descriptor_t register_simple_face(
    grace::sc_array_view_t<p4est_iter_face_side_t> const& sides 
) 
{
    face_descriptor_t this_face ; 
    this_face.kind = SIMPLE  ;

    if ( sides[0].is_ghost ) {
        this_face.loc = REMOTE ; 
        auto offset = amr::get_local_quadrants_offset(sides[1].treeid);
        this_face.qid_a = offset + sides[1].is.full.quadid ; 
        this_face.qid_b = sides[0].is.full.quadid ; 
    } else if ( sides[1].is_ghost) {
        this_face.loc = REMOTE ; 
        auto offset = amr::get_local_quadrants_offset(sides[0].treeid);
        this_face.qid_a = offset + sides[0].is.full.quadid ; 
        this_face.qid_b = sides[1].is.full.quadid ; 
    } else {
        this_face.loc = LOCAL  ;
        auto offset = amr::get_local_quadrants_offset(sides[0].treeid);
        this_face.qid_a = offset + sides[0].is.full.quadid ; 
        offset = amr::get_local_quadrants_offset(sides[1].treeid);
        this_face.qid_b = offset + sides[1].is.full.quadid ; 
    }

    return this_face ;

}

grace::face_descriptor_t register_hanging_face(
    grace::sc_array_view_t<p4est_iter_face_side_t> const& sides 
) 
{
    face_descriptor_t this_face ; 
    this_face.kind = HANGING  ;
    this_face.loc  = LOCAL    ;

    int8_t f, c ;
    if ( sides[0].is_hanging ) {
        f = 0 ; c = 1 ; 
    } else {
        f = 1 ; c = 0 ; 
    }

    this_face.faceA = sides[c].face ; 
    this_face.faceB = sides[f].face ; 

    if ( sides[c].is_ghost ) {
        this_face.coarse_remote = 1 ; 
        this_face.loc  = REMOTE     ;
        this_face.qid_coarse = sides[c].is.full.quadid ; 
    } else {
        this_face.coarse_remote = 0 ; 
        auto const offset = amr::get_local_quadrants_offset(sides[c].treeid) ; 
        this_face.qid_coarse = sides[c].is.full.quadid + offset ; 
    } 
    auto offset =  amr::get_local_quadrants_offset(sides[f].treeid) ; 
    for( int ic=0; ic<P4EST_CHILDREN / 2 ) {
        if ( sides[f].is.hanging.is_ghost[ic] ) {
            this_face.child_remote_mask[ic] = 1 ; 
            this_face.loc = REMOTE ; 
            this_face.qid_f[ic] = sides[f].is.hanging.quadid[ic] ;  
        } else {
            this_face.child_remote_mask[ic] = 0 ; 
            this_face.child_rank[ic] = /*TODO*/ 0 ; 
            this_face.qid_f[ic] = sides[f].is.hanging.quadid[ic] + offset ;  
        }
    }


    return this_face ;

}

void grace_iterate_faces(
    p4est_iter_faces_info_t* info,
    void* user_data 
)
{ 
    auto buffer = reinterpret_cast<neighbor_descriptor_t*>(user_data) ; 
    sc_array_view_t<p4est_iter_face_side_t> sides{
        &(info->sides)
    } ;
    auto& faces = buffer->faces ; 
    face_descriptor_t this_face ; 

    if ( sides.size() == 1 ) {
        this_face = register_phys_boundary_face(sides) ; 
        return ; 
    }

    if ( sides[0].is_hanging || sides[1].is_hanging) {
        this_face =  register_hanging_face(sides) ;         
    } else {
        this_face =  register_simple_face(sides) ; 
    }

    if ( this_face.loc == REMOTE ) {
        remote_data_descriptor_t desc ; 
        desc.topology = FACE ; 
        switch (this_face.kind)
        {
            case SIMPLE:


            break ; 
            case HANGING:
            break ; 
            case default
            ASSERT(0, "Invalid face state in p4est_iterate") ; 
            break;  
        }
     
    }
}


void grace_iterate_faces( p4est_iter_face_info_t* info 
                          , void* user_data ) 
{
    auto ghost_layer = reinterpret_cast<amr_ghosts_impl_t*>(user_data) ;

    auto& local_faces  = ghost_layer->local_face_info  ; 
    auto& remote_faces = ghost_layer->remote_face_info ;
    auto& phys_faces   = ghost_layer->phys_face_info   ;
    auto& hanging_local_faces  = ghost_layer->local_hanging_face_info   ;
    auto& hanging_remote_faces = ghost_layer->local_hanging_face_info   ;


    sc_array_view_t<p4est_iter_face_side_t> sides{
        &(info->sides)
    } ;

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
    int8_t l, g, f, c ; 

    auto const check_remote = [&] (
        int8_t f, int8_t c,
        bool& coarse_remote,
        bool fine_remote[P4EST_CHILDREN/2]
    ) {
        
        coarse_remote = sides[c].is.full.is_ghost ; 
        any_remote = false ; 
        for( int ic=0; ic<P4EST_CHILDREN/2; ++ic) {
            fine_remote[ic] = sides[f].is.hanging.is_ghost[ic] ; 
            any_remote = any_remote or fine_remote[ic] ; 
        }
        return any_remote ; 
    }

    auto const append_simple_remote = [&] (
            int8_t l, int8_t g
        ) {
        remote_faces.n_faces ++ ; 
        auto offset = amr::get_local_quadrants_offset(sides[l].treeid);
        remote_faces.face_a_h.push_back(
            sides[l].is.full.face
        ) ; 
        remote_faces.face_b_h.push_back(
            sides[g].is.full.face
        ) ; 
        remote_faces.qid_a_h.push_back(
            sides[l].is.full.quadid + offset 
        ) ; 
        remote_faces.qid_b_h.push_back(
            remote_faces.n_faces - 1 
        ) ; 
    }; 


    auto const append_hanging_local [&] (
        int8_t f, int8_t c
    ) {
        hanging_local_faces.n_coarse ++ ; 

        auto const& fine_side = sides[f] ; 
        auto const& coarse_side = sides[c] ; 
        
        auto offset_c = amr::get_local_quadrants_offset(
            coarse_side.treeid 
        ) ; 
        auto offset_f = amr::get_local_quadrants_offset(
            fine_side.treeid 
        ) ; 
        hanging_local_faces.qid_c_h.push_back(
            coarse_side.is.full.quadid + offset_c 
        ) ; 
        hanging_local_faces.face_c_h.push_back(
            coarse_side.face
        ) ; 
        hanging_local_faces.face_f_h.push_back(
            fine_side.face
        ) ; 
        for( int ic=0; ic<P4EST_CHILDREN/2; ++ic) {
            auto const qid_f = fine_side.is.hanging.quadid[ic] + offset_f  ;
            hanging_local_faces.qid_f_h.push_back(
                qid_f
            ) ; 
            hanging_local_faces.qid_f_to_bid_h[qid_f] = hanging_local_faces.n_fine - 1 ; 
            hanging_local_faces.n_fine ++ ; 
        }

    } ; 

    bool coarse_remote, any_remote{false}, any_hanging{false} ; 
    bool fine_remote[P4EST_CHILDREN/2] ; 

    int8_t f,c ; 
    int8_t l,g ; 

    

    if ( sides[0].is_hanging ) {
        f = 0 ; c = 1 ;
        any_hanging = true ; 
    } else if ( sides[1].is_hanging ) {
        f = 1 ; c=0 ; 
        any_hanging = true ; 
    } 
    
    if ( any_hanging ) {
        auto any_remote_f = check_remote(f,c,coarse_remote,fine_remote) ; 
        if ( any_remote_f ) {

        } else if (coarse_remote) {

        } else {
            append_hanging_local(f,c) ; 
        }
    } else {
        if ( sides[0].is.full.is_ghost ) {
            g = 0, l = 1 ;
            append_simple_remote(l,g) ; 
        } else if ( sides[1].is.full.is_ghost ) {
            g = 1; l = 0 ;
            append_simple_remote(l,g) ; 
        } else {
            local_faces.n_faces++ ; 
            auto offset_a = amr::get_local_quadrants_offset(sides[0].treeid);
            auto offset_b = amr::get_local_quadrants_offset(sides[1].treeid);
            local_faces.face_a_h.push_back(sides[0].is.full.face) ; 
            local_faces.face_b_h.push_back(sides[1].is.full.face) ; 
            local_faces.qid_a_h.push_back(sides[0].is.full.quadid + offset_a) ; 
            local_faces.qid_b_h.push_back(sides[1].is.full.quadid + offset_b) ;
        }
    }

}



void amr_ghosts_impl_t::update() {
    
    
    std::vector<face_type_t> stack ; 
    stack.reserve()

}

} /* namespace grace */