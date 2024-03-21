/**
 * @file bc_helpers.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
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
#include <thunder/amr/p4est_headers.hh>
#include <thunder/amr/bc_helpers.tpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

#include <bc_helpers.hh> 

namespace thunder { namespace amr {

void thunder_iterate_faces( p4est_iterater_face_info_t * info 
                          , void* user_data  )
{
    using namespace thunder; 

    auto face_iter_data = reinterpret_cast<iterate_face_data_t*>(user_data) ; 
    auto physical_boundary_info = face_iter_data->phys_bc_idx ; 
    
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
            physical_boundary_info->push_back(
                (offset+sides[0].hanging.quadid[0]) * P4EST_FACES + sides[0].face
            ); 
            locid = 
            physical_boundary_info->push_back(
                (offset+sides[0].hanging.quadid[1]) * P4EST_FACES + sides[0].face
            ); 
        } else {
            physical_boundary_info->push_back(
                (offset+sides[0].full.quadid) * P4EST_FACES + sides[0].face
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
    auto& vars = variables::get().getstate() ;  
    int8_t const orientation = info->orientation ; 
    ASSERT( orientation == 0
          , "Twisted grid topologies not yet implemented"
            " in ghost exchange." ) ; 
    auto mpi_context = face_data->context ; 

    p4est_ghost_t* ghost_quads = info->ghost_layer ; 
    
    ASSERT_DBG( ghost_quads != nullptr, "Ghost layer is null.");
    
    for( auto const& side: sides )
    {
        if( side.is_ghost )
        {

        } else {
            
        }
    }


}

}} /* namespace thunder::amr */