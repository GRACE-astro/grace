/**
 * @file bc_helpers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
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

#ifndef GRACE_AMR_BC_ITERATE_HH 
#define GRACE_AMR_BC_ITERATE_HH

#include <grace_config.h>

#include <grace/parallel/mpi_wrappers.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/utils/numerics/interpolators.hh>
#include <grace/amr/boundary_conditions.hh>
#include <grace/utils/numerics/prolongation.hh>
#include <grace/utils/numerics/limiters.hh> 
#include <grace/data_structures/variable_properties.hh>

#include <Kokkos_Vector.hpp>

namespace grace{ namespace amr {
/**************************************************************************************************/
/**
 * @brief Iterate through all the faces of grid quadrants to
 *        store boundary information.
 * \ingroup amr
 * @param info <code>p4est</code>'s struct containing information   
 *             regarding the quadrant face.
 * @param user_data Type erased <code>grace_neighbor_info_t</code> 
 *                  where information is stored.
 * This function is used as callback in <code>p4est_iterate</code> to store 
 * all the necessary information to apply interior and exterior boundary conditions.
 * In particular, this function stores, for all faces, the quadrant id's which share 
 * this face, whether this face is hanging or simple, whether it's internal or external,
 * its face orientation code, the tree(s) containing the quadrants on each side, and whether
 * any of the quadrants on this face are in the halo.
 */
void grace_iterate_faces( p4est_iter_face_info_t* info 
                          , void* user_data ) ;
/**
 * @brief Iterate through all the corners of grid quadrants to
 *        store boundary information.
 * \ingroup amr
 * @param info <code>p4est</code>'s struct containing information   
 *             regarding the quadrant corner.
 * @param user_data Type erased <code>grace_neighbor_info_t</code> 
 *                  where information is stored.
 * This function is used as callback in <code>p4est_iterate</code> to store 
 * all the necessary information to apply interior and exterior boundary conditions.
 * In particular, this function stores, for all corners, the quadrant id's which share 
 * this corner, whether this corner is hanging or simple, whether it's internal or external,
 * its corner orientation code, the tree(s) containing the quadrants on each side, and whether
 * any of the quadrants on this corner are in the halo.
 */
void grace_iterate_corners( p4est_iter_corner_info_t* info 
                          , void* user_data ) ;
#ifdef GRACE_3D 
/**
 * @brief Iterate through all the edges of grid quadrants to
 *        store boundary information.
 * \ingroup amr
 * @param info <code>p4est</code>'s struct containing information   
 *             regarding the quadrant edge.
 * @param user_data Type erased <code>grace_neighbor_info_t</code> 
 *                  where information is stored.
 * This function is used as callback in <code>p4est_iterate</code> to store 
 * all the necessary information to apply interior and exterior boundary conditions.
 * In particular, this function stores, for all edges, the quadrant id's which share 
 * this edge, whether this edge is hanging or simple, whether it's internal or external,
 * its edge orientation code, the tree(s) containing the quadrants on each side, and whether
 * any of the quadrants on this edge are in the halo.
 */
void grace_iterate_edges( p8est_iter_edge_info_t* info 
                          , void* user_data ) ;
#endif 
/**************************************************************************************************/
/**************************************************************************************************/
}} /* namespace grace::amr */

#endif /* GRACE_AMR_BC_ITERATE_HH */