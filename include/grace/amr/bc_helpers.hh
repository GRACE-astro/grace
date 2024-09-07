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

#ifndef GRACE_AMR_BC_HELPERS_HH 
#define GRACE_AMR_BC_HELPERS_HH

#include <grace_config.h>

#include <grace/parallel/mpi_wrappers.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/utils/interpolators.hh>
#include <grace/amr/boundary_conditions.hh>
#include <grace/utils/prolongation.hh>
#include <grace/utils/limiters.hh> 
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
/**
 * @brief Copy ghostzones across simple interior faces.
 * \ingroup amr
 * @param vars State array.
 * @param halo Halo quadrants. 
 * @param interior_faces Information about the faces where data
 *                       needs to be copied.
 * Interior faces are the ones that do not face the grid boundary,
 * simple faces are those where quadrants on each side are at the 
 * same refinement level.
 */
void copy_interior_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& vars
    , grace::var_array_t<GRACE_NSPACEDIM>& halo
    , Kokkos::vector<simple_face_info_t>&  interior_faces 
    , Kokkos::vector<simple_corner_info_t>&  interior_corners 
    #ifdef GRACE_3D 
    , Kokkos::vector<simple_edge_info_t>&  interior_edges 
    #endif 
    ) ;
/**************************************************************************************************/
/**
 * @brief Restrict all variables to ghostzones across all hanging faces.
 * \ingroup amr
 * @param state State array.
 * @param halo  Halo quadrants.
 * @param vols  Cell volumes.
 * @param halo_vols Halo cell volumes.
 * @param hanging_faces Information on hanging face neighbors.
 * @param hanging_corners Information on hanging corner neighbors.
 * @param hanging_edges Information on hanging edge neighbors.
 * Hanging faces are those where quadrants are not at the same refinement level
 * on the two sides. This function performs restriction of all variables on the 
 * coarse side of all hanging faces. Restriction is performed in a second order 
 * accurate, volume-average preserving way. 
 */
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
) ;
/**************************************************************************************************/
/**
 * @brief Prolongate all variables to ghostzones across all hanging faces.
 * \ingroup amr
 * @tparam InterpT Prolongator.
 * @param state State array.
 * @param halo  Halo quadrants.
 * @param vols  Cell volumes.
 * @param halo_vols Halo cell volumes.
 * @param hanging_faces Information on hanging faces.
 * @param hanging_corners Information on hanging corner neighbors.
 * @param hanging_edges Information on hanging edge neighbors.
 * Hanging faces are those where quadrants are not at the same refinement level
 * on the two sides. This function performs prolongation of all variables on the 
 * fine side of all hanging faces. By default prolongation is performed in a second order 
 * accurate, volume-average preserving way using a slope limited second order interpolation
 * scheme. The default slope limiter is \ref minmod.
 */
template< typename InterpT >
void prolongate_hanging_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols_halo
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
    , Kokkos::vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>& hanging_edges
    #endif
) ;
/**************************************************************************************************/
/*                                  Instantiate templates                                         */
/**************************************************************************************************/
extern template void 
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
/**************************************************************************************************/
extern template void 
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
/**************************************************************************************************/
}}

#endif /* GRACE_AMR_BC_HELPERS_HH */