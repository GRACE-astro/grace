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

#ifndef GRACE_AMR_BC_COPY_HH 
#define GRACE_AMR_BC_COPY_HH

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
/**************************************************************************************************/
/**
 * @brief Copy ghostzones across simple interior faces.
 * \ingroup amr
 * @param vars State array.
 * @param halo Halo quadrants. 
 * @param staggered_vars Staggered state array.
 * @param staggered_halo Staggered halo array.
 * @param interior_faces Information about the faces where data
 *                       needs to be copied.
 * @param interior_corners Information about corners across which 
 *                         data needs to be copied 
 * @param interior_edges Information about edges across which 
 *                       data needs to be copied 
 * Interior faces are the ones that do not face the grid boundary,
 * simple faces are those where quadrants on each side are at the 
 * same refinement level.
 */
void copy_interior_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& vars
    , grace::var_array_t<GRACE_NSPACEDIM>& halo
    , grace::staggered_variable_arrays_t& staggered_state
    , grace::staggered_variable_arrays_t& staggered_halo
    , Kokkos::vector<simple_face_info_t>&  interior_faces 
    , Kokkos::vector<simple_corner_info_t>&  interior_corners 
    #ifdef GRACE_3D 
    , Kokkos::vector<simple_edge_info_t>&  interior_edges 
    #endif 
    ) ;
/**************************************************************************************************/
/**
 * @brief Copy ghostzones of cell-centered variables across simple interior faces.
 * \ingroup amr
 * @param vars State array.
 * @param halo Halo quadrants. 
 * @param interior_faces Information about the faces where data
 *                       needs to be copied.
 * @param interior_corners Information about corners across which 
 *                         data needs to be copied 
 * @param interior_edges Information about edges across which 
 *                       data needs to be copied
 * Interior faces are the ones that do not face the grid boundary,
 * simple faces are those where quadrants on each side are at the 
 * same refinement level.
 */
void copy_interior_ghostzones_cell_centers(
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
 * @brief Copy ghostzones of corner-centered variables across simple interior faces.
 * \ingroup amr
 * @param vars Staggered state array.
 * @param halo Staggered halo quadrants. 
 * @param interior_faces Information about the faces where data
 *                       needs to be copied.
 * @param interior_corners Information about corners across which 
 *                         data needs to be copied 
 * @param interior_edges Information about edges across which 
 *                       data needs to be copied
 * Interior faces are the ones that do not face the grid boundary,
 * simple faces are those where quadrants on each side are at the 
 * same refinement level.
 */
void copy_interior_ghostzones_corners(
      grace::var_array_t<GRACE_NSPACEDIM>& vars
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , Kokkos::vector<simple_face_info_t>& interior_faces
    , Kokkos::vector<simple_corner_info_t>& interior_corners
    #ifdef GRACE_3D
    , Kokkos::vector<simple_edge_info_t>& interior_edges
    #endif 
) ; 
/**************************************************************************************************/
/**************************************************************************************************/
}}

#endif /* GRACE_AMR_BC_COPY_HH */