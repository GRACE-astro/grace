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

#ifndef GRACE_AMR_BC_RESTRICTION_HH 
#define GRACE_AMR_BC_RESTRICTION_HH

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
 * @brief Restrict all variables to ghostzones across all hanging faces.
 * \ingroup amr
 * @param state State array.
 * @param halo  Halo quadrants.
 * @param staggered_state Staggered state array.
 * @param staggered_hall  Staggered halo array. 
 * @param vols  Cell volumes.
 * @param halo_vols Halo cell volumes.
 * @param hanging_faces Information on hanging face neighbors.
 * @param hanging_corners Information on hanging corner neighbors.
 * @param hanging_edges Information on hanging edge neighbors.
 * Hanging faces are those where quadrants are not at the same refinement level
 * on the two sides. This function performs restriction of all variables on the 
 * coarse side of all hanging faces. Restriction is performed in a second order 
 * accurate, volume-average preserving way for cell-centered variables. Corner 
 * staggered variables are simply copied from the corresponding fine point. 
 */
void restrict_hanging_ghostzones(
      var_array_t<GRACE_NSPACEDIM>& state
    , var_array_t<GRACE_NSPACEDIM>& halo 
    , staggered_variable_arrays_t& staggered_state
    , staggered_variable_arrays_t& staggered_halo 
    , cell_vol_array_t<GRACE_NSPACEDIM>& vols 
    , cell_vol_array_t<GRACE_NSPACEDIM>& halo_vols 
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
    , Kokkos::vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>& hanging_edges
    #endif 
);
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
void restrict_hanging_ghostzones_cell_centers(
      var_array_t<GRACE_NSPACEDIM>& state
    , var_array_t<GRACE_NSPACEDIM>& halo
    , cell_vol_array_t<GRACE_NSPACEDIM>& vols
    , cell_vol_array_t<GRACE_NSPACEDIM>& halo_vols
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
    , Kokkos::vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>& hanging_edges
    #endif 
) ;
/**************************************************************************************************/
/**
 * @brief Restrict all variables to ghostzones across all hanging faces.
 * \ingroup amr
 * @param state Staggered state array.
 * @param halo  Staggered halo quadrants.
 * @param vols  Cell volumes.
 * @param halo_vols Halo cell volumes.
 * @param hanging_faces Information on hanging face neighbors.
 * @param hanging_corners Information on hanging corner neighbors.
 * @param hanging_edges Information on hanging edge neighbors.
 * Hanging faces are those where quadrants are not at the same refinement level
 * on the two sides. This function performs restriction of all variables on the 
 * coarse side of all hanging faces. 
 */
void restrict_hanging_ghostzones_corners(
      var_array_t<GRACE_NSPACEDIM>& state
    , var_array_t<GRACE_NSPACEDIM>& halo 
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
    , Kokkos::vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>& hanging_edges
    #endif 
)  ;
/**************************************************************************************************/
/**************************************************************************************************/
}}

#endif /* GRACE_AMR_BC_RESTRICTION_HH */