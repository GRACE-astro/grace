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

#ifndef GRACE_AMR_BC_PROLONGATION_HH 
#define GRACE_AMR_BC_PROLONGATION_HH

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
 * @brief Prolongate all variables to ghostzones across all hanging faces.
 * \ingroup amr
 * @param state State array.
 * @param halo  Halo quadrants.
 * @param staggered_state staggered state.
 * @param staggered_halo  halo for staggered vars.
 * @param vols  Cell volumes.
 * @param halo_vols Halo cell volumes.
 * @param hanging_faces Information on hanging faces.
 * @param hanging_corners Information on hanging corner neighbors.
 * @param hanging_edges Information on hanging edge neighbors.
 * Hanging faces are those where quadrants are not at the same refinement level
 * on the two sides. This function performs prolongation of all variables on the 
 * fine side of all hanging faces. By default prolongation is performed in a second order 
 * accurate, volume-average preserving way using a slope limited second order interpolation
 * scheme for cell centered vars. The default slope limiter is \ref minmod. 
 * Edge and corner staggered vars are interpolated by Lagrange interpolation of customizable order. 
 */
void prolongate_hanging_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::staggered_variable_arrays_t& staggered_state
    , grace::staggered_variable_arrays_t& staggered_halo
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols 
    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& vols_halo
    , Kokkos::vector<hanging_face_info_t>& hanging_faces
    , Kokkos::vector<hanging_corner_info_t>& hanging_corners
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>& hanging_edges
    #endif
) ;
/**************************************************************************************************/
/**
 * @brief Prolongate all cell-centered variables to ghostzones across all hanging faces.
 * \ingroup amr
 * @tparam InterpT Prolongator.
 * @param state State array.
 * @param halo  Halo quadrants.
 * @param vols  Cell volumes.
 * @param halo_vols Halo cell volumes.
 * @param hanging_faces Information on hanging faces.
 * @param hanging_corners Information on hanging corner neighbors.
 * @param hanging_edges Information on hanging edge neighbors.
 * The interpolator should be a second order slope-limited method. The slope limiter used 
 * defaults to \ref minmod and can be customized.
 */
template< typename InterpT >
void prolongate_hanging_ghostzones_cell_centers(
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
/**
 * @brief Prolongate all cell-centered variables to ghostzones across all hanging faces.
 * \ingroup amr
 * @tparam InterpT Prolongator.
 * @param state State array.
 * @param halo  Halo quadrants.
 * @param vols  Cell volumes.
 * @param halo_vols Halo cell volumes.
 * @param hanging_faces Information on hanging faces.
 * @param hanging_corners Information on hanging corner neighbors.
 * @param hanging_edges Information on hanging edge neighbors.
 * The interpolator must be a Lagrange interpolator (or one with a compatible API)
 * of a certain order.
 */
template< typename InterpT > 
void prolongate_hanging_ghostzones_corners(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
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
prolongate_hanging_ghostzones_cell_centers<utils::linear_prolongator_t<grace::minmod>>(
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
prolongate_hanging_ghostzones_cell_centers<utils::linear_prolongator_t<grace::MCbeta>>(
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
prolongate_hanging_ghostzones_corners<utils::lagrange_prolongator_t<2>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&   
    , Kokkos::vector<hanging_face_info_t>& 
    , Kokkos::vector<hanging_corner_info_t>&
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>&
    #endif
) ;
/**************************************************************************************************/ 
extern template void 
prolongate_hanging_ghostzones_corners<utils::lagrange_prolongator_t<4>>(
      grace::var_array_t<GRACE_NSPACEDIM>& 
    , grace::var_array_t<GRACE_NSPACEDIM>&   
    , Kokkos::vector<hanging_face_info_t>& 
    , Kokkos::vector<hanging_corner_info_t>&
    #ifdef GRACE_3D 
    , Kokkos::vector<hanging_edge_info_t>&
    #endif
) ; 
/**************************************************************************************************/
}}

#endif /* GRACE_AMR_BC_PROLONGATION_HH */