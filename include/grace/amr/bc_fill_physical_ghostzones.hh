/**
 * @file bc_fill_physical_ghostzones.cpp
 * @author  Carlo Musolino
 * @brief 
 * @date 2024-09-17
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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
#ifndef GRACE_AMR_BC_FILL_PHYSICAL_GHOSTZONES_HH
#define GRACE_AMR_BC_FILL_PHYSICAL_GHOSTZONES_HH

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/device/device_vector.hh> 

#include <Kokkos_Core.hpp>


namespace grace { namespace amr {

/**
 * @brief Fill the ghostzones outside of the grid.
 * \ingroup amr 
 * @param vars State array.
 * @param staggered_vars Staggered state array.
 * @param face_phys_bc   Information about faces of the grid.
 * @param corner_phys_bc Information about corners of the grid.
 * @param edge_phys_bc   Information about edges of the grid.
 */
void fill_physical_boundaries(
  grace::var_array_t<GRACE_NSPACEDIM>& vars 
  , grace::var_array_t<GRACE_NSPACEDIM>& old_vars 
  , grace::staggered_variable_arrays_t& staggered_vars 
  , grace::staggered_variable_arrays_t& old_staggered_vars 
  , grace::device_vector<grace::amr::grace_phys_bc_info_t>& face_phys_bc 
  , grace::device_vector<grace::amr::grace_phys_bc_info_t>& corner_phys_bc
  #ifdef GRACE_3D 
  , grace::device_vector<grace::amr::grace_phys_bc_info_t>& edge_phys_bc 
  #endif
  , double const dt, double const dtfact
) ; 

}} /* namespace grace::amr */
#endif /* GRACE_AMR_BC_FILL_PHYSICAL_GHOSTZONES_HH */
