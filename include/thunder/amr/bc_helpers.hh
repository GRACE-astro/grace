/**
 * @file bc_helpers.hh
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

#ifndef THUNDER_AMR_BC_HELPERS_HH 
#define THUNDER_AMR_BC_HELPERS_HH

#include <thunder_config.h>

#include <thunder/parallel/mpi_wrappers.hh>
#include <thunder/amr/p4est_headers.hh>
#include <thunder/utils/interpolators.hh>
#include <thunder/amr/boundary_conditions.hh>
#include <thunder/utils/prolongation.hh>
#include <thunder/utils/limiters.hh> 
#include <thunder/data_structures/variable_properties.hh>

#include <Kokkos_Vector.hpp>

namespace thunder{ namespace amr {

void thunder_iterate_faces( p4est_iter_face_info_t* info 
                          , void* user_data ) ;


void copy_interior_ghostzones(
      thunder::var_array_t<THUNDER_NSPACEDIM>& 
    , thunder::var_array_t<THUNDER_NSPACEDIM>&  
    , Kokkos::vector<simple_face_info_t>&  ) ;

void restrict_hanging_ghostzones(
      thunder::var_array_t<THUNDER_NSPACEDIM>& 
    , thunder::var_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
) ;

template< typename InterpT >
void prolongate_hanging_ghostzones(
      thunder::var_array_t<THUNDER_NSPACEDIM>& 
    , thunder::var_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
) ;

extern template void 
prolongate_hanging_ghostzones<utils::linear_prolongator_t<thunder::minmod>>(
      thunder::var_array_t<THUNDER_NSPACEDIM>& 
    , thunder::var_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
) ;

extern template void 
prolongate_hanging_ghostzones<utils::linear_prolongator_t<thunder::MCbeta>>(
      thunder::var_array_t<THUNDER_NSPACEDIM>& 
    , thunder::var_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::scalar_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , thunder::cell_vol_array_t<THUNDER_NSPACEDIM>&  
    , Kokkos::vector<hanging_face_info_t>& 
) ; 

}}

#endif /* THUNDER_AMR_BC_HELPERS_HH */