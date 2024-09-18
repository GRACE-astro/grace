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
#include <grace/parallel/mpi_wrappers.hh>
#include <Kokkos_Vector.hpp>

namespace grace{ namespace amr {
/**************************************************************************************************/
/**
 * @brief Initiate an asynchronous exchange of halo quadrant data.
 * 
 */
void grace_init_halo_transfer(
    parallel::grace_transfer_context_t& context       ,
    p4est_ghost_t*                     halos          ,
    sc_array_view_t<p4est_quadrant_t>& halo_quads     , 
    sc_array_view_t<p4est_quadrant_t>& mirror_quads   ,
    grace::var_array_t<GRACE_NSPACEDIM>& halo          , 
    grace::staggered_variable_arrays_t& staggered_halo,
    cell_vol_array_t<GRACE_NSPACEDIM>&  halo_vols     ,
    grace::var_array_t<GRACE_NSPACEDIM>& vars          , 
    grace::staggered_variable_arrays_t& staggered_vars,
    cell_vol_array_t<GRACE_NSPACEDIM>&  vols          ,
    bool exchange_vols 
);
/**************************************************************************************************/
/**
 * @brief Initiate an asynchronous exchange of halo quadrant data.
 */
void grace_init_halo_transfer_custom(
    parallel::grace_transfer_context_t& context       ,
    std::vector<int64_t> const& snd_quadid            ,
    std::vector<int64_t> const& rcv_quadid            ,  
    std::vector<std::set<int>> const& snd_procid      , 
    std::vector<int>     const& rcv_procid            ,
    grace::var_array_t<GRACE_NSPACEDIM>& halo          , 
    grace::staggered_variable_arrays_t& staggered_halo,
    cell_vol_array_t<GRACE_NSPACEDIM>&  halo_vols     ,
    grace::var_array_t<GRACE_NSPACEDIM>& vars          , 
    grace::staggered_variable_arrays_t& staggered_vars,
    cell_vol_array_t<GRACE_NSPACEDIM>&  vols          ,
    bool exchange_cell_volumes
);
/**************************************************************************************************/
/**
 * @brief Call mpi_waitall() and wait for all transfer in progress to complete.
 * @param context The MPI context.
 */
void grace_finalize_halo_transfer(parallel::grace_transfer_context_t& context) ;
/**************************************************************************************************/
}}

#endif /* GRACE_AMR_BC_HELPERS_HH */