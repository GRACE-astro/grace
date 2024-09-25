/**
 * @file grid_transfer_operators.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-25
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

#ifndef GRACE_AMR_REGRID_HELPERS_GRID_TRANSFER_OPERATORS_HH
#define GRACE_AMR_REGRID_HELPERS_GRID_TRANSFER_OPERATORS_HH

#include <grace_config.h>

#include <grace/amr/regrid_helpers.hh>
#include <grace/amr/amr_functions.hh> 
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/config/config_parser.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/device_vector.hh>
#include <grace/utils/limiters.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/amr/p4est_headers.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <string> 

namespace grace { namespace amr {

void grace_prolongate_refined_quadrants(
    grace::var_array_t<GRACE_NSPACEDIM>& state,
    grace::var_array_t<GRACE_NSPACEDIM>& state_swap,
    grace::staggered_variable_arrays_t & sstate,
    grace::staggered_variable_arrays_t & sstate_swap,
    grace::cell_vol_array_t<GRACE_NSPACEDIM> in_vol,
    grace::device_vector<int> const& refine_incoming,
    grace::device_vector<int> const& refine_outgoing
) ;

void grace_restrict_coarsened_quadrants(
    grace::var_array_t<GRACE_NSPACEDIM>& state,
    grace::var_array_t<GRACE_NSPACEDIM>& state_swap,
    grace::staggered_variable_arrays_t & sstate,
    grace::staggered_variable_arrays_t & sstate_swap,
    grace::cell_vol_array_t<GRACE_NSPACEDIM> out_vol,
    grace::device_vector<int> const& coarsen_incoming,
    grace::device_vector<int> const& coarsen_outgoing,
) ; 

template< typename LimT > 
void prolongate_variables_cell_centered(
    grace::var_array_t<GRACE_NSPACEDIM>& in_state,
    grace::var_array_t<GRACE_NSPACEDIM>& out_state,
    grace::cell_vol_array_t<GRACE_NSPACEDIM> in_vol,
    grace::device_vector<int> const& in_idx,
    grace::device_vector<int> const& out_idx
) ; 


template< int order > 
void prolongate_variables_corner_staggered(
    grace::var_array_t<GRACE_NSPACEDIM>& in_state,
    grace::var_array_t<GRACE_NSPACEDIM>& out_state,
    grace::device_vector<int> const& in_idx,
    grace::device_vector<int> const& out_idx
) ; 

void restrict_variables_cell_centered(
    grace::var_array_t<GRACE_NSPACEDIM>& in_state,
    grace::var_array_t<GRACE_NSPACEDIM>& out_state,
    grace::cell_vol_array_t<GRACE_NSPACEDIM> out_vol,
    grace::device_vector<int> const& in_idx,
    grace::device_vector<int> const& out_idx
) ; 

void restrict_variables_corner_staggered(
    grace::var_array_t<GRACE_NSPACEDIM>& in_state,
    grace::var_array_t<GRACE_NSPACEDIM>& out_state,
    grace::cell_vol_array_t<GRACE_NSPACEDIM> out_vol,
    grace::device_vector<int> const& in_idx,
    grace::device_vector<int> const& out_idx
) ; 

/***********************************************************/
/*                  Intantiate templates                   */
/***********************************************************/
#define INSTANTIATE_TEMPLATES(limiter,order)        \
extern template                                     \
void prolongate_variables_cell_centered<limiter>(   \
    grace::var_array_t<GRACE_NSPACEDIM>& ,          \
    grace::var_array_t<GRACE_NSPACEDIM>& ,          \
    grace::cell_vol_array_t<GRACE_NSPACEDIM> ,      \
    grace::device_vector<int> const& ,              \
    grace::device_vector<int> const&                \
) ;                                                 \
extern template                                     \
void prolongate_variables_corner_staggered<order>(  \
    grace::var_array_t<GRACE_NSPACEDIM>& ,          \
    grace::var_array_t<GRACE_NSPACEDIM>& ,          \
    grace::device_vector<int> const& ,              \
    grace::device_vector<int> const&                \
) 

INSTANTIATE_TEMPLATE(grace::minmod, 2) ; 
INSTANTIATE_TEMPLATE(grace::minmod, 4) ; 
INSTANTIATE_TEMPLATE(grace::MCBeta, 2) ; 
INSTANTIATE_TEMPLATE(grace::MCBeta, 4) ;
}}

#endif /* GRACE_AMR_REGRID_HELPERS_GRID_TRANSFER_OPERATORS_HH */