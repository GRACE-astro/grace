/**
 * @file partition_grid.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-24
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

#ifndef GRACE_AMR_REGRID_HELPERS_PARTITION_GRID_HH
#define GRACE_AMR_REGRID_HELPERS_PARTITION_GRID_HH

#include <grace_config.h>

#include <grace/data_structures/grace_data_structures.hh>
#include <grace/amr/p4est_headers.hh> 

#include <vector>

namespace grace { namespace amr {

std::vector<p4est_transfer_context_t *> 
grace_partition_begin(
    grace::var_array_t<GRACE_NSPACEDIM>& state, 
    grace::var_array_t<GRACE_NSPACEDIM>& state_swap,
    grace::staggered_variable_arrays_t& sstate, 
    grace::staggered_variable_arrays_t& sstate_swap
) ; 

void 
grace_partition_finalize(std::vector<p4est_transfer_context_t *>const & context) ;


}} /* namespace grace::amr */


#endif /* GRACE_AMR_REGRID_HELPERS_PARTITION_GRID_HH */