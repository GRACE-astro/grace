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

namespace thunder{ namespace amr {

struct iterate_face_data_t 
{
    parallel::thunder_transfer_context_t context ;
    Kokkos::vector<int64_t> phys_bc_idx          ; 
}


}}

#endif /* THUNDER_AMR_BC_HELPERS_HH */