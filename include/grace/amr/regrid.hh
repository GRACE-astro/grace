/**
 * @file regrid.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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
#ifndef GRACE_AMR_REGRID_HH
#define GRACE_AMR_REGRID_HH 
/* config file */ 
#include <grace_config.h>
/* grace includes */
#include <grace/system/grace_system.hh>
#include <grace/utils/grace_utils.hh>
/* grace::amr includes */
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/quadrant.hh>
#include <grace/amr/tree.hh>
#include <grace/amr/connectivity.hh>
#include <grace/amr/forest.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/amr/amr_flags.hh>
#include <grace/amr/amr_functions.hh>

namespace grace { namespace amr { 

/**
 * @brief Perform a regridding operation.
 * \ingroup amr
 * This function refines and coarsens the grid based on 
 * the user-provided refinement criterion, it prolongates 
 * and restricts state variables on the new grid structure,
 * and then partitions the grid in parallel. Auxiliary variables
 * are re-allocated according to the new grid structure but 
 * they are not re-computed, and the aux array is empty coming
 * out of this routine. The coordinates are re-computed on the 
 * new grid.
 */
void regrid() ; 

/**
 * @brief Perform a regridding operation given the criterion and
 *        the variable on which it should be applied.
 * \ingroup amr
 * 
 * @param criterion Regridding criterion, consult documentation for options.
 * @param var       Regridding variable.
 * 
 * This function refines and coarsens the grid based on 
 * the user-provided refinement criterion, it prolongates 
 * and restricts state variables on the new grid structure,
 * and then partitions the grid in parallel. Auxiliary variables
 * are re-allocated according to the new grid structure but 
 * they are not re-computed, and the aux array is empty coming
 * out of this routine. The coordinates are re-computed on the 
 * new grid.
 */
void regrid(std::string const& criterion, std::string const& var) ; 

/**
 * @brief Set the all quadrants to DEFAULT_STATE
 * \ingroup amr 
 * \cond grace_detail
 */
void set_quadrants_to_default(); 

}} /* namespace grace::amr */ 

#endif /* GRACE_AMR_REGRID_HH */