/**
 * @file boundary_conditions.hh
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

#ifndef GRACE_AMR_BC_HH 
#define GRACE_AMR_BC_HH 

#include <grace/data_structures/variable_properties.hh>

#include <vector> 
#include <set> 

namespace grace { namespace amr {

/**
 * @brief Apply all boundary conditions and fill ghostzones on state array.
 * \ingroup amr
 * This function fills all the ghost-cells in the <code>state</code> array.
 * This includes applying physical boundary conditions at domain edges, as 
 * well as filling all internal ghost-zones across simple and hanging faces.
 * The state array needs to be in a valid state at all interior points when entering 
 * this function. No assumptions are made on the content of the ghostzones of each
 * quadrant and each variable for the state array when entering this function. Auxiliries
 * and scratch state are left untouched by this function, unless a non-trivial boundary 
 * condition is requested on an auxiliary variable. When this function returns, \b all 
 * ghostzones for all quadrants are in a valid state for the <code>state</code> array.
 * All interior ghost-zones operations are guaranteed to be second order accurate, total 
 * variation diminishing, and volume average preserving.
 */
void apply_boundary_conditions() ;
/**
 * @brief Apply all boundary conditions on the var array.
 * \ingroup amr
 * @param vars The state array where BCs are applied.
 * Specialized version of \ref apply_boundary_conditions which allows 
 * the caller to specify which state array needs its ghostzones to be filled.
 */
void apply_boundary_conditions(grace::var_array_t<GRACE_NSPACEDIM>& vars) ;


}} /* namespace grace::amr */

#endif /* GRACE_AMR_BC_HH */
