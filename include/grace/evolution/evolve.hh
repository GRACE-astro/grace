/**
 * @file evolve.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-13
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

#ifndef GRACE_EVOLVE_HH
#define GRACE_EVOLVE_HH
//*****************************************************************************************************
#include <grace_config.h>
//*****************************************************************************************************
#include <grace/data_structures/variable_properties.hh>
//*****************************************************************************************************
#include <grace/physics/eos/eos_types.hh>
//*****************************************************************************************************
namespace grace {
//*****************************************************************************************************
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Perform a timestep.
 * \ingroup evol
 * This function advances all variables in the state array by a full timestep. The timestep size is 
 * controlled by the function \ref find_stable_timestep. The kind of timestepper used is controlled 
 * by the parameter evolution::time_stepper. Coming out of this routine all the variables in the state
 * array are in a valid state (at all gridpoints) and at time \f$t+dt\f$. Auxiliaries are not filled 
 * by this function and neither is the scratched space, both of which are left in an invalid state. 
 * This function assumes that the state is in a valid state as input, for all gridpoints including 
 * ghostzones. It also assumes that auxiliaries are filled at all gridpoints and up to date w.r.t. the 
 * evolution time.
 */
void evolve() ; 
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Perform a timestep.
 * @tparam eos_t Type of the active EOS.
 * \ingroup evol
 * \cond grace_detail
 * This function implements the actual evolution for a concrete EOS type.
 */
template< typename eos_t >
void evolve_impl() ; 
//*****************************************************************************************************

//*****************************************************************************************************
/**
 * @brief Advance all variables by a substep.
 * \ingroup evol
 * @tparam eos_t Type of active EOS.
 * @param t Current time.
 * @param dt Timestep size.
 * @param dtfact Timestep factor.
 * @param state  State array.
 * @param state_p Scratch state array.
 * @param aux Auxiliaries.
 * @param cvol Cell volumes.
 * @param surfs_and_edges Cell face surfaces and edge lengths.
 * 
 * This routine advances all variables by a substep. It is agnostic to the time-stepper used 
 * and assumes that all input variable arrays are in a valid state at all gridcells. The output 
 * is applied in-place on <code>state</code>, whereas <code>state_p</code> and <code>aux</code>
 * are left unchanged.
 */
template< typename eos_t >
void advance_substep( double const t, double const dt, double const dtfact 
                    , grace::var_array_t<GRACE_NSPACEDIM>& state 
                    , grace::var_array_t<GRACE_NSPACEDIM>& state_p 
                    , grace::var_array_t<GRACE_NSPACEDIM>& aux 
                    , grace::cell_vol_array_t<GRACE_NSPACEDIM>& cvol
                    , grace::staggered_coordinate_arrays_t& surfs_and_edges ) ; 
//*****************************************************************************************************
//*****************************************************************************************************
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                     \
extern template                                                       \
void advance_substep<EOS>( double const , double const , double const \
                         , grace::var_array_t<GRACE_NSPACEDIM>&       \
                         , grace::var_array_t<GRACE_NSPACEDIM>&       \
                         , grace::var_array_t<GRACE_NSPACEDIM>&       \
                         , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  \
                         , grace::staggered_coordinate_arrays_t&  ) ; \
extern template                                                       \
void evolve_impl<EOS>()

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
} /* namespace grace */
//*****************************************************************************************************
#endif /* GRACE_EVOLVE_HH */