/**
 * @file find_stable_timestep.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-16
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

#ifndef GRACE_EVOLUTION_FIND_STABLE_TIMESTEP_HH
#define GRACE_EVOLUTION_FIND_STABLE_TIMESTEP_HH

#include <grace_config.h> 
#include <grace/physics/eos/eos_types.hh>

namespace grace {
//*****************************************************************************************************
/**
 * @brief Find a stable timestep by computing the maximum eigenspeed
 *        over the whole grid. Notice that the timestep is modified by this 
 *        function to be the maximum stable timestep over the whole grid 
 *        multiplied by the cfl_factor parameter.
 * \ingroup evol
 */
void find_stable_timestep() ;
//*****************************************************************************************************
/**
 * @brief Implementation of find_stable_timestep for a concrete EOS type.
 * \ingroup evol
 * \cond grace_detail
 * @tparam eos_t Type of concrete EOS.
 */
template< typename eos_t >
void find_stable_timestep_impl() ;  
//*****************************************************************************************************
#define INSTANTIATE_TEMPLATE(EOS)     \
extern template                       \
void find_stable_timestep_impl<EOS>()
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
#undef INSTANTIATE_TEMPLATE
}

#endif 