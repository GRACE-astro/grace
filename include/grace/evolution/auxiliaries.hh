/**
 * @file auxiliaries.hh
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

#ifndef GRACE_EVOLUTION_AUXILIARIES_HH
#define GRACE_EVOLUTION_AUXILIARIES_HH

#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/eos/eos_types.hh>

namespace grace {
//*****************************************************************************************************
/**
 * @brief Fill the <code>aux</code> array.
 * \ingroup evol
 */
void compute_auxiliary_quantities() ; 
//*****************************************************************************************************
/**
 * @brief Fill the <code>aux</code> array
 * \ingroup evol
 * @tparam eos_t Type of active EOS.
 * @param state The state to be used to compute auxiliaries.
 * @param aux   The array where to store computed aux variables.
 */
template< typename eos_t >
void compute_auxiliary_quantities(
      grace::var_array_t& state
    , grace::staggered_variable_arrays_t& sstate
    , grace::var_array_t& aux 
) ; 
//*****************************************************************************************************
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                       \
extern template                                                         \
void compute_auxiliary_quantities<EOS>(                                 \
                           grace::var_array_t&         \
                         , grace::staggered_variable_arrays_t& \
                         , grace::var_array_t& aux )

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE
}

#endif /* GRACE_EVOLUTION_AUXILIARIES_HH */