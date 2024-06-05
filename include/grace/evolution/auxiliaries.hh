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
 * @param state The state to be used to compute auxiliaries.
 * @param aux   The array where to store computed aux variables.
 */
void compute_auxiliary_quantities(
      grace::var_array_t<GRACE_NSPACEDIM>& state
    , grace::var_array_t<GRACE_NSPACEDIM>& aux 
) ; 
//*****************************************************************************************************
}

#endif /* GRACE_EVOLUTION_AUXILIARIES_HH */