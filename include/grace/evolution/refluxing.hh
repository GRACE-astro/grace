/**
 * @file refluxing.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-10-16
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
#ifndef GRACE_EVOLUTION_REFLUXING_HH
#define GRACE_EVOLUTION_REFLUXING_HH
#include <grace_config.h>

#include <grace/amr/amr_ghosts.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/amr/amr_functions.hh>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

namespace grace {

//*****************************************************************************************************
/** @brief Fill flux buffers for refluxing
 * @return Transfer context containing send and receive requests for fluxes.
 * \ingroup evol
 */
parallel::grace_transfer_context_t reflux_fill_flux_buffers();
//*****************************************************************************************************
/** @brief Fill emf buffers for refluxing
 * @return Transfer context containing send and receive requests for emfs.
 * \ingroup evol
 */
parallel::grace_transfer_context_t reflux_fill_emf_buffers() ; 
//*****************************************************************************************************
/** @brief Correct fluxes at fine-coarse interfaces
 * @param context Transfer context
 * @param t Time 
 * @param dt Time step 
 * @param dtfact Time step factor
 * @param new_state New state 
 * \ingroup evol
*/
void reflux_correct_fluxes(
    parallel::grace_transfer_context_t& context,
    double t, double dt, double dtfact,
    var_array_t & new_state 
) ; 
//*****************************************************************************************************
/** @brief Correct EMFs at fine-coarse interfaces
 * @param context Transfer context
 * \ingroup evol
*/
void reflux_correct_emfs(
    parallel::grace_transfer_context_t& context
) ;
//*****************************************************************************************************
//*****************************************************************************************************
} /* namespace grace */

#endif /*GRACE_EVOLUTION_REFLUXING_HH*/