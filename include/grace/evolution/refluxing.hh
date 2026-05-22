/**
 * @file refluxing.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Free-function entry points for face-flux and constrained-transport EMF refluxing at coarse-fine AMR interfaces.
 * @date 2025-10-16
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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
 * \ingroup evolution
 */
parallel::grace_transfer_context_t reflux_fill_flux_buffers();
//*****************************************************************************************************
/** @brief Fill emf buffers for refluxing
 * @return Transfer context containing send and receive requests for emfs.
 * \ingroup evolution
 */
parallel::grace_transfer_context_t reflux_fill_emf_buffers() ; 
//*****************************************************************************************************
/** @brief Replace coarse-side fluxes at fine-coarse interfaces with the
 *         area-averaged fine fluxes, in place on the flux array.
 *
 *  Must be called AFTER `compute_fluxes` and AFTER `reflux_fill_flux_buffers`,
 *  but BEFORE `add_fluxes_and_source_terms` (the divergence update that
 *  consumes the flux array).  Mirrors the `reflux_correct_emfs` pattern.
 *  Replacing the flux directly (rather than patching `new_state` after the
 *  fact) makes the operation race-free: each (qid, idir, side, i, j, ivar)
 *  slot in the flux array is written by at most one descriptor, so the
 *  kernel needs no atomics and the result is bit-invariant under MPI
 *  repartition.
 *
 * @param context Transfer context populated by `reflux_fill_flux_buffers`.
 * \ingroup evolution
*/
void reflux_correct_fluxes(
    parallel::grace_transfer_context_t& context
) ;
//*****************************************************************************************************
/** @brief Correct EMFs at fine-coarse interfaces
 * @param context Transfer context
 * \ingroup evolution
*/
void reflux_correct_emfs(
    parallel::grace_transfer_context_t& context
) ;
//*****************************************************************************************************
//*****************************************************************************************************
} /* namespace grace */

#endif /*GRACE_EVOLUTION_REFLUXING_HH*/