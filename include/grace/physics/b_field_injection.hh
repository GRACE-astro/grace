/**
 * @file b_field_injection.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Inject a confined poloidal B field at a user-specified
 *        simulation time, using co_tracker locations as the
 *        centers. Reuses the existing Avec_ID prescription.
 * @date 2026-04-18
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
 */

#ifndef GRACE_PHYSICS_B_FIELD_INJECTION_HH
#define GRACE_PHYSICS_B_FIELD_INJECTION_HH

#include <grace_config.h>

namespace grace {

/**
 * @brief If enabled and not yet fired and the current time is past
 *        the configured trigger, inject a confined poloidal B field
 *        at each currently tracked compact object and update the
 *        conservative variables. Otherwise a no-op.
 */
void maybe_inject_b_field() ;

/**
 * @brief Has the mid-run B-field injection already fired?
 */
bool b_field_injection_has_fired() ;

/**
 * @brief Set the fired flag (used by checkpoint restore).
 */
void b_field_injection_set_fired(bool v) ;

}

#endif /* GRACE_PHYSICS_B_FIELD_INJECTION_HH */
