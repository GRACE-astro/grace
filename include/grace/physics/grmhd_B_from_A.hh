/**
 * @file grmhd_B_from_A.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-04-08
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
 * 
 */

#ifndef GRACE_PHYSICS_GRMHD_B_FROM_A_HH
#define GRACE_PHYSICS_GRMHD_B_FROM_A_HH

#include <grace_config.h>

#include <Kokkos_Core.hpp>

#include <array>

namespace grace {

/**
 * @brief Initialize B field from A following
 *        the usual prescription where A is
 *        directed along phi and proportional
 *        to a given power of pressure or density.
 */
void setup_confined_poloidal_B_field() ;

/**
 * @brief Initialize a single confined poloidal B field
 *        patch around @p center with the given cut radius
 *        and target |B|_max. Uses the Avec_ID cutoff_var,
 *        cutoff_fact and A_n parameters internally.
 */
void setup_confined_poloidal_B_field_single(
    std::array<double,3> const& center, double radius, double Btarget
) ;

}


#endif 