/**
 * @file particle_utilities.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Support for tracer particles
 * @date 2026-04-14
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
#ifndef GRACE_UTILS_TRACERS_HH
#define GRACE_UTILS_TRACERS_HH

#include <grace_config.h>

#include <Kokkos_Core.hpp>
#include <Cabana_Core.hpp>

namespace grace {


    
// concept 
// load balancing policy 
// methods: 
//  balance_begin() / balance_end() -> transfers particle data
//  fetch_data_begin/end()          -> transfers metric / hydro data to particle locations 

struct patch_owner_lb_policy_t {

    

} ; 


enum particle_status_flag_t: uint8_t {
    PARTICLE_DEFAULT=0,
    PARTICLE_INSIDE_BH,
    PARTICLE_OUTSIDE_DOMAIN,
    N_PARTICLE_STATUSES
} ; 

}

#endif 