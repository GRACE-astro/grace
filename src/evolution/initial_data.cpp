/**
 * @file initial_data.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-15
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

#include <grace_config.h>

#include <grace/evolution/initial_data.hh>

#include <grace/physics/grace_physical_systems.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>

#include <Kokkos_Core.hpp>

namespace grace {

void set_initial_data() {
    Kokkos::Profiling::pushRegion("ID") ; 
    using namespace grace ;

    #ifdef GRACE_ENABLE_SCALAR_ADV 
    set_scalar_advection_initial_data() ; 
    #endif 
    #ifdef GRACE_ENABLE_BURGERS
    set_burgers_initial_data() ; 
    #endif 
    Kokkos::Profiling::popRegion() ; 
} 

}