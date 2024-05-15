/**
 * @file thunder_physical_systems.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-15
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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
#ifndef THUNDER_PHYSICS_THUNDER_PHYSICAL_SYSTEMS_HH
#define THUNDER_PHYSICS_THUNDER_PHYSICAL_SYSTEMS_HH

#include <thunder_config.h> 

#ifdef THUNDER_ENABLE_BURGERS
#include<thunder/physics/burgers.hh>
#endif 
#ifdef THUNDER_ENABLE_SCALAR_ADV
#include <thunder/physics/scalar_advection.hh>
#endif 


#endif /* THUNDER_PHYSICS_THUNDER_PHYSICAL_SYSTEMS_HH */