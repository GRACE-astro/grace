/**
 * @file regrid.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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
#ifndef THUNDER_AMR_REGRID_HH
#define THUNDER_AMR_REGRID_HH 
/* config file */ 
#include <thunder_config.h>
/* thunder includes */
#include <thunder/system/thunder_system.hh>
#include <thunder/utils/thunder_utils.hh>
/* thunder::amr includes */
#include <thunder/amr/p4est_headers.hh>
#include <thunder/amr/quadrant.hh>
#include <thunder/amr/tree.hh>
#include <thunder/amr/connectivity.hh>
#include <thunder/amr/forest.hh>
#include <thunder/amr/coordinates.hh>
#include <thunder/amr/amr_flags.hh>
#include <thunder/amr/amr_functions.hh>

namespace thunder { namespace amr { 

void regrid() ; 

void refine() ; 

void coarsen() ; 

void partition() ; 

}} /* namespace thunder::amr */ 

#endif /* THUNDER_AMR_REGRID_HH */