/**
 * @file thunder_amr.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Single include for all amr related utilities in thunder.
 * @date 2024-03-14
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
#ifndef THUNDER_AMR_THUNDER_AMR_HH
#define THUNDER_AMR_THUNDER_AMR_HH

#include <thunder_config.h>

#include <thunder/utils/inline.h>
#include <thunder/utils/device.h>

#include <thunder/data_structures/macros.hh>

#include<thunder/amr/p4est_headers.hh>

#include <thunder/amr/quadrant.hh>
#include <thunder/amr/tree.hh>
#include <thunder/amr/amr_flags.hh>
#include <thunder/amr/connectivity.hh>
#include <thunder/amr/forest.hh>
#include <thunder/amr/amr_functions.hh>
#include <thunder/amr/regrid.hh>
#include <thunder/amr/boundary_conditions.hh>

#endif /* THUNDER_AMR_THUNDER_AMR_HH */
