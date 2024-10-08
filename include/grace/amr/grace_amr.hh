/**
 * @file grace_amr.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Single include for all amr related utilities in grace.
 * @date 2024-03-14
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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
#ifndef GRACE_AMR_GRACE_AMR_HH
#define GRACE_AMR_GRACE_AMR_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>

#include<grace/amr/p4est_headers.hh>

#include <grace/amr/quadrant.hh>
#include <grace/amr/tree.hh>
#include <grace/amr/amr_flags.hh>
#include <grace/amr/connectivity.hh>
#include <grace/amr/forest.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/regrid.hh>
#include <grace/amr/boundary_conditions.hh>

#endif /* GRACE_AMR_GRACE_AMR_HH */
