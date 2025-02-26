/**
 * @file tabulated_reader_compose.cpp
 * @author Khalil Pierre (khalil3.14erre@gmail.com)
 * @brief 
 * @date 2025-01-24
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



#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/utils/format_utils.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/system/grace_system.hh>
#include <grace/config/config_parser.hh>

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/eos/tabulated_eos.hh> //Addition to eos_storage inclusions

#include <grace/physics/eos/eos_storage.hh>

#include <Kokkos_Core.hpp>

namespace grace {

}
