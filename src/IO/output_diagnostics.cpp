/**
 * @file output_diagnostics.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-17
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

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/device_vector.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/IO/spherical_surfaces.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/amr/amr_functions.hh>

#include <grace/config/config_parser.hh>

#include <grace/IO/diagnostics/black_hole_diagnostics.hh>
#ifdef GRACE_ENABLE_Z4C_METRIC
#include <grace/IO/diagnostics/gw_integrals.hh>
#endif
#include <Kokkos_Core.hpp>

#include <array>
#include <memory>


namespace grace { namespace IO { 


void output_diagnostics() {
    bh_diagnostics bh_diag{};
    bh_diag.compute_and_write() ;  
    #ifdef GRACE_ENABLE_Z4C_METRIC
    gw_integrals gw_ints{} ; 
    gw_ints.compute_and_write() ;  
    #endif
}

void initialize_diagnostic_files() {
    bh_diagnostics bh_diag{};
    bh_diag.initialize_files() ; 
    #ifdef GRACE_ENABLE_Z4C_METRIC
    gw_integrals gw_ints{} ; 
    gw_ints.initialize_files() ;  
    #endif
    parallel::mpi_barrier() ;
}

} } 