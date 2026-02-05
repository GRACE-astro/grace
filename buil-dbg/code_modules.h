/**
 * @file code_modules.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-06-07
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
#ifndef BAF4B02D_A6A3_4DB8_87E0_FB248A6ECFA4
#define BAF4B02D_A6A3_4DB8_87E0_FB248A6ECFA4

#include <grace_config.h>

#include <string> 
#include <vector> 


namespace grace { namespace detail { 
//*****************************************************************************************************
const std::vector< std::string > code_modules { 
    "amr",
    "system",
    "IO",
    "evolution"
    #ifdef GRACE_ENABLE_SCALAR_ADV
    , "scalar_advection"
    #endif 
    #ifdef GRACE_ENABLE_BURGERS
    , "burgers_equation"
    #endif 
    #ifdef GRACE_ENABLE_GRMHD
    , "grmhd"
    , "admbase"
    #endif 
    #ifdef GRACE_ENABLE_Z4C_METRIC
    , "z4c"
    #endif
    #ifdef GRACE_ENABLE_M1
    , "m1"
    #endif 
    , "eos"
    , "profiling"
    , "checkpoints"
    , "coordinate_system"
    , "spherical_surfaces"
    , "bh_diagnostics"
    #ifdef GRACE_ENABLE_Z4C_METRIC
    , "gw_integrals"
    , "puncture_tracker"
    #endif
} ;
//*****************************************************************************************************
const std::vector< std::string > code_modules_default_configs { 
     "/u/cmusolino/grace-pvt/include/grace/amr/amr-defaults.yaml",
     "/u/cmusolino/grace-pvt/include/grace/system/system-defaults.yaml",
     "/u/cmusolino/grace-pvt/include/grace/IO/IO-defaults.yaml",
     "/u/cmusolino/grace-pvt/include/grace/evolution/evolution-defaults.yaml"
     #ifdef GRACE_ENABLE_SCALAR_ADV
     , "/u/cmusolino/grace-pvt/include/grace/physics/scalar_advection-defaults.yaml"
     #endif 
     #ifdef GRACE_ENABLE_BURGERS
     , "/u/cmusolino/grace-pvt/include/grace/physics/burgers_equation-defaults.yaml"
     #endif 
     #ifdef GRACE_ENABLE_GRMHD
     , "/u/cmusolino/grace-pvt/include/grace/physics/grmhd-defaults.yaml"
     , "/u/cmusolino/grace-pvt/include/grace/physics/admbase-defaults.yaml"
     #endif
     #ifdef GRACE_ENABLE_Z4C_METRIC
     , "/u/cmusolino/grace-pvt/include/grace/physics/z4c-defaults.yaml"
     #endif
     #ifdef GRACE_ENABLE_M1
     , "/u/cmusolino/grace-pvt/include/grace/physics/m1-defaults.yaml"
     #endif
     , "/u/cmusolino/grace-pvt/include/grace/physics/eos/eos-defaults.yaml"
     , "/u/cmusolino/grace-pvt/include/grace/profiling/profiling-defaults.yaml"
     , "/u/cmusolino/grace-pvt/include/grace/system/checkpointing-defaults.yaml"
     , "/u/cmusolino/grace-pvt/include/grace/coordinates/coordinate_system-defaults.yaml"
     , "/u/cmusolino/grace-pvt/include/grace/IO/spherical_surface-defaults.yaml"
     , "/u/cmusolino/grace-pvt/include/grace/IO/bh_diagnostics-defaults.yaml"
     #ifdef GRACE_ENABLE_Z4C_METRIC
     , "/u/cmusolino/grace-pvt/include/grace/IO/gw_integrals-defaults.yaml"
     , "/u/cmusolino/grace-pvt/include/grace/IO/puncture_tracker-defaults.yaml"
     #endif
} ;
//*****************************************************************************************************
}} /* namespace grace::detail */
#endif /* BAF4B02D_A6A3_4DB8_87E0_FB248A6ECFA4 */
