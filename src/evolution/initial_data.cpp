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
#include <grace/config/config_parser.hh>
#include <grace/physics/grace_physical_systems.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#ifdef GRACE_ENABLE_GRMHD
#include <grace/physics/grmhd.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#endif
#include <grace/physics/eos/eos_types.hh>

#include <Kokkos_Core.hpp>

namespace grace {

void set_initial_data() {
    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    if( eos_type == "hybrid" ) {
        auto const cold_eos_type = 
            grace::get_param<std::string>("eos", "cold_eos_type") ;
        if( cold_eos_type == "piecewise_polytrope" ) {
            set_initial_data_impl<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>() ; 
        } else if ( cold_eos_type == "tabulated" ) {
            ERROR("Not implemented yet.") ;
        }
    } else if ( eos_type == "tabulated" ) {
        ERROR("Not implemented yet.") ; 
    }
}

template< typename eos_t >
void set_initial_data_impl() {
    Kokkos::Profiling::pushRegion("ID") ; 
    using namespace grace ;

    #ifdef GRACE_ENABLE_SCALAR_ADV 
    set_scalar_advection_initial_data() ; 
    #endif 
    #ifdef GRACE_ENABLE_BURGERS
    set_burgers_initial_data() ; 
    #endif 
    #ifdef GRACE_ENABLE_GRMHD
    set_grmhd_initial_data<eos_t>();
    #endif 
    Kokkos::Profiling::popRegion() ; 
} 
#define INSTANTIATE_TEMPLATE(EOS)   \
template                            \
void set_initial_data_impl<EOS>()
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}