/**
 * @file boundary_conditions.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-21
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

#include <Kokkos_Core.hpp>

#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/system/grace_system.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/prolongation.hh>
#include <grace/utils/limiters.hh>
#include <grace/data_structures/macros.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/config/config_parser.hh>
#include <grace/amr/amr_ghosts.hh>

#include <spdlog/stopwatch.h>

namespace grace { namespace amr {

void apply_boundary_conditions() {
    auto& vars = variable_list::get().getstate() ;
    auto& stag_vars = variable_list::get().getstaggeredstate() ; 
    apply_boundary_conditions(vars, stag_vars,vars,stag_vars,0,0)              ; 
}

void apply_boundary_conditions(
    grace::var_array_t& vars, grace::staggered_variable_arrays_t& stag_vars,
    grace::var_array_t& vars_p, grace::staggered_variable_arrays_t& stag_vars_p,
    double dt, double dtfact
) {
    Kokkos::Profiling::pushRegion("BC") ; 
    GRACE_VERBOSE("Initiating ghost-zone filling.") ; 
    using namespace grace ;
    /******************************************************/
    /* First step:                                        */
    /* Asynchronous data exchange for quadrants in the    */
    /* halo.                                              */
    /******************************************************/
    spdlog::stopwatch sw ; 
    auto& ghost = grace::amr_ghosts::get() ; 
    auto& halo_executor = ghost.get_task_executor() ; 
    halo_executor.run(view_alias_t{&vars,&vars_p,&stag_vars,&stag_vars_p,dt,dtfact}) ; 
    halo_executor.reset();
    Kokkos::fence() ; 
    parallel::mpi_barrier() ; 

    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = get_quadrant_extents() ;
    int64_t ngz = get_n_ghosts() ;
    int64_t nq  = get_local_num_quadrants()  ;  
    size_t nvars = variables::get_n_evolved() ;
    GRACE_TRACE("All done in BC. Total number of cells processed: {}.\n"
                  "Total time elapsed {} s.\n"\
                , EXPR((nx+2*ngz), *(ny+2*ngz), *(nz+2*ngz)) * nq * nvars 
                , sw ) ; 
    Kokkos::Profiling::popRegion() ; 
}

}} /* namespace grace::amr */