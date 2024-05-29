/**
 * @file find_stable_timestep.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-16
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
#include <grace_config.h>
#include <grace/evolution/find_stable_timestep.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/config/config_parser.hh>
#include <grace/physics/grace_physical_systems.hh>
#include <grace/evolution/evolution_kernel_tags.hh> 
#include <grace/utils/reconstruction.hh>
#include <grace/utils/riemann_solvers.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <Kokkos_Core.hpp>

namespace grace {

void find_stable_timestep() {
    Kokkos::Profiling::pushRegion("Timestep update") ; 
    using namespace Kokkos ;
    using namespace grace ;


    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ;    
    int64_t nq = amr::get_local_num_quadrants() ; 

    auto& state = variable_list::get().getstate()   ; 
    auto& aux   = variable_list::get().getaux()     ; 
    auto& cvol  = variable_list::get().getvolumes() ; 

    auto& params = config_parser::get() ; 
    double const CFL = params["evolution"]["cfl_factor"].as<double>() ; 

    #ifdef GRACE_ENABLE_SCALAR_ADV 
    double VEC(ax,ay,az) ; 
    EXPR(
    ax = params["scalar_advection"]["ax"].as<double>() ;,
    ay = params["scalar_advection"]["ay"].as<double>() ;,
    az = params["scalar_advection"]["az"].as<double>() ; )
    scalar_advection_system_t<slope_limited_reconstructor_t<minmod>>  
        scalar_adv_system{ state, aux, VEC(ax,ay,az) } ; 
    #endif 
    #ifdef GRACE_ENABLE_BURGERS 
    burgers_equation_system_t<slope_limited_reconstructor_t<minmod>,hll_riemann_solver_t>
        burgers_eq_system{ state, aux } ;
    #endif 

    double dt_local ; 

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy( {VEC(ngz,ngz,ngz),0}, {VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 

    parallel_reduce( GRACE_EXECUTION_TAG("EVOL","find_timestep")
                   , policy
                   , KOKKOS_LAMBDA(VEC(int const& i, int const& j, int const& k), int const& q, double& dtmax)
    {
        double cmax ; 
        #ifdef GRACE_ENABLE_SCALAR_ADV 
        cmax = scalar_adv_system(eigenspeed_kernel_t{}, VEC(i,j,k),q) ; 
        #endif 
        #ifdef GRACE_ENABLE_BURGERS
        cmax = burgers_eq_system(eigenspeed_kernel_t{}, VEC(i,j,k),q) ; 
        #endif 
        double L    ; 
        #ifdef GRACE_3D 
        L = Kokkos::cbrt(cvol(VEC(i,j,k),q)) ; 
        #else 
        L = Kokkos::sqrt(cvol(VEC(i,j,k),q)) ;
        #endif 
        dtmax = dtmax > CFL/cmax*L ? CFL/cmax*L : dtmax ;  

    }, Kokkos::Min<double>(dt_local)) ; 
    double dt_new ; 
    parallel::mpi_allreduce(&dt_local,&dt_new,1,sc_MPI_MIN) ; 
    grace::set_timestep(dt_new) ; 
    Kokkos::Profiling::popRegion() ; 
}

}