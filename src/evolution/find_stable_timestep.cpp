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
#ifdef GRACE_ENABLE_GRMHD
#include <grace/physics/grmhd.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#endif
#ifdef GRACE_ENABLE_M1
#include <grace/physics/m1_helpers.hh>
#include <grace/physics/m1.hh>
#endif 
#include <grace/physics/eos/eos_types.hh>

#include <Kokkos_Core.hpp>

namespace grace {

void find_stable_timestep() {
    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    GRACE_VERBOSE("Computing timestep at iteration {}, old timestep {}", grace::get_iteration(), grace::get_timestep()) ; 
    if( eos_type == "hybrid" ) {
        auto const cold_eos_type = 
            get_param<std::string>("eos","hybrid_eos","cold_eos_type") ;  
        if( cold_eos_type == "piecewise_polytrope" ) {
            find_stable_timestep_impl<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>() ; 
        } else if ( cold_eos_type == "tabulated" ) {
            ERROR("Not implemented yet.") ;
        }
    } else if ( eos_type == "tabulated" ) {
        ERROR("Not implemented yet.") ; 
    }
    GRACE_VERBOSE("New timestep {}", grace::get_timestep()) ; 
}

template< typename eos_t >
void find_stable_timestep_impl() {
    Kokkos::Profiling::pushRegion("Timestep update") ; 
    using namespace Kokkos ;
    using namespace grace ;


    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ;    
    int64_t nq = amr::get_local_num_quadrants() ; 

    auto& state = variable_list::get().getstate()   ;
    auto& sstate = variable_list::get().getstaggeredstate()   ; 
    auto& aux   = variable_list::get().getaux()     ; 
    auto& dx  = variable_list::get().getspacings() ; 

    auto& params = config_parser::get() ; 
    double const CFL = params["evolution"]["cfl_factor"].as<double>() ; 

    #ifdef GRACE_ENABLE_SCALAR_ADV 
    double VEC(ax,ay,az) ; 
    EXPR(
    ax = params["scalar_advection"]["ax"].as<double>() ;,
    ay = params["scalar_advection"]["ay"].as<double>() ;,
    az = params["scalar_advection"]["az"].as<double>() ; )
    scalar_advection_system_t  
        scalar_adv_system{ state, aux, VEC(ax,ay,az) } ; 
    #define GET_CMAX \
    scalar_adv_system(eigenspeed_kernel_t{}, VEC(i,j,k),q)
    #endif 
    #ifdef GRACE_ENABLE_BURGERS 
    burgers_equation_system_t
        burgers_eq_system{ state, aux } ;
    #define GET_CMAX \
    burgers_eq_system(eigenspeed_kernel_t{}, VEC(i,j,k),q)
    #endif 
    #ifdef GRACE_ENABLE_GRMHD
    auto eos = eos::get().get_eos<eos_t>() ;  
    grmhd_equations_system_t<eos_t>
        grmhd_eq_system(eos,state,sstate,aux) ; 
    #define GET_CMAX \
    grmhd_eq_system(eigenspeed_kernel_t{}, VEC(i,j,k),q)
    #endif 
    #ifdef GRACE_ENABLE_M1 
    m1_equations_system_t m1_eq_system(state,sstate,aux) ;
    #endif 
    double dt_local ; 

    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy( {VEC(ngz,ngz,ngz),0}, {VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 

    parallel_reduce( GRACE_EXECUTION_TAG("EVOL","find_timestep")
                   , policy
                   , KOKKOS_LAMBDA(VEC(int const& i, int const& j, int const& k), int const& q, double& dtmax)
    {
        #if !defined(GRACE_ENABLE_Z4C_METRIC) && !defined(GRACE_ENABLE_BSSN_METRIC)
        double cmax = GET_CMAX; 
        #ifdef GRACE_ENABLE_M1 
        cmax = fmax(cmax,m1_eq_system(eigenspeed_kernel_t{}, VEC(i,j,k),q)) ; 
        #endif 
        #else 
        double cmax = 1 ; 
        #endif 
        double L = dx(0,q);        
        dtmax = dtmax > CFL/cmax*L ? CFL/cmax*L : dtmax ;  

    }, Kokkos::Min<double>(dt_local)) ; 
    double dt_new ; 
    parallel::mpi_allreduce(&dt_local,&dt_new,1,sc_MPI_MIN) ; 
    grace::set_timestep(dt_new) ; 
    #undef GET_CMAX
    Kokkos::Profiling::popRegion() ; 
}

/* Instantiate templates */
#define INSTANTIATE_TEMPLATE(EOS)     \
template                              \
void find_stable_timestep_impl<EOS>()
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}