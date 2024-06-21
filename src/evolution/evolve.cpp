/**
 * @file evolve.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-13
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

#include <grace/evolution/evolve.hh>
#include <grace/evolution/auxiliaries.hh>
#include <grace/evolution/evolution_kernel_tags.hh>

#include <grace/system/grace_system.hh>

#include <grace/config/config_parser.hh>

#include <grace/amr/boundary_conditions.hh>

#include <grace/data_structures/grace_data_structures.hh>
#include <grace/profiling/profiling.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/reconstruction.hh>
#include <grace/utils/weno_reconstruction.hh>
#include <grace/utils/riemann_solvers.hh>
#ifdef GRACE_ENABLE_BURGERS 
#include <grace/physics/burgers.hh>
#endif 
#ifdef GRACE_ENABLE_SCALAR_ADV
#include <grace/physics/scalar_advection.hh>
#endif  
#ifdef GRACE_ENABLE_GRMHD
#include <grace/physics/grmhd.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#endif
#include <grace/physics/eos/eos_types.hh>

#include <grace/amr/grace_amr.hh>

#include <string> 

namespace grace {

void evolve() {
    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    if( eos_type == "hybrid" ) {
        auto const cold_eos_type = 
            grace::get_param<std::string>("eos", "cold_eos_type") ;
        if( cold_eos_type == "piecewise_polytrope" ) {
            evolve_impl<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>() ;
        } else if ( cold_eos_type == "tabulated" ) {
            ERROR("Not implemented yet.") ;
        }
    } else if ( eos_type == "tabulated" ) {
        ERROR("Not implemented yet.") ; 
    }
}

template< typename eos_t >
void evolve_impl() {
    using namespace grace ; 

    auto& parser = grace::config_parser::get() ;

    std::string tstepper = 
        parser["evolution"]["time_stepper"].as<std::string>() ; 

    double const t  = get_simulation_time() ; 
    double const dt = get_timestep()        ;
    
    auto& state   = grace::variable_list::get().getstate()   ; 
    auto& state_p = grace::variable_list::get().getscratch() ;

    auto& aux     = grace::variable_list::get().getaux()     ; 

    auto& cvol    = grace::variable_list::get().getvolumes() ; 
    auto& fsurf   = grace::variable_list::get().getstaggeredcoords() ; 
    /* Copy the current state to scratch memory */
    //amr::apply_boundary_conditions(state) ; 
    Kokkos::deep_copy(state_p, state) ; 

    if ( tstepper == "euler" ) {
        compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,aux,cvol,fsurf) ; 
        amr::apply_boundary_conditions() ; 
    } else if (tstepper == "rk2" ) {
        /* Compute auxiliaries at current timelevel */
        compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,0.5,state_p,state,aux,cvol,fsurf) ; 
        amr::apply_boundary_conditions(state_p) ; 
        compute_auxiliary_quantities<eos_t>(state_p, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,aux,cvol,fsurf) ;
        amr::apply_boundary_conditions(state) ; 
    } else if (tstepper == "rk3" ) {
        ERROR("Not implemented yet.") ; 
    } else {
        ERROR("Unrecognised time-stepper.") ; 
    }
    Kokkos::deep_copy(state_p,state) ; 
}
template< typename eos_t >
void advance_substep( double const t, double const dt, double const dtfact 
                    , var_array_t<GRACE_NSPACEDIM>& new_state 
                    , var_array_t<GRACE_NSPACEDIM>& old_state 
                    , var_array_t<GRACE_NSPACEDIM>& aux 
                    , cell_vol_array_t<GRACE_NSPACEDIM>& cvol
                    , staggered_coordinate_arrays_t& surfs_and_edges )
{
    GRACE_PROFILING_PUSH_REGION("evol") ;
    using namespace grace ; 
    using namespace Kokkos  ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;
    
    int nvars_hrsc = variables::get_n_evolved() ;
    /*********************************************/ 
    /* Define the flux array and allocate memory */
    /* NB this array has NO ghostzones!          */
    /*********************************************/ 
    flux_array_t fluxes(
          "Fluxes"
        , VEC( nx + 1 
             , ny + 1 
             , nz + 1)
        , nvars_hrsc 
        , GRACE_NSPACEDIM
        , nq 
    ) ; 
    /* Define the equation system (a couple ugly ifdef's!)*/
    #ifdef GRACE_ENABLE_SCALAR_ADV
    double VEC(ax,ay,az) ; 
    EXPR(
    ax = grace::get_param<double>("scalar_advection","ax");,
    ay = grace::get_param<double>("scalar_advection","ay");,
    az = grace::get_param<double>("scalar_advection","az"); )
    scalar_advection_system_t<slope_limited_reconstructor_t<minmod>>  
        scalar_adv_system{ old_state, aux, VEC(ax,ay,az) } ; 
    #define GET_X_FLUX \
    scalar_adv_system(x_flux_computation_kernel_t{}, team, VEC(i,j,k), ngz, fluxes) 
    #define GET_Y_FLUX \
    scalar_adv_system(y_flux_computation_kernel_t{}, team, VEC(i,j,k), ngz, fluxes)
    #define GET_Z_FLUX \
    scalar_adv_system(z_flux_computation_kernel_t{}, team, VEC(i,j,k), ngz, fluxes)
    #define GET_SOURCES \
    scalar_adv_system(sources_computation_kernel_t{}, team, VEC(i,j,k), new_state, dt, dtfact )
    #endif 
    #ifdef GRACE_ENABLE_BURGERS 
    burgers_equation_system_t<weno_reconstructor_t<3>,hll_riemann_solver_t>
        burgers_eq_system{ old_state, aux } ; 
    #define GET_X_FLUX \
    burgers_eq_system(x_flux_computation_kernel_t{}, team, VEC(i,j,k), ngz, fluxes) 
    #define GET_Y_FLUX \
    burgers_eq_system(y_flux_computation_kernel_t{}, team, VEC(i,j,k), ngz, fluxes)
    #define GET_Z_FLUX \
    burgers_eq_system(z_flux_computation_kernel_t{}, team, VEC(i,j,k), ngz, fluxes)
    #define GET_SOURCES \
    burgers_eq_system(sources_computation_kernel_t{}, team, VEC(i,j,k), new_state, dt, dtfact )
    #endif 
    #ifdef GRACE_ENABLE_GRMHD
    auto eos = eos::get().get_eos<eos_t>() ;  
    grmhd_equations_system_t<eos_t,weno_reconstructor_t<3>,hll_riemann_solver_t>
        grmhd_eq_system(eos,old_state,aux) ; 
    #define GET_X_FLUX \
    grmhd_eq_system(x_flux_computation_kernel_t{}, team, VEC(i,j,k), ngz, fluxes) 
    #define GET_Y_FLUX \
    grmhd_eq_system(y_flux_computation_kernel_t{}, team, VEC(i,j,k), ngz, fluxes)
    #define GET_Z_FLUX \
    grmhd_eq_system(z_flux_computation_kernel_t{}, team, VEC(i,j,k), ngz, fluxes)
    #define GET_SOURCES \
    grmhd_eq_system(sources_computation_kernel_t{}, team, VEC(i,j,k), new_state, dt, dtfact )
    #endif 
    
    TeamPolicy<default_execution_space> policy( nq, AUTO() ) ;
    using member_type = TeamPolicy<default_execution_space>::member_type ; 

    parallel_for( GRACE_EXECUTION_TAG("EVOL", "advance_substep")
                , policy 
                , KOKKOS_LAMBDA (member_type team)
    {
        auto team_range_x = 
        Kokkos::TeamThreadMDRange<Kokkos::Rank<GRACE_NSPACEDIM>, member_type>( 
              team 
            , VEC(nx+1,ny,nz) ) ;
        auto team_range_y = 
        Kokkos::TeamThreadMDRange<Kokkos::Rank<GRACE_NSPACEDIM>, member_type>( 
              team 
            , VEC(nx,ny+1,nz) ) ;
        #ifdef GRACE_3D 
        auto team_range_z = 
        Kokkos::TeamThreadMDRange<Kokkos::Rank<GRACE_NSPACEDIM>, member_type>( 
              team 
            , VEC(nx,ny,nz+1) ) ;
        #endif 
        auto team_range = 
        Kokkos::TeamThreadMDRange<Kokkos::Rank<GRACE_NSPACEDIM>, member_type>( 
              team 
            , VEC(nx,ny,nz) ) ;

        int64_t q = team.league_rank() ; 

        auto surfx = subview( surfs_and_edges.cell_face_surfaces_x 
                             , VEC(ALL(),ALL(),ALL()), q ) ; 
        auto surfy = subview( surfs_and_edges.cell_face_surfaces_y 
                             , VEC(ALL(),ALL(),ALL()), q ) ; 
        auto surfz = subview( surfs_and_edges.cell_face_surfaces_z 
                             , VEC(ALL(),ALL(),ALL()), q ) ; 
        
        parallel_for( team_range_x 
                    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k))
        {
            GET_X_FLUX ;
        }) ; 
        parallel_for( team_range_y 
                    , KOKKOS_LAMBDA ( VEC(int const& i, int const& j, int const& k))
        {
            GET_Y_FLUX ; 
        }) ; 
        #ifdef GRACE_3D 
        parallel_for( team_range_z 
                    , KOKKOS_LAMBDA ( VEC(int const& i, int const& j, int const& k))
        {
            GET_Z_FLUX ;
        }) ; 
        #endif 
        parallel_for( team_range 
                    , KOKKOS_LAMBDA ( VEC(int const& i, int const& j, int const& k))
        {
            GET_SOURCES ; 
        }) ;
        team.team_barrier() ; 

        auto team_range_vars = 
        Kokkos::TeamThreadMDRange<Kokkos::Rank<GRACE_NSPACEDIM+1>, member_type>( 
              team 
            , VEC(nx,ny,nz), nvars_hrsc ) ;
        parallel_for( team_range_vars 
                    , KOKKOS_LAMBDA ( VEC(int const& i, int const& j, int const& k), int const& ivar)
        {
            int const VEC(I{i+ngz},J{j+ngz},K{k+ngz}) ; 
            new_state(VEC(I,J,K),ivar,team.league_rank()) += 
                dt * dtfact * (
                EXPR(   ( surfx(VEC(I,J,K))   * fluxes(VEC(i,j,k)  ,ivar,0,q) 
                        - surfx(VEC(I+1,J,K)) * fluxes(VEC(i+1,j,k),ivar,0,q) )
                    , + ( surfy(VEC(I,J,K))   * fluxes(VEC(i,j,k)  ,ivar,1,q) 
                        - surfy(VEC(I,J+1,K)) * fluxes(VEC(i,j+1,k),ivar,1,q) )
                    , + ( surfz(VEC(I,J,K))   * fluxes(VEC(i,j,k)  ,ivar,2,q) 
                        - surfz(VEC(I,J,K+1)) * fluxes(VEC(i,j,k+1),ivar,2,q) ) )
                ) / cvol(VEC(I,J,K),q) ; 
        }) ;
    }) ; 
    #undef GET_X_FLUX
    #undef GET_Y_FLUX
    #undef GET_Z_FLUX
    #undef GET_SOURCES
    GRACE_PROFILING_POP_REGION ; 
}
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                     \
template                                                              \
void advance_substep<EOS>( double const , double const , double const \
                         , grace::var_array_t<GRACE_NSPACEDIM>&       \
                         , grace::var_array_t<GRACE_NSPACEDIM>&       \
                         , grace::var_array_t<GRACE_NSPACEDIM>&       \
                         , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  \
                         , grace::staggered_coordinate_arrays_t&  ) ; \
template                                                              \
void evolve_impl<EOS>()

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}