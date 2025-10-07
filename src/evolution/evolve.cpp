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
#include <grace/utils/advanced_riemann_solvers.hh>
#endif
#include <grace/physics/eos/eos_types.hh>

#include <grace/amr/grace_amr.hh>

#include <string> 

namespace grace {

void evolve() {
    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    GRACE_VERBOSE("Performing timestep integration at iteration {}", grace::get_iteration()) ; 
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
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& dx     = grace::variable_list::get().getspacings() ;  
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ;  

    /* Copy the current state to scratch memory */
    //amr::apply_boundary_conditions(state) ; 
    Kokkos::deep_copy(state_p, state) ; 

    if ( tstepper == "euler" ) {
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,aux,idx,dx,cvol,fsurf,fluxes) ; 
        amr::apply_boundary_conditions(state) ; 
        compute_auxiliary_quantities<eos_t>(state, aux) ;
    } else if (tstepper == "rk2" ) {
        /* Compute auxiliaries at current timelevel */
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,0.5,state_p,state,aux,idx,dx,cvol,fsurf,fluxes) ; 
        amr::apply_boundary_conditions(state_p) ; 
        compute_auxiliary_quantities<eos_t>(state_p, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,aux,idx,dx,cvol,fsurf,fluxes) ;
        amr::apply_boundary_conditions(state) ; 
        compute_auxiliary_quantities<eos_t>(state, aux) ;
    } else if (tstepper == "rk3" ) {
        ERROR("Not implemented yet.") ; 
    } else {
        ERROR("Unrecognised time-stepper.") ; 
    }
    Kokkos::deep_copy(state_p,state) ; 
}
template< typename eos_t >
void advance_substep( double const t, double const dt, double const dtfact 
                    , var_array_t& new_state 
                    , var_array_t& old_state 
                    , var_array_t& aux 
                    , scalar_array_t<GRACE_NSPACEDIM>& idx
                    , scalar_array_t<GRACE_NSPACEDIM>& dx
                    , cell_vol_array_t<GRACE_NSPACEDIM>& cvol
                    , staggered_coordinate_arrays_t& surfs_and_edges
                    , flux_array_t& fluxes  )
{
    GRACE_PROFILING_PUSH_REGION("evol") ;
    using namespace grace ; 
    using namespace Kokkos  ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;
    
    int nvars_hrsc = variables::get_n_hrsc() ;
    /*********************************************/ 
    /* Define the flux array and allocate memory */
    /* NB this array has NO ghostzones!          */
    /*********************************************/ 
    /* 
    flux_array_t fluxes(
          "Fluxes"
        , VEC( nx + 1 
             , ny + 1 
             , nz + 1)
        , nvars_hrsc 
        , GRACE_NSPACEDIM
        , nq 
    ) ; 
    */
    /* Define the equation system (a couple ugly ifdef's!)*/
    #ifdef GRACE_ENABLE_SCALAR_ADV
    double VEC(ax,ay,az) ; 
    EXPR(
    ax = grace::get_param<double>("scalar_advection","ax");,
    ay = grace::get_param<double>("scalar_advection","ay");,
    az = grace::get_param<double>("scalar_advection","az"); )
    scalar_advection_system_t 
        scalar_adv_system{ old_state, aux, VEC(ax,ay,az) } ; 
    #define GET_X_FLUX \
    scalar_adv_system.template compute_x_flux<slope_limited_reconstructor_t<minmod>>(team, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact) 
    #define GET_Y_FLUX \
    scalar_adv_system.template compute_y_flux<slope_limited_reconstructor_t<minmod>>(team, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact)
    #define GET_Z_FLUX \
    scalar_adv_system.template compute_z_flux<slope_limited_reconstructor_t<minmod>>(team, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact)
    #define GET_SOURCES \
    scalar_adv_system(sources_computation_kernel_t{}, team, VEC(i+ngz,j+ngz,k+ngz), idx, new_state, dt, dtfact )
    #endif 
    #ifdef GRACE_ENABLE_BURGERS 
    burgers_equation_system_t
        burgers_eq_system{ old_state, aux } ; 
    #define GET_X_FLUX \
    burgers_eq_system.template compute_x_flux<hll_riemann_solver_t,weno_reconstructor_t<3>>(team, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact) 
    #define GET_Y_FLUX \
    burgers_eq_system.template compute_y_flux<hll_riemann_solver_t,weno_reconstructor_t<3>>(team, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact)
    #define GET_Z_FLUX \
    burgers_eq_system.template compute_z_flux<hll_riemann_solver_t,weno_reconstructor_t<3>>(eam, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact)
    #define GET_SOURCES \
    burgers_eq_system(sources_computation_kernel_t{}, team, VEC(i+ngz,j+ngz,k+ngz), idx, new_state, dt, dtfact )
    #endif 
    #ifdef GRACE_ENABLE_GRMHD
    auto eos = eos::get().get_eos<eos_t>() ;  
    grmhd_equations_system_t<eos_t>
        grmhd_eq_system(eos,old_state,aux) ; 
    #define RECON slope_limited_reconstructor_t<MCbeta>
    #define GET_X_FLUX \
    grmhd_eq_system.template compute_x_flux<hll_riemann_solver_t,RECON>(q, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact) 
    #define GET_Y_FLUX \
    grmhd_eq_system.template compute_y_flux<hll_riemann_solver_t,RECON>(q, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact)
    #define GET_Z_FLUX \
    grmhd_eq_system.template compute_z_flux<hll_riemann_solver_t,RECON>(q, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact)
    #define GET_SOURCES \
    grmhd_eq_system(sources_computation_kernel_t{}, q, VEC(i+ngz,j+ngz,k+ngz), idx, new_state, dt, dtfact )
    #endif 
    //**************************************************************************************************/
    auto flux_x_policy = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,0),0}
            , {VEC(nx+1,ny+1,nz+1),nq}
        ) ; 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_x_flux")
                , flux_x_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        GET_X_FLUX ;
    }) ; 
    //**************************************************************************************************/
    auto flux_y_policy = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,0),0}
            , {VEC(nx+1,ny+1,nz+1),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_y_flux")
                , flux_y_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        GET_Y_FLUX ;
    }) ; 
    //**************************************************************************************************/
    auto flux_z_policy = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,0),0}
            , {VEC(nx+1,ny+1,nz+1),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_z_flux")
                , flux_z_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        GET_Z_FLUX ;
    }) ; 
    //**************************************************************************************************/
    auto geom_sources_policy = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,0),0}
            , {VEC(nx,ny,nz),nq}
        ) ; 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_sources")
                , geom_sources_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        GET_SOURCES ;
    }) ;  
    //**************************************************************************************************/
    Kokkos::fence() ; 
    //**************************************************************************************************/
    auto advance_policy = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+2>> (
              {VEC(0,0,0),0,0}
            , {VEC(nx,ny,nz),nvars_hrsc,nq}
        ) ; 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
                , advance_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& ivar, int const& q)
    {

        int const VEC(I{i+ngz},J{j+ngz},K{k+ngz}) ; 
        #ifndef GRACE_CARTESIAN_COORDINATES
        auto surfx = subview( surfs_and_edges.cell_face_surfaces_x 
                             , VEC(ALL(),ALL(),ALL()), q ) ; 
        auto surfy = subview( surfs_and_edges.cell_face_surfaces_y 
                             , VEC(ALL(),ALL(),ALL()), q ) ; 
        auto surfz = subview( surfs_and_edges.cell_face_surfaces_z 
                             , VEC(ALL(),ALL(),ALL()), q ) ; 
        new_state(VEC(I,J,K),ivar,q) += 
            dt * dtfact * (
            EXPR(   ( surfx(VEC(I,J,K))   * fluxes(VEC(i,j,k)  ,ivar,0,q) 
                    - surfx(VEC(I+1,J,K)) * fluxes(VEC(i+1,j,k),ivar,0,q) )
                , + ( surfy(VEC(I,J,K))   * fluxes(VEC(i,j,k)  ,ivar,1,q) 
                    - surfy(VEC(I,J+1,K)) * fluxes(VEC(i,j+1,k),ivar,1,q) )
                , + ( surfz(VEC(I,J,K))   * fluxes(VEC(i,j,k)  ,ivar,2,q) 
                    - surfz(VEC(I,J,K+1)) * fluxes(VEC(i,j,k+1),ivar,2,q) ) )
            ) / cvol(VEC(I,J,K),q) ; 
        #else 
        new_state(VEC(I,J,K),ivar,q) += 
            dt * dtfact * (
            EXPR(   ( fluxes(VEC(i,j,k)  ,ivar,0,q) - fluxes(VEC(i+1,j,k),ivar,0,q) ) * idx(0,q)
                , + ( fluxes(VEC(i,j,k)  ,ivar,1,q) - fluxes(VEC(i,j+1,k),ivar,1,q) ) * idx(1,q)
                , + ( fluxes(VEC(i,j,k)  ,ivar,2,q) - fluxes(VEC(i,j,k+1),ivar,2,q) ) * idx(2,q))
            ) ; 
        #endif 
    }) ; 
    auto advance_stag_policy_x = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz,nz+ngz),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
                , advance_stag_policy_x 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& ivar, int const& q)
    {
        // E^y_{i+1/2,j,k+1/2} = 1/4 ( F^x_{i+1/2,j,k} + F^x_{i+1/2,j,k+1} - F^z_{i+1,j,k+1/2} - F^z_{i,j,k+1/2} )
        auto const Eyp = 0.25 * ( fluxes(VEC(i,j,k), BSZ_, 0, q) + fluxes(VEC(i,j,k+1), BSZ_, 0, q)
                                - fluxes(VEC(i+1,j,k), BSX_, 2, q) - fluxes(VEC(i,j,k), BSX_, 2, q) ) ;
        // E^y_{i+1/2,j,k-1/2} = 1/4 ( F^x_{i+1/2,j,k} + F^x_{i+1/2,j,k-1} - F^z_{i+1,j,k-1/2} - F^z_{i,j,k-1/2} )
        auto const Ey  = 0.25 * ( fluxes(VEC(i,j,k), BSZ_, 0, q) + fluxes(VEC(i,j,k-1), BSZ_, 0, q)
                                - fluxes(VEC(i+1,j,k-1), BSX_, 2, q) - fluxes(VEC(i,j,k-1), BSX_, 2, q) ) ;
        // E^z_{i+1/2,j+1/2,k} = 1/4 ( F^y_{i+1,j+1/2,k} + F^y_{i,j+1/2,k} - F^x_{i+1/2,j,k} - F^x{i+1/2,j+1,k} )
        auto const Ezp = 0.25 * ( fluxes(VEC(i,j,k),BSX_,1,q) + fluxes(VEC(i+1,j,k),BSX_,1,q) 
                                - fluxes(VEC(i,j,k),BSY_,0,q) - fluxes(VEC(i,j+1,k),BSY_,0,q) ) ;
        // E^z_{i+1/2,j-1/2,k} = 1/4 ( F^y_{i+1,j-1/2,k} + F^y_{i,j-1/2,k} - F^x_{i+1/2,j,k} - F^x{i+1/2,j-1,k} )
        auto const Ez =0.25 * ( fluxes(VEC(i,j-1,k),BSX_,1,q) + fluxes(VEC(i+1,j-1,k),BSX_,1,q) 
                                - fluxes(VEC(i,j,k),BSY_,0,q) - fluxes(VEC(i,j-1,k),BSY_,0,q) ) ;

        new_state(VEC(i,j,k),BSX_,q) += dt * dtfact * (
            (Eyp-Ey) * idx(1,q)
          + (Ez-Ezp) * idx(2,q)
        )  ; /* yuck! */
    } ) ; 

    auto advance_stag_policy_y = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz+1,nz+ngz),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
                , advance_stag_policy_y 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& ivar, int const& q)
    {
        // Corner E-fields needed for B_y update
        // E^z_{i+1/2,j+1/2,k}
        auto const Ezp_y = 0.25 * (
              fluxes(VEC(i,j  ,k), BSY_, 0, q)    // F_x(i+1/2,j,k)
            + fluxes(VEC(i,j+1,k), BSY_, 0, q)    // F_x(i+1/2,j+1,k)
            - fluxes(VEC(i  ,j,k), BSX_, 1, q)    // F_y(i,j+1/2,k)
            - fluxes(VEC(i+1,j,k), BSX_, 1, q)    // F_y(i+1,j+1/2,k)
        );
        // E^z (i-1/2,j+1/2,k)
        auto const Ez_y  = 0.25 * (
              fluxes(VEC(i-1,j  ,k), BSY_, 0, q)    // F_x(i-1/2,j,k)
            + fluxes(VEC(i-1,j+1,k), BSY_, 0, q)    // F_x(i-1/2,j+1,k)
            - fluxes(VEC(i  ,j,k), BSX_, 1, q)    // F_y(i,j+1/2,k)
            - fluxes(VEC(i-1,j,k), BSX_, 1, q)    // F_y(i-1,j+1/2,k)
        );

        // E^x (i,j+1/2,k+1/2)
        auto const Exp_y = 0.25 * (
                fluxes(VEC(i,j,k  ), BSZ_, 1, q) // F_y(i,j+1/2,k)
              + fluxes(VEC(i,j,k+1), BSZ_, 1, q) // F_y(i,j+1/2,k+1)
              - fluxes(VEC(i,j  ,k), BSY_, 2, q) // F_z(i,j,k+1/2)
              - fluxes(VEC(i,j+1,k), BSY_, 2, q) // F_z(i,j,k+1/2)

        );
        // E^x (i,j+1/2,k-1/2)
        auto const Ex_y = 0.25 * (
                fluxes(VEC(i,j,k  ), BSZ_, 1, q) // F_y(i,j+1/2,k)
              + fluxes(VEC(i,j,k-1), BSZ_, 1, q) // F_y(i,j+1/2,k-1)
              - fluxes(VEC(i,j  ,k-1), BSY_, 2, q) // F_z(i,j,k-1/2)
              - fluxes(VEC(i,j+1,k-1), BSY_, 2, q) // F_z(i,j,k-1/2)
        );

        // Update B_y
        new_state(VEC(i,j,k), BSY_, q) += dt * dtfact * (
            (Ezp_y - Ez_y) * idx(0,q)  // z derivative
            + (Ex_y - Exp_y) * idx(2,q)  // x derivative
        );

    } ) ; 

    auto advance_stag_policy_z = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz+1),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
                , advance_stag_policy_z 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& ivar, int const& q)
    {
        // Corner E-fields needed for B_z update
        // E^x (i,j+1/2,k+1/2)
        auto const Exp_z = 0.25 * (
               fluxes(VEC(i,j,k  ),BSZ_,1)
             + fluxes(VEC(i,j,k+1),BSZ_,1)
             - fluxes(VEC(i,j  ,k),BSY_,2)
             - fluxes(VEC(i,j+1,k),BSY_,2)
        );
        // E^x (i,j-1/2,k+1/2)
        auto const Ex_z = 0.25 * (
               fluxes(VEC(i,j-1,k  ),BSZ_,1)
             + fluxes(VEC(i,j-1,k+1),BSZ_,1)
             - fluxes(VEC(i,j  ,k),BSY_,2)
             - fluxes(VEC(i,j-1,k),BSY_,2)
        );

        // E^y at front and back corners in x
        auto const Eyp_z = 0.25 * (
               fluxes(VEC(i,j,k),BSX_,2)
             + fluxes(VEC(i+1,j,k),BSX_,2)
             - fluxes(VEC(i,j,k),BSZ_,0)
             - fluxes(VEC(i,j,k+1),BSZ_,0)
        );
        // E^y (i-1/2,j,k+1/2)
        auto const Ey_z = 0.25 * (
               fluxes(VEC(i,j,k),BSX_,2)
             + fluxes(VEC(i-1,j,k),BSX_,2)
             - fluxes(VEC(i-1,j,k),BSZ_,0)
             - fluxes(VEC(i-1,j,k+1),BSZ_,0)
        );

        // Update B_z
        new_state(VEC(I,J,K), BSZ_, q) += dt * dtfact * (
            (Ey_z - Eyp_z) * idx(1,q)  // y derivative
            + (Exp_z - Ex_z) * idx(0,q)  // x derivative
        );


    } ) ; 
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
                         , grace::var_array_t&       \
                         , grace::var_array_t&       \
                         , grace::var_array_t&       \
                         , grace::scalar_array_t<GRACE_NSPACEDIM>&    \
                         , grace::scalar_array_t<GRACE_NSPACEDIM>&    \
                         , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  \
                         , grace::staggered_coordinate_arrays_t&      \
                         , grace::flux_array_t&                   ) ; \
template                                                              \
void evolve_impl<EOS>()

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}