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
#ifdef GRACE_ENABLE_BSSN_METRIC
#include <grace/physics/bssn.hh>
#include <grace/physics/bssn_helpers.hh>
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

    DECLARE_GRID_EXTENTS ; 

    auto& parser = grace::config_parser::get() ;
    
    std::string tstepper = 
        parser["evolution"]["time_stepper"].as<std::string>() ; 

    double const t  = get_simulation_time() ; 
    double const dt = get_timestep()        ;
    
    auto& state   = grace::variable_list::get().getstate()   ; 
    auto& state_p = grace::variable_list::get().getscratch() ;

    auto& sstate   = grace::variable_list::get().getstaggeredstate()   ; 
    auto& sstate_p = grace::variable_list::get().getstaggeredscratch() ;

    auto& aux     = grace::variable_list::get().getaux()     ; 
    auto& saux    = grace::variable_list::get().getstaggeredaux() ;

    auto& cvol    = grace::variable_list::get().getvolumes() ; 
    auto& fsurf   = grace::variable_list::get().getstaggeredcoords() ;
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& dx     = grace::variable_list::get().getspacings() ;  

    auto const nvars_corner = sstate.corner_staggered_fields.extent(GRACE_NSPACEDIM) ;
    auto const nvars_cc     = state.extent(GRACE_NSPACEDIM) ;
    
    /* Copy the current state to scratch memory */
    //amr::apply_boundary_conditions(state) ; 
    Kokkos::deep_copy(state_p, state) ; 
    sstate_p.deep_copy(sstate)        ;  
    if ( tstepper == "euler" ) {
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(
            t,dt,1.0,state,state_p,aux,
            sstate,sstate_p,saux,
            idx,dx,cvol,fsurf) ; 
        amr::apply_boundary_conditions(state,sstate) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux, saux) ;
    } else if (tstepper == "rk2" ) {
        /* Compute auxiliaries at current timelevel */
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(
            t,dt,0.5,
            state_p,state,aux,
            sstate_p,sstate,saux,
            idx,dx,cvol,fsurf) ; 
        amr::apply_boundary_conditions(state_p,sstate_p) ; 
        compute_auxiliary_quantities<eos_t>(state_p, sstate_p, aux, saux) ;
        advance_substep<eos_t>(
            t,dt,1.0,
            state,state_p,aux,
            sstate,sstate_p,saux,
            idx,dx,cvol,fsurf) ;
        amr::apply_boundary_conditions(state,sstate) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux, saux) ;
    } else if (tstepper == "rk3" ) {
        auto staggered_update_policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+2>> (
               {VEC(0,0,0), 0, 0}
            , {VEC(nx+1,ny+1,nz+1),nvars_corner, nq}
        ) ;
        auto update_policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+2>> (
               {VEC(0,0,0), 0, 0}
            , {VEC(nx,ny,nz),nvars_cc, nq}
        ) ;
        // step 1: state_p -> u^1 = u^n + dt L( u^n )
        advance_substep<eos_t>(
            t,dt,1.0,
            state_p,state,aux,
            sstate_p,sstate,saux,
            idx,dx,cvol,fsurf) ; 
        amr::apply_boundary_conditions(state_p,sstate_p) ; 
        compute_auxiliary_quantities<eos_t>(state_p, sstate_p, aux, saux) ;
        // Allocate state_pp and sstate_pp 
        auto state_pp  = grace::variable_list::get().allocate_state() ;
        auto sstate_pp = grace::variable_list::get().allocate_staggered_state() ;
        // step 2: state_pp = 3/4 u^n + 1/4 u^1
        Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
            , staggered_update_policy
            , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
            {
                sstate_pp.corner_staggered_fields(VEC(i+ngz,j+ngz,k+ngz), ivar, q)
                    = 0.75 * sstate.corner_staggered_fields(VEC(i+ngz,j+ngz,k+ngz), ivar, q)
                    + 0.25 * sstate_p.corner_staggered_fields(VEC(i+ngz,j+ngz,k+ngz), ivar, q) ; 
            }
        ) ;
        Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
            , update_policy
            , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
            {
                state_pp(VEC(i+ngz,j+ngz,k+ngz), ivar, q)
                    = 0.75 * state(VEC(i+ngz,j+ngz,k+ngz), ivar, q)
                    + 0.25 * state_p(VEC(i+ngz,j+ngz,k+ngz), ivar, q) ; 
            }
        ) ;
        // step 3: state_pp -> u^2 = 3/4 u^n + 1/4 u^1 + 1/4 dt L( u^1 )
        advance_substep<eos_t>(
            t,dt,0.25,
            state_pp,state_p,aux,
            sstate_pp,sstate_p,saux,
            idx,dx,cvol,fsurf) ;
        amr::apply_boundary_conditions(state_pp,sstate_pp) ; 
        compute_auxiliary_quantities<eos_t>(state_pp, sstate_pp, aux, saux) ;
        // step 4: state = 1/3 u^n + 2/3 u^2
        Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
            , staggered_update_policy
            , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
            {
                sstate.corner_staggered_fields(VEC(i+ngz,j+ngz,k+ngz), ivar, q)
                    = 1./3. * sstate.corner_staggered_fields(VEC(i+ngz,j+ngz,k+ngz), ivar, q)
                    + 2./3. * sstate_pp.corner_staggered_fields(VEC(i+ngz,j+ngz,k+ngz), ivar, q) ; 
            }
        ) ;
        Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
            , update_policy
            , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
            {
                state(VEC(i+ngz,j+ngz,k+ngz), ivar, q)
                    = 1./3. * state(VEC(i+ngz,j+ngz,k+ngz), ivar, q)
                    + 2./3. * state_pp(VEC(i+ngz,j+ngz,k+ngz), ivar, q) ; 
            }
        ) ;
        // step 5: state -> u^n+1 = 1/3 u^n + 2/3 u^2 + 2/3 dt L( u^2 )
        advance_substep<eos_t>(
            t,dt,2./3.,
            state,state_pp,aux,
            sstate,sstate_pp,saux,
            idx,dx,cvol,fsurf) ;
        amr::apply_boundary_conditions(state,sstate) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux, saux) ;
    } else {
        ERROR("Unrecognised time-stepper.") ; 
    }
    Kokkos::deep_copy(state_p,state) ; 
    sstate_p.deep_copy(sstate)        ;
}
template< typename eos_t >
void advance_substep( double const t, double const dt, double const dtfact 
                    , var_array_t<GRACE_NSPACEDIM>& new_state 
                    , var_array_t<GRACE_NSPACEDIM>& old_state 
                    , var_array_t<GRACE_NSPACEDIM>& aux 
                    , staggered_variable_arrays_t& staggered_new_state
                    , staggered_variable_arrays_t& staggered_old_state 
                    , staggered_variable_arrays_t& staggered_aux
                    , scalar_array_t<GRACE_NSPACEDIM>& idx
                    , scalar_array_t<GRACE_NSPACEDIM>& dx
                    , cell_vol_array_t<GRACE_NSPACEDIM>& cvol
                    , staggered_coordinate_arrays_t& surfs_and_edges )
{
    GRACE_PROFILING_PUSH_REGION("evol") ;
    using namespace grace ; 
    using namespace Kokkos  ; 

    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    grace::fill_physical_coordinates(pcoords) ; 

    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int nq = amr::get_local_num_quadrants() ;
    
    int nvars_hrsc = variables::get_n_hrsc() ;
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
    burgers_eq_system.template compute_z_flux<hll_riemann_solver_t,weno_reconstructor_t<3>>(team, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact)
    #define GET_SOURCES \
    burgers_eq_system(sources_computation_kernel_t{}, team, VEC(i+ngz,j+ngz,k+ngz), idx, new_state, dt, dtfact )
    #endif 
    #ifdef GRACE_ENABLE_GRMHD
    auto eos = eos::get().get_eos<eos_t>() ;  
    grmhd_equations_system_t<eos_t>
        grmhd_eq_system(eos,old_state,aux,staggered_old_state) ; 
    #define RECON slope_limited_reconstructor_t<minmod>
    #define GET_X_FLUX \
    grmhd_eq_system.template compute_x_flux<hll_riemann_solver_t,RECON>(q, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact) 
    #define GET_Y_FLUX \
    grmhd_eq_system.template compute_y_flux<hll_riemann_solver_t,RECON>(q, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact)
    #define GET_Z_FLUX \
    grmhd_eq_system.template compute_z_flux<hll_riemann_solver_t,RECON>(q, VEC(i,j,k), ngz, fluxes, dx, dt, dtfact)
    #define GET_SOURCES \
    grmhd_eq_system(sources_computation_kernel_t{}, q, VEC(i+ngz,j+ngz,k+ngz), idx, new_state, staggered_new_state, dt, dtfact )
    #endif 
    #ifdef GRACE_ENABLE_BSSN_METRIC
    bssn_system_t
        bssn_eq_system(old_state,aux,staggered_old_state) ; 
    double const epsdiss = grace::get_param<double>("bssn","epsdiss") ;
    #endif 
    //**************************************************************************************************/
    device_event_t x_flux_finished{}, y_flux_finished{}, z_flux_finished{}, sources_finished{} ; 

    int threadsPerBlock = 256; 

    DEVICE_MARK_TRACING_POINT("x_flux") ; // roctx tracing on HIP, nvtx on CUDA

    /* Get stream */
    auto& pool = grace::device_stream_pool::get(); 
    auto& stream = pool.next() ; 

    /* Create loop range and set number of blocks */
    auto flux_x_range  = MDRange<GRACE_NSPACEDIM+1, int> (
          {VEC(0,0,0),0}
        , {VEC(nx+1,ny,nz),nq}
    ) ; 
    int numBlocks      = (flux_x_range.tot_iterations + threadsPerBlock - 1)/threadsPerBlock ;
    launch_grace_kernel(flux_x_range, KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {  GET_X_FLUX ; },
                        (dim3) numBlocks, (dim3) threadsPerBlock, 0, stream ) ; 

    x_flux_finished.record(stream) ; 
    //**************************************************************************************************/
    DEVICE_MARK_TRACING_POINT("y_flux") ; 
    auto flux_y_range  = MDRange<GRACE_NSPACEDIM+1, int> (
          {VEC(0,0,0),0}
        , {VEC(nx,ny+1,nz),nq}
    ) ;
    numBlocks      = (flux_y_range.tot_iterations + threadsPerBlock - 1)/threadsPerBlock ;
    launch_grace_kernel(flux_y_range, KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {  GET_Y_FLUX ; },
                        (dim3) numBlocks, (dim3) threadsPerBlock, 0, stream ) ;  
    y_flux_finished.record(stream) ; 
    //**************************************************************************************************/
    DEVICE_MARK_TRACING_POINT("z_flux") ;
    auto flux_z_range  = MDRange<GRACE_NSPACEDIM+1, int> (
          {VEC(0,0,0),0}
        , {VEC(nx,ny,nz+1),nq}
    ) ;
    numBlocks      = (flux_z_range.tot_iterations + threadsPerBlock - 1)/threadsPerBlock ;
    launch_grace_kernel(flux_z_range, KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {  GET_Z_FLUX ; },
                        (dim3) numBlocks, (dim3) threadsPerBlock, 0, stream ) ; 
    z_flux_finished.record(stream) ;  
    //**************************************************************************************************/
    DEVICE_MARK_TRACING_POINT("sources") ;
    auto sources_range  = MDRange<GRACE_NSPACEDIM+1, int> (
          {VEC(0,0,0),0}
        , {VEC(nx,ny,nz),nq}
    ) ;
    numBlocks      = (sources_range.tot_iterations + threadsPerBlock - 1)/threadsPerBlock ;
    launch_grace_kernel(sources_range, KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {  GET_SOURCES ; },
                        (dim3) numBlocks, (dim3) threadsPerBlock, 0, stream ) ;  
    sources_finished.record(stream) ; 
    //**************************************************************************************************/
    Kokkos::fence() ;
    GRACE_TRACE(
        "Doing BSSN {} {} {} {} {}", 
        staggered_old_state.corner_staggered_fields.extent(0), 
        staggered_old_state.corner_staggered_fields.extent(1), 
        staggered_old_state.corner_staggered_fields.extent(2), 
        staggered_old_state.corner_staggered_fields.extent(3), 
        staggered_old_state.corner_staggered_fields.extent(4)
    ) ; 
    
    auto bssn_rhs_policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
               {VEC(0,0,0),0}
            , {VEC(nx+1,ny+1,nz+1),nq}
        ) ;
    Kokkos::parallel_for(
          GRACE_EXECUTION_TAG("EVOL","BSSN_RHS")
        , bssn_rhs_policy
        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int q)
        {
            /*
            bssn_eq_system.template compute_update_custom<2>(
                q,
                VEC(i+ngz,j+ngz,k+ngz),
                idx,
                new_state,
                staggered_new_state,
                dt,
                dtfact, 
                t,
                pcoords(VEC(i+ngz,j+ngz,k+ngz),0,q),
                epsdiss
            ) ; 
            */
        }
    ) ; 
    
    /* Device sync */
    Kokkos::fence() ;
    GRACE_TRACE("BSSN done.") ; 
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
                         , grace::staggered_variable_arrays_t&        \
                         , grace::staggered_variable_arrays_t&        \
                         , grace::staggered_variable_arrays_t&        \
                         , grace::scalar_array_t<GRACE_NSPACEDIM>&    \
                         , grace::scalar_array_t<GRACE_NSPACEDIM>&    \
                         , grace::cell_vol_array_t<GRACE_NSPACEDIM>&  \
                         , grace::staggered_coordinate_arrays_t&  ) ; \
template                                                              \
void evolve_impl<EOS>()

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}