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
#include <grace/evolution/refluxing.hh>
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
#include <grace/utils/tstep_utils.hh>
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
//#include <grace/utils/advanced_riemann_solvers.hh>
#endif
#ifdef GRACE_ENABLE_Z4C_METRIC
#include <grace/physics/z4c.hh>
#include <grace/physics/z4c_helpers.hh>
#endif 
#ifdef GRACE_ENABLE_BSSN_METRIC
#include <grace/physics/bssn.hh>
#include <grace/physics/bssn_helpers.hh>
#endif 
#ifdef GRACE_ENABLE_M1 
#include <grace/physics/m1_helpers.hh>
#include <grace/physics/m1.hh>
#endif 
#include <grace/physics/eos/eos_types.hh>

#include <grace/amr/grace_amr.hh>

#include <string> 


#include <fstream> 
#include <iomanip> 

namespace grace {

void evolve() {
    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    GRACE_VERBOSE("Performing timestep integration at iteration {}", grace::get_iteration()) ; 
    if( eos_type == "hybrid" ) {
        auto const cold_eos_type = 
            get_param<std::string>("eos","hybrid_eos","cold_eos_type") ;  
        if( cold_eos_type == "piecewise_polytrope" ) {
            evolve_impl<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>() ;
        } else if ( cold_eos_type == "tabulated" ) {
            ERROR("Not implemented yet.") ;
        } else {
            ERROR("Unknown cold eos type " << cold_eos_type) ;
        }
    } else if ( eos_type == "tabulated" ) {
        evolve_impl<grace::tabulated_eos_t>() ;
    } else if  ( eos_type == "ideal_gas") {
        evolve_impl<grace::ideal_gas_eos_t>() ;
    } else {
        ERROR("Unknown EOS " << eos_type) ; 
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

    auto& sstate = grace::variable_list::get().getstaggeredstate() ; 
    auto& sstate_p = grace::variable_list::get().getstaggeredscratch() ; 

    auto& aux     = grace::variable_list::get().getaux()     ; 
    
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& dx     = grace::variable_list::get().getspacings() ;  
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ; 
    auto& emf  = grace::variable_list::get().getemfarray() ; 

    auto nvars_face  = sstate.face_staggered_fields_x.extent(GRACE_NSPACEDIM) ;
    auto nvars_cc  = state.extent(GRACE_NSPACEDIM) ;

    /* Copy the current state to scratch memory */
    //amr::apply_boundary_conditions(state) ; 
    Kokkos::deep_copy(state_p, state) ; 
    grace::deep_copy(sstate_p, sstate) ; 
    // reset mass error once per timestep
    #ifdef GRACE_ENABLE_GRMHD 
    #ifndef GRACE_FREEZE_HYDRO
    Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ;
    parallel_for(GRACE_EXECUTION_TAG("EVOL","reset_d_err"), policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        aux(i,j,k,C2P_DENS_ERR_,q) = 0.0 ; 
    });
    #endif
    #endif 
    if ( tstepper == "euler" ) {
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ; 
        amr::apply_boundary_conditions(state, sstate,state_p,sstate_p,dt,1.0) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
    } else if (tstepper == "rk2" ) {
        /* Compute auxiliaries at current timelevel */
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,0.5,state_p,state,sstate_p,sstate) ; 
        amr::apply_boundary_conditions(state_p,sstate_p,state,sstate,dt,0.5) ; 
        compute_auxiliary_quantities<eos_t>(state_p, sstate_p, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ;
        amr::apply_boundary_conditions(state,sstate,state_p,sstate_p,dt,1.0) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
    } else if (tstepper == "rk3" ) {
        // step 1: state_p -> u^1 = u^n + dt L( u^n )
        advance_substep<eos_t>(
            t,dt,1.0,
            state_p,state,
            sstate_p,sstate) ; 
        amr::apply_boundary_conditions(state_p,sstate_p,state,sstate,dt,1.0) ; 
        compute_auxiliary_quantities<eos_t>(state_p, sstate_p, aux) ;
        // Allocate state_pp and sstate_pp 
        auto state_pp  = grace::variable_list::get().getstagingbuffer()[0] ;
        auto sstate_pp = grace::variable_list::get().getstagstagingbuffer()[0];
        // step 2: state_pp = 3/4 u^n + 1/4 u^1
        linop_apply(state_pp,state,state_p,
                    sstate_pp,sstate,sstate_p, 0.75, 0.25) ;
        
        // step 3: state_pp -> u^2 = 3/4 u^n + 1/4 u^1 + 1/4 dt L( u^1 )
        advance_substep<eos_t>(
            t,dt,0.25,
            state_pp/*new*/,state_p/*old*/,
            sstate_pp,sstate_p) ;
        amr::apply_boundary_conditions(state_pp,sstate_pp,state_p,sstate_p,dt,0.25) ; 
        compute_auxiliary_quantities<eos_t>(state_pp, sstate_pp, aux) ;
        // step 4: state = 1/3 u^n + 2/3 u^2
        linop_apply(state,state,state_pp,
                    sstate,sstate,sstate_pp, 1./3, 2./3.) ; 
        
        // step 5: state -> u^n+1 = 1/3 u^n + 2/3 u^2 + 2/3 dt L( u^2 )
        advance_substep<eos_t>(
            t,dt,2./3.,
            state,state_pp,
            sstate,sstate_pp) ;
        amr::apply_boundary_conditions(state,sstate,state_pp,sstate_pp,dt,2./3.) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
    } else if ( tstepper == "rk4" ) { 
        // get storage 
        auto& s1  = state    ; 
        auto& ss1 = sstate   ;
        auto& s2  = state_p  ;
        auto& ss2 = sstate_p ;
        auto s3   = grace::variable_list::get().getstagingbuffer()[0]    ;
        auto ss3  = grace::variable_list::get().getstagstagingbuffer()[0];
        // S2 method of 
        // coefficients 
        double delta, gamma1, gamma2, beta ; 
        // stage 1 
        delta = 1.0 ; 
        gamma1 = 0.0 ; gamma2 = 1.0;
        beta = 1.193743905974738;
        // s2 = s2 + delta s1 
        // we already start with s2 = s1 
        // s3 = gamma1 s1 + gamma2 s2 
        linop_apply(
            s3,s1,s2,
            ss3,ss1,ss2,
            gamma1,gamma2
        ) ; 
        // s3 = s3 + dt F(s1)
        advance_substep<eos_t>(
            t,dt,beta,
            s3,s1,
            ss3,ss1
        ) ; 
        // apply bc 
        amr::apply_boundary_conditions(s3,ss3,s1,ss1,dt,beta) ;
        compute_auxiliary_quantities<eos_t>(s3, ss3, aux) ;  
        // stage 2 
        delta  = 0.217683334308543;
        gamma1 = 0.121098479554482 ; gamma2 = 0.721781678111411;
        beta = 0.099279895495783;
        // s2 = s2 + delta s3 
        linop_apply(
            s2,s2,s3,
            ss2,ss2,ss3,
            1.0,delta 
        ) ;
        // s1 = gamma1 s3 + gamma2 s2 
        linop_apply(
            s1,s3,s2,
            ss1,ss3,ss2,
            gamma1,gamma2
        ) ; 
        // s1 += beta dt F(s3)
        advance_substep<eos_t>(
            t,dt,beta,
            s1,s3,
            ss1,ss3
        ) ; 
        // apply bc 
        amr::apply_boundary_conditions(s1,ss1,s3,ss3,dt,beta) ;
        compute_auxiliary_quantities<eos_t>(s1, ss1, aux) ; 
        // stage 3
        delta  = 1.065841341361089;
        gamma1 = -3.843833699660025 ; gamma2 = 2.121209265338722;
        beta = 1.131678018054042;
        // s2 = s2 + delta s1 
        linop_apply(
            s2,s2,s1,
            ss2,ss2,ss1,
            1.0,delta
        ) ;
        // s3 = gamma1 s1 + gamma2 s2 
        linop_apply(
            s3,s1,s2,
            ss3,ss1,ss2,
            gamma1,gamma2
        ) ; 
        // s3 = s3 + dt F(s1)
        advance_substep<eos_t>(
            t,dt,beta,
            s3,s1,
            ss3,ss1
        ) ; 
        // apply bc 
        amr::apply_boundary_conditions(s3,ss3,s1,ss1,dt,beta) ;
        compute_auxiliary_quantities<eos_t>(s3, ss3, aux) ; 
        // stage 4
        delta  = 0.000000000000000;
        gamma1 = 0.546370891121863 ; gamma2 = 0.198653035682705;
        beta = 0.310665766509336;
        // s2 = s2 + delta s3 (delta == 0, skip!)
        // -- 
        // s1 = gamma1 s3 + gamma2 s2 
        linop_apply(
            s1,s3,s2,
            ss1,ss3,ss2,
            gamma1,gamma2
        ) ; 
        // s1 += beta dt F(s3)
        advance_substep<eos_t>(
            t,dt,beta,
            s1,s3,
            ss1,ss3
        ) ; 
        // apply bc 
        amr::apply_boundary_conditions(s1,ss1,s3,ss3,dt,beta) ;
        compute_auxiliary_quantities<eos_t>(s1, ss1, aux) ; 
    } else if (tstepper == "imex1") { 
        advance_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ;
        amr::apply_boundary_conditions(state,sstate,state_p,sstate_p,dt,1.0) ; 
        advance_implicit_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ;
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;  
    } else if (tstepper == "imex222" ) {
        
        double const lambda = 1.0 - 1.0/sqrt(2.0); 
        // Fetch state_pp and sstate_pp 
        auto& stage  = grace::variable_list::get().getstagingbuffer() ;
        auto& sstage = grace::variable_list::get().getstagstagingbuffer();

        // initialize s2 as yˆn 
        auto state_pp = stage[0] ; 
        auto sstate_pp = sstage[0] ; 
        Kokkos::deep_copy(state_pp,state) ; 
        deep_copy(sstate_pp,sstate) ; 

        // xi1 = yˆn + lambda dt G(xi1) 
        // store xi1 in s2 
        advance_implicit_substep<eos_t>(
            t,dt,lambda,
            /*new*/state_pp,/*old*/state,
            /*new*/sstate_pp,/*old*/sstate
        ) ;
        // no bc as implicit update acts in the gzs 
        compute_auxiliary_quantities<eos_t>(state_pp, sstate_pp, aux) ; 
        // set s1 = yˆn + dt X(xi1) 
        advance_substep<eos_t>(
            t,dt,1.0,
            /*new*/state_p,/*old*/state_pp,
            /*new*/sstate_p,/*old*/sstate_pp
        ) ; 
        amr::apply_boundary_conditions(state_p,sstate_p,state_pp,sstate_pp,dt,1.0) ;
        compute_auxiliary_quantities<eos_t>(state_p, sstate_p, aux) ; 
        // set s2 = dt G(xi1)
        linop_apply(
            state_pp, state_pp, state,
            sstate_pp, sstate_pp, sstate,
            1/lambda,-1/lambda
        ) ; 
        // set s  = y^n + dt/2 F(xi1)
        linop_apply(
            state, state, state_p, state_pp,
            sstate, sstate, sstate_p, sstate_pp,
            0.5,0.5,0.5
        ) ;
        // set s1 = yˆn + dt X(xi1) + (1-2 lambda) dt G(xi1)
        linop_apply(
            state_p, state_p, state_pp, 
            sstate_p, sstate_p, sstate_pp,
            1.0, (1.0-2.0*lambda)
        ) ;
        // apply BC 
        // For Sommerfeld: here we keep the BC frozen since 
        // 
        amr::apply_boundary_conditions(state_p,sstate_p,state_p,sstate_p,dt,0.0/*FIXME*/) ; 
        // reset s2 = xi1
        linop_apply(
            state_pp, state_pp, state_p,
            sstate_pp, sstate_pp, sstate_p,
            (2.0-lambda), -1.0
        ) ; 
        // solve xi2 = s1 + lambda dt G(xi2)
        advance_implicit_substep<eos_t>(
            t,dt,lambda,
            /*new*/state_pp,/*old*/state_p,
            /*new*/sstate_pp,/*old*/sstate_p
        );
        // add dt / 2 G(xi2) to s
        linop_apply(
            state, state, state_pp, state_p,
            sstate, sstate, sstate_pp, sstate_p,
            1.0, 1./(2*lambda), -1./(2*lambda)
        ) ; 
        compute_auxiliary_quantities<eos_t>(state_pp, sstate_pp, aux) ; 
        // set s = yˆn + dt/2 (F(xi1)+F(xi2)) == yˆ{n+1}
        advance_substep<eos_t>(
            t, dt, 0.5,
            /*new*/ state, /*old*/ state_pp,
            /*new*/ sstate, /*old*/ sstate_pp
        ) ; 
        amr::apply_boundary_conditions(state,sstate,state_pp,sstate_pp,dt,0.5) ;
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ; 
        /* done */
    } else if (tstepper == "imex232" ) { 
        ERROR("Imex 3 not implemented yet") ; 
    } else {
        ERROR("Unrecognised time-stepper.") ; 
    }
    Kokkos::deep_copy(state_p,state) ;
    grace::deep_copy(sstate_p,sstate) ;

    // Hamiltonian/momentum constraints are now filled in-flight by the
    // curvature half of the Z4c update, so no separate pass is needed here.
    // The standalone compute_constraint_violations() entry point is kept
    // for post-regrid diagnostic refreshes and initial-data setup.
}

template< typename eos_t >
void compute_fluxes(
    double const t, double const dt, double const dtfact 
    , var_array_t& new_state 
    , var_array_t& old_state 
    , staggered_variable_arrays_t & new_stag_state 
    , staggered_variable_arrays_t & old_stag_state 
) 
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& dx     = grace::variable_list::get().getspacings() ;  
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ; 
    auto& aux     = grace::variable_list::get().getaux()     ; 
    #ifdef GRACE_GRMHD_USE_GS
    auto& vbar  = grace::variable_list::get().getefarray() ;
    #else 
    auto& vbar  = grace::variable_list::get().getvbararray() ;
    #endif 
    //**************************************************************************************************/
    // construct grmhd object 
    using recon_t = GRACE_RECONSTRUCTION_T ; 
    auto eos = eos::get().get_eos<eos_t>() ;  
    grmhd_equations_system_t<eos_t>
        grmhd_eq_system(eos,old_state,old_stag_state,aux) ; 
    //**************************************************************************************************/
    #ifdef GRACE_ENABLE_M1
    m1_equations_system_t m1_eq_system(old_state,old_stag_state,aux) ;
    // normalize 
    auto m1_norm_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,0),0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}
        ) ; 
    parallel_for(
          GRACE_EXECUTION_TAG("evol", "m1_normalize_conservs")
        , m1_norm_policy
        , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
            metric_array_t metric ; 
            FILL_METRIC_ARRAY(metric,old_state,q,VEC(i,j,k)) ;
            old_state(VEC(i,j,k),NRAD_,q)  /=  old_state(VEC(i,j,k),ERAD_,q); 
            old_state(VEC(i,j,k),FRADX_,q) /=  old_state(VEC(i,j,k),ERAD_,q); 
            old_state(VEC(i,j,k),FRADY_,q) /=  old_state(VEC(i,j,k),ERAD_,q); 
            old_state(VEC(i,j,k),FRADZ_,q) /=  old_state(VEC(i,j,k),ERAD_,q); 
            old_state(VEC(i,j,k),ERAD_,q)  /= metric.sqrtg() ; 
            #ifdef M1_NU_THREESPECIES
            old_state(VEC(i,j,k),NRAD1_,q)  /=  old_state(VEC(i,j,k),ERAD1_,q); 
            old_state(VEC(i,j,k),FRADX1_,q) /=  old_state(VEC(i,j,k),ERAD1_,q); 
            old_state(VEC(i,j,k),FRADY1_,q) /=  old_state(VEC(i,j,k),ERAD1_,q); 
            old_state(VEC(i,j,k),FRADZ1_,q) /=  old_state(VEC(i,j,k),ERAD1_,q); 
            old_state(VEC(i,j,k),ERAD1_,q)  /= metric.sqrtg() ; 

            old_state(VEC(i,j,k),NRAD2_,q)  /=  old_state(VEC(i,j,k),ERAD2_,q); 
            old_state(VEC(i,j,k),FRADX2_,q) /=  old_state(VEC(i,j,k),ERAD2_,q); 
            old_state(VEC(i,j,k),FRADY2_,q) /=  old_state(VEC(i,j,k),ERAD2_,q); 
            old_state(VEC(i,j,k),FRADZ2_,q) /=  old_state(VEC(i,j,k),ERAD2_,q); 
            old_state(VEC(i,j,k),ERAD2_,q)  /= metric.sqrtg() ; 
            #endif 
        }
    ) ; 
    #endif 
    //**************************************************************************************************/
    // loop ranges: extended for mhd (need vbar at faces for emf)
    auto flux_x_policy_mhd = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,0,0),0}
            , {VEC(nx+ngz+1,ny+2*ngz,nz+2*ngz),nq}
        ) ;
    auto flux_y_policy_mhd = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,ngz,0),0}
            , {VEC(nx+2*ngz,ny+ngz+1,nz+2*ngz),nq}
        ) ;
    auto flux_z_policy_mhd = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,ngz),0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+ngz+1),nq}
        ) ; 
    // non-mhd 
    auto flux_x_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz,nz+ngz),nq}
        ) ;
    auto flux_y_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz+1,nz+ngz),nq}
        ) ;
    auto flux_z_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz+1),nq}
        ) ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    // compute x flux 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_grmhd_x_flux")
                , flux_x_policy_mhd 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        #ifndef GRACE_FREEZE_HYDRO
        grmhd_eq_system.template compute_x_flux<recon_t>(q, VEC(i,j,k), fluxes, vbar, dx, dt, dtfact) ;
        #endif 
    }) ; 
    #ifdef GRACE_ENABLE_M1
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_M1_x_flux")
                , flux_x_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
         m1_eq_system.template compute_x_flux<slope_limited_reconstructor_t<MCbeta>,0>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        ) ; 
        #ifdef M1_NU_THREESPECIES
        m1_eq_system.template compute_x_flux<slope_limited_reconstructor_t<MCbeta>,1>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        );
        m1_eq_system.template compute_x_flux<slope_limited_reconstructor_t<MCbeta>,2>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        );
        #endif
    }) ; 
    #endif 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_grmhd_y_flux")
                , flux_y_policy_mhd 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        #ifndef GRACE_FREEZE_HYDRO
        grmhd_eq_system.template compute_y_flux<recon_t>(q, VEC(i,j,k), fluxes, vbar, dx, dt, dtfact);
        #endif 
    }) ;
    #ifdef GRACE_ENABLE_M1
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_M1_y_flux")
                , flux_y_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
         m1_eq_system.template compute_y_flux<slope_limited_reconstructor_t<MCbeta>,0>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        ) ; 
        #ifdef M1_NU_THREESPECIES
        m1_eq_system.template compute_y_flux<slope_limited_reconstructor_t<MCbeta>,1>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        );
        m1_eq_system.template compute_y_flux<slope_limited_reconstructor_t<MCbeta>,2>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        );
        #endif
    }) ; 
    #endif 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_grmhd_z_flux")
                , flux_z_policy_mhd 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        #ifndef GRACE_FREEZE_HYDRO
        grmhd_eq_system.template compute_z_flux<recon_t>(q, VEC(i,j,k), fluxes, vbar, dx, dt, dtfact);
        #endif 
    }) ; 
    #ifdef GRACE_ENABLE_M1
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_M1_z_flux")
                , flux_z_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
         m1_eq_system.template compute_z_flux<slope_limited_reconstructor_t<MCbeta>,0>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        ) ; 
        #ifdef M1_NU_THREESPECIES
        m1_eq_system.template compute_z_flux<slope_limited_reconstructor_t<MCbeta>,1>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        );
        m1_eq_system.template compute_z_flux<slope_limited_reconstructor_t<MCbeta>,2>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        );
        #endif
    }) ; 
    #endif 
    //**************************************************************************************************/
    Kokkos::fence() ; 
    #ifdef GRACE_ENABLE_M1
    // un-normalize
    parallel_for(
          GRACE_EXECUTION_TAG("evol", "m1_unnormalize_conservs")
        , m1_norm_policy
        , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {

            metric_array_t metric ; 
            FILL_METRIC_ARRAY(metric,old_state,q,VEC(i,j,k)) ;
            old_state(VEC(i,j,k),ERAD_,q)  *= metric.sqrtg() ; 
            old_state(VEC(i,j,k),NRAD_,q)  *=  old_state(VEC(i,j,k),ERAD_,q); 
            old_state(VEC(i,j,k),FRADX_,q) *=  old_state(VEC(i,j,k),ERAD_,q); 
            old_state(VEC(i,j,k),FRADY_,q) *=  old_state(VEC(i,j,k),ERAD_,q); 
            old_state(VEC(i,j,k),FRADZ_,q) *=  old_state(VEC(i,j,k),ERAD_,q); 
            #ifdef M1_NU_THREESPECIES
            old_state(VEC(i,j,k),ERAD1_,q)  *= metric.sqrtg() ; 
            old_state(VEC(i,j,k),NRAD1_,q)  *=  old_state(VEC(i,j,k),ERAD1_,q); 
            old_state(VEC(i,j,k),FRADX1_,q) *=  old_state(VEC(i,j,k),ERAD1_,q); 
            old_state(VEC(i,j,k),FRADY1_,q) *=  old_state(VEC(i,j,k),ERAD1_,q); 
            old_state(VEC(i,j,k),FRADZ1_,q) *=  old_state(VEC(i,j,k),ERAD1_,q);

            old_state(VEC(i,j,k),ERAD2_,q)  *= metric.sqrtg() ; 
            old_state(VEC(i,j,k),NRAD2_,q)  *=  old_state(VEC(i,j,k),ERAD2_,q); 
            old_state(VEC(i,j,k),FRADX2_,q) *=  old_state(VEC(i,j,k),ERAD2_,q); 
            old_state(VEC(i,j,k),FRADY2_,q) *=  old_state(VEC(i,j,k),ERAD2_,q); 
            old_state(VEC(i,j,k),FRADZ2_,q) *=  old_state(VEC(i,j,k),ERAD2_,q);
            #endif 
        }
    ) ; 
    #endif 
}
#ifdef GRACE_GRMHD_USE_GS
void compute_emfs(
    double const t, double const dt, double const dtfact 
    , var_array_t& new_state 
    , var_array_t& old_state 
    , staggered_variable_arrays_t & new_stag_state 
    , staggered_variable_arrays_t & old_stag_state 
) 
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    using recon_t = GRACE_RECONSTRUCTION_T ; 
    auto& idx     = grace::variable_list::get().getinvspacings() ;
    auto& fluxes = grace::variable_list::get().getfluxesarray() ;
    auto& Eface  = grace::variable_list::get().getefarray() ;
    auto& Ecenter  = grace::variable_list::get().getecarray() ;
    auto& emf  = grace::variable_list::get().getemfarray() ;
    //**************************************************************************************************/
    // loop ranges 
    auto emf_policy_x = 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
            {VEC(ngz,ngz,ngz),0}
        , {VEC(nx+ngz,ny+ngz+1,nz+ngz+1),nq}
    ) ;
    auto emf_policy_y = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz,nz+ngz+1),nq}
    ) ;
    auto emf_policy_z = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz+1,nz+ngz),nq}
    ) ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    // compute EMF -- x (stag yz)
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_X")
                , emf_policy_x 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // Here direction 0 is y and direction 1 is z 
        // meaning that South-North is -+ z and West-East is -+ y
        double Exzf  = Eface(i,j,k,0,2,q)    ;
        double Exzfm = Eface(i,j-1,k,0,2,q)  ;
        double Exyf  = Eface(i,j,k,0,1,q)    ;
        double Exyfm = Eface(i,j,k-1,0,1,q)  ;
        // and centered 
        double ExNE = Ecenter(i,j,k,0,q)     ; 
        double ExNW = Ecenter(i,j-1,k,0,q)   ; 
        double ExSE = Ecenter(i,j,k-1,0,q)   ; 
        double ExSW = Ecenter(i,j-1,k-1,0,q) ;
        // first compute arithmetic average 
        double Eavg = 0.25 * ( 
            Exzf + Exzfm
          + Exyf + Exyfm    
        ) ;
        // first derivative piece:
        // dE/dz (north south)
        // get signs 
        double Sy  = Kokkos::copysign(1.,fluxes(i,j,k,DENS_,1,q))   ; 
        double Sym = Kokkos::copysign(1.,fluxes(i,j,k-1,DENS_,1,q)) ; 
        // Get "North"
        double dEdzN = (1-Sy) * (ExNE-Exzf)
                     + (1+Sy) * (ExNW-Exzfm) ; 
        // "South"
        double dEdzS = (1-Sym) * (Exzf-ExSE) 
                     + (1+Sym) * (Exzfm-ExSW) ; 
        // Assemble 
        double dEdz = 1./8. * (dEdzS - dEdzN) ; 
        // second derivative piece 
        // Other signs 
        double Sz  = Kokkos::copysign(1.,fluxes(i,j,k,DENS_,2,q)) ; 
        double Szm = Kokkos::copysign(1.,fluxes(i,j-1,k,DENS_,2,q)) ; 
        // Get "East"
        double dEdyE = (1-Sz) * (ExNE - Exyf)
                     + (1+Sz) * (ExSE - Exyfm) ; 
        // "West"
        double dEdyW = (1-Szm) * (Exyf  - ExNW) 
                     + (1+Szm) * (Exyfm - ExSW) ; 
        // Assemble 
        double dEdy = 1./8. * (dEdyW - dEdyE) ; 
        // finally 
        emf(i,j,k,0,q) = Eavg + dEdz + dEdy ; 
    } );
    //**************************************************************************************************/
    // compute EMF -- y (stag xz)
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_Y")
                , emf_policy_y 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // Here direction 0 is x and direction 1 is z 
        // meaning that South-North is -+ z and West-East is -+ x 
        // copy locally 
        double Eyzf  = Eface(i,j,k,1,2,q)    ;
        double Eyzfm = Eface(i-1,j,k,1,2,q)  ;
        double Eyxf  = Eface(i,j,k,0,0,q)    ;
        double Eyxfm = Eface(i,j,k-1,0,0,q)  ;
        // and centered 
        double EyNE = Ecenter(i,j,k,1,q)     ; 
        double EyNW = Ecenter(i-1,j,k,1,q)   ; 
        double EySE = Ecenter(i,j,k-1,1,q)   ; 
        double EySW = Ecenter(i-1,j,k-1,1,q) ; 
        // first compute arithmetic average 
        double Eavg = 0.25 * ( 
            Eyzf + Eyzfm
          + Eyxf + Eyxfm    
        ) ;
        // first derivative piece:
        // dE/dz (north south)
        // get signs 
        double Sx  = Kokkos::copysign(1.,fluxes(i,j,k,DENS_,0,q)) ; 
        double Sxm = Kokkos::copysign(1.,fluxes(i,j,k-1,DENS_,0,q)) ; 
        // Get "North"
        double dEdzN = (1-Sx) * (EyNE-Eyzf)
                     + (1+Sx) * (EyNW-Eyzfm) ; 
        // "South"
        double dEdzS = (1-Sxm) * (Eyzf-EySE) 
                     + (1+Sxm) * (Eyzfm-EySW) ; 
        // Assemble (Avengers!)
        double dEdz = 1./8. * (dEdzS - dEdzN) ; 
        // Other signs 
        double Sz  = Kokkos::copysign(1.,fluxes(i,j,k,DENS_,2,q)) ; 
        double Szm = Kokkos::copysign(1.,fluxes(i-1,j,k,DENS_,2,q)) ; 
        // "East"
        double dEdxE = (1-Sz) * (EyNE - Eyxf)
                     + (1+Sz) * (EySE - Eyxfm) ; 
        // "West"
        double dEdxW = (1-Szm) * (Eyxf  - EyNW) 
                     + (1+Szm) * (Eyxfm - EySW) ; 
        // Assemble 
        double dEdx = 1./8. * (dEdxW - dEdxE) ; 
        // finally 
        emf(i,j,k,1,q) = Eavg + dEdz + dEdx ; 

    } ) ; 
    //**************************************************************************************************/
    // compute EMF -- z (stag xy)
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_Z")
                , emf_policy_z 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // E^z_{i-1/2, j-1/2, k} = 1/4 ( avg E from faces )
        //                       + (1+Sxf)/2 (Fx_{yf} - F_x)
        // copy locally 
        double Ezyf  = Eface(i,j,k,1,1,q)    ;
        double Ezyfm = Eface(i-1,j,k,1,1,q)  ;
        double Ezxf  = Eface(i,j,k,1,0,q)    ;
        double Ezxfm = Eface(i,j-1,k,1,0,q)  ;
        // and centered 
        double EzNE = Ecenter(i,j,k,2,q)     ; 
        double EzNW = Ecenter(i-1,j,k,2,q)   ; 
        double EzSE = Ecenter(i,j-1,k,2,q)   ; 
        double EzSW = Ecenter(i-1,j-1,k,2,q) ; 
        // first compute arithmetic average 
        double Eavg = 0.25 * ( 
            Ezyf + Ezyfm   // E_z(i,j-1/2,k) + E_z(i-1,j-1/2,k)
          + Ezxf + Ezxfm   // E_z(i-1/2,j,k) + E_z(i-1/2,j-1,k) 
        ) ; 

        // then we need the derivatives 
        // get the sign of vbar at x face 
        auto Sx  = Kokkos::copysign(1.0, fluxes(i,j,k,DENS_,0,q))   ; 
        auto Sxm = Kokkos::copysign(1.0, fluxes(i,j-1,k,DENS_,0,q)) ;
        // first derivative term  
        // "north"
        double dEdyN =  (1.-Sx) * ( EzNE - Ezyf  )    // select if v < 0
                     +  (1.+Sx) * ( EzNW - Ezyfm ) ;  // select if v > 0 
        // "south"
        double dEdyS =  (1.-Sxm) * ( Ezyf  - EzSE )
                     +  (1.+Sxm) * ( Ezyfm - EzSW ) ; 
        double dEdy  = 1./ 8. * ( dEdyS - dEdyN ) ; 
        // Get the sign of vbar at y face 
        auto Sy  = Kokkos::copysign(1.0, fluxes(i,j,k,DENS_,1,q))   ; 
        auto Sym = Kokkos::copysign(1.0, fluxes(i-1,j,k,DENS_,1,q))   ; 
        // second derivative term 
        // "east"
        double dEdxE = (1-Sy) * ( EzNE - Ezxf  ) 
                     + (1+Sy) * ( EzSE - Ezxfm ) ;
        // "west" 
        double dEdxW = (1-Sym) * ( Ezxf  - EzNW  ) 
                     + (1+Sym) * ( Ezxfm - EzSW ) ;
        // combine 
        double dEdx = 1./8. * ( dEdxW - dEdxE ) ; 
        
        // finally 
        emf(i,j,k,2,q) = Eavg + dEdy + dEdx ; 

    } ) ;
    //**************************************************************************************************/
    // all done! 
}
#else 
void compute_emfs(
    double const t, double const dt, double const dtfact 
    , var_array_t& new_state 
    , var_array_t& old_state 
    , staggered_variable_arrays_t & new_stag_state 
    , staggered_variable_arrays_t & old_stag_state 
) 
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    using recon_t = GRACE_RECONSTRUCTION_T ; 
    auto& idx     = grace::variable_list::get().getinvspacings() ;
    auto& vbar  = grace::variable_list::get().getvbararray() ;
    auto& emf  = grace::variable_list::get().getemfarray() ; 
    //**************************************************************************************************/
    // some ugly macros 
    #define RECONSTRUCT(vview,vidx,q,i,j,k,uL,uR,dir) \
    do { \
    auto sview = subview(vview, \
                                ALL(), \
                                ALL(), \
                                ALL(), \
                                static_cast<size_t>(vidx), \
                                q     ) ; \
    reconstructor(sview,i,j,k,uL,uR,dir) ; \
    } while(false)
    #define RECONSTRUCT_V(vview,jdir,vidx,q,i,j,k,uL,uR,dir) \
    do { \
    auto sview = subview(vview, \
                                ALL(), \
                                ALL(), \
                                ALL(), \
                                static_cast<size_t>(vidx), \
                                jdir, \
                                q     ) ; \
    reconstructor(sview,i,j,k,uL,uR,dir) ; \
    } while(false)
    //**************************************************************************************************/
    // loop ranges 
    auto emf_policy_x = 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
            {VEC(ngz,ngz,ngz),0}
        , {VEC(nx+ngz,ny+ngz+1,nz+ngz+1),nq}
    ) ;
    auto emf_policy_y = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz,nz+ngz+1),nq}
    ) ;
    auto emf_policy_z = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz+1,nz+ngz),nq}
    ) ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    // compute EMF -- x (stag yz)
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_X")
                , emf_policy_x 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // i, j-1/2, k-1/2 
        // Ex = vz By - vy Bz 
        // reconstruct vz and By
        recon_t reconstructor {} ; 
        hll_riemann_solver_t solver {} ;
        double ByL,ByR;
        RECONSTRUCT(
            old_stag_state.face_staggered_fields_y, BSY_, q, i,j,k, ByL, ByR, 2 /*recon along z*/
        ) ; 
        double vbarL_z,vbarR_z;
        RECONSTRUCT_V(
            vbar, 1 /*y-stagger*/, 1 /*vx,vz so 1*/, q, i,j,k, vbarL_z,vbarR_z, 2 /*recon along z*/
        ) ;
        // reconstruct vy and Bz
        double BzL,BzR;
        RECONSTRUCT(
            old_stag_state.face_staggered_fields_z, BSZ_, q, i,j,k, BzL, BzR, 1 /*recon along y*/
        ) ; 
        double vbarL_y,vbarR_y;
        RECONSTRUCT_V(
            vbar, 2 /*z-stagger*/, 1 /*vx,vy so 1*/, q, i,j,k, vbarL_y,vbarR_y, 1 /*recon along y*/
        ) ;

        // now we find the wavespeeds 
        // this is min(cmin_y(i,j-1/2,k), cmin_y(i,j-1/2,k-1))
        auto cmin_y = Kokkos::max(vbar(VEC(i,j,k),2,1,q),vbar(VEC(i,j,k-1),2,1,q)) ;
        // this is max(cmax_y(i,j-1/2,k), cmax_y(i,j-1/2,k-1)) 
        auto cmax_y = Kokkos::max(vbar(VEC(i,j,k),3,1,q),vbar(VEC(i,j,k-1),3,1,q)) ;
        // this is min(cmin_z(i,j,k-1/2), cmin_z(i,j-1,k-1/2))
        auto cmin_z = Kokkos::max(vbar(VEC(i,j,k),2,2,q),vbar(VEC(i,j-1,k),2,2,q)) ;
        // this is max(cmax_z(i,j,k-1/2), cmax_y(i,j-1,k-1/2)) 
        auto cmax_z = Kokkos::max(vbar(VEC(i,j,k),3,2,q),vbar(VEC(i,j-1,k),3,2,q)) ;

        // now we can finally compute the EMF 
        // E^x_{i,j-1/2,k-1/2} = ( cmax_z vbarL_z ByL + cmin_z vbarR_z ByR - cmax_z cmin_z (ByR -ByL) ) / ( cmax_z + cmin_z )
        //                     - ( cmax_y vbarL_y BzL + cmin_y vbarR_y BzR - cmax_y cmin_y (BzR -BzL) ) / ( cmax_y + cmin_y )
        emf(VEC(i,j,k),0,q) = solver(vbarL_z*ByL, vbarR_z*ByR, ByL, ByR, cmin_z, cmax_z)
                            - solver(vbarL_y*BzL, vbarR_y*BzR, BzL, BzR, cmin_y, cmax_y) ; 
        
    } ) ;
    //**************************************************************************************************/
    // compute EMF -- y (stag xz)
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_Y")
                , emf_policy_y 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // i-1/2, j, k-1/2 
        // Ex = vx Bz - vz Bx 
        // reconstruct vx and Bz
        recon_t reconstructor {} ; 
        hll_riemann_solver_t solver {} ;
        // Bz_{i,j,k-1/2} --> Bz_{i-1/2,j,k-1/2} 
        double BzL,BzR;
        RECONSTRUCT(
            old_stag_state.face_staggered_fields_z, BSZ_, q, i,j,k, BzL, BzR, 0 /*recon along x*/
        ) ; 
        // vx_{i,j,k-1/2} --> vx_{i-1/2,j,k-1/2} 
        double vbarL_x,vbarR_x;
        RECONSTRUCT_V(
            vbar, 2 /*z-stagger*/, 0 /*vx,vy so 0*/, q, i,j,k, vbarL_x,vbarR_x, 0 /*recon along x*/
        ) ;
        // reconstruct vz and Bx
        // Bx_{i-1/2,j,k} --> Bx_{i-1/2,j,k-1/2} 
        double BxL,BxR;
        RECONSTRUCT(
            old_stag_state.face_staggered_fields_x, BSX_, q, i,j,k, BxL, BxR, 2 /*recon along z*/
        ) ; 
        // vz_{i-1/2,j,k} --> vz_{i-1/2,j,k-1/2} 
        double vbarL_z,vbarR_z;
        RECONSTRUCT_V(
            vbar, 0 /*x-stagger*/, 1 /*vy,vz so 1*/, q, i,j,k, vbarL_z,vbarR_z, 2 /*recon along z*/
        ) ;

        // now we find the wavespeeds 
        // this is min(cmin_z(i,j,k-1/2), cmin_z(i-1,j,k-1/2))
        auto cmin_z = Kokkos::max(vbar(VEC(i,j,k),2,2,q),vbar(VEC(i-1,j,k),2,2,q)) ;
        // this is max(cmax_z(i,j,k-1/2), cmax_z(i-1,j,k-1/2))
        auto cmax_z = Kokkos::max(vbar(VEC(i,j,k),3,2,q),vbar(VEC(i-1,j,k),3,2,q)) ;

        // this is min(cmin_x(i-1/2,j,k), cmin_z(i-1/2,j,k-1))
        auto cmin_x = Kokkos::max(vbar(VEC(i,j,k),2,0,q),vbar(VEC(i,j,k-1),2,0,q)) ;
        // this is max(cmax_x(i-1/2,j,k), cmax_x(i-1/2,j,k-1))
        auto cmax_x = Kokkos::max(vbar(VEC(i,j,k),3,0,q),vbar(VEC(i,j,k-1),3,0,q)) ;

        // now we can finally compute the EMF 
        // E^y_{i-1/2,j,k-1/2} = ( cmax_x vbarL_x BzL + cmin_x vbarR_x BzR - cmax_x cmin_x (BzR -BzL) ) / ( cmax_x + cmin_x )
        //                     - ( cmax_z vbarL_z BxL + cmin_z vbarR_z BxR - cmax_z cmin_z (BxR -BxL) ) / ( cmax_z + cmin_z )
        emf(VEC(i,j,k),1,q) = solver(vbarL_x*BzL,vbarR_x*BzR,BzL,BzR,cmin_x,cmax_x) 
                            - solver(vbarL_z*BxL,vbarR_z*BxR,BxL,BxR,cmin_z,cmax_z) ; 
        
    } ) ;
    //**************************************************************************************************/
    // compute EMF -- z (stag xy)
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_Z")
                , emf_policy_z 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // i-1/2, j-1/2, k 
        // Ez = vy Bx - vx By 
        // reconstruct vy and Bx
        recon_t reconstructor {} ; 
        hll_riemann_solver_t solver {} ;
        // Bx_{i-1/2,j,k} --> Bx_{i-1/2,j-1/2,k} 
        double BxL,BxR;
        RECONSTRUCT(
            old_stag_state.face_staggered_fields_x, BSX_, q, i,j,k, BxL, BxR, 1 /*recon along y*/
        ) ; 
        // vy_{i-1/2,j,k} --> vx_{i-1/2,j-1/2,k} 
        double vbarL_y,vbarR_y;
        RECONSTRUCT_V(
            vbar, 0 /*x-stagger*/, 0 /*vy,vz so 0*/, q, i,j,k, vbarL_y,vbarR_y, 1 /*recon along y*/
        ) ;
        // reconstruct vx and By
        // By_{i,j-1/2,k} --> Bx_{i-1/2,j-1/2,k} 
        double ByL,ByR;
        RECONSTRUCT(
            old_stag_state.face_staggered_fields_y, BSY_, q, i,j,k, ByL, ByR, 0 /*recon along x*/
        ) ; 
        // vz_{i,j-1/2,k} --> vz_{i-1/2,j-1/2,k} 
        double vbarL_x,vbarR_x;
        RECONSTRUCT_V(
            vbar, 1 /*y-stagger*/, 0 /*vx,vz so 0*/, q, i,j,k, vbarL_x,vbarR_x, 0 /*recon along x*/
        ) ;

        // now we find the wavespeeds 
        // this is min(cmin_x(i-1/2,j,k), cmin_x(i-1/2,j-1,k)
        auto cmin_x = Kokkos::max(vbar(VEC(i,j,k),2,0,q),vbar(VEC(i,j-1,k),2,0,q)) ;
        // this is max(cmax_x(i-1/2,j,k), cmax_x(i-1/2,j-1,k)
        auto cmax_x = Kokkos::max(vbar(VEC(i,j,k),3,0,q),vbar(VEC(i,j-1,k),3,0,q)) ;

        // this is min(cmin_y(i,j-1/2,k), cmin_y(i-1,j-1/2,k))
        auto cmin_y = Kokkos::max(vbar(VEC(i,j,k),2,1,q),vbar(VEC(i-1,j,k),2,1,q)) ;
        // this is max(cmax_y(i,j-1/2,k), cmax_y(i-1,j-1/2,k))
        auto cmax_y = Kokkos::max(vbar(VEC(i,j,k),3,1,q),vbar(VEC(i-1,j,k),3,1,q)) ;

        // now we can finally compute the EMF 
        // E^z_{i-1/2,j,k-1/2} = ( cmax_y vbarL_y BxL + cmin_y vbarR_y BxR - cmax_y cmin_y (BxR -BxL) ) / ( cmax_y + cmin_y )
        //                     - ( cmax_x vbarL_x ByL + cmin_x vbarR_x ByR - cmax_x cmin_x (ByR -ByL) ) / ( cmax_x + cmin_x )
        emf(VEC(i,j,k),2,q) = solver(vbarL_y*BxL,vbarR_y*BxR,BxL,BxR,cmin_y,cmax_y) 
                            - solver(vbarL_x*ByL,vbarR_x*ByR,ByL,ByR,cmin_x,cmax_x) ; 
        
    } ) ;
    //**************************************************************************************************/
    Kokkos::fence() ; 
    //**************************************************************************************************/
}
#endif 
template< typename eos_t >
void add_fluxes_and_source_terms(
    double const t, double const dt, double const dtfact 
    , var_array_t& new_state 
    , var_array_t& old_state 
    , staggered_variable_arrays_t & new_stag_state 
    , staggered_variable_arrays_t & old_stag_state 
)
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& dx     = grace::variable_list::get().getspacings() ;  
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ; 
    auto& aux     = grace::variable_list::get().getaux()     ; 
    int nvars_hrsc = variables::get_n_hrsc() ;
    //**************************************************************************************************/
    // construct grmhd object 
    auto eos = eos::get().get_eos<eos_t>() ;  
    grmhd_equations_system_t<eos_t>
        grmhd_eq_system(eos,old_state,old_stag_state,aux) ;
    #ifdef GRACE_ENABLE_M1 
    m1_equations_system_t m1_eq_system(old_state,old_stag_state,aux) ; 
    #endif 
    //**************************************************************************************************/
    // loop range 
    auto policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
        ) ;
    //**************************************************************************************************/
    #ifdef GRACE_ENABLE_M1 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_sources_M1")
                , policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        
        m1_eq_system.compute_source_terms<0>(q, VEC(i,j,k), idx, new_state, dt, dtfact );
        #ifdef M1_NU_THREESPECIES
        m1_eq_system.compute_source_terms<1>(q, VEC(i,j,k), idx, new_state, dt, dtfact );
        m1_eq_system.compute_source_terms<2>(q, VEC(i,j,k), idx, new_state, dt, dtfact );
        #endif 
    });
    #endif 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_sources")
                , policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        #ifndef GRACE_FREEZE_HYDRO
        grmhd_eq_system(sources_computation_kernel_t{}, q, VEC(i,j,k), idx, new_state, dt, dtfact );
        #endif 
        for( int ivar=0; ivar<nvars_hrsc; ++ivar) {
            new_state(VEC(i,j,k),ivar,q) += 
                dt * dtfact * (
                EXPR(   ( fluxes(VEC(i,j,k)  ,ivar,0,q) - fluxes(VEC(i+1,j,k),ivar,0,q) ) * idx(0,q)
                    , + ( fluxes(VEC(i,j,k)  ,ivar,1,q) - fluxes(VEC(i,j+1,k),ivar,1,q) ) * idx(1,q)
                    , + ( fluxes(VEC(i,j,k)  ,ivar,2,q) - fluxes(VEC(i,j,k+1),ivar,2,q) ) * idx(2,q))
            ) ;
        }
        
    }) ; 
    // fixme better to have two kernels? --> I don't think so nvars_hrsc is small.
}

void update_CT(
    double const t, double const dt, double const dtfact 
    , var_array_t& new_state 
    , var_array_t& old_state 
    , staggered_variable_arrays_t & new_stag_state 
    , staggered_variable_arrays_t & old_stag_state 
)
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;
    auto& emf  = grace::variable_list::get().getemfarray() ; 
    //**************************************************************************************************/
    // loop ranges 
    auto advance_stag_policy_x = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz,nz+ngz),nq}
        ) ;
    auto advance_stag_policy_y = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz+1,nz+ngz),nq}
        ) ;
    auto advance_stag_policy_z = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz+1),nq}
        ) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "CT_advance_BX")
                , advance_stag_policy_x 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // d/dt B^x_i-1/2,j,k = d/dz E^y - d/dy E^z
        //                    = 1/dz (E^y_{i-1/2,j,k+1/2}-E^y_{i-1/2,j,k-1/2})
        //                    + 1/dy (E^z_{i-1/2,j-1/2,k}-E^z_{i-1/2,j+1/2,k})
        new_stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) += dt * dtfact * (
            (emf(VEC(i,j,k+1),1,q)-emf(VEC(i,j,k),1,q)) * idx(2,q)
          + (emf(VEC(i,j,k),2,q)-emf(VEC(i,j+1,k),2,q)) * idx(1,q)
        )  ; 
    } ) ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "CT_advance_BY")
                , advance_stag_policy_y 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // d/dt B^y_i,j-1/2,k = d/dx E^z - d/dz E^x
        //                    = 1/dx (E^z_{i+1/2,j-1/2,k}-E^z_{i-1/2,j-1/2,k})
        //                    + 1/dz (E^x_{i,j-1/2,k-1/2}-E^x_{i,j-1/2,k+1/2})
        new_stag_state.face_staggered_fields_y(VEC(i,j,k), BSY_, q) += dt * dtfact * (
              (emf(VEC(i+1,j,k),2,q) - emf(VEC(i,j,k),2,q)) * idx(0,q)
            + (emf(VEC(i,j,k),0,q) - emf(VEC(i,j,k+1),0,q)) * idx(2,q)
        );
    } ) ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "CT_advance_BZ")
                , advance_stag_policy_z 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // d/dt B^z_i,j,k-1/2 = d/dy E^x - d/dx E^y 
        //                    = 1/dy (E^x_{i,j+1/2,k-1/2}-E^x_{i,j-1/2,k-1/2})
        //                    + 1/dx (E^y_{i,j,k-1/2}-E^y_{i+1/2,j,k-1/2})
        new_stag_state.face_staggered_fields_z(VEC(i,j,k), BSZ_, q) += dt * dtfact * (
              (emf(VEC(i,j+1,k),0,q) - emf(VEC(i,j,k),0,q)) * idx(1,q)
            + (emf(VEC(i,j,k),1,q) - emf(VEC(i+1,j,k),1,q)) * idx(0,q)
        );
    } ) ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
}

void update_fd(
    double const t, double const dt, double const dtfact 
    , var_array_t& new_state 
    , var_array_t& old_state 
    , staggered_variable_arrays_t & new_stag_state 
    , staggered_variable_arrays_t & old_stag_state
) 
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ;
    //**************************************************************************************************/
    #ifdef GRACE_ENABLE_Z4C_METRIC
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& aux     = grace::variable_list::get().getaux()     ; 
    auto dev_coords = grace::coordinate_system::get().get_device_coord_system() ; 
    //**************************************************************************************************/
    z4c_system_t z4c_eq_system(old_state,aux,old_stag_state) ;
    //**************************************************************************************************/
    // Launch bounds: the advective kernel is bandwidth-bound with a small
    // register footprint, so we ask for high occupancy.  The curvature
    // kernel is heavily register-pressured and benefits from the compiler
    // allocating more registers per thread at the cost of occupancy.
    //
    // Override per-architecture via compile flags if needed.
    #ifndef GRACE_Z4C_ADV_LB
      #define GRACE_Z4C_ADV_LB  Kokkos::LaunchBounds<256, 4>
    #endif
    #ifndef GRACE_Z4C_CURV_LB
      #define GRACE_Z4C_CURV_LB Kokkos::LaunchBounds<128, 1>
    #endif
    using adv_policy_t =
        MDRangePolicy< Rank<GRACE_NSPACEDIM+1>, GRACE_Z4C_ADV_LB > ;
    using curv_policy_t =
        MDRangePolicy< Rank<GRACE_NSPACEDIM+1>, GRACE_Z4C_CURV_LB > ;
    adv_policy_t advective_policy (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
        ) ;
    curv_policy_t curvature_policy (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
        ) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_advective_update")
                , advective_policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    z4c_eq_system.compute_advective_update(q,VEC(i,j,k),idx,new_state,new_stag_state,dt,dtfact,dev_coords);
                }) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_curvature_update")
                , curvature_policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    z4c_eq_system.compute_curvature_update(q,VEC(i,j,k),idx,new_state,new_stag_state,dt,dtfact,dev_coords);
                }) ;
    //**************************************************************************************************/
    #elif defined(GRACE_ENABLE_BSSN_METRIC)
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& aux     = grace::variable_list::get().getaux()     ; 
    //**************************************************************************************************/
    bssn_system_t bssn_eq_system(old_state,aux,old_stag_state) ; 
    //**************************************************************************************************/
    auto advance_policy = 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
        ) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL","bssn_update")
                , advance_policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) 
                {
                    bssn_eq_system.compute_update(q,VEC(i,j,k),idx,new_state,new_stag_state,dt,dtfact);
                }) ; 
    //**************************************************************************************************/
    #endif 
}


// new_state = old_state + dt * dtfact * G(new_state)
template< typename eos_t >
void advance_implicit_substep( double const t, double const dt, double const dtfact 
                    , var_array_t& new_state 
                    , var_array_t& old_state 
                    , staggered_variable_arrays_t & new_stag_state 
                    , staggered_variable_arrays_t & old_stag_state )
{/*to do*/

    DECLARE_GRID_EXTENTS ; 

    using namespace grace ; 
    using namespace Kokkos ; 

    Kokkos::deep_copy(new_state,old_state) ; 
    deep_copy(new_stag_state,old_stag_state) ; 

    auto& _idx = variable_list::get().getinvspacings() ; 
    auto& aux  = variable_list::get().getaux() ; 

    #ifdef GRACE_ENABLE_M1 
    auto policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,0),0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}
        ) ;
    m1_equations_system_t m1_eq_system(old_state,old_stag_state,aux) ; 
    parallel_for(
          GRACE_EXECUTION_TAG("evol", "m1_implicit_sources")
        , policy
        , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
            m1_eq_system.compute_implicit_update<0>(
                q, VEC(i,j,k), _idx, new_state, dt, dtfact
            );
            #ifdef M1_NU_THREESPECIES
            m1_eq_system.compute_implicit_update<1>(
                q, VEC(i,j,k), _idx, new_state, dt, dtfact
            );
            m1_eq_system.compute_implicit_update<2>(
                q, VEC(i,j,k), _idx, new_state, dt, dtfact
            );
            #endif 
        }
    ) ; 
    #endif
    Kokkos::fence() ; 
}

template< typename eos_t >
void advance_substep( double const t, double const dt, double const dtfact 
                    , var_array_t& new_state 
                    , var_array_t& old_state 
                    , staggered_variable_arrays_t & new_stag_state 
                    , staggered_variable_arrays_t & old_stag_state )
{
    GRACE_PROFILING_PUSH_REGION("evol") ;
    using namespace grace ; 
    using namespace Kokkos  ; 

    //**************************************************************************************************/
    compute_fluxes<eos_t>(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ; 
    //**************************************************************************************************/    
    auto flux_context = reflux_fill_flux_buffers() ;
    //**************************************************************************************************/
    compute_emfs(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ; 
    //**************************************************************************************************/
    auto emf_context = reflux_fill_emf_buffers() ; 
    //**************************************************************************************************/
    add_fluxes_and_source_terms<eos_t>(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ;
    //**************************************************************************************************/
    reflux_correct_emfs(emf_context) ;
    //**************************************************************************************************/
    update_CT(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ;
    //**************************************************************************************************/
    update_fd(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ; 
    //**************************************************************************************************/
    reflux_correct_fluxes(flux_context,t,dt,dtfact,new_state) ;
    //**************************************************************************************************/
    parallel::mpi_barrier() ;  
    Kokkos::fence() ; 
    GRACE_PROFILING_POP_REGION ; 
}

#ifdef GRACE_ENABLE_Z4C_METRIC
void compute_constraint_violations() {
    using namespace grace ; 
    using namespace Kokkos  ; 
    DECLARE_GRID_EXTENTS ;
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& aux     = grace::variable_list::get().getaux()     ; 
    auto& state   = grace::variable_list::get().getstate()   ; 
    auto dev_coords = grace::coordinate_system::get().get_device_coord_system() ; 
    auto dummy = grace::variable_list::get().getstaggeredstate() ; 
    //**************************************************************************************************/
    z4c_system_t z4c_eq_system(state,aux,dummy) ; 
    //**************************************************************************************************/
    auto policy = 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_compute_constraint_violations")
                , policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) 
                {
                    z4c_eq_system(auxiliaries_computation_kernel_t{}, VEC(i,j,k), q, idx, dev_coords);
                }) ; 
    //**************************************************************************************************/
}

void enforce_algebraic_constraints(grace::var_array_t& state) {
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ;
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& aux     = grace::variable_list::get().getaux()     ; 
    auto dev_coords = grace::coordinate_system::get().get_device_coord_system() ; 
    auto dummy = grace::variable_list::get().getstaggeredstate() ; 
    //**************************************************************************************************/
    z4c_system_t z4c_eq_system(state,aux,dummy) ; 
    //**************************************************************************************************/
    auto policy = 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,0),0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}
        ) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_update")
                , policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) 
                {
                    z4c_eq_system.impose_algebraic_constraints(state,i,j,k,q) ; 
                }) ; 
    //**************************************************************************************************/
}; 
#endif 


// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)                                     \
template                                                              \
void advance_substep<EOS>( double const , double const , double const \
                         , grace::var_array_t&                        \
                         , grace::var_array_t&                        \
                         , grace::staggered_variable_arrays_t &       \
                         , grace::staggered_variable_arrays_t &       \
                        ) ;                                           \
template                                                              \
void advance_implicit_substep<EOS>(                                   \
                           double const , double const , double const \
                         , grace::var_array_t&                        \
                         , grace::var_array_t&                        \
                         , grace::staggered_variable_arrays_t &       \
                         , grace::staggered_variable_arrays_t &       \
                        ) ;                                           \
template                                                              \
void compute_fluxes<EOS>( double const , double const , double const \
                        , grace::var_array_t&                        \
                        , grace::var_array_t&                        \
                        , grace::staggered_variable_arrays_t &       \
                        , grace::staggered_variable_arrays_t &       \
                        ) ;                                          \
template                                                             \
void add_fluxes_and_source_terms<EOS>( double const , double const , double const \
                        , grace::var_array_t&                        \
                        , grace::var_array_t&                        \
                        , grace::staggered_variable_arrays_t &       \
                        , grace::staggered_variable_arrays_t &       \
                        ) ;                                          \
template                                                              \
void evolve_impl<EOS>()

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE
}
