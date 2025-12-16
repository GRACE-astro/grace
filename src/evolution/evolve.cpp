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
#include <grace/utils/advanced_riemann_solvers.hh>
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

    auto& sstate = grace::variable_list::get().getstaggeredstate() ; 
    auto& sstate_p = grace::variable_list::get().getstaggeredscratch() ; 

    auto& aux     = grace::variable_list::get().getaux()     ; 

    auto& cvol    = grace::variable_list::get().getvolumes() ; 
    auto& fsurf   = grace::variable_list::get().getstaggeredcoords() ;
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& dx     = grace::variable_list::get().getspacings() ;  
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ; 
    auto& emf  = grace::variable_list::get().getemfarray() ; 
    auto& vbar  = grace::variable_list::get().getvbararray() ;

    auto nvars_face  = sstate.face_staggered_fields_x.extent(GRACE_NSPACEDIM) ;
    auto nvars_cc  = state.extent(GRACE_NSPACEDIM) ;

    /* Copy the current state to scratch memory */
    //amr::apply_boundary_conditions(state) ; 
    Kokkos::deep_copy(state_p, state) ; 
    grace::deep_copy(sstate_p, sstate) ; 
    if ( tstepper == "euler" ) {
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ; 
        amr::apply_boundary_conditions(state, sstate) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
    } else if (tstepper == "rk2" ) {
        /* Compute auxiliaries at current timelevel */
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,0.5,state_p,state,sstate_p,sstate) ; 
        amr::apply_boundary_conditions(state_p,sstate_p) ; 
        compute_auxiliary_quantities<eos_t>(state_p, sstate_p, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ;
        amr::apply_boundary_conditions(state,sstate) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
    } else if (tstepper == "rk3" ) {
        // step 1: state_p -> u^1 = u^n + dt L( u^n )
        advance_substep<eos_t>(
            t,dt,1.0,
            state_p,state,
            sstate_p,sstate) ; 
        amr::apply_boundary_conditions(state_p,sstate_p) ; 
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
            state_pp,state_p,
            sstate_pp,sstate_p) ;
        amr::apply_boundary_conditions(state_pp,sstate_pp) ; 
        compute_auxiliary_quantities<eos_t>(state_pp, sstate_pp, aux) ;
        // step 4: state = 1/3 u^n + 2/3 u^2
        linop_apply(state,state,state_pp,
                    sstate,sstate,sstate_pp, 1./3, 2./3.) ; 
        // step 5: state -> u^n+1 = 1/3 u^n + 2/3 u^2 + 2/3 dt L( u^2 )
        advance_substep<eos_t>(
            t,dt,2./3.,
            state,state_pp,
            sstate,sstate_pp) ;
        amr::apply_boundary_conditions(state,sstate) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
    } else if (tstepper == "imex1") { 
        advance_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ;
        amr::apply_boundary_conditions(state,sstate) ; 
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
        amr::apply_boundary_conditions(state_p,sstate_p) ;
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
        amr::apply_boundary_conditions(state_p,sstate_p) ; 
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
        amr::apply_boundary_conditions(state,sstate) ;
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ; 
        /* done */
    } else if (tstepper == "imex232" ) { 
        ERROR("Imex 3 not implemented yet") ; 
    } else {
        ERROR("Unrecognised time-stepper.") ; 
    }
    Kokkos::deep_copy(state_p,state) ; 
    grace::deep_copy(sstate_p,sstate) ; 
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
    auto& vbar  = grace::variable_list::get().getvbararray() ;
    //**************************************************************************************************/
    // construct grmhd object 
    using recon_t = weno_reconstructor_t<5> ; 
    auto atmo_params = get_atmo_params();
    auto excision_params = get_excision_params() ; 
    auto eos = eos::get().get_eos<eos_t>() ;  
    grmhd_equations_system_t<eos_t>
        grmhd_eq_system(eos,old_state,old_stag_state,aux,atmo_params,excision_params) ; 
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
        }
    ) ; 
    #endif 
    //**************************************************************************************************/
    // loop ranges 
    auto flux_x_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,0,0),0}
            , {VEC(nx+ngz+1,ny+2*ngz,nz+2*ngz),nq}
        ) ;
    auto flux_y_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,ngz,0),0}
            , {VEC(nx+2*ngz,ny+ngz+1,nz+2*ngz),nq}
        ) ;
    auto flux_z_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,ngz),0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+ngz+1),nq}
        ) ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    // compute x flux 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_x_flux")
                , flux_x_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        #ifndef GRACE_FREEZE_HYDRO
        grmhd_eq_system.template compute_x_flux<hll_riemann_solver_t,recon_t>(q, VEC(i,j,k), fluxes, vbar, dx, dt, dtfact) ;
        #endif 
        #ifdef GRACE_ENABLE_M1
        m1_eq_system.template compute_x_flux<hll_riemann_solver_t, slope_limited_reconstructor_t<MCbeta>>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        ) ; 
        #endif 
    }) ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_y_flux")
                , flux_y_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        #ifndef GRACE_FREEZE_HYDRO
        grmhd_eq_system.template compute_y_flux<hll_riemann_solver_t,recon_t>(q, VEC(i,j,k), fluxes, vbar, dx, dt, dtfact);
        #endif 
        #ifdef GRACE_ENABLE_M1
        m1_eq_system.template compute_y_flux<hll_riemann_solver_t, slope_limited_reconstructor_t<MCbeta>>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        ) ; 
        #endif
    }) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_z_flux")
                , flux_z_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        #ifndef GRACE_FREEZE_HYDRO
        grmhd_eq_system.template compute_z_flux<hll_riemann_solver_t,recon_t>(q, VEC(i,j,k), fluxes, vbar, dx, dt, dtfact);
        #endif 
        #ifdef GRACE_ENABLE_M1
        m1_eq_system.template compute_z_flux<hll_riemann_solver_t, slope_limited_reconstructor_t<MCbeta>>(
            q, VEC(i,j,k), fluxes, vbar, dx, t, dtfact
        ) ; 
        #endif
    }) ; 
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
        }
    ) ; 
    #endif 
}

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
    using recon_t = weno_reconstructor_t<5> ; 
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
                                vidx, \
                                q     ) ; \
    reconstructor(sview,i,j,k,uL,uR,dir) ; \
    } while(false)
    #define RECONSTRUCT_V(vview,jdir,vidx,q,i,j,k,uL,uR,dir) \
    do { \
    auto sview = subview(vview, \
                                ALL(), \
                                ALL(), \
                                ALL(), \
                                vidx, \
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
    using recon_t = weno_reconstructor_t<5> ; 
    auto atmo_params = get_atmo_params();
    auto excision_params = get_excision_params() ; 
    auto eos = eos::get().get_eos<eos_t>() ;  
    grmhd_equations_system_t<eos_t>
        grmhd_eq_system(eos,old_state,old_stag_state,aux,atmo_params,excision_params) ;
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
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_sources")
                , policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        #ifndef GRACE_FREEZE_HYDRO
        grmhd_eq_system(sources_computation_kernel_t{}, q, VEC(i,j,k), idx, new_state, dt, dtfact );
        #endif 
        #ifdef GRACE_ENABLE_M1
        //m1_eq_system(sources_computation_kernel_t{}, q, VEC(i,j,k), idx, new_state, dt, dtfact );
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
    using recon_t = weno_reconstructor_t<5> ; 
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
    #ifdef GRACE_ENABLE_BSSN_METRIC
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& dx     = grace::variable_list::get().getspacings() ;  
    auto& aux     = grace::variable_list::get().getaux()     ; 
    //**************************************************************************************************/
    auto k1 = grace::get_param<double>("bssn","k1") ;
    auto eta = grace::get_param<double>("bssn","eta") ;
    auto epsdiss = grace::get_param<double>("bssn","epsdiss") ; 
    bssn_system_t bssn_eq_system(old_state,aux,old_stag_state,k1,eta,epsdiss) ; 
    //**************************************************************************************************/
    auto advance_bssn_policy = 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
        ) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL","BSSN_update")
                , advance_bssn_policy
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
            m1_eq_system.compute_implicit_update(
                q, VEC(i,j,k), _idx, new_state, dt, dtfact
            );
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
#undef INSTANTIATE_TEMPLATE
}