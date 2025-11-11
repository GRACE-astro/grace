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
#include <grace/amr/amr_ghosts.hh>

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
        
        auto update_policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+2>> (
               {VEC(0,0,0), 0, 0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nvars_cc, nq}
        ) ;
        auto const stag_update = [&] (
            grace::staggered_variable_arrays_t& A,
            grace::staggered_variable_arrays_t& B,
            grace::staggered_variable_arrays_t& C, double b, double c
        ) {
            auto staggered_update_policy_x =
            Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+2>> (
                {VEC(0,0,0), 0, 0}
                , {VEC(nx+2*ngz+1,ny+2*ngz,nz+2*ngz),nvars_face, nq}
            ) ;
            Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
                , staggered_update_policy_x
                , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
                {
                    A.face_staggered_fields_x(VEC(i,j,k), ivar, q)
                        = b * B.face_staggered_fields_x(VEC(i,j,k), ivar, q)
                        + c * C.face_staggered_fields_x(VEC(i,j,k), ivar, q) ; 
                }
            ) ;
            auto staggered_update_policy_y =
            Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+2>> (
                {VEC(0,0,0), 0, 0}
                , {VEC(nx+2*ngz,ny+2*ngz+1,nz+2*ngz),nvars_face, nq}
            ) ;
            Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
                , staggered_update_policy_y
                , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
                {
                    A.face_staggered_fields_y(VEC(i,j,k), ivar, q)
                        = b * B.face_staggered_fields_y(VEC(i,j,k), ivar, q)
                        + c * C.face_staggered_fields_y(VEC(i,j,k), ivar, q) ; 
                }
            ) ;
            auto staggered_update_policy_z =
            Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+2>> (
                {VEC(0,0,0), 0, 0}
                , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz+1),nvars_face, nq}
            ) ;
            Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
                , staggered_update_policy_z
                , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
                {
                    A.face_staggered_fields_z(VEC(i,j,k), ivar, q)
                        = b * B.face_staggered_fields_z(VEC(i,j,k), ivar, q)
                        + c * C.face_staggered_fields_z(VEC(i,j,k), ivar, q) ; 
                }
            ) ;
        } ; 
        // step 1: state_p -> u^1 = u^n + dt L( u^n )
        advance_substep<eos_t>(
            t,dt,1.0,
            state_p,state,
            sstate_p,sstate) ; 
        amr::apply_boundary_conditions(state_p,sstate_p) ; 
        compute_auxiliary_quantities<eos_t>(state_p, sstate_p, aux) ;
        // Allocate state_pp and sstate_pp 
        auto state_pp  = grace::variable_list::get().allocate_state() ;
        auto sstate_pp = grace::variable_list::get().allocate_staggered_state() ;
        // step 2: state_pp = 3/4 u^n + 1/4 u^1
        stag_update(sstate_pp, sstate, sstate_p, 0.75, 0.25) ; 
        Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
            , update_policy
            , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
            {
                state_pp(VEC(i,j,k), ivar, q)
                    = 0.75 * state(VEC(i,j,k), ivar, q)
                    + 0.25 * state_p(VEC(i,j,k), ivar, q) ; 
            }
        ) ;
        // step 3: state_pp -> u^2 = 3/4 u^n + 1/4 u^1 + 1/4 dt L( u^1 )
        advance_substep<eos_t>(
            t,dt,0.25,
            state_pp,state_p,
            sstate_pp,sstate_p) ;
        amr::apply_boundary_conditions(state_pp,sstate_pp) ; 
        compute_auxiliary_quantities<eos_t>(state_pp, sstate_pp, aux) ;
        // step 4: state = 1/3 u^n + 2/3 u^2
        stag_update(sstate, sstate, sstate_pp, 1./3, 2./3.) ; 
        Kokkos::parallel_for(
            GRACE_EXECUTION_TAG("EVOL","RK3_substep")
            , update_policy
            , KOKKOS_LAMBDA (VEC(int i, int j, int k), int ivar, int q)
            {
                state(VEC(i,j,k), ivar, q)
                    = 1./3. * state(VEC(i,j,k), ivar, q)
                    + 2./3. * state_pp(VEC(i,j,k), ivar, q) ; 
            }
        ) ;
        // step 5: state -> u^n+1 = 1/3 u^n + 2/3 u^2 + 2/3 dt L( u^2 )
        advance_substep<eos_t>(
            t,dt,2./3.,
            state,state_pp,
            sstate,sstate_pp) ;
        amr::apply_boundary_conditions(state,sstate) ; 
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
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
    // loop ranges 
    auto flux_x_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,0,0),0}
            , {VEC(nx+ngz+1,ny+2*ngz,nz+2*ngz),nq}
        ) ;
    auto flux_y_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,ngz,0),0}
            , {VEC(nx+ngz,ny+2*ngz+1,nz+2*ngz),nq}
        ) ;
    auto flux_z_policy = 
        MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(0,0,ngz),0}
            , {VEC(nx+ngz,ny+2*ngz,nz+2*ngz+1),nq}
        ) ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    // compute x flux 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_x_flux")
                , flux_x_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        grmhd_eq_system.template compute_x_flux<hll_riemann_solver_t,recon_t>(q, VEC(i,j,k), fluxes, vbar, dx, dt, dtfact) ;
    }) ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_y_flux")
                , flux_y_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        grmhd_eq_system.template compute_y_flux<hll_riemann_solver_t,recon_t>(q, VEC(i,j,k), fluxes, vbar, dx, dt, dtfact);
    }) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "compute_z_flux")
                , flux_z_policy 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        grmhd_eq_system.template compute_z_flux<hll_riemann_solver_t,recon_t>(q, VEC(i,j,k), fluxes, vbar, dx, dt, dtfact);
    }) ; 
    //**************************************************************************************************/
    Kokkos::fence() ; 
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
        grmhd_eq_system(sources_computation_kernel_t{}, q, VEC(i,j,k), idx, new_state, dt, dtfact );
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
    auto& vbar  = grace::variable_list::get().getvbararray() ;
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
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
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
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
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
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
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

parallel::grace_transfer_context_t reflux_fill_flux_buffers() 
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ; 
    int nvars_hrsc = variables::get_n_hrsc() ;
    //**************************************************************************************************/
    // and some more 
    auto& ghost_layer = grace::amr_ghosts::get();
    auto sbuf = ghost_layer.get_reflux_send_buffer() ; 
    auto rbuf = ghost_layer.get_reflux_recv_buffer() ; 
    auto desc = ghost_layer.get_reflux_face_send_list() ; 
    //**************************************************************************************************/
    View<hanging_remote_reflux_desc_t*> info ; 
    grace::deep_copy_vec_to_const_view(info,desc) ; 
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<4>> (
            {0
            ,0
            ,0
            ,0},
            {static_cast<long>(nx/2)
            ,static_cast<long>(nx/2)
            ,static_cast<long>(nvars_hrsc)
            ,static_cast<long>(desc.size())}
        ) ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_flux_buffers")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& ivar, int const& iq) {
                auto const dsc = info(iq) ; 

                auto const iface = dsc.elem_id ; 
                auto const rank = dsc.rank ; // could be same as ours! 
                auto const idir = iface / 2; 
                auto const iside = iface%2 ; 
                auto const qid = dsc.qid ; 

                size_t ijk_s[3] ; 
                ijk_s[idir] = iside ? nx + ngz : ngz ; 
                ijk_s[other_dirs[idir][0]] = ngz + 2*i ; 
                ijk_s[other_dirs[idir][1]] = ngz + 2*j ; 
                
                double flux = 0 ; 
                for( int ii=0; ii<=(idir!=0); ++ii) {
                    for( int jj=0; jj<=(idir!=1); ++jj) {
                        for( int kk=0; kk<=(idir!=2); ++kk) {
                            flux += fluxes(ijk_s[0]+ii, ijk_s[1]+jj, ijk_s[2]+kk, ivar, idir, qid) ; 
                        }
                    }
                }
                
                auto bid = dsc.buf_id ; 
                sbuf(i,j,ivar,bid,rank) = 0.25*flux ; 
            }
        ) ; 
    /* now we send and receive */
    auto soffsets = ghost_layer.get_reflux_buffer_rank_send_offsets() ; 
    auto ssizes = ghost_layer.get_reflux_buffer_rank_send_sizes() ;
    
    auto roffsets = ghost_layer.get_reflux_buffer_rank_recv_offsets() ; 
    auto rsizes = ghost_layer.get_reflux_buffer_rank_recv_sizes() ;

    parallel::grace_transfer_context_t context ; 
    auto nprocs = parallel::mpi_comm_size() ;
    auto proc = parallel::mpi_comm_rank() ; 
    for( int iproc=0; iproc<nprocs; ++iproc) {
        if ( iproc == proc ) continue ; 
        // send 
        if ( ssizes[iproc] > 0 ) {
            context._send_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_isend(
                sbuf.data() + soffsets[iproc],
                ssizes[iproc],
                iproc,
                0,
                MPI_COMM_WORLD,
                &context._send_requests.back()
            );  
        }
        if ( rsizes[iproc] > 0 ) {
            context._recv_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_irecv(
                rbuf.data() + roffsets[iproc],
                rsizes[iproc],
                iproc,
                0,
                MPI_COMM_WORLD,
                &context._recv_requests.back()
            );  
        }
    }

    return context ;
}

parallel::grace_transfer_context_t reflux_fill_emf_buffers() 
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& emf  = grace::variable_list::get().getemfarray() ; 
    int nvars_hrsc = variables::get_n_hrsc() ;
    //**************************************************************************************************/
    // and some more 
    auto& ghost_layer = grace::amr_ghosts::get();
    auto sbuf = ghost_layer.get_reflux_emf_send_buffer() ; 
    auto rbuf = ghost_layer.get_reflux_emf_recv_buffer() ; 
    auto desc = ghost_layer.get_reflux_face_send_list() ; 
    //**************************************************************************************************/
    View<hanging_remote_reflux_desc_t*> info ; 
    grace::deep_copy_vec_to_const_view(info,desc) ; 
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx/2+1)
            ,static_cast<long>(nx/2+1)
            ,static_cast<long>(desc.size())}
        ) ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_emf_buffers")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                auto const dsc = info(iq) ; 
                 
                auto const iface = dsc.elem_id ; 
                auto const rank = dsc.rank ; // could be same as ours! 

                auto const fdir = iface / 2;
                auto const idir = other_dirs[iface / 2][0]; 
                auto const jdir = other_dirs[iface / 2][1]; 
                auto const iside = iface % 2 ;

                auto const qid = dsc.qid ; 

                size_t ijk_s[3] ; 
                ijk_s[fdir] = iside ? nx + ngz : ngz ; 
                ijk_s[idir] = 2*i + ngz ; 
                ijk_s[idir] = 2*i + ngz ; 
                // note that the range is n+1 in both dirs
                // for each, one iteration is out of bounds.
                // however the arrays have padding of ngz and the garbage
                // value will be unused
                double emf_i = 0.5 * (
                    emf(ijk_s[0],ijk_s[1],ijk_s[2],idir,qid) + 
                    emf(ijk_s[0] + (idir==0),ijk_s[1] + (idir==1),ijk_s[2] + (idir==2),idir,qid)
                );
                double emf_j = 0.5 * (
                    emf(ijk_s[0],ijk_s[1],ijk_s[2],jdir,qid) + 
                    emf(ijk_s[0] + (jdir==0),ijk_s[1] + (jdir==1),ijk_s[2] + (jdir==2),jdir,qid)
                );
                
                auto bid = dsc.buf_id ; 
                sbuf(i,j,0,bid,rank) = emf_i ; 
                sbuf(i,j,1,bid,rank) = emf_j ; 
            }
        ) ; 
    // send - receive face buffers
    auto soffsets = ghost_layer.get_reflux_buffer_rank_send_emf_offsets() ; 
    auto ssizes = ghost_layer.get_reflux_buffer_rank_send_emf_sizes() ;
    
    auto roffsets = ghost_layer.get_reflux_buffer_rank_recv_emf_offsets() ; 
    auto rsizes = ghost_layer.get_reflux_buffer_rank_recv_emf_sizes() ;

    parallel::grace_transfer_context_t context ; 
    auto nprocs = parallel::mpi_comm_size() ;
    auto proc = parallel::mpi_comm_rank() ; 
    for( int iproc=0; iproc<nprocs; ++iproc) {
        if ( ssizes[iproc] > 0 ) {
            context._send_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_isend(
                sbuf.data() + soffsets[iproc],
                ssizes[iproc],
                iproc,
                0,
                MPI_COMM_WORLD,
                &context._send_requests.back()
            );  
        }
        if ( rsizes[iproc] > 0 ) {
            context._recv_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_irecv(
                rbuf.data() + roffsets[iproc],
                rsizes[iproc],
                iproc,
                0,
                MPI_COMM_WORLD,
                &context._recv_requests.back()
            );  
        }
    }
    //**************************************************************************************************/
    auto sbuf_edge = ghost_layer.get_reflux_emf_edge_send_buffer() ; 
    auto rbuf_edge = ghost_layer.get_reflux_emf_edge_recv_buffer() ; 
    auto desc_edge = ghost_layer.get_reflux_edge_send_list() ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    View<hanging_remote_reflux_desc_t*> info_edge ; 
    grace::deep_copy_vec_to_const_view(info_edge,desc_edge) ; 
    //**************************************************************************************************/
    //**************************************************************************************************/
    auto edge_policy = 
        MDRangePolicy<Rank<2>> (
            {0,0},
            {static_cast<long>(nx),static_cast<long>(desc_edge.size())}
        ) ; 
    // fill edge buffers 
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_fill_emf_edge_buffers")
            , edge_policy 
            , KOKKOS_LAMBDA (int const& i, int const& iq) {
                auto const dsc = info_edge(iq) ; 
                
                auto const iedge = dsc.elem_id ; 
                int idir = (iedge/4) ; 
                int jside = (iedge>>0)&1 ; 
                int kside = (iedge>>1)&1 ;
                int jdir = other_dirs[idir][0] ; 
                int kdir = other_dirs[idir][1] ; 

                auto const qid = dsc.qid ; 

                size_t ijk_s[3] ;
                ijk_s[idir] = ngz + i ; 
                ijk_s[jdir] = jside ? nx + ngz : ngz ; 
                ijk_s[kdir] = kside ? nx + ngz : ngz ; 

                auto const rank = dsc.rank ; // could be same as ours!
                auto bid = dsc.buf_id ;
                sbuf_edge(i, bid, rank) = emf(ijk_s[0],ijk_s[1],ijk_s[2],idir,qid) ; 
            }
        ) ;
        // todo maybe edge bufs can be separate, this seems wasteful 
    // send - receive edge buffers 
    auto soffsets_edge = ghost_layer.get_reflux_buffer_rank_send_emf_edge_offsets() ; 
    auto ssizes_edge   = ghost_layer.get_reflux_buffer_rank_send_emf_edge_sizes()   ;
    
    auto roffsets_edge = ghost_layer.get_reflux_buffer_rank_recv_emf_edge_offsets() ; 
    auto rsizes_edge   = ghost_layer.get_reflux_buffer_rank_recv_emf_sizes()   ;
    for( int iproc=0; iproc<nprocs; ++iproc) {
        if ( ssizes_edge[iproc] > 0 ) {
            context._send_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_isend(
                sbuf_edge.data() + soffsets_edge[iproc],
                ssizes_edge[iproc],
                iproc,
                0,
                MPI_COMM_WORLD,
                &context._send_requests.back()
            );  
        }
        if ( rsizes_edge[iproc] > 0 ) {
            context._recv_requests.push_back(MPI_Request{}) ; 
            parallel::mpi_irecv(
                rbuf_edge.data() + roffsets_edge[iproc],
                rsizes_edge[iproc],
                iproc,
                0,
                MPI_COMM_WORLD,
                &context._recv_requests.back()
            );  
        }
    }
    return context ; 
}

void reflux_correct_fluxes(
    parallel::grace_transfer_context_t& context,
    double t, double dt, double dtfact,
    var_array_t & new_state 
)
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ; 
    int nvars_hrsc = variables::get_n_hrsc() ;
    //**************************************************************************************************/
    auto& ghost_layer = grace::amr_ghosts::get();
    auto rbuf = ghost_layer.get_reflux_recv_buffer() ; 
    auto desc = ghost_layer.get_reflux_face_descriptors() ; 
    //**************************************************************************************************/
    View<hanging_face_reflux_desc_t*> info ; 
    grace::deep_copy_vec_to_const_view(info,desc) ; 
    //**************************************************************************************************/
    parallel::mpi_waitall(context) ; 
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<4>> (
            {0,0,0,0},
            {static_cast<long>(nx/2)
            ,static_cast<long>(nx/2)
            ,static_cast<long>(nvars_hrsc)
            ,static_cast<long>(desc.size())}
        ) ;
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_apply")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& ivar, int const& iq) {
                auto const dsc = info(iq) ; 

                auto const qid_c  = dsc.coarse_qid     ; 
                auto const iface_c = dsc.coarse_face_id ; 

                auto const idir = iface_c / 2; 
                auto const side = iface_c % 2;

                size_t ijk_fs[3], ijk_cc[3] ; 
                ijk_fs[idir] = (iface_c % 2)    
                            ? ngz + nx 
                            : ngz ;
                // cell center index 
                ijk_cc[idir] = (iface_c % 2)    
                            ? ngz + nx - 1 
                            : ngz ;
                ijk_fs[other_dirs[idir][0]] = ijk_cc[other_dirs[idir][0]] = ngz + i ; 
                ijk_fs[other_dirs[idir][1]] = ijk_cc[other_dirs[idir][1]] = ngz + j ; 

                // compute child id 
                int8_t ichild = (2*i>=nx) + 2 * (2*j>=nx) ; 

                double flux_correction = 0 ; 
                if ( dsc.fine_is_remote[ichild] ) {
                    flux_correction = rbuf(i%(nx/2),j%(nx/2),ivar,dsc.fine_qid[ichild],dsc.fine_owner_rank[ichild]) ; 
                } else {
                    // compute flux correction 
                    size_t qid_f = dsc.fine_qid[ichild] ; 
                    size_t ijk_f[3] ; 
                    // on fine side the side is opposite 
                    ijk_f[idir] = (iface_c % 2)    
                                ? ngz  
                                : ngz + nx ;
                    ijk_f[other_dirs[idir][0]] = ngz + (2*i%nx) ; 
                    ijk_f[other_dirs[idir][1]] = ngz + (2*j%nx) ; 

                    for( int ii=0; ii<=(idir!=0); ++ii) {
                        for( int jj=0; jj<=(idir!=1); ++jj) {
                            for( int kk=0; kk<=(idir!=2); ++kk) {
                                flux_correction += fluxes(ijk_f[0]+ii, ijk_f[1]+jj, ijk_f[2]+kk, ivar, idir, qid_f) ; 
                            }
                        }
                    }
                    flux_correction *= 0.25 ; 
                }
                int sign = side ? -1 : +1 ; 
                new_state(ijk_cc[0],ijk_cc[1],ijk_cc[2],ivar,qid_c) += sign * dt * dtfact * idx(idir,qid_c) * (
                    flux_correction - fluxes(ijk_fs[0],ijk_fs[1],ijk_fs[2],ivar,idir,qid_c)
                ) ; 
            }
        ) ;
    //**************************************************************************************************/
    //**************************************************************************************************/
    //**************************************************************************************************/ 

}

void reflux_correct_emfs(
    parallel::grace_transfer_context_t& context,
    double t, double dt, double dtfact,
    staggered_variable_arrays_t& new_stag_state
)
{
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ; 
    //**************************************************************************************************/
    // fetch some stuff 
    auto& idx     = grace::variable_list::get().getinvspacings() ;  
    auto& emf  = grace::variable_list::get().getemfarray() ;  
    int nvars_hrsc = variables::get_n_hrsc() ;
    //**************************************************************************************************/
    auto& ghost_layer = grace::amr_ghosts::get();
    auto rbuf = ghost_layer.get_reflux_emf_recv_buffer() ; 
    auto desc = ghost_layer.get_reflux_face_descriptors() ; 
    //**************************************************************************************************/
    View<hanging_face_reflux_desc_t*> info ; 
    grace::deep_copy_vec_to_const_view(info,desc) ; 
    //**************************************************************************************************/
    parallel::mpi_waitall(context) ;
    //**************************************************************************************************/
    auto policy = 
        MDRangePolicy<Rank<3>> (
            {0,0,0},
            {static_cast<long>(nx+1)
            ,static_cast<long>(nx+1)
            ,static_cast<long>(desc.size())}
        ) ;
    //**************************************************************************************************/
    constexpr std::array<std::array<int,2>,3> other_dirs = {{
        {{1,2}}, {{0,2}}, {{0,1}}
    }} ; 
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_face")
            , policy 
            , KOKKOS_LAMBDA (VECD(int const& i, int const& j), int const& iq) {
                auto const dsc = info(iq) ; 

                auto const iface_c = dsc.coarse_face_id ; 
                auto const fdir = iface_c / 2 ; 
                auto const idir = other_dirs[fdir][0]; 
                auto const jdir = other_dirs[fdir][1]; 
                auto const iside = iface_c % 2 ;
                auto const qid_c = dsc.coarse_qid ; 

                // indices of face center (coarse)
                // edge center (coarse)
                // edge center (fine)
                size_t ijk_c[3], ijk_f[3] ; 

                int ichild = (2*i>=nx) + 2*(2*j>=nx) ;
                // upper child indices need to be shifted
                // down by half nx for accessing the receive 
                // buffer 
                int off_i = (2*i>=nx) ? -nx/2 : 0 ;
                int off_j = (2*j>=nx) ? -nx/2 : 0 ; 
                // fine qid 
                auto const qid_f = dsc.fine_qid[ichild] ; 
                // emf correction 
                double emf_corr_i{0}, emf_corr_j{0} ; 
                if ( dsc.fine_is_remote[ichild] ) {
                    auto rank = dsc.fine_owner_rank[ichild] ; 
                    emf_corr_i = rbuf(i-off_i,j-off_j,0,qid_f,rank) ; 
                    emf_corr_j = rbuf(i-off_i,j-off_j,1,qid_f,rank) ; 
                } else {
                    // fine side so iside is opposite
                    ijk_f[fdir] = iside ? ngz : nx + ngz ; 
                    ijk_f[idir] = (2*i)%nx + ngz ; 
                    ijk_f[jdir] = (2*j)%nx + ngz ; 
                    emf_corr_i = 0.5 * (
                        emf(ijk_f[0],ijk_f[1],ijk_f[2],idir,qid_f) + 
                        emf(ijk_f[0] + (idir==0),ijk_f[1] + (idir==1),ijk_f[2] + (idir==2),idir,qid_f)
                    );
                    emf_corr_j = 0.5 * (
                        emf(ijk_f[0],ijk_f[1],ijk_f[2],jdir,qid_f) + 
                        emf(ijk_f[0] + (jdir==0),ijk_f[1] + (jdir==1),ijk_f[2] + (jdir==2),jdir,qid_f)
                    ) ; 
                }
                
                // face indices --> cells to be corrected 
                // coarse side so iside is correct
                ijk_c[fdir] = iside ? nx + ngz : ngz ; 
                ijk_c[idir] = i + ngz ; 
                ijk_c[jdir] = j + ngz ; 

                emf_corr_i -= emf(ijk_c[0], ijk_c[1], ijk_c[2], idir, qid_c) ; 
                emf_corr_j -= emf(ijk_c[0], ijk_c[1], ijk_c[2], jdir, qid_c) ;

                const var_array_t* views[3] = {
                    &new_stag_state.face_staggered_fields_x,
                    &new_stag_state.face_staggered_fields_y,
                    &new_stag_state.face_staggered_fields_z
                };
                
                // add emf_i to the cell shifted 
                // down by one in j
                if ( j>0 ) {
                    int signs[3] = {+1,-1,+1} ; 
                    (*views[fdir])(ijk_c[0]-(jdir==0),ijk_c[1]-(jdir==1),ijk_c[2]-(jdir==2),0,qid_c) += 
                        signs[fdir] * dt * dtfact * idx(jdir,qid_c) * emf_corr_i ; 
                }

                if ( j < nx ) {
                    int signs[3] = {+1,-1,+1} ; 
                    (*views[fdir])(ijk_c[0]-(jdir==0),ijk_c[1]-(jdir==1),ijk_c[2]-(jdir==2),0,qid_c) += 
                        - signs[fdir] * dt * dtfact * idx(jdir,qid_c) * emf_corr_i ; 
                }

                if ( i>0 ) {
                    int signs[3] = {-1,+1,-1} ; 
                    (*views[fdir])(ijk_c[0]-(idir==0),ijk_c[1]-(idir==1),ijk_c[2]-(idir==2),0,qid_c) += 
                        signs[fdir] * dt * dtfact * idx(idir,qid_c) * emf_corr_j ; 
                }

                if ( i < nx ) {
                    int signs[3] = {-1,+1,-1} ; 
                    (*views[fdir])(ijk_c[0]-(idir==0),ijk_c[1]-(idir==1),ijk_c[2]-(idir==2),0,qid_c) += 
                        - signs[fdir] * dt * dtfact * idx(idir,qid_c) * emf_corr_j ; 
                }
                

                // moreover we correct the cell at fdir-1 of B^jdir using 
                // emf_i 
                {
                    int signs[3] = {-1,+1,-1}; // Ey in Bz, Ex in Bz, Ex in By 
                    (*views[jdir])(ijk_c[0]-(fdir==0),ijk_c[1]-(fdir==1),ijk_c[2]-(fdir==2),0,qid_c) += 
                        signs[fdir] * dt * dtfact * idx(fdir,qid_c) * emf_corr_i ;
                }
                {
                    int signs[3] = {+1,-1,+1}; // Ez in By, Ez in Bx, Ey in Bx 
                    // and the cell at fdir-1 of B^idir using emf_j 
                    (*views[idir])(ijk_c[0]-(fdir==0),ijk_c[1]-(fdir==1),ijk_c[2]-(fdir==2),0,qid_c) += 
                        signs[fdir] * dt * dtfact * idx(fdir,qid_c) * emf_corr_j ;
                }
                

            }
                
        ) ; 
    
    //**************************************************************************************************/
    auto edge_rbuf = ghost_layer.get_reflux_emf_edge_recv_buffer() ; 
    auto edge_desc = ghost_layer.get_reflux_edge_descriptors() ; 
    //**************************************************************************************************/
    View<hanging_edge_reflux_desc_t*> edge_info ; 
    grace::deep_copy_vec_to_const_view(edge_info,edge_desc) ; 
    //**************************************************************************************************/
    auto edge_policy = 
        MDRangePolicy<Rank<2>> (
            {0,0},
            {static_cast<long>(nx),static_cast<long>(desc.size())}
        ) ;
    //**************************************************************************************************/
    // two phases, first we need to compute the correction, then we apply
    auto emf_edge_correction = ghost_layer.get_reflux_edge_emf_accumulation_buffer() ;  
    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_emf_compute_edge"),
        edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            auto& desc = edge_info(iq) ; 
            auto n_fine = desc.n_fine ; 
            size_t ijk[3] ; 
            for( int iside=0; iside<4; ++iside) {
                auto& side = desc.sides[iside] ; 
                if ( ! side.is_fine ) continue ; 
                auto edge_id = side.edge_id ; 
                int edge_dir = edge_id / 4 ; 
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;
                int ichild = 2*i>=nx ;
                auto qid = side.octants.fine.quad_id[ichild];
                if ( side.octants.fine.is_remote[ichild] ) {
                    auto rank = side.octants.fine.owner_rank[ichild] ; 
                    emf_edge_correction(2*i%nx,ichild,iq) = edge_rbuf((2*i)%nx,qid,rank) ; 
                } else {
                    ijk[edge_dir] = ngz + (2*i)%nx ; 
                    ijk[other_dirs[edge_dir][0]] = side_i ? nx + ngz : ngz ; 
                    ijk[other_dirs[edge_dir][1]] = side_j ? nx + ngz : ngz ; 
                    emf_edge_correction(2*i%nx,ichild,iq) = emf(ijk[0],ijk[1],ijk[2],edge_dir,qid); 
                }
            }
            emf_edge_correction(i,0,iq) /= n_fine ; 
            emf_edge_correction(i,1,iq) /= n_fine ; 
        }
    );
    #if 0
    // apply 
    parallel_for(
        GRACE_EXECUTION_TAG("EVOL", "reflux_emf_apply_edge"),
        edge_policy,
        KOKKOS_LAMBDA (int const& i, int const& iq) {
            const grace::var_array_t* views[3] = {
                &new_stag_state.face_staggered_fields_x,
                &new_stag_state.face_staggered_fields_y,
                &new_stag_state.face_staggered_fields_z
            } ;
            auto& desc = edge_info(iq) ; 
            size_t ijk[3] ; 
            for( int iside=0; iside<4; ++iside) {
                auto& side = desc.sides[iside] ;
                auto edge_id = side.edge_id ;  
                int edge_dir = edge_id / 4 ; 
                int side_i = (edge_id>>0)&1;
                int side_j = (edge_id>>1)&1;
                int ichild = 2*i>=nx ;
                if ( ! side.is_fine ) {
                    if ( side.octants.coarse.is_remote ) continue ;
                    auto qid = side.octants.coarse.quad_id ; 
                    ijk[edge_dir] = ngz + i ; 
                    ijk[other_dirs[edge_dir][0]] = side_i ? nx + ngz : ngz ;  
                    ijk[other_dirs[edge_dir][1]] = side_j ? nx + ngz : ngz ;
                    double emf_correction = 
                        -emf(ijk[0],ijk[1],ijk[2],edge_dir,qid)
                        +0.5*(emf_edge_correction((2*i)%nx,ichild,iq) + emf_edge_correction((2*i)%nx+1,ichild,iq));
                    // 2 B field values need to be corrected 
                    {
                        int signs[3] = {side_j ? -1 : +1, side_j ? +1 : -1, side_j ? -1 : +1 } ; 
                        int off_i = (other_dirs[edge_dir][1] == 0) ? (side_j ? -1 : 0) : 0 ;
                        int off_j = (other_dirs[edge_dir][1] == 1) ? (side_j ? -1 : 0) : 0 ; 
                        int off_k = (other_dirs[edge_dir][1] == 2) ? (side_j ? -1 : 0) : 0 ; 
                        (*views[other_dirs[edge_dir][0]])(ijk[0]+off_i,ijk[1]+off_j,ijk[2]+off_k,0,qid) += 
                            signs[edge_dir] * dt * dtfact * idx(other_dirs[edge_dir][1],qid) * emf_correction ; 
                    }
                    {
                        int signs[3] = {side_i ? +1 : -1, side_i ? -1 : +1, side_i ? +1 : -1} ; 
                        int off_i = (other_dirs[edge_dir][0] == 0) ? (side_i ? -1 : 0) : 0 ;
                        int off_j = (other_dirs[edge_dir][0] == 1) ? (side_i ? -1 : 0) : 0 ; 
                        int off_k = (other_dirs[edge_dir][0] == 2) ? (side_i ? -1 : 0) : 0 ; 
                        (*views[other_dirs[edge_dir][1]])(ijk[0]+off_i,ijk[1]+off_j,ijk[2]+off_k,0,qid) += 
                            signs[edge_dir] * dt * dtfact * idx(other_dirs[edge_dir][0],qid) * emf_correction ;
                    }

                } else {
                    if ( side.octants.fine.is_remote[ichild] ) continue ;
                    auto qid = side.octants.fine.quad_id[ichild] ;
                    ijk[edge_dir] = ngz + i ;  
                    ijk[other_dirs[edge_dir][0]] = side_i ? nx + ngz : ngz ;  
                    ijk[other_dirs[edge_dir][1]] = side_j ? nx + ngz : ngz ;
                    double emf_correction = 
                        -emf(ijk[0],ijk[1],ijk[2],edge_dir,qid)
                        +emf_edge_correction(i,ichild,iq);
                    // 2 B field values need to be corrected 
                    {
                        int signs[3] = {side_j ? -1 : +1, side_j ? +1 : -1, side_j ? -1 : +1 } ; 
                        int off_i = (other_dirs[edge_dir][1] == 0) ? (side_j ? -1 : 0) : 0 ;
                        int off_j = (other_dirs[edge_dir][1] == 1) ? (side_j ? -1 : 0) : 0 ; 
                        int off_k = (other_dirs[edge_dir][1] == 2) ? (side_j ? -1 : 0) : 0 ; 
                        (*views[other_dirs[edge_dir][0]])(ijk[0]+off_i,ijk[1]+off_j,ijk[2]+off_k,0,qid) += 
                            signs[edge_dir] * dt * dtfact * idx(other_dirs[edge_dir][1],qid) * emf_correction ; 
                    }
                    {
                        int signs[3] = {side_i ? +1 : -1, side_i ? -1 : +1, side_i ? +1 : -1} ; 
                        int off_i = (other_dirs[edge_dir][0] == 0) ? (side_i ? -1 : 0) : 0 ;
                        int off_j = (other_dirs[edge_dir][0] == 1) ? (side_i ? -1 : 0) : 0 ; 
                        int off_k = (other_dirs[edge_dir][0] == 2) ? (side_i ? -1 : 0) : 0 ; 
                        (*views[other_dirs[edge_dir][1]])(ijk[0]+off_i,ijk[1]+off_j,ijk[2]+off_k,0,qid) += 
                            signs[edge_dir] * dt * dtfact * idx(other_dirs[edge_dir][0],qid) * emf_correction ;
                    }
                }
            }
        }
    ) ; 
    #endif 
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
    auto flux_context = reflux_fill_flux_buffers() ; // todo 
    //**************************************************************************************************/
    compute_emfs(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ; 
    //**************************************************************************************************/
    auto emf_context = reflux_fill_emf_buffers() ; //todo 
    //**************************************************************************************************/
    add_fluxes_and_source_terms<eos_t>(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ;
    //**************************************************************************************************/
    update_CT(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ; 
    //**************************************************************************************************/
    update_fd(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ; 
    //**************************************************************************************************/
    reflux_correct_fluxes(flux_context,t,dt,dtfact,new_state) ;
    //**************************************************************************************************/
    reflux_correct_emfs(emf_context,t,dt,dtfact,new_stag_state) ; 
    //**************************************************************************************************/
    parallel::mpi_barrier() ; 
    #if 0
    #define RECONSTRUCT(vview,vidx,q,i,j,k,uL,uR,dir) \
    do { \
    auto sview = Kokkos::subview(vview, \
                                Kokkos::ALL(), \
                                Kokkos::ALL(), \
                                Kokkos::ALL(), \
                                vidx, \
                                q     ) ; \
    reconstructor(sview,i,j,k,uL,uR,dir) ; \
    } while(false)
    #define RECONSTRUCT_V(vview,jdir,vidx,q,i,j,k,uL,uR,dir) \
    do { \
    auto sview = Kokkos::subview(vview, \
                                Kokkos::ALL(), \
                                Kokkos::ALL(), \
                                Kokkos::ALL(), \
                                vidx, \
                                jdir, \
                                q     ) ; \
    reconstructor(sview,i,j,k,uL,uR,dir) ; \
    } while(false)
    // compute EMF -- x (stag yz)
    auto emf_policy_x = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz+1,nz+ngz+1),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_X")
                , emf_policy_x 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // i, j-1/2, k-1/2 
        // Ex = vz By - vy Bz 
        // reconstruct vz and By
        RECON reconstructor {} ; 
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
    
    // compute EMF -- y (stag xz)
    auto emf_policy_y = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz,nz+ngz+1),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_Y")
                , emf_policy_y 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // i-1/2, j, k-1/2 
        // Ex = vx Bz - vz Bx 
        // reconstruct vx and Bz
        RECON reconstructor {} ; 
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


    // compute EMF -- z (stag xy)
    auto emf_policy_z = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz+1,nz+ngz),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_Z")
                , emf_policy_z 
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        // i-1/2, j-1/2, k 
        // Ez = vy Bx - vx By 
        // reconstruct vy and Bx
        RECON reconstructor {} ; 
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



    auto advance_stag_policy_x = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz+1,ny+ngz,nz+ngz),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
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

    auto advance_stag_policy_y = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz+1,nz+ngz),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
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

    auto advance_stag_policy_z = 
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz+1),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "add_fluxes")
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
    #endif 
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