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
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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
#include <grace/evolution/boundary_outflow.hh>

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
#include <grace/physics/grmhd.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/eos_storage.hh>
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
#include <grace/physics/z4c.hh>
#include <grace/physics/z4c_helpers.hh>
#endif
#ifdef GRACE_ENABLE_M1 
#include <grace/physics/m1_helpers.hh>
#include <grace/physics/m1.hh>
#endif 
#include <grace/physics/eos/eos_types.hh>

#include <grace/amr/grace_amr.hh>

#ifdef GRACE_ENABLE_PARTICLES
#include <grace/particles/particles_module.hh>
#endif

#include <string>


#include <fstream> 
#include <iomanip> 

namespace grace {

// After AMR ghost-zone prolongation, the 5-point Lagrange interpolation of
// the conformal metric / extrinsic curvature leaves an O(dx^5) defect in
// det γ̃ and tr Ã in the ghost cells.  The subsequent 4th-order FD stencils
// would copy those defects into the interior at each substep — exactly the
// kind of AMR-boundary noise that drags BNS convergence below its formal
// rate.  Re-impose the BSSN/Z4c algebraic constraints (det γ̃ = 1, tr Ã = 0,
// χ ≥ χ_safe, α ≥ α_min) on the full extent after every BC fill; interior
// re-imposition is idempotent (rescale factor 1, trace already zero) and
// cheap relative to a single Z4c RHS kernel.  No-op under Cowling.
static inline void enforce_algebraic_constraints_after_bc(
    [[maybe_unused]] grace::var_array_t& state)
{
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    enforce_algebraic_constraints(state) ;
#endif
}

void evolve() {
    auto const eos_type = grace::get_param<std::string>("eos", "eos_type") ;
    GRACE_VERBOSE("Performing timestep integration at iteration {}", grace::get_iteration()) ; 
    if( eos_type == "hybrid" ) {
        auto const cold_eos_type = 
            get_param<std::string>("eos","hybrid_eos","cold_eos_type") ;  
        if( cold_eos_type == "piecewise_polytrope" ) {
            evolve_impl<grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>>() ;
        } else if ( cold_eos_type == "tabulated" ) {
            evolve_impl<grace::hybrid_eos_t<grace::tabulated_cold_eos_t>>() ;
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
    // Reset per-step c2p diagnostics once per timestep. C2P_DENS_ERR_ is a
    // signed mass-error accumulator; C2P_ERR_ holds a packed bit-pattern of
    // c2p failure modes (sticky-OR'd over the RK substages in grmhd.hh's
    // compute_auxiliaries — see c2p_err_enum_t for the layout).
    #ifndef GRACE_FREEZE_HYDRO
    Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ;
    parallel_for(GRACE_EXECUTION_TAG("EVOL","reset_c2p_diagnostics"), policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        aux(i,j,k,C2P_DENS_ERR_,q) = 0.0 ;
        aux(i,j,k,C2P_ERR_,q)      = 0.0 ;
    });
    #endif

    #ifdef GRACE_ENABLE_PARTICLES
    // Tracer push: sample fluid SRC state (state, aux at t^n) and FE-push
    // by dt. One call per full RK step is sub-CFL safe (|v_fluid| <=
    // |v_signal|, fluid CFL bounds the latter). No-op when the module is
    // disabled or empty. PIC species, when added, will need a per-substage
    // hook with field deposition between substages — not this entry point.
    grace::particles::particles_module_t::get().advance_step(dt);
    #endif

    if ( tstepper == "euler" ) {
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ;
        amr::apply_boundary_conditions(state, sstate,state_p,sstate_p,dt,1.0) ;
        enforce_algebraic_constraints_after_bc(state) ;
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
    } else if (tstepper == "rk2" ) {
        /* Compute auxiliaries at current timelevel */
        //compute_auxiliary_quantities<eos_t>(state, aux) ;
        advance_substep<eos_t>(t,dt,0.5,state_p,state,sstate_p,sstate) ;
        amr::apply_boundary_conditions(state_p,sstate_p,state,sstate,dt,0.5) ;
        enforce_algebraic_constraints_after_bc(state_p) ;
        compute_auxiliary_quantities<eos_t>(state_p, sstate_p, aux) ;
        advance_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ;
        amr::apply_boundary_conditions(state,sstate,state_p,sstate_p,dt,1.0) ;
        enforce_algebraic_constraints_after_bc(state) ;
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
    } else if (tstepper == "rk3" ) {
        // step 1: state_p -> u^1 = u^n + dt L( u^n )
        advance_substep<eos_t>(
            t,dt,1.0,
            state_p,state,
            sstate_p,sstate) ;
        amr::apply_boundary_conditions(state_p,sstate_p,state,sstate,dt,1.0) ;
        enforce_algebraic_constraints_after_bc(state_p) ;
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
        enforce_algebraic_constraints_after_bc(state_pp) ;
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
        enforce_algebraic_constraints_after_bc(state) ;
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
        enforce_algebraic_constraints_after_bc(s3) ;
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
        enforce_algebraic_constraints_after_bc(s1) ;
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
        enforce_algebraic_constraints_after_bc(s3) ;
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
        enforce_algebraic_constraints_after_bc(s1) ;
        compute_auxiliary_quantities<eos_t>(s1, ss1, aux) ;
    } else if (tstepper == "imex1") {
        advance_substep<eos_t>(t,dt,1.0,state,state_p,sstate,sstate_p) ;
        amr::apply_boundary_conditions(state,sstate,state_p,sstate_p,dt,1.0) ;
        enforce_algebraic_constraints_after_bc(state) ;
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
        enforce_algebraic_constraints_after_bc(state_p) ;
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
        enforce_algebraic_constraints_after_bc(state_p) ;
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
        enforce_algebraic_constraints_after_bc(state) ;
        compute_auxiliary_quantities<eos_t>(state, sstate, aux) ;
        /* done */
    } else if (tstepper == "imex232" ) { 
        ERROR("Imex 3 not implemented yet") ; 
    } else {
        ERROR("Unrecognised time-stepper.") ; 
    }
    Kokkos::deep_copy(state_p,state) ;
    grace::deep_copy(sstate_p,sstate) ;

    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    // Fill Hamiltonian and momentum constraints once per full RK step.
    // Reuses Ricci/Γ̃/matter cached in _z4c_curv_scratch by the last set
    // of pre-kernels (matter/wave/conn) of this RK step — much cheaper
    // than rebuilding the whole geometry
    // pass.  Post-regrid / post-initial-data paths call the full
    // compute_constraint_violations() in their own files.
    compute_constraint_violations_fast() ;
    #endif
}

/**
 * @brief FOFC stage 3: tentative-update + bad-cell flag.
 *
 * Builds the hypothetical post-substep cons (new_state base + dt*dtfact*divF,
 * i.e. what add_fluxes_and_source_terms would add next) and dry-runs c2p at
 * every interior cell.  Cells where c2p would have floored / reset to
 * atmosphere are compacted atomically into the singleton's FOFC index lists
 * (q,i,j,k) and the per-substep count.  apply_fofc_correction then iterates
 * [0, count) and re-runs the surrounding fluxes with donor + LLF.
 *
 * Geometric sources are omitted from the tentative — they're tiny relative
 * to the flux artifacts FOFC is designed to catch.  Cell-centered B comes
 * from old_stag_state (CT hasn't run yet); metric from old_state, matching
 * the geometry compute_fluxes used.
 */
template< typename eos_t >
void flag_fofc_cells(
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
    // FOFC widens the flag range by one cell into the ghost zone on each axis
    // so that the boundary face between two same-level neighbors is
    // recomputed with donor+LLF from both sides using mirror-consistent
    // primitives (FOFC algorithm: arXiv:2409.10384).  The
    // widest reconstruction stencil for the widened FACE at j=ngz-1 spans
    // cells [j-3, j+2] = [ngz-4, ngz+1]; the leftmost (ngz-4 = 0) is the
    // outermost ghost cell, so we need ngz >= 4 for PPM/WENO5/WENOZ.
    ASSERT(ngz >= 4,
        "FOFC requires ngz >= 4 with PPM/WENO5/WENOZ recon (widened "
        "boundary face at j=ngz-1 reads cell ngz-4). Current ngz = " << ngz) ;
    #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_UCT
        ASSERT(0, "FOFC is not supported with UCT due to the stencil involved in EMF evaluation.") ; 
    #endif 

    auto& idx      = grace::variable_list::get().getinvspacings() ;
    auto& fluxes   = grace::variable_list::get().getfluxesarray() ;
    auto& emf      = grace::variable_list::get().getemfarray() ;
    auto& fofc_faces    = grace::variable_list::get().getfofcfacetags() ;
    auto& fofc_edges    = grace::variable_list::get().getfofcedgetags() ;
    auto& fofc_face_cnt = grace::variable_list::get().getfofcfcnt() ;
    auto& fofc_edge_cnt = grace::variable_list::get().getfofcecnt() ;

    auto eos      = eos::get().get_eos<eos_t>() ;
    auto atmo     = get_atmo_params() ;
    auto excision = get_excision_params() ;
    auto c2p_pars = get_c2p_params() ;
    // Force c2p_is_lenient = true everywhere during the dry-run: a
    // Kokkos::abort would defeat FOFC's purpose, which is precisely to
    // catch tentative states the inversion can't handle.  alp_bh_thresh
    // above any realistic lapse triggers the lenient branch unconditionally.
    c2p_pars.alp_bh_thresh = 1e30 ;
    auto dcoords  = grace::coordinate_system::get().get_device_coord_system() ;

    // Per-substep reset: byte-flag tag views and the 3-element compact-slot
    // counters.  Lists are oversized to (worst-case interior) * nq so no
    // bound check is needed once the slot is claimed.
    Kokkos::deep_copy(fofc_faces,    int{0}) ;
    Kokkos::deep_copy(fofc_edges,    int{0}) ;
    Kokkos::deep_copy(fofc_face_cnt, int{0}) ;
    Kokkos::deep_copy(fofc_edge_cnt, int{0}) ;

    auto Bx = old_stag_state.face_staggered_fields_x ;
    auto By = old_stag_state.face_staggered_fields_y ;
    auto Bz = old_stag_state.face_staggered_fields_z ;

    // Widened by one ghost cell on each axis (vs. the original [ngz, nx+ngz)
    // interior-only range) so flagged ghost cells trigger boundary-face
    // recomputation from this rank's side.  The neighbor rank flags the
    // same physical cell as its own interior under deterministic local
    // computation from mirror-consistent ghost primitives, and rewrites
    // the same boundary face from its side with bit-identical LLF.
    auto fofc_flag_policy = MDRangePolicy<Rank<GRACE_NSPACEDIM+1>>(
          {VEC(ngz-1,ngz-1,ngz-1),0}
        , {VEC(nx+ngz+1,ny+ngz+1,nz+ngz+1),nq}
    ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "flag_fofc_cells")
                , fofc_flag_policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) {
        /************************************************************************************/
        grmhd_cons_array_t cons ;
        /************************************************************************************/
        // evol_hrsc_var_cc_idx [0..ENTROPYSTAR_] and GRMHD_CONS_LOC_INDICES
        // [DENSL..ENTSL] share order, so a single linear copy suffices.
        /************************************************************************************/
        for (int ivar = 0; ivar <= static_cast<int>(ENTROPYSTAR_); ++ivar) {
            cons[ivar] = new_state(VEC(i,j,k),ivar,q) + dt * dtfact * (
                EXPR(   ( fluxes(VEC(i,j,k)  ,ivar,0,q) - fluxes(VEC(i+1,j,k),ivar,0,q) ) * idx(0,q)
                    , + ( fluxes(VEC(i,j,k)  ,ivar,1,q) - fluxes(VEC(i,j+1,k),ivar,1,q) ) * idx(1,q)
                    , + ( fluxes(VEC(i,j,k)  ,ivar,2,q) - fluxes(VEC(i,j,k+1),ivar,2,q) ) * idx(2,q))
            ) ;
        }
        /************************************************************************************/

        /************************************************************************************/
        // Tentative post-CT face B values for the c2p dry-run.  Same FD as the
        // production CT update (see ct_update_B*_face in grmhd_helpers.hh);
        // factored to keep this kernel and apply_fofc_correction in sync.
        double const dt_eff = dt * dtfact ;
        double const Bxi  = ct_update_Bx_face(Bx, emf, idx, VEC(i,   j, k), q, dt_eff) ;
        double const Bxip = ct_update_Bx_face(Bx, emf, idx, VEC(i+1, j, k), q, dt_eff) ;
        double const Byj  = ct_update_By_face(By, emf, idx, VEC(i, j,   k), q, dt_eff) ;
        double const Byjp = ct_update_By_face(By, emf, idx, VEC(i, j+1, k), q, dt_eff) ;
        double const Bzk  = ct_update_Bz_face(Bz, emf, idx, VEC(i, j, k  ), q, dt_eff) ;
        double const Bzkp = ct_update_Bz_face(Bz, emf, idx, VEC(i, j, k+1), q, dt_eff) ;
        cons[BSXL] = 0.5 * ( Bxi + Bxip ) ;
        cons[BSYL] = 0.5 * ( Byj + Byjp ) ;
        cons[BSZL] = 0.5 * ( Bzk + Bzkp ) ;
        /************************************************************************************/

        /************************************************************************************/
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric, old_state, q, VEC(i,j,k)) ;

        double rtp[3] ;
        dcoords.get_physical_coordinates_sph(i,j,k,q,rtp) ;

        grmhd_prims_array_t prims{} ;
        c2p_err_t c2p_errors ;
        bool const floored = conservs_to_prims<eos_t>(
            cons, prims, metric, eos,
            atmo, excision, c2p_pars, rtp,
            c2p_errors ) ;
        /************************************************************************************/

        /************************************************************************************/
        if (floored) {
            // 6 faces touching cell (i,j,k):
            Kokkos::atomic_or(&fofc_faces(VEC(i,  j,  k  ), 0, q), int8_t{1});  // -x face
            Kokkos::atomic_or(&fofc_faces(VEC(i+1,j,  k  ), 0, q), int8_t{1});  // +x face
            Kokkos::atomic_or(&fofc_faces(VEC(i,  j,  k  ), 1, q), int8_t{1});  // -y
            Kokkos::atomic_or(&fofc_faces(VEC(i,  j+1,k  ), 1, q), int8_t{1});  // +y
            Kokkos::atomic_or(&fofc_faces(VEC(i,  j,  k  ), 2, q), int8_t{1});  // -z
            Kokkos::atomic_or(&fofc_faces(VEC(i,  j,  k+1), 2, q), int8_t{1});  // +z

            // 12 edges touching cell (i,j,k):
            // E^x parallel to x-axis at the 4 (j,k) corners of the cell
            Kokkos::atomic_or(&fofc_edges(VEC(i,j,  k  ), 0, q), int8_t{1});
            Kokkos::atomic_or(&fofc_edges(VEC(i,j+1,k  ), 0, q), int8_t{1});
            Kokkos::atomic_or(&fofc_edges(VEC(i,j,  k+1), 0, q), int8_t{1});
            Kokkos::atomic_or(&fofc_edges(VEC(i,j+1,k+1), 0, q), int8_t{1});
            // E^y parallel to y-axis at the 4 (i,k) corners
            Kokkos::atomic_or(&fofc_edges(VEC(i,  j,k  ), 1, q), int8_t{1});
            Kokkos::atomic_or(&fofc_edges(VEC(i+1,j,k  ), 1, q), int8_t{1});
            Kokkos::atomic_or(&fofc_edges(VEC(i,  j,k+1), 1, q), int8_t{1});
            Kokkos::atomic_or(&fofc_edges(VEC(i+1,j,k+1), 1, q), int8_t{1});
            // E^z parallel to z-axis at the 4 (i,j) corners
            Kokkos::atomic_or(&fofc_edges(VEC(i,  j,  k), 2, q), int8_t{1});
            Kokkos::atomic_or(&fofc_edges(VEC(i+1,j,  k), 2, q), int8_t{1});
            Kokkos::atomic_or(&fofc_edges(VEC(i,  j+1,k), 2, q), int8_t{1});
            Kokkos::atomic_or(&fofc_edges(VEC(i+1,j+1,k), 2, q), int8_t{1});
        }
        /************************************************************************************/
    }) ;

    // compact
    auto& fofc_fx  = grace::variable_list::get().getfofcfx() ;
    auto& fofc_fy  = grace::variable_list::get().getfofcfy() ;
    auto& fofc_fz  = grace::variable_list::get().getfofcfz() ;
    auto& fofc_eyz = grace::variable_list::get().getfofceyz() ;
    auto& fofc_exz = grace::variable_list::get().getfofcexz() ;
    auto& fofc_exy = grace::variable_list::get().getfofcexy() ;
    // fofc_face_cnt / fofc_edge_cnt are already bound at the top of the function.

    auto fofc_compact_policy = MDRangePolicy<Rank<GRACE_NSPACEDIM+1>>(
          {VEC(ngz-1,ngz-1,ngz-1),0}
        , {VEC(nx+ngz+1,ny+ngz+1,nz+ngz+1),nq}
    ) ;
    parallel_for("fofc_compact_faces_and_edges", fofc_compact_policy,
        KOKKOS_LAMBDA(int i, int j, int k, int q) {
        // X face
        // needed range: i \in [ngz,  nx + ngz] (i.e. inclusive!) for flux update
        //               j \in [ngz-1, ny + ngz], k \in [ngz-1, nz + ngz] for emf computation (GS only!)
        if ( (i >= ngz) && (fofc_faces(VEC(i,j,k), 0, q)) ) {
            int slot = Kokkos::atomic_fetch_add(&fofc_face_cnt(0), 1);
            fofc_index_tag_t tag ; 
            tag.q = q ; tag.i = i ; tag.j = j ; tag.k = k ; 
            fofc_fx(slot) = tag ; 
        }
        // Y face
        // needed range: j \in [ngz,  ny + ngz] (i.e. inclusive!) for flux update
        //               i \in [ngz-1, nx + ngz], k \in [ngz-1, nz + ngz] for emf computation (GS only!)
        if ((j >= ngz) && (fofc_faces(VEC(i,j,k), 1, q))) {
            int slot = Kokkos::atomic_fetch_add(&fofc_face_cnt(1), 1);
            fofc_index_tag_t tag ; 
            tag.q = q ; tag.i = i ; tag.j = j ; tag.k = k ; 
            fofc_fy(slot) = tag ;
        }
        // Z face
        // needed range: k \in [ngz,  nz + ngz] (i.e. inclusive!) for flux update
        //               i \in [ngz-1, nx + ngz], j \in [ngz-1, ny + ngz] for emf computation (GS only!)
        if ((k >= ngz) && (fofc_faces(VEC(i,j,k), 2, q))) {
            int slot = Kokkos::atomic_fetch_add(&fofc_face_cnt(2), 1);
            fofc_index_tag_t tag ; 
            tag.q = q ; tag.i = i ; tag.j = j ; tag.k = k ; 
            fofc_fz(slot) = tag ;
        }
        // YZ EDGE (E^x, parallel to x-axis, staggered in y and z)
        // needed range: i \in [ngz, nx+ngz-1] for CT update
        //               j \in [ngz, ny+ngz],   k \in [ngz, nz+ngz] for CT update
        if ( (i>=ngz) && (j>=ngz) && (k>=ngz) && (i<nx+ngz) && (fofc_edges(i,j,k,0,q)) ) {
            int slot = Kokkos::atomic_fetch_add(&fofc_edge_cnt(0), 1);
            fofc_index_tag_t tag ; 
            tag.q = q ; tag.i = i ; tag.j = j ; tag.k = k ; 
            fofc_eyz(slot) = tag ;
        }
        // XZ EDGE (E^y, parallel to y-axis, staggered in x and z)
        // needed range: j \in [ngz, ny+ngz-1] for CT update
        //               i \in [ngz, nx+ngz],   k \in [ngz, nz+ngz] for CT update
        if ( (j>=ngz) && (i>=ngz) && (k>=ngz) && (j<ny+ngz) && (fofc_edges(i,j,k,1,q)) ) {
            int slot = Kokkos::atomic_fetch_add(&fofc_edge_cnt(1), 1);
            fofc_index_tag_t tag ; 
            tag.q = q ; tag.i = i ; tag.j = j ; tag.k = k ; 
            fofc_exz(slot) = tag ;
        }
        // XY EDGE (E^z, parallel to z-axis, staggered in x and y)
        // needed range: k \in [ngz, nz+ngz-1] for CT update
        //               i \in [ngz, nx+ngz],   j \in [ngz, ny+ngz] for CT update
        if ( (k>=ngz) && (i>=ngz) && (j>=ngz) && (k<nz+ngz) && (fofc_edges(i,j,k,2,q)) ) {
            int slot = Kokkos::atomic_fetch_add(&fofc_edge_cnt(2), 1);
            fofc_index_tag_t tag ; 
            tag.q = q ; tag.i = i ; tag.j = j ; tag.k = k ; 
            fofc_exy(slot) = tag ;
        }
    });
}

template< typename eos_t >
void apply_fofc_correction(
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
    auto& dx      = grace::variable_list::get().getspacings() ;
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ;
    auto& emf     = grace::variable_list::get().getemfarray() ;
    auto& aux     = grace::variable_list::get().getaux()     ;
    #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS
    auto& Eface  = grace::variable_list::get().getefarray() ;
    auto& Ecenter  = grace::variable_list::get().getecarray() ;
    #else
    ASSERT(0, "Should have been caught earlier, FOFC and UCT are incompatible.") ; 
    #endif
    // FOFC index lists were populated atomically by flag_fofc_cells.  The
    // counter scalar was already copied back to host there (for the log line),
    // but flag_fofc_cells is a separate translation-unit-level call, so we
    // bring the count over again here — it's a single int.
    auto& fofc_fx = grace::variable_list::get().getfofcfx() ;
    auto& fofc_fy = grace::variable_list::get().getfofcfy() ;
    auto& fofc_fz = grace::variable_list::get().getfofcfz() ;
    auto& fofc_eyz = grace::variable_list::get().getfofceyz() ;
    auto& fofc_exz = grace::variable_list::get().getfofcexz() ;
    auto& fofc_exy = grace::variable_list::get().getfofcexy() ;

    auto& fofc_face_cnt = grace::variable_list::get().getfofcfcnt() ;
    auto& fofc_edge_cnt = grace::variable_list::get().getfofcecnt() ;
    // Stage through a HostSpace mirror of the same View<int[1]> shape rather
    // than deep_copy-ing into a bare int (which isn't portable across Kokkos
    // backends for rank-0/rank-1 sources).
    Kokkos::View<int[3], Kokkos::HostSpace> host_face_cnt("fofc_face_count_host") ;
    Kokkos::deep_copy(host_face_cnt, fofc_face_cnt) ;
    Kokkos::View<int[3], Kokkos::HostSpace> host_edge_cnt("fofc_edge_count_host") ;
    Kokkos::deep_copy(host_edge_cnt, fofc_edge_cnt) ;
    //**************************************************************************************************/
    using recon_t   = donor_cell_reconstructor_t ;
    using riemann_t = llf_riemann_tag_t ;
    //**************************************************************************************************/
    auto eos = eos::get().get_eos<eos_t>() ;
    grmhd_equations_system_t<eos_t>
        grmhd_eq_system(eos,old_state,old_stag_state,aux) ;

    #ifndef GRACE_FREEZE_HYDRO
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "correct_x_fluxes_fofc")
                , host_face_cnt(0)
                , KOKKOS_LAMBDA (int idx) {
        auto qijk = fofc_fx(idx) ; 
        grmhd_eq_system.template compute_x_flux<recon_t,riemann_t>(qijk.q,qijk.i,qijk.j,qijk.k, fluxes, Eface, dx, dt, dtfact) ;
    }) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "correct_y_fluxes_fofc")
                , host_face_cnt(1)
                , KOKKOS_LAMBDA (int idx) {
        auto qijk = fofc_fy(idx) ; 
        grmhd_eq_system.template compute_y_flux<recon_t,riemann_t>(qijk.q,qijk.i,qijk.j,qijk.k, fluxes, Eface, dx, dt, dtfact) ;
    }) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "correct_z_fluxes_fofc")
                , host_face_cnt(2)
                , KOKKOS_LAMBDA (int idx) {
        auto qijk = fofc_fz(idx) ; 
        grmhd_eq_system.template compute_z_flux<recon_t,riemann_t>(qijk.q,qijk.i,qijk.j,qijk.k, fluxes, Eface, dx, dt, dtfact) ;
    }) ;
    // Recompute the GS edge EMF on every flagged edge using the (partly
    // updated) Eface / Ecenter / fluxes.  Same arithmetic as compute_emfs
    // — see gs_edge_emf_{x,y,z} in grmhd_helpers.hh for the discretization.
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "correct_yz_edge_fofc")
                , host_edge_cnt(0)
                , KOKKOS_LAMBDA (int idx_) {
        auto qijk = fofc_eyz(idx_) ;
        emf(qijk.i, qijk.j, qijk.k, 0, qijk.q) =
            gs_edge_emf_x(Eface, Ecenter, fluxes,
                          VEC(qijk.i, qijk.j, qijk.k), qijk.q) ;
    }) ;

    parallel_for( GRACE_EXECUTION_TAG("EVOL", "correct_xz_edge_fofc")
                , host_edge_cnt(1)
                , KOKKOS_LAMBDA (int idx_) {
        auto qijk = fofc_exz(idx_) ;
        emf(qijk.i, qijk.j, qijk.k, 1, qijk.q) =
            gs_edge_emf_y(Eface, Ecenter, fluxes,
                          VEC(qijk.i, qijk.j, qijk.k), qijk.q) ;
    }) ;

    parallel_for( GRACE_EXECUTION_TAG("EVOL", "correct_xy_edge_fofc")
                , host_edge_cnt(2)
                , KOKKOS_LAMBDA (int idx_) {
        auto qijk = fofc_exy(idx_) ;
        emf(qijk.i, qijk.j, qijk.k, 2, qijk.q) =
            gs_edge_emf_z(Eface, Ecenter, fluxes,
                          VEC(qijk.i, qijk.j, qijk.k), qijk.q) ;
    }) ;
    #endif
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
    #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS
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
    //
    // Launch-bounds note: the WENO5 reconstruction + HLL/LLF flux mix pushes
    // these kernels past 128 VGPR under the default heuristic, so the compiler
    // spills ~1.5-2 KB/thread to scratch.  LaunchBounds<256, 2> lifts the cap
    // to ~256 VGPR (2 waves/EU instead of 4) in exchange for half the latency
    // hiding; worthwhile because the kernels are compute-heavy (WENO smoothness
    // indicators, Riemann branches) rather than bandwidth-bound.  Tile product
    // must match MaxThreadsPerBlock or HIP silently rejects the launch — see
    // z4c_update notes below.
    #ifndef GRACE_FLUX_LB
      #define GRACE_FLUX_LB Kokkos::LaunchBounds<256, 2>
    #endif
    // GRACE_LB_ARG expands to ", <LB>" or to nothing, depending on whether
    // the active perf-tuning header asked us to drop launch_bounds entirely
    // (GRACE_NO_LB).  Letting the policy template omit the LaunchBounds
    // parameter is preferable to passing LaunchBounds<0,0>, whose behaviour
    // is unspecified by NVIDIA's CUDA C++ Programming Guide.
    #ifdef GRACE_NO_LB
      #define GRACE_LB_ARG(LB) /* no launch bounds */
    #else
      #define GRACE_LB_ARG(LB) , LB
    #endif
    using flux_policy_t =
        MDRangePolicy< Rank<GRACE_NSPACEDIM+1> GRACE_LB_ARG(GRACE_FLUX_LB) > ;
    // FOFC widening: compute one extra HO flux on each side of the interior
    // so that the FOFC test at one ghost cell deep on each axis has a valid
    // F(i)/F(i+1) pair.  Requires ngz >= recon_half_width + 1 (4 for
    // WENO5/WENOZ/PPM, runtime-checked in flag_fofc_cells).
    flux_policy_t flux_x_policy_mhd (
              {VEC(ngz-1,0,0),0}
            , {VEC(nx+ngz+2,ny+2*ngz,nz+2*ngz),nq}
            , {VEC(16,4,4),1}
        ) ;
    flux_policy_t flux_y_policy_mhd (
              {VEC(0,ngz-1,0),0}
            , {VEC(nx+2*ngz,ny+ngz+2,nz+2*ngz),nq}
            , {VEC(16,4,4),1}
        ) ;
    flux_policy_t flux_z_policy_mhd (
              {VEC(0,0,ngz-1),0}
            , {VEC(nx+2*ngz,ny+2*ngz,nz+ngz+2),nq}
            , {VEC(16,4,4),1}
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
    // Surface silent launch failures (mismatched LaunchBounds/tile or exceeded
    // occupancy constraints).  Kokkos does not abort on these by default.
    #if defined(KOKKOS_ENABLE_HIP)
    Kokkos::fence("compute_fluxes post-launch error check") ;
    {
        auto _err = hipGetLastError() ;
        if (_err != hipSuccess) {
            ERROR("compute_fluxes kernel launch failed: " << hipGetErrorString(_err)) ;
        }
    }
    #elif defined(KOKKOS_ENABLE_CUDA)
    Kokkos::fence("compute_fluxes post-launch error check") ;
    {
        auto _err = cudaGetLastError() ;
        if (_err != cudaSuccess) {
            ERROR("compute_fluxes kernel launch failed: " << cudaGetErrorString(_err)) ;
        }
    }
    #else
    Kokkos::fence() ;
    #endif
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
#if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS
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
    auto& fluxes  = grace::variable_list::get().getfluxesarray() ;
    auto& Eface   = grace::variable_list::get().getefarray() ;
    auto& Ecenter = grace::variable_list::get().getecarray() ;
    auto& emf     = grace::variable_list::get().getemfarray() ;
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
    // The GS edge EMF assembly is identical to apply_fofc_correction's
    // edge-recompute kernels and shares one implementation in
    // gs_edge_emf_{x,y,z} (grmhd_helpers.hh).
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_X")
                , emf_policy_x
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        emf(i,j,k,0,q) = gs_edge_emf_x(Eface, Ecenter, fluxes, VEC(i,j,k), q);
    });
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_Y")
                , emf_policy_y
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        emf(i,j,k,1,q) = gs_edge_emf_y(Eface, Ecenter, fluxes, VEC(i,j,k), q);
    });
    parallel_for( GRACE_EXECUTION_TAG("EVOL", "EMF_Z")
                , emf_policy_z
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        emf(i,j,k,2,q) = gs_edge_emf_z(Eface, Ecenter, fluxes, VEC(i,j,k), q);
    });
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
            // this way of writing is explicitly bit-wise equivariant
            double dFx = ( fluxes(VEC(i,j,k)  ,ivar,0,q) - fluxes(VEC(i+1,j,k),ivar,0,q) ) * idx(0,q);
            double dFy = ( fluxes(VEC(i,j,k)  ,ivar,1,q) - fluxes(VEC(i,j+1,k),ivar,1,q) ) * idx(1,q);
            double dFz = ( fluxes(VEC(i,j,k)  ,ivar,2,q) - fluxes(VEC(i,j,k+1),ivar,2,q) ) * idx(2,q);
            new_state(VEC(i,j,k),ivar,q) += dt * dtfact * ( (dFx + dFy) + dFz ) ; 
        }
        
    }) ; 
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
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    //**************************************************************************************************/
    // fetch some stuff
    auto& idx     = grace::variable_list::get().getinvspacings() ;
    auto& aux     = grace::variable_list::get().getaux()     ;
    auto& curv_scratch = grace::variable_list::get().getz4ccurvscratch() ;
    auto dev_coords = grace::coordinate_system::get().get_device_coord_system() ;
    //**************************************************************************************************/
    z4c_system_t z4c_eq_system(old_state,aux,old_stag_state,curv_scratch) ;
    //**************************************************************************************************/
    // Launch-bounds note: a too-tight LaunchBounds (in particular a
    // MaxThreadsPerBlock smaller than the policy's tile-product) lets hipcc
    // emit a kernel the runtime silently rejects, leaving the kernel a no-op
    // while reporting a tiny runtime in the trace.  Always pair LaunchBounds
    // with an explicit tile whose product equals MaxThreadsPerBlock, and rely
    // on the post-launch hipGetLastError below to surface any mismatch.
    //
    // Defaults below: 256-thread blocks (4 wave64s on MI300A) for all three kernels.
    //   adv:      LaunchBounds<256, 4> — bandwidth-bound, ask for high occupancy
    //             (4 waves/EU min ≈ 128 VGPR/lane cap, fine for the small footprint).
    //   curv_pre: LaunchBounds<256, 1> — second-derivative heavy (ddgtdd_dx2[36],
    //             ddchi_dx2[6], dGammat_dx[9]); needs the full register file.
    //   curv:     LaunchBounds<256, 1> — still register-pressured even after
    //             scratch read, give it the full register file.
    //
    // Override per-architecture via -DGRACE_Z4C_*_LB / -DGRACE_Z4C_*_TILE.
    #ifndef GRACE_Z4C_ADV_LB
      #define GRACE_Z4C_ADV_LB  Kokkos::LaunchBounds<256, 4>
    #endif
    #ifndef GRACE_Z4C_CURV_PRE_LB
      #define GRACE_Z4C_CURV_PRE_LB Kokkos::LaunchBounds<256, 1>
    #endif
    #ifndef GRACE_Z4C_CURV_LB
      #define GRACE_Z4C_CURV_LB Kokkos::LaunchBounds<256, 1>
    #endif
    // GRACE_LB_ARG (defined earlier in this file) expands to ", <LB>" or to
    // nothing when GRACE_NO_LB is set by the active perf-tuning header.
    using adv_policy_t =
        MDRangePolicy< Rank<GRACE_NSPACEDIM+1> GRACE_LB_ARG(GRACE_Z4C_ADV_LB) > ;
    using curv_pre_policy_t =
        MDRangePolicy< Rank<GRACE_NSPACEDIM+1> GRACE_LB_ARG(GRACE_Z4C_CURV_PRE_LB) > ;
    using curv_policy_t =
        MDRangePolicy< Rank<GRACE_NSPACEDIM+1> GRACE_LB_ARG(GRACE_Z4C_CURV_LB) > ;
    // Tile chosen so the product = 256 = MaxThreadsPerBlock above.
    // Shape (16,4,4,1) on LayoutLeft data: one wavefront of 64 lanes covers
    // 16 contiguous i × 4 j × 1 k = four full 128B cache lines.  q=1 so a
    // single block never straddles two patches (patches are tens of MB apart
    // along the last view dim).
    adv_policy_t advective_policy (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
            , {VEC(16,4,4),1}
        ) ;
    curv_pre_policy_t curvature_pre_policy (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
            , {VEC(16,4,4),1}
        ) ;
    curv_policy_t curvature_policy (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
            , {VEC(16,4,4),1}
        ) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_advective_update")
                , advective_policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    z4c_eq_system.compute_advective_update(q,VEC(i,j,k),idx,new_state,new_stag_state,dt,dtfact,dev_coords);
                }) ;
    //**************************************************************************************************/
    // Matter sources — cheap, mostly GRMHD/aux reads; dispatched first so it
    // can overlap with wave on hardware that exposes concurrent kernel
    // execution.  Body is a near-no-op in vacuum runs (branched on is_vacuum
    // inside the functor).
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_matter_sources")
                , curvature_pre_policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    z4c_eq_system.compute_matter_sources(q,VEC(i,j,k),idx,new_state,new_stag_state,dt,dtfact,dev_coords);
                }) ;
    //**************************************************************************************************/
    // Ricci wave — writes W2R_ij (wave) into _curv_scratch[RICCI_*].
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_ricci_wave")
                , curvature_pre_policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    z4c_eq_system.compute_curvature_wave(q,VEC(i,j,k),idx,new_state,new_stag_state,dt,dtfact,dev_coords);
                }) ;
    //**************************************************************************************************/
    // Ricci connection + conformal — reads wave piece back, accumulates,
    // stores final Ricci + Rtrace + Gammatudd.  Must follow z4c_ricci_wave.
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_ricci_conn")
                , curvature_pre_policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    z4c_eq_system.compute_curvature_conn(q,VEC(i,j,k),idx,new_state,new_stag_state,dt,dtfact,dev_coords);
                }) ;
    //**************************************************************************************************/
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_curvature_update")
                , curvature_policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    z4c_eq_system.compute_curvature_update(q,VEC(i,j,k),idx,new_state,new_stag_state,dt,dtfact,dev_coords);
                }) ;
    // Surface silent launch failures.  HIP can return e.g. hipErrorInvalidConfiguration
    // for a kernel whose register/scratch footprint is incompatible with the chosen
    // launch geometry, and Kokkos does not abort on it by default.
    #if defined(KOKKOS_ENABLE_HIP)
    Kokkos::fence("z4c_update post-launch error check") ;
    {
        auto _err = hipGetLastError() ;
        if (_err != hipSuccess) {
            ERROR("z4c update kernel launch failed: " << hipGetErrorString(_err)) ;
        }
    }
    #elif defined(KOKKOS_ENABLE_CUDA)
    Kokkos::fence("z4c_update post-launch error check") ;
    {
        auto _err = cudaGetLastError() ;
        if (_err != cudaSuccess) {
            ERROR("z4c update kernel launch failed: " << cudaGetErrorString(_err)) ;
        }
    }
    #endif
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
    compute_emfs(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ;
    //**************************************************************************************************/
#ifdef GRACE_ENABLE_FOFC
    flag_fofc_cells<eos_t>(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ;
    //**************************************************************************************************/
    apply_fofc_correction<eos_t>(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ;
    //**************************************************************************************************/
#endif
    auto flux_context = reflux_fill_flux_buffers() ;
    //**************************************************************************************************/
    auto emf_context = reflux_fill_emf_buffers() ;
    //**************************************************************************************************/
    reflux_correct_fluxes(flux_context) ;
    //**************************************************************************************************/
    // Mass-conservation diagnostic: integrate the DENS face flux over the
    // outer (non-periodic) domain boundary using the SAME (corrected) flux
    // about to be applied to the state.  Weighted by dt·dtfact so each RK
    // substage contributes correctly; sticky across substages within an
    // output interval, MPI-reduced and flushed by the diagnostic.
    boundary_outflow_t::get().accumulate(dt, dtfact) ;
    //**************************************************************************************************/
    add_fluxes_and_source_terms<eos_t>(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ;
    //**************************************************************************************************/
    reflux_correct_emfs(emf_context) ;
    //**************************************************************************************************/
    update_CT(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ;
    //**************************************************************************************************/
    update_fd(t,dt,dtfact,new_state,old_state,new_stag_state,old_stag_state) ;
    //**************************************************************************************************/
    parallel::mpi_barrier() ;  
    Kokkos::fence() ; 
    GRACE_PROFILING_POP_REGION ; 
}

#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
void compute_constraint_violations() {
    using namespace grace ;
    using namespace Kokkos  ;
    DECLARE_GRID_EXTENTS ;
    //**************************************************************************************************/
    // fetch some stuff
    auto& idx     = grace::variable_list::get().getinvspacings() ;
    auto& aux     = grace::variable_list::get().getaux()     ;
    auto& state   = grace::variable_list::get().getstate()   ;
    auto& curv_scratch = grace::variable_list::get().getz4ccurvscratch() ;
    auto dev_coords = grace::coordinate_system::get().get_device_coord_system() ;
    auto dummy = grace::variable_list::get().getstaggeredstate() ;
    //**************************************************************************************************/
    z4c_system_t z4c_eq_system(state,aux,dummy,curv_scratch) ;
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

// Fast variant: requires _z4c_curv_scratch to hold Ricci + Gammatudd + matter
// sources consistent with the current state (true immediately after the last
// matter/wave/conn dispatch of an RK step, invalid after regrid or fresh
// initial data).
// Avoids rebuilding Christoffel / Ricci / matter from scratch; only recomputes
// gtuu/Atuu/AA, GammatDu and 5 centered first-deriv stencils before calling
// z4c_get_constraints.
void compute_constraint_violations_fast() {
    using namespace grace ;
    using namespace Kokkos ;
    DECLARE_GRID_EXTENTS ;
    auto& idx          = grace::variable_list::get().getinvspacings() ;
    auto& aux          = grace::variable_list::get().getaux() ;
    auto& state        = grace::variable_list::get().getstate() ;
    auto& curv_scratch = grace::variable_list::get().getz4ccurvscratch() ;
    auto dummy         = grace::variable_list::get().getstaggeredstate() ;
    z4c_system_t z4c_eq_system(state,aux,dummy,curv_scratch) ;
    auto policy =
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>> (
              {VEC(ngz,ngz,ngz),0}
            , {VEC(nx+ngz,ny+ngz,nz+ngz),nq}
        ) ;
    parallel_for( GRACE_EXECUTION_TAG("EVOL","z4c_compute_constraint_violations_fast")
                , policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    z4c_eq_system.compute_constraints_fast(VEC(i,j,k), q, idx) ;
                }) ;
}

void enforce_algebraic_constraints(grace::var_array_t& state) {
    using namespace grace ; 
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS ;
    //**************************************************************************************************/
    // fetch some stuff
    auto& idx     = grace::variable_list::get().getinvspacings() ;
    auto& aux     = grace::variable_list::get().getaux()     ;
    auto& curv_scratch = grace::variable_list::get().getz4ccurvscratch() ;
    auto dev_coords = grace::coordinate_system::get().get_device_coord_system() ;
    auto dummy = grace::variable_list::get().getstaggeredstate() ;
    //**************************************************************************************************/
    z4c_system_t z4c_eq_system(state,aux,dummy,curv_scratch) ;
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
void flag_fofc_cells<EOS>( double const , double const , double const \
                         , grace::var_array_t&                        \
                         , grace::var_array_t&                        \
                         , grace::staggered_variable_arrays_t &       \
                         , grace::staggered_variable_arrays_t &       \
                         ) ;                                          \
template                                                             \
void apply_fofc_correction<EOS>( double const , double const , double const \
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
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::tabulated_cold_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE
}
