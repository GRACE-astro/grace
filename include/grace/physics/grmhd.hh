/**
 * @file grmhd.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-28
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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
#ifndef GRACE_PHYSICS_GRMHD_HH
#define GRACE_PHYSICS_GRMHD_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/evolution/hrsc_evolution_system.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/reconstruction.hh>
#include <grace/utils/weno_reconstruction.hh>
#include <grace/utils/riemann_solvers.hh>
#include <grace/utils/rootfinding.hh>
#include "grmhd_subexpressions.hh"
#include "tetrad_subexpressions.hh"
#include "fd_subexpressions.hh"
#include "z4c_subexpressions.hh"
#include <Kokkos_Core.hpp>

#include <type_traits>
//**************************************************************************************************/
/**
 * \defgroup physics Physics Modules.
 */
namespace grace {
//**************************************************************************************************/ 
//**************************************************************************************************
/**
 * @brief GRMHD equations system.
 * \ingroup physics 
 * @tparam eos_t Type of equation of state used.
 */
//**************************************************************************************************/
template< typename eos_t >
struct grmhd_equations_system_t 
    : public hrsc_evolution_system_t<grmhd_equations_system_t<eos_t>>
{
 private:
    //! Base class type 
    using base_t = hrsc_evolution_system_t<grmhd_equations_system_t<eos_t>>;

 public:
    /**
     * @brief Constructor
     *
     * @param eos_ eos object.
     * @param state_ State array.
     * @param aux_ Auxiliary array.
     */
    grmhd_equations_system_t( eos_t eos_
                            , grace::var_array_t state_
                            , grace::staggered_variable_arrays_t stag_state_
                            , grace::var_array_t aux_ )
     : base_t(state_,stag_state_,aux_), _eos(eos_)
    {
        excision_params = get_excision_params() ;
        atmo_params = get_atmo_params() ;
        c2p_params = get_c2p_params() ;
        dcoords = grace::coordinate_system::get().get_device_coord_system();
    } ;

    /**
     * @brief Compute GRMHD fluxes in direction \f$x^1\f$
     * 
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename recon_t, typename riemann_t = default_riemann_tag_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_x_flux_impl( int const q
                       , VEC( const int i
                       ,      const int j
                       ,      const int k)
                       , grace::flux_array_t const  fluxes
                       , grace::flux_array_t const  vbar
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt
                       , double const dtfact ) const
    {
        getflux<0,recon_t,riemann_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
    }
    /**
     * @brief Compute GRMHD fluxes in direction \f$x^2\f$
     * 
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename recon_t, typename riemann_t = default_riemann_tag_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_y_flux_impl( int const q
                       , VEC( const int i
                       ,      const int j
                       ,      const int k)
                       , grace::flux_array_t const  fluxes
                       , grace::flux_array_t const  vbar
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt
                       , double const dtfact ) const
    {
        getflux<1,recon_t,riemann_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
    }
    /**
     * @brief Compute GRMHD fluxes in direction \f$x^3\f$
     * 
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename recon_t, typename riemann_t = default_riemann_tag_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_z_flux_impl( int const q
                       , VEC( const int i
                       ,      const int j
                       ,      const int k)
                       , grace::flux_array_t const  fluxes
                       , grace::flux_array_t const  vbar
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                       , double const dt
                       , double const dtfact ) const
    {
        getflux<2,recon_t,riemann_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
    }
    /**
     * @brief Compute geometric source terms for GRMHD equations.
     * 
     * @tparam thread_team_t Thread team type.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param idx Inverse cell coordinate spacings.
     * @param state_new State where sources are added.
     * @param dt Timestep.
     * @param dtfact Timestep factor.
     */
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_source_terms( const int q 
                         , VEC( const int i 
                         ,      const int j 
                         ,      const int k)
                         , grace::scalar_array_t<GRACE_NSPACEDIM> const idx
                         , grace::var_array_t const state_new
                         , double const dt 
                         , double const dtfact ) const 
    {
        using namespace grace  ;
        using namespace Kokkos ;
        auto s = subview(this->_state,i,j,k,ALL(),q) ;
        /**************************************************************************************************/
        /* Read in the metric                                                                             */
        /**************************************************************************************************/
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;        
        /**************************************************************************************************/
        /* Read the primitive variables                                                                   */
        /**************************************************************************************************/
        grmhd_prims_array_t prims ; 
        FILL_PRIMS_ARRAY_ZVEC(prims,this->_aux,q,VEC(i,j,k))   ;
        double const eps   = prims[EPSL]   ; 
        double const rho   = prims[RHOL]   ; 
        double const p     = prims[PRESSL] ;
        double const * const   z   = &(prims[ZXL])       ;  
        double const * const   B   = &(prims[BXL])       ;  
        double const * const betau = metric._beta.data() ; 
        double const * const gdd   = metric._g.data()    ; 
        double const * const guu   = metric._ginv.data() ; 
        double const sqrtg         = metric.sqrtg()      ; 
        double const alp           = metric.alp()        ; 
        /**************************************************************************************************/
        double W ; 
        grmhd_get_W(gdd,z,&W) ; 
        double b2, smallb[4] ; 
        grmhd_get_smallbu_smallb2(alp,betau,gdd,B,z,W,&smallb,&b2) ;
        /**************************************************************************************************/
        /* Metric derivatives                                                                             */
        /**************************************************************************************************/
        double dalpha_dx[3], dgdd_dx[18], dbetau_dx[9]; 
        fill_deriv_scalar<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, ALP_, q, dalpha_dx, idx(0,q)) ;
        fill_deriv_vector<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, BETAX_, q, dbetau_dx, idx(0,q)) ;
        #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
        fill_deriv_tensor<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, GXX_, q, dgdd_dx, idx(0,q)) ;
        #else
        // conformal factor W = 1/gamma^{1/6}
        double Wt  = s(CHI_) ;
        // 1/W
        double ooW = 1./Wt ;
        // 1/W^2
        double ooWsqr = SQR(ooW) ;
        // dW_dx/y/z
        double dchi_dx[3] ;
        fill_deriv_scalar<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, CHI_, q, dchi_dx, idx(0,q)) ;
        fill_deriv_tensor<MATTER_METRIC_DER_ORDER>(this->_state, i,j,k, GTXX_, q, dgdd_dx, idx(0,q)) ;
        // gdd = gtdd/W^2
        // dgdd/dx = d gtdd / dx / W^2 - 2 gtdd d W / dx / W^3 
        for( int idir=0; idir<3; ++idir) {
            for( int a=0; a<6; ++a) {
                dgdd_dx[a + 6*idir] = ooWsqr * dgdd_dx[a + 6*idir] - 2. * ooW * dchi_dx[idir] * gdd[a] ;  
            }
        }
        #endif 
        

        /**************************************************************************************************/
        /* Extrinsic curvature                                                                            */
        /**************************************************************************************************/
        double Kdd[6] ; 
        #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
        Kdd[0] = s(KXX_) ; Kdd[1] = s(KXY_) ; Kdd[2] = s(KXZ_) ; 
        Kdd[3] = s(KYY_) ; Kdd[4] = s(KYZ_) ; Kdd[5] = s(KZZ_) ; 
        #else
        // K_{ij} = \tilde{A}_{ij} / \tilde{W}^2 + 1/3 (Khat + 2 \Theta) gamma_{ij}
        double Atdd[6] = { 
              s(ATXX_), s(ATXY_), s(ATXZ_),
              s(ATYY_), s(ATYZ_), s(ATZZ_)
        } ;
        double const Khat  = s(KHAT_);
        double const theta = s(THETA_);
        double const Ktr = Khat + 2. * theta ;
        for( int a=0; a<6; ++a ) {
            Kdd[a] = ooWsqr * Atdd[a] + (Ktr) * gdd[a] / 3. ; 
        }
        #endif 
        /**************************************************************************************************/
        /* Compute source terms                                                                           */
        /**************************************************************************************************/
        double tau_src, stilde_src[3] ; 
        grmhd_get_geom_sources(
            alp, betau, gdd, guu, Kdd, dalpha_dx, dgdd_dx, dbetau_dx,
            rho, p, eps, B, z, W,
            &tau_src, &stilde_src
        ) ;
        /**************************************************************************************************/
        /* Add energy source terms                                                                        */
        /**************************************************************************************************/
        state_new(VEC(i,j,k),TAU_,q)     += sqrtg * dt * dtfact * tau_src ;
        /**************************************************************************************************/
        state_new(VEC(i,j,k),SX_,q)      += sqrtg * dt * dtfact * stilde_src[0] ;
        /**************************************************************************************************/
        state_new(VEC(i,j,k),SY_,q)      += sqrtg * dt * dtfact * stilde_src[1] ;
        /**************************************************************************************************/
        state_new(VEC(i,j,k),SZ_,q)      += sqrtg * dt * dtfact * stilde_src[2] ;
        /**************************************************************************************************/
    } ;
    /**
     * @brief Compute GRMHD auxiliary quantities.
     *        This is essentially a call to c2p.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index.
     */
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_auxiliaries(  VEC( const int i 
                        ,      const int j 
                        ,      const int k) 
                        , int64_t q 
                        , grace::device_coordinate_system coords) const 
    {
        using namespace grace ;
        using namespace Kokkos ; 

        double rtp[3] ; 
        coords.get_physical_coordinates_sph(i,j,k,q,rtp) ; 

        auto vars = subview(
              this->_state
            , VEC( i
                 , j
                 , k )
            , ALL()
            , q
        ) ;
        auto aux = subview(
              this->_aux
            , VEC( i
                 , j
                 , k )
            , ALL()
            , q
        ) ; 
        auto Bx = subview(
              this->_stag_state.face_staggered_fields_x
            , VEC( ALL()
                 , ALL()
                 , ALL() )
            , static_cast<size_t>(BSX_)
            , q
        ) ; 
        auto By = subview(
              this->_stag_state.face_staggered_fields_y
            , VEC( ALL()
                 , ALL()
                 , ALL() )
            , static_cast<size_t>(BSY_)
            , q
        ) ;
        auto Bz = subview(
              this->_stag_state.face_staggered_fields_z
            , VEC( ALL()
                 , ALL()
                 , ALL() )
            , static_cast<size_t>(BSZ_)
            , q
        ) ;
        grmhd_cons_array_t cons ;
        cons[DENSL] = vars(DENS_)        ; 
        cons[STXL]  = vars(SX_)          ;
        cons[STYL]  = vars(SY_)          ;
        cons[STZL]  = vars(SZ_)          ;
        cons[TAUL]  = vars(TAU_)         ;
        cons[YESL]  = vars(YESTAR_)      ; 
        cons[ENTSL] = vars(ENTROPYSTAR_) ; 
        cons[BSXL]  = 0.5*(Bx(VEC(i,j,k)) + Bx(VEC(i+1,j,k))) ; 
        cons[BSYL]  = 0.5*(By(VEC(i,j,k)) + By(VEC(i,j+1,k))) ; 
        cons[BSZL]  = 0.5*(Bz(VEC(i,j,k)) + Bz(VEC(i,j,k+1))) ; 
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;
        // Set cell-centered **primitive** B^i
        aux(BX_) = cons[BSXL] / metric.sqrtg() ;
        aux(BY_) = cons[BSYL] / metric.sqrtg() ;
        aux(BZ_) = cons[BSZL] / metric.sqrtg() ;
        c2p_err_t c2p_errors ;
        grmhd_prims_array_t prims ;
        conservs_to_prims<eos_t>(
            cons, prims, metric, this->_eos,
            this->atmo_params, this->excision_params, this->c2p_params, rtp,
            c2p_errors ) ;


        /* Write new prims */
        aux(RHO_)     = prims[RHOL]    ;
        aux(EPS_)     = prims[EPSL]    ;
        aux(PRESS_)   = prims[PRESSL]  ;
        aux(TEMP_)    = prims[TEMPL]   ;
        aux(ENTROPY_) = prims[ENTL]    ;
        aux(YE_)      = prims[YEL]     ;
        aux(ZVECX_)   = prims[ZXL]     ;
        aux(ZVECY_)   = prims[ZYL]     ;
        aux(ZVECZ_)   = prims[ZZL]     ;

        // Conservative rewrites driven by the err's reset bits.
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_DENS) ) {
            // Signed mass-error accumulator (separate diagnostic — see
            // C2P_DENS_ERR_, reset once per timestep in evolve.cpp).
            aux(C2P_DENS_ERR_) += vars(DENS_) - cons[DENSL] ;
            vars(DENS_) = cons[DENSL] ;
        }
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_STILDE) ) {
            for( int ii=0; ii<3; ++ii) {
                vars(SX_+ii)=cons[STXL+ii] ;
            }
        }
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_TAU) ) {
            vars(TAU_)=cons[TAUL];
        }
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_ENTROPY) ) {
            vars(ENTROPYSTAR_) = cons[ENTSL] ;
        }
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_YE) ) {
            vars(YESTAR_) = cons[YESL] ;
        }

        // Pack the c2p_err bits (reset + diagnostic SIG_* + outcome) into
        // aux(C2P_ERR_) with sticky-OR semantics over the timestep. Reset
        // to 0 once per step at the top of evolve(), then OR-accumulated
        // here on every RK substep so the value at step end is the union
        // of failure modes seen across all substages.
        //
        // Storage: c2p_err_t is a bitset_t<C2P_N_ERR=21>, kWords=1, so the
        // full bit pattern lives in words[0]. Reinterpret existing aux
        // value as uint64_t (it was either 0 from the per-step reset or a
        // previously-stored bit pattern from an earlier substep), OR with
        // the new bits, cast back to double. All values stay integer-valued
        // and ≤ 2^21, so the double<->uint64 round-trip is exact.
        {
            uint64_t const prev = static_cast<uint64_t>(aux(C2P_ERR_)) ;
            uint64_t const curr = c2p_errors.words[0] ;
            aux(C2P_ERR_) = static_cast<double>(prev | curr) ;
        }
    };
    /**
     * @brief Compute maximum absolute value eigenspeed.
     * 
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index.
     * @return double Maximum eigenspeed of GRMHD equations.
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_max_eigenspeed( VEC( const int i 
                          ,      const int j 
                          ,      const int k) 
                          , int64_t q ) const 
    {
        /****************************************************/
        /****************************************************/

        using namespace grace; 
        using namespace Kokkos ; 

        /****************************************************/
        /* Get prims */
        grmhd_prims_array_t prims ;
        FILL_PRIMS_ARRAY_ZVEC(prims,this->_aux,q,VEC(i,j,k)) ;
        /* Get metric */
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k));
        /* Get some pointers */
        double * const  betau = metric._beta.data() ;
        double * const  gdd   = metric._g.data()    ;
        double const alp = metric.alp() ; 

        double * const z = &(prims[ZXL]) ; 
        double * const B = &(prims[BXL]) ; 
        double rho    = prims[RHOL]  ; 
        double T      = prims[TEMPL] ;
        double ye     = prims[YEL] ; 
        double eps    = prims[EPSL] ; 
        double press  = prims[PRESSL] ;
        /****************************************************/

        /* Get soundspeed, enthalpy */
        double csnd2, h ; 
        eos_err_t err ; 
        double dummy = _eos.press_h_csnd2__temp_rho_ye( h, csnd2, T, rho, ye, err);

        /* Compute Lorentz factor */
        double W ;
        grmhd_get_W(
            gdd, z, &W
        ) ; 

        /* Compute smallb */
        double smallbu[4] ; 
        double b2;
        grmhd_get_smallbu_smallb2(
            alp,betau,gdd,B,z,W,
            &smallbu,&b2
        ) ;

        /* Compute vtilde */
        double vt[3] ;
        grmhd_get_vtildeu(
            alp, betau, z, W, &vt
        ) ;
        /****************************************************/
        /* Find maximum eigenvalue (amongst all directions) */
        double cmax {0};
        std::array<unsigned int, 3> const metric_comp{ 0, 3, 5 } ;
        for( int idir=0; idir<3; ++idir){
            double cp, cm ;
            grmhd_get_cm_cp(
                alp, betau, rho, press, eps, csnd2, W, b2, vt,
                metric.invgamma(metric_comp[idir]),
                idir,
                &cm, &cp
            ) ;
            cmax = math::max(cmax,math::abs(cp),math::abs(cm)) ; 
        }
        /****************************************************/
        return cmax ; 
        /****************************************************/
        /****************************************************/
    };

 private:
    /***********************************************************************/
    //! Number of reconstructed variables.
    static constexpr unsigned int GRMHD_NUM_RECON_VARS = 10 ; 
    //! Equation of State object.
    eos_t _eos ;    
    //! Parameters for atmosphere
    atmo_params_t atmo_params;
    //! Parameters for excision
    excision_params_t excision_params; 
    //! con2prim parameters 
    c2p_params_t c2p_params ; 
    //! Coordinate helper 
    grace::device_coordinate_system dcoords ; 
    /***********************************************************************/
    /**
     * @brief Compute fluxes for gmrmhd equations.
     * 
     * @tparam idir Direction the fluxes are computed in.
     * @tparam recon_t Type of reconstruction.
     * @tparam riemann_t Type of Riemann solver.
     * @param i zero-offset x cell index.
     * @param j zero-offset y cell index.
     * @param k zero-offset z cell index.
     * @param q quadrant index.
     * @param ngz Number of ghost-zones.
     * @param fluxes Flux array.
     */
    template< int idir
            , typename recon_t
            , typename riemann_t = default_riemann_tag_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    getflux(  VEC( const int i
            ,      const int j
            ,      const int k)
            , const int64_t q
            , grace::flux_array_t const fluxes
            , grace::flux_array_t const vbar /* this is actually eface for GS */
            , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
            , double const dt
            , double const dtfact ) const
    {
        /***********************************************************************/
        /* Initialize reconstructor and riemann solver                         */
        /***********************************************************************/
        recon_t reconstructor{} ; 

        /***********************************************************************/
        /* Define and interpolate metric                                       */
        /***********************************************************************/
        metric_array_t metric_l, metric_r;
        FILL_METRIC_ARRAY( metric_l, this->_state, q
                         , VEC( i-utils::delta(idir,0)
                              , j-utils::delta(idir,1)
                              , k-utils::delta(idir,2))) ; 
        FILL_METRIC_ARRAY( metric_r, this->_state, q
                         , VEC( i
                              , j
                              , k )) ;
        /***********************************************************************/
        /* 4-point Lagrange interpolation of the metric at the cell interface, */
        /* with pair-symmetric summation so the face values are bit-mirror     */
        /* under the discrete symmetries (see grmhd_helpers.hh).               */
        /***********************************************************************/
        auto const metric_face = compute_face_metric(this->_state, VEC(i,j,k), q, idir);
        /***********************************************************************/
        /*              Reconstruct primitive variables                        */
        /***********************************************************************/
        /* Indices of variables being reconstructed                            */
        /* NB: reconstruction is done on zvec = W v_n                          */
        /*     to avoid getting acausal velocities at the                      */
        /*     interface.                                                      */
        /***********************************************************************/
        /* Thermo primitive reconstructed at faces is selected at configure   */
        /* time via GRACE_RECON_THERMO (TEMP|PRESS).  TEMP is the GRACE       */
        /* default (cold-K-preserving on polytropic equilibria); PRESS is     */
        /* continuous across contact discontinuities.                         */
        #if defined(GRACE_RECON_THERMO_PRESS)
        constexpr int recon_thermo_idx_aux   = PRESS_ ;
        constexpr int recon_thermo_idx_local = PRESSL ;
        #else
        constexpr int recon_thermo_idx_aux   = TEMP_ ;
        constexpr int recon_thermo_idx_local = TEMPL ;
        #endif
        std::array<int, GRMHD_NUM_RECON_VARS>
            recon_indices{
                  RHO_
                , ZVECX_
                , ZVECY_
                , ZVECZ_
                , YE_
                , recon_thermo_idx_aux
                , ENTROPY_
                , BX_
                , BY_
                , BZ_
            } ;
        /* Local indices in prims array (note z^k -> v^k) */
        std::array<int, GRMHD_NUM_RECON_VARS>
            recon_indices_loc{
                  RHOL
                , ZXL
                , ZYL
                , ZZL
                , YEL
                , recon_thermo_idx_local
                , ENTL
                , BXL
                , BYL
                , BZL
            } ;
        /* Reconstruction                                  */
        grmhd_prims_array_t primL, primR ;
        #pragma unroll GRMHD_NUM_RECON_VARS
        for( int ivar=0; ivar<GRMHD_NUM_RECON_VARS; ++ivar) {
            auto u = Kokkos::subview( this->_aux
                                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                                    , recon_indices[ivar]
                                    , q ) ;
            reconstructor( u, VEC(i,j,k)
                         , primL[recon_indices_loc[ivar]]
                         , primR[recon_indices_loc[ivar]]
                         , idir) ;
        }
        /***********************************************************************/
        /* Donor-cell override near excised cells.                             *
         * If either of the two cells adjacent to the face sits in the         *
         * excised region (α ≤ α_ex), discard the high-order reconstruction    *
         * and use the cell-center primitives on each side. Recon stencils     *
         * reading inside the horizon return sentinel/garbage state; donor     *
         * cell from the live side is the standard cure.                       *
         ***********************************************************************/
        if ( metric_l.alp() <= excision_params.alp_ex
          || metric_r.alp() <= excision_params.alp_ex ) {
            int const i_L = i - utils::delta(idir, 0) ;
            int const j_L = j - utils::delta(idir, 1) ;
            int const k_L = k - utils::delta(idir, 2) ;
            #pragma unroll GRMHD_NUM_RECON_VARS
            for (int ivar = 0; ivar < GRMHD_NUM_RECON_VARS; ++ivar) {
                primL[recon_indices_loc[ivar]] =
                    this->_aux(VEC(i_L, j_L, k_L), recon_indices[ivar], q) ;
                primR[recon_indices_loc[ivar]] =
                    this->_aux(VEC(i,   j,   k  ), recon_indices[ivar], q) ;
            }
        }
        /***********************************************************************/
        /* Replace B^d_L/R with face staggered                                 */
        /***********************************************************************/
        if constexpr ( idir == 0 ) {
            primL[BXL] = primR[BXL] = this->_stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) / metric_face.sqrtg() ; 
        } else if constexpr ( idir == 1 ) {
            primL[BYL] = primR[BYL] = this->_stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) / metric_face.sqrtg(); 
        } else {
            primL[BZL] = primR[BZL] = this->_stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) / metric_face.sqrtg(); 
        }
        /***********************************************************************/
        /* EOS: derive remaining thermo state from the reconstructed thermo   */
        /* primitive.  T-recon path uses press_eps_csnd2__temp_rho_ye (forward */
        /* hook, returns P from T).  P-recon path uses                         */
        /* eps_h_csnd2_temp_entropy__press_rho_ye (inverse hook, returns eps   */
        /* from P and fills T).  Tabulated EOS currently aborts in the inverse */
        /* hook — switch to TEMP recon if you need tabulated EOS.              */
        /***********************************************************************/
        eos_err_t eos_err;
        #if defined(GRACE_RECON_THERMO_PRESS)
        {
            /* Dummies for outputs the recon path doesn't consume.  Entropy   */
            /* in particular is *reconstructed*; we must not let the EOS      */
            /* clobber primL/R[ENTL] with the EOS-derived value, since        */
            /* grmhd_get_fluxes consumes the reconstructed entropy for the    */
            /* ENTSL flux (same as the T-recon path).                         */
            double h_dummy_L, h_dummy_R, ent_dummy_L, ent_dummy_R ;
            primL[EPSL] = _eos.eps_h_csnd2_temp_entropy__press_rho_ye(
                h_dummy_L, primL[CS2L], primL[TEMPL], ent_dummy_L,
                primL[PRESSL], primL[RHOL], primL[YEL], eos_err
            );
            primR[EPSL] = _eos.eps_h_csnd2_temp_entropy__press_rho_ye(
                h_dummy_R, primR[CS2L], primR[TEMPL], ent_dummy_R,
                primR[PRESSL], primR[RHOL], primR[YEL], eos_err
            );
        }
        #else
        {
            primL[PRESSL] = _eos.press_eps_csnd2__temp_rho_ye(
                primL[EPSL], primL[CS2L], primL[TEMPL], primL[RHOL], primL[YEL], eos_err
            );
            primR[PRESSL] = _eos.press_eps_csnd2__temp_rho_ye(
                primR[EPSL], primR[CS2L], primR[TEMPL], primR[RHOL], primR[YEL], eos_err
            );
        }
        #endif
        // Compute Riemann-solver fluxes.
        grmhd_cons_array_t f_HLL ;
        #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS
        std::array<double,2> vb_HLL  ;
        #else
        std::array<double,4> vb_HLL ;
        #endif
        /***********************************************************************/
        /* Riemann solver dispatch on tag.  default_riemann_tag_t matches the  */
        /* build-time GRACE_GRMHD_USE_* choice, so the main flux pass is       */
        /* unchanged.  FOFC calls this kernel with llf_riemann_tag_t to force  */
        /* a Rusanov flux at faces of flagged cells, irrespective of build.    */
        /***********************************************************************/
        if constexpr (std::is_same_v<riemann_t, hlld_riemann_tag_t>) {
            /* ADV: HLLD (MHD) / HLLC (hydro) with HLLE fallback.              */
            int solver_used = 0;
            bool flux_computed = compute_mhd_fluxes_hlld<idir>(
                primL, primR, metric_face, f_HLL, vb_HLL, &solver_used
            );
            if (!flux_computed) {
                compute_mhd_fluxes<idir>(primL, primR, metric_face, f_HLL, vb_HLL);
            }
        } else if constexpr (std::is_same_v<riemann_t, llf_riemann_tag_t>) {
            /* Local Lax-Friedrichs (Rusanov): symmetric HLL with              *
             * cmin = cmax = max(|c±_L|, |c±_R|) — the largest local fast-     *
             * magnetosonic speed.                                             */
            compute_mhd_fluxes<idir, /*rusanov=*/true>(
                primL, primR, metric_face, f_HLL, vb_HLL
            ) ;
        } else {
            /* hll_riemann_tag_t — plain HLLE.                                 */
            compute_mhd_fluxes<idir>( primL, primR, metric_face, f_HLL, vb_HLL ) ;
        }
        /***********************************************************************/
        /* Direct write of the Riemann-solver flux into the persistent fluxes  */
        /* buffer. Positivity is no longer enforced here at recon time         */
        /* (the old GRMHD_USE_PPLIM blend was retired in favour of FOFC); the  */
        /* FOFC pass downstream catches cells that would need flooring and    */
        /* recomputes the relevant face fluxes with donor+LLF.                 */
        /***********************************************************************/
        fluxes(VEC(i,j,k),DENS_,idir,q)        = f_HLL[DENSL] ;
        fluxes(VEC(i,j,k),YESTAR_,idir,q)      = f_HLL[YESL] ;
        fluxes(VEC(i,j,k),ENTROPYSTAR_,idir,q) = f_HLL[ENTSL] ;
        fluxes(VEC(i,j,k),TAU_,idir,q)         = f_HLL[TAUL] ;
        fluxes(VEC(i,j,k),SX_,idir,q)          = f_HLL[STXL] ;
        fluxes(VEC(i,j,k),SY_,idir,q)          = f_HLL[STYL] ;
        fluxes(VEC(i,j,k),SZ_,idir,q)          = f_HLL[STZL] ;
        /***********************************************************************/
        #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS
        // fill emf array
        vbar(VEC(i,j,k),0,idir,q) = vb_HLL[0] ;
        vbar(VEC(i,j,k),1,idir,q) = vb_HLL[1] ;
        #else
	    // fill vbar and cmin/max for later
        vbar(VEC(i,j,k),0,idir,q) = vb_HLL[0] ;
        vbar(VEC(i,j,k),1,idir,q) = vb_HLL[1] ;
        vbar(VEC(i,j,k),2,idir,q) = vb_HLL[2] ;
        vbar(VEC(i,j,k),3,idir,q) = vb_HLL[3] ;
        #endif
    }
    template< size_t idir
            , bool use_rusanov_speeds = false >
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    void compute_mhd_fluxes( grmhd_prims_array_t& primL
                           , grmhd_prims_array_t& primR
                           , metric_array_t const& metric_face
                           , grmhd_cons_array_t& f
                           #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS
                           , std::array<double,2>& emf
                           #else
                           , std::array<double,4>& vbar
                           #endif
                           ) const
    {
        /***********************************************************************/
        hll_riemann_solver_t solver     {} ;
        /***********************************************************************/
        /* Get some pointers                                                   */
        /***********************************************************************/
        double const * const gdd   = metric_face._g.data();
        double const * const guu   = metric_face._ginv.data();
        double const * const betau = metric_face._beta.data(); 
        double const alp           = metric_face.alp() ; 
        double const sqrtg         = metric_face.sqrtg() ; 
        double const * const zl    = &(primL[ZXL]) ; 
        double const * const zr    = &(primR[ZXL]) ; 
        double const * const Bl    = &(primL[BXL]) ; 
        double const * const Br    = &(primR[BXL]) ; 
        double& rhol          = primL[RHOL]   ; 
        double& rhor          = primR[RHOL]   ; 
        double& sl            = primL[ENTL]   ; 
        double& sr            = primR[ENTL]   ;
        double& tl            = primL[TEMPL]  ;
        double& tr            = primR[TEMPL]  ;
        double& yel           = primL[YEL]    ;
        double& yer           = primR[YEL]    ;
        double const epsl = primL[EPSL], epsr = primR[EPSL];
        double const pl   = primL[PRESSL], pr = primR[PRESSL] ; 
        double const cs2l = primL[CS2L], cs2r = primR[CS2L] ;  
        /***********************************************************************/
        /* Compute W on both sides                                             */
        /***********************************************************************/
        double wl,wr ;
        grmhd_get_W(gdd, zl, &wl) ; 
        grmhd_get_W(gdd, zr, &wr) ; 
        /***********************************************************************/
        /* Compute b and b2 on both sides                                      */
        /***********************************************************************/
        double smallbl[4], smallbr[4] ; 
        double b2l, b2r ;
        grmhd_get_smallbu_smallb2(
            alp, betau, gdd, Bl, zl, wl, &smallbl, &b2l
        ) ;
        grmhd_get_smallbu_smallb2(
            alp, betau, gdd, Br, zr, wr, &smallbr, &b2r
        ) ;
        /***********************************************************************/
        /* Compute vtilde on both sides                                        */
        /***********************************************************************/
        double vtildel[3], vtilder[3] ;
        grmhd_get_vtildeu(
            alp, betau, zl, wl, &vtildel
        ) ;
        grmhd_get_vtildeu(
            alp, betau, zr, wr, &vtilder
        ) ;
        /***********************************************************************/
        /* Compute cm/cp                                                       */
        /***********************************************************************/
        double cmin, cmax ;
        {
            double cpr, cmr, cpl, cml;
            int metric_comps[3] {0, 3, 5} ;
            grmhd_get_cm_cp(
                alp, betau, rhol, pl, epsl, cs2l, wl, b2l, vtildel,
                guu[metric_comps[idir]], idir, &cml, &cpl
            ) ;
            grmhd_get_cm_cp(
                alp, betau, rhor, pr, epsr, cs2r, wr, b2r, vtilder,
                guu[metric_comps[idir]], idir, &cmr, &cpr
            ) ;
            if constexpr ( use_rusanov_speeds ) {
                /* LLF / Rusanov: symmetric diffusion with the largest local  *
                 * |fast magnetosonic speed| of either state. Matches the     *
                 * GRMHD literature. The result is less diffusive than the    *
                 * fixed-c LLF and more diffusive than upwind HLL.            */
                cmax = Kokkos::max(
                    Kokkos::max(Kokkos::fabs(cml), Kokkos::fabs(cmr)),
                    Kokkos::max(Kokkos::fabs(cpl), Kokkos::fabs(cpr))
                ) ;
                cmin = cmax ;
            } else {
                cmin = -Kokkos::min(0., Kokkos::min(cml,cmr)) ;
                cmax =  Kokkos::max(0., Kokkos::max(cpl,cpr)) ;
            }
            /* Add some diffusion in weakly hyperbolic limit */
            if( cmin < 1e-12 and cmax < 1e-12 ) { cmin=1; cmax=1; }
        }
        #if GRACE_EMF_SCHEME != GRACE_EMF_SCHEME_GS
        /* Fill vbar with HLL-averaged vtildes and cmin/cmax for the EMF.    */
        {
            int const jk[3][2] = { {1,2}, {0,2}, {0,1} } ;
            vbar[0] = solver(vtildel[jk[idir][0]],vtilder[jk[idir][0]],0,0,cmin,cmax) ;
            vbar[1] = solver(vtildel[jk[idir][1]],vtilder[jk[idir][1]],0,0,cmin,cmax) ;
            vbar[2] = cmin; vbar[3] = cmax ;
        }
        #endif
        /***********************************************************************/
        /* Compute fluxes and conserved on both sides                          */
        /***********************************************************************/
        double densl, taul, entsl, densr, taur, entsr ; 
        double stl[3], str[3] ; 
        double fdl, ftl, fel, fstl[3] ; 
        double fdr, ftr, fer, fstr[3] ;

        grmhd_get_fluxes(
            alp, betau, gdd, rhol, pl, epsl, zl, sl, wl, b2l, smallbl, vtildel, idir,
            &densl, &taul, &stl, &entsl,
            &fdl, &ftl, &fstl, &fel
        ) ;

        grmhd_get_fluxes(
            alp, betau, gdd, rhor, pr, epsr, zr, sr, wr, b2r, smallbr, vtilder, idir,
            &densr, &taur, &str, &entsr,
            &fdr, &ftr, &fstr, &fer
        ) ;
        /***********************************************************************/
        /* Use Riemann solver                                                  */
        /***********************************************************************/
        f[DENSL] = sqrtg * solver(fdl,fdr,densl,densr,cmin,cmax) ; 
        /***********************************************************************/
        f[ENTSL] = sqrtg * solver(fel,fer,entsl,entsr,cmin,cmax) ;
        /***********************************************************************/
        f[TAUL]  = sqrtg * solver(ftl,ftr,taul,taur,cmin,cmax) ; 
        /***********************************************************************/
        f[STXL] = sqrtg * solver(fstl[0],fstr[0],stl[0],str[0],cmin,cmax) ; 
        f[STYL] = sqrtg * solver(fstl[1],fstr[1],stl[1],str[1],cmin,cmax) ; 
        f[STZL] = sqrtg * solver(fstl[2],fstr[2],stl[2],str[2],cmin,cmax) ; 
        /***********************************************************************/
        f[YESL] = sqrtg * solver(yel*fdl,yer*fdr,yel*densl,yer*densr,cmin,cmax) ;
        /***********************************************************************/
        #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS 
        if constexpr ( idir == 0 ) {
            // cross directions are y, z
            // Ey = Fx(Bz)
            // Ez = - Fx(By)
            double Fbzl = - Bl[0] * vtildel[2] + Bl[2] * vtildel[0] ; 
            double Fbzr = - Br[0] * vtilder[2] + Br[2] * vtilder[0] ; 
            emf[0] = sqrtg * solver(Fbzl, Fbzr, Bl[2], Br[2], cmin,cmax) ; 
            double Fbyl = - Bl[0] * vtildel[1] + Bl[1] * vtildel[0] ;
            double Fbyr = - Br[0] * vtilder[1] + Br[1] * vtilder[0] ;
            emf[1] = - sqrtg * solver(Fbyl, Fbyr, Bl[1], Br[1], cmin,cmax) ; 
        } else if constexpr (idir == 1) {
            // cross directions are x, z
            // Ex = - Fy(Bz)
            // Ez = Fy(Bx)
            double Fbzl = - Bl[1] * vtildel[2] + Bl[2] * vtildel[1] ;
            double Fbzr = - Br[1] * vtilder[2] + Br[2] * vtilder[1] ; 
            emf[0] = - sqrtg * solver(Fbzl,Fbzr,Bl[2],Br[2],cmin,cmax) ; 
            double Fbxl = Bl[0] * vtildel[1] - Bl[1] * vtildel[0] ;
            double Fbxr = Br[0] * vtilder[1] - Br[1] * vtilder[0] ;
            emf[1] = sqrtg * solver(Fbxl,Fbxr,Bl[0],Br[0],cmin,cmax) ; 
        } else {
            // cross directions are x, y
            // Ex = Fz(By)
            // Ey = - Fz(Bx)
            double Fbyl = Bl[1] * vtildel[2] - Bl[2] * vtildel[1] ; 
            double Fbyr = Br[1] * vtilder[2] - Br[2] * vtilder[1] ; 
            emf[0] = sqrtg * solver(Fbyl,Fbyr,Bl[1],Br[1],cmin,cmax) ; 
            double Fbxl = Bl[0] * vtildel[2] - Bl[2] * vtildel[0] ;
            double Fbxr = Br[0] * vtilder[2] - Br[2] * vtilder[0] ; 
            emf[1] = - sqrtg * solver(Fbxl,Fbxr,Bl[0],Br[0],cmin,cmax) ; 
        }
        #endif 
    }
    /***********************************************************************/
    /* HLLD/HLLC ADV Riemann solver: not implemented in this build.         */
    /*                                                                      */
    /* Dispatch harness is present (cmake selector, riemann tag, dispatch   */
    /* arm in getflux) but the implementation will land in a follow-up      */
    /* release.  Build with -DGRACE_RIEMANN_SOLVER=HLL or =LLF; ADV will    */
    /* compile but `Kokkos::abort` at the first face flux evaluation.       */
    /***********************************************************************/
    template< size_t idir >
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    bool compute_mhd_fluxes_hlld( [[maybe_unused]] grmhd_prims_array_t& primL
                                , [[maybe_unused]] grmhd_prims_array_t& primR
                                , [[maybe_unused]] metric_array_t const& metric_face
                                , [[maybe_unused]] grmhd_cons_array_t& f
                                #if GRACE_EMF_SCHEME == GRACE_EMF_SCHEME_GS
                                , [[maybe_unused]] std::array<double,2>& emf
                                #else
                                , [[maybe_unused]] std::array<double,4>& emf
                                #endif
                                , [[maybe_unused]] int* out_solver = nullptr
                                ) const
    {
        Kokkos::abort(
            "compute_mhd_fluxes_hlld: ADV (HLLD/HLLC) Riemann solver is not "
            "implemented in this build.  Rebuild with "
            "-DGRACE_RIEMANN_SOLVER=HLL or =LLF."
        );
        return false;
    }
    /***********************************************************************/
} ;
/***********************************************************************/
template< typename eos_t >
void set_grmhd_initial_data() ; 
/***********************************************************************/
void set_conservs_from_prims() ;
/***********************************************************************/
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)        \
extern template                          \
void set_grmhd_initial_data<EOS>( )

INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::tabulated_cold_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE
/***********************************************************************/
}

#endif /*GRACE_PHYSICS_GRMHD_HH*/
