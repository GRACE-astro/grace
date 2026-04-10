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
#include "grmhd_subexpressions.hh"
//#include "hllc_subexpressions.hh"
#include "fd_subexpressions.hh"
#include "z4c_subexpressions.hh"
#include <Kokkos_Core.hpp>

#include <type_traits>
#define GRMHD_USE_PPLIM
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
    template< typename recon_t >
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
        getflux<0,recon_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
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
    template< typename recon_t >
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
        getflux<1,recon_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
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
    template< typename recon_t >
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
        getflux<2,recon_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
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
        grmhd_get_smallbu_smallb2(betau,gdd,B,z,W,alp,&smallb,&b2) ; 
        /**************************************************************************************************/
        /* Metric derivatives                                                                             */
        /**************************************************************************************************/
        double dalpha_dx[3], dgdd_dx[18], dbetau_dx[9]; 
        fill_deriv_scalar_4(this->_state, i,j,k, ALP_, q, dalpha_dx, idx(0,q)) ; 
        fill_deriv_vector_4(this->_state, i,j,k, BETAX_, q, dbetau_dx, idx(0,q)) ;
        #ifdef GRACE_ENABLE_COWLING_METRIC
        fill_deriv_tensor_4(this->_state, i,j,k, GXX_, q, dgdd_dx, idx(0,q)) ;
        #else 
        // conformal factor W = 1/gamma^{1/6}
        double Wt  = s(CHI_) ; 
        // 1/W 
        double ooW = 1./Wt ; 
        // 1/W^2
        double ooWsqr = SQR(ooW) ; 
        // dW_dx/y/z 
        double dchi_dx[3] ; 
        fill_deriv_scalar_4(this->_state, i,j,k, CHI_, q, dchi_dx, idx(0,q)) ;
        fill_deriv_tensor_4(this->_state, i,j,k, GTXX_, q, dgdd_dx, idx(0,q)) ;
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
        #ifdef GRACE_ENABLE_COWLING_METRIC
        Kdd[0] = s(KXX_) ; Kdd[1] = s(KXY_) ; Kdd[2] = s(KXZ_) ; 
        Kdd[3] = s(KYY_) ; Kdd[4] = s(KYZ_) ; Kdd[5] = s(KZZ_) ; 
        #else
        double Atdd[6] = { 
              s(ATXX_), s(ATXY_), s(ATXZ_),
              s(ATYY_), s(ATYZ_), s(ATZZ_)
        } ;
        #ifdef GRACE_ENABLE_Z4C_METRIC
        double const Khat  = s(KHAT_);
        double const theta = s(THETA_);
        double const Ktr = Khat + 2. * theta ; 
        #elif defined(GRACE_ENABLE_BSSN_METRIC)
        double const Ktr = s(KTR_) ; 
        #endif 
        for( int a=0; a<6; ++a ) {
            Kdd[a] = ooWsqr * Atdd[a] + (Ktr) * gdd[a] / 3. ; 
        }
        #endif 
        /**************************************************************************************************/
        /* Compute source terms                                                                           */
        /**************************************************************************************************/
        double tau_src, stilde_src[3] ; 
        grmhd_get_geom_sources(
            betau, z, Kdd, dalpha_dx, guu, B, gdd, eps, alp, W, rho, p, dgdd_dx, dbetau_dx, &tau_src, &stilde_src
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
        // this used to return vtilde (TODO: changeme!)   
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

        aux(C2P_ERR_) = 0;
        
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_DENS) ) {
            aux(C2P_ERR_) += Kokkos::fabs(cons[DENSL]-vars(DENS_));
            vars(DENS_) = cons[DENSL] ; 
        }
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_STILDE) ) {
            for( int ii=0; ii<3; ++ii) {
                aux(C2P_ERR_) += Kokkos::fabs(cons[STXL+ii]-vars(SX_+ii));
                vars(SX_+ii)=cons[STXL+ii] ; 
            }
        }
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_TAU) ) {
            aux(C2P_ERR_) += Kokkos::fabs(cons[TAUL]-vars(TAU_));
            vars(TAU_)=cons[TAUL];
        }
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_ENTROPY) ) {
            vars(ENTROPYSTAR_) = cons[ENTSL] ; 
        }
        if ( c2p_errors.test(c2p_err_enum_t::C2P_RESET_YE) ) {
            aux(C2P_ERR_) += Kokkos::fabs(cons[YESL]-vars(YESTAR_));
            vars(YESTAR_) = cons[YESL] ; 
        }
        /* Compute W */
        double const W = Kokkos::sqrt(1.+metric.square_vec({prims[ZXL],prims[ZYL],prims[ZZL]})) ;
        /* Compute smallb2 */
        double smallbu[4] ; 
        double b2 ;
        grmhd_get_smallbu_smallb2(
            metric._beta.data(), metric._g.data(),
            &(prims[BXL]), &(prims[ZXL]), W, metric.alp(),
            &smallbu, &b2 
        ) ; 
        aux(SMALLB2_) = b2 ; 
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
            betau,gdd,B,z,W,alp,
            &smallbu,&b2
        ) ; 

        /* Compute vtilde */
        double vt[3] ; 
        grmhd_get_vtildeu(
            betau, W, z, alp, &vt
        ) ;
        /****************************************************/
        /* Find maximum eigenvalue (amongst all directions) */
        double cmax {0}; 
        std::array<unsigned int, 3> const metric_comp{ 0, 3, 5 } ; 
        for( int idir=0; idir<3; ++idir){ 
            double cp, cm ; 
            grmhd_get_cm_cp(
                csnd2, vt, b2, betau, W, eps, rho, 
                metric.invgamma(metric_comp[idir]),
                alp, press, idir, 
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
            , typename recon_t   >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    getflux(  VEC( const int i 
            ,      const int j 
            ,      const int k)
            , const int64_t q 
            , grace::flux_array_t const fluxes
            , grace::flux_array_t const vbar
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
        /* 3rd order interpolation at cell interface                           */
        /***********************************************************************/
        metric_array_t metric_face ; 
        COMPUTE_FCVAL(metric_face,this->_state,i,j,k,q,idir) ; 
        /***********************************************************************/
        /*              Reconstruct primitive variables                        */
        /***********************************************************************/
        /* Indices of variables being reconstructed                            */
        /* NB: reconstruction is done on zvec = W v_n                          */
        /*     to avoid getting acausal velocities at the                      */
        /*     interface.                                                      */
        /***********************************************************************/
        std::array<int, GRMHD_NUM_RECON_VARS>
            recon_indices{
                  RHO_
                , ZVECX_
                , ZVECY_
                , ZVECZ_
                , YE_
                , TEMP_
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
                , TEMPL
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
        /* Replace B^d_L/R with face staggered                                 */
        /***********************************************************************/
        if constexpr ( idir == 0 ) {
            primL[BXL] = primR[BXL] = this->_stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) / metric_face.sqrtg() ; 
        } else if constexpr ( idir == 1 ) {
            primL[BYL] = primR[BYL] = this->_stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) / metric_face.sqrtg(); 
        } else {
            primL[BZL] = primR[BZL] = this->_stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) / metric_face.sqrtg(); 
        }
        // Compute HLL fluxes
        grmhd_cons_array_t f_HLL ; 
        std::array<double,4> vb_HLL ; 
        compute_mhd_fluxes<idir,true>( primL, primR, metric_face, f_HLL, vb_HLL, 1, 1) ; 
        #ifdef GRMHD_USE_PPLIM
        /***********************************************************************/
        // And LLF fluxes to mix in for positivity preserving limiter 
        /***********************************************************************/
        FILL_PRIMS_ARRAY_ZVEC( primL, this->_aux, q 
                        , VEC( i-utils::delta(idir,0)
                             , j-utils::delta(idir,1)
                             , k-utils::delta(idir,2) )) ;
        FILL_PRIMS_ARRAY_ZVEC( primR, this->_aux, q 
                        , VEC( i
                             , j
                             , k )) ; 
        if constexpr ( idir == 0 ) {
            primL[BXL] = primR[BXL] = this->_stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) / metric_face.sqrtg() ; 
        } else if constexpr ( idir == 1 ) {
            primL[BYL] = primR[BYL] = this->_stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) / metric_face.sqrtg(); 
        } else {
            primL[BZL] = primR[BZL] = this->_stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) / metric_face.sqrtg(); 
        }
        /***********************************************************************/ 
        /*                      Compute LLF flux                               */
        /***********************************************************************/
        grmhd_cons_array_t f_LLF ;
	    std::array<double,4> dummy ; 
        compute_mhd_fluxes<idir,false>( primL, primR, metric_face, f_LLF, dummy, 1., 1.) ;
        /***********************************************************************/
        // Get conserves 
        grmhd_cons_array_t consL, consR ;
        FILL_CONS_ARRAY(consL, this->_state, q 
                     , VEC(   i-utils::delta(idir,0)
                            , j-utils::delta(idir,1)
                            , k-utils::delta(idir,2) ) ) ; 
        FILL_CONS_ARRAY(consR, this->_state, q
                       , VEC(i,j,k)) ; 
        /***********************************************************************/
        // Mix fluxes 
        double const a2CFL = 6. * (dt*dtfact/dx(idir,q)) ; 
        double theta = 1 ; 
        double rho_atm = fmin(atmo_params.rho_fl, excision_params.rho_ex) ; 
        
        double const dens_min_r = rho_atm * metric_r.sqrtg() ; 
        double const dens_min_l = rho_atm * metric_l.sqrtg() ; 

        double const dens_LLF_m = consR[DENSL] + a2CFL * f_LLF[DENSL] ; 
        double const dens_LLF_p = consL[DENSL] - a2CFL * f_LLF[DENSL] ;

        double const dens_m = consR[DENSL] + a2CFL * f_HLL[DENSL] ; 
        double const dens_p = consL[DENSL] - a2CFL * f_HLL[DENSL] ; 

        double theta_p = 1.; 
        double theta_m = 1.; 
        /*
        if(rho_star_m < rho_star_min)
                theta_m = MIN(theta,MAX(0.0,(rho_star_min - rho_star_LLF_m)/(a2CFL*(rho_star_flux[index] - rho_star_flux_LO[index]))));

        if(rho_star_p < rho_star_minm1)
                theta_p = MIN(theta,MAX(0.0,-( rho_star_minm1 - rho_star_LLF_p)/(a2CFL*(rho_star_flux[index] - rho_star_flux_LO[index]))));
        */
        if (dens_m < dens_min_r) {
            theta_m = math::min(theta, math::max(0, (dens_min_r-dens_LLF_m)/(a2CFL*(f_HLL[DENSL]-f_LLF[DENSL])))) ; 
        }
        if ( dens_p < dens_min_l ) {
            theta_p = math::min(theta, math::max(0, -(dens_min_l-dens_LLF_p)/(a2CFL*(f_HLL[DENSL]-f_LLF[DENSL])))) ; 
        }

        theta = math::min(theta_m, theta_p) ;
        if ( std::isnan(theta) ) theta = 1. ; 
        
        //if ( metric_face.alp() < 0.2 ) theta = 0 ; 
        /***********************************************************************/
        /***********************************************************************/
        fluxes(VEC(i,j,k),DENS_,idir,q)        = theta * f_HLL[DENSL]    
                                               + (1. - theta) * f_LLF[DENSL] ; 
        fluxes(VEC(i,j,k),YESTAR_,idir,q)      = theta * f_HLL[YESL]    
                                               + (1. - theta) * f_LLF[YESL] ; 
        fluxes(VEC(i,j,k),ENTROPYSTAR_,idir,q) = theta * f_HLL[ENTSL]    
                                               + (1. - theta) * f_LLF[ENTSL] ; 
        fluxes(VEC(i,j,k),TAU_,idir,q)         = theta * f_HLL[TAUL]    
                                               + (1. - theta) * f_LLF[TAUL] ; 
        fluxes(VEC(i,j,k),SX_,idir,q)          = theta * f_HLL[STXL]    
                                               + (1. - theta) * f_LLF[STXL] ; 
        fluxes(VEC(i,j,k),SY_,idir,q)          = theta * f_HLL[STYL]    
                                               + (1. - theta) * f_LLF[STYL] ; 
        fluxes(VEC(i,j,k),SZ_,idir,q)          = theta * f_HLL[STZL]    
                                               + (1. - theta) * f_LLF[STZL] ; 
        /***********************************************************************/
        #else 
        /***********************************************************************/
        fluxes(VEC(i,j,k),DENS_,idir,q)        = f_HLL[DENSL] ; 
        fluxes(VEC(i,j,k),YESTAR_,idir,q)      = f_HLL[YESL] ; 
        fluxes(VEC(i,j,k),ENTROPYSTAR_,idir,q) = f_HLL[ENTSL] ; 
        fluxes(VEC(i,j,k),TAU_,idir,q)         = f_HLL[TAUL] ; 
        fluxes(VEC(i,j,k),SX_,idir,q)          = f_HLL[STXL] ; 
        fluxes(VEC(i,j,k),SY_,idir,q)          = f_HLL[STYL] ; 
        fluxes(VEC(i,j,k),SZ_,idir,q)          = f_HLL[STZL] ;
        /***********************************************************************/
        #endif
	    // fill vbar and cmin/max for later
        vbar(VEC(i,j,k),0,idir,q) = vb_HLL[0] ; 
        vbar(VEC(i,j,k),1,idir,q) = vb_HLL[1] ; 
        vbar(VEC(i,j,k),2,idir,q) = vb_HLL[2] ; 
        vbar(VEC(i,j,k),3,idir,q) = vb_HLL[3] ; 
    }
    template< size_t idir
            , bool recompute_cp_cm >
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    void compute_mhd_fluxes( grmhd_prims_array_t& primL
                           , grmhd_prims_array_t& primR 
                           , metric_array_t const& metric_face 
                           , grmhd_cons_array_t& f
                           , std::array<double,4>& vbar
                           , double const cmin_loc = 1
                           , double const cmax_loc = 1 ) const 
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

        
        /***********************************************************************/
        /* Compute W on both sides                                             */
        /***********************************************************************/
        double wl,wr ;
        grmhd_get_W(gdd, zl, &wl) ; 
        grmhd_get_W(gdd, zr, &wr) ; 
        /***********************************************************************/
        /* Compute press and cs2 on both sides                                 */
        /***********************************************************************/
        double epsl,epsr,pl,pr,cs2l,cs2r ; 
        eos_err_t eos_err ; 
        pl = _eos.press_eps_csnd2__temp_rho_ye(epsl, cs2l, tl, rhol, yel, eos_err) ; 
        pr = _eos.press_eps_csnd2__temp_rho_ye(epsr, cs2r, tr, rhor, yer, eos_err) ; 
        /***********************************************************************/
        /* Compute b and b2 on both sides                                      */
        /***********************************************************************/
        double smallbl[4], smallbr[4] ; 
        double b2l, b2r ;
        grmhd_get_smallbu_smallb2(
            betau, gdd, Bl, zl, wl, alp, &smallbl, &b2l
        ) ;
        grmhd_get_smallbu_smallb2(
            betau, gdd, Br, zr, wr, alp, &smallbr, &b2r
        ) ;
        /***********************************************************************/
        /* Compute vtilde on both sides                                        */
        /***********************************************************************/
        double vtildel[3], vtilder[3] ; 
        grmhd_get_vtildeu(
            betau, wl, zl, alp, &vtildel
        ) ; 
        grmhd_get_vtildeu(
            betau, wr, zr, alp, &vtilder
        ) ; 
        /***********************************************************************/
        /* Compute cm/cp if needed                                             */
        /***********************************************************************/
        double cmin, cmax ; 
        if constexpr ( recompute_cp_cm ) {
            double cpr, cmr, cpl, cml;
            int metric_comps[3] {0, 3, 5} ; 
            int jk[3][2] = {
                {1,2},
                {0,2},
                {0,1}
            } ; 
            grmhd_get_cm_cp( 
                cs2l, vtildel, b2l, betau, wl, epsl, rhol, guu[metric_comps[idir]],
                alp, pl, idir, &cml, &cpl
            ) ;
            grmhd_get_cm_cp( 
                cs2r, vtilder, b2r, betau, wr, epsr, rhor, guu[metric_comps[idir]],
                alp, pr, idir, &cmr, &cpr
            ) ;
            cmin = -Kokkos::min(0., Kokkos::min(cml,cmr)) ; 
            cmax =  Kokkos::max(0., Kokkos::max(cpl,cpr)) ; 
            /* Add some diffusion in weakly hyperbolic limit */
            if( cmin < 1e-12 and cmax < 1e-12 ) { cmin=1; cmax=1; }
            /* Store cmin/cmax and vtilde for EMF            */
            vbar[0] = solver(vtildel[jk[idir][0]],vtilder[jk[idir][0]],0,0,cmin,cmax) ;
            vbar[1] = solver(vtildel[jk[idir][1]],vtilder[jk[idir][1]],0,0,cmin,cmax) ; 
            vbar[2] = cmin; vbar[3] = cmax ; 
        } else {
            cmin = cmin_loc ; 
            cmax = cmax_loc ; 
        }
        /***********************************************************************/
        /* Compute fluxes and conserved on both sides                          */
        /***********************************************************************/
        double densl, taul, entsl, densr, taur, entsr ; 
        double stl[3], str[3] ; 
        double fdl, ftl, fel, fstl[3] ; 
        double fdr, ftr, fer, fstr[3] ;

        grmhd_get_fluxes(
            wl, rhol, smallbl, b2l, alp, epsl, pl,
            betau, zl, gdd, sl, vtildel, idir,
            &densl, &taul, &stl, &entsl,
            &fdl, &ftl, &fstl, &fel
        ) ; 

        grmhd_get_fluxes(
            wr, rhor, smallbr, b2r, alp, epsr, pr,
            betau, zr, gdd, sr, vtilder, idir,
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
    }
    /***********************************************************************/
    /***********************************************************************/
    #if 0
    template< size_t idir
            , bool recompute_cp_cm >
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    void compute_mhd_fluxes_hllc( grmhd_prims_array_t& primL
                                , grmhd_prims_array_t& primR 
                                , metric_array_t const& metric_face 
                                , grmhd_cons_array_t& f
                                , std::array<double,4>& vbar
                                , double const cmin_loc = 1
                                , double const cmax_loc = 1 ) const 
    {
        /***********************************************************************/
        hll_riemann_solver_t hlle_solver     {} ;
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
        #ifdef GRACE_EVOLVE_YE
        double& yel           = primL[YEL]    ;
        double& yer           = primR[YEL]    ;
        #else 
        double yel            = 0.0           ;
        double yer            = 0.0           ;
        #endif
        
        /***********************************************************************/
        /* Compute W on both sides                                             */
        /***********************************************************************/
        double wl,wr ;
        double vl[3],vr[3];
        hllc_get_W_vel(gdd, zl, &wl, &vl) ; 
        hllc_get_W_vel(gdd, zr, &wr, &vr) ; 
        /***********************************************************************/
        /* Compute press and cs2 on both sides                                 */
        /***********************************************************************/
        double epsl,epsr,pl,pr,cs2l,cs2r ; 
        unsigned int eos_err; 
        pl = _eos.press_eps_csnd2__temp_rho_ye(epsl, cs2l, tl, rhol, yel, eos_err) ; 
        pr = _eos.press_eps_csnd2__temp_rho_ye(epsr, cs2r, tr, rhor, yer, eos_err) ; 
        /***********************************************************************/
        /* Get tetrad                                                          */
        /***********************************************************************/
        int jdir = idir == 0 ? 1 : 0 ; 
        int kdir = idir == 2 ? 1 : 2 ;
        double gamma_dd[3][3] = {
            {gdd[0], gdd[1], gdd[2]},
            {gdd[1], gdd[3], gdd[4]},
            {gdd[2], gdd[4], gdd[5]}
        } ; 
        double gamma_uu[3][3] = {
            {guu[0], guu[1], guu[2]},
            {guu[1], guu[3], guu[4]},
            {guu[2], guu[4], guu[5]}
        } ; 
        double betad[3] = {0.0, 0.0, 0.0} ; 
        for( int a=0; a<3; ++a) {
            for( int b=0; b<3; ++b){
                betad[a] += gamma_dd[a][b] * betau[b] ; 
            }
        }

        double eUU[4][4], edd[4][4] ; 
        hllc_get_tetrad(
            idir,jdir,kdir,
            gamma_uu, gamma_dd,
            betau, betad,
            &(eUU[0]),
            &(eUU[idir+1]),
            &(eUU[jdir+1]),
            &(eUU[kdir+1]),
            &(edd[0]),
            &(edd[idir+1]),
            &(edd[jdir+1]),
            &(edd[kdir+1])
        ) ; 

        /***********************************************************************/
        /* Transform vectors to tetrad frame                                   */
        /***********************************************************************/
        double vhatl[3], Bhatl[3], uhatl[3] ; 
        hllc_transform_vectors(
            alp, betau, Bl, zl, wl,
            edd[0], edd[1], edd[2], edd[3],
            &uhatl, &vhatl, &Bhatl
        ) ;
        double vhatr[3], Bhatr[3], uhatr[3] ; 
        hllc_transform_vectors(
            alp, betau, Br, zr, wr,
            edd[0], edd[1], edd[2], edd[3],
            &uhatr, &vhatr, &Bhatr
        ) ;
        /***********************************************************************/
        /* Get L/R states and fluxes                                           */
        /***********************************************************************/
        double densl, taul, stildel[3], entsl, yesl ; 
        double fdensl, ftaul, fstildel[3], fentsl, fyesl ; 
        hllc_get_state_and_fluxes(
            rhol, pl, epsl, wl, idir,
            vhatl, uhatl, Bhatl,
            &densl, &stildel, &taul,
            &fdensl, &fstildel, &ftaul
        ) ; 
        entsl  = densl * sl   ; 
        yesl   = densl * yel  ; 
        fentsl = fdensl * sl  ; 
        fyesl  = fdensl * yel ; 

        double densr, taur, stilder[3], entsr, yesr ; 
        double fdensr, ftaur, fstilder[3], fentsr, fyesr ; 
        hllc_get_state_and_fluxes(
            rhor, pr, epsr, wr, idir,
            vhatr, uhatr, Bhatr,
            &densr, &stilder, &taur,
            &fdensr, &fstilder, &ftaur
        ) ; 
        entsr  = densr * sr   ; 
        yesr   = densr * yer  ; 
        fentsr = fdensr * sr  ; 
        fyesr  = fdensr * yer ; 

        /***********************************************************************/
        /* Get wavespeeds cmax/cmin                                            */
        /***********************************************************************/
        double cpl, cml ;
        hllc_get_wavespeeds(
            rhol, pl, epsl, cs2l, wl,
            vhatl, uhatl, Bhatl,
            &cml, &cpl
        ) ;
        double cpr, cmr ;
        hllc_get_wavespeeds(
            rhor, pr, epsr, cs2r, wr,
            vhatr, uhatr, Bhatr,
            &cmr, &cpr
        ) ;
        double cmin = - Kokkos::fmin(0.0,Kokkos::fmin(cml,cmr)) ; 
        double cmax =   Kokkos::fmax(0.0,Kokkos::fmax(cpl,cpr)) ; 

        /***********************************************************************/
        /* Get contact speed                                                   */
        /***********************************************************************/
        double fdenshlle = hlle_solver(fdensl,fdensr,densl,densr,cmin,cmax) ;
        double denshlle  = hlle_solver.get_state(fdensl,fdensr,densl,densr,cmin,cmax) ; 
        double ftauhlle  = hlle_solver(ftaul,ftaur,taul,taur,cmin,cmax) ;
        double tauhlle   = hlle_solver.get_state(ftaul,ftaur,taul,taur,cmin,cmax) ; 
        double fstildehlle[3], stildehlle[3] ; 
        for( int a=0; a<3; ++a ) {
            fstildehlle[a] = hlle_solver(fstildel[a],fstilder[a],stildel[a],stilder[a],cmin,cmax) ; 
            stildehlle[a]  = hlle_solver.get_state(fstildel[a],fstilder[a],stildel[a],stilder[a],cmin,cmax) ; 
        }
        double lambda_c ; 
        hllc_get_contact_speed(fstildehlle,stildehlle,ftauhlle,fdenshlle,tauhlle,denshlle,&lambda_c) ; 

        /***********************************************************************/
        /* Get interface speed                                                 */
        /***********************************************************************/
        double lambda_i ; 
        hllc_get_interface_velocity(
            alp, gamma_uu, betau, &lambda_i
        ) ; 

        /***********************************************************************/
        /* Get cL/cR state and fluxes                                          */
        /***********************************************************************/
        double fdenscl, ftaucl, fstildecl[3], fentscl, fyescl ; 
        double denscl, taucl, stildecl[3], entscl, yescl ; 
        hllc_get_central_state_and_fluxes(
            idir, fstildehll, fdenshll, lambda_c, -cmin,
            densl, taul, stildel, vhatl, pl, fdensl, ftaul, fstildel,
            &denscl, &stildecl, &taucl, &fdenscl, &fstildecl, &ftaucl
        ) ; 
        entscl  = entsl * denscl   ; 
        yescl   = yesl * denscl    ; 
        fentscl = entscl * fdenscl ; 
        fyescl  = yescl * fyescl   ; 

        double fdenscr, ftaucr, fstildecr[3], fentscr, fyescr ; 
        double denscr, taucr, stildecr[3], entscr, yescr ; 
        hllc_get_central_state_and_fluxes(
            idir, fstildehll, fdenshll, lambda_c, cmax,
            densr, taur, stilder, vhatr, pr, fdensr, ftaur, fstilder,
            &denscr, &stildecr, &taucr, &fdenscr, &fstildecr, &ftaucr
        ) ; 
        entscr  = entsr * denscr   ; 
        yescr   = yesr * denscr    ; 
        fentscr = entscr * fdenscr ; 
        fyescr  = yescr * fyescr   ; 
        
        /***********************************************************************/
        /* Assemble HLLC fluxes                                                */
        /***********************************************************************/
        double fdens, fstilde[3], ftau, fents, fyes ; 
        double dens, stilde[3], tau, ents, yes ;
        if ( (-cmin) > lambda_i ) {
            // left 
            fdens      = fdensl      ;
            fstilde[0] = fstildel[0] ; 
            fstilde[1] = fstildel[1] ; 
            fstilde[2] = fstildel[2] ;
            ftau       = ftaul       ;
            fents      = fentsl      ; 
            fyes       = fyesl       ;
            dens       = densl       ; 
            stilde[0]  = stildel[0]  ; 
            stilde[1]  = stildel[1]  ; 
            stilde[2]  = stildel[2]  ;
            tau        = taul        ;
            ents       = entsl       ; 
            yes        = yesl        ;
        } else if ( ((-cmin) <= lambda_i) and ( lambda_i < lambda_c )) {
            // center-left 
            fdens      = fdenscl      ;
            fstilde[0] = fstildecl[0] ; 
            fstilde[1] = fstildecl[1] ; 
            fstilde[2] = fstildecl[2] ;
            ftau       = ftaucl       ;
            fents      = fentscl      ; 
            fyes       = fyescl       ;
            dens       = denscl       ; 
            stilde[0]  = stildecl[0]  ; 
            stilde[1]  = stildecl[1]  ; 
            stilde[2]  = stildecl[2]  ;
            tau        = taucl        ;
            ents       = entscl       ; 
            yes        = yescl        ;
        } else if ( (lambda_c <= lambda_i) and (lambda_i < lambda_r) ) {
            // center-right 
            fdens      = fdenscr      ;
            fstilde[0] = fstildecr[0] ; 
            fstilde[1] = fstildecr[1] ; 
            fstilde[2] = fstildecr[2] ;
            ftau       = ftaucr       ;
            fents      = fentscr      ; 
            fyes       = fyescr       ;
            dens       = denscr       ; 
            stilde[0]  = stildecr[0]  ; 
            stilde[1]  = stildecr[1]  ; 
            stilde[2]  = stildecr[2]  ;
            tau        = taucr        ;
            ents       = entscr       ; 
            yes        = yescr        ;
        } else {
            // right 
            fdens      = fdensr      ;
            fstilde[0] = fstilder[0] ; 
            fstilde[1] = fstilder[1] ; 
            fstilde[2] = fstilder[2] ;
            ftau       = ftaur       ;
            fents      = fentsr      ; 
            fyes       = fyesr       ;
            dens       = densr       ; 
            stilde[0]  = stilder[0]  ; 
            stilde[1]  = stilder[1]  ; 
            stilde[2]  = stilder[2]  ;
            tau        = taur        ;
            ents       = entsr       ; 
            yes        = yesr        ;
        }
        /***********************************************************************/
        /* Transform back                                                      */
        /***********************************************************************/
        hllc_transform_fluxes_to_grid_frame(
            alp, idir, eUU, edd, 
            dens, fdens, stilde, fstilde, tau, ftau,
            &(f[DENSL]), &(f[STXL]), &(f[TAUL])
        ) ;
        f[ENTSL] = f[DENSL] * ents ; 
        f[YESL]  = f[DENSL] * yes  ;
    }
    #endif 
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
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE
/***********************************************************************/
}

#endif /*GRACE_PHYSICS_GRMHD_HH*/
