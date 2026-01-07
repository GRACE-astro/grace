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
//#include <grace/utils/advanced_riemann_solvers.hh>
#include "grmhd_subexpressions.hh"

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
    { } ;
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
                            , grace::var_array_t aux_ 
                            , atmo_params_t _atmo_pars
                            , excision_params_t _excision_pars) 
     : base_t(state_,stag_state_,aux_), _eos(eos_), atmo_params(_atmo_pars), excision_params(_excision_pars)
    {} ;
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
    template< typename riemann_t 
            , typename recon_t >
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
        getflux<0,riemann_t,recon_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
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
    template< typename riemann_t 
            , typename recon_t >
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
        getflux<1,riemann_t,recon_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
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
    template< typename riemann_t 
            , typename recon_t >
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
        getflux<2,riemann_t,recon_t>(VEC(i,j,k),q,fluxes,vbar,dx,dt,dtfact);
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
        /**************************************************************************************************/
        /* Convenience indices to make the code slightly less unreadable                                  */
        static constexpr int TT4=0; 
        static constexpr int TX4=1;
        static constexpr int TY4=2;
        static constexpr int TZ4=3;
        static constexpr int XX4=4;
        static constexpr int XY4=5;
        static constexpr int XZ4=6;
        static constexpr int YY4=7;
        static constexpr int YZ4=8;
        static constexpr int ZZ4=9;
        /**************************************************************************************************/
        /* Read in the metric                                                                             */
        /**************************************************************************************************/
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;
        double const alp           = metric.alp()        ; 
        double const * const betau = metric._beta.data() ; 
        double const * const gdd   = metric._g.data()    ;
        double const * const guu   = metric._ginv.data() ;
        /**************************************************************************************************/
        /* Read in the extrinsic curvature                                                                */
        /**************************************************************************************************/
        std::array<double,6> Kij ;
        get_extrinsic_curvature(Kij,this->_state,VEC(i,j,k),q) ; 
        double const * const Kdd = Kij.data() ; 
        /**************************************************************************************************/
        /* Read the primitive variables                                                                   */
        /**************************************************************************************************/
        grmhd_prims_array_t prims ; 
        FILL_PRIMS_ARRAY_ZVEC(prims,this->_aux,q,VEC(i,j,k))   ;
        double const eps   = prims[EPSL]   ; 
        double const rho   = prims[RHOL]   ; 
        double const p     = prims[PRESSL] ;
        double const * const   z = &(prims[ZXL]) ;  
        double const * const   B = &(prims[BXL]) ;  
        /**************************************************************************************************/
        double W ; 
        grmhd_get_W(gdd,z,&W) ; 
        double b2, smallb[4] ; 
        grmhd_get_smallbu_smallb2(betau,gdd,B,z,W,alp,&smallb,&b2) ; 
        /**************************************************************************************************/
        /* Metric derivatives                                                                             */
        /**************************************************************************************************/
        double dalpha_dx[3], dgdd_dx[18], dbetau_dx[9] ; 
        for( int idir=0; idir<3; ++idir) {
            /* Read metric components at neighor cell centres for metric derivative                       */
            metric_array_t metric_m, metric_p ; 
            FILL_METRIC_ARRAY( metric_m, this->_state
                             , q
                             , VEC( i-utils::delta(0,idir)
                                  , j-utils::delta(1,idir)
                                  , k-utils::delta(2,idir)) ) ; 
            FILL_METRIC_ARRAY( metric_p, this->_state
                             , q
                             , VEC( i+utils::delta(0,idir)
                                  , j+utils::delta(1,idir)
                                  , k+utils::delta(2,idir) ) ) ; 
            dalpha_dx[idir] = 0.5 * idx(idir,q) * (metric_p.alp() - metric_m.alp()) ; 
            int icomp=0;
            for ( int ii=0; ii<3; ii++) {
                dbetau_dx[3*idir+ii] = 0.5 * idx(idir,q) * (metric_p.beta(ii) - metric_m.beta(ii)) ; 
                for( int jj=ii; jj<3; ++jj) {
                    dgdd_dx[6*idir + icomp] = 0.5 * idx(idir,q) * (metric_p.gamma(icomp) - metric_m.gamma(icomp)) ; 
                    icomp ++ ; 
                }
            }
        }
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
        state_new(VEC(i,j,k),TAU_,q)     += metric.sqrtg() * dt * dtfact * tau_src ;
        /**************************************************************************************************/
        state_new(VEC(i,j,k),SX_,q)      += metric.sqrtg() * dt * dtfact * stilde_src[0] ;
        /**************************************************************************************************/
        state_new(VEC(i,j,k),SY_,q)      += metric.sqrtg() * dt * dtfact * stilde_src[1] ;
        /**************************************************************************************************/
        state_new(VEC(i,j,k),SZ_,q)      += metric.sqrtg() * dt * dtfact * stilde_src[2] ;
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
            , BSX_
            , q
        ) ; 
        auto By = subview(
              this->_stag_state.face_staggered_fields_y
            , VEC( ALL()
                 , ALL()
                 , ALL() )
            , BSY_
            , q
        ) ;
        auto Bz = subview(
              this->_stag_state.face_staggered_fields_z
            , VEC( ALL()
                 , ALL()
                 , ALL() )
            , BSZ_
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
            this->atmo_params, this->excision_params, rtp,
            c2p_errors ) ;
        
        
        /* Write new prims */
        aux(RHO_) = prims[RHOL]     ; 
        aux(EPS_) = prims[EPSL]     ; 
        aux(PRESS_) = prims[PRESSL] ; 
        aux(TEMP_) = prims[TEMPL]   ; 
        aux(ENTROPY_) = prims[ENTL]  ; 
        aux(YE_)   = prims[YEL]     ;
        aux(ZVECX_) = prims[ZXL] ; 
        aux(ZVECY_) = prims[ZYL] ; 
        aux(ZVECZ_) = prims[ZZL] ; 
        /* Overwrite conserved */
        #if 0
        vars(DENS_)  = cons[DENSL]       ; 
        vars(SX_)    = cons[STXL]        ; 
        vars(SY_)    = cons[STYL]        ;
        vars(SZ_)    = cons[STZL]        ;
        vars(TAU_)   = cons[TAUL]        ;
        vars(YESTAR_) = cons[YESL]       ; 
        vars(ENTROPYSTAR_) = cons[ENTSL] ; 
        #endif

        aux(C2P_ERR_) = 0;
        
        if ( c2p_errors.adjust_d ) {
            aux(C2P_ERR_) += fabs(cons[DENSL]-vars(DENS_));
            vars(DENS_) = cons[DENSL] ; 
        }
        if ( c2p_errors.adjust_s ) {
            for( int ii=0; ii<3; ++ii) {
                aux(C2P_ERR_) += fabs(cons[STXL+ii]-vars(SX_+ii));
                vars(SX_+ii)=cons[STXL+ii] ; 
            }
        }
        if ( c2p_errors.adjust_tau ) {
            aux(C2P_ERR_) += fabs(cons[TAUL]-vars(TAU_));
            vars(TAU_)=cons[TAUL];
        }
        if ( c2p_errors.adjust_ent ) {
            vars(ENTROPYSTAR_) = cons[ENTSL] ; 
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
        unsigned int err ; 
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
    /***********************************************************************/
    #if 0
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
            , typename riemann_t
            , typename recon_t   >
    std::enable_if<std::is_same_v<riemann_t,grace::hllc_riemann_solver_t<idir>>,void>::type
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    getflux(  VEC( const int i 
            ,      const int j 
            ,      const int k)
            , const int64_t q 
            , grace::flux_array_t const fluxes
            , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
            , double const dt 
            , double const dtfact ) const 
    {
        /***********************************************************************/
        /* Initialize reconstructor                                            */
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
        /* 2nd order interpolation at cell interface                           */
        /***********************************************************************/
        metric_array_t const metric_face{
            { 0.5*(metric_l.gamma(0) + metric_r.gamma(0))
            , 0.5*(metric_l.gamma(1) + metric_r.gamma(1))
            , 0.5*(metric_l.gamma(2) + metric_r.gamma(2))
            , 0.5*(metric_l.gamma(3) + metric_r.gamma(3))
            , 0.5*(metric_l.gamma(4) + metric_r.gamma(4))
            , 0.5*(metric_l.gamma(5) + metric_r.gamma(5))}
        ,   { 0.5*(metric_l.beta(0) + metric_r.beta(0))
            , 0.5*(metric_l.beta(1) + metric_r.beta(1))
            , 0.5*(metric_l.beta(2) + metric_r.beta(2))}
        ,   0.5 * (metric_l.alp() + metric_r.alp())
        } ; 
        
        /***********************************************************************/
        /* Initialize Riemann solver                                           */
        /***********************************************************************/
        riemann_t solver     {metric_face} ;

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
            primL[BXL] = this->stag_state_.face_staggered_fields_x(VEC(i,j,k),BSX_,q) ; 
            primR[BXL] = this->stag_state_.face_staggered_fields_x(VEC(i,j,k),BSX_,q) ; 
        } else if constexpr ( idir == 1 ) {
            primL[BYL] = this->stag_state_.face_staggered_fields_y(VEC(i,j,k),BSY_,q) ; 
            primR[BYL] = this->stag_state_.face_staggered_fields_y(VEC(i,j,k),BSY_,q) ; 
        } else {
            primL[BZL] = this->stag_state_.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) ; 
            primR[BZL] = this->stag_state_.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) ; 
        }
        
        /***********************************************************************/
        /* Compute u0 on both sides                                            */
        /***********************************************************************/
        /* Lorentz factors  */
        /* W = sqrt(1+z^2)  */
        double const alp = metric_face.alp() ;
        double const wl   = Kokkos::sqrt(1. 
            + metric_face.square_vec({primL[ZXL], primL[ZYL], primL[ZZL]}));
        double const wr   = Kokkos::sqrt(1. 
            + metric_face.square_vec({primR[ZXL], primR[ZYL], primR[ZZL]}));
        
        /* u^0             */
        double u0_l = wl / alp ; 
        double u0_r = wr / alp ; 

        /***********************************************************************/
        /* Fill up primitive array on both sides of the face.                  */
        /* Right now we have:                                                  */
        /* 1) The correct rho                                                  */
        /* 2) No pressure (computed below)                                     */
        /* 3) The temperature but no eps                                       */
        /* 4) The z vector (W v_{n}^i) as opposed to v^i (swapped below)       */
        /***********************************************************************/
        
        /* Left */
        double cs2l, cs2r ; 
        unsigned int eos_err; 
        primL[PRESSL] = _eos.press_eps_csnd2__temp_rho_ye(primL[EPSL], cs2l, primL[TEMPL], primL[RHOL], primL[YEL], eos_err) ; 

        /* Right */
        primR[PRESSL] = _eos.press_eps_csnd2__temp_rho_ye(primR[EPSL], cs2r, primR[TEMPL], primR[RHOL], primR[YEL], eos_err) ; 

        std::array<double,3> uD_l, uD_r ; 
        solver.transform_velocities_to_tetrad_frame(u0_l, primL, uD_l) ; 
        solver.transform_velocities_to_tetrad_frame(u0_r, primR, uD_r) ; 

        /* Compute specific enthalpies */
        double h_l = 1 + primL[EPSL] + primL[PRESSL]/primL[RHOL] ;
        double h_r = 1 + primR[EPSL] + primR[PRESSL]/primR[RHOL] ;
        
        grmhd_cons_array_t fL, fR, uL, uR; 

        /* Get wavespeeds      */ 
        double cpr, cmr, cpl, cml;
        compute_cp_cm( cpl, cml, cs2l, u0_l, primL[ZXL+idir], 1
                     , 0, 1) ;
        compute_cp_cm( cpr, cmr, cs2r, u0_r, primR[ZXL+idir], 1
                     , 0, 1) ;
        double cmin = -math::min(0., math::min(cml,cmr)) ; 
        double cmax =  math::max(0., math::max(cpl,cpr)) ; 
        /* Add some diffusion in weakly hyperbolic limit */
        if( cmin < 1e-12 and cmax < 1e-12 ) { cmin=1; cmax=1; }
        /***********************************************************************/
        /*                          Get dens flux                              */
        /***********************************************************************/
        double const alpha_sqrtgamma = alp * metric_face.sqrtg() ;
        uL[DENSL] = alp * primL[RHOL] * u0_l ;
        uR[DENSL] = alp * primR[RHOL] * u0_r ;

        fL[DENSL] = uL[DENSL] * primL[ZXL+idir] ; 
        fR[DENSL] = uR[DENSL] * primR[ZXL+idir] ; 

        /***********************************************************************/
        /*                          Get ye_star flux                           */
        /***********************************************************************/
        uL[YESL] = uL[DENSL] * primL[YEL] ; 
        uR[YESL] = uR[DENSL] * primR[YEL] ; 
        
        fL[YESL] = uL[YESL] * primL[ZXL+idir] ; 
        fR[YESL] = uR[YESL] * primR[ZXL+idir] ; 

        /***********************************************************************/
        /*                          Get s_star flux                            */
        /***********************************************************************/
        uL[ENTSL] = uL[DENSL] * primL[ENTL] ; 
        uR[ENTSL] = uR[DENSL] * primR[ENTL] ; 

        fL[ENTSL] = uL[ENTSL] * primL[ZXL+idir] ; 
        fR[ENTSL] = uR[ENTSL] * primR[ZXL+idir] ; 

        /***********************************************************************/ 
        /*                           Get tau flux                              */
        /***********************************************************************/
        double const tau_plus_P_l = uL[DENSL] * ( alp * h_l * u0_l - 1. ) ; 
        double const tau_plus_P_r = uR[DENSL] * ( alp * h_r * u0_r - 1. ) ;
        /***************************************************************************/
        /* \tau = \sqrt{\gamma} D (Wh-P/D-1)                                       */
        /***************************************************************************/
        uL[TAUL] = tau_plus_P_l - primL[PRESSL] ; 
        uR[TAUL] = tau_plus_P_r - primR[PRESSL] ;
        /***************************************************************************/
        /* F^{d}_{\rm tau} = \sqrt{\gamma} (\tau + P) v^d                          */
        /***************************************************************************/
        fL[TAUL] = tau_plus_P_l * primL[ZXL+idir] ;
        fR[TAUL] = tau_plus_P_r * primR[ZXL+idir] ;
        /***********************************************************************/
        /* Momentum flux in direction d for S_j : \alpha \sqrt{\gamma} T^d_j   */
        /***********************************************************************/

        /***********************************************************************/
        /* Get S_x flux                                                        */
        /***********************************************************************/
        
        /***********************************************************************/
        /* F^d_{S_x} = \alpha \sqrt{\gamma} T^d_x                              */
        /*  = \alpha \sqrt{\gamma} ( (\rho h + b^2) u^0 v^d u_x                */
        /*                         + p \delta^d_x - b^d b_x )                  */  
        /***********************************************************************/
        double const D_h_l = uL[DENSL] * h_l ; 
        double const D_h_r = uR[DENSL] * h_r ; 

        uL[STXL] = D_h_l * uD_l[0] ;
        uR[STXL] = D_h_r * uD_r[0] ;

        fL[STXL] = uL[STXL] * primL[ZXL+idir] + primL[PRESSL] * utils::delta(idir,0) ;
        fR[STXL] = uR[STXL] * primR[ZXL+idir] + primR[PRESSL] * utils::delta(idir,0) ;

        /***********************************************************************/
        /* Get S_y flux                                                        */
        /***********************************************************************/

        /***********************************************************************/
        /* F^d_{S_y} = \alpha \sqrt{\gamma} T^d_y                              */
        /*  = \alpha \sqrt{\gamma} ( (\rho h + b^2) u^0 v^d u_y                */
        /*                         + p \delta^d_y - b^d b_y )                  */  
        /***********************************************************************/
        uL[STYL] = D_h_l * uD_l[1] ;
        uR[STYL] = D_h_r * uD_r[1] ;

        fL[STYL] = uL[STYL] * primL[ZXL+idir] + primL[PRESSL] * utils::delta(idir,1) ;
        fR[STYL] = uR[STYL] * primR[ZXL+idir] + primR[PRESSL] * utils::delta(idir,1) ; 

        /***********************************************************************/
        /* Get S_z flux                                                        */
        /***********************************************************************/

        /***********************************************************************/
        /* F^d_{S_z} = \alpha \sqrt{\gamma} T^d_z                              */
        /*  = \alpha \sqrt{\gamma} ( (\rho h + b^2) u^0 v^d u_z                */
        /*                         + p \delta^d_z - b^d b_z )                  */  
        /***********************************************************************/
        uL[STZL] = D_h_l * uD_l[2] ;
        uR[STZL] = D_h_r * uD_r[2] ;

        fL[STZL] = uL[STZL] * primL[ZXL+idir] 
            + primL[PRESSL] * utils::delta(idir,2) ;
        fR[STZL] = uR[STZL] * primR[ZXL+idir] 
            + primR[PRESSL] * utils::delta(idir,2) ; 
        /***********************************************************************/
        grmhd_cons_array_t fHLLC = 
            solver(fL,fR,uL,uR,primL,primR,cmin,cmax) ; 
        /***********************************************************************/
        fluxes(VEC(i,j,k),DENS_,idir,q)        = alpha_sqrtgamma * fHLLC[DENSL] ; 
        fluxes(VEC(i,j,k),YESTAR_,idir,q)      = alpha_sqrtgamma * fHLLC[YESL]  ; 
        fluxes(VEC(i,j,k),ENTROPYSTAR_,idir,q) = alpha_sqrtgamma * fHLLC[ENTSL] ;
        fluxes(VEC(i,j,k),TAU_,idir,q)         = alpha_sqrtgamma * fHLLC[TAUL]  ;
        fluxes(VEC(i,j,k),SX_,idir,q)          = alpha_sqrtgamma * fHLLC[STXL]  ;
        fluxes(VEC(i,j,k),SY_,idir,q)          = alpha_sqrtgamma * fHLLC[STYL]  ;
        fluxes(VEC(i,j,k),SZ_,idir,q)          = alpha_sqrtgamma * fHLLC[STZL]  ;
        /***********************************************************************/
    };
    #endif
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
            , typename riemann_t 
            , typename recon_t   >
    std::enable_if<std::is_same_v<riemann_t,grace::hll_riemann_solver_t>,void>::type 
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
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
            primL[BXL] = this->_stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) / metric_face.sqrtg() ; 
            primR[BXL] = this->_stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) / metric_face.sqrtg(); 
        } else if constexpr ( idir == 1 ) {
            primL[BYL] = this->_stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) / metric_face.sqrtg(); 
            primR[BYL] = this->_stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) / metric_face.sqrtg(); 
        } else {
            primL[BZL] = this->_stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) / metric_face.sqrtg(); 
            primR[BZL] = this->_stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) / metric_face.sqrtg(); 
        }
        // Compute HLL fluxes
        grmhd_cons_array_t f_HLL ; 
        std::array<double,4> vb_HLL ; 
        compute_mhd_fluxes<idir,riemann_t,true>( primL, primR, metric_face, f_HLL, vb_HLL, 1, 1) ; 
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
            primL[BXL] = this->_stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) / metric_face.sqrtg() ; 
            primR[BXL] = this->_stag_state.face_staggered_fields_x(VEC(i,j,k),BSX_,q) / metric_face.sqrtg(); 
        } else if constexpr ( idir == 1 ) {
            primL[BYL] = this->_stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) / metric_face.sqrtg(); 
            primR[BYL] = this->_stag_state.face_staggered_fields_y(VEC(i,j,k),BSY_,q) / metric_face.sqrtg(); 
        } else {
            primL[BZL] = this->_stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) / metric_face.sqrtg(); 
            primR[BZL] = this->_stag_state.face_staggered_fields_z(VEC(i,j,k),BSZ_,q) / metric_face.sqrtg(); 
        }
        /***********************************************************************/ 
        /*                      Compute LLF flux                               */
        /***********************************************************************/
        grmhd_cons_array_t f_LLF ;
	    std::array<double,4> dummy ; 
        compute_mhd_fluxes<idir,riemann_t,false>( primL, primR, metric_face, f_LLF, dummy, 1., 1.) ;
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
            , typename riemann_t 
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
        riemann_t solver     {} ;
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
        grmhd_get_W(gdd, zl, &wl) ; 
        grmhd_get_W(gdd, zr, &wr) ; 
        /***********************************************************************/
        /* Compute press and cs2 on both sides                                 */
        /***********************************************************************/
        double epsl,epsr,pl,pr,cs2l,cs2r ; 
        unsigned int eos_err; 
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
        #ifdef GRACE_EVOLVE_YE
        f[YESL] = sqrtg * solver(yel*fdl,yer*fdr,yel*densl,yer*densr,cmin,cmax) ;
        /***********************************************************************/
        #endif  
    }
    /***********************************************************************/
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
#undef INSTANTIATE_TEMPLATE
/***********************************************************************/
}

#endif /*GRACE_PHYSICS_GRMHD_HH*/
