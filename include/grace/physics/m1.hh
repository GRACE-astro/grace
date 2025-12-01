/**
 * @file m1.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2024-11-21
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
#ifndef GRACE_PHYSICS_M1_HH
#define GRACE_PHYSICS_M1_HH

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
#include <grace/utils/reconstruction.hh>
#include <grace/utils/weno_reconstruction.hh>
#include <grace/utils/riemann_solvers.hh>
#include <grace/utils/advanced_riemann_solvers.hh>
#include <grace/physics/m1_helpers.hh>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace grace {
//**************************************************************************************************/ 
//**************************************************************************************************
/**
 * @brief M1 equations system.
 * \ingroup physics 
 */
//**************************************************************************************************/
struct m1_equations_system_t
    : public hrsc_evolution_system_t<m1_equations_system_t>
{
    private:
    //! Base class type 
    using base_t = hrsc_evolution_system_t<m1_equations_system_t>;

    public:

    m1_equations_system_t(grace::var_array_t state_
                        , grace::staggered_variable_arrays_t stag_state_
                        , grace::var_array_t aux_ )
    : base_t(state_,stag_state_,aux_)
    {} ; 

    m1_equations_system_t(grace::var_array_t state_
                        , grace::staggered_variable_arrays_t stag_state_
                        , grace::var_array_t aux_ 
                        , m1_atmo_params_t _atmo_pars 
                        , m1_excision_params_t _excision_pars )
    : base_t(state_,stag_state_,aux_)
    , atmo_params(_atmo_pars)
    , excision_params(_excision_pars)
    {} ;

    /**
     * @brief Compute M1 fluxes in direction \f$x^1\f$
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
        getflux<0,riemann_t,recon_t>(VEC(i,j,k),q,fluxes,dx,dt,dtfact);
    }
    /**
     * @brief Compute M1 fluxes in direction \f$x^2\f$
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
        getflux<1,riemann_t,recon_t>(VEC(i,j,k),q,fluxes,dx,dt,dtfact);
    }
    /**
     * @brief Compute M1 fluxes in direction \f$x^3\f$
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
        getflux<2,riemann_t,recon_t>(VEC(i,j,k),q,fluxes,dx,dt,dtfact);
    }
    

    /**
     * @brief Compute geometric source terms for M1 equations.
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
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;
        /**************************************************************************************************/
        // construct closure and get pressure 
        m1_prims_array_t prims ; 
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,VEC(i,j,k)) ; 
        prims[ERADL] /= metric.sqrtg() ; 
        prims[FXL] /= metric.sqrtg(); 
        prims[FYL] /= metric.sqrtg(); 
        prims[FZL] /= metric.sqrtg(); 

        m1_closure_t cl{prims,metric} ;
        cl.update_closure(0.) ; 
        cl.compute_pressure() ; 
        auto const& PUU = cl.PUU ; 
        /**************************************************************************************************/
        /* Indices for contraction of P^{ij} onto \partial_i \beta_j (see E source below)                 */  
        int spatial_index[3][3] = {
            {VEC(0,1,2)},
            {VEC(1,3,4)},
            {VEC(2,4,5)},
        };
        /**************************************************************************************************/
        double E_source{0.} ; 
        /**************************************************************************************************/
        for( int idir=0; idir<3; ++idir) {
            /**************************************************************************************************/
            /**************************************************************************************************/
            /* Read metric components at neighor cell centres for metric derivative                           */
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
            /**************************************************************************************************/
            /* Compute metric derivatives                                                                     */
            /* We need \partial_i \gamma_{ij} and \partial_i \alpha                                           */
            /**************************************************************************************************/
            std::array<double, 6> dgab_dxi  ;

            /**************************************************************************************************/
            /* Compute metric derivative (factor of 1./dx introduced after)                                 */
            #pragma unroll 6
            for( int ii=0; ii<6; ++ii) { 
                dgab_dxi[ii] =  0.5*(metric_p.gamma(ii) - metric_m.gamma(ii)) ;
            }
            /**************************************************************************************************/
            /* Compute lapse derivative (factor of 1./dx introduced after)                                    */
            double const dalp_dxi =  0.5*(metric_p.alp() - metric_m.alp()) ;
            double const PAB_dgab_dxi = metric.contract_sym2tens_sym2tens(dgab_dxi,PUU) ;
            #ifdef GRACE_ENABLE_COWLING_METRIC
            /**************************************************************************************************/
            /* In Cowling approx we can use the simpler form of the source term which reads                   */
            /* S_{E} +=  1/2 P^{ij} \beta^k \partial_k \gamma_{ij} + P^i_j \partial_i \beta^j                 */
            /**************************************************************************************************/ 
            std::array<double,3> dbetaj_dxi = {
                0.5 * (metric_p.beta(0) - metric_m.beta(0)),
                0.5 * (metric_p.beta(1) - metric_m.beta(1)),
                0.5 * (metric_p.beta(2) - metric_m.beta(2)),
            } ; 

            E_source += idx(idir,q) * (
                0.5 * metric.beta(idir) * PAB_dgab_dxi
              + metric.contract_vec_vec( 
                    dbetaj_dxi
                ,   {PUU[idir][0],PUU[idir][1],PUU[idir][2]})
            ) ; 
            #endif 
            /**************************************************************************************************/
            /* Other piece of the source term                                                                 */
            /* S_{E} += - F^i \partial_i \alpha                                                               */
            /**************************************************************************************************/ 
            E_source -= idx(idir,q) * dalp_dxi * cl.FU[idir] ; 
            /**************************************************************************************************/
            /* S_{F_i} = -E \partial_i alpha + F_k \partial_i \betaˆk                                         */
            /*         + \alpha/2 Pˆ{jk} \partial_i \gamma_{jk}                                               */
            /**************************************************************************************************/
            double F_source = idx(idir,q) * ( -cl.E*dalp_dxi + metric.contract_vec_covec(cl.FD,dbetaj_dxi) 
                                              + 0.5 * metric.alp() *  PAB_dgab_dxi ) ; 
            state_new(VEC(i,j,k),FRADX_+idir,q) += dt * dtfact * metric.sqrtg() * F_source ; 
        }
        state_new(VEC(i,j,k),ERAD_,q) += dt * dtfact * metric.sqrtg() * E_source ; 
    }

    /**
     * @brief Compute M1 auxiliary quantities.
     * 
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
                        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords) const 
    {
        using namespace grace ;
        using namespace Kokkos ; 
        m1_prims_array_t prims ; 
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,VEC(i,j,k)) ; 
        
        metric_array_t metric; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ; 

        prims[ERADL] /= metric.sqrtg() ; 
        prims[FXL] /= metric.sqrtg() ; 
        prims[FYL] /= metric.sqrtg() ; 
        prims[FZL] /= metric.sqrtg() ; 

        m1_closure_t cl{
            prims, metric
        } ; 
        // rescale if superluminal
        if ( cl.F >= cl.E ) {
            double fact = 0.9999 * cl.E / cl.F ; 
            this->_state(VEC(i,j,k),FRADX_,q) *= fact ; 
            this->_state(VEC(i,j,k),FRADY_,q) *= fact ; 
            this->_state(VEC(i,j,k),FRADZ_,q) *= fact ; 
        }

        // Set atmosphere / excision 
        double r = pcoords(VEC(i,j,k),0,q) ; 
        bool excise = excision_params.excise_by_radius 
                ? r <= excision_params.r_ex 
                : metric.alp() <= excision_params.alp_ex ; 
        double E_atmo = atmo_params.E_fl * Kokkos::pow(r,atmo_params.E_fl_scaling) ; 
        if ( cl.E < E_atmo * (1. + 1.e-3 ) 
            or excise ) 
        {
            this->_state(VEC(i,j,k),ERAD_,q) = 
                excise ? metric.sqrtg() * excision_params.E_ex
                       : metric.sqrtg() * E_atmo ; 
            this->_state(VEC(i,j,k),FRADX_,q) = 0.0 ; 
            this->_state(VEC(i,j,k),FRADY_,q) = 0.0 ; 
            this->_state(VEC(i,j,k),FRADZ_,q) = 0.0 ;
        }
        
    }

    /**
     * @brief Compute M1 implicit update.
     * 
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param q Quadrant index.
     */
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_implicit_update( const int q 
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
        /* Read in the metric                                                                             */
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;
        /**************************************************************************************************/
        // read in eas 
        m1_eas_array_t eas ; 
        eas[KAL]  = this->_aux(VEC(i,j,k),KAPPAA_,q) ; 
        eas[KSL]  = this->_aux(VEC(i,j,k),KAPPAS_,q) ; 
        eas[ETAL] = this->_aux(VEC(i,j,k),ETA_,q) ; 
        /**************************************************************************************************/
        // construct closure and update
        m1_prims_array_t prims ; 
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,VEC(i,j,k)) ; 
        prims[ERADL] /= metric.sqrtg() ; 
        prims[FXL] /= metric.sqrtg(); 
        prims[FYL] /= metric.sqrtg(); 
        prims[FZL] /= metric.sqrtg(); 

        m1_closure_t cl{prims,metric} ;
        cl.update_closure(0.) ;
        /**************************************************************************************************/
        // store explicitly updated state 
        double W[4] ; 
        W[0] = prims[ERADL] ; W[1] = prims[FXL] ; W[2] = prims[FYL] ; W[3] = prims[FZL] ;   
        /**************************************************************************************************/
        // construct the initial guess 
        double U[4] ; 
        cl.get_implicit_update_initial_guess(eas, U, dt, dtfact);
        /**************************************************************************************************/
        // construct the lambdas for the evaluation of the update 
        auto const func = [&cl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4]) {
            cl.implicit_update_func(eas,u,W,s,dt,dtfact) ; 
        } ; 
        auto const dfunc = [&cl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4], double (&J)[4][4]) {
            cl.implicit_update_dfunc(eas,u,W,s,J,dt,dtfact) ; 
        } ; 
        /**************************************************************************************************/
        // call rootfinder 
        unsigned long maxiter = 30 ; 
        int err ; 
        utils::rootfind_nd_newton_raphson<4>(
            func, dfunc, U, maxiter, 1e-15, err
        ) ; 
        /**************************************************************************************************/
        if ( err != utils::nr_err_t::SUCCESS ) {
            // assume optically thick closure and 
            // repeat 
            cl = m1_closure_t(prims,metric) ;
            cl.update_closure(0.,false /*nb no update here*/) ;

            cl.get_implicit_update_initial_guess(eas, U, dt, dtfact);

            auto const fixed_closure_func = [&cl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4]) {
                cl.implicit_update_func(eas,u,W,s,dt,dtfact,false) ; 
            } ; 
            auto const fixed_closure_dfunc = [&cl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4], double (&J)[4][4]) {
                cl.implicit_update_dfunc(eas,u,W,s,J,dt,dtfact) ; 
            } ; 
            utils::rootfind_nd_newton_raphson<4>(
                fixed_closure_func, fixed_closure_dfunc, U, maxiter, 1e-15, err
            ) ; 
            // if we failed again we just take a linear step and call it 
            if ( err != utils::nr_err_t::SUCCESS ) {
                cl.update_closure(prims,0,true) ; 
                double J[4][4] ; 
                double S[4] ; 
                cl.get_implicit_jacobian(eas,J) ; 
                for( int i=0; i<4; ++i ) {
                    U[i] = - W[i] ; 
                    for ( int j=0; j<4; ++j) {
                        J[i][j] = dt * dtfact * J[i][j] - (i==j) ; 
                    }
                }
                int piv[5];
                LUPDecompose<4>(J,1e-15,piv) ; 
                LUPSolve(J,piv,U) ; 
            }
        }
        /**************************************************************************************************/
        // write back to the new state 
        state_new(VEC(i,j,k),ERAD_,q)  = metric.sqrtg() * U[0] ; 
        state_new(VEC(i,j,k),FRADX_,q) = metric.sqrtg() * U[1] ;
        state_new(VEC(i,j,k),FRADY_,q) = metric.sqrtg() * U[2] ;
        state_new(VEC(i,j,k),FRADZ_,q) = metric.sqrtg() * U[3] ;
        /**************************************************************************************************/
    }
    private:
    /***********************************************************************/
    //! Number of reconstructed variables.
    static constexpr unsigned int M1_NUM_RECON_VARS = 7 ; 
  
    //! Parameters for atmosphere
    m1_atmo_params_t atmo_params;
    //! Parameters for excision
    m1_excision_params_t excision_params; 
    /***********************************************************************/
    /***********************************************************************/
    /**
     * @brief Compute fluxes for m1 equations.
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
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE void
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
        /* Initialize reconstructor and riemann solver                         */
        /***********************************************************************/
        recon_t reconstructor{} ; 
        /***********************************************************************/
        /* 3rd order interpolation of metric at cell interface                 */
        /***********************************************************************/
        metric_array_t metric_face ; 
        COMPUTE_FCVAL(metric_face,this->_state,i,j,k,q,idir) ; 
        /***********************************************************************/
        /*              Reconstruct primitive variables                        */
        /***********************************************************************/
        std::array<int, 4>
            recon_indices{
                  ERAD_
                , FRADX_
                , FRADY_
                , FRADZ_
            } ; 
        /* Local indices in prims array (note z^k -> v^k) */
        std::array<int, 4>
            recon_indices_loc{
                  ERADL
                , FXL
                , FYL 
                , FZL 
            } ;
        /* Reconstruction                                  */
        m1_prims_array_t primL, primR ; 
        #pragma unroll 4
        for( int ivar=0; ivar<4; ++ivar) {
            auto u = Kokkos::subview( this->_state
                                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()) 
                                    , recon_indices[ivar] 
                                    , q ) ;
            reconstructor( u, VEC(i,j,k)
                         , primL[recon_indices_loc[ivar]]
                         , primR[recon_indices_loc[ivar]]
                         , idir) ;
        }
        // now we need to reconstruct zvec from the hydro 
        std::array<int, 3>
            recon_indices_aux{
                  ZVECX_
                , ZVECY_
                , ZVECZ_
            } ; 
        /* Local indices in prims array (note z^k -> v^k) */
        std::array<int, 3>
            recon_indices_aux_loc{
                  ZXL 
                , ZYL 
                , ZZL
            } ;
        #pragma unroll 3
        for( int ivar=0; ivar<3; ++ivar) {
            auto u = Kokkos::subview( this->_aux
                                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()) 
                                    , recon_indices_aux[ivar] 
                                    , q ) ;
            reconstructor( u, VEC(i,j,k)
                         , primL[recon_indices_aux_loc[ivar]]
                         , primR[recon_indices_aux_loc[ivar]]
                         , idir) ;
        }
        // note that at this stage F is actually F/E, we need to fix that here
        for( int ii=0; ii<3; ++ii) {
            primL[FXL+ii] *= primL[ERADL] ; 
            primR[FXL+ii] *= primR[ERADL] ;  
        }
        // closures 
        m1_closure_t cl{
            primL[ERADL],
            {primL[FXL], primL[FYL], primL[FZL]},
            {primL[ZXL], primL[ZYL], primL[ZZL]},
            metric_face
        }, 
        cr{
            primR[ERADL],
            {primR[FXL], primR[FYL], primR[FZL]},
            {primR[ZXL], primR[ZYL], primR[ZZL]},
            metric_face
        }; 

        cl.update_closure(0) ; cr.update_closure(0) ;  
        cl.compute_pressure(); cr.compute_pressure() ; 
        // compute P^i_j 
        int imap[3][3] = {
            {0,1,2}, {1,3,4}, {2,4,5}
        } ; 
        auto const PUU_l = cl.PUU ; auto const PUU_r = cr.PUU ; 
        auto const PUD_l = metric_face.lower(
            {PUU_l[idir][0], PUU_l[idir][1],PUU_l[idir][2]}
        ) ; 
        auto const PUD_r = metric_face.lower(
            {PUU_r[idir][0], PUU_r[idir][1],PUU_r[idir][2]}
        ) ; 
        // compute the A factor for asymptotic flux correction 
        double const kappa = this->_aux(VEC(i,j,k),KAPPAA_,q) + this->_aux(VEC(i,j,k),KAPPAS_,q) + 1e-20 ; 
        double const A = fmin(1., 1. / dx(idir,q) / kappa ) ; 
        // compute one component of the upper-index flux for the E flux 
        
        double FUd_l = metric_face.invgamma(imap[idir][0]) * primL[FXL] 
                     + metric_face.invgamma(imap[idir][1]) * primL[FYL]
                     + metric_face.invgamma(imap[idir][2]) * primL[FZL] ; 
        double FUd_r = metric_face.invgamma(imap[idir][0]) * primR[FXL] 
                     + metric_face.invgamma(imap[idir][1]) * primR[FYL]
                     + metric_face.invgamma(imap[idir][2]) * primR[FZL] ; 

        // compute wave speeds 
        double cmin, cmax ; 
        double cpr, cmr, cpl, cml;
        compute_cp_cm<idir>(cpl,cml, cl, metric_face) ;
        compute_cp_cm<idir>(cpr,cmr, cr, metric_face) ; 
        cmin = -Kokkos::min(0., Kokkos::min(cml,cmr)) ; 
        cmax =  Kokkos::max(0., Kokkos::max(cpl,cpr)) ; 
        /* Add some diffusion in weakly hyperbolic limit */
        if( cmin < 1e-12 and cmax < 1e-12 ) { cmin=1; cmax=1; }

        // compute the fluxes 
        // E 
        double E_l = primL[ERADL] * metric_face.sqrtg() ;
        double E_r = primR[ERADL] * metric_face.sqrtg() ; 
        double f_E_l = metric_face.sqrtg() * (metric_face.alp() * FUd_l - metric_face.beta(idir) * primL[ERADL]) ; 
        double f_E_r = metric_face.sqrtg() * (metric_face.alp() * FUd_r - metric_face.beta(idir) * primR[ERADL]) ; 
        fluxes(VEC(i,j,k),ERAD_,idir,q) = (cmax*f_E_l + cmin*f_E_r - A * cmax * cmin * (E_r-E_l))/(cmax+cmin) ; 
        // Fx 
        double Fx_l = primL[FXL] * metric_face.sqrtg() ;
        double Fx_r = primR[FXL] * metric_face.sqrtg() ;
        double f_Fx_l = metric_face.sqrtg() * (metric_face.alp() * PUD_l[0] - metric_face.beta(idir) * primL[FXL]) ; 
        double f_Fx_r = metric_face.sqrtg() * (metric_face.alp() * PUD_r[0] - metric_face.beta(idir) * primR[FXL]) ; 
        fluxes(VEC(i,j,k),FRADX_,idir,q) = (SQR(A)*(cmax*f_Fx_l + cmin*f_Fx_r) - A * cmax * cmin * (Fx_r-Fx_l))/(cmax+cmin) 
                                    + (1-SQR(A)) * 0.5 * (f_Fx_l+f_Fx_r); 
        // Fy 
        double Fy_l = primL[FYL] * metric_face.sqrtg() ;
        double Fy_r = primR[FYL] * metric_face.sqrtg() ;
        double f_Fy_l = metric_face.sqrtg() * (metric_face.alp() * PUD_l[1] - metric_face.beta(idir) * primL[FYL]) ; 
        double f_Fy_r = metric_face.sqrtg() * (metric_face.alp() * PUD_r[1] - metric_face.beta(idir) * primR[FYL]) ;
        fluxes(VEC(i,j,k),FRADY_,idir,q) = (SQR(A)*(cmax*f_Fy_l + cmin*f_Fy_r) - A * cmax * cmin * (Fy_r-Fy_l))/(cmax+cmin) 
                                    + (1-SQR(A)) * 0.5 * (f_Fy_l+f_Fy_r); 
        // Fz 
        double Fz_l = primL[FZL] * metric_face.sqrtg() ;
        double Fz_r = primR[FZL] * metric_face.sqrtg() ;
        double f_Fz_l = metric_face.sqrtg() * (metric_face.alp() * PUD_l[2] - metric_face.beta(idir) * primL[FZL]) ; 
        double f_Fz_r = metric_face.sqrtg() * (metric_face.alp() * PUD_r[2] - metric_face.beta(idir) * primR[FZL]) ;
        fluxes(VEC(i,j,k),FRADZ_,idir,q) = (SQR(A)*(cmax*f_Fz_l + cmin*f_Fz_r) - A * cmax * cmin * (Fz_r-Fz_l))/(cmax+cmin) 
                                    + (1-SQR(A)) * 0.5 * (f_Fz_l+f_Fz_r); 
    }

    template< size_t idir >
    GRACE_HOST_DEVICE void compute_cp_cm(
        double& cp, double &cm, m1_closure_t const& cl, metric_array_t const& metric
    ) const 
    {

        int const icomp = (idir==0)*0 + (idir==1)*3 + (idir==2)*5 ;

        double dthin = cl.chi * 1.5 - 0.5 ; 
        double dthick = 1.5 - cl.chi * 1.5 ;

        m1_wavespeeds(
            dthin, dthick, metric.alp(),
            cl.F, cl.W, metric.beta(idir),
            cl.FU[idir], metric.invgamma(icomp),
            cl.vU[idir], &cp, &cm
        ) ; 

    }

} ; 

/**************************************************************************************************/
/* Standalone functions for m1 initial data and eas calculations                                  */
/**************************************************************************************************/
template < typename eos_t >
void set_m1_eas(
      grace::var_array_t& state
    , grace::staggered_variable_arrays_t& sstate
    , grace::var_array_t& aux
) ; 

template < typename eos_t >
void set_m1_eas() ;

template < typename eos_t >
void set_m1_initial_data() ; 

/***********************************************************************/
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS)        \
extern template                          \
void set_m1_initial_data<EOS>( );        \
extern template                          \
void set_m1_eas<EOS>(                    \
      grace::var_array_t&                \
    , grace::staggered_variable_arrays_t&\
    , grace::var_array_t&                \
);                                       \
extern template                          \
void set_m1_eas<EOS>()


INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
/***********************************************************************/
} /* namespace grace */

#endif /*GRACE_PHYSICS_M1_HH*/