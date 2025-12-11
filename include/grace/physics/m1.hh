/**
 * @file m1.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-26
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
#include <grace/coordinates/coordinate_systems.hh>
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
        prims[NRADL] /= metric.sqrtg() ; 
        prims[FXL] /= metric.sqrtg(); 
        prims[FYL] /= metric.sqrtg(); 
        prims[FZL] /= metric.sqrtg(); 

        m1_closure_t cl{prims,metric} ;
        cl.update_closure(0.) ; 
        cl.compute_pressure() ; 
        auto const& PUU = cl.PUU ; 
        /**************************************************************************************************/
        // we need derivative of gamma_{ij}, derivatives of beta^i and beta_i, derivatives of 
        // alpha 
        double dalpha[3], dbetau_x[3], dbetau_y[3], dbetau_z[3] ; 
        double dgdd_x[3][3], dgdd_y[3][3], dgdd_z[3][3] ; 
        double *dbetau[3] = {dbetau_x,dbetau_y,dbetau_z};
        double (*dgdd[3])[3] = {dgdd_x, dgdd_y, dgdd_z};
        #pragma unroll 
        for( int idir=0; idir<3; ++idir) {
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
            dalpha[idir] = 0.5*(metric_p.alp() - metric_m.alp()) * idx(idir,q) ;
            int ii=0; 
            for( int j=0; j<3; ++j) {
                dbetau[idir][j] =  0.5*(metric_p.beta(j) - metric_m.beta(j)) * idx(idir,q) ;
                for( int k=j; k<3; ++k) {
                    dgdd[idir][j][k] = dgdd[idir][k][j] = 0.5*(metric_p.gamma(ii) - metric_m.gamma(ii)) * idx(idir,q) ;
                    ii++ ; 
                }
            }
            
        }
        /**************************************************************************************************/
        auto betad = metric.lower(metric._beta) ; 
        double source[4] ; 
        m1_source_terms(
            dgdd_x,dgdd_y,dgdd_z,
            dbetau_x,dbetau_y,dbetau_z,
            dalpha,betad.data(),metric._g.data(),metric._ginv.data(),
            metric.alp(),metric.sqrtg(),
            cl.E, cl.FD.data(), cl.FU.data(), PUU, source
        ) ; 
        
        state_new(VEC(i,j,k),ERAD_,q)  += dt * dtfact * source[0] ;
        state_new(VEC(i,j,k),FRADX_,q) += dt * dtfact * source[1] ;
        state_new(VEC(i,j,k),FRADY_,q) += dt * dtfact * source[2] ;
        state_new(VEC(i,j,k),FRADZ_,q) += dt * dtfact * source[3] ;
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
                        , grace::device_coordinate_system coords) const 
    {
        using namespace grace ;
        using namespace Kokkos ; 

        double rtp[3] ; 
        coords.get_physical_coordinates_sph(i,j,k,q,rtp) ; 


        m1_prims_array_t prims ; 
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,VEC(i,j,k)) ; 
        
        metric_array_t metric; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ; 

        prims[ERADL] /= metric.sqrtg() ;
        prims[NRADL] /= metric.sqrtg() ; 
        prims[FXL] /= metric.sqrtg() ; 
        prims[FYL] /= metric.sqrtg() ; 
        prims[FZL] /= metric.sqrtg() ; 

        m1_closure_t cl{
            prims, metric
        } ; 
        cl.update_closure(0) ; 
        // rescale if superluminal
        if ( cl.F >= cl.E ) {
            double fact = 0.9999 * cl.E / cl.F ; 
            this->_state(VEC(i,j,k),FRADX_,q) *= fact ; 
            this->_state(VEC(i,j,k),FRADY_,q) *= fact ; 
            this->_state(VEC(i,j,k),FRADZ_,q) *= fact ; 
        }
        // compute radiation avg energy 
        double epsilon = cl.J / prims[NRADL] * cl.Gamma ; 
        // Set atmosphere / excision 
        double r = rtp[0] ; 
        bool excise = excision_params.excise_by_radius 
                ? r <= excision_params.r_ex 
                : metric.alp() <= excision_params.alp_ex ; 
        double E_atmo = atmo_params.E_fl * Kokkos::pow(r,atmo_params.E_fl_scaling) ; 
        double eps_atmo = atmo_params.eps_fl * Kokkos::pow(r,atmo_params.eps_fl_scaling) ; 
        if ( cl.E < E_atmo * (1. + 1.e-3 ) ) 
        {
            double atmo_state[4] = {E_atmo,0.0, 0.0, 0.0} ; 
            this->_state(VEC(i,j,k),ERAD_,q)  = metric.sqrtg() * atmo_state[0]; 
            this->_state(VEC(i,j,k),FRADX_,q) = atmo_state[1] ; 
            this->_state(VEC(i,j,k),FRADY_,q) = atmo_state[2] ; 
            this->_state(VEC(i,j,k),FRADZ_,q) = atmo_state[3] ;
            // We set N in order to ensure a sensible average energy 
            cl.update_closure(atmo_state,0,true) ; 
            this->_state(VEC(i,j,k),NRAD_,q)  = metric.sqrtg() * cl.Gamma * cl.J / eps_atmo ; 
            epsilon = eps_atmo ; 
        } else if ( excise ) {
            this->_state(VEC(i,j,k),ERAD_,q)  = metric.sqrtg() * excision_params.E_ex ; 
            this->_state(VEC(i,j,k),FRADX_,q) = 0.0 ; 
            this->_state(VEC(i,j,k),FRADY_,q) = 0.0 ; 
            this->_state(VEC(i,j,k),FRADZ_,q) = 0.0 ;
            // here since we are in excision it's safe to assume v^i == 0 :
            // Gamma == 1 and N = sqrtg E / eps_target 
            this->_state(VEC(i,j,k),NRAD_,q)  = metric.sqrtg() * excision_params.E_ex/excision_params.eps_ex ;
            epsilon = excision_params.eps_ex;  
        } 
        // Finally check epsilon, if out of range 
        // we adjust **only** Nrad
        if ( epsilon < atmo_params.eps_min ) {
            this->_state(VEC(i,j,k),NRAD_,q)  = metric.sqrtg() * cl.Gamma * cl.J / atmo_params.eps_min ; 
        } else if ( epsilon > atmo_params.eps_max ) {
            // avoid subnormal 
            double n = fmax(1e-200, cl.J / atmo_params.eps_max ) ; 
            this->_state(VEC(i,j,k),NRAD_,q)  = metric.sqrtg() * cl.Gamma * n ; 
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
    void KOKKOS_INLINE_FUNCTION 
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
        eas[KAL]   = this->_aux(VEC(i,j,k),KAPPAA_,q) ; 
        eas[KSL]   = this->_aux(VEC(i,j,k),KAPPAS_,q) ; 
        eas[ETAL]  = this->_aux(VEC(i,j,k),ETA_,q) ; 
        eas[ETANL] = this->_aux(VEC(i,j,k),ETAN_,q) ; 
        eas[KANL]  = this->_aux(VEC(i,j,k),KAPPAAN_,q) ; 
        /**************************************************************************************************/
        // construct closure and update
        m1_prims_array_t prims ; 
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,VEC(i,j,k)) ; 
        prims[ERADL] /= metric.sqrtg() ; 
        prims[NRADL] /= metric.sqrtg() ; 
        prims[FXL] /= metric.sqrtg(); 
        prims[FYL] /= metric.sqrtg(); 
        prims[FZL] /= metric.sqrtg(); 

        m1_closure_t cl{prims,metric} ;
        cl.update_closure(0.) ;
        /**************************************************************************************************/
        // store explicitly updated state 
        double  W[4]  ; 
        W[0] = prims[ERADL] ; W[1] = prims[FXL] ; W[2] = prims[FYL] ; W[3] = prims[FZL] ;   
        /**************************************************************************************************/
        // construct the initial guess 
        double  U[4]  ; 
        cl.get_implicit_update_initial_guess(eas, U, dt, dtfact);
        //U[0] = ( prims[ERADL] + dt * dtfact * eas[ETAL]) / ( 1. + dt * dtfact * eas[KAL]) ; 
        #if 0
        if ( i == 4 and j == 4 and k == 4 ) {
            printf("E_guess %.16g eta %.16g kappa %.16g \n", U[0], eas[ETAL], eas[KAL]) ; 
        }
        #endif 
        // take a pointer so we can capture it 
        // in the lambda 
        m1_closure_t* pcl = &cl; 
        /**************************************************************************************************/
        // construct the lambdas for the evaluation of the update 
        auto const func = [pcl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4]) {
            pcl->implicit_update_func(eas,u,W,s,dt,dtfact) ; 
        } ; 
        auto const dfunc = [pcl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4], double (&J)[4][4]) {
            pcl->implicit_update_dfunc(eas,u,W,s,J,dt,dtfact) ; 
        } ; 
        /**************************************************************************************************/
        // call rootfinder 
        unsigned long maxiter = 30 ; 
        int err = 0; 
        utils::rootfind_nd_newton_raphson<4>(
            func, dfunc, U, maxiter, 1e-15, err
        ) ; 
        /**************************************************************************************************/
        if ( err != utils::nr_err_t::SUCCESS ) {
            // assume optically thick closure and 
            // repeat 
            cl.update_closure(prims,0.,false /*nb no update here*/) ;

            cl.get_implicit_update_initial_guess(eas, U, dt, dtfact);

            auto const fixed_closure_func = [pcl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4]) {
                pcl->implicit_update_func(eas,u,W,s,dt,dtfact,false) ; 
            } ; 
            auto const fixed_closure_dfunc = [pcl,eas,W,dt,dtfact] (double (&u)[4], double (&s)[4], double (&J)[4][4]) {
                pcl->implicit_update_dfunc(eas,u,W,s,J,dt,dtfact) ; 
            } ; 
            utils::rootfind_nd_newton_raphson<4>(
                fixed_closure_func, fixed_closure_dfunc, U, maxiter, 1e-15, err
            ) ; 
            // if we failed again we just take a linear step and call it 
            if ( err != utils::nr_err_t::SUCCESS ) {
                cl.update_closure(prims,0,true) ; 
                cl.get_implicit_update_initial_guess(eas, U, dt, dtfact); 
            }
        }
        // the compiler seems to sometimes think U is never 
        // modified and just elides the whole function....
        volatile double U0 = U[0];
        volatile double U1 = U[1];
        volatile double U2 = U[2];
        volatile double U3 = U[3];
        /**************************************************************************************************/
        // write back to the new state 
        #define VOLATILE_WRITE(ivar,val) \
        state_new(i,j,k,ivar,q) = val
        //VOLATILE_WRITE(ERAD_ , metric.sqrtg() * U0) ; 
        //VOLATILE_WRITE(FRADX_, metric.sqrtg() * U1) ; 
        //VOLATILE_WRITE(FRADY_, metric.sqrtg() * U2) ; 
        //VOLATILE_WRITE(FRADZ_, metric.sqrtg() * U3) ; 
        state_new(i,j,k,ERAD_,q)  = metric.sqrtg() * U[0] ; 
        state_new(i,j,k,FRADX_,q) = metric.sqrtg() * U[1] ; 
        state_new(i,j,k,FRADY_,q) = metric.sqrtg() * U[2] ; 
        state_new(i,j,k,FRADZ_,q) = metric.sqrtg() * U[3] ; 
        /**************************************************************************************************/
        #ifndef GRACE_FREEZE_HYDRO
        double const dE = this->_state(VEC(i,j,k),ERAD_,q) - state_new(VEC(i,j,k),ERAD_,q) ; 
        state_new(VEC(i,j,k),TAU_,q) += dE ; 
        double const dSx = this->_state(VEC(i,j,k),FRADX_,q) - state_new(VEC(i,j,k),FRADX_,q) ;
        state_new(VEC(i,j,k),SX_,q) += dSx ; 
        double const dSy = this->_state(VEC(i,j,k),FRADY_,q) - state_new(VEC(i,j,k),FRADY_,q) ;
        state_new(VEC(i,j,k),SY_,q) += dSy ; 
        double const dSz = this->_state(VEC(i,j,k),FRADZ_,q) - state_new(VEC(i,j,k),FRADZ_,q) ; 
        state_new(VEC(i,j,k),SZ_,q) += dSz ; 
        #endif
        /**************************************************************************************************/
        // Number source is linear
        // we need to update the closure on the starred state 
        // to get the correct Gamma factor! 
        double N, dN ; 
        cl.update_closure(U0, {U1,U2,U3},0,true) ; 
        // prims here are **not** the implicitly updated ones
        cl.get_N_implicit_update(
            prims, eas, dt, dtfact, &N, &dN 
        ) ; 
        //VOLATILE_WRITE(NRAD_,metric.sqrtg()*N) ; 
        state_new(VEC(i,j,k),NRAD_,q)  = metric.sqrtg() * N ; 
        /**************************************************************************************************/
        // if needed add dN to ye here! 
        #if 0
        if ( i == 4 and j == 4 and k == 4 ) {
            printf("E_old %.16g E_new %.16g eta %.16g kappa %.16g \n", prims[ERADL], U[0], eas[ETAL], eas[KAL]) ; 
        }
        #endif 
    }

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
        using namespace grace; 
        using namespace Kokkos ; 
        /**************************************************************************************************/
        /* Read in the metric                                                                             */
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;
        /**************************************************************************************************/
        // read in eas 
        m1_eas_array_t eas ; 
        eas[KAL]   = this->_aux(VEC(i,j,k),KAPPAA_,q) ; 
        eas[KSL]   = this->_aux(VEC(i,j,k),KAPPAS_,q) ; 
        eas[ETAL]  = this->_aux(VEC(i,j,k),ETA_,q) ; 
        eas[ETANL] = this->_aux(VEC(i,j,k),ETAN_,q) ; 
        eas[KANL]  = this->_aux(VEC(i,j,k),KAPPAAN_,q) ; 
        /**************************************************************************************************/
        // construct closure and update
        m1_prims_array_t prims ; 
        FILL_M1_PRIMS_ARRAY(prims,this->_state,this->_aux,q,VEC(i,j,k)) ; 
        prims[ERADL] /= metric.sqrtg() ; 
        prims[NRADL] /= metric.sqrtg() ; 
        prims[FXL] /= metric.sqrtg(); 
        prims[FYL] /= metric.sqrtg(); 
        prims[FZL] /= metric.sqrtg(); 

        m1_closure_t cl{prims,metric} ;
        cl.update_closure(0.) ;
        /**************************************************************************************************/
        double cmax=0. ; 
        {
            double cp, cm ; 
            compute_cp_cm<0>(cp,cm,cl,metric) ; 
            cmax = fmax(cmax,fmax(fabs(cp),fabs(cm))) ; 
        }
        {
            double cp, cm ; 
            compute_cp_cm<1>(cp,cm,cl,metric) ; 
            cmax = fmax(cmax,fmax(fabs(cp),fabs(cm))) ; 
        }
        {
            double cp, cm ; 
            compute_cp_cm<2>(cp,cm,cl,metric) ; 
            cmax = fmax(cmax,fmax(fabs(cp),fabs(cm))) ; 
        }
        return cmax ; 
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
        std::array<int, 5>
            recon_indices{
                  ERAD_
                , NRAD_
                , FRADX_
                , FRADY_
                , FRADZ_
            } ; 
        /* Local indices in prims array (note z^k -> v^k) */
        std::array<int, 5>
            recon_indices_loc{
                  ERADL
                , NRADL
                , FXL
                , FYL 
                , FZL 
            } ;
        /* Reconstruction                                  */
        m1_prims_array_t primL, primR ; 
        #pragma unroll 5
        for( int ivar=0; ivar<5; ++ivar) {
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
        // ditto for N 
        primL[NRADL] *= primL[ERADL] ; 
        primR[NRADL] *= primR[ERADL] ;
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
        double const kappa_a = this->_aux(VEC(i,j,k),KAPPAA_,q);
        double const kappa_s = this->_aux(VEC(i,j,k),KAPPAS_,q);
        double const _dx = dx(idir,q);
        // this prevents division by zero while also clamping it 
        // in [0,1]... I think! 
        double const A = 1./( _dx * fmax(kappa_a+kappa_s,1./_dx) ) ; 
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
        // Nrad 
        double N_l = primL[NRADL] *  metric_face.sqrtg() ;
        double N_r = primR[NRADL] *  metric_face.sqrtg() ;
        double f_N_l = metric_face.sqrtg() * metric_face.alp() * N_l/cl.Gamma * ( cl.W * (cl.vU[idir]-metric_face.beta(idir)/metric_face.alp()) + cl.HU[idir]/cl.J ) ; 
        double f_N_r = metric_face.sqrtg() * metric_face.alp() * N_r/cr.Gamma * ( cr.W * (cr.vU[idir]-metric_face.beta(idir)/metric_face.alp()) + cr.HU[idir]/cr.J ) ; 
        fluxes(VEC(i,j,k),NRAD_,idir,q) = (cmax*f_N_l + cmin*f_N_r - A * cmax * cmin * (N_r-N_l))/(cmax+cmin) ; 

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