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
#include <grace/utils/reconstruction.hh>
#include <grace/utils/weno_reconstruction.hh>
#include <grace/utils/riemann_solvers.hh>
#include <grace/utils/advanced_riemann_solvers.hh>

#include <Kokkos_Core.hpp>

#include <type_traits>
//#define GRMHD_USE_PPLIM
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
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k)) ;

        /* Compute inverse (contravariant) four-metric                                                    */
        auto const gupmunu = metric.invgmunu() ;
        
        /**************************************************************************************************/
        /* Computation of T^{\mu\nu}, needed for the source term of \tau and \tilde{S}_i                  */
        /**************************************************************************************************/
        std::array<double, 10> Tupmunu ;

        /* Read the primitive variables                                                                   */
        grmhd_prims_array_t prims ; 
        FILL_PRIMS_ARRAY(prims,this->_aux,q,VEC(i,j,k))   ;

        /* Get fluid 4-velocity                                                                           */
        auto const u0 = compute_u0(prims,metric) ; 
        double umu[4] ; 
        umu[0] = u0; 
        #pragma unroll 3
        for(int ii=0; ii<3; ++ii) {
            umu[ii+1] = u0 * prims[VXL+ii] ; 
        }

        /* Compute common factors for T^{\mu\nu}                                                          */
        double b2{0.} ;
        std::array<double,4> smallb{0,0,0,0} ;
        // get comoving b-field 
        compute_smallb(
            smallb, b2, u0 * metric.alp(), prims, metric 
        ) ; 

        double const rho0_h_plus_b2 = 
            prims[RHOL] * ( 1 + prims[EPSL] ) + prims[PRESSL] + b2 ; 
        double const P_plus_half_b2 = 
            prims[PRESSL] + 0.5 * b2 ; 
        

        int icomp{0} ; 
        for( int mu=0; mu<4; ++mu) {
            for(int nu=mu; nu<4; ++nu) {
                Tupmunu[icomp] = rho0_h_plus_b2 * umu[mu] * umu[nu] 
                               + P_plus_half_b2 * gupmunu[icomp]
                               - smallb[mu] * smallb[nu] ; 
                icomp ++ ;
            }
        }
        

        /* Source for the conserved energy (added piece by piece below)                                   */
        double tau_source{0.};

        std::array<double,3> const shift {metric.beta(0),metric.beta(1),metric.beta(2)};

        #ifndef GRACE_ENABLE_COWLING_METRIC
        /* Read in the extrinsic curvature                                                                */
        std::array<double,6> Kij ;
        get_extrinsic_curvature(Kij,this->_state,VEC(i,j,k),q) ; 
        /**************************************************************************************************/
        /* Compute first piece of conserved energy source term (curvature terms)                          */
        /*      S_{\tau} += \alpha (T^{00} \beta^i\beta^j + 2T^{0i}\beta^j  + T^{ij}) K_{ij}              */
        /**************************************************************************************************/
        tau_source += metric.alp() * (
              Tupmunu[0] * metric.contract_vec_sym2tens(shift, Kij) 
            + 2. * metric.contract_vec_vec_sym2tens(shift,{Tupmunu[TX4],Tupmunu[TY4],Tupmunu[TZ4]}, Kij)
            + metric.contract_sym2tens_sym2tens({Tupmunu[XX4],Tupmunu[XY4],Tupmunu[XZ4],Tupmunu[YY4],Tupmunu[YZ4],Tupmunu[ZZ4]}, Kij) ); 
        #endif 

        /* Indices for contraction of T^{0i} onto \partial_i \alpha (see tau source below)                     */
        int index_4d[GRACE_NSPACEDIM] = {VEC(TX4,TY4,TZ4)} ;

        /* More indices for contraction of T^{ij} onto \partial_i \beta_j (see tau source below)               */  
        int spatial_index_4d[GRACE_NSPACEDIM][3] = {
            {VEC(XX4,XY4,XZ4)},
            {VEC(XY4,YY4,YZ4)},
            {VEC(XZ4,YZ4,ZZ4)},
        };

        /* Overall factor of dt \alpha \sqrt{\gamma} to be multiplied to source terms                          */
        double const alpha_sqrtgamma    = metric.alp()*metric.sqrtg();
        double const alpha_sqrtgamma_dt = dt*dtfact*alpha_sqrtgamma;

        /*******************************************************************************************************/
        /* Direction loop for source terms                                                                     */
        /* Although the source term of \tau is scalar (obviously) we add it piece by piece because it contains */
        /* the derivative of \alpha which is more convenient to compute one direction at a time. This loop is  */
        /* anyway needed for the momentum source terms which are a vector and which also contain directional   */
        /* derivatives.                                                                                        */
        /*******************************************************************************************************/
        for( int idir=0; idir<GRACE_NSPACEDIM; ++idir) {

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
            /* We need \partial_i g_{\alpha\beta} for the momentum source term and \partial_i \alpha for the  */
            /* conserved energy source term.                                                                  */
            /**************************************************************************************************/
            std::array<double, 10> dgab_dxi  ;

            /* Get 4 metric                                                                                   */
            auto const gmunu_m = metric_m.gmunu() ;
            auto const gmunu_p = metric_p.gmunu() ;

            /* Compute 4 metric derivative (factor of 1./dx introduced after)                                 */
            #pragma unroll 10
            for( int ii=0; ii<10; ++ii) { 
                dgab_dxi[ii] =  0.5*(gmunu_p[ii] - gmunu_m[ii]) ;
            }
            /* Compute lapse derivative (factor of 1./dx introduced after)                                    */
            double const dalp_dxi =  0.5*(metric_p.alp() - metric_m.alp()) ;

            #ifdef GRACE_ENABLE_COWLING_METRIC
            /**************************************************************************************************/
            /* In Cowling approx we can use the simpler form of the source term which reads                   */
            /* S_{\tau} +=  1/2 T^{ij} \beta^k \partial_k \gamma_{ij} + W^i_j \partial_i \beta^j              */
            /**************************************************************************************************/ 
            std::array<double,3> dbetaj_dxi = {
                0.5 * (metric_p.beta(0) - metric_m.beta(0)),
                0.5 * (metric_p.beta(1) - metric_m.beta(1)),
                0.5 * (metric_p.beta(2) - metric_m.beta(2)),
            } ; 
            
            tau_source += idx(idir,q) * ( metric.contract_vec_vec(
                dbetaj_dxi,
                {Tupmunu[spatial_index_4d[idir][0]],Tupmunu[spatial_index_4d[idir][1]],Tupmunu[spatial_index_4d[idir][2]]}
            )  + 0.5 * shift[idir] * metric.contract_sym2tens_sym2tens(
                {dgab_dxi[XX4],dgab_dxi[XY4],dgab_dxi[XZ4],dgab_dxi[YY4],dgab_dxi[YZ4],dgab_dxi[ZZ4]},
                {Tupmunu[XX4],Tupmunu[XY4],Tupmunu[XZ4],Tupmunu[YY4],Tupmunu[YZ4],Tupmunu[ZZ4]}
            )) ;
            #endif 
            /**************************************************************************************************/
            /* Momentum source term:                                                                          */
            /* S_{\tilde{S}_i} = \frac{1}{2} T^{\alpha\beta} g_{\alpha\beta, i}                               */
            /* NB: The overall factor of \alpha \sqrt{\gamma} is introduced at the end                        */
            /**************************************************************************************************/
            double const st_i_source = 
                0.5 * metric.contract_4dsym2tens_4dsym2tens(dgab_dxi, Tupmunu) * idx(idir,q) ; 

            /**************************************************************************************************/
            /* Second part of conserved energy source term:                                                   */
            /* S_{\tau} +=  - (T^{0 i} +  T^{00} beta^i) \partial_i \alpha                                    */
            /**************************************************************************************************/
            tau_source  -= 
                metric.alp() * ( Tupmunu[index_4d[idir]] + Tupmunu[TT4] * metric.beta(idir) ) * dalp_dxi * idx(idir,q) ; 

            /**************************************************************************************************/
            /* Add momentum source terms                                                                      */
            /**************************************************************************************************/
            state_new(VEC(i,j,k),SX_+idir,q) += alpha_sqrtgamma_dt*st_i_source ; 
            /**************************************************************************************************/
        }
        /**************************************************************************************************/
        /* Add energy source terms                                                                        */
        /**************************************************************************************************/
        state_new(VEC(i,j,k),TAU_,q)     += dt*dtfact*metric.sqrtg()*tau_source ;
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
                        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords) const 
    {
        using namespace grace ;
        using namespace Kokkos ; 
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
        
        grmhd_prims_array_t prims ;        
        conservs_to_prims<eos_t>( cons, prims, metric
                                , this->_eos
                                , {pcoords(VEC(i,j,k),0,q),pcoords(VEC(i,j,k),1,q),pcoords(VEC(i,j,k),2,q)}
                                , atmo_params, excision_params ) ;
        
        
        /* Write new prims */
        aux(RHO_) = prims[RHOL]     ; 
        aux(EPS_) = prims[EPSL]     ; 
        aux(PRESS_) = prims[PRESSL] ; 
        aux(VELX_) = prims[VXL]     ;
        aux(VELY_) = prims[VYL]     ; 
        aux(VELZ_) = prims[VZL]     ; 
        aux(TEMP_) = prims[TEMPL]   ; 
        aux(ENTROPY_) = prims[ENTL]  ; 
        aux(YE_)   = prims[YEL]     ;
        
        /* Compute ZVEC */
        double const one_over_alp = 1./metric.alp(); 
        std::array<double,3> const vN {
              one_over_alp * (prims[VXL] + metric.beta(0))
            , one_over_alp * (prims[VYL] + metric.beta(1))
            , one_over_alp * (prims[VZL] + metric.beta(2))
        } ; 

        double const W = 1./Kokkos::sqrt(1.-metric.square_vec(vN)) ;

        aux(ZVECX_) = W * vN[0] ; 
        aux(ZVECY_) = W * vN[1] ; 
        aux(ZVECZ_) = W * vN[2] ; 
        /* Overwrite conserved */
        #if 1
        vars(DENS_)  = cons[DENSL]       ; 
        vars(SX_)    = cons[STXL]        ; 
        vars(SY_)    = cons[STYL]        ;
        vars(SZ_)    = cons[STZL]        ;
        vars(TAU_)   = cons[TAUL]        ;
        vars(YESTAR_) = cons[YESL]       ; 
        vars(ENTROPYSTAR_) = cons[ENTSL] ; 
        #endif
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
        using namespace grace; 
        using namespace Kokkos ; 

        /* Get prims */
        grmhd_prims_array_t prims ;
        FILL_PRIMS_ARRAY(prims,this->_aux,q,VEC(i,j,k)) ;
        /* Get metric */
        metric_array_t metric ; 
        FILL_METRIC_ARRAY(metric,this->_state,q,VEC(i,j,k));
        /* Get soundspeed, enthalpy */
        double csnd2, h ; 
        unsigned int err ; 
        double dummy = _eos.press_h_csnd2__temp_rho_ye( h, csnd2, prims[TEMPL]
                                                      , prims[RHOL], prims[YEL], err ) ;
        /* Compute magnetosonic speed */
        double b2{0.} ;
        std::array<double,4> dummy2 ; 
        double const u0 = compute_u0(prims,metric) ;  
        auto W = u0 * metric.alp() ; 
        compute_smallb(dummy2,b2,W,prims,metric) ; 
        
        double const v_A_sq =  b2 / ( b2 + prims[RHOL]*h) ; 
        double const v02 = v_A_sq + csnd2 * ( 1. - v_A_sq ) ;
        /* Find maximum eigenvalue (amongst all directions) */
        double cmax {0}; 
        std::array<unsigned int, 3> const metric_comp{ 0, 3, 5 } ; 
        
        for( int idir=0; idir<3; ++idir){ 
            double cp, cm ; 
            compute_cp_cm( cp, cm, v02, u0, prims[VXL+idir]
                         , 1./math::int_pow<2>(metric.alp())
                         , metric.beta(idir)
                         , metric.invgamma(metric_comp[idir]) );
            cmax = math::max(cmax,math::abs(cp),math::abs(cm)) ; 
        }
        return cmax ; 
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
                , VXL
                , VYL
                , VZL
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
            + metric_face.square_vec({primL[VXL], primL[VYL], primL[VZL]}));
        double const wr   = Kokkos::sqrt(1. 
            + metric_face.square_vec({primR[VXL], primR[VYL], primR[VZL]}));
        
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
        primL[VXL] = alp * primL[VXL] / wl - metric_face.beta(0) ;
        primL[VYL] = alp * primL[VYL] / wl - metric_face.beta(1) ;
        primL[VZL] = alp * primL[VZL] / wl - metric_face.beta(2) ; 

        /* Right */
        primR[PRESSL] = _eos.press_eps_csnd2__temp_rho_ye(primR[EPSL], cs2r, primR[TEMPL], primR[RHOL], primR[YEL], eos_err) ; 
        primR[VXL] = alp * primR[VXL] / wr - metric_face.beta(0) ;
        primR[VYL] = alp * primR[VYL] / wr - metric_face.beta(1) ;
        primR[VZL] = alp * primR[VZL] / wr - metric_face.beta(2) ;

        std::array<double,3> uD_l, uD_r ; 
        solver.transform_velocities_to_tetrad_frame(u0_l, primL, uD_l) ; 
        solver.transform_velocities_to_tetrad_frame(u0_r, primR, uD_r) ; 

        /* Compute specific enthalpies */
        double h_l = 1 + primL[EPSL] + primL[PRESSL]/primL[RHOL] ;
        double h_r = 1 + primR[EPSL] + primR[PRESSL]/primR[RHOL] ;
        
        grmhd_cons_array_t fL, fR, uL, uR; 

        /* Get wavespeeds      */ 
        double cpr, cmr, cpl, cml;
        compute_cp_cm( cpl, cml, cs2l, u0_l, primL[VXL+idir], 1
                     , 0, 1) ;
        compute_cp_cm( cpr, cmr, cs2r, u0_r, primR[VXL+idir], 1
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

        fL[DENSL] = uL[DENSL] * primL[VXL+idir] ; 
        fR[DENSL] = uR[DENSL] * primR[VXL+idir] ; 

        /***********************************************************************/
        /*                          Get ye_star flux                           */
        /***********************************************************************/
        uL[YESL] = uL[DENSL] * primL[YEL] ; 
        uR[YESL] = uR[DENSL] * primR[YEL] ; 
        
        fL[YESL] = uL[YESL] * primL[VXL+idir] ; 
        fR[YESL] = uR[YESL] * primR[VXL+idir] ; 

        /***********************************************************************/
        /*                          Get s_star flux                            */
        /***********************************************************************/
        uL[ENTSL] = uL[DENSL] * primL[ENTL] ; 
        uR[ENTSL] = uR[DENSL] * primR[ENTL] ; 

        fL[ENTSL] = uL[ENTSL] * primL[VXL+idir] ; 
        fR[ENTSL] = uR[ENTSL] * primR[VXL+idir] ; 

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
        fL[TAUL] = tau_plus_P_l * primL[VXL+idir] ;
        fR[TAUL] = tau_plus_P_r * primR[VXL+idir] ;
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

        fL[STXL] = uL[STXL] * primL[VXL+idir] + primL[PRESSL] * utils::delta(idir,0) ;
        fR[STXL] = uR[STXL] * primR[VXL+idir] + primR[PRESSL] * utils::delta(idir,0) ;

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

        fL[STYL] = uL[STYL] * primL[VXL+idir] + primL[PRESSL] * utils::delta(idir,1) ;
        fR[STYL] = uR[STYL] * primR[VXL+idir] + primR[PRESSL] * utils::delta(idir,1) ; 

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

        fL[STZL] = uL[STZL] * primL[VXL+idir] 
            + primL[PRESSL] * utils::delta(idir,2) ;
        fR[STZL] = uR[STZL] * primR[VXL+idir] 
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
        /* 2nd order interpolation at cell interface                           */
        /***********************************************************************/
        #if 0
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
        #else 
        metric_array_t metric_face ; 
        COMPUTE_FCVAL(metric_face,this->_state,i,j,k,q,idir) ; 
        #endif
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
                , VXL
                , VYL
                , VZL
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
        double rho_atm = 1e-6 ; //FIXME //_eos.rho_atmosphere() ; 
        
        double const dens_min_r = rho_atm * metric_r.sqrtg() ; 
        double const dens_min_l = rho_atm * metric_l.sqrtg() ; 

        double const dens_LLF_m = consR[DENSL] + a2CFL * f_LLF[DENSL] ; 
        double const dens_LLF_p = consL[DENSL] - a2CFL * f_LLF[DENSL] ;

        double const dens_m = consR[DENSL] + a2CFL * f_HLL[DENSL] ; 
        double const dens_p = consL[DENSL] - a2CFL * f_HLL[DENSL] ; 

        double theta_p = 1.; 
        double theta_m = 1.; 

        if (dens_m < dens_min_r) {
            theta_m = math::min(theta, math::max(0, (dens_min_r-dens_LLF_m)/(a2CFL*(f_HLL[DENSL]-f_LLF[DENSL])))) ; 
        }
        if ( dens_p < dens_min_l ) {
            theta_p = math::min(theta, math::max(0, -(dens_min_l-dens_LLF_p)/(a2CFL*(f_HLL[DENSL]-f_LLF[DENSL])))) ; 
        }

        theta = math::min(theta_m, theta_p) ;
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

        riemann_t solver     {} ;
        /***********************************************************************/
        /* Compute u0 on both sides                                            */
        /***********************************************************************/
        /* Lorentz factors  */
        /* W = sqrt(1+z^2)  */
        double const alp = metric_face.alp() ;
        double const wl   = Kokkos::sqrt(1. 
            + metric_face.square_vec({primL[VXL], primL[VYL], primL[VZL]}));
        double const wr   = Kokkos::sqrt(1. 
            + metric_face.square_vec({primR[VXL], primR[VYL], primR[VZL]}));
        
        /* u^0             */
        double const u0_l = wl / alp ; 
        double const u0_r = wr / alp ; 

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
        primL[VXL] = alp * primL[VXL] / wl - metric_face.beta(0) ;
        primL[VYL] = alp * primL[VYL] / wl - metric_face.beta(1) ;
        primL[VZL] = alp * primL[VZL] / wl - metric_face.beta(2) ; 

        /* Right */
        primR[PRESSL] = _eos.press_eps_csnd2__temp_rho_ye(primR[EPSL], cs2r, primR[TEMPL], primR[RHOL], primR[YEL], eos_err) ; 
        primR[VXL] = alp * primR[VXL] / wr - metric_face.beta(0) ;
        primR[VYL] = alp * primR[VYL] / wr - metric_face.beta(1) ;
        primR[VZL] = alp * primR[VZL] / wr - metric_face.beta(2) ;

        /* Compute small b */
        std::array<double,4> smallbL{0,0,0,0}, smallbR{0,0,0,0} ; 
        double b2l{0.}, b2r{0.}; 
        compute_smallb(smallbL, b2l, wl, primL, metric_face) ; 
        compute_smallb(smallbR, b2r, wr, primR, metric_face) ; 

        /* Compute Alfvén speeds */
        double v02r,v02l, h_r,h_l;
        compute_v02(h_l, v02l, cs2l, b2l, primL) ; 
        compute_v02(h_r, v02r, cs2r, b2r, primR) ;

        /* Get wavespeeds      */
        double const one_over_alp2 = 1./math::int_pow<2>(alp); 

        double cmin, cmax ;
        if constexpr ( recompute_cp_cm ) {
            double cpr, cmr, cpl, cml;
            int metric_comps[3] { 0, 3, 5} ; 
            compute_cp_cm( cpl, cml, v02l, u0_l, primL[VXL+idir], one_over_alp2
                        , metric_face.beta(idir), metric_face.invgamma(metric_comps[idir])) ;
            compute_cp_cm( cpr, cmr, v02r, u0_r, primR[VXL+idir], one_over_alp2
                        , metric_face.beta(idir), metric_face.invgamma(metric_comps[idir])) ;
            cmin = -Kokkos::min(0., Kokkos::min(cml,cmr)) ; 
            cmax =  Kokkos::max(0., Kokkos::max(cpl,cpr)) ; 
            /* Add some diffusion in weakly hyperbolic limit */
            if( cmin < 1e-12 and cmax < 1e-12 ) { cmin=1; cmax=1; }
        } else {
            cmin = cmin_loc ; 
            cmax = cmax_loc ; 
        }
        /***********************************************************************/
        /*                          Get dens flux                              */
        /***********************************************************************/
        double const alpha_sqrtgamma = alp * metric_face.sqrtg() ;
        double const dens_l = alpha_sqrtgamma * primL[RHOL] * u0_l ;
        double const dens_r = alpha_sqrtgamma * primR[RHOL] * u0_r ;

        double fl = dens_l * primL[VXL+idir] ; 
        double fr = dens_r * primR[VXL+idir] ; 

        f[DENSL] = solver(fl,fr,dens_l,dens_r,cmin,cmax) ; 

        /***********************************************************************/
        /*                          Get ye_star flux                           */
        /***********************************************************************/
        double const ye_star_l = dens_l * primL[YEL] ; 
        double const ye_star_r = dens_r * primR[YEL] ; 
        
        fl = ye_star_l * primL[VXL+idir] ; 
        fr = ye_star_l * primR[VXL+idir] ; 

        f[YESL] = solver(fl,fr,ye_star_l,ye_star_r,cmin,cmax) ;

        /***********************************************************************/
        /*                          Get s_star flux                            */
        /***********************************************************************/
        double const s_star_l = dens_l * primL[ENTL] ; 
        double const s_star_r = dens_r * primR[ENTL] ; 

        fl = s_star_l * primL[VXL+idir] ; 
        fr = s_star_r * primR[VXL+idir] ; 

        f[ENTSL] = solver(fl,fr,s_star_l,s_star_r,cmin,cmax) ;

        /***********************************************************************/ 
        /*                           Get tau flux                              */
        /***********************************************************************/
        /* Auxiliary metric quantitites */
        double const alp2_sqrtgamma = math::int_pow<2>(alp) * metric_face.sqrtg() ; 
        double const g4uptd = one_over_alp2 * metric_face.beta(idir) ; 
        double const g4uptt = -one_over_alp2 ; 

        /***************************************************************************/
        /***************************************************************************/
        /* Left flux */
        double const rho0_h_plus_b2_l = primL[RHOL]*(1+primL[EPSL]) + primL[PRESSL] 
                                      + b2l ;
        
        double const P_plus_half_b2_l = primL[PRESSL] + 0.5*b2l;
        /***************************************************************************/
        /* T^{td} = (rho h + b^2) u^0 (u^0 v^d) + (P + b^2/2) g^{td} - b^t b^d     */
        /***************************************************************************/
        double const TUPtd_l = rho0_h_plus_b2_l * math::int_pow<2>(u0_l) * primL[VXL+idir] 
            + P_plus_half_b2_l*g4uptd - smallbL[0]*smallbL[1+idir] ;
        /***************************************************************************/
        /* F^{d}_{\rm tau} = \alpha^2 \sqrt{\gamma} T^{td} - D v^d                 */
        /***************************************************************************/
        fl = alp2_sqrtgamma * TUPtd_l - dens_l * primL[VXL+idir] ;
        /***************************************************************************/
        /* T^{tt} = (rho h + b^2) u^0 u^0 + (P + b^2/2 ) g^{tt} - b^t b^t          */
        /***************************************************************************/
        double const Tuptt_l = rho0_h_plus_b2_l*math::int_pow<2>(u0_l) 
            + P_plus_half_b2_l*g4uptt - math::int_pow<2>(smallbL[0]) ;
        /***************************************************************************/
        /* \tau = \alpha^2 \sqrt{\gamma} T^{tt} - D                                */
        /***************************************************************************/
        double const tau_l = alp2_sqrtgamma * Tuptt_l - dens_l ;

        /***************************************************************************/
        /***************************************************************************/
        /* Right flux */
        double const rho0_h_plus_b2_r = primR[RHOL]*(1+primR[EPSL]) + primR[PRESSL] 
                                      + b2r ;

        double const P_plus_half_b2_r = primR[PRESSL] + 0.5*b2r;
        /***************************************************************************/
        /* T^{td} = (rho h + b^2) u^0 (u^0 v^d) + (P + b^2/2) g^{td} - b^t b^d     */
        /***************************************************************************/
        double const TUPtd_r = rho0_h_plus_b2_r * math::int_pow<2>(u0_r) * primR[VXL+idir] 
            + P_plus_half_b2_r*g4uptd - smallbR[0]*smallbR[1+idir] ; 
        /**************************************************************************/
        /* F^{d}_{\rm tau} = \alpha^2 \sqrt{\gamma} T^{td} - D v^d                */
        /**************************************************************************/
        fr = alp2_sqrtgamma * TUPtd_r - dens_r * primR[VXL+idir] ;
        /**************************************************************************/
        /* T^{tt} = (rho h + b^2) u^0 u^0 + (P + b^2/2 ) g^{tt} - b^t b^t         */
        /**************************************************************************/
        double const Tuptt_r = rho0_h_plus_b2_r*math::int_pow<2>(u0_r) 
            + P_plus_half_b2_r*g4uptt - math::int_pow<2>(smallbR[0]) ; 
        /***********************************************************************/
        /* \tau = \alpha^2 \sqrt{\gamma} T^{tt} - D                            */
        /***********************************************************************/
        double const tau_r = alp2_sqrtgamma * Tuptt_r - dens_r ; 

        /***************************************************************************/
        f[TAUL] = solver(fl,fr,tau_l,tau_r,cmin,cmax) ; 
        /***************************************************************************/

        /***********************************************************************/
        /* Momentum flux in direction d for S_j : \alpha \sqrt{\gamma} T^d_j   */
        /***********************************************************************/
        /* Compute u_i */
        auto uD_l = metric_face.lower({ primL[VXL]+metric_face.beta(0)
                                      , primL[VYL]+metric_face.beta(1)
                                      , primL[VZL]+metric_face.beta(2) }) ; 
        for(auto& uu: uD_l) uu *= u0_l ; 
        auto uD_r = metric_face.lower({ primR[VXL]+metric_face.beta(0)
                                      , primR[VYL]+metric_face.beta(1)
                                      , primR[VZL]+metric_face.beta(2) }) ; 
        for(auto& uu: uD_r) uu *= u0_r ; 
        /***********************************************************************/
        /* Get S_x flux                                                        */
        /***********************************************************************/
        auto smallbDL = metric_face.lower_4vec(smallbL) ; 
        auto smallbDR = metric_face.lower_4vec(smallbR) ; 
        /***********************************************************************/
        /* F^d_{S_x} = \alpha \sqrt{\gamma} T^d_x                              */
        /*  = \alpha \sqrt{\gamma} ( (\rho h + b^2) u^0 v^d u_x                */
        /*                         + p \delta^d_x - b^d b_x )                  */  
        /***********************************************************************/
        fl = alpha_sqrtgamma * ( rho0_h_plus_b2_l * (u0_l*primL[VXL+idir])*uD_l[0]
           + P_plus_half_b2_l*utils::delta(0,idir) - smallbL[idir+1]*smallbDL[1] ) ; 
        fr = alpha_sqrtgamma * ( rho0_h_plus_b2_r * (u0_r*primR[VXL+idir])*uD_r[0]
           + P_plus_half_b2_r*utils::delta(0,idir) - smallbR[idir+1]*smallbDR[1] ) ;  

        double const s_x_l = alpha_sqrtgamma * (  rho0_h_plus_b2_l*u0_l*uD_l[0]
                                                - smallbL[0]*smallbDL[1] ) ; 

        double const s_x_r = alpha_sqrtgamma * (  rho0_h_plus_b2_r*u0_r*uD_r[0]
                                                - smallbR[0]*smallbDR[1] ) ; 

        /***********************************************************************/
        f[STXL] = solver(fl,fr,s_x_l,s_x_r,cmin,cmax) ; 
        /***********************************************************************/

        /***********************************************************************/
        /* Get S_y flux                                                        */
        /***********************************************************************/

        /***********************************************************************/
        /* F^d_{S_y} = \alpha \sqrt{\gamma} T^d_y                              */
        /*  = \alpha \sqrt{\gamma} ( (\rho h + b^2) u^0 v^d u_y                */
        /*                         + p \delta^d_y - b^d b_y )                  */  
        /***********************************************************************/
        fl = alpha_sqrtgamma * ( rho0_h_plus_b2_l * (u0_l*primL[VXL+idir])*uD_l[1]
           + P_plus_half_b2_l*utils::delta(1,idir) - smallbL[idir+1]*smallbDL[2] ) ; 
        fr = alpha_sqrtgamma * ( rho0_h_plus_b2_r * (u0_r*primR[VXL+idir])*uD_r[1]
           + P_plus_half_b2_r*utils::delta(1,idir) - smallbR[idir+1]*smallbDR[2] ) ;
         
        
        double const s_y_l = alpha_sqrtgamma * ( rho0_h_plus_b2_l*u0_l*uD_l[1]
                                               - smallbL[0]*smallbDL[2] ) ; 

        double const s_y_r = alpha_sqrtgamma * ( rho0_h_plus_b2_r*u0_r*uD_r[1]
                                               - smallbR[0]*smallbDR[2] ) ; 
        
        /***********************************************************************/
        f[STYL] = solver(fl,fr,s_y_l,s_y_r,cmin,cmax) ;
        /***********************************************************************/

        /***********************************************************************/
        /* Get S_z flux                                                        */
        /***********************************************************************/

        /***********************************************************************/
        /* F^d_{S_z} = \alpha \sqrt{\gamma} T^d_z                              */
        /*  = \alpha \sqrt{\gamma} ( (\rho h + b^2) u^0 v^d u_z                */
        /*                         + p \delta^d_z - b^d b_z )                  */  
        /***********************************************************************/
        fl = alpha_sqrtgamma * ( rho0_h_plus_b2_l * u0_l*primL[VXL+idir]*uD_l[2]
           + P_plus_half_b2_l*utils::delta(2,idir) - smallbL[idir+1]*smallbDL[3] ) ; 
        fr = alpha_sqrtgamma * ( rho0_h_plus_b2_r * u0_r*primR[VXL+idir]*uD_r[2]
           + P_plus_half_b2_r*utils::delta(2,idir) - smallbR[idir+1]*smallbDR[3] ) ;  

        double const s_z_l = alpha_sqrtgamma * ( rho0_h_plus_b2_l*u0_l*uD_l[2]
                                               - smallbL[0]*smallbDL[3] ) ; 

        double const s_z_r = alpha_sqrtgamma * ( rho0_h_plus_b2_r*u0_r*uD_r[2]
                                               - smallbR[0]*smallbDR[3] ) ; 

        /***********************************************************************/
        f[STZL] = solver(fl,fr,s_z_l,s_z_r,cmin,cmax) ; 
        /***********************************************************************/
        auto const sqrtg = metric_face.sqrtg() ;
        std::array<int, 2> jk ;  
        if constexpr ( idir == 0 ) {
            jk[0] = 1 ; jk[1] = 2; 
        } else if constexpr ( idir == 1 ) {
            jk[0] = 0 ; jk[1] = 2; 
        } else {
            jk[0] = 0 ; jk[1] = 1; 
        }
        vbar[0] = solver(primL[VXL+jk[0]],primR[VXL+jk[0]],0,0,cmin,cmax) ; 
        vbar[1] = solver(primL[VXL+jk[1]],primR[VXL+jk[1]],0,0,cmin,cmax) ; 
        vbar[2] = cmin ; vbar[3] = cmax ; 
        
    };
    /***********************************************************************/
    /**
     * @brief Compute Alfvén speed, specific enthalpy 
     *        and approximate magnetosonic wave speed.
     * 
     * @param h Enthalpy.
     * @param v02 Squared magnetosonic wave (approximate) speed.
     * @param cs2 Squared sound speed.
     * @param b2  Square comoving magnetic field.
     * @param prims Primitive variables.
     */
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_v02( double& h, double& v02, double const& cs2, double const& b2
               , grmhd_prims_array_t const& prims ) const
    {
        h = 1. + prims[EPSL] + prims[PRESSL] / prims[RHOL] ; 
        double const v_A_sq =  b2 / ( b2 + prims[RHOL]*h) ; 
        v02 = v_A_sq + cs2 * ( 1. - v_A_sq ) ; 
    }
    void GRACE_HOST_DEVICE compute_smallb(  std::array<double,4>& smallb, double& b2, double const& W
                        , grmhd_prims_array_t& prims, metric_array_t const& metric ) const
    {
        #if 1
        // simple minded, can be optimized later
        double const u0 = W / metric.alp() ; 
        std::array<double,3> const vi = { 
            (prims[VXL]  + metric.beta(0))/metric.alp(),
            (prims[VYL]  + metric.beta(1))/metric.alp(),
            (prims[VZL]  + metric.beta(2))/metric.alp(),
        } ; 
        std::array<double,3> const ui = { 
            prims[VXL] * u0,
            prims[VYL] * u0,
            prims[VZL] * u0,
        } ;  
        smallb[0] = metric.contract_vec_vec(vi,{prims[BXL],prims[BYL],prims[BZL]}) * u0 ; 
        for( int i=0; i<3; ++i) {
            smallb[i+1] = (prims[BXL+i] + metric.alp() * smallb[0] * ui[i])/W ; 
        }
        b2 = ( metric.square_vec({prims[BXL],prims[BYL],prims[BZL]}) + metric.alp()*metric.alp() * smallb[0] * smallb[0] ) / W / W ; 
        return ;
        #else 
        b2 = 0 ; 
        smallb = {0,0,0,0} ; 
        #endif 

    }
    /***********************************************************************/
    /**
     * @brief Compute approximate GRMHD wave-speeds according to 
     *        eq. (28) in https://iopscience.iop.org/article/10.1086/374594/pdf.
     * 
     * @param cp Maximum wavespeed
     * @param cm Minimum wavespeed
     * @param v02 Squared v0.
     * @param vd  3 velocity in direction d.
     * @param one_over_alp2 One over lapse squared.
     * @param betad Shift in direction d.
     * @param gupdd (d,d) component of contravariant metric.
     */
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_cp_cm( double& cp, double& cm
                 , double const& v02, double const& u0
                 , double const& vd, double const& one_over_alp2 
                 , double const& betad, double const& gupdd ) const
    {
        double const u0_sq = math::int_pow<2>(u0) ; 

        double const a = u0_sq * ( 1- v02 ) + v02 * one_over_alp2 ; 

        double const b = 2. * ( betad * one_over_alp2 * v02 - u0_sq * vd * (1. - v02 )) ; 

        double const c = u0_sq * math::int_pow<2>(vd) * ( 1.-v02 ) 
            - v02 * ( gupdd - math::int_pow<2>(betad)*one_over_alp2) ;
        
        //double det = math::int_pow<2>(b) - 4. * a * c ; 
        double const det = Kokkos::sqrt(math::max(0., math::int_pow<2>(b) - 4. * a * c))  ; 

        double const c1 =  0.5*(det-b) / a ; 
        double const c2 = -0.5*(det+b) / a ; 

        cp = Kokkos::max(c1,c2) ; 
        cm = Kokkos::min(c1,c2) ; 
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
