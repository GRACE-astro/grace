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

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/evolution/hrsc_evolution_system.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/evolution/evolution_kernel_tags.hh>

#include <Kokkos_Core.hpp>

#define FILL_METRIC_ARRAY(g, view, q, ...) \
grace::metric_array_t g{  { view(VEC(__VA_ARGS__,GXX_,q)) \
                          , view(VEC(__VA_ARGS__,GXY_,q)) \
                          , view(VEC(__VA_ARGS__,GXZ_,q)) \
                          , view(VEC(__VA_ARGS__,GYY_,q)) \
                          , view(VEC(__VA_ARGS__,GYZ_,q)) \
                          , view(VEC(__VA_ARGS__,GZZ_,q)) } \
                        , { view(VEC(__VA_ARGS__,BETAX_,q)) \
                          , view(VEC(__VA_ARGS__,BETAY_,q)) \
                          , view(VEC(__VA_ARGS__,BETAZ_,q)) } \
                        , view(VEC(__VA_ARGS__,ALP_,q)) } 

namespace grace {
//**************************************************************************************************/
/**
 * \defgroup physics Physics Modules.
 */
//**************************************************************************************************/
/* Auxiliaries */
//**************************************************************************************************/
enum GRMHD_PRIMS_LOC_INDICES {
    RHOL = 0,
    PRESSL,
    VXL,
    VYL,
    VZL,
    YEL,
    TEMPL,
    EPSL,
    ENTL,
    #ifdef GRACE_DO_MHD
    BXL,
    BYL,
    BZL,
    #endif 
    NUM_PRIMS_LOC
} ; 
enum GRMHD_CONS_LOC_INDICES {
    DENSL=0,
    STXL,
    STYL,
    STZL,
    TAUL,
    YESL,
    ENTSL,
    NUM_CONS_LOC
} ; 
using grmhd_prims_array_t = std::array<double,GRMHD_PRIMS_LOC> ; 
using grmhd_cons_array_t  = std::array<double,GRMHD_CONS_LOC>  ;
//**************************************************************************************************/ 
//**************************************************************************************************
/**
 * @brief GRMHD equations system.
 * \ingroup physics 
 * @tparam eos_t Type of equation of state used.
 * @tparam recon_t Type of reconstruction used.
 * @tparam riemann_t Type of Riemann solver used.
 */
//**************************************************************************************************/
template< typename eos_t 
        , typename recon_t 
        , typename riemann_t >
struct grmhd_equations_system_t 
    : public hrsc_evolution_system_t<grmhd_equations_system_t<eos_t,recon_t,riemann_t>>
{
 private:
    //! Base class type 
    using base_t = hrsc_evolution_system_t<grmhd_equations_system_t<eos_t,recon_t,riemann_t>>;

 public:

    /**
     * @brief Constructor
     * 
     * @param eos_ eos object.
     * @param state_ State array.
     * @param aux_ Auxiliary array.
     */
    grmhd_equations_system_t( eos_t eos_ 
                            , grace::var_array_t<GRACE_NSPACEDIM> state_
                            , grace::var_array_t<GRACE_NSPACEDIM> aux_   ) 
     : base_t(state_,aux_), _eos(eos_)
    { } ;
    /**
     * @brief Compute GRMHD fluxes in direction \f$x^1\f$
     * 
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_x_flux( thread_team_t& team 
                  , VEC( const int i 
                  ,      const int j 
                  ,      const int k)
                  , int ngz
                  , grace::flux_array_t const  fluxes) const 
    {
        getflux<0>(VEC(i,j,k),team.league_rank(),ngz,fluxes);
    }
    /**
     * @brief Compute GRMHD fluxes in direction \f$x^2\f$
     * 
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_y_flux( thread_team_t& team 
                  , VEC( const int i 
                  ,      const int j 
                  ,      const int k)
                  , int ngz
                  , grace::flux_array_t const  fluxes) const 
    {
        getflux<1>(VEC(i,j,k),team.league_rank(),ngz,fluxes); 
    }
    /**
     * @brief Compute GRMHD fluxes in direction \f$x^3\f$
     * 
     * @tparam thread_team_t Type of the thread team.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param ngz  Number of ghost cells.
     * @param fluxes Flux array.
     */
    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_z_flux(  thread_team_t& team 
                  , VEC( const int i 
                  ,      const int j 
                  ,      const int k)
                  , int ngz
                  , grace::flux_array_t const  fluxes) const 
    {
        getflux<2>(VEC(i,j,k),team.league_rank(),ngz,fluxes); 
    }
    /**
     * @brief Compute geometric source terms for GRMHD equations.
     * 
     * @tparam thread_team_t Thread team type.
     * @param team Thread team.
     * @param i Cell index in \f$x^1\f$ direction.
     * @param j Cell index in \f$x^2\f$ direction.
     * @param k Cell index in \f$x^3\f$ direction.
     * @param state_new State where sources are added.
     * @param dt Timestep.
     * @param dtfact Timestep factor.
     */
    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_source_terms( thread_team_t& team 
                         , VEC( const int i 
                         ,      const int j 
                         ,      const int k)
                         , grace::var_array_t<GRACE_NSPACEDIM> const state_new
                         , double const dt 
                         , double const dtfact ) const ;
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
                        , int64_t q ) const ;
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
                          , int64_t q ) const ;

 private:
    //! Number of reconstructed variables.
    static constexpr unsigned int GRMHD_NUM_RECON_VARS = 7 ; 
    //! Equation of State object.
    eos_t _eos ;
    /**
     * @brief 
     * 
     * @tparam idir 
     * @param q 
     * @param ngz 
     * @param fluxes 
     */
    template< int idir >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    getflux(  VEC( const int i 
            ,      const int j 
            ,      const int k)
            , const int64_t q 
            , int ngz
            , grace::flux_array_t const fluxes) const 
    {
        recon_t reconstructor ; 
        riemann_t solver      ;
        /***********************************************************************/
        /* Define and interpolate metric                                       */
        /***********************************************************************/
        FILL_METRIC_ARRAY( metric_l, this->_aux, q
                         , VEC( i+ngz-utils::delta(idir,0)
                              , j+ngz-utils::delta(idir,1)
                              , k+ngz-utils::delta(idir,2))) ; 
        FILL_METRIC_ARRAY( metric_r, this->_aux, q
                         , VEC( i+ngz
                              , j+ngz
                              , k+ngz )) ;
        metric_array_t metric_center{
            { 0.5*(metric_l.gamma(0) + metric_r.gamma(0))
            , 0.5*(metric_l.gamma(1) + metric_r.gamma(1))
            , 0.5*(metric_l.gamma(2) + metric_r.gamma(2))
            , 0.5*(metric_l.gamma(3) + metric_r.gamma(3))
            , 0.5*(metric_l.gamma(4) + metric_r.gamma(4))
            , 0.5*(metric_l.gamma(5) + metric_r.gamma(5))}
        ,   { 0.5*(metric_l.shift(0) + metric_r.shift(0))
            + 0.5*(metric_l.shift(1) + metric_r.shift(1))
            + 0.5*(metric_l.shift(2) + metric_r.shift(2))}
        ,   0.5 * (metric_l.alp() + metric_r.alp())
        } ; 
        /***********************************************************************/
        /*              Reconstruct primitive variables                        */
        /***********************************************************************/
        std::array<size_t, GRMHD_NUM_RECON_VARS>
            recon_indices{
                  RHO_
                , VELX_
                , VELY_
                , VELZ_
                , YE_
                , TEMP_
                , ENTROPY_
            } ; 
        
        std::array<size_t, GRMHD_NUM_RECON_VARS>
            recon_indices_loc{
                  RHOL
                , VXL
                , VYL
                , VZL
                , YEL
                , TEMPL
                , ENTL
            } ;
        grmhd_prims_array_t primL, primR ; 
        for( int i=0; i<GRMHD_NUM_RECON_VARS; ++i) {
            auto u = Kokkos::subview( this->_aux
                                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()) 
                                    , recon_indices[i] 
                                    , q ) ;   
            reconstructor( u, VEC(i+ngz,j+ngz,k+ngz)
                         , primL[recon_indices_loc[i]]
                         , primR[recon_indices_loc[i]] 
                         , idir) ;
        }
        /***********************************************************************/
        /* Compute u0 on both sides                                            */
        /***********************************************************************/
        /* Lorentz factors */
        double const wl   = Kokkos::sqrt(1. 
            + metric_center.square_vec({primL[VXL], primL[VYL], primL[VZL]}));
        double const wr   = Kokkos::sqrt(1. 
            + metric_center.square_vec({primR[VXL], primR[VYL], primR[VZL]}));
        /* u^0             */
        double const u0_l = wl / metric_center.alp() ; 
        double const u0_r = wr / metric_center.alp() ; 
        /***********************************************************************/
        /* Fill up primitive array on both sides of the face.                  */
        /* Right now we have:                                                  */
        /* 1) The correct rho                                                  */
        /* 2) No pressure (computed below)                                     */
        /* 3) The temperature in place of eps (swapped below)                  */
        /* 4) The z vector (w v_{n}^i) as opposed to v^i (swapped below)       */
        /***********************************************************************/
        double const alp = metric_center.alp() ; 
        /* Left */
        double epsl, epsr ;
        double cs2l, cs2r ; 
        primL[PRESSL] = _eos.press_eps_csnd2__temp_rho_ye(epsl, cs2l, primL[TEMPL], primL[RHOL], primL[YEL], eos_err) ; 
        primL[EPSL]   = epsl ;
        primL[VXL] = alp * primL[VXL] / wl - metric_center.shift(0) ;
        primL[VYL] = alp * primL[VYL] / wl - metric_center.shift(1) ;
        primL[VZL] = alp * primL[VZL] / wl - metric_center.shift(2) ; 
        z_to_v(primL[VXL], primL[VYL], primL[VZL], metric_face) ; 
        /* Right */
        primR[PRESSL] = _eos.press_eps_csnd2__temp_rho_ye(epsr, cs2r, primR[EPSL], primR[RHOL], primR[YEL], eos_err) ; 
        primR[EPSL]   = epsr ; 
        primR[VXL] = alp * primR[VXL] / wr - metric_center.shift(0) ;
        primR[VYL] = alp * primR[VYL] / wr - metric_center.shift(1) ;
        primR[VZL] = alp * primR[VZL] / wr - metric_center.shift(2) ; 
        /* Compute small b */
        std::array<double,4> smallbL{0,0,0,0}, smallbR{0,0,0,0} ; 
        double b2l{0.}, b2r{0.}; 
        //compute_smallb(smallbL, b2l, primL, metric_center) ; 
        //compute_smallb(smallbR, b2r, primR, metric_center) ; 
        /* Compute Alfven speeds */
        double v02r,v02l, h_r,h_l;
        compute_v02(h_l, v02l, cs2l, b2l, primL) ; 
        compute_v02(h_r, v02r, cs2r, b2r, primR) ;
        /* Get wavespeeds      */
        double const one_over_alp2 = 1./math::int_pow<2>(metric_face.alp()); 
        double cpr, cmr, cpl, cml;
        int metric_component = (idir==0 ? 0 : (idir==1 ? 3 : 5)) ;
        compute_cp_cm( cpl, cml, v02l, u0_l, primL[VXL+idir], one_over_alp2
                     , metric_face.shift(idir), metric_face.gammainv(metric_component)) ;
        compute_cp_cm( cpr, cmr, v02r, u0_r, primR[VXL+idir], one_over_alp2
                     , metric_face.shift(idir), metric_face.gammainv(metric_component)) ;
        double const cmin = -math::min(0, math::min(cml,cmr)) ; 
        double const cmax =  math::max(0, math::max(cpl,cpr)) ; 
        /***********************************************************************/
        /*                          Get dens flux                              */
        /***********************************************************************/
        double const alpha_sqrtgamma = metric_center.alp() * metric_center.sqrtg() ;
        double const dens_l = alpha_sqrtgamma * primL[RHOL] * u0_l ;
        double const dens_r = alpha_sqrtgamma * primR[RHOL] * u0_r ;

        double fl = dens_l * primL[VXL+idir] ; 
        double fr = dens_r * primR[VXL+idir] ; 

        fluxes(VEC(i,j,k),DENS_,idir,q) = solver(fl,fr,dens_l,dens_r,cmin,cmax) ; 
        /***********************************************************************/
        /*                          Get ye_star flux                           */
        /***********************************************************************/
        double const ye_star_l = dens_l * primL[YEL] ; 
        double const ye_star_r = dens_r * primR[YEL] ; 
        
        fl = ye_star_l * primL[VXL+idir] ; 
        fr = ye_star_l * primR[VXL+idir] ; 

        fluxes(VEC(i,j,k),YESTAR_,idir,q) = solver(fl,fr,ye_star_l,ye_star_r,cmin,cmax) ;
        /***********************************************************************/
        /*                          Get s_star flux                            */
        /***********************************************************************/
        double const s_star_l = dens_l * primL[ENTL] ; 
        double const s_star_r = dens_r * primR[ENTL] ; 

        fl = s_star_l * primL[VXL+idir] ; 
        fr = s_star_r * primR[VXL+idir] ; 

        fluxes(VEC(i,j,k),ENTROPYSTAR_,idir,q) = solver(fl,fr,s_star_l,s_star_r,cmin,cmax) ;
        /***********************************************************************/ 
        /*                           Get tau flux                              */
        /***********************************************************************/
        /* Auxiliary metric quantitites */
        double const alp2_sqrtgamma = math::int_pow<2>(metric_center.alp()) * metric_center.sqrtgamma() ; 
        double const g4uptd = one_over_alp2 * metric_center.beta(idir) ; 
        double const g4uptt = one_over_alp2 ; 
        /* Left flux */
        double const rho0_h_plus_b2_l = (primL[RHOL]*(1+primL[EPSL])) + primL[PRESSL] + b2l ;
        double const P_plus_half_b2_l = (primL[PRESSL] + 0.5*b2l);
        double const TUPtd_l = rho0_h_plus_b2_l * math::int_pow<2>(u0_l) * primL[VXL+idir] 
            + P_plus_half_b2_l*g4uptd - smallbL[0]*smallbL[1+idir] ; 
        fl = alp2_sqrtgamma * TUPtd_l - dens_l * primL[VXL+idir] ;
        double const Tuptt_l = rho0_h_plus_b2_l*math::int_pow<2>(u0_l) 
            + P_plus_half_b2_l*g4uptt - math::int_pow<2>(smallbL[0])
        double const tau_l = alp2_sqrtgamma * TUPtt_l - dens_l ;
        /* Right flux */
        double const rho0_h_plus_b2_r = (primR[RHOL]*(1+primR[EPSL])) + primR[PRESSL] + b2r ;
        double const P_plus_half_b2_r = (primR[PRESSL] + 0.5*b2r);
        double const TUPtd_r = rho0_h_plus_b2_r * math::int_pow<2>(u0_r) * primR[VXL+idir] 
            + P_plus_half_b2_r*g4uptd - smallbR[0]*smallbR[1+idir] ; 
        fr = alp2_sqrtgamma * TUPtd_r - dens_r * primR[VXL+idir] ;
        double const Tuptt_r = rho0_h_plus_b2_r*math::int_pow<2>(u0_r) 
            + P_plus_half_b2_r*g4uptt - math::int_pow<2>(smallbR[0])
        double const tau_r = alp2_sqrtgamma * TUPtt_r - dens_r ; 

        fluxes(VEC(i,j,k),TAU_,idir,q) = solver(fl,fr,tau_l,tau_r,cmin,cmax) ; 
        /***********************************************************************/
        /* Momentum flux in direction d for S_j : \alpha \sqrt{\gamma} T^d_j   */
        /***********************************************************************/
        /* Compute u_i */
        auto uD_l = metric_center.lower({primL[VX],primL[VY],primL[VZ]}) ; 
        for(auto& uu: uD_l) uu /= wl ; 
        auto uD_r = metric_center.lower({primR[VX],primR[VY],primR[VZ]}) ; 
        for(auto& uu: uD_r) uu /= wr ; 
        /***********************************************************************/
        /* Get S_x flux                                                        */
        /***********************************************************************/
        std::array<double,3> smallbDL{0,0,0}, smallbDR{0,0,0,0};
        fl = alpha_sqrtgamma * ( rho0_h_plus_b2_l * (u0_l*primL[VXL+idir])*uD_l[0]
           + P_plus_half_b2_l*utils::delta(0,idir) - smallbL[idir+1]*smallbDL[0] ) ; 
        fr = alpha_sqrtgamma * ( rho0_h_plus_b2_r * (u0_r*primR[VXL+idir])*uD_r[0]
           + P_plus_half_b2_r*utils::delta(0,idir) - smallbR[idir+1]*smallbDR[0] ) ;  

        double const s_x_l = alpha_sqrtgamma * (rho0_h_plus_b2_l*u0_l*uD_l[0]-smallbL[0]*smallbDL[0]) ; 
        double const s_x_r = alpha_sqrtgamma * (rho0_h_plus_b2_r*u0_r*uD_r[0]-smallbR[0]*smallbDR[0]) ; 

        fluxes(VEC(i,j,k),SX_,idir,q) = solver(fl,fr,s_x_l,s_x_r,cmin,cmax) ; 
        /***********************************************************************/
        /* Get S_y flux                                                        */
        /***********************************************************************/
        fl = alpha_sqrtgamma * ( rho0_h_plus_b2_l * (u0_l*primL[VXL+idir])*uD_l[1]
           + P_plus_half_b2_l*utils::delta(1,idir) - smallbL[idir+1]*smallbDL[1] ) ; 
        fr = alpha_sqrtgamma * ( rho0_h_plus_b2_r * (u0_r*primR[VXL+idir])*uD_r[1]
           + P_plus_half_b2_r*utils::delta(1,idir) - smallbR[idir+1]*smallbDR[1] ) ;  

        double const s_y_l = alpha_sqrtgamma * (rho0_h_plus_b2_l*u0_l*uD_l[1]-smallbL[0]*smallbDL[1]) ; 
        double const s_y_r = alpha_sqrtgamma * (rho0_h_plus_b2_r*u0_r*uD_r[1]-smallbR[0]*smallbDR[1]) ; 

        fluxes(VEC(i,j,k),SY_,idir,q) = solver(fl,fr,s_y_l,s_y_r,cmin,cmax) ;
        /***********************************************************************/
        /* Get S_z flux                                                        */
        /***********************************************************************/
        fl = alpha_sqrtgamma * ( rho0_h_plus_b2_l * (u0_l*primL[VXL+idir])*uD_l[2]
           + P_plus_half_b2_l*utils::delta(2,idir) - smallbL[idir+1]*smallbDL[2] ) ; 
        fr = alpha_sqrtgamma * ( rho0_h_plus_b2_r * (u0_r*primR[VXL+idir])*uD_r[2]
           + P_plus_half_b2_r*utils::delta(2,idir) - smallbR[idir+1]*smallbDR[2] ) ;  

        double const s_z_l = alpha_sqrtgamma * (rho0_h_plus_b2_l*u0_l*uD_l[2]-smallbL[0]*smallbDL[2]) ; 
        double const s_z_r = alpha_sqrtgamma * (rho0_h_plus_b2_r*u0_r*uD_r[2]-smallbR[0]*smallbDR[2]) ; 

        fluxes(VEC(i,j,k),SZ_,idir,q) = solver(fl,fr,s_z_l,s_z_r,cmin,cmax) ; 
        /***********************************************************************/
        /***********************************************************************/
    };
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
               , grmhd_prims_array_t const& prims )
    {
        h = 1. + prims[EPSL] + prims[PRESSL] / prims[RHOL] ; 
        double const v_A_sq = b2 / ( b2 + prims[RHOL]*h) ; 
        v02 = v_A_sq + cs2 * ( 1. - v_A_sq ) ; 
    }

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
                 , double const& betad, double const& gupdd )
    {
        double const u0_sq = math::int_pow<2>(u0) ; 
        double const a = u0_sq * ( 1- v02 ) + v02 * one_over_alp2 ; 
        double const b = 2. * ( betad * one_over_alp2 * v02 - u0_sq * vd * (1. - v02 )) ; 
        double const c = u0_sq * math::int_pow<2>(vd) * ( 1.-v02 ) 
            - v02 * ( gupdd - math::int_pow<2>(betad)*one_over_alp2) ;
        double det = math::int_pow<2>(b) - 4. * a * c ; 
        det = Kokkos::sqrt(0.5*(det+Kokkos::fabs(det)) ) ; 

        double const c1 = 0.5*(det-b) / a ; 
        double const c2 = -0.5*(det+b) / a ; 
        cp = math::max(c1,c2) ; 
        cm = math::min(c1,c2) ; 
    }
} ; 

}

