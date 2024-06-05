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
    EPSL,
    ENTL,
    BXL,
    BYL,
    BZL,
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

    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_source_terms( thread_team_t& team 
                         , VEC( const int i 
                         ,      const int j 
                         ,      const int k)
                         , grace::var_array_t<GRACE_NSPACEDIM> const state_new
                         , double const dt 
                         , double const dtfact ) const ;

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_auxiliaries(  VEC( const int i 
                        ,      const int j 
                        ,      const int k) 
                        , int64_t q ) const ;

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_max_eigenspeed( VEC( const int i 
                          ,      const int j 
                          ,      const int k) 
                          , int64_t q ) const ;

 private:
    static constexpr unsigned int GRMHD_NUM_RECON_VARS = 7 ; 
    
    eos_t _eos ;

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    z_to_v( double const& zx, double const& zy, double const& zz
          , double& vx, double& vy, double& vz, metric_array_t const& metric) const
    {

    }

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
        /* Define and interpolate metric */
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
        /* Step 1: reconstruction of primitive variables */
        std::array<size_t, GRMHD_NUM_RECON_VARS>
            recon_indices{
                  RHO_
                , VELX_
                , VELY_
                , VELZ_
                , YE_
                , EPS_
                , ENTROPY_
            } ; 
        std::array<double, GRMHD_NUM_RECON_VARS>
            _primL, _primR ; 
        reconstructor( this->_aux, VEC(i+ngz,j+ngz,k+ngz), q, recon_indices 
                     , _primL, _primR, idir ) ; 
        unsigned int eos_err; 
        /* Fill left and right prim arrays based on recon vars */
        grmhd_prims_array_t primL, primR ; 
        /* Left */
        primL[RHOL]   = _primL[RHOL] ; 
        primL[PRESSL] = _eos.press__eps_rho_ye(_primL[5], _primL[0], _primL[4], eos_err) ; 
        z_to_v(_primL[1], _primL[2], _primL[3], primL[VXL], primL[VYL], primL[VZL], metric_face) ; 
        primL[YEL] = _primL[4] ; 
        primL[ENTL] = _primL[6] ; 
        /* Right */
        primR[RHOL]   = _primR[RHOL] ;  
        primR[PRESSL] = _eos.press__eps_rho_ye(_primR[5], _primR[0], _primR[4], eos_err) ; 
        z_to_v(_primR[1], _primR[2], _primR[3], primR[VXL], primR[VYL], primR[VZL], metric_face) ; 
        primR[YEL] = _primR[4] ; 
        primR[ENTL] = _primR[6] ; 
        /* Compute u0 on both sides */
        double const zl   = Kokkos::sqrt(metric_center.square_vec({_primL[1],_primL[2],_primL[3]}));
        double const zr   = Kokkos::sqrt(metric_center.square_vec({_primR[1],_primR[2],_primR[3]}));
        double const u0_l = zl / metric_center.alp() ; 
        double const u0_r = zr / metric_center.alp() ; 
        /* Compute small b */
        std::array<double,4> smallbl, smallbr ; 
        double b2l, b2r; 
        compute_smallb(smallbl, b2l, primL, metric_center) ; 
        compute_smallb(smallbr, b2r, primR, metric_center) ; 
        /* Compute Alfven speeds */
        double v02r,v02l, h_r,h_l;
        compute_v02(h_l, v02l, primL, smallbL) ; 
        compute_v02(h_r, v02r, primR, smallbR) ;
        /* Get wavespeeds      */
        double const one_over_alp2 = 1./math::int_pow<2>(metric_face.alp()); 
        double cpr, cmr, cpl, cml;
        compute_cp_cm( cpl, cml, primL[VXL+idir], one_over_alp2
                     , u0_l, metric_face.gammainv(idir)) ;
        compute_cp_cm( cpr, cmr, primR[VXL+idir], one_over_alp2
                     , u0_r, metric_face.gammainv(idir)) ;
        double const cmin = -math::min(0, math::min(cml,cmr)) ; 
        double const cmax =  math::max(0, math::max(cpl,cpr)) ; 
        /* Get dens flux */
        double const alpha_sqrtgamma = metric_center.alp() * metric_center.sqrtg() ;
        double const dens_l = alpha_sqrtgamma * primL[RHOL] * u0_l ;
        double const dens_r = alpha_sqrtgamma * primR[RHOL] * u0_r ;

        double fl = dens_l * primL[VXL+idir] ; 
        double fr = dens_r * primR[VXL+idir] ; 

        fluxes(VEC(i,j,k),DENS_,idir,q) = solver(fl,fr,dens_l,dens_r,cmin,cmax) ; 
        /* Get ye_star flux */
        double const ye_star_l = dens_l * primL[YEL] ; 
        double const ye_star_r = dens_r * primR[YEL] ; 
        


    };


} ; 

}

