/**
 * @file c2p.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-10
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

#include <grace/physics/eos/c2p.hh>
#include <grace/physics/eos/grhd_c2p.hh>

#include <Kokkos_Core.hpp>

#define C2P_TOLERANCE 10

namespace grace {

template< typename eos_t >
void GRACE_HOST_DEVICE
conservs_to_prims( grmhd_cons_array_t& cons 
                 , grmhd_prims_array_t& prims
                 , metric_array_t const& metric 
                 , eos_t const& eos
                 , double const& lapse_excision ) 
{
    using c2p_impl_t = grhd_c2p_t<eos_t> ;
    bool c2p_failed{ false }             ;
    double W                             ;
    /* Undensitize conservs */
    for( auto& c: cons) c /= metric.sqrtg() ;
    /* First we check whether we are in the atmosphere */
    auto const dens_atmo = eos.rho_atmosphere() ;
    if( cons[DENSL] > dens_atmo ) {
        c2p_impl_t c2p(eos,metric,cons) ;
        double residual ;
        prims =  c2p.invert(residual) ;
        c2p_failed = (math::abs(residual) > C2P_TOLERANCE) ;
        W = prims[PRESSL] ; // W was stored here for convenience
    } else {
        c2p_failed = true ;
    }
    if(   prims[RHOL] < (1.+1e-03) * dens_atmo
      or  c2p_failed
      or  metric.alp() < lapse_excision )
    {  
        prims[RHOL]  = dens_atmo ;
        prims[TEMPL] = eos.temp_atmosphere() ;
        prims[YEL]   = eos.ye_atmosphere()   ;
        prims[EPSL]  = eos.eps_atmosphere()  ; 
        prims[VXL]   = 0. ;
        prims[VYL]   = 0. ;
        prims[VZL]   = 0. ;
        W = 1. ;
    }
    /* Set pressure entropy and temperature */
    double h, csnd2;
    unsigned int err ;
    prims[PRESSL] = eos.press_h_csnd2_temp_entropy__eps_rho_ye(
        h,csnd2,prims[TEMPL],prims[ENTL],prims[EPSL],prims[RHOL],prims[YEL], err
    ) ;
    /* Go from z-vec to velocity and remove */
    /* shift contribution.                  */
    double const u0 = W / metric.alp() ;
    /* The 3-velocity in grace is not in the */
    /* ZAMO frame.                           */
    prims[VXL] = metric.alp()*prims[VXL] - metric.beta(0) ;
    prims[VYL] = metric.alp()*prims[VYL] - metric.beta(1) ;
    prims[VZL] = metric.alp()*prims[VZL] - metric.beta(2) ;
    /* Re-compute conservative variables based  */
    /* on new primitives.                       */
    prims_to_conservs(prims,cons,metric) ;
    /* Re-densitize conservs */
    //for( auto& c: cons) c *= metric.sqrtg() ;
}

void GRACE_HOST_DEVICE
prims_to_conservs( grace::grmhd_prims_array_t& prims
                 , grace::grmhd_cons_array_t& cons 
                 , grace::metric_array_t const& metric )
{
    std::array<double,3> vZAMO {
          (prims[VXL]+metric.beta(0))/metric.alp()
        , (prims[VYL]+metric.beta(1))/metric.alp()
        , (prims[VZL]+metric.beta(2))/metric.alp()
    } ; 
    double const v2 = metric.square_vec(vZAMO) ; 
    double const W  = 1./Kokkos::sqrt(1-v2) ; 
    double const u0 = W / metric.alp();
    double const alp_sqrtgamma = metric.alp() * metric.sqrtg() ;

    cons[DENSL] = alp_sqrtgamma * u0 * prims[RHOL] ; 

    double const b2{0.}, smallbt{0.} ; 
    double const one_over_alp2 = 1./math::int_pow<2>(metric.alp());
    double const rho0_h_plus_b2 = (prims[RHOL]*(1+prims[EPSL])) + prims[PRESSL] + b2 ;
    double const alp2_sqrtgamma = math::int_pow<2>(metric.alp()) * metric.sqrtg() ;
    double const g4uptt = -one_over_alp2 ; 
    
    double const P_plus_half_b2 = (prims[PRESSL] + 0.5*b2);
    double const Tuptt = rho0_h_plus_b2 * math::int_pow<2>(u0) + P_plus_half_b2 * g4uptt - math::int_pow<2>(smallbt) ; 
    cons[TAUL] = alp2_sqrtgamma * Tuptt - cons[DENSL] ;

    std::array<double,4> smallb{0.,0.,0.,0.}, smallbD{0.,0.,0.,0.} ; 
    auto uD = metric.lower({prims[VXL]+metric.beta(0),prims[VYL]+metric.beta(1),prims[VZL]+metric.beta(2)}) ; 
    for(auto & uu: uD) uu *= u0 ; 

    cons[STXL] = alp_sqrtgamma * (rho0_h_plus_b2*u0*uD[0]-smallb[0]*smallbD[1]) ; 
    cons[STYL] = alp_sqrtgamma * (rho0_h_plus_b2*u0*uD[1]-smallb[0]*smallbD[2]) ; 
    cons[STZL] = alp_sqrtgamma * (rho0_h_plus_b2*u0*uD[2]-smallb[0]*smallbD[3]) ;

    cons[YESL] = cons[DENSL] * prims[YEL] ;
    cons[ENTSL] = cons[DENSL] * prims[ENTL] ;

    return ; 
}

#define INSTANTIATE_TEMPLATE(EOS) \
template \
void GRACE_HOST_DEVICE \
conservs_to_prims<EOS>( grace::grmhd_cons_array_t&  \
                      , grace::grmhd_prims_array_t&  \
                      , grace::metric_array_t const&  \
                      , EOS const& eos \
                      , double const& ) 
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;  
#undef INSTANTIATE_TEMPLATE

}