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

#include <Kokkos_Core.hpp>

#define C2P_TOLERANCE 1e-12

namespace grace {

template< typename eos_t >
void conservs_to_prims( grmhd_cons_array_t& cons 
                      , grmhd_prims_array_t& prims
                      , metric_array_t const& metric 
                      , eos_t const& eos
                      , double const& lapse_excision ) 
{
    using c2p_impl_t = grhd_c2p_t ; 
    bool c2p_failed{ false }      ; 
    double W ; 
    /* Undensitize conservs */
    for( auto& c: cons) c /= metric.sqrtgamma() ; 
    /* First we check whether we are in the atmosphere */
    auto const dens_atmo = eos.rho_atmosphere() ; 
    if( cons[DENSL] > dens_atmo ) {
        c2p_impl_t c2p(eos,metric,cons) ; 
        double residual ;
        prims =  c2p.invert(residual) ; 
        c2p_failed = (residual > C2P_TOLERANCE) ; 
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
    prims[PRESSL] = eos.press_h_csnd2_temp_entropy__eps_rho_ye(
        h,csnd2,prims[TEMPL],prims[ENTL],prims[EPSL],prims[RHOL],prims[YEL]
    ) ; 
    /* Go from z-vec to velocity and remove */
    /* shift contribution.                  */
    double const u0 = W / metric.alp() ; 
    /* The 3-velocity in grace is not in the */
    /* ZAMO frame.                           */
    prims[VXL] = prims[VXL]/u0 - metric.beta(0) ;
    prims[VYL] = prims[VYL]/u0 - metric.beta(1) ;
    prims[VZL] = prims[VZL]/u0 - metric.beta(2) ;
    /* Re-compute conservative variables based  */
    /* on new primitives.                       */
    prims_to_conservs(prims,cons,metric,eos) ; 
    /* Re-densitize conservs */
    for( auto& c: cons) c *= metric.sqrtgamma() ; 
}

}