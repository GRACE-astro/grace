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

#include <grace/physics/grmhd_subexpressions.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/eos/grhd_c2p.hh>
#include <grace/physics/eos/kastaun_c2p.hh>
#include <grace/physics/eos/ent_based_c2p.hh>

#include <Kokkos_Core.hpp>

namespace grace {

static double KOKKOS_FUNCTION 
compute_beta(
    grmhd_prims_array_t const& prims,
    metric_array_t const& metric
)
{
    double const * const betau = metric._beta.data() ; 
    double const * const gdd   = metric._g.data() ; 
    double const * const z     = &(prims[ZXL]) ; 
    double const * const B     = &(prims[BXL]) ; 
    double const alp{metric.alp()} ; 
    
    double W;
    grmhd_get_W(gdd,z,&W) ; 

    double smallbu[4];
    double smallb2;
    grmhd_get_smallbu_smallb2(
        betau,gdd,B,z,W,alp,&smallbu,&smallb2
    ) ; 
    return 2.0 * prims[PRESSL]/fmax(smallb2, 1e-50) ; 
}

template< typename eos_t >
static void KOKKOS_INLINE_FUNCTION 
limit_conserved(
    grace::grmhd_cons_array_t&  cons,
    metric_array_t const& metric,
    eos_t const& eos,
    c2p_err_t& c2p_err
)
{
    // Tau cannot be lower than
    // D * (1 + eps_min) + B^2 / 2 - D
    // FIXME here we assume eps_min == 0
    double const tau_floor = metric.square_vec({cons[BSXL],cons[BSYL],cons[BSZL]}) ; 
    if ( cons[TAUL] < tau_floor ) {
        c2p_err.adjust_tau = true ; 
        cons[TAUL] = tau_floor ;
    }
}

template< typename eos_t >
static void KOKKOS_INLINE_FUNCTION 
limit_primitives(
    grace::grmhd_prims_array_t&  prims,
    metric_array_t const& metric,
    eos_t const& eos,
    double max_w,
    double max_sigma,
    c2p_err_t& c2p_err
)
{  
    // limit velocities 
    double const W2 = 1 + metric.square_vec({prims[ZXL],prims[ZYL],prims[ZZL]}) ; 
    if ( W2 >= max_w*max_w ) {
        double znorm = 0.99 * max_w / sqrt(W2) ; 
        prims[ZXL] *= znorm ; 
        prims[ZYL] *= znorm ; 
        prims[ZZL] *= znorm ; 
        // W has changed so the conserved are outdated
        c2p_err.adjust_s = c2p_err.adjust_tau = c2p_err.adjust_ent = c2p_err.adjust_d = true; 
    }
    // limit magnetization 
    double const * const betau = metric._beta.data() ; 
    double const * const gdd   = metric._g.data() ; 
    double const * const z     = &(prims[ZXL]) ; 
    double const * const B     = &(prims[BXL]) ; 
    double const alp{metric.alp()} ; 
    
    double W;
    grmhd_get_W(gdd,z,&W) ; 

    double smallbu[4];
    double smallb2;
    grmhd_get_smallbu_smallb2(
        betau,gdd,B,z,W,alp,&smallbu,&smallb2
    ) ;
    double const sigma = smallb2 / prims[RHOL] ; 
    if ( sigma >= max_sigma ) {
        // add some density here! 
        double const rhofact = 1.001 * sigma/max_sigma ;
        prims[RHOL] *= rhofact ; 
        // recompute other prims 
        unsigned int eos_err ; 
        double csnd2; 
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(
            prims[EPSL], csnd2, prims[ENTL], prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // rho, P and entropy have changed so the conserved are outdated
        c2p_err.adjust_s = c2p_err.adjust_tau = c2p_err.adjust_ent = c2p_err.adjust_d = true; 
    }
}

template< typename eos_t >
void GRACE_HOST_DEVICE
conservs_to_prims(  grace::grmhd_cons_array_t&  cons
                  , grace::grmhd_prims_array_t& prims
                  , grace::metric_array_t const& metric
                  , eos_t const& eos 
                  , atmo_params_t const& atmo 
                  , excision_params_t const& excision 
                  , double * rtp
                  , c2p_err_t& c2p_err)
{
    
    using c2p_impl_t = kastaun_c2p_t<eos_t> ;
    using c2p_backup_t = entropy_fix_c2p_t<eos_t> ;
    bool c2p_failed{ false }             ;

    // by default we overwrite S_star 
    c2p_err.adjust_ent = true ; 
    c2p_err.adjust_tau = c2p_err.adjust_d = c2p_err.adjust_s = false ; 

    // store 
    const double c2p_tolerance = atmo.c2p_tol ; 

    /* Undensitize conservs */
    for( auto& c: cons) c /= metric.sqrtg() ;
    
    /* Set B */
    /* NB now the cons contains   */
    /* Cell centered undensitized */
    /* B                          */
    prims[BXL] = cons[BSXL] ; 
    prims[BYL] = cons[BSYL] ; 
    prims[BZL] = cons[BSZL] ; 

    /* Get atmo rho and temp */
    auto const dens_atmo =  atmo.rho_fl * pow(rtp[0], atmo.rho_fl_scaling);
    auto const temp_atmo =  atmo.temp_fl * pow(rtp[0], atmo.temp_fl_scaling);
    
    /* If D>rho_atm we solve*/
    if( cons[DENSL] > dens_atmo ) {
        c2p_sig_t c2p_ret ; 
        c2p_impl_t c2p(eos,metric,cons) ;
        double residual = c2p.invert(prims,c2p_ret) ;
        c2p_failed = (math::abs(residual) > c2p_tolerance) ;
        double beta = compute_beta(prims,metric) ; 
        if ( (     c2p_failed 
                or beta <= atmo.beta_fallback
                or c2p_ret == C2P_EPS_TOO_HIGH ) 
            and atmo.use_ent_backup ) 
        {
            c2p_err.adjust_ent = false ; 
            c2p_err.adjust_tau = true  ; 
            c2p_backup_t e_c2p(eos,metric,cons) ;
            double residual = e_c2p.invert(prims,c2p_ret) ; 
            c2p_failed = (math::abs(residual) > c2p_tolerance) ;
        }
    } else {
        c2p_failed = true ;
    }

    // now we check for atmo / excision 

    // excision criterion 
    bool excise = excision.excise_by_radius     
                ? rtp[0] <= excision.r_ex 
                : metric.alp() <= excision.alp_ex ; 
    if ( excise ) {
        prims[RHOL]  = excision.rho_ex  ; 
        prims[TEMPL] = excision.temp_ex ; 
        prims[ZXL]   = 0. ;
        prims[ZYL]   = 0. ;
        prims[ZZL]   = 0. ;
        prims[YEL]   = 0   ;
        // get pressure, eps and entropy
        double csnd2 ;  
        unsigned int eos_err;  
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(
            prims[EPSL], csnd2, prims[ENTL], prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // reset all conserved
        c2p_err.adjust_tau = c2p_err.adjust_d = c2p_err.adjust_s = c2p_err.adjust_ent = true ; 
    } else if ((prims[RHOL] < (1.+1e-03) * dens_atmo) or c2p_failed ) {
        prims[RHOL]  = dens_atmo  ; 
        prims[TEMPL] = temp_atmo  ; 
        prims[ZXL]   = 0. ;
        prims[ZYL]   = 0. ;
        prims[ZZL]   = 0. ;
        prims[YEL]   = 0. ;
        // get pressure, eps and entropy
        double csnd2 ;  
        unsigned int eos_err;  
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(
            prims[EPSL], csnd2, prims[ENTL], prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // reset all conserved
        c2p_err.adjust_tau = c2p_err.adjust_d = c2p_err.adjust_s = c2p_err.adjust_ent = true ; 
    } else if (prims[TEMPL] < temp_atmo) {
        // In this case we only reset 
        // T, eps and press 
        prims[TEMPL] = temp_atmo  ; 
        // get pressure, eps and entropy
        double csnd2 ;  
        unsigned int eos_err;  
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(
            prims[EPSL], csnd2, prims[ENTL], prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // reset all conserved except for D, since rho and W are unchanged
        c2p_err.adjust_tau = c2p_err.adjust_s = c2p_err.adjust_ent = true ; 
    } 
    #if 1
    else {
        /* Limit lorentz fact and magnetization  */
        limit_primitives<eos_t>(
            prims, metric, eos, atmo.max_w, atmo.max_sigma, c2p_err
        ) ;
    }
    #endif 
    
    /* Re-compute conservative variables based  */
    /* on new primitives.                       */
    prims_to_conservs(prims,cons,metric) ;
}

void GRACE_HOST_DEVICE
prims_to_conservs( grace::grmhd_prims_array_t& prims
                 , grace::grmhd_cons_array_t& cons 
                 , grace::metric_array_t const& metric )
{
    double const * const betau = metric._beta.data() ; 
    double const * const gdd   = metric._g.data() ; 
    double const * const z     = &(prims[ZXL]) ; 
    double const * const B     = &(prims[BXL]) ; 
    double const alp{metric.alp()} ; 
    
    double W;
    grmhd_get_W(gdd,z,&W) ; 

    double smallbu[4];
    double smallb2;
    grmhd_get_smallbu_smallb2(
        betau,gdd,B,z,W,alp,&smallbu,&smallb2
    ) ; 

    double D,tau,sstar ;
    double Stilde[3] ; 
    grmhd_get_conserved(
        W, prims[RHOL], smallbu, smallb2,
        alp, prims[EPSL], prims[PRESSL],
        betau, z, gdd, prims[ENTL],
        &D, &tau, &Stilde, &sstar
    ) ; 

    double const sqrtg = metric.sqrtg() ; 
    cons[DENSL] = sqrtg * D ; 
    cons[STXL]  = sqrtg * Stilde[0];
    cons[STYL]  = sqrtg * Stilde[1];
    cons[STZL]  = sqrtg * Stilde[2];
    cons[TAUL]  = sqrtg * tau ;
    cons[ENTSL] = sqrtg * sstar ; 
    cons[YESL]  = cons[DENSL] * prims[YEL] ;
    ////
    return ; 
}

#define INSTANTIATE_TEMPLATE(EOS) \
template \
void GRACE_HOST_DEVICE \
conservs_to_prims<EOS>( grace::grmhd_cons_array_t&  \
                      , grace::grmhd_prims_array_t&  \
                      , grace::metric_array_t const&  \
                      , EOS const& eos \
                      , atmo_params_t const& atmo \
                      , excision_params_t const& excision \
                      , double * rtp \
                      , c2p_err_t& c2p_err ) 
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE

}