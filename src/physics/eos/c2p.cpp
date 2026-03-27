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
limit_primitives(
    grace::grmhd_prims_array_t&  prims,
    metric_array_t const& metric,
    eos_t const& eos,
    double max_w,
    double max_sigma,
    c2p_sig_t& sig
) {  
    // limit velocities 
    // w^2 = z^2 + 1 
    // but we limit z directly 
    double const z2max = SQR(max_w) - 1. ;
    double const z2 =  metric.square_vec({prims[ZXL],prims[ZYL],prims[ZZL]}) ; 
    if ( z2 >= z2max ) {
        double znorm = 0.99 * sqrt(z2max/z2) ; 
        prims[ZXL] *= znorm ; 
        prims[ZYL] *= znorm ; 
        prims[ZZL] *= znorm ; 
        // W has changed so the conserved are outdated
        sig.set(c2p_sig_enum_t::C2P_VEL_TOO_HIGH) ; 
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
        eos_err_t eos_err ; 
        double csnd2; 
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(
            prims[EPSL], csnd2, prims[ENTL], prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // Set the signal for the caller to handle 
        sig.set(c2p_sig_enum_t::C2P_SIGMA_TOO_HIGH) ; 
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
                  , c2p_params_t const& c2p_pars
                  , double * rtp
                  , c2p_err_t& c2p_err)
{
    
    using c2p_mhd_t    = kastaun_c2p_t<eos_t>     ;
    using c2p_backup_t = entropy_fix_c2p_t<eos_t> ;
    bool c2p_failed{ false }                      ;  

    // initialize ret code
    c2p_sig_t c2p_ret ;

    // by default we overwrite S_star 
    c2p_err.reset() ; 
    c2p_err.set(c2p_err_enum_t::C2P_RESET_ENTROPY) ; 

    /* Undensitize conservs */
    for( auto& c: cons) c /= metric.sqrtg() ;
    
    /* Set B */
    /* NB now the cons contains   */
    /* Cell centered undensitized */
    /* B                          */
    prims[BXL] = cons[BSXL] ; 
    prims[BYL] = cons[BSYL] ; 
    prims[BZL] = cons[BSZL] ; 

    /* Figure out if we are inside a bh   */
    /* in which case we should be lenient */
    bool c2p_is_lenient = (
        metric.sqrtg() > c2p_pars.psi6_bh_thresh 
    ) ; 
    // FIXME!!! Sometimes tabeos randomly puts one point in the 
    // ejecta at max eps, we don't want that to crash the whole 
    // simulation if possible. This is a hack, we need to find 
    // out why.


    /* Get atmo rho and temp */
    auto const dens_atmo =  atmo.rho_fl * pow(rtp[0], atmo.rho_fl_scaling);
    auto const temp_atmo =  atmo.temp_fl * pow(rtp[0], atmo.temp_fl_scaling);
    auto const ye_atmo   =  atmo.ye_fl ; 
    
    /* If D>rho_atm we solve*/
    if( cons[DENSL] > dens_atmo * (1+1e-2) ) { 
        // Call main c2p 
        c2p_mhd_t c2p(eos,metric,cons) ;
        double residual = c2p.invert(prims,c2p_ret) ;
        // handle return codes: 
        // if we are inside a black hole, or the backup is enabled, 
        // we allow more c2p errors without aborting, in the hope 
        // that the backup will fix them and / or that they don't 
        // affect the region outside the black hole 
        bool lenient = ( c2p_is_lenient || c2p_pars.use_ent_backup ) ; 
        c2p_handle_signals(c2p_ret, lenient, c2p_err) ; 
        // decide if we need a backup. The criteria are as follows:
        // 1) C2P failed due to not finite residual or residual 
        //    exceeding the threshold 
        // 2) A severe exception was raised (rho>rho_max or eps>eps_max)
        // 3) The plasma beta is below the threshold 
        c2p_failed = (Kokkos::fabs(residual) > c2p_pars.tol) 
                    || (c2p_ret.test(c2p_sig_enum_t::C2P_RHO_TOO_HIGH)) 
                    || (c2p_ret.test(c2p_sig_enum_t::C2P_EPS_TOO_HIGH))
                    || (!Kokkos::isfinite(residual)) ;
        // we enforce the user-provided limit here too.
        c2p_failed |= (prims[EPSL] >= eos.get_c2p_eps_max()) ; 
        
        double beta = compute_beta(prims,metric) ; 
        
        // backup if needed and allowed 
        if ( (     c2p_failed 
                or beta <= c2p_pars.beta_fallback ) 
               and c2p_pars.use_ent_backup ) 
        {
            // reset the c2p signals 
            c2p_ret.reset() ; 
            // reset the c2p errors,
            // since we are trying another
            // inversion 
            c2p_err.reset() ; 
            // call the backup c2p 
            c2p_backup_t e_c2p(eos,metric,cons) ;
            double residual = e_c2p.invert(prims,c2p_ret) ; 
            // test the returns 
            c2p_handle_signals(c2p_ret, c2p_is_lenient, c2p_err) ;
            // decide if we can accept the inversion 
            c2p_failed = (math::abs(residual) > c2p_pars.tol) || (!Kokkos::isfinite(residual)) ;
            c2p_failed |= (prims[EPSL] >= eos.get_c2p_eps_max()) ; 
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
        prims[YEL]   = excision.ye_ex   ;
        prims[ZXL]   = 0. ;
        prims[ZYL]   = 0. ;
        prims[ZZL]   = 0. ;
        prims[YEL]   = 0   ;
        // get pressure, eps and entropy
        double csnd2 ;  
        eos_err_t eos_err;  
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(
            prims[EPSL], csnd2, prims[ENTL], prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // no need to check EOS errs, everything is reset regardless 
        // reset all conserved
        c2p_err.set_all() ; 
    } else if ((prims[RHOL] < (1.+1e-2) * dens_atmo) or c2p_failed ) {
        prims[RHOL]  = dens_atmo  ; 
        prims[TEMPL] = temp_atmo  ; 
        prims[YEL]   = ye_atmo    ;
        prims[ZXL]   = 0. ;
        prims[ZYL]   = 0. ;
        prims[ZZL]   = 0. ;
        
        // get pressure, eps and entropy
        double csnd2 ;  
        eos_err_t eos_err;  
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(
            prims[EPSL], csnd2, prims[ENTL], prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // no need to check EOS errs, everything is reset regardless 
        // reset all conserved
        c2p_err.set_all() ; 
    } else if (prims[TEMPL] < temp_atmo) {
        // In this case we only reset 
        // T, eps and press 
        prims[TEMPL] = temp_atmo  ; 
        // get pressure, eps and entropy
        double csnd2 ;  
        eos_err_t eos_err;  
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(
            prims[EPSL], csnd2, prims[ENTL], prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // check if the EOS had any sort of problem 
        c2p_handle_eos_signals(
            eos_err,
            c2p_is_lenient,
            c2p_err
        ) ; 
        // reset all conserved except for D, since rho and W are unchanged
        c2p_err.set(c2p_err_enum_t::C2P_RESET_TAU)     ; 
        c2p_err.set(c2p_err_enum_t::C2P_RESET_STILDE)  ; 
        c2p_err.set(c2p_err_enum_t::C2P_RESET_ENTROPY) ; 
    } else {
        /* Limit lorentz fact and magnetization  */
        c2p_sig_t c2p_sig ; 
        limit_primitives<eos_t>(
            prims, metric, eos, c2p_pars.max_w, c2p_pars.max_sigma, c2p_sig
        ) ;
        c2p_handle_signals(c2p_sig, c2p_is_lenient, c2p_err) ; 
    }
    
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
                      , c2p_params_t const& c2p_pars \
                      , double * rtp \
                      , c2p_err_t& c2p_err ) 
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
#undef INSTANTIATE_TEMPLATE

}