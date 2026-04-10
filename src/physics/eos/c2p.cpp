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
static void KOKKOS_INLINE_FUNCTION 
limit_conservatives(
    grace::grmhd_cons_array_t&  cons,
    metric_array_t const& metric,
    eos_t const& eos,
    c2p_err_t& err
) 
{ 
    
    double rhoL = cons[DENSL] ; 
    double yeL  = cons[YESL]  / (cons[DENSL]) ;  

    double epsmax,epsmin ; 
    eos_err_t eoserr; 
    eos.eps_range__rho_ye(epsmin,epsmax,rhoL,yeL,eoserr) ; 

    // note: this is not really staggered! 
    auto B2L = metric.square_vec({cons[BSXL],cons[BSYL],cons[BSZL]}) ; 
    // This is Etienne+2008 (A46) (https://arxiv.org/pdf/1112.0568)
    if ( cons[TAUL]/cons[DENSL] - 0.5 * B2L/cons[DENSL] < Kokkos::fmin(0.0, epsmin)) 
    {
        cons[TAUL] = epsmin * cons[DENSL] + 0.5 * B2L ; 
        err.set(c2p_err_enum_t::C2P_RESET_TAU) ; 
    }
    // This is the dominant energy condition, see 
    // e.g. Galeazzi+2014 (C13) (https://arxiv.org/pdf/1306.4953)
    double st2_max = SQR(cons[TAUL]/cons[DENSL] + 1) ; 
    auto st2L = metric.square_covec({cons[STXL],cons[STYL],cons[STZL]})/SQR(cons[DENSL]) ; 
    if ( st2L > st2_max ) {
        double const fact = sqrt(st2_max/st2L) ; 
        cons[STXL] *= fact ;
        cons[STYL] *= fact ;
        cons[STZL] *= fact ;
        err.set(c2p_err_enum_t::C2P_RESET_STILDE) ;  
    }

}

template< typename eos_t >
void GRACE_HOST_DEVICE
reset_to_atmosphere(
    grace::grmhd_cons_array_t&  cons,
    grace::grmhd_prims_array_t& prims,
    grace::metric_array_t const& metric,
    atmo_params_t const& atmo,
    eos_t const& eos,
    c2p_err_t& err  
)
{
    prims[RHOL]  = atmo.rho_fl ; 
    prims[YEL]   = atmo.ye_fl  ; 
    prims[TEMPL] = atmo.temp_fl ; 
    prims[ZXL]   = 0. ;
    prims[ZYL]   = 0. ;
    prims[ZZL]   = 0. ;
    // get pressure, eps and entropy
    double csnd2 ;  
    eos_err_t eos_err;  
    prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(
        prims[EPSL], csnd2, prims[ENTL], prims[TEMPL], prims[RHOL], prims[YEL], eos_err
    ); 
    // all conserved need to be reset 
    prims_to_conservs(prims,cons,metric) ; 
    err.set_all() ; 
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

    /* Guard against negative density */
    if ( cons[DENSL] < 0 ) {
        reset_to_atmosphere(
            cons,prims,metric,atmo,eos,c2p_err
        ) ; 
        return ; 
    }

    /* Check that the ye is within bounds */
    prims[YEL] = cons[YESL]/cons[DENSL] ; 
    double yemax = eos.get_c2p_ye_max();
    double yemin = eos.get_c2p_ye_min();
    if ( prims[YEL] > yemax ) {
        prims[YEL] = yemax ;
        c2p_err.set(c2p_err_enum_t::C2P_RESET_YE) ; 
    }
    if ( prims[YEL] < yemin ) {
        prims[YEL] = yemin ;
        c2p_err.set(c2p_err_enum_t::C2P_RESET_YE) ; 
    }

    /* Figure out if we are inside a bh   */
    /* in which case we should be lenient */
    bool c2p_is_lenient = (
        metric.alp() < c2p_pars.alp_bh_thresh 
    ) ; 

    // enforce limits on conserved variables
    limit_conservatives(cons,metric,eos,c2p_err) ; 
    
    /* If D>rho_atm we solve*/
    if( cons[DENSL] > atmo.rho_fl * (1+atmo.atmo_tol) ) { 
        // Call main c2p 
        c2p_mhd_t c2p(eos,metric,cons) ;
        double residual = c2p.invert(prims,c2p_ret) ;
        

        // decide if we need a backup. The criteria are as follows:
        // 1) C2P failed due to not finite residual or residual 
        //    exceeding the threshold 
        // 2) A severe exception was raised (rho>rho_max or eps>eps_max )
        // 3) The plasma beta is below the threshold 
        c2p_failed = (Kokkos::fabs(residual) > c2p_pars.tol) 
                    || (c2p_ret.test(c2p_sig_enum_t::C2P_RHO_TOO_HIGH)) 
                    || (c2p_ret.test(c2p_sig_enum_t::C2P_EPS_TOO_HIGH))
                    || (!Kokkos::isfinite(residual)) ;

        double beta = compute_beta(prims,metric) ; 
        
        // backup if needed and allowed 
        if ( (     c2p_failed 
                or beta <= c2p_pars.beta_fallback ) 
               and c2p_pars.use_ent_backup ) 
        {
            // reset the c2p signals since we are
            // trying again
            c2p_ret.reset() ; 
            // call the backup c2p 
            c2p_backup_t e_c2p(eos,metric,cons) ;
            double residual = e_c2p.invert(prims,c2p_ret) ; 
            // decide if we can accept the inversion 
            c2p_failed = (math::abs(residual) > c2p_pars.tol) || (!Kokkos::isfinite(residual)) ;
            c2p_failed |= (prims[EPSL] >= eos.get_c2p_eps_max()) ; 
        }
        // handle the return signals from within the 
        // c2p operators 
        c2p_handle_signals(c2p_ret, c2p_is_lenient, c2p_err) ; 
    } else {
        c2p_failed = true ;
    }

    // excision criterion 
    bool excise = excision.excise_by_radius     
                ? rtp[0] <= excision.r_ex 
                : metric.alp() <= excision.alp_ex ; 
    if ((prims[RHOL] < (1.+atmo.atmo_tol) * atmo.rho_fl) or c2p_failed or excise ) {
        // reset everything to atmosphere 
        reset_to_atmosphere(
            cons,prims,metric,atmo,eos,c2p_err
        ) ;
    } else if (prims[TEMPL] < atmo.temp_fl) {
        // NB: When using tabeos, if temp_fl == min(T_tab)
        // this check is effectively a no-op since 
        // the eos will never return T below the table bound.
        // nonetheless it's useful if either of the 
        // above does not hold.

        // In this case we only reset 
        // T, eps and press 
        prims[TEMPL] = atmo.temp_fl  ; 
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
        // reset all conserved except for D (and S_i), since rho and W are unchanged
        c2p_err.set(c2p_err_enum_t::C2P_RESET_TAU)     ; 
        //c2p_err.set(c2p_err_enum_t::C2P_RESET_STILDE)  ; stick to the convention used for eps fix
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
    /* NB: only conservatives flagged in the    */
    /* c2p errors are overwritten.              */
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
INSTANTIATE_TEMPLATE(grace::ideal_gas_eos_t) ;
#undef INSTANTIATE_TEMPLATE

}