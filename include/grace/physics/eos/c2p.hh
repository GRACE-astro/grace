/**
 * @file c2p.hh
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
#ifndef GRACE_PHYSICS_EOS_C2P_HH
#define GRACE_PHYSICS_EOS_C2P_HH

#include <grace_config.h>

#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/grmhd_helpers.hh>

/* These were previously only included in c2p.cpp */
#include <grace/physics/grmhd_subexpressions.hh>
#include <grace/physics/eos/grhd_c2p.hh>
#include <grace/physics/eos/kastaun_c2p.hh>
#include <grace/physics/eos/ent_based_c2p.hh>

#include <Kokkos_Core.hpp>

namespace grace {

// -----------------------------------------------------------------------
// Internal helpers (file-local linkage via static)
// -----------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
static double
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
KOKKOS_INLINE_FUNCTION
static void
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

// -----------------------------------------------------------------------
// Public interface declarations
// -----------------------------------------------------------------------

/**
 * @brief Convert conservative variables to primitive ones.
 *
 * @tparam eos_t Type of EOS.
 * @param cons Conservative variables (at one cell).
 * @param prims Primitive variables (at one cell).
 * @param metric Metric utilities.
 * @param eos Equation of State.
 * @param atmo Atmosphere parameters.
 * @param excision Excision parameters.
 * @param c2p_pars C2P solver parameters.
 * @param rtp Radial/angular coordinates of the cell.
 * @param c2p_err Error/status flags for conserved variable adjustment.
 *
 * Atmosphere conditions are enforced by this routine.
 * Conserved variables are recomputed to be consistent with inverted
 * primitives.
 */
template< typename eos_t >
KOKKOS_FUNCTION
void conservs_to_prims( grace::grmhd_cons_array_t&
                      , grace::grmhd_prims_array_t&
                      , grace::metric_array_t const&
                      , eos_t const& eos
                      , atmo_params_t const& atmo
                      , excision_params_t const& excision
                      , c2p_params_t const& c2p_pars
                      , double * rtp
                      , c2p_err_t& c2p_err ) ;

KOKKOS_FUNCTION
void prims_to_conservs( grace::grmhd_prims_array_t& prims
                      , grace::grmhd_cons_array_t& cons
                      , grace::metric_array_t const& metric ) ;

// -----------------------------------------------------------------------
// Definitions (kept in the header so every TU that includes this header
// gets device-callable code, regardless of CUDA / HIP / serial backend)
// -----------------------------------------------------------------------

template< typename eos_t >
KOKKOS_FUNCTION
void conservs_to_prims(  grace::grmhd_cons_array_t&  cons
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
    using c2p_hydro_t  = kastaun_c2p_t<eos_t>      ;
    using c2p_backup_t = entropy_fix_c2p_t<eos_t>  ;
    bool c2p_failed{ false }                       ;

    // by default we overwrite S_star
    c2p_err.adjust_ent = true ;
    c2p_err.adjust_tau = c2p_err.adjust_d = c2p_err.adjust_s = false ;

    /* Undensitize conservs */
    for( auto& c: cons) c /= metric.sqrtg() ;

    /* Set B */
    /* NB now the cons contains   */
    /* Cell centered undensitized */
    /* B                          */
    prims[BXL] = cons[BSXL] ;
    prims[BYL] = cons[BSYL] ;
    prims[BZL] = cons[BSZL] ;

    /* limit conservative vars */
    //limit_conserved(cons,metric,atmo,eos,c2p_err) ;

    /* Get atmo rho and temp */
    auto const dens_atmo = atmo.rho_fl  * pow(rtp[0], atmo.rho_fl_scaling);
    auto const temp_atmo = atmo.temp_fl * pow(rtp[0], atmo.temp_fl_scaling);

    /* If D>rho_atm we solve */
    if( cons[DENSL] > dens_atmo ) {
        // initialize ret code
        c2p_sig_t c2p_ret{C2P_SUCCESS} ;
        // initialize beta
        double beta=1e100 ;
        // check if we need mhd c2p
        double hydro_thresh = 1e-15 * fmin(cons[TAUL]/cons[DENSL], 1e-5) ;
        double Btilde2 = metric.square_vec({cons[BSXL],cons[BSYL],cons[BSZL]}) / cons[DENSL] ;
        if ( Btilde2 < hydro_thresh ) {
            c2p_hydro_t c2p(eos,metric,cons) ;
            double residual = c2p.invert(prims,c2p_ret) ;
            c2p_failed = (math::abs(residual) > c2p_pars.tol) || (c2p_ret == C2P_EPS_TOO_HIGH);
        } else {
            c2p_mhd_t c2p(eos,metric,cons) ;
            double residual = c2p.invert(prims,c2p_ret) ;
            c2p_failed = (math::abs(residual) > c2p_pars.tol) || (c2p_ret == C2P_EPS_TOO_HIGH);
            beta = compute_beta(prims,metric) ;
        }

        if ( (     c2p_failed
                or beta <= c2p_pars.beta_fallback )
               and c2p_pars.use_ent_backup )
        {
            c2p_ret = C2P_SUCCESS ;
            c2p_err.adjust_ent = false ;
            c2p_err.adjust_tau = true  ;
            c2p_backup_t e_c2p(eos,metric,cons) ;
            double residual = e_c2p.invert(prims,c2p_ret) ;
            c2p_failed = (math::abs(residual) > c2p_pars.tol) || (c2p_ret == C2P_EPS_TOO_HIGH);
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
    } else {
        /* Limit lorentz factor and magnetization */
        limit_primitives<eos_t>(
            prims, metric, eos, c2p_pars.max_w, c2p_pars.max_sigma, c2p_err
        ) ;
    }

    /* Re-compute conservative variables based  */
    /* on new primitives.                       */
    prims_to_conservs(prims,cons,metric) ;
}

KOKKOS_FUNCTION
inline void prims_to_conservs( grace::grmhd_prims_array_t& prims
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
    return ;
}

} // namespace grace

#endif /* GRACE_PHYSICS_EOS_C2P_HH */