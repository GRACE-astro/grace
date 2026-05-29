/**
 * @file c2p.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Shared c2p signal / error bitsets and post-recovery handlers (resets, atmosphere fall-back, EOS-level signal dispatch).
 * @date 2024-06-10
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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
#include <grace/physics/eos/eos_types.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/utils/bitset.hh>

namespace grace {

enum c2p_sig_enum_t : uint8_t {
    C2P_EPS_TOO_HIGH=0,
    C2P_EPS_TOO_LOW,
    C2P_YE_TOO_LOW,
    C2P_YE_TOO_HIGH,
    C2P_RHO_TOO_HIGH,
    C2P_RHO_TOO_LOW,
    C2P_ENT_TOO_LOW,
    C2P_ENT_TOO_HIGH,
    C2P_VEL_TOO_HIGH,
    C2P_SIGMA_TOO_HIGH,
    C2P_NSIG
} ; 

// c2p_err bit layout:
//   bits 0-4:  reset-which-conserved flags. These drive the conservative
//              rewrite logic in compute_auxiliaries — DO NOT renumber.
//   bits 5-18: raw-signal mirror bits. Set whenever the corresponding
//              c2p_sig or eos_err fired during a c2p call. Pure diagnostic
//              — they do not affect the reset logic. Sticky-OR'd over the
//              full timestep into aux(C2P_ERR_) so downstream analysis
//              tools can decode WHICH failure modes triggered the resets.
//   bit 19:    entropy-backup-used. Set when the primary energy-based c2p
//              inversion was distrusted and the entropy-backup path
//              successfully recovered the primitives.
//   bit 20:    atmosphere-reset. Set when the cell fell all the way through
//              to reset_to_atmosphere — primary c2p failed AND backup didn't
//              save it, OR rho landed in the atmosphere range, OR excised.
//   bit 21:    T-floored. Set when the cell entered the T < temp_fl branch:
//              rho was fine but T fell below the atmosphere temperature, so
//              T (and derived eps, press, entropy) were reset while velocity
//              was preserved. Disjoint from C2P_ATMO_RESET — that branch is
//              the full atmosphere reset which also zeros velocity.
//
// Together bits 19-21 distinguish four diagnostic outcomes per cell:
//   - all clear:                       clean c2p (any SIG_* bits = quiet clamp)
//   - bit 21 set (only):               T-only floor (velocity preserved)
//   - bit 19 set, bits 20-21 clear:    backup saved a distrusted inversion
//   - bit 20 set:                      hard atmosphere reset (most severe)
// And the reset/sig bits explain WHY in each case.
//
// To decode aux(C2P_ERR_) in Python:
//   v = int(aux[..., C2P_ERR_, ...])
//   reset_dens          = bool(v & (1<<0))
//   sig_eps_too_low     = bool(v & (1<<7))
//   entropy_backup_used = bool(v & (1<<19))
//   atmosphere_reset    = bool(v & (1<<20))
//   t_floored           = bool(v & (1<<21))
//   ...
//
// Total bits used: 22 — comfortably below double mantissa (2^53).
enum c2p_err_enum_t : uint8_t {
    C2P_RESET_DENS=0,
    C2P_RESET_TAU,
    C2P_RESET_STILDE,
    C2P_RESET_ENTROPY,
    C2P_RESET_YE,
    // Signal-mirror bits begin. Pure diagnostic; not consumed by the
    // conservative-reset dispatch in compute_auxiliaries.
    C2P_SIG_RHO_TOO_LOW,
    C2P_SIG_RHO_TOO_HIGH,
    C2P_SIG_EPS_TOO_LOW,
    C2P_SIG_EPS_TOO_HIGH,
    C2P_SIG_YE_TOO_LOW,
    C2P_SIG_YE_TOO_HIGH,
    C2P_SIG_ENT_TOO_LOW,
    C2P_SIG_ENT_TOO_HIGH,
    C2P_SIG_TEMP_TOO_LOW,
    C2P_SIG_TEMP_TOO_HIGH,
    C2P_SIG_PRESS_TOO_LOW,
    C2P_SIG_PRESS_TOO_HIGH,
    C2P_SIG_VEL_TOO_HIGH,
    C2P_SIG_SIGMA_TOO_HIGH,
    // c2p outcome bits — set by the dispatch logic in conservs_to_prims,
    // not the per-signal handlers. Use these to distinguish "primary c2p
    // succeeded with quiet flooring" from "primary c2p failed and the
    // entropy backup recovered the cell" from "atmosphere reset fired".
    C2P_ENT_BACKUP_USED,
    C2P_ATMO_RESET,
    C2P_T_FLOORED,
    C2P_N_ERR
} ;

using c2p_sig_t = bitset_t<C2P_NSIG> ;

using c2p_err_t = bitset_t<C2P_N_ERR> ;

// Internal helper: set just the five "reset conservatives" bits, leaving
// the diagnostic SIG_* and outcome bits untouched. Replaces the historic
// bitset_t::set_all() calls that predated the diagnostic mirror bits and
// would otherwise spuriously mark every atmosphere reset / lenient-rho-clamp
// as having triggered every failure mode.
KOKKOS_INLINE_FUNCTION
void c2p_set_all_resets(c2p_err_t& err)
{
    err.set(C2P_RESET_DENS)    ;
    err.set(C2P_RESET_TAU)     ;
    err.set(C2P_RESET_STILDE)  ;
    err.set(C2P_RESET_ENTROPY) ;
    err.set(C2P_RESET_YE)      ;
}

/**
 * @brief Handles c2p signals coming from
 *        the low level routines and decides which
 *        conserved variables need to be changed.
 *        Also folds the raw signal bits into the err's
 *        diagnostic SIG_* mirror slots so downstream can
 *        decode which failure modes triggered the resets.
 *        NB: Might abort on fatal errors!
 *
 * @tparam eos_t EOS type. Used to compile-time gate Y_e handling
 *               via has_ye_v<eos_t> — composition-free EOSes don't
 *               carry meaningful Y_e info, so the corresponding
 *               signals fire stochastically and should be ignored.
 */
template <typename eos_t>
void KOKKOS_INLINE_FUNCTION c2p_handle_signals(
    c2p_sig_t const& sig,
    bool be_lenient,
    c2p_err_t& err
)
{
    if ( sig.test(C2P_RHO_TOO_HIGH) ) {
        err.set(C2P_SIG_RHO_TOO_HIGH) ;
        if ( be_lenient ) {
            c2p_set_all_resets(err) ;
        } else {
            Kokkos::abort("In c2p: maximum density exceeded") ;
        }
    }
    if ( sig.test(C2P_RHO_TOO_LOW) ) {
        // if rho was modified everything has to change
        err.set(C2P_RESET_DENS)    ;
        err.set(C2P_RESET_TAU)     ;
        err.set(C2P_RESET_STILDE)  ;
        err.set(C2P_RESET_ENTROPY) ;
        if constexpr (has_ye_v<eos_t>) err.set(C2P_RESET_YE) ;
        err.set(C2P_SIG_RHO_TOO_LOW) ;
    }

    if constexpr (has_ye_v<eos_t>) {
        if (sig.test(C2P_YE_TOO_LOW))  { err.set(C2P_RESET_YE); err.set(C2P_SIG_YE_TOO_LOW)  ; }
        if (sig.test(C2P_YE_TOO_HIGH)) { err.set(C2P_RESET_YE); err.set(C2P_SIG_YE_TOO_HIGH) ; }
    }

    if (sig.test(C2P_ENT_TOO_LOW))  { err.set(C2P_RESET_ENTROPY); err.set(C2P_SIG_ENT_TOO_LOW)  ; }
    if (sig.test(C2P_ENT_TOO_HIGH)) { err.set(C2P_RESET_ENTROPY); err.set(C2P_SIG_ENT_TOO_HIGH) ; }

    if (sig.test(C2P_EPS_TOO_HIGH)) {
        // ε changed → h changed → S̃_i = ρhW²v_i must follow τ, otherwise
        // the conserved momentum is inconsistent with the clamped primitives
        // and c2p drifts on subsequent steps.
        err.set(C2P_RESET_STILDE) ;
        err.set(C2P_RESET_TAU)    ;
        err.set(C2P_SIG_EPS_TOO_HIGH) ;
    }

    if (sig.test(C2P_EPS_TOO_LOW)) {
        err.set(C2P_RESET_STILDE) ;
        err.set(C2P_RESET_TAU)    ;
        err.set(C2P_SIG_EPS_TOO_LOW) ;
    }

    if (sig.test(C2P_VEL_TOO_HIGH)) {
        // W enters everything
        err.set(C2P_RESET_DENS)    ;
        err.set(C2P_RESET_TAU)     ;
        err.set(C2P_RESET_STILDE)  ;
        err.set(C2P_RESET_ENTROPY) ;
        if constexpr (has_ye_v<eos_t>) err.set(C2P_RESET_YE) ;
        err.set(C2P_SIG_VEL_TOO_HIGH) ;
    }

    // magnetization
    if ( sig.test(C2P_SIGMA_TOO_HIGH) ) {
        // rho enters everything
        err.set(C2P_RESET_DENS)    ;
        err.set(C2P_RESET_TAU)     ;
        err.set(C2P_RESET_STILDE)  ;
        err.set(C2P_RESET_ENTROPY) ;
        if constexpr (has_ye_v<eos_t>) err.set(C2P_RESET_YE) ;
        err.set(C2P_SIG_SIGMA_TOO_HIGH) ;
    }

}

/**
 * @brief Handles eos signals coming from
 *        within the c2p routines and decides which
 *        conserved variables need to be changed.
 *        Also folds the raw EOS-signal bits into the err's
 *        diagnostic SIG_* mirror slots.
 *        NB: Might abort on fatal errors!
 *
 * @tparam eos_t EOS type. Used to compile-time gate Y_e handling
 *               via has_ye_v<eos_t>.
 */
template <typename eos_t>
void KOKKOS_INLINE_FUNCTION c2p_handle_eos_signals(
    eos_err_t const& eos_err,
    bool be_lenient,
    c2p_err_t& err
) {
    if ( eos_err.test(EOS_RHO_TOO_HIGH) ) {
        err.set(C2P_SIG_RHO_TOO_HIGH) ;
        Kokkos::abort("In EOS: maximum density exceeded") ;
    }

    if (eos_err.test(EOS_RHO_TOO_LOW)) {
        err.set(C2P_RESET_DENS)    ;
        err.set(C2P_RESET_TAU)     ;
        err.set(C2P_RESET_STILDE)  ;
        err.set(C2P_RESET_ENTROPY) ;
        if constexpr (has_ye_v<eos_t>) err.set(C2P_RESET_YE) ;
        err.set(C2P_SIG_RHO_TOO_LOW) ;
    }

    if constexpr (has_ye_v<eos_t>) {
        if (eos_err.test(EOS_YE_TOO_LOW))  { err.set(C2P_RESET_YE); err.set(C2P_SIG_YE_TOO_LOW)  ; }
        if (eos_err.test(EOS_YE_TOO_HIGH)) { err.set(C2P_RESET_YE); err.set(C2P_SIG_YE_TOO_HIGH) ; }
    }

    if (eos_err.test(EOS_ENTROPY_TOO_HIGH)) { err.set(C2P_RESET_ENTROPY); err.set(C2P_SIG_ENT_TOO_HIGH) ; }
    if (eos_err.test(EOS_ENTROPY_TOO_LOW))  { err.set(C2P_RESET_ENTROPY); err.set(C2P_SIG_ENT_TOO_LOW)  ; }

    if (eos_err.test(EOS_EPS_TOO_HIGH)) {
        err.set(C2P_SIG_EPS_TOO_HIGH) ;
        if (be_lenient) {
            err.set(C2P_RESET_STILDE) ;
            err.set(C2P_RESET_TAU)    ;
        } else {
            Kokkos::abort("In EOS: maximum eps exceeded") ;
        }
    }

    if (eos_err.test(EOS_EPS_TOO_LOW)) {
        err.set(C2P_RESET_STILDE) ;
        err.set(C2P_RESET_TAU)    ;
        err.set(C2P_SIG_EPS_TOO_LOW) ;
    }

    if ( eos_err.test(EOS_TEMPERATURE_TOO_HIGH)) {
        err.set(C2P_SIG_TEMP_TOO_HIGH) ;
        if (be_lenient) {
            err.set(C2P_RESET_STILDE) ;
            err.set(C2P_RESET_TAU)    ;
        } else {
            Kokkos::abort("In EOS: maximum temperature exceeded") ;
        }
    }
    if (eos_err.test(EOS_TEMPERATURE_TOO_LOW)) {
        err.set(C2P_RESET_STILDE) ;
        err.set(C2P_RESET_TAU)    ;
        err.set(C2P_SIG_TEMP_TOO_LOW) ;
    }

    /* Pressure-range signals are produced by the P-recon inverse hook       */
    /* (eps_h_csnd2_temp_entropy__press_rho_ye, used when                    */
    /* GRACE_RECON_THERMO=PRESS).  Standard c2p paths invert energy/entropy  */
    /* and don't fire these; the dispatch is defensive in case a future c2p  */
    /* variant uses the P-based hook.  Treat the same way as the EPS and    */
    /* TEMPERATURE pairs: low = quiet reset of momentum/energy, high =      */
    /* lenient reset or hard abort.                                         */
    if (eos_err.test(EOS_PRESS_TOO_HIGH)) {
        err.set(C2P_SIG_PRESS_TOO_HIGH) ;
        if (be_lenient) {
            err.set(C2P_RESET_STILDE) ;
            err.set(C2P_RESET_TAU)    ;
        } else {
            Kokkos::abort("In EOS: maximum pressure exceeded") ;
        }
    }
    if (eos_err.test(EOS_PRESS_TOO_LOW)) {
        err.set(C2P_RESET_STILDE) ;
        err.set(C2P_RESET_TAU)    ;
        err.set(C2P_SIG_PRESS_TOO_LOW) ;
    }
}

/**
 * @brief Convert conservative variables to primitive ones.
 *
 * @tparam eos_t Type of EOS.
 * @param cons Conservative variables (at one cell).
 * @param prims Primitive variables (at one cell).
 * @param metric Metric utilities.
 * @param eos Equation of State.
 * Atmosphere conditions are enforced by this routine. Conserved variables
 * are recomputed to be consistent with inverted primitives.
 *
 * @returns true if any "real" flooring/failure event occurred during the
 *          inversion (atmosphere reset, T-floor branch fired). Consumed by
 *          the FOFC pass downstream to flag cells whose high-order flux
 *          would have produced an unphysical state. Routine adjustments
 *          (limit_conservatives, primitive W/sigma clamps, the unconditional
 *          RESET_ENTROPY) do NOT count as flooring for this purpose.
 */
template< typename eos_t >
bool GRACE_HOST_DEVICE GRACE_DEVICE_EXTERNAL_LINKAGE
conservs_to_prims( grace::grmhd_cons_array_t&
                      , grace::grmhd_prims_array_t&
                      , grace::metric_array_t const&
                      , eos_t const& eos
                      , atmo_params_t const& atmo
                      , excision_params_t const& excision
                      , c2p_params_t const& c2p_pars
                      , double * rtp
                      , c2p_err_t& c2p_err 
                      , bool dry_run = false ) ;

void GRACE_HOST_DEVICE GRACE_DEVICE_EXTERNAL_LINKAGE
prims_to_conservs( grace::grmhd_prims_array_t& prims
                 , grace::grmhd_cons_array_t& cons
                 , grace::metric_array_t const& metric ) ;
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS) \
extern template \
bool GRACE_HOST_DEVICE \
conservs_to_prims<EOS>( grace::grmhd_cons_array_t&  \
                      , grace::grmhd_prims_array_t&  \
                      , grace::metric_array_t const&  \
                      , EOS const& eos \
                      , atmo_params_t const& atmo \
                      , excision_params_t const& excision \
                      , c2p_params_t const& c2p_pars \
                      , double * rtp \
                      , c2p_err_t& c2p_err \
                      , bool dry_run )
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::tabulated_cold_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;
#undef INSTANTIATE_TEMPLATE
}

#endif /* GRACE_PHYSICS_EOS_C2P_HH */