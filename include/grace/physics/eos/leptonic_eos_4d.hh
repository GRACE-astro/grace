/**
 * @file leptonic_eos_4d.hh
 * @author Marie Cassing (mcassing@itp.uni-frankfurt.de)
 * @author Keneth Miler (miler@itp.uni-frankfurt.de)
 * @brief  GRACE wrapper around the Margherita-style additive leptonic EOS.
 *
 *         The total thermodynamic state is built additively from three
 *         independent 3D tables:
 *
 *             baryon     (rho, T, yp = Y_le + Y_mu) -- existing GRACE tabulated_eos
 *             electronic (rho, T, Y_le)             -- from the leptonic HDF5
 *             muonic     (rho, T, Y_mu)             -- from the leptonic HDF5
 *
 *         where yp is the proton (charge) fraction (charge neutrality)
 *         clamped to the baryon-table Y_e axis bounds.
 *
 *         Margherita default semantics that this class follows:
 *           - the baryon contribution already contains the electron
 *             contribution, so the electronic table's pressure / eps /
 *             entropy are NOT added into the totals.  The electronic
 *             table is consulted only for mu_e.
 *           - yp is clamped to the baryon table's [yemin, yemax] before
 *             every baryon lookup (charge neutrality + Y_e axis range).
 *           - when ye + ymu >= yemax, mu_e is taken from the baryon
 *             table (the leptonic mu_e is unreliable in that corner)
 *             and mu_mu is reported as its rest mass (atmosphere value).
 *           - sound speed comes from the baryon table alone.
 *           - the Y_le axis is linear; the Y_mu axis is log-spaced
 *             (ymu_table in the HDF5 stores log(Ymu)).
 *
 *         The class follows the existing GRACE CRTP / eos_base_t pattern.
 *
 * @date   2026-05-15
 *
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale (GRACE).
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas.
 * Copyright (C) 2026 Marie Cassing, Keneth Miler
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef GRACE_PHYSICS_EOS_LEPTONIC_4D_HH
#define GRACE_PHYSICS_EOS_LEPTONIC_4D_HH

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/bitset.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/tabulated_eos.hh>   // tabeos_linterp_t, cold_eos_linterp_t

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <limits>

namespace grace {

class leptonic_eos_4d_t
    : public eos_base_t<leptonic_eos_4d_t>
{
    using err_t  = eos_err_t ;
    using base_t = eos_base_t<leptonic_eos_4d_t> ;

  public:

    // ----------------------------------------------------------
    //  Baryon table variable indices (same as tabulated_eos_t).
    // ----------------------------------------------------------
    enum TEOS_VIDX : int {
        TABPRESS = 0,
        TABEPS,
        TABCSND2,
        TABENTROPY,
        TABMUE,
        TABMUP,
        TABMUN,
        TABXA,
        TABXH,
        TABXN,
        TABXP,
        TABABAR,
        TABZBAR,
        N_TAB_VARS_BARYON
    } ;

    // ----------------------------------------------------------
    //  Electronic table (9 vars, Margherita EELE ordering).
    // ----------------------------------------------------------
    enum ELE_VIDX : int {
        TABMUELE = 0,
        TABYLE_MINUS,
        TABYLE_PLUS,
        TABPRESS_E_MINUS,
        TABPRESS_E_PLUS,
        TABEPS_E_MINUS,
        TABEPS_E_PLUS,
        TABS_E_MINUS,
        TABS_E_PLUS,
        N_TAB_VARS_ELE
    } ;

    // ----------------------------------------------------------
    //  Muonic table (9 vars, Margherita EMUON ordering).
    // ----------------------------------------------------------
    enum MUON_VIDX : int {
        TABMUMU = 0,
        TABYMU_MINUS,
        TABYMU_PLUS,
        TABPRESS_MU_MINUS,
        TABPRESS_MU_PLUS,
        TABEPS_MU_MINUS,
        TABEPS_MU_PLUS,
        TABS_MU_MINUS,
        TABS_MU_PLUS,
        N_TAB_VARS_MUON
    } ;

    // ----------------------------------------------------------
    //  Cold-slice file layout: 8 columns
    //   log(rho)  temp  ye_cold  ymu_cold  press  eps  cs2  entropy
    // ----------------------------------------------------------
    enum COLD_VIDX : int {
        CTABTEMP = 0,
        CTABYE,
        CTABYMU,
        CTABPRESS,
        CTABEPS,
        CTABCSND2,
        CTABENTROPY,
        N_CTAB_VARS
    } ;

    leptonic_eos_4d_t() = default ;

    /**
     * @brief Construct from pre-built interpolators.
     *
     * The baryon-table 4D view, axes and energy_shift come from the
     * GRACE read_eos_table() pipeline; the lepton views and Y_e/Y_mu
     * axes come from the leptonic HDF5.  rho and T axes are *always*
     * those of the baryon table (the reader asserts the leptonic file
     * lives on the same (rho,T) grid).
     */
    leptonic_eos_4d_t(
        // baryon table (shape and layout of tabulated_eos_t::tables)
        Kokkos::View<double ****, grace::default_space> tab_baryon,
        Kokkos::View<double *,    grace::default_space> bar_logrho,
        Kokkos::View<double *,    grace::default_space> bar_logT,
        Kokkos::View<double *,    grace::default_space> bar_ye,
        // electronic table  (axis = linear Y_le)
        Kokkos::View<double ****, grace::default_space> tab_ele,
        Kokkos::View<double *,    grace::default_space> ele_yle,
        // muonic table      (axis = log Y_mu, as stored in HDF5 ymu_table)
        Kokkos::View<double ****, grace::default_space> tab_muon,
        Kokkos::View<double *,    grace::default_space> mu_ymu,
        // cold slice  [nrho, N_CTAB_VARS]
        Kokkos::View<double **,   grace::default_space> cold_tab,
        Kokkos::View<double *,    grace::default_space> cold_logrho,
        // thermodynamic ranges
        double rhomax,   double rhomin,
        double tempmax,  double tempmin,
        double yemax,    double yemin,
        double ymumax,   double ymumin,
        double baryon_mass,
        double energy_shift_,
        double c2p_epsmin, double c2p_epsmax,
        double c2p_hmin,   double c2p_hmax,
        double c2p_temp_atm,
        double c2p_ye_atm,
        double c2p_ymu_atm,
        bool   atmo_is_beta_eq,
        // When false (default): the baryon table already includes the
        // electron contribution; the electronic table is used only for
        // mu_e.  Set true only if the baryon table was generated
        // without electrons and the electronic table must be added.
        bool   add_ele_contribution_ = false
    )
    : base_t( rhomax, rhomin,
              tempmax, tempmin,
              yemax, yemin,
              ymumax, ymumin,
              baryon_mass,
              c2p_epsmin, c2p_epsmax,
              c2p_hmin,   c2p_hmax,
              c2p_temp_atm,
              c2p_ye_atm,
              c2p_ymu_atm,
              atmo_is_beta_eq )
    , baryon_table(tab_baryon, bar_logrho, bar_logT, bar_ye)
    , ele_table   (tab_ele,    bar_logrho, bar_logT, ele_yle)
    , muon_table  (tab_muon,   bar_logrho, bar_logT, mu_ymu)
    , cold_table  (cold_tab,   cold_logrho)
    , nrho(bar_logrho.size()), nT(bar_logT.size())
    , nye(ele_yle.size()),     nymu(mu_ymu.size())
    , energy_shift(energy_shift_)
    , add_ele_contribution(add_ele_contribution_)
    {
        lrhomin  = bar_logrho[0] ; lrhomax  = bar_logrho[bar_logrho.size()-1] ;
        ltempmin = bar_logT[0]   ; ltempmax = bar_logT[bar_logT.size()-1]    ;
    }

    // ===========================================================
    //  CRTP _impl methods
    // ===========================================================

    double GRACE_HOST_DEVICE
    press__eps_rho_ye_ymu_impl(double& eps, double& rho,
                               double& ye, double& ymu, err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        return total_press(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_temp__eps_rho_ye_ymu_impl(double& temp, double& eps, double& rho,
                                    double& ye, double& ymu, err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        temp = Kokkos::exp(ltemp) ;
        return total_press(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press__temp_rho_ye_ymu_impl(double& temp, double& rho,
                                double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        return total_press(Kokkos::log(rho), Kokkos::log(temp), ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    eps__temp_rho_ye_ymu_impl(double& temp, double& rho,
                              double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        return total_eps(Kokkos::log(rho), Kokkos::log(temp), ye, ymu) ;
    }

    void GRACE_HOST_DEVICE
    eps_range__rho_ye_ymu_impl(double& epsmin, double& epsmax,
                               double& rho, double& ye, double& ymu,
                               err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double const lrho = Kokkos::log(rho) ;
        epsmin = total_eps(lrho, ltempmin, ye, ymu) ;
        epsmax = total_eps(lrho, ltempmax, ye, ymu) ;
    }

    void GRACE_HOST_DEVICE
    entropy_range__rho_ye_ymu_impl(double& smin, double& smax,
                                   double& rho, double& ye, double& ymu,
                                   err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double const lrho = Kokkos::log(rho) ;
        smin = total_entropy(lrho, ltempmin, ye, ymu) ;
        smax = total_entropy(lrho, ltempmax, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2__eps_rho_ye_ymu_impl( double& h, double& csnd2, double& eps,
                                        double& rho, double& ye, double& ymu,
                                        err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        double const press = total_press(lrho, ltemp, ye, ymu) ;
        csnd2 = baryon_csnd2(lrho, ltemp, ye, ymu) ;
        h     = 1. + eps + press/rho ;
        return press ;
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2__temp_rho_ye_ymu_impl( double& h, double& csnd2, double& temp,
                                         double& rho, double& ye, double& ymu,
                                         err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        double const press = total_press(lrho, ltemp, ye, ymu) ;
        double const eps   = total_eps  (lrho, ltemp, ye, ymu) ;
        csnd2 = baryon_csnd2(lrho, ltemp, ye, ymu) ;
        h     = 1. + eps + press/rho ;
        return press ;
    }

    double GRACE_HOST_DEVICE
    press_eps_csnd2__temp_rho_ye_ymu_impl(double& eps, double& csnd2,
                                          double& temp, double& rho,
                                          double& ye, double& ymu,
                                          err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        eps   = total_eps  (lrho, ltemp, ye, ymu) ;
        csnd2 = baryon_csnd2(lrho, ltemp, ye, ymu) ;
        return total_press(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_entropy__eps_rho_ye_ymu_impl(
        double& h, double& csnd2, double& temp, double& entropy,
        double& eps, double& rho, double& ye, double& ymu, err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__eps_lrho_ye_ymu(eps, lrho, ye, ymu, err) ;
        temp = Kokkos::exp(ltemp) ;
        double const press = total_press   (lrho, ltemp, ye, ymu) ;
        csnd2   = baryon_csnd2(lrho, ltemp, ye, ymu) ;
        entropy = total_entropy(lrho, ltemp, ye, ymu) ;
        h = 1. + eps + press / rho ;
        return press ;
    }

    double GRACE_HOST_DEVICE
    eps_csnd2_entropy__temp_rho_ye_ymu_impl(
        double& csnd2, double& entropy, double& temp,
        double& rho, double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        csnd2   = baryon_csnd2 (lrho, ltemp, ye, ymu) ;
        entropy = total_entropy(lrho, ltemp, ye, ymu) ;
        return total_eps(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_eps_csnd2_entropy__temp_rho_ye_ymu_impl(
        double& eps, double& csnd2, double& entropy,
        double& temp, double& rho, double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        eps     = total_eps    (lrho, ltemp, ye, ymu) ;
        csnd2   = baryon_csnd2 (lrho, ltemp, ye, ymu) ;
        entropy = total_entropy(lrho, ltemp, ye, ymu) ;
        return total_press(lrho, ltemp, ye, ymu) ;
    }

    double GRACE_HOST_DEVICE
    press_h_csnd2_temp_eps__entropy_rho_ye_ymu_impl(
        double& h, double& csnd2, double& temp, double& eps,
        double& entropy, double& rho, double& ye, double& ymu,
        err_t& err) const
    {
        limit_rho(rho, err) ;
        limit_ye (ye,  err) ;
        limit_ymu(ymu, err) ;
        limit_entropy_rho_ye_ymu(entropy, rho, ye, ymu, err) ;
        double const lrho  = Kokkos::log(rho) ;
        double const ltemp = ltemp__entropy_lrho_ye_ymu(entropy, lrho, ye, ymu) ;
        temp = Kokkos::exp(ltemp) ;
        double const press = total_press   (lrho, ltemp, ye, ymu) ;
        eps   = total_eps    (lrho, ltemp, ye, ymu) ;
        csnd2 = baryon_csnd2 (lrho, ltemp, ye, ymu) ;
        h     = 1. + eps + press / rho ;
        return press ;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    eps_h_csnd2_temp_entropy__press_rho_ye_ymu_impl(
        double&, double&, double&, double&, double&, double&,
        double&, double&, err_t&) const
    {
        Kokkos::abort("eps_h_csnd2_temp_entropy__press_rho_ye_ymu_impl"
                      " is not implemented for leptonic_eos_4d_t.") ;
        return 0. ;
    }

    /**
     * @brief Chemical potentials and composition.
     *
     * mu_p, mu_n, X_*, Abar, Zbar are taken from the baryon table at
     * yp = clamp(ye + ymu, yemin, yemax) (charge neutrality);
     * mu_e is taken from the electronic table at Y_le = ye, unless
     * ye + ymu has saturated at yemax -- in that case we fall back
     * to the baryon table's mu_e, since the leptonic mu_e becomes
     * unreliable there (Margherita's convention).
     */
    double GRACE_HOST_DEVICE
    mue_mumu_mup_mun_Xa_Xh_Xn_Xp_Abar_Zbar__temp_rho_ye_ymu_impl(
        double &mumu, double& mup, double& mun, double& Xa, double& Xh,
        double& Xn,  double& Xp,  double& Abar, double& Zbar,
        double& temp, double& rho, double& ye, double& ymu,
        err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        double const lrho  = Kokkos::log(rho)  ;
        double const ltemp = Kokkos::log(temp) ;
        double const yp    = clamp_yp(ye + ymu) ;

        mup  = baryon_table.interp(lrho,ltemp,yp,TABMUP)  ;
        mun  = baryon_table.interp(lrho,ltemp,yp,TABMUN)  ;
        Xa   = baryon_table.interp(lrho,ltemp,yp,TABXA)   ;
        Xh   = baryon_table.interp(lrho,ltemp,yp,TABXH)   ;
        Xn   = baryon_table.interp(lrho,ltemp,yp,TABXN)   ;
        Xp   = baryon_table.interp(lrho,ltemp,yp,TABXP)   ;
        Abar = baryon_table.interp(lrho,ltemp,yp,TABABAR) ;
        Zbar = baryon_table.interp(lrho,ltemp,yp,TABZBAR) ;

        if (ye + ymu >= this->eos_yemax) {
            // in this case, we use baryonic table mu_e and force mu_mu = 0
           	mumu = 105.6583755; // mu_mu at atmosphere value is not 0 mev
            return baryon_table.interp(lrho,ltemp,yp,TABMUE)  ;
        }
        mumu = muon_table.interp(lrho, ltemp, ymu, TABMUE) ;
        return ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABMUELE) ;
    }

    /**
     * @brief Muon chemical potential.  Not part of the standard CRTP
     *        API but needed by the M1 muon-leakage and beta-equilibrium
     *        solvers.
     */
    double GRACE_HOST_DEVICE
    mumu__temp_rho_ye_ymu(double& temp, double& rho,
                          double& ye, double& ymu, err_t& err) const
    {
        limit_rho (rho,  err) ;
        limit_ye  (ye,   err) ;
        limit_ymu (ymu,  err) ;
        limit_temp(temp, err) ;
        if (ye + ymu >= this->eos_yemax) {
            return 105.6583755 ;        // muon rest mass [MeV] -- atmosphere
        }
        return muon_table.interp(Kokkos::log(rho),
                                 Kokkos::log(temp),
                                 Kokkos::log(ymu),
                                 MUON_VIDX::TABMUMU) ;
    }

    // ----- Cold-slice accessors -----
    double GRACE_HOST_DEVICE
    press_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return Kokkos::exp(cold_table.interp(Kokkos::log(rho), CTABPRESS)) ;
    }
    double GRACE_HOST_DEVICE
    eps_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return Kokkos::exp(cold_table.interp(Kokkos::log(rho), CTABEPS))
               - energy_shift ;
    }
    double GRACE_HOST_DEVICE
    ye_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABYE) ;
    }
    double GRACE_HOST_DEVICE
    ymu_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABYMU) ;
    }
    double GRACE_HOST_DEVICE
    temp_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return Kokkos::exp(cold_table.interp(Kokkos::log(rho), CTABTEMP)) ;
    }
    double GRACE_HOST_DEVICE
    entropy_cold__rho_impl(double& rho, err_t& err) const {
        limit_rho(rho, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABENTROPY) ;
    }
    double GRACE_HOST_DEVICE
    rho__press_cold_impl(double& press_cold, err_t& err) const
    {
        double const lp = cold_lpress__press_limited(press_cold, err) ;
        auto rootfun = [this, lp] (double lrho) {
            return cold_table.interp(lrho, CTABPRESS) - lp ;
        } ;
        double const lrmin = cold_table._logrho(0) ;
        double const lrmax = cold_table._logrho(cold_table._logrho.size()-1) ;
        return Kokkos::exp(utils::brent(rootfun, lrmin, lrmax, 1e-14)) ;
    }
    double GRACE_HOST_DEVICE
    rho__energy_cold_impl(double& e_cold, err_t& err) const
    {
        int const n = cold_table._logrho.size() ;
        double const eps_min = Kokkos::exp(cold_table._tables(0,   CTABEPS)) - energy_shift ;
        double const eps_max = Kokkos::exp(cold_table._tables(n-1, CTABEPS)) - energy_shift ;
        double const e_min   = (1.+eps_min) * Kokkos::exp(cold_table._logrho(0))   ;
        double const e_max   = (1.+eps_max) * Kokkos::exp(cold_table._logrho(n-1)) ;
        if (e_cold < e_min) { e_cold = e_min ; err.set(EOS_EPS_TOO_LOW)  ; return Kokkos::exp(cold_table._logrho(0))   ; }
        if (e_cold > e_max) { e_cold = e_max ; err.set(EOS_EPS_TOO_HIGH) ; return Kokkos::exp(cold_table._logrho(n-1)) ; }
        auto rootfun = [this, e_cold] (double lrho) {
            double eps = Kokkos::exp(cold_table.interp(lrho, CTABEPS)) - energy_shift ;
            return (1.+eps) * Kokkos::exp(lrho) - e_cold ;
        } ;
        return Kokkos::exp(utils::brent(rootfun,
            cold_table._logrho(0), cold_table._logrho(n-1), 1e-14)) ;
    }
    double GRACE_HOST_DEVICE
    energy_cold__press_cold_impl(double& press_cold, err_t& err) const
    {
        double const lp = cold_lpress__press_limited(press_cold, err) ;
        auto rootfun = [this, lp] (double lrho) {
            return cold_table.interp(lrho, CTABPRESS) - lp ;
        } ;
        double const lrmin = cold_table._logrho(0) ;
        double const lrmax = cold_table._logrho(cold_table._logrho.size()-1) ;
        double const lrho  = utils::brent(rootfun, lrmin, lrmax, 1e-14) ;
        double const eps   = Kokkos::exp(cold_table.interp(lrho, CTABEPS)) - energy_shift ;
        return Kokkos::exp(lrho) * (1.+eps) ;
    }
    double GRACE_HOST_DEVICE
    ye_cold__press_impl(double& press, err_t& err) const {
        double rho = rho__press_cold_impl(press, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABYE) ;
    }
    double GRACE_HOST_DEVICE
    ymu_cold__press_impl(double& press, err_t& err) const {
        double rho = rho__press_cold_impl(press, err) ;
        return cold_table.interp(Kokkos::log(rho), CTABYMU) ;
    }

    // ===========================================================
    //  Public data members (captured into kernels by value)
    // ===========================================================
    tabeos_linterp_t   baryon_table ;  ///< rho, T, yp = ye + ymu  (clamped)
    tabeos_linterp_t   ele_table    ;  ///< rho, T, Y_le (linear axis)
    tabeos_linterp_t   muon_table   ;  ///< rho, T, log(Y_mu) axis — queries must pass log(ymu)
    cold_eos_linterp_t cold_table   ;

    int nrho, nT, nye, nymu ;
    double energy_shift ;
    double lrhomin, lrhomax ;
    double ltempmin, ltempmax ;
    // If true, add electronic pressure/eps/entropy on top of the
    // baryon table.  Keep false when the baryon table already
    // contains the electron contribution (the usual case for SFHo,
    // DD2, etc.).  Mirrors Margherita's add_ele_contribution flag.
    bool add_ele_contribution = false ;

  private:

    // ----------------------------------------------------------
    //  Effective proton fraction yp = clamp(ye + ymu, yemin, yemax).
    //  Charge neutrality: the baryon table is sampled at the total
    //  charge fraction, which is the sum of the lepton fractions.
    // ----------------------------------------------------------
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double clamp_yp(double yp_raw) const {
        if (yp_raw < this->eos_yemin) return this->eos_yemin ;
        if (yp_raw > this->eos_yemax) return this->eos_yemax ;
        return yp_raw ;
    }

    // ----------------------------------------------------------
    //  Total quantities: baryon + muon, optionally + electron.
    //  add_ele_contribution = false (default): the baryon table
    //  already includes electrons; the electronic table is used
    //  only for mu_e.  Set true only if the baryon table was built
    //  without electrons.  Mirrors Margherita's add_ele_contribution.
    // ----------------------------------------------------------
    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    total_press(double lrho, double ltemp, double ye, double ymu) const
    {
        double const yp   = clamp_yp(ye + ymu) ;
        double const lymu = Kokkos::log(ymu) ;
        double const pb   = Kokkos::exp(baryon_table.interp(lrho, ltemp, yp, TABPRESS)) ;
        double const pmm  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABPRESS_MU_MINUS) ;
        double const pmp  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABPRESS_MU_PLUS)  ;
        double const pe   = add_ele_contribution
            ? ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABPRESS_E_MINUS)
            + ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABPRESS_E_PLUS)
            : 0.0 ;
        return pb + pmm + pmp + pe ;
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    total_eps(double lrho, double ltemp, double ye, double ymu) const
    {
        double const yp   = clamp_yp(ye + ymu) ;
        double const lymu = Kokkos::log(ymu) ;
        double const eb   = Kokkos::exp(baryon_table.interp(lrho, ltemp, yp, TABEPS)) - energy_shift ;
        double const emm  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABEPS_MU_MINUS) ;
        double const emp  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABEPS_MU_PLUS)  ;
        double const ee   = add_ele_contribution
            ? ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABEPS_E_MINUS)
            + ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABEPS_E_PLUS)
            : 0.0 ;
        return eb + emm + emp + ee ;
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    total_entropy(double lrho, double ltemp, double ye, double ymu) const
    {
        double const yp   = clamp_yp(ye + ymu) ;
        double const lymu = Kokkos::log(ymu) ;
        double const sb   = baryon_table.interp(lrho, ltemp, yp, TABENTROPY) ;
        double const smm  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABS_MU_MINUS) ;
        double const smp  = muon_table.interp(lrho, ltemp, lymu, MUON_VIDX::TABS_MU_PLUS)  ;
        double const se   = add_ele_contribution
            ? ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABS_E_MINUS)
            + ele_table.interp(lrho, ltemp, ye, ELE_VIDX::TABS_E_PLUS)
            : 0.0 ;
        return sb + smm + smp + se ;
    }

    GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE double
    baryon_csnd2(double lrho, double ltemp, double ye, double ymu) const
    {
        return baryon_table.interp(lrho, ltemp,
                                   clamp_yp(ye + ymu),
                                   TABCSND2) ;
    }

    // ----------------------------------------------------------
    //  Bounds clamping
    // ----------------------------------------------------------
    KOKKOS_INLINE_FUNCTION void limit_rho(double& rho, err_t& err) const {
        if ( rho < this->eos_rhomin ) { rho = (1.+1e-5)*this->eos_rhomin ; err.set(EOS_RHO_TOO_LOW)  ; }
        if ( rho > this->eos_rhomax ) { rho = (1.-1e-5)*this->eos_rhomax ; err.set(EOS_RHO_TOO_HIGH) ; }
    }
    KOKKOS_INLINE_FUNCTION void limit_ye(double& ye, err_t& err) const {
        if ( ye < this->eos_yemin ) { ye = this->eos_yemin ; err.set(EOS_YE_TOO_LOW)  ; }
        if ( ye > this->eos_yemax ) { ye = this->eos_yemax ; err.set(EOS_YE_TOO_HIGH) ; }
    }
    KOKKOS_INLINE_FUNCTION void limit_ymu(double& ymu, err_t& err) const {
        if ( ymu < this->eos_ymumin ) {
            ymu = this->eos_ymumin ;
            #ifdef M1_NU_FIVESPECIES
            err.set(EOS_YMU_TOO_LOW) ;
            #else
            err.set(EOS_YE_TOO_LOW) ;
            #endif
        }
        if ( ymu > this->eos_ymumax ) {
            ymu = this->eos_ymumax ;
            #ifdef M1_NU_FIVESPECIES
            err.set(EOS_YMU_TOO_HIGH) ;
            #else
            err.set(EOS_YE_TOO_HIGH) ;
            #endif
        }
    }
    KOKKOS_INLINE_FUNCTION void limit_temp(double& temp, err_t& err) const {
        double const tmin = Kokkos::exp(ltempmin) ;
        double const tmax = Kokkos::exp(ltempmax) ;
        if ( temp < tmin ) { temp = (1.+1e-2)*tmin ; err.set(EOS_TEMPERATURE_TOO_LOW)  ; }
        if ( temp > tmax ) { temp = (1.-1e-2)*tmax ; err.set(EOS_TEMPERATURE_TOO_HIGH) ; }
    }
    KOKKOS_INLINE_FUNCTION void limit_entropy_rho_ye_ymu(double& entropy,
        double& rho, double& ye, double& ymu, err_t& err) const
    {
        double smin, smax ;
        entropy_range__rho_ye_ymu_impl(smin, smax, rho, ye, ymu, err) ;
        if ( entropy < smin ) { entropy = smin ; err.set(EOS_ENTROPY_TOO_LOW)  ; }
        if ( entropy > smax ) { entropy = smax ; err.set(EOS_ENTROPY_TOO_HIGH) ; }
    }

    // ----------------------------------------------------------
    //  Brent on the additive totals.  eps and entropy are monotone
    //  in T over the table range, so a single Brent in lT works.
    // ----------------------------------------------------------
    KOKKOS_INLINE_FUNCTION double
    ltemp__eps_lrho_ye_ymu(double& eps, double lrho,
                           double ye, double ymu, err_t& err) const
    {
        double const eps_lo = total_eps(lrho, ltempmin, ye, ymu) ;
        double const eps_hi = total_eps(lrho, ltempmax, ye, ymu) ;
        if (eps <= eps_lo) { eps = eps_lo ; err.set(EOS_EPS_TOO_LOW)  ; return ltempmin ; }
        if (eps >= eps_hi) { eps = eps_hi ; err.set(EOS_EPS_TOO_HIGH) ; return ltempmax ; }
        auto rootfun = [this, lrho, ye, ymu, eps] (double lt) {
            return total_eps(lrho, lt, ye, ymu) - eps ;
        } ;
        return utils::brent(rootfun, ltempmin, ltempmax, 1e-12) ;
    }

    KOKKOS_INLINE_FUNCTION double
    ltemp__entropy_lrho_ye_ymu(double entropy, double lrho,
                               double ye, double ymu) const
    {
        double const s_lo = total_entropy(lrho, ltempmin, ye, ymu) ;
        double const s_hi = total_entropy(lrho, ltempmax, ye, ymu) ;
        if (entropy <= s_lo) return ltempmin ;
        if (entropy >= s_hi) return ltempmax ;
        auto rootfun = [this, lrho, ye, ymu, entropy] (double lt) {
            return total_entropy(lrho, lt, ye, ymu) - entropy ;
        } ;
        return utils::brent(rootfun, ltempmin, ltempmax, 1e-12) ;
    }

    KOKKOS_INLINE_FUNCTION double
    cold_lpress__press_limited(double& press_cold, err_t& err) const
    {
        int const n = cold_table._logrho.size() ;
        double const p_min = Kokkos::exp(cold_table._tables(0,   CTABPRESS)) ;
        double const p_max = Kokkos::exp(cold_table._tables(n-1, CTABPRESS)) ;
        if (press_cold < p_min) { press_cold = p_min * (1.+1e-10) ; err.set(EOS_RHO_TOO_LOW)  ; }
        if (press_cold > p_max) { press_cold = p_max * (1.-1e-10) ; err.set(EOS_RHO_TOO_HIGH) ; }
        return Kokkos::log(press_cold) ;
    }

} ; // class leptonic_eos_4d_t

} /* namespace grace */

#endif /* GRACE_PHYSICS_EOS_LEPTONIC_4D_HH */
