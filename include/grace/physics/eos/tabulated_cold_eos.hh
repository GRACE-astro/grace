/**
 * @file tabulated_cold_eos.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Tabulated barotropic cold-EOS backbone for hybrid_eos_t.
 * @date 2026-05-04
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
 */

#ifndef GRACE_PHYSICS_EOS_TABULATED_COLD_EOS_HH
#define GRACE_PHYSICS_EOS_TABULATED_COLD_EOS_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/utils/rootfinding.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/physics/eos/tabulated_eos.hh>

#include <Kokkos_Core.hpp>

namespace grace {

/**
 * @brief Tabulated barotropic cold-EOS backbone for use with hybrid_eos_t.
 *
 * Reuses the GRACE cold-table file format (logrho, [logtemp], [ye],
 * logpress, logeps, cs2, [entropy]) — but only logrho, logpress, logeps,
 * and cs2 are read.  The temperature and Y_e columns can be set to bogus
 * values (typically 0) by the generator since they are unused here.
 *
 * Storage convention matches tabulated_eos_t: log(eps + energy_shift) is
 * what's tabulated, and energy_shift is subtracted on every read of the
 * eps column.  Pressure is stored as log(P).  cs² is stored linearly.
 *
 * Implements the minimal cold-EOS interface required by hybrid_eos_t:
 *   - press_cold_eps_cold__rho(eps, rho)
 *   - dpress_cold_drho__rho(rho)             (= cs²·h_cold for a barotrope)
 *   - rho__press_cold(press)
 *   - rho__energy_cold(e)
 *   - energy_cold__press_cold(press)
 */
class tabulated_cold_eos_t
{
public:

    tabulated_cold_eos_t() = default ;

    tabulated_cold_eos_t(
        Kokkos::View<double **, grace::default_space> _cold_tabeos,
        Kokkos::View<double  *, grace::default_space> _cold_tabeos_logrho,
        double _rhomin, double _rhomax,
        double _baryon_mass, double _energy_shift)
      : cold_table(_cold_tabeos, _cold_tabeos_logrho)
      , eos_rhomin(_rhomin), eos_rhomax(_rhomax)
      , baryon_mass(_baryon_mass), energy_shift(_energy_shift)
    {
        // Cache table-edge eps so floor / ceiling lookups don't pay
        // for an interpolation every time.
        int const n = cold_table._logrho.size() ;
        cold_eps_min = Kokkos::exp(cold_table._tables(0,    tabulated_eos_t::CTABEPS)) - energy_shift ;
        cold_eps_max = Kokkos::exp(cold_table._tables(n-1,  tabulated_eos_t::CTABEPS)) - energy_shift ;
        cold_press_min = Kokkos::exp(cold_table._tables(0,   tabulated_eos_t::CTABPRESS)) ;
        cold_press_max = Kokkos::exp(cold_table._tables(n-1, tabulated_eos_t::CTABPRESS)) ;
        // h_minimum is consumed by hybrid_eos_t / Kastaun c2p as the bracket
        // upper bound on μ.  Computed at the cold-table low-density edge.
        h_minimum = 1.0 + cold_eps_min + cold_press_min / Kokkos::exp(cold_table._logrho(0)) ;
    }

    /************************************************/
    /** Cold pressure and specific internal energy at given rho. */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    press_cold_eps_cold__rho(double& eps_cold, double& rho) const
    {
        rho = Kokkos::fmax(eos_rhomin, Kokkos::fmin(eos_rhomax, rho)) ;
        double const lrho = Kokkos::log(rho) ;
        std::array<int,2>    idx{tabulated_eos_t::CTABPRESS, tabulated_eos_t::CTABEPS} ;
        std::array<double,2> res{0.0, 0.0} ;
        cold_table.template interp<2>(lrho, idx, res) ;
        double const press_cold = Kokkos::exp(res[0]) ;
        eps_cold = Kokkos::exp(res[1]) - energy_shift ;
        return press_cold ;
    }

    /** dP_cold/drho.  For a barotrope, dP/dρ = cs²·h_cold. */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    dpress_cold_drho__rho(double& rho) const
    {
        rho = Kokkos::fmax(eos_rhomin, Kokkos::fmin(eos_rhomax, rho)) ;
        double const lrho = Kokkos::log(rho) ;
        std::array<int,3> idx{tabulated_eos_t::CTABPRESS,
                              tabulated_eos_t::CTABEPS,
                              tabulated_eos_t::CTABCSND2} ;
        std::array<double,3> res{0.0, 0.0, 0.0} ;
        cold_table.template interp<3>(lrho, idx, res) ;
        double const press_cold = Kokkos::exp(res[0]) ;
        double const eps_cold   = Kokkos::exp(res[1]) - energy_shift ;
        double const cs2_cold   = res[2] ;
        double const h_cold     = 1.0 + eps_cold + press_cold / rho ;
        return cs2_cold * h_cold ;
    }

    /** Inverse: rho such that P_cold(rho) = press.  Brent in log(rho). */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    rho__press_cold(double& press) const
    {
        if (press <= cold_press_min) { press = cold_press_min ; return Kokkos::exp(cold_table._logrho(0)) ; }
        if (press >= cold_press_max) { press = cold_press_max ; return Kokkos::exp(cold_table._logrho(cold_table._logrho.size()-1)) ; }
        double const lp = Kokkos::log(press) ;
        auto rootfun = [this, lp] (double lrho) {
            return cold_table.interp(lrho, tabulated_eos_t::CTABPRESS) - lp ;
        } ;
        double const lrmin = cold_table._logrho(0) ;
        double const lrmax = cold_table._logrho(cold_table._logrho.size()-1) ;
        return Kokkos::exp(utils::brent(rootfun, lrmin, lrmax, 1e-14)) ;
    }

    /** Inverse: rho such that e_cold(rho) = e, where e = rho(1+eps). */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    rho__energy_cold(double& e) const
    {
        int const n = cold_table._logrho.size() ;
        double const e_min = (1.0 + cold_eps_min) * Kokkos::exp(cold_table._logrho(0))   ;
        double const e_max = (1.0 + cold_eps_max) * Kokkos::exp(cold_table._logrho(n-1)) ;
        if (e <= e_min) { e = e_min ; return Kokkos::exp(cold_table._logrho(0))   ; }
        if (e >= e_max) { e = e_max ; return Kokkos::exp(cold_table._logrho(n-1)) ; }
        auto rootfun = [this, e] (double lrho) {
            double const eps = Kokkos::exp(cold_table.interp(lrho, tabulated_eos_t::CTABEPS)) - energy_shift ;
            return (1.0 + eps) * Kokkos::exp(lrho) - e ;
        } ;
        return Kokkos::exp(utils::brent(rootfun,
            cold_table._logrho(0), cold_table._logrho(n-1), 1e-14)) ;
    }

    /** Total cold energy density e = rho(1+eps) given press.  May clamp
     *  press to the table range, mirroring piecewise_polytropic_eos_t. */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    energy_cold__press_cold(double& press) const
    {
        if (press <= 0.0) return 0.0 ;
        double const rho = rho__press_cold(press) ;
        double const lrho = Kokkos::log(rho) ;
        double const eps = Kokkos::exp(cold_table.interp(lrho, tabulated_eos_t::CTABEPS)) - energy_shift ;
        return rho * (1.0 + eps) ;
    }

private:
    cold_eos_linterp_t cold_table ;
    double             cold_eps_min, cold_eps_max ;
    double             cold_press_min, cold_press_max ;

public:
    double eos_rhomin, eos_rhomax ;
    double baryon_mass, energy_shift ;
    double h_minimum ;
} ;

} /* namespace grace */

#endif /* GRACE_PHYSICS_EOS_TABULATED_COLD_EOS_HH */
