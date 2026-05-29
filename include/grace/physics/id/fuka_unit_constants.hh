/**
 * @file fuka_unit_constants.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief FUKA-geo to GRACE-geo rescale factors applied at the ID import boundary.
 *
 * @copyright This file is part of GRACE.
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
 * FUKA hardcodes legacy Whisky/Cactus geometric-unit constants in its
 * standalone Margherita EOS library
 * (`<HOME_KADATH>/include/EOS/standalone/margherita.hh`, e.g.
 *  `LENGTHGF = 6.77269222552442e-06`). GRACE uses CODATA-2018 G + IAU-2015
 * nominal GM_sun, which yields `LENGTHGF = 6.77220e-06`. The 73 ppm length
 * mismatch propagates to 213 ppm in density and pressure conversion
 * factors, silently biasing FUKA-ingested initial data on the GRACE side
 * whenever a tabulated EOS converted with GRACE constants is used
 * downstream.
 *
 * This header keeps GRACE's own conversions intact and instead provides
 * a `fuka_to_grace_rescale_t` struct of multiplicative factors to apply
 * at the importer boundary, one per tensor type. The FUKA-side values
 * are pulled in directly from `EOS/standalone/margherita.hh` so they
 * always track upstream — no hand-copied floats to keep in sync.
 *
 * Convention: every factor is the multiplier `x_GRACE = x_FUKA * <factor>`.
 * Dimensionless quantities (alpha, beta^i, gamma_ij, eps, W, v^i) have
 * factor 1 and are not represented here.
 */
#ifndef GRACE_PHYSICS_ID_FUKA_UNIT_CONSTANTS_HH
#define GRACE_PHYSICS_ID_FUKA_UNIT_CONSTANTS_HH

#include <grace_config.h>

#if !defined(GRACE_ENABLE_FUKA)
#error "fuka_unit_constants.hh requires GRACE_ENABLE_FUKA"
#endif

#include <grace/physics/eos/physical_constants.hh>

// FUKA's hardcoded Margherita geometric-unit constants. Pulled in
// directly so we automatically track any future upstream change rather
// than maintaining a shadow copy that can silently drift.
#include <EOS/standalone/margherita.hh>

namespace grace { namespace fuka_units {

// GRACE's canonical conversion factors, derived inline from
// grace::physical_constants (CODATA 2018 G + IAU 2015 GM_sun).
// Everything in cgs, matching Margherita's convention.
struct grace_constants_t {
 private:
    static constexpr double c_cgs  = grace::physical_constants::c_si    * 1.0e2;
    static constexpr double G_cgs  = grace::physical_constants::G_si    * 1.0e3;
    static constexpr double Msun_g = grace::physical_constants::Msun_si * 1.0e3;
    static constexpr double GM_cgs = G_cgs * Msun_g;
 public:
    static constexpr double LENGTHGF = c_cgs * c_cgs / GM_cgs;
    static constexpr double RHOGF    = (G_cgs * G_cgs * G_cgs * Msun_g * Msun_g)
                                     / (c_cgs * c_cgs * c_cgs * c_cgs * c_cgs * c_cgs);
    static constexpr double PRESSGF  = RHOGF / (c_cgs * c_cgs);
    static constexpr double EPSGF    = 1.0 / (c_cgs * c_cgs);
};

// Multiplicative factors: x_GRACE = x_FUKA * <factor>.
// Use one entry per tensor type at the importer boundary.
struct fuka_to_grace_rescale_t {
    // [L]       x, y, z, r, dx                              -- ~ 1 - 73 ppm
    static constexpr double length        = grace_constants_t::LENGTHGF
                                          / Margherita_constants::LENGTHGF;
    // [1/L]     K_ij, K, A_ij_tilde, Gamma^i_tilde          -- ~ 1 + 73 ppm
    static constexpr double inv_length    = 1.0 / length;
    // [M/L^3]   rho, e = rho*(1+eps), any energy density    -- ~ 1 + 213 ppm
    static constexpr double density       = grace_constants_t::RHOGF
                                          / Margherita_constants::RHOGF;
    // [M/L/t^2] = [M/L^3] in c=1 : pressure                 -- ~ 1 + 213 ppm
    static constexpr double pressure      = grace_constants_t::PRESSGF
                                          / Margherita_constants::PRESSGF;
    // [sqrt(M/L^3)] B (Heaviside-Lorentz, u_mag = B^2/2)    -- ~ 1 + 107 ppm
    // Not currently exercised: FUKA importer does not consume a B field.
    static constexpr double bfield        = grace::physical_constants::detail::sqrt(density);
    // [B * L]   A_i (vector potential)                      -- ~ 1 + 34 ppm
    static constexpr double vector_potential = bfield * length;
};

} } // namespace grace::fuka_units

#endif // GRACE_PHYSICS_ID_FUKA_UNIT_CONSTANTS_HH
