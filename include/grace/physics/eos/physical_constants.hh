/**
 * @file physical_constants.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Fundamental physical constants (c, G, k_B, ...) and astrophysical conversion factors used across GRACE physics modules.
 * @date 2024-05-29
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
#ifndef GRACE_PHYS_CONSTANTS_HH
#define GRACE_PHYS_CONSTANTS_HH
namespace grace { namespace physical_constants {

namespace detail {
// constexpr sqrt by Newton iteration (std::sqrt isn't constexpr before C++26).
// Used only inside this header to derive CU_to_Tesla.
constexpr double _sqrt_iter(double x, double y, int n) {
    return n == 0 ? y : _sqrt_iter(x, 0.5*(y + x/y), n-1);
}
constexpr double sqrt(double x) {
    return x <= 0.0 ? 0.0 : _sqrt_iter(x, x, 64);
}
} // namespace detail


#define CONSTDEF(x,y) static constexpr double x = y

// ---------------------------------------------------------------------------
// Fundamental SI constants.
// Where possible, use post-2019 SI redefinition exact values; otherwise
// CODATA 2018 (latest published as of 2022). All comments give the unit.
// ---------------------------------------------------------------------------
CONSTDEF(c_si,   299792458.0);              // m/s (exact, SI definition)
CONSTDEF(h_si,   6.62607015e-34);           // J s (exact, post-2019 SI)
CONSTDEF(e_si,   1.602176634e-19);          // C — elementary charge (exact, post-2019 SI)
CONSTDEF(Kb_si,  1.380649e-23);             // J/K (exact, post-2019 SI)
CONSTDEF(NA_si,  6.02214076e23);            // 1/mol (exact, post-2019 SI)

// Newtonian G — CODATA 2018. The Msun-based unit system below uses the IAU
// 2015 exact nominal GM_sun and derives Msun from this G, so G and Msun are
// always internally consistent (G * Msun_si == GMsun_IAU exactly).
CONSTDEF(G_si,   6.67430e-11);              // m^3 / (kg s^2) — CODATA 2018

// Permeability of free space. Post-2019 SI: measured, no longer exactly 4π·1e-7.
// CODATA 2018 value (relative uncertainty ~1.5e-10). Pre-2019 was 4π*1e-7 exactly.
CONSTDEF(mu0_si, 1.25663706212e-6);         // N/A^2 — CODATA 2018

// Length. Definition.
CONSTDEF(fm_si,  1.0e-15);                  // m (exact, definition)

// ---------------------------------------------------------------------------
// Astrophysical constants
// ---------------------------------------------------------------------------
// IAU 2015 nominal solar mass parameter (exact resolution).
CONSTDEF(GMsun_IAU, 1.3271244e20);          // m^3/s^2 — IAU 2015 nominal
// Solar mass derived from IAU GMsun and our G, by construction.
CONSTDEF(Msun_si, GMsun_IAU / G_si);        // kg (derived, exact in G·Msun)

// ---------------------------------------------------------------------------
// Particle masses (CODATA 2018, in MeV/c^2)
// ---------------------------------------------------------------------------
CONSTDEF(me_MeV, 0.51099895000);            // electron mass
CONSTDEF(mp_MeV, 938.27208816);             // proton mass
CONSTDEF(mn_MeV, 939.56542052);             // neutron mass
CONSTDEF(mu_MeV, 931.49410242);             // atomic mass unit — used by FUKA/LORENE

// Fine structure constant (CODATA 2018)
CONSTDEF(alpha_fine, 7.2973525693e-3);      // dimensionless

// Thomson scattering cross-section (CODATA 2018)
CONSTDEF(sigmaT_cgs, 6.6524587321e-25);     // cm^2

// ---------------------------------------------------------------------------
// Derived conversions
// ---------------------------------------------------------------------------
// MeV (energy) → kg of mass: M = E/c^2 = (e_si * 1e6 J) / c_si^2.
// Computed inline so it always tracks e_si and c_si.
CONSTDEF(MeV_to_kg, e_si * 1.0e6 / (c_si * c_si));  // kg per MeV/c^2 (exact)
CONSTDEF(MeV_to_g,  MeV_to_kg * 1.0e3);             // g per MeV/c^2

// Code-unit (c = G = Msun = 1) magnetic field → Tesla, in Heaviside-Lorentz SI
// where u_mag = B^2 / (2 μ_0). Equivalently:
//     B[T] = sqrt(μ_0) * c^4 / (G^(3/2) * Msun)
// Computed inline so it tracks G_si / Msun_si / μ_0_si — never silently drifts.
CONSTDEF(CU_to_Tesla,
    detail::sqrt(mu0_si) * (c_si*c_si*c_si*c_si)
      / (detail::sqrt(G_si*G_si*G_si) * Msun_si));

} } /* namespace grace::physical_constants */

#endif 