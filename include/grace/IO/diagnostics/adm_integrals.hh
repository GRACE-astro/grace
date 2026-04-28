/**
 * @file adm_integrals.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief ADM-mass surface integral on a 2-sphere, evaluated by Hermite-cubic
 *        interpolation of the spatial metric components and their analytic
 *        gradients.
 *
 *        Asymptotic flat-space form (e.g. Gourgoulhon 8.59):
 *
 *            M_ADM = (1/16 pi) oint_S sum_{i,j} (d_j gamma_{ij} - d_i gamma_{jj}) n^i dA
 *
 *        with dA = r^2 sin(theta) dtheta dphi for a sphere of radius r and
 *        n^i the flat-space outward unit normal, both with respect to the
 *        sphere's centre.
 *
 *        For Z4c the physical metric and its gradient are reconstructed from
 *        the conformal pair (gamma~, chi):
 *
 *            gamma_ij     = gamma~_ij / chi^2
 *            d_k gamma_ij = d_k gamma~_ij / chi^2 - 2 gamma~_ij d_k chi / chi^3
 *
 *        For the Cowling / non-conformal metric the physical components are
 *        evolved directly and no conversion is needed.
 *
 *        Each sphere named under adm_integrals.detector_names produces one
 *        scalar output stream M_ADM_<sphere>.dat.
 *
 * @date 2026-04-27
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference / Volume
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
 */

#ifndef GRACE_IO_ADM_INTEGRALS_HH
#define GRACE_IO_ADM_INTEGRALS_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/grid_interpolator.hh>
#include <grace/IO/spherical_surfaces.hh>

#include <vector>
#include <string>
#include <unordered_map>


namespace grace {

/**
 * @brief Surface ADM-mass diagnostic.
 *
 * Owns one Hermite-cubic interpolator per registered detector sphere. The
 * interpolator is rebuilt every call from the sphere's already-computed
 * (intersected_cells_h, intersecting_points_h, points_h) — that data is
 * refreshed by the spherical_surface_manager whenever the mesh or the
 * sphere moves.
 *
 * This class deliberately does not derive from diagnostic_base_t: the base
 * pipeline routes through interpolate_on_sphere, which only knows about the
 * detector's Lagrange interpolator and yields values, not gradients.
 */
struct adm_integrals {

    adm_integrals();

    /// Open per-sphere output files and write headers (rank 0 only).
    void initialize_files();

    /// Recompute and append one row per sphere to its output file.
    void compute_and_write();

    /// Recompute M_ADM at every registered sphere and return them in the
    /// same order as sphere_indices(). All ranks return the globally-reduced
    /// value. Exposed for programmatic / unit-test use; compute_and_write()
    /// is a thin wrapper around it.
    std::vector<double> compute();

    /// View of the (sorted, deduplicated) sphere indices this diagnostic
    /// is registered against.
    std::vector<size_t> const& sphere_indices_view() const { return sphere_indices; }

private:

    /// Rebuild the Hermite interpolator for sphere idx from its current
    /// detector data. Idempotent — safe to call every step.
    void refresh_interpolator(size_t sphere_idx);

    /// Local (rank-owned) integrand sum for one sphere.
    double compute_local(size_t sphere_idx);

    using hermite_interp_t = grid_interpolator_t<poly_kind::hermite, 3>;

    std::vector<size_t>                          sphere_indices;
    std::unordered_map<size_t, hermite_interp_t> interpolators;
    /// Indices of metric variables we need to interpolate (with grad).
    std::vector<int>                             var_interp_idx;
};

}  // namespace grace

#endif  // GRACE_IO_ADM_INTEGRALS_HH
