/**
 * @file lagrange_interpolation.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Backward-compat shim for the (now unified) grid interpolator.
 *
 *        The Lagrange-based grid-to-point interpolator, plus the shared
 *        search infrastructure (point_host_t, intersected_cell_descriptor_t,
 *        grace_search_points, intersected_cell_set_t, interp_weights_t),
 *        live in <grace/utils/grid_interpolator.hh> as the
 *        grid_interpolator_t<poly_kind, Degree> template plus its Lagrange
 *        specialization.
 *
 *        This header re-exports the old name as
 *
 *            template<int Order>
 *            using lagrange_interpolator_t =
 *                grid_interpolator_t<poly_kind::lagrange, Order>;
 *
 *        so the existing call sites (spherical_surface_iface, the AH
 *        finder, the centre-of-object tracker, etc.) keep working
 *        unchanged.
 *
 * @date 2024-04-09
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference / Volume
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

#ifndef GRACE_UTILS_LAGRANGE_INTERP_HH
#define GRACE_UTILS_LAGRANGE_INTERP_HH

#include <grace/utils/grid_interpolator.hh>

namespace grace {

template <int Order>
using lagrange_interpolator_t = grid_interpolator_t<poly_kind::lagrange, Order>;

}  // namespace grace

#endif  // GRACE_UTILS_LAGRANGE_INTERP_HH
