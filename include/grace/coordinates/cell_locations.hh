/**
 * @file cell_locations.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief FP-symmetric helpers for cell-center / cell-edge physical positions.
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
 * The naive form `x = xmin + l * (xmax - xmin)` accumulates round-off
 * proportional to the cell index, breaking bit-exact mirror symmetry
 * about the tree midpoint when the cells-per-block is not a power of 2.
 * The reorganized form below yields a per-cell round-off bounded at one
 * ulp regardless of the cell index, and is bit-exact for any tree whose
 * `(xmin, xmax)` are FP-exact (so that `0.5*(xmin+xmax)` is FP-exact too).
 * This dramatically reduces the FP-asymmetry pumping rate of the m=4
 * cubic mode at non-power-of-2 cells-per-block resolutions.
 */
#ifndef GRACE_COORDINATES_CELL_LOCATIONS_HH
#define GRACE_COORDINATES_CELL_LOCATIONS_HH

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

namespace grace { namespace coordinates {

/**
 * @brief FP-symmetric physical position from a fractional coordinate in [0,1].
 *
 * Mathematically equivalent to `xmin + l * (xmax - xmin)`, but evaluated in a
 * grouping that arranges the round-off errors to cancel under the substitution
 * `l -> 1 - l`. Cells at mirror-symmetric fractional positions about l=0.5
 * therefore land bit-exactly mirror-symmetric about (xmin+xmax)/2, even when
 * the multiplication `l * (xmax - xmin)` would not be FP-exact.
 */
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
constexpr double fp_symmetric_phys_from_logical(
    double l, double xmin, double xmax) noexcept
{
    return (l * xmax - l * xmin)
         - (0.5 * xmax - 0.5 * xmin)
         + (0.5 * xmin + 0.5 * xmax);
}

} } // namespace grace::coordinates

#endif // GRACE_COORDINATES_CELL_LOCATIONS_HH
