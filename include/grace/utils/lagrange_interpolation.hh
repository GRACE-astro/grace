/**
 * @file lagrange_interpolation.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Slope limiters for use in reconstruction and/or prolongation.
 * @version 0.1
 * @date 2024-04-09
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef GRACE_UTILS_LAGRANGE_INTERP_HH
#define GRACE_UTILS_LAGRANGE_INTERP_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/math.hh>
#include <Kokkos_Core.hpp> 

namespace grace {

template< size_t order >
struct lagrange_restrictor {
    
};

}

#endif /* GRACE_UTILS_LAGRANGE_INTERP_HH */