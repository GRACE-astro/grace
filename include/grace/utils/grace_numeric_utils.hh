/**
 * @file grace_numeric_utils.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-10-08
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

#ifndef GRACE_NUMERIC_UTILS_HH
#define GRACE_NUMERIC_UTILS_HH

#include <grace/utils/numerics/advanced_riemann_solvers.hh>
#include <grace/utils/numerics/affine_transformation.hh>
#include <grace/utils/numerics/constexpr.hh>
//#include <grace/utils/numerics/fd_utils.hh>
#include <grace/utils/numerics/gridloop.hh>
#include <grace/utils/numerics/integration.hh>
#include <grace/utils/numerics/interpolators.hh>
#include <grace/utils/numerics/limiters.hh>
#include <grace/utils/numerics/math.hh>
#include <grace/utils/numerics/matrix_helpers.tpp>
#include <grace/utils/numerics/metric_utils.hh>
#include <grace/utils/numerics/prolongation.hh>
#include <grace/utils/numerics/reconstruction.hh>
#include <grace/utils/numerics/reductors.hh>
#include <grace/utils/numerics/restriction.hh>
#include <grace/utils/numerics/riemann_solvers.hh>
#include <grace/utils/numerics/rootfinding.hh>
#include <grace/utils/numerics/runge_kutta.hh>
#include <grace/utils/numerics/weno_reconstruction.hh>

#endif /* GRACE_NUMERIC_UTILS_HH */