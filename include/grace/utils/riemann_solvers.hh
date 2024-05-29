/**
 * @file riemann_solvers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-13
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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

#ifndef GRACE_UTILS_RIEMANN_SOLVERS_HH 
#define GRACE_UTILS_RIEMANN_SOLVERS_HH

#include <grace_config.h>

#include <grace/utils/math.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device.h> 
#include <grace/data_structures/macros.hh>

namespace grace {

struct hll_riemann_solver_t 
{
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() (
          double const fL
        , double const fR 
        , double const uL 
        , double const uR 
        , double const cmin 
        , double const cmax  ) 
    {
        return ( cmin <= speed_eps ) ? fL : (
            (cmax <= speed_eps) ? fR : (cmin*fL + cmax*fR - cmax*cmin*(uR-uL))/(cmax+cmin)
        ) ; 
    }
 private:
    static constexpr double speed_eps = 1e-15 ; 
} ; 

}

#endif /* GRACE_UTILS_RIEMANN_SOLVERS_HH */