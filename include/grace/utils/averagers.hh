/**
 * @file averagers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-20
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
#ifndef GRACE_UTILS_AVERAGERS
#define GRACE_UTILS_AVERAGERS

#include <grace_config.h>

namespace utils {

template< size_t ndim >
struct point_values_averager_t {} ; 

template<>
point_values_averager_t<1>
{
    static constexpr size_t stencil_size = 2UL ; 
    
    double *x, *y; 
    
    static GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    void get_parametric_coordinates(int xp[])
    {   
        xp[0] = 0; xp[1] = 1;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate( double const& x0 )
    {
        return detail::linterp1d(x[0],x[1],y[0],y[1],x0) ; 
    }
} ; 
} /* namespace utils */ 

#endif /* UTILS_AVERAGERS */
