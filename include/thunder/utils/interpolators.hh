/**
 * @file iterator.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-02-29
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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

#ifndef THUNDER_UTILS_INTERPOLATORS_HH 
#define THUNDER_UTILS_INTERPOLATORS_HH

#include <thunder/utils/inline.h>
#include <thunder/utils/device.h>

namespace utils {


namespace detail {

double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
linterp1d(double x0, double x1, double y0, double y1, double const& xt)
{
    double const lambda = ( xt - x0 ) / ( x1 - x0 ) ; 
    return y0*(1-lambda) + y1 * lambda ; 
}

}

template< size_t ndim > 
struct linear_interp_t 
{ } ; 

template<>
struct linear_interp_t<1>
{
    static constexpr size_t stencil_size = 2UL ; 

    double *x, *y; 
    
    static THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    void get_parametric_coordinates(int xp[])
    {   
        xp[0] = 0; xp[1] = 1;
    }

    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    interpolate( double const& x0 )
    {
        return detail::linterp1d(x[0],x[1],y[0],y[1],x0) ; 
    }
} ; 

template<>
struct linear_interp_t<2>
{
    static constexpr size_t stencil_size = 2UL ; 

    double *x, *y;

    static THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    void get_parametric_coordinates(int xp[])
    {   
        for( int is=0; is<4; ++is){
            xp[2*is + 0UL] = (is%2) ; 
            xp[2*is + 1UL] = (is/2) ; 
        }
    }

    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    interpolate( double const& x0, double const& y0 )
    {
        double const c00 =  detail::linterp1d(x[2*0+0],x[2*1+0],y[0],y[1],x0) ; 
        double const c10 =  detail::linterp1d(x[2*2+0],x[2*3+0],y[2],y[3],x0) ; 
        return detail::linterp1d(x[2*0+1],x[2*2+1], c00, c10, y0) ; 
    }
} ; 

template<>
struct linear_interp_t<3>
{
    static constexpr size_t stencil_size = 2UL ; 

    double *x, *y; 

    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    interpolate( double const& x0, double const& y0, double const& z0 )
    {
        double const c00 =  detail::linterp1d(x[3*0+0],x[3*1+0], y[0],y[1], x0) ; 
        double const c10 =  detail::linterp1d(x[3*2+0],x[3*3+0], y[2],y[3], x0) ; 
        double const c0  =  detail::linterp1d(x[3*0+1],x[3*2+1], c00 , c10, y0) ; 
        double const c01 =  detail::linterp1d(x[3*4+0],x[3*5+0], y[4],y[5], x0) ; 
        double const c11 =  detail::linterp1d(x[3*6+0],x[3*7+0], y[6],y[7], x0) ; 
        double const c1  =  detail::linterp1d(x[3*4+1],x[3*6+1], c01 , c11, y0) ;
        return detail::linterp1d(x[3*0+2], x[3*4+2], c0, c1, z0) ;  
    }
} ; 

} /* namespace utils */

#endif /* THUNDER_UTILS_INTERPOLATORS_HH */