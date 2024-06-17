/**
 * @file rootfinding.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-10
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

#ifndef GRACE_UTILS_ROOTFINDING_HH
#define GRACE_UTILS_ROOTFINDING_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>

namespace utils {

template< typename F >
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
bisection(F&& func, double const& a, double const& b, double const& tol)
{
    double xa{a}, xb{b}, xc; 
    double fa{func(a)}, fb{func(b)}, fc ; 
    if ( fa * fb > 0 ) {
        return std::numeric_limits<double>::quiet_NaN(); 
    }
    if ( fa == 0  ) {
        return a ;
    } else if ( fb == 0 ) {
        return b ; 
    }
    do {
        xc = 0.5 * ( xa + xb ) ; 
        fc = func(xc) ; 
        if( fa * fc < 0 ) { 
            fb = fc ; 
            xb = xc ; 
        } else if(fb*fc < 0) {
            fa = fc ; 
            xa = xc ;
        } else if ( fa == 0 ) {
            return xa ; 
        } else if ( fb == 0 ) {
            return xb ; 
        } else if ( fc == 0 ) {
            return xc ; 
        }
    } while( math::abs(xa-xb) > tol ) ; 
    return xc ; 
}
void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
brent_bracket(double& a, double& b, double& fa, double& fb) {
    double s, fs ; 
    if ( math::abs(fa) < math::abs(fb) ) {
        s = b ; 
        b = a ; 
        a = s ; 
        fs = fb ; 
        fb = fa ;
        fa = fs ; 
    }
}
template< typename F >
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
brent(F&& func, double const& a, double const& b, double const& tol)
{
    double xa{a}, xb{b}, xc, xd, xs, fa{func(a)}, fb{func(b)}, fs, fc, fd ;
    if ( fa * fb > 0 ) {
        return std::numeric_limits<double>::quiet_NaN(); 
    }
    if ( fa == 0  ) {
        return a ;
    } else if ( fb == 0 ) {
        return b ; 
    }
    brent_bracket(xa,xb,fa,fb) ; 
    xc = xa ; fc = fa ;
    bool mflag{true} ; 

    bool noconv{true} ;
    do {
        if ( fa != fs and fb != fs ) {
            xs = xb * fb*fs / ( fa - fb ) / ( fa - fs ) 
               + xa * fa*fs / ( fb - fa ) / ( fb - fs ) 
               + xs * fa*fb / ( fs - fa ) / ( fs - fb ) ; 
        } else {
            xs = xb - fb * (xb-xa) / (fb-fa) ; 
        }
        if (  not ( 0.25*(3.*xa+xb) < xs and xs < xb)
           or (mflag and ( math::abs(xs-xb)>=0.5*math::abs(xb-xc)))
           or (!mflag and (math::abs(xs-xb)>=0.5*math::abs(xc-xd)))
           or (mflag and math::abs(xb-xc) < tol)
           or (!mflag and math::abs(xc-xd)<tol )) {
            xs = 0.5 * (xa+xb) ; 
        } else {
            mflag = false; 
        }
        fs = func(xs) ; 
        xd = xc ;
        fd = fc ; 
        xc = xb ;
        fc = fb ; 
        if( fa*fs < 0 ) {
            xb = xs ;
            fb = fs ;
        } else {
            xa = xs ; 
            fs = fs ;
        }

        noconv =  math::abs(xa-xb) < tol 
              or fs == 0. ; 
    } while( noconv ) ;
    return xs ; 
}

}

#endif /* GRACE_UTILS_ROOTFINDING_HH */