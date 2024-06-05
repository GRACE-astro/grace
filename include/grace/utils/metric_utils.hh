/**
 * @file metric_utils.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-04
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

#ifndef GRACE_UTILS_METRIC_HH
#define GRACE_UTILS_METRIC_HH

#include <grace_config.h>

#include <grace/utils/math.hh>
#include <grace/utils/constexpr.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <array> 

namespace grace {


struct metric_array_t {

GRACE_HOST_DEVICE
metric_array_t( std::array<double,6>const& g_
              , std::array<double,3>const& beta_ 
              , double const& alp_ )
    : _g(g_), _ginv(), _beta(beta_), _alp(alp_), _sqrtg(1.)
{
    _sqrtg = -(math::int_pow<2>(_g[2])*_g[3]) 
             + 2*_g[1]*_g[2]*_g[4] 
             - _g[0]*math::int_pow<2>(_g[4]) 
             - math::int_pow<2>(_g[1])*_g[5] 
             + _g[0]*_g[3]*_g[5]    ;
    _ginv[0] = (_g[3] - math::int_pow<2>(_g[4]))/_sqrtg;
    _ginv[1] = (-_g[1] + _g[2]*_g[4])/_sqrtg;
    _ginv[2] = (-(_g[2]*_g[3]) + _g[1]*_g[4])/_sqrtg;
    _ginv[3] = (2*_g[0] - math::int_pow<2>(_g[2]))/_sqrtg;
    _ginv[4] = (_g[1]*_g[2] - _g[0]*_g[4])/_sqrtg ; 
    _ginv[5] = (-math::int_pow<2>(_g[1]) + _g[0]*_g[3]) / _sqrtg;
}

double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
gamma(int i) const {
    return _g[i] ; 
}

double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
invgamma(int i) const {
    return _ginv[i] ;
}

double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
beta(int i) const {
    return _beta[i] ; 
}

double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
alp() const { return _alp ; }

std::array<double,3> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
raise(std::array<double,3> const& v ) const {
    return std::array<double,3> {
          _ginv[XX] * v[X] + _ginv[XY] * v[Y] + _ginv[XZ] * v[Z]
        , _ginv[XY] * v[X] + _ginv[YY] * v[Y] + _ginv[YZ] * v[Z]
        , _ginv[XZ] * v[X] + _ginv[YZ] * v[Y] + _ginv[ZZ] * v[Z]
    } ; 
}

std::array<double,3> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
lower(std::array<double,3> const& v ) const {
    return std::array<double,3> {
          _g[XX] * v[X] + _g[XY] * v[Y] + _g[XZ] * v[Z]
        , _g[XY] * v[X] + _g[YY] * v[Y] + _g[YZ] * v[Z]
        , _g[XZ] * v[X] + _g[YZ] * v[Y] + _g[ZZ] * v[Z]
    } ; 
}

double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
square_vec(std::array<double,3> const& v ) const {
    return  _g[XX] * v[X] * v[X] 
          + _g[YY] * v[Y] * v[Y] 
          + _g[ZZ] * v[Z] * v[Z] 
          + 2. * ( _g[XY] * v[X] * v[Y]  
                 + _g[XZ] * v[X] * v[Z] 
                 + _g[YZ] * v[Y] * v[Z] ) ; 
}

double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
square_covec(std::array<double,3> const& v ) const {
    return  _ginv[XX] * v[X] * v[X] 
          + _ginv[YY] * v[Y] * v[Y] 
          + _ginv[ZZ] * v[Z] * v[Z] 
          + 2. * ( _ginv[XY] * v[X] * v[Y]  
                 + _ginv[XZ] * v[X] * v[Z] 
                 + _ginv[YZ] * v[Y] * v[Z] ) ; 
}

std::array<double,6> _g, _ginv ;
std::array<double,3> _beta     ;
double _alp,_sqrtg             ; 

 private: 
    static constexpr int XX = 0 ; 
    static constexpr int XY = 1 ; 
    static constexpr int XZ = 2 ; 
    static constexpr int YY = 3 ; 
    static constexpr int YZ = 4 ; 
    static constexpr int ZZ = 5 ; 
    static constexpr int X = 0 ; 
    static constexpr int Y = 1 ; 
    static constexpr int Z = 2 ; 

} ;

}

#endif /* GRACE_UTILS_METRIC_HH */