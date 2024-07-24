/**
 * @file runge_kutta.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-07-22
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

#ifndef GRACE_UTILS_RUNGEKUTTA_HH
#define GRACE_UTILS_RUNGEKUTTA_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <array> 

namespace grace {


template< typename N >
struct rk45_t {

template< typename F> 
std::array<double,N> GRACE_HOST_DEVICE 
solve(F&& rhs) const 
{
    std::array<double, N> state{id} ; 
    t = domain[0] ; 
    dt = ( domain[1] - domain[0] ) / 100. ; 
    while( t < domain[1] ) {
        advance_step(rhs, state) ; 
        t += dt ; 
    }
    return state ; 
}
template< typename F> 
void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
advance_step(F&& rhs, std::array<double,N>& state) {
    bool accepted = false ; 
    do{
        if ( dt < dt_min )
            break ; 
        auto k = compute_k(rhs,state) ; 
        double err = 0. ; 
        #pragma unroll 6
        for( int ik=0; ik<6; ++ik) {
            for( int iv=0; iv<N; ++iv){
                err += Kokkos::fabs((b5[ik] - b4[ik]) * k[ik][iv]) ;
            } 
        }
        err *= dt / N ; 
        if ( err > tol ) {
            accepted = false ; 
            dt *= 0.9 * math::int_pow<2>(tol/err); 
        } else if ( err < tol * 1e-2 ) {
            accepted = true ; 
            dt *= 0.9 * math::int_pow<2>((tol * 1e-2)/err) ; 
        }    
    } while ( not accepted  ) ;

    for( int iv=0 ;iv < N; ++iv ){
        double update = 0; 
        #pragma unroll 6
        for( int ik=0; ik<6; ++ik) {
            update += b4[ik] * k[ik] ; 
        }
        state[iv] += dt * update ; 
    }
}

template< typename F> 
std::array<std::array<double,N>,6> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
compute_k(F&& rhs, std::array<double,N>& state)
{
    std::array<std::array<double,N>,6> k ; 
    k[0] = rhs(t, state) ; 
    for( int ik=1; ik<6; ++ik) {
        auto tmpstate{ state } ; 
        auto tmpt { t }
        
        for( int jk=0; jk<ik; ++jk) {
            for( int iv=0; iv<N; ++iv){
                tmpstate[iv] += a[ik][jk] * k[jk][iv] ;
            }
            tmpt += c[jk] * dt ; 
        }
        
        k[ik] = rhs(tmpt, tmpstate) ; 
    }
    return std::move(k) ; 
}

std::array<double,2> domain ; 
std::array<double,N> id     ; 
double tol ; 
double t, dt  ; 

constexpr const std::array<double,6> c  { 0., 0.25, 3./8., 12./13., 1., 0.5 } ; 
constexpr const std::array<double,6> b5 {16./135., 0., 6656./12825., 28561./56430, -9./50., 2./55.} ; 
constexpr const std::array<double,6> b4 {25./216., 0., 1408./2565., 2197./4104., -0.2, 0} ; 
constexpr const std::array<std::array<double, 6>, 6> a {{
    { 0., 0., 0., 0., 0., 0. },
    { 0.25, 0., 0., 0., 0., 0. },
    { 3./32., 9./32., 0., 0., 0., 0. },
    { 1932./2197., -7200./2197., 7296./2197., 0., 0., 0. },
    { 439./216., -8., 3680./513., -845./4104., 0., 0. },
    { -8./27., 2., -3544./2565., 1859./4104., -11./40., 0. }
}};
constexpr const double dt_min = 1e-13 ; 

} ; 


}

#endif /* GRACE_UTILS_RUNGEKUTTA_HH */