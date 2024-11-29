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

#include <Kokkos_Core.hpp>

namespace grace {

template< size_t N >
struct rk4_t {

GRACE_HOST_DEVICE
rk4_t(std::array<double,2> _domain, std::array<double,N> _id, size_t Nt)
    : domain(_domain), id(_id), state(id), t(domain[0]), dt((domain[1]-domain[0])/Nt),  _it(0), _Nt(Nt)
    , c({ 0., 0.5, 0.5, 1. })
    , b({1./6., 1./3., 1./3., 1./6.})
    , a({{ 0., 0., 0., 0. },{ 0.5, 0., 0., 0. },{ 0., 0.5, 0., 0. },{ 0., 0.,  0., 1. }})
{ }


template< typename F>
void GRACE_HOST_DEVICE
solve(F&& rhs)
{
    while( _it < _Nt ) {
      advance_step(std::forward<F>(rhs)) ;
    }
}

template< typename F>
void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
advance_step(F&& rhs) {
    std::array<std::array<double,N>,4> k ;
    k = compute_k(std::forward<F>(rhs)) ;
    update_state(k);
    _it ++ ; 
    t+=dt ;   
}

private: 

void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
update_state(std::array<std::array<double,N>,4> const& k) {
    for( int iv=0 ;iv < N; ++iv ){
        double update = 0;
        #pragma unroll 4
        for( int ik=0; ik<4; ++ik) {
            update += b[ik] * k[ik][iv] ;
        }
        state[iv] += dt * update ;
    }

}

template< typename F>
std::array<std::array<double,N>,4> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
compute_k(F&& rhs)
{
    std::array<std::array<double,N>,4> k ;
    k[0] = rhs(t, state) ;
    for( int ik=1; ik<4; ++ik) {
        auto tmpstate{ state } ;
        auto tmpt { t } ;

        for( int jk=0; jk<ik; ++jk) {
            for( int iv=0; iv<N; ++iv){
                tmpstate[iv] += dt * a[ik][jk] * k[jk][iv] ;
            }
            tmpt += c[jk] * dt ;
        }

        k[ik] = rhs(tmpt, tmpstate) ;
    }
    return std::move(k) ;
}

public:

std::array<double,2> domain ; 
std::array<double,N> id     ; 
double abs_tol, rel_tol ;
double t, dt  ; 
std::array<double, N> state;

size_t _Nt, _it ; 

const std::array<double,4> c ; 
const std::array<double,4> b ; 
const std::array<std::array<double,4>,4> a ;

} ; 


template< size_t N >
struct rk45_t {

GRACE_HOST_DEVICE
rk45_t(std::array<double,2> _domain, std::array<double,N> _id, double _abs_tol, double _rel_tol=0.)
    : domain(_domain), id(_id), abs_tol(_abs_tol), rel_tol(_rel_tol), state(id), t(domain[0]), dt((domain[1]-domain[0])/100)
    , c({ 0., 0.25, 3./8., 12./13., 1., 0.5 })
    , b5({16./135., 0., 6656./12825., 28561./56430, -9./50., 2./55.})
    , b4({25./216., 0., 1408./2565., 2197./4104., -0.2, 0})
    , a({{ 0., 0., 0., 0., 0., 0. },
         { 0.25, 0., 0., 0., 0., 0. },
         { 3./32., 9./32., 0., 0., 0., 0. },
         { 1932./2197., -7200./2197., 7296./2197., 0., 0., 0. },
         { 439./216., -8., 3680./513., -845./4104., 0., 0. },
         { -8./27., 2., -3544./2565., 1859./4104., -11./40., 0. }})
  {}

template< typename F>
void GRACE_HOST_DEVICE
solve(F&& rhs)
{
    while( t < domain[1] ) {
      advance_step(std::forward<F>(rhs)) ;
    }
}

template< typename F>
void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
advance_step(F&& rhs) {
    bool accepted = false ;
    bool must_accept = false ;
    std::array<std::array<double,N>,6> k ;

    double dt_max = domain[1]-t ;
    do{
      dt = std::min(dt, dt_max) ;
      k = compute_k(std::forward<F>(rhs)) ;
      double const err = compute_error(k);
      double ynorm = 0 ;
      for( int i=0; i<N; ++i) {
        ynorm += state[i] * state[i] ;
      }
      double const tol = abs_tol + rel_tol * Kokkos::sqrt(ynorm) ;
      if ( err < tol or must_accept ) {
        t += dt ;
        update_state(k);
        dt *= Kokkos::min(5.0, 0.9 * Kokkos::pow(tol/err,0.2)) ;
        accepted = true ;
      } else {
        dt *= Kokkos::max(0.1, 0.9 * Kokkos::pow(tol/err, 0.2)) ;
        accepted = false ;
      }

      if( dt < dt_min ) {
        dt = dt_min ;
        must_accept = true ;
      }

    } while ( not accepted ) ;
}

private: 

void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
update_state(std::array<std::array<double,N>,6> const& k) {
    for( int iv=0 ;iv < N; ++iv ){
        double update = 0;
        #pragma unroll 6
        for( int ik=0; ik<6; ++ik) {
            update += b4[ik] * k[ik][iv] ;
        }
        state[iv] += dt * update ;
    }

}

double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
compute_error(std::array<std::array<double,N>,6> const& k) const {

  auto const scale = compute_scale() ;
  double err = 0 ;
  for( int i=0 ; i< N; ++i  ){
    double eps = 0 ;
    for( int j=0; j<6; ++j) {
      eps += ( b5[j] - b4[j] ) * k[j][i] ;
    }
    err += math::int_pow<2>(eps / scale[i] ) ;
  }

  return Kokkos::sqrt(err / N) * dt + 1e-99 ;

}

std::array<double,N> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
compute_scale() const {
    std::array<double, N> scale ;

    #pragma unroll N
    for( int i=0; i<N; ++i) {
        scale[i] = abs_tol + rel_tol * Kokkos::fabs(state[i]) ;
    }

    return std::move(scale) ;
}

template< typename F>
std::array<std::array<double,N>,6> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
compute_k(F&& rhs)
{
    std::array<std::array<double,N>,6> k ;
    k[0] = rhs(t, state) ;
    for( int ik=1; ik<6; ++ik) {
        auto tmpstate{ state } ;
        auto tmpt { t } ;

        for( int jk=0; jk<ik; ++jk) {
            for( int iv=0; iv<N; ++iv){
                tmpstate[iv] += dt * a[ik][jk] * k[jk][iv] ;
            }
            tmpt += c[jk] * dt ;
        }

        k[ik] = rhs(tmpt, tmpstate) ;
    }
    return std::move(k) ;
}

public:

std::array<double,2> domain ; 
std::array<double,N> id     ; 
double abs_tol, rel_tol ;
double t, dt  ; 
std::array<double, N> state;

const std::array<double,6> c  ; 
const std::array<double,6> b5 ; 
const std::array<double,6> b4 ; 
const std::array<std::array<double,6>,6> a ;
static constexpr const double dt_min = 1e-13 ; 

} ; 


}

#endif /* GRACE_UTILS_RUNGEKUTTA_HH */