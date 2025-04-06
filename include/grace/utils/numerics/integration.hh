/**
 * @file integration.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-04-15
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
#ifndef GRACE_UTILS_INTEGRATION_HH 
#define GRACE_UTILS_INTEGRATION_HH

#include <grace/utils/inline.h>
#include <grace/utils/device/device.h> 
#include <grace/utils/numerics/constexpr.hh> 
#include <grace/utils/numerics/affine_transformation.hh>

#include <cmath> 
#include <array> 
#include <vector>
#include <functional>
#include <numeric>

namespace utils {

namespace detail {

namespace helpers{
constexpr const std::array<double,1> xGauLeg1   = {0};
constexpr const std::array<double,2> xGauLeg2   = {-0.577350269189626, 0.57735026918963};
constexpr const std::array<double,3> xGauLeg3   = {-0.77459666924148, 0., 0.77459666924148}; 
constexpr const std::array<double,4> xGauLeg4   = {-0.86113631159405, -0.33998104358486,0.33998104358486, 0.86113631159405};
constexpr const std::array<double,5> xGauLeg5   = {-0.90617984593866, -0.53846931010568, 0., 0.5384693101057, 0.9061798459387};
constexpr const std::array<double,6> xGauLeg6   = {-0.93246951420315, -0.66120938646626, -0.23861918608320, 0.2386191860832, 0.6612093864663, 0.9324695142032}; 
constexpr const std::array<double,7> xGauLeg7   = {-0.9491079123428,-0.74153118559939, -0.4058451513774, 0., 0.4058451513774, 0.7415311855994, 0.9491079123428};
constexpr const std::array<double,8> xGauLeg8   = {-0.9602898564975, -0.79666647741363, -0.5255324099163, -0.1834346424956, 0.1834346424956, 0.5255324099163, 0.7966664774136, 0.9602898564975}; 
constexpr const std::array<double,9> xGauLeg9   = {-0.9681602395076, -0.83603110732664, -0.6133714327006, -0.3242534234038,0., 0.3242534234038, 0.6133714327006, 0.8360311073266, 0.9681602395076}; 
constexpr const std::array<double,10> xGauLeg10 = {-0.9739065285172, -0.8650633666890, -0.6794095682990, -0.4333953941292, -0.1488743389816, 0.1488743389816, 0.4333953941292, 0.6794095682990, 0.8650633666890, 0.9739065285172};

constexpr const std::array<double,1> wGauLeg1   = {2.000000000000000};
constexpr const std::array<double,2> wGauLeg2   = {1.00000000000000, 1.00000000000000};
constexpr const std::array<double,3> wGauLeg3   = {0.555555555555556, 0.888888888888889, 0.555555555555556}; 
constexpr const std::array<double,4> wGauLeg4   = {0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454};
constexpr const std::array<double,5> wGauLeg5   = {0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189};
constexpr const std::array<double,6> wGauLeg6   = {0.171324492379170, 0.360761573048139, 0.467913934572691, 0.467913934572691, 0.360761573048139, 0.171324492379170};
constexpr const std::array<double,7> wGauLeg7   = {0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469, 0.381830050505119, 0.279705391489277, 0.129484966168870} ; 
constexpr const std::array<double,8> wGauLeg8   = {0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362, 0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376};
constexpr const std::array<double,9> wGauLeg9   = {0.0812743883615744, 0.180648160694857, 0.260610696402935, 0.312347077040003, 0.330239355001260, 0.312347077040003, 0.260610696402935, 0.180648160694857, 0.0812743883615744};
constexpr const std::array<double,10> wGauLeg10 = {0.0666713443086881, 0.149451349150581, 0.219086362515982, 0.269266719309996, 0.295524224714753, 0.295524224714753, 0.269266719309996, 0.219086362515982, 0.149451349150581, 0.0666713443086881};
} /* namespace helpers */

/**
 * @brief Perform recursive trapezoidal integration in 1D.
 * \cond grace_detail
 * \ingroup utils
 * @tparam T Type of integrand argument.
 * @tparam F Type of integrand function.
 * @param s Result of previous recursion.
 * @param a Lower bound of integration interval.
 * @param b Upper bound of integration interval.
 * @param n Level of recursion.
 * @param func Integrand function.
 * @return T Integral of <code>func</code> between <code>a</code> and <code>b</code>.
 */
template<  typename T
        ,  typename F>
T GRACE_ALWAYS_INLINE  
trapezoid_impl(T const& s, T const& a, T const& b, int n, F&& func)
{
    if( n==1 ) {
        return 0.5*(b-a)*(func(a)+func(b)) ;
    } else {
        int it = 1 << (n-2) ;
        T tnm = static_cast<T>(it) ;
        T del = (b-a)/tnm ;
        T x = a + 0.5*del;
        T sum{0.} ;
        for( int j=1; j<=it; ++j){
            sum += func(x) ;
            x+= del ;
        }
        return 0.5*(s+(b-a)*sum/tnm );
    }
}
/**
 * @brief Compute nodes and weights for gauss legendre quadrature
 *        in [-1,1].
 * 
 * @tparam N Number of nodes
 * @tparam T Type of data
 * @param x Node array 
 * @param w Weights Array 
 * @param eps 
 */
template< size_t N, typename T> 
void gauleg(std::array<T,N>& x, std::array<T,N>& w, T const& eps) {
  
  int m = (N+1)/2 ;
  for( int i=1; i<=m; ++i) {
    T z = cos(M_PI*(i-0.25)/(N+0.5));
    T z1,pp; 
    do {
      T p1{1.}, p2{0.0} ;
      for( int j=1; j<=N; ++j) {
        T p3=p2 ;
        p2 = p1 ;
        p1 = ((2.*j-1.)*z*p2-(j-1.)*p3)/static_cast<T>(j) ;
      }
      pp = N*(z*p1-p2)/(z*z-1.) ;
      z1 = z ;
      z = z1-p1/pp ;
    } while( fabs(z-z1) > eps ) ; 
    x[i-1] = - z;
    x[N-i] =   z ;
    w[i-1] = 2./((1.-z*z)*pp*pp) ;
    w[N-i] = w[i-1] ; 
  }
}

}
/**
 * @brief Perform simpson integration of a 1D function
 *        to a required tolerance using a recursive trapezoidal rule.
 * \ingroup utils
 * @tparam T Type of integrand argument.
 * @tparam F Type of integrand function.
 * @param a Lower bound of integration interval.
 * @param b Upper bound of integration interval.
 * @param eps Relative error tolerance.
 * @param func Integrand function.
 * @param imax Maximum recursion level.
 * @return T Integral of <code>func</code> between <code>a</code> and <code>b</code>.
 */
template< typename T
        , typename F >
T GRACE_ALWAYS_INLINE  
simpson(T const& a, T const& b, T const& eps, F&& func, size_t const imax=30) {
  T s,st,os{-1e30},ost{-1e30} ;
  for( int j=1; j<=imax; ++j) {
    st = detail::trapezoid_impl(st,a,b,j,func) ;
    s = ( 4.0*st - ost ) / 3.0 ;
    if( j > 5 ) {
      if( ( fabs(s-os) < eps*fabs(os) )  ||
          ( s==0. && os==0. ) ) {
        return s ;
      }
    }
    os=s ; ost = st ;
  }
  return 0. ;
}

template< size_t N
        , typename T
	      , typename F >
T inline
gauss_legendre_quadrature(T const& a, T const& b, T const& eps, F const& func) {
  using namespace detail ; 
  std::array<T,N> x{0.},w{0.} ;
  if constexpr (N==1) {
    x = helpers::xGauLeg1 ; w = helpers::wGauLeg1 ;
  } else if constexpr (N==2) {
    x = helpers::xGauLeg2 ; w = helpers::wGauLeg2 ;
  } else if constexpr (N==3) {  
    x = helpers::xGauLeg3 ; w = helpers::wGauLeg3 ;
  } else if constexpr (N==4) {
    x = helpers::xGauLeg4 ; w = helpers::wGauLeg4 ;
  } else if constexpr (N==5) {
    x = helpers::xGauLeg5 ; w = helpers::wGauLeg5 ;
  } else if constexpr (N==6) {
    x = helpers::xGauLeg6 ; w = helpers::wGauLeg6 ;
  } else if constexpr (N==7) {
    x = helpers::xGauLeg7 ; w = helpers::wGauLeg7 ;
  } else if constexpr (N==8) {
    x = helpers::xGauLeg8 ; w = helpers::wGauLeg8 ;
  } else if constexpr (N==9) {
    x = helpers::xGauLeg9 ; w = helpers::wGauLeg9 ;
  } else if constexpr (N==10) {
    x = helpers::xGauLeg10 ; w = helpers::wGauLeg10 ;
  } else {
    detail::gauleg<N,T>(x,w,eps) ;
  }
  T sum = 0. ;
  utils::ForwardLoop<N,0>([&] (size_t i){ 
    sum += w[i] * func(utils::affine_transformation(x[i],-1.,1.,a,b));
  } ) ;
 
  return 0.5*(b-a)*sum ; 
}

namespace detail {

/**
 * @brief Helper class for N dimensional integration over 
 *        a hyperrectangle.
 * \cond grace_detail
 * \ingroup utils
 * @tparam T Type of integrand argument(s).
 * @tparam F Type of integrand function.
 * @tparam Ndim Number of dimensions.
 */
template< typename T
        , typename F
        , size_t Ndim >
struct integrator_t {
  /**
   * @brief Perform the integral.
   */
  static GRACE_ALWAYS_INLINE T 
  integrate( std::array<T,Ndim> const& a
           , std::array<T,Ndim> const& b
           , T const& eps, F const& func
           , size_t const imax=30 ) 
  {
    auto const _oned_integrand = [&] (T const& xi) {
      auto const _ndm1_func = [&] ( auto ... args ) {
        return func(args...,xi) ;
      } ;
      using _ndm1_integrator =
        integrator_t<T, decltype(_ndm1_func),Ndim-1> ;
      std::array<T,Ndim-1> _a,_b;
      for(int i=0; i<Ndim-1;++i) {
        _a[i] = a[i] ; _b[i] = b[i] ;
      }
      return _ndm1_integrator::integrate(_a,_b,eps, _ndm1_func, imax) ;
    } ;
    using _oned_integrator = integrator_t<T,decltype(_oned_integrand),1> ;
    return _oned_integrator::integrate({a[Ndim-1]}, {b[Ndim-1]}, eps, _oned_integrand, imax);
  } ;
} ;

/**
 * @brief Specialization of helper class for N dimensional integration over 
 *        a hyperrectangle to 1D.
 * \cond grace_detail
 * \ingroup utils
 * @tparam T Type of integrand argument(s).
 * @tparam F Type of integrand function.
 */
template< typename T
        , typename F >
struct integrator_t<T,F,1> {
  /**
   * @brief Perform the integral.
   */
  static GRACE_ALWAYS_INLINE T 
  integrate( std::array<T,1> const& a
           , std::array<T,1> const& b
           , T const& eps, F const& func
           , size_t const imax=30) {
    return simpson(a[0],b[0],eps,func,imax) ;
  } ;
} ;

template< typename T
	      , typename F
	      , size_t Ndim 
        , size_t Npoints>
  struct quadrature_integrator_t {
    static inline T integrate(std::array<T,Ndim> const& a, std::array<T,Ndim> const& b, F const& func, T const eps_gaussleg = 1e-15) {
      auto const _oned_integrand = [&] (T const& xi) {
        auto const _ndm1_func = [&] ( auto ... args ) {
          return func(args...,xi) ; 
        } ;
        using _ndm1_integrator =
          quadrature_integrator_t<T, decltype(_ndm1_func),Ndim-1,Npoints> ;
        std::array<T,Ndim-1> _a,_b;
        utils::ForwardLoop<Ndim-1,0>([&] (size_t i) { _a[i] = a[i]; _b[i] = b[i]; } ) ;
        return _ndm1_integrator::integrate(std::move(_a),std::move(_b), _ndm1_func, eps_gaussleg) ;
      } ;
      using _oned_integrator = quadrature_integrator_t<T,decltype(_oned_integrand),1,Npoints> ;
      return _oned_integrator::integrate(std::move(std::array<double,1>{a[Ndim-1]}), std::move(std::array<double,1>{b[Ndim-1]}),  _oned_integrand, eps_gaussleg); 
    } ;
  } ; 
  
  template< typename T
	        , typename F 
          , size_t Npoints >
  struct quadrature_integrator_t<T,F,1,Npoints> {
    static inline T integrate(std::array<T,1> const& a, std::array<T,1> const& b,F const& func, T const eps_gaussleg=1e-15) {
	    return  gauss_legendre_quadrature<Npoints>(a[0],b[0],eps_gaussleg,func) ;
    } ;
  };
} /* namespace detail */

/**
 * @brief Perform an integral of a scalar function over an N-dimensional 
 *        hyperrectangle using Simpson's rule.
 * \ingroup utils
 * @tparam Ndim Number of dimensions.
 * @tparam T Type of integrand argument(s).
 * @tparam F Type of integrand function.
 * @param a Array containing the lower bounds of integration.
 * @param b Array containing the upper bounds of integration.
 * @param eps Relative error tolerance.
 * @param func Integrand function \f$ f: I^{N_d} \subset R^{N_d} \rightarrow R\f$
 * @param imax Maximum level of recursion for achieving requested tolerance.
 * @return T The integral of the specified function over the hyper-rectangular domain
 *           \f$~\bigotimes_{i=1}^{N_d}~[a_i,b_i]\f$
 */
template< size_t Ndim
        , typename T
        , typename F >
T nd_simpson_integrate(std::array<T,Ndim>const& a, std::array<T,Ndim> const& b, T const& eps, F const & func, size_t const imax=30) {
  return detail::integrator_t<T,F,Ndim>::integrate(a,b,eps,func,imax) ;
}

template< size_t Ndim
        , size_t Npoints
        , typename T
	      , typename F >
T nd_quadrature_integrate(std::array<T,Ndim>const& a, std::array<T,Ndim> const& b, F const & func, T const eps_gaussleg=1e-15) {
  return detail::quadrature_integrator_t<T,F,Ndim,Npoints>::integrate(a,b,func,eps_gaussleg) ; 
}

template< typename F >
static double GRACE_ALWAYS_INLINE 
eval_2d_primitive(double a1, double a2, double b1, double b2, F&& func)
{
    return (func(a1,a2) + func(b1,b2)) - (func(a1,b2) + func(b1,a2)) ; 
}

template< typename F >
static double GRACE_ALWAYS_INLINE 
eval_3d_primitive(double a1, double a2, double a3, double b1, double b2, double b3, F&& func)
{
    auto const f2d_a = [&] (double x, double y) {
        return func(x,y,a3) ; 
    }  ; 
    auto const f2d_b = [&] (double x, double y) {
        return func(x,y,b3) ; 
    } ; 
    return eval_2d_primitive(a1,a2,b1,b2,f2d_b) - eval_2d_primitive(a1,a2,b1,b2,f2d_a); 
}




/**
 * @brief Perform a simple one-d (uniform-resolution) integral of an array using the midpoint's rule and knowledge of the function sampling
 * \ingroup utils
 * @tparam Ndim Number of dimensions.
 * @tparam T Type of integrand argument(s).
 * @param dx grid spacings in each dimension
 * @param func flattened integrand function values 
 * @return T The integral of the sampled values over the hyper-rectangular domain
 */
template< size_t Ndim, typename T>
T oned_midpoint_integrator(std::array<T,Ndim> const& dx_i,
                         std::vector<T> const& f_i) {
        
    return  std::accumulate(dx_i.begin(), dx_i.end(), 1.0, std::multiplies<T>()) *\
             std::accumulate(f_i.begin(), f_i.end(), 0.0, std::plus<T>());
}



} /* namespace utils */

#endif 