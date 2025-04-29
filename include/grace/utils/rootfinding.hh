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


template <typename F>
double GRACE_HOST_DEVICE
brent(F&& f, const double &a, const double &b, const double t)
//brent(F& f, const double &a, const double &b, const double t)

//****************************************************************************80
//
//  Purpose:
//
//    ZERO seeks the root of a function F(X) in an interval [A,B].
//
//  Discussion:
//
//    The interval [A,B] must be a change of sign interval for F.
//    That is, F(A) and F(B) must be of opposite signs.  Then
//    assuming that F is continuous implies the existence of at least
//    one value C between A and B for which F(C) = 0.
//
//    The location of the zero is determined to within an accuracy
//    of 6 * MACHEPS * abs ( C ) + 2 * T.
//
//    Thanks to Thomas Secretin for pointing out a transcription error in the
//    setting of the value of P, 11 February 2013.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 February 2013
//
//  Author:
//
//    Original FORTRAN77 version by Richard Brent.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Richard Brent,
//    Algorithms for Minimization Without Derivatives,
//    Dover, 2002,
//    ISBN: 0-486-41998-3,
//    LC: QA402.5.B74.
//
//  Parameters:
//
//    Input, double A, B, the endpoints of the change of sign interval.
//
//    Input, double T, a positive error tolerance.
//
//    Input, func_base& F, the name of a user-supplied c++ functor
//    whose zero is being sought.  The input and output
//    of F() are of type double.
//
//    Output, double ZERO, the estimated value of a zero of
//    the function F.
//
{
  double c;
  double d;
  double e;
  double fa;
  double fb;
  double fc;
  double m;
  double p;
  double q;
  double r;
  double s;
  double sa;
  double sb;
  double tol;
  //
  //  Make local copies of A and B.
  //
  sa = a;
  sb = b;
  #define INVK(FUNC, ARG) std::invoke(std::forward<F>(FUNC), ARG);

  fa = INVK(f,sa);
  fb = INVK(f,sb);

  //fa = f(sa);
  //fb = f(sb);

  c = sa;
  fc = fa;
  e = sb - sa;
  d = e;

  constexpr double macheps = std::numeric_limits<double>::epsilon();

  for (;;) {
    if (std::fabs(fc) < std::fabs(fb)) {
      sa = sb;
      sb = c;
      c = sa;
      fa = fb;
      fb = fc;
      fc = fa;
    }

    tol = 2.0 * macheps * std::fabs(sb) + t;
    m = 0.5 * (c - sb);

    if (std::fabs(m) <= tol || fb == 0.0) {
      break;
    }

    if (std::fabs(e) < tol || std::fabs(fa) <= std::fabs(fb)) {
      e = m;
      d = e;
    } else {
      s = fb / fa;

      if (sa == c) {
        p = 2.0 * m * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        r = fb / fc;
        p = s * (2.0 * m * q * (q - r) - (sb - sa) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }

      if (0.0 < p) {
        q = -q;
      } else {
        p = -p;
      }

      s = e;
      e = d;

      if (2.0 * p < 3.0 * m * q - std::fabs(tol * q) &&
          p < std::fabs(0.5 * s * q)) {
        d = p / q;
      } else {
        e = m;
        d = e;
      }
    }
    sa = sb;
    fa = fb;

    if (tol < std::fabs(d)) {
      sb = sb + d;
    } else if (0.0 < m) {
      sb = sb + tol;
    } else {
      sb = sb - tol;
    }

    //fb = f(sb);
    fb = INVK(f,sb);

    #undef INVK

    if ((0.0 < fb && 0.0 < fc) || (fb <= 0.0 && fc <= 0.0)) {
      c = sa;
      fc = fa;
      e = sb - sa;
      d = e;
    }
  }
  return sb;
}
//****************************************************************************80

  /** @brief Newton-Raphson root finding algorithm. 
   *  @tparam F  callable object (struct with the suitable operator() defined, a lambda, an std::function object ...)
   *  @tparam DF : callable object (struct with the suitable operator() defined, a lambda, an std::function object ...)
   *  @param x0 : Initial guess
   *  @param a  : Low end of the bracket, feel free to set to a very high negative value for unconstrained newton raphson
   *  @param b  : High end of the bracket, feel free to set to a very high value for unconstrained newton raphson
   *  @param f  : object of class F representing the function 
   *  @param df  : object of class F representing the function's derivative 
   *  @param tol : Tolerance
   *  @param iter: initially set to the max iteration count, upon return contains the number of iterations the code went through 
   *  @return : a double which results from a Newton Rapshon step s.t. xk-1, xk satisfy the stopif criterion or whatever the last computed value is if iter > maxiter 
   *  Removed feature: noexcept (The function never throws. The user is responsible to check for failure by verifying that iter < maxiter.)
   * @details: 
   * Two template parameters are necessary in case when lambdas enter as f and df.
   * In that case, each automatically deduced lambda type is different,
   * and that necessitates F and DF.
   * The callable objects as passed by forward referencing to the std::invoke
   * In this way, we are not restricting ourselves to lvalues and can invoke this function also 
   * on lambdas 
   */ 
  template <typename F, typename DF>
  double GRACE_HOST_DEVICE
  rootfind_newton_raphson(double const& a, double const& b,
                            F&& f, DF&& df,
                            double const& tol, unsigned long& iter)
    {
      unsigned long const maxiter = iter ;

      constexpr const double macheps = std::numeric_limits<double>::epsilon() ;
      
      #define INVK(TYPE, FUNC, ARG) std::invoke(std::forward<TYPE>(FUNC), ARG);

      auto const fa = INVK(F, f, a); //f(a);
      auto const fb = INVK(F, f, b); //f(b);
      
      auto x0 = ( fa * a - fb * b ) / ( fa - fb ) ;
      //auto f0 = f(x0) ; 
      auto f0 = INVK(F, f,x0) ; 
    
      iter = 0 ; 
      auto x1 = x0 ;
    
      do {
        x0  =    x1 ;
        //f0  =  f(x1);
        //auto df0 = df(x1);
        f0 = INVK(F, f, x1);
        auto df0 = INVK(DF, df, x1);

        #undef INVK

        x1  = x0 - f0 / df0 ;

        if ( std::fabs(x0-x1) <= tol + 2. * macheps * std::fabs(x1) )
        return x1 ;

        if ( x1 < a || x1 > b ) { 
          if ( f0 * fa < 0 )
            x1 = ( fa * a - f0*x0 ) / ( fa - f0 ) ;
          else
            x1 = ( fb * b - f0*x0 ) / ( fb - f0 ) ;
        }
  
        iter ++ ;
        } while(  iter < maxiter  ) ;

        return x1 ; 
    }

}

#endif /* GRACE_UTILS_ROOTFINDING_HH */
