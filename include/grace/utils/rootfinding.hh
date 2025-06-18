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

template <typename F>
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
safe_brent(F&& f, const double &a, const double &b, const double t, int& err, int max_iter = 100) {
    // err = 0 Good run
    // err = 1 fa and fb have same sign (no root bracketed)
    // err = 2 exceeded max iteration
    
    err = 0; // Initialize error code
    
    double fa = f(a);
    double fb = f(b);
    double tol = t;
    
    // Check for same signs
    if (fa * fb > 0) {
        err = 1;
        return (a + b) * 0.5; // Return midpoint as fallback
    }
    
    // Check if we already have the root
    if (std::abs(fa) < tol) return a;
    if (std::abs(fb) < tol) return b;
    
    // Initialize variables for Brent's method
    double c;
    double d;
    double e;
    double fc;
    double m;
    double p;
    double q;
    double r;
    double s;
    double sa;
    double sb;
    
    // Make local copies of A and B
    sa = a;
    sb = b;
    
#define INVK(FUNC, ARG) std::invoke(std::forward<F>(FUNC), ARG)
    
    fa = INVK(f, sa);
    fb = INVK(f, sb);
    c = sa;
    fc = fa;
    e = sb - sa;
    d = e;
    
    constexpr double macheps = std::numeric_limits<double>::epsilon();
    
    int iter = 0; // Iteration counter
    
    for (;;) {
        // Check iteration limit
        if (iter >= max_iter) {
            err = 2;
            break;
        }
        
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
        
        fb = INVK(f, sb);
        
        if ((0.0 < fb && 0.0 < fc) || (fb <= 0.0 && fc <= 0.0)) {
            c = sa;
            fc = fa;
            e = sb - sa;
            d = e;
        }
        
        iter++; // Increment iteration counter
    }
    
#undef INVK
  return sb;
}
template <typename F>
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE safe_secant(F&& f, double x0, double x1, double tol, int& err, int max_iter = 20) {
    err = 0;
    
    double f0 = f(x0);
    double f1 = f(x1);
    
    // Check if we already have a root
    if (fabs(f0) < tol) {
        return x0;
    }
    if (fabs(f1) < tol) {
        return x1;
    }
    
    // Main secant iteration
    for (int iter = 0; iter < max_iter; ++iter) {
        // Check for divide by zero
        double df = f1 - f0;
        if (fabs(df) < 1e-15) {
            err = 3; // Derivative too small
            return x1;
        }
        
        // Secant step
        double x2 = x1 - f1 * (x1 - x0) / df;
        
        // Check convergence
        if (fabs(x2 - x1) < tol) {
            return x2;
        }
        
        // Update for next iteration
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x2);
        
        // Check if function value is small enough
        if (fabs(f1) < tol) {
            return x1;
        }
        
        // Safety check for non-finite values
        if (!isfinite(x2) || !isfinite(f1)) {
            err = 4; // Non-finite values
            return x1;
        }
    }
    
    err = 2; // Max iterations exceeded
    return x1;
}

// Alternative: Hybrid secant-bisection for better robustness
template <typename F>
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE safe_secant_bisection(F&& f, double a, double b, double tol, int& err, int max_iter = 25) {
    err = 0;
    
    double fa = f(a);
    double fb = f(b);
    
    // Ensure root is bracketed
  if (fa * fb > 0) {
    err = 1;
    return (a + b) * (double)0.5;
  }
  // Early exit if endpoints are roots
  if (fabs(fa) <= tol) return a;
  if (fabs(fb) <= tol) return b;

  // Initialize secant points
  double x0 = a, x1 = b;
  double f0 = fa, f1 = fb;

  // Main iteration
  for (int i = 0; i < max_iter; ++i) {
    // Secant candidate (safe division)
    double df = f1 - f0;
    double x_sec = x1 - f1 * (x1 - x0) /
                   ((fabs(df) > std::numeric_limits<double>::epsilon()) ? df : std::numeric_limits<double>::epsilon());

    // Choose secant or midpoint
    // Avoid divergence: few branches, rely on GPU predication
    bool in_bounds = (x_sec > a) & (x_sec < b);
    double x2      = in_bounds ? x_sec : (double)0.5 * (a + b);
    double f2      = f(x2);

    // Convergence: either function value small or interval small
    if (fabs(f2) <= tol || fabs(b - a) <= tol) {
      return x2;
    }

    // Update bracket
    bool left = (f2 * fa < 0);
    b  = left ? x2 : b;
    fb = left ? f2 : fb;
    a  = left ? a  : x2;
    fa = left ? fa : f2;

    // Shift secant points
    x0 = x1; f0 = f1;
    x1 = x2; f1 = f2;

    // Safety: check nonfinite
    if (!isfinite(x2) || !isfinite(f2)) {
      err = 4;
      return x2;
    }
  }

  // Max iterations reached
  err = 2;
  return x1;
}

}

#endif /* GRACE_UTILS_ROOTFINDING_HH */
