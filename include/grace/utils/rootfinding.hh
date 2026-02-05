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
#include <grace/utils/LU_utils.hh>

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
brent(F& f, const double &a, const double &b, const double t)

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
  fa = f(sa);
  fb = f(sb);

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

    fb = f(sb);

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
  template <typename F>
double KOKKOS_FUNCTION
rootfind_newton_raphson(double sa, double sb,
                        F&& f, unsigned long const maxiter, double tol, int& err)
    {

      constexpr const double macheps = std::numeric_limits<double>::epsilon() ;
      double a{sa}, b{sb} ;       
      double fa,fb,dummy ; 
      f(a,fa,dummy); 
      f(b,fb,dummy); 
      if ( fa == 0 ) {
        err = 0 ; 
        return a ; 
      } else if ( fb == 0 ) {
        err = 0 ; 
        return b; 
      }

      // check the root is bracketed 
      if ( fa * fb > 0 ) { 
        err = 1 ;
        return 0 ; 
      }
      double x = ( fb * a - fa * b ) / ( fb - fa ) ;    
      int iter = 0 ; 
      
      double t, xold, fx, dfx; 
      do {
        xold = x ; 
        f(x,fx,dfx) ; 
        if ( fx * fb > 0 ) {
          fb = fx ; 
          b = x ;
        } else if ( fx * fa > 0 ) {
          fa = fx ; 
          a = x ; 
        }
        x -= fx / dfx ; 
        t = 2. * macheps * fabs(x) + tol;
        if ( fabs(x-xold) < t || fx == 0 ) {
          err = 0 ; 
          return x ; 
        }
        if ( x > b or x < a) {
          x = 0.5 * ( b + a ) ; 
        }
        iter ++ ;
      } while(  iter < maxiter  ) ;
      err = 1 ; 
      return 0 ; 
}

  template <typename F>
double KOKKOS_FUNCTION
rootfind_newton_raphson_unsafe(double x0, F&& f, unsigned long const maxiter, double tol, int& err)
    {

      constexpr const double macheps = std::numeric_limits<double>::epsilon() ;
      double x, fx, dfx ; 
      x = x0; 
      f(x,fx,dfx) ; 
      if ( fx == 0 ) {
        err = 0 ;
        return x0 ; 
      } 
 
      int iter = 0 ; 
      
      double t, xold; 
      do {
        if (fabs(dfx) < 10 * macheps) {
            err = 2;
            return x;
        }

        xold = x ; 
        x -= fx / dfx ; 
        f(x,fx,dfx) ; 
        // Robust convergence check
        t = 2. * macheps * fabs(x) + tol;
        if (fabs(x-xold) < t || fabs(fx) < tol) {
            err = 0;
            return x;
        }
        iter ++ ;
      } while(  iter < maxiter  ) ;
      err = 1 ; 
      return 0 ; 
}
//****************************************************************************
enum nr_err_t {
  SUCCESS=0,
  ERR_ROUNDOFF,
  ERR_SMALLSTEP,
  ERR_STAGNATION
} ; 

template < size_t ND,  typename FT  >
void GRACE_HOST_DEVICE
lnsrch( 
  FT&& func, double (&xold)[ND], double fold, double (&g)[ND], double (&p)[ND], 
  double (&x)[ND], double *f, double stpmax, int *check )
{
    static constexpr double TOLX = 5. * std::numeric_limits<double>::epsilon()  ; 
    static constexpr double ALF = 1e-4 ; 
    *check=0; 
    double sum = 0 ; 
    for( int i=0; i<ND; ++i) sum+=SQR(p[i]) ;
    sum = sqrt(sum) ;  
    if ( sum > stpmax ) {
        for( int i=0; i<ND; ++i) p[i] /= stpmax/sum ; 
    }
    double slope = 0 ; 
    for( int i=0; i<ND; ++i) slope += g[i]*p[i] ; 
    if (slope>=0) {
        *check=ERR_ROUNDOFF;
        return ;
    }
    double test=0.;
    for( int i=0; i<ND; ++i) {
        test = fmax(test, fabs(p[i])/fmax(fabs(xold[i]),1.)) ; 
    }
    double alamin = TOLX/test ;
    double alam = 1.0;
    int nfix = 0; 
    do {
        for( int i=0; i<ND; ++i) x[i] = xold[i] + alam * p[i]; 
        *f = func(x) ; 
        if (*f <= fold + ALF * alam * slope) { 
            return ; // good enough 
        } else if ( alam < alamin ) {
            for ( int i=0; i<ND; ++i) x[i] = xold[i] ; 
            *check = ERR_SMALLSTEP;
            return ;
        } else {
            double tmplam{0}, alam2{0}, f2{0} ; // initialize to silence warnings
            if ( nfix == 0 ) {
                tmplam = - slope / (2.*(*f-fold-slope)) ; 
            } else {
                double r1 = *f-fold-alam*slope;
                double r2 = f2-fold-alam2*slope;
                double a = (r1/SQR(alam)-r2/SQR(alam2)/(alam-alam2));
                double b = (-alam2*r1/SQR(alam)+alam*r2/SQR(alam2)/(alam-alam2));
                if ( a==0 ) {
                tmplam = -slope/(2.*b) ; 
                } else {
                double disc = SQR(b) - 3. * a * slope ;
                if ( disc < 0. ){ 
                    tmplam = 0.5 * alam ; 
                } else if ( b<=0 ){ 
                    tmplam=(-b+sqrt(disc))/(3.*a) ; 
                } else {
                    tmplam = -slope/(b+sqrt(disc)) ; 
                }
                tmplam = fmin(tmplam,0.5*alam) ; 
                }
            } // not first rodeo
            alam2 = alam ; 
            f2 = *f ; 
            alam = fmax(tmplam,0.1*alam) ; 
            nfix ++ ; 
        }
    } while(true) ; 
}
//****************************************************************************

//****************************************************************************
template< size_t ND, typename FT, typename DFT >
void inline GRACE_HOST_DEVICE
rootfind_nd_newton_raphson(FT&& func, DFT&& dfunc, double (&x)[ND], unsigned long maxiter, double t, int& err)
{
    int iter = 0 ; 
    static constexpr double macheps = std::numeric_limits<double>::epsilon() ; 
    double dx[ND], J[ND][ND], F[ND], g[ND], xold[ND] ; 
    int piv[ND+1] ; 
    double tol ; 
    /* f -> 1/2 F^i F_j */
    auto const fmin = [&] (double (&xL)[ND]) {
        func(xL,F) ; 
        double sum = 0 ;
        for (int i=0; i<ND; ++i) sum += SQR(F[i]);
        return 0.5 * sum ; 
    } ; 

    double f = fmin(x) ; 
    double test = 0.0 ; 
    double xmax = 0; 
    for ( int i=0; i<ND; ++i ) {
      test = fmax(test, fabs(F[i])) ; 
      xmax = fmax(xmax, fabs(x[i])) ; 
    }
    
    tol = 2.0 * macheps * xmax + t;
    if ( test < tol ) {
      err = SUCCESS ; 
      return ; 
    }
    double sum=0. ; 
    for( int i=0; i<ND; ++i ) sum += SQR(x[i]) ; 
    double stpmax = 100 * fmax(sqrt(sum), static_cast<double>(ND)) ; 

    do {
        dfunc(x, F, J) ;
        for( int i=0; i<ND; ++i) {
            sum = 0 ; 
            // compute grad f
            for( int j=0; j<ND; ++j) {
                sum += F[j] * J[j][i] ; 
            }
            g[i] = sum ; 
        }
        for( int i=0; i<ND; ++i) {
            xold[i] = x[i] ; 
        }
        double fold = f ; 
        for( int i=0; i<ND; ++i ) dx[i] = -F[i] ;
        LUPDecompose<ND>(J,1e-15,piv) ; 
        LUPSolve<ND>(J,piv,dx) ; 
        int check ; 
        // this function fills x, f, and F
        lnsrch(fmin,xold,fold,g,dx,x,&f,stpmax,&check);
        if ( check == ERR_ROUNDOFF ) {
            err = ERR_ROUNDOFF ; 
            return ; 
        } else if ( check == ERR_SMALLSTEP ) {
            test = 0. ; 
            double den = fmax(f,0.5*static_cast<double>(ND));
            for( int i=0; i<ND; ++i)
                test = fmax(test, fabs(g[i])*fmax(fabs(x[i]),1.0));
            tol = 2.0 * macheps * den + 100 * t;
            if ( test < tol ) {
                err = SUCCESS ; 
                return ; 
            } else {
                err = ERR_SMALLSTEP ; 
                return ; 
            }
        }
        // test for convergence 
        test = 0. ; 
        double ftest = 0. ; 
        double scale = 0 ;
        for( int i=0; i<ND; ++i){
            test = fmax(test, fabs(x[i]-xold[i])) ; 
            scale = fmax(scale, fabs(x[i])) ; 
            ftest = fmax(ftest, fabs(F[i])) ; 
        }
        tol = 2.0 * macheps * scale + t;
        if ( test < tol || ftest == 0.0 ) {
            err = SUCCESS ; 
            return ; 
        }
        iter++ ; 
    } while( iter<maxiter );
    err = ERR_STAGNATION ; 
    return ; 
}
}

#endif /* GRACE_UTILS_ROOTFINDING_HH */