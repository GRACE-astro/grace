/**
 * @file hermite_interp.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 1D Hermite interpolation on a single cell [x_a, x_b], templated
 *        on the number of derivatives carried per knot. Constructs the
 *        cell-local polynomial once (as a utils::polynomial_t in the
 *        normalized coordinate s = (x - x_a)/dx) and provides evaluate,
 *        evaluate_derivative, and cell-integral operations.
 *
 *        NDerivsPerKnot == 2  ->  cubic Hermite (C^1 across knots),
 *                                 polynomial degree 3.
 *        NDerivsPerKnot == 3  ->  quintic Hermite (C^2 across knots),
 *                                 polynomial degree 5.  This is the
 *                                 variant used by the Timmes/Helmholtz
 *                                 EOS biquintic-Hermite interpolant.
 *
 *        All operations are GRACE_HOST_DEVICE for Kokkos compatibility.
 *        The polynomial-in-s lives in the object, so construction costs
 *        a handful of mul/adds; queries cost one Horner evaluation per
 *        derivative order requested, with the polynomial differentiated
 *        at compile time.
 *
 * @date 2026-04-24
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Volume methods
 * to simulate relativistic spacetimes and plasmas
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
 */

#ifndef GRACE_UTILS_HERMITE_INTERP_HH
#define GRACE_UTILS_HERMITE_INTERP_HH

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/utils/hermite_polynomial.hh>

#include <array>

namespace utils {

namespace detail {

/**
 * @brief Compile-time n-th derivative of a polynomial.
 *
 * polynomial_t::derivative() returns a polynomial of degree one less;
 * this template recurses at compile time to apply it N times.
 */
template <int N, int Degree>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
constexpr auto nth_derivative(polynomial_t<Degree> const& p) {
    if constexpr (N == 0) {
        return p;
    } else if constexpr (Degree == 0) {
        // Further differentiation of a constant is the zero constant.
        return polynomial_t<0>{};
    } else {
        return nth_derivative<N - 1>(p.derivative());
    }
}

}  // namespace detail

/**
 * @brief 1D Hermite interpolator on a single cell.
 *
 * Given the function value and NDerivsPerKnot-1 derivatives at each of
 * two endpoints x_a and x_b, this constructs the unique polynomial of
 * degree 2*NDerivsPerKnot - 1 that matches the specified values and
 * derivatives at both knots, and provides evaluation (and derivative
 * evaluation) at arbitrary query points within (or outside) the cell.
 *
 * Derivatives supplied by the user are in the physical coordinate x;
 * internally the polynomial is stored in the normalized coordinate
 * s = (x - x_a) / dx, so that construction doesn't depend on dx except
 * through a few scalings. This also keeps the interior arithmetic in
 * O(1)-magnitude ranges, which is numerically friendlier.
 *
 * @tparam NDerivsPerKnot 2 (cubic) or 3 (quintic). Other values are
 *                       rejected at compile time; extending is a matter
 *                       of adding another `if constexpr` branch in the
 *                       construction below.
 */
template <int NDerivsPerKnot>
class hermite_interp_1d_t {
    static_assert(NDerivsPerKnot == 2 || NDerivsPerKnot == 3,
                  "hermite_interp_1d_t currently supports NDerivsPerKnot "
                  "in {2, 3} (cubic and quintic Hermite).");

 public:
    static constexpr int n_derivs_per_knot = NDerivsPerKnot;
    static constexpr int polynomial_degree = 2 * NDerivsPerKnot - 1;

    using knot_data_t  = std::array<double, NDerivsPerKnot>;
    using polynomial_type = polynomial_t<polynomial_degree>;

 private:
    double           _x_a;     //!< Left knot position.
    double           _dx;      //!< Cell width x_b - x_a (must be non-zero).
    polynomial_type  _p_s;     //!< Polynomial in s = (x - x_a)/dx.

 public:
    /**
     * @brief Construct the interpolant.
     *
     * @param x_a  Left knot position.
     * @param x_b  Right knot position (must differ from x_a).
     * @param f_at_a  {f(x_a), f'(x_a), ..., f^{(N-1)}(x_a)} in physical x.
     * @param f_at_b  {f(x_b), f'(x_b), ..., f^{(N-1)}(x_b)} in physical x.
     */
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    hermite_interp_1d_t(double x_a, double x_b,
                        knot_data_t const& f_at_a,
                        knot_data_t const& f_at_b)
        : _x_a(x_a), _dx(x_b - x_a), _p_s{} {
        // Change-of-variables: derivatives in s are x-derivatives scaled
        // by dx^order:  d^k f/ds^k = dx^k * d^k f/dx^k.
        if constexpr (NDerivsPerKnot == 2) {
            // Cubic Hermite: 4 coefficients from 4 constraints.
            // Using the normalized polynomial p(s) with
            //   p(0)=fa, p'(0)=dx*fa', p(1)=fb, p'(1)=dx*fb'.
            double const fa  = f_at_a[0];
            double const fpa = f_at_a[1];
            double const fb  = f_at_b[0];
            double const fpb = f_at_b[1];

            double const d_fa = _dx * fpa;   // p'(0)
            double const d_fb = _dx * fpb;   // p'(1)

            double const A = fb - fa - d_fa;              // p(1) - p(0) - p'(0)
            double const B = d_fb - d_fa;                 // p'(1) - p'(0)

            _p_s.coeffs[0] = fa;
            _p_s.coeffs[1] = d_fa;
            _p_s.coeffs[2] = 3.0 * A - B;
            _p_s.coeffs[3] = B - 2.0 * A;
        } else {  // NDerivsPerKnot == 3: quintic Hermite.
            // 6 coefficients from 6 constraints:
            //   p(0)=fa, p'(0)=dx*fa', p''(0)=dx^2*fa'',
            //   p(1)=fb, p'(1)=dx*fb', p''(1)=dx^2*fb''.
            double const fa   = f_at_a[0];
            double const fpa  = f_at_a[1];
            double const fppa = f_at_a[2];
            double const fb   = f_at_b[0];
            double const fpb  = f_at_b[1];
            double const fppb = f_at_b[2];

            double const dxsq   = _dx * _dx;
            double const d_fa   = _dx * fpa;       // p'(0)
            double const d_fb   = _dx * fpb;       // p'(1)
            double const dd_fa  = dxsq * fppa;     // p''(0)
            double const dd_fb  = dxsq * fppb;     // p''(1)

            // Closed-form Cramer solve of the 3x3 system for c3, c4, c5.
            double const A = fb  - fa  - d_fa - 0.5 * dd_fa;
            double const B = d_fb - d_fa - dd_fa;
            double const C = dd_fb - dd_fa;

            _p_s.coeffs[0] = fa;
            _p_s.coeffs[1] = d_fa;
            _p_s.coeffs[2] = 0.5 * dd_fa;
            _p_s.coeffs[3] =  10.0 * A - 4.0 * B + 0.5 * C;
            _p_s.coeffs[4] = -15.0 * A + 7.0 * B -       C;
            _p_s.coeffs[5] =   6.0 * A - 3.0 * B + 0.5 * C;
        }
    }

    //-----------------------------------------------------------------
    // Query.
    //-----------------------------------------------------------------

    /// Evaluate the interpolant at physical x.
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double evaluate(double x) const {
        double const s = (x - _x_a) / _dx;
        return _p_s.evaluate(s);
    }

    /**
     * @brief Evaluate the Order-th derivative of the interpolant at x.
     *
     * Chain rule: d^n f / dx^n = (1/dx)^n * d^n p / ds^n, evaluated at
     * s = (x - x_a)/dx. The derivative polynomial is built at compile
     * time from the stored normalized polynomial.
     *
     * Order must be non-negative; Order > polynomial_degree yields the
     * identically-zero polynomial (correct mathematically and cheap to
     * evaluate).
     */
    template <int Order>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double evaluate_derivative(double x) const {
        static_assert(Order >= 0, "Derivative order must be non-negative.");
        auto const dp_s = detail::nth_derivative<Order>(_p_s);
        double const s = (x - _x_a) / _dx;
        double dx_pow = 1.0;
        for (int i = 0; i < Order; ++i) dx_pow *= _dx;
        return dp_s.evaluate(s) / dx_pow;
    }

    /// Runtime-order version, for when Order isn't known at compile time.
    /// Restricted to [0, polynomial_degree] to avoid an open-ended runtime
    /// recursion; out-of-range returns 0 (mathematically correct).
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double evaluate_derivative_rt(double x, int order) const {
        if (order < 0 || order > polynomial_degree) return 0.0;
        // Build the nth-derivative polynomial by hand at runtime.
        // The coefficient of s^k in the order-th derivative is
        //   (k+order)! / k! * c_{k+order}.
        double const s = (x - _x_a) / _dx;
        double dx_pow = 1.0;
        for (int i = 0; i < order; ++i) dx_pow *= _dx;

        double acc = 0.0;
        // Compute via Horner on the transformed coefficients.
        for (int k = polynomial_degree - order; k >= 0; --k) {
            // Multiply c_{k+order} by (k+order)*(k+order-1)*...*(k+1)
            double factor = 1.0;
            for (int m = 0; m < order; ++m) factor *= static_cast<double>(k + order - m);
            acc = acc * s + factor * _p_s.coeffs[k + order];
        }
        return acc / dx_pow;
    }

    /**
     * @brief Definite integral of the interpolant over [x_a, x_b].
     *
     * Using the integral of p(s) over [0, 1], then scaling by dx.
     */
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double integral_over_cell() const {
        auto const I = _p_s.integral();
        return _dx * (I.evaluate(1.0) - I.evaluate(0.0));
    }

    //-----------------------------------------------------------------
    // Accessors.
    //-----------------------------------------------------------------

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double x_a() const { return _x_a; }
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double x_b() const { return _x_a + _dx; }
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE double dx()  const { return _dx; }

    /// The underlying polynomial in the normalized coordinate s.
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    polynomial_type const& polynomial_in_s() const { return _p_s; }
};

//---------------------------------------------------------------------
// Convenience aliases.
//---------------------------------------------------------------------

using cubic_hermite_interp_1d_t   = hermite_interp_1d_t<2>;
using quintic_hermite_interp_1d_t = hermite_interp_1d_t<3>;

}  // namespace utils

#endif  // GRACE_UTILS_HERMITE_INTERP_HH
