/**
 * @file hermite_polynomial.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Compile-time-templated 1D polynomial type with arithmetic,
 *        evaluation (Horner), symbolic differentiation, and integration.
 *        Storage is std::array so all operations are host/device-friendly
 *        and trivially copyable for Kokkos.
 *
 *        This is the algebraic foundation for the Hermite interpolation
 *        layer (include/grace/utils/hermite_interp.hh) and for any other
 *        code that wants cheap symbolic polynomial arithmetic -- in
 *        particular, differentiating an interpolant to evaluate its
 *        derivatives at arbitrary points (e.g. for ADM-mass integrals).
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

#ifndef GRACE_UTILS_HERMITE_POLYNOMIAL_HH
#define GRACE_UTILS_HERMITE_POLYNOMIAL_HH

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <array>
#include <cstddef>

namespace utils {

/**
 * @brief Compile-time-degree 1D polynomial over doubles.
 *
 * Coefficients are stored in ascending-power order:
 *     p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[Degree]*x^Degree
 *
 * The default-constructed polynomial is identically zero. The template
 * parameter Degree must be non-negative; there are Degree+1 coefficients.
 *
 * All operations (evaluate, differentiate, integrate, add, multiply) are
 * GRACE_HOST_DEVICE-callable and constexpr where the language allows,
 * so instances can be used inside Kokkos kernels and as compile-time
 * building blocks for higher-order Hermite-interpolation machinery.
 *
 * @tparam Degree Polynomial degree. Must be >= 0.
 */
template <int Degree>
struct polynomial_t {
    static_assert(Degree >= 0, "Polynomial degree must be non-negative.");

    static constexpr int degree     = Degree;
    static constexpr int num_coeffs = Degree + 1;

    std::array<double, num_coeffs> coeffs;

    /// Default construction yields the zero polynomial.
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    constexpr polynomial_t() : coeffs{} {}

    /// Construct from an explicit coefficient array (ascending-power order).
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    constexpr explicit polynomial_t(std::array<double, num_coeffs> const& c)
        : coeffs(c) {}

    /// Coefficient read access.
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    constexpr double operator[](int i) const { return coeffs[i]; }

    /// Coefficient write access.
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    constexpr double& operator[](int i) { return coeffs[i]; }

    /**
     * @brief Evaluate p(x) via Horner's method.
     *
     * Requires Degree+1 multiply-adds. Numerically stable and the
     * fastest general-purpose scheme for low-to-moderate degree.
     */
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    constexpr double evaluate(double x) const {
        double acc = coeffs[Degree];
        for (int i = Degree - 1; i >= 0; --i) {
            acc = acc * x + coeffs[i];
        }
        return acc;
    }

    /**
     * @brief Analytic derivative as a polynomial of degree Degree-1.
     *
     * For Degree == 0 the derivative is identically zero; to keep the
     * return type a valid polynomial the result is a degree-0 zero
     * polynomial rather than a degree-(-1) object.
     */
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    constexpr polynomial_t<(Degree > 0 ? Degree - 1 : 0)> derivative() const {
        if constexpr (Degree == 0) {
            return polynomial_t<0>{};
        } else {
            polynomial_t<Degree - 1> d{};
            for (int i = 1; i <= Degree; ++i) {
                d.coeffs[i - 1] = static_cast<double>(i) * coeffs[i];
            }
            return d;
        }
    }

    /**
     * @brief Analytic indefinite integral with integration constant 0.
     *
     * Returns a polynomial of degree Degree+1 with I(0) == 0.
     */
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    constexpr polynomial_t<Degree + 1> integral() const {
        polynomial_t<Degree + 1> I{};
        I.coeffs[0] = 0.0;
        for (int i = 0; i <= Degree; ++i) {
            I.coeffs[i + 1] = coeffs[i] / static_cast<double>(i + 1);
        }
        return I;
    }
};

//---------------------------------------------------------------------
// Free-function arithmetic.
//---------------------------------------------------------------------

/// Same-degree addition.
template <int Degree>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
constexpr polynomial_t<Degree>
operator+(polynomial_t<Degree> const& a, polynomial_t<Degree> const& b) {
    polynomial_t<Degree> r;
    for (int i = 0; i <= Degree; ++i) r.coeffs[i] = a.coeffs[i] + b.coeffs[i];
    return r;
}

/// Same-degree subtraction.
template <int Degree>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
constexpr polynomial_t<Degree>
operator-(polynomial_t<Degree> const& a, polynomial_t<Degree> const& b) {
    polynomial_t<Degree> r;
    for (int i = 0; i <= Degree; ++i) r.coeffs[i] = a.coeffs[i] - b.coeffs[i];
    return r;
}

/// Unary negation.
template <int Degree>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
constexpr polynomial_t<Degree>
operator-(polynomial_t<Degree> const& a) {
    polynomial_t<Degree> r;
    for (int i = 0; i <= Degree; ++i) r.coeffs[i] = -a.coeffs[i];
    return r;
}

/// Scalar multiplication (scalar on the left).
template <int Degree>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
constexpr polynomial_t<Degree>
operator*(double s, polynomial_t<Degree> const& a) {
    polynomial_t<Degree> r;
    for (int i = 0; i <= Degree; ++i) r.coeffs[i] = s * a.coeffs[i];
    return r;
}

/// Scalar multiplication (scalar on the right).
template <int Degree>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
constexpr polynomial_t<Degree>
operator*(polynomial_t<Degree> const& a, double s) {
    return s * a;
}

/**
 * @brief Polynomial-by-polynomial product.
 *
 * Result degree is the sum of operand degrees (this is the whole point
 * of templating Degree: the output type is known at compile time).
 */
template <int A, int B>
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
constexpr polynomial_t<A + B>
operator*(polynomial_t<A> const& a, polynomial_t<B> const& b) {
    polynomial_t<A + B> r{};
    for (int i = 0; i <= A; ++i) {
        for (int j = 0; j <= B; ++j) {
            r.coeffs[i + j] += a.coeffs[i] * b.coeffs[j];
        }
    }
    return r;
}

}  // namespace utils

#endif  // GRACE_UTILS_HERMITE_POLYNOMIAL_HH
