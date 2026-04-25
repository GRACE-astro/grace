#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace/utils/hermite_interp.hh>

#include <array>
#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kTightTol = 1e-12;
constexpr double kLooseTol = 1e-9;

// Helper: smooth test function and its first three derivatives.
struct analytic_f {
    // f(x) = sin(2x)
    static double f  (double x) { return std::sin(2.0 * x); }
    static double fp (double x) { return 2.0 * std::cos(2.0 * x); }
    static double fpp(double x) { return -4.0 * std::sin(2.0 * x); }
};

}  // namespace

//---------------------------------------------------------------------
// Cubic Hermite (NDerivs = 2): reproduces any cubic exactly.
//---------------------------------------------------------------------
TEST_CASE("Cubic Hermite reproduces an arbitrary cubic exactly",
          "[hermite_interp][cubic]") {
    // p(x) = 2 + 3x - 1.5 x^2 + 0.5 x^3
    auto f  = [](double x) { return 2.0 + 3.0 * x - 1.5 * x * x + 0.5 * x * x * x; };
    auto fp = [](double x) { return 3.0 - 3.0 * x + 1.5 * x * x; };

    double const x_a = -1.0;
    double const x_b =  2.3;

    utils::hermite_interp_1d_t<2> H(x_a, x_b,
                                    {f(x_a), fp(x_a)},
                                    {f(x_b), fp(x_b)});

    for (double x : {x_a, -0.3, 0.0, 0.7, 1.9, x_b}) {
        REQUIRE_THAT(H.evaluate(x), WithinAbs(f(x), kTightTol));
        REQUIRE_THAT(H.template evaluate_derivative<1>(x),
                     WithinAbs(fp(x), kTightTol));
    }
}

//---------------------------------------------------------------------
// Quintic Hermite (NDerivs = 3): reproduces any quintic exactly.
//---------------------------------------------------------------------
TEST_CASE("Quintic Hermite reproduces an arbitrary quintic exactly",
          "[hermite_interp][quintic]") {
    // p(x) = 1 - 0.7x + 1.2 x^2 - 0.5 x^3 + 0.2 x^4 - 0.1 x^5
    auto f   = [](double x) { return 1.0 - 0.7*x + 1.2*x*x - 0.5*x*x*x + 0.2*x*x*x*x - 0.1*x*x*x*x*x; };
    auto fp  = [](double x) { return -0.7 + 2.4*x - 1.5*x*x + 0.8*x*x*x - 0.5*x*x*x*x; };
    auto fpp = [](double x) { return 2.4 - 3.0*x + 2.4*x*x - 2.0*x*x*x; };

    double const x_a = -0.5;
    double const x_b =  1.5;

    utils::hermite_interp_1d_t<3> H(x_a, x_b,
                                    {f(x_a), fp(x_a), fpp(x_a)},
                                    {f(x_b), fp(x_b), fpp(x_b)});

    for (double x : {x_a, -0.2, 0.0, 0.6, 1.1, x_b}) {
        REQUIRE_THAT(H.evaluate(x),
                     WithinAbs(f(x), kTightTol));
        REQUIRE_THAT(H.template evaluate_derivative<1>(x),
                     WithinAbs(fp(x), kTightTol));
        REQUIRE_THAT(H.template evaluate_derivative<2>(x),
                     WithinAbs(fpp(x), kTightTol));
    }
}

//---------------------------------------------------------------------
// Both variants match values and derivatives at the knots by construction.
//---------------------------------------------------------------------
TEST_CASE("Hermite interpolant matches the input data at the knots",
          "[hermite_interp]") {
    double const x_a = 0.0;
    double const x_b = 1.0;

    SECTION("cubic") {
        std::array<double, 2> fa = { 7.0, -1.3};
        std::array<double, 2> fb = {-2.1,  3.9};
        utils::hermite_interp_1d_t<2> H(x_a, x_b, fa, fb);

        REQUIRE_THAT(H.evaluate(x_a),                       WithinAbs(fa[0], kTightTol));
        REQUIRE_THAT(H.evaluate(x_b),                       WithinAbs(fb[0], kTightTol));
        REQUIRE_THAT(H.template evaluate_derivative<1>(x_a),WithinAbs(fa[1], kTightTol));
        REQUIRE_THAT(H.template evaluate_derivative<1>(x_b),WithinAbs(fb[1], kTightTol));
    }

    SECTION("quintic") {
        std::array<double, 3> fa = { 2.0,  0.5, -1.1};
        std::array<double, 3> fb = {-1.4, -0.7,  2.3};
        utils::hermite_interp_1d_t<3> H(x_a, x_b, fa, fb);

        REQUIRE_THAT(H.evaluate(x_a),                       WithinAbs(fa[0], kTightTol));
        REQUIRE_THAT(H.evaluate(x_b),                       WithinAbs(fb[0], kTightTol));
        REQUIRE_THAT(H.template evaluate_derivative<1>(x_a),WithinAbs(fa[1], kTightTol));
        REQUIRE_THAT(H.template evaluate_derivative<1>(x_b),WithinAbs(fb[1], kTightTol));
        REQUIRE_THAT(H.template evaluate_derivative<2>(x_a),WithinAbs(fa[2], kTightTol));
        REQUIRE_THAT(H.template evaluate_derivative<2>(x_b),WithinAbs(fb[2], kTightTol));
    }
}

//---------------------------------------------------------------------
// Convergence on a smooth function: error on sin(2x) should scale
// as O(dx^4) for cubic and O(dx^6) for quintic Hermite.
//---------------------------------------------------------------------
TEST_CASE("Hermite convergence rates on a smooth function",
          "[hermite_interp][convergence]") {
    auto max_error = [](int n_derivs_per_knot, double dx) {
        double const x_a = 0.2;
        double const x_b = x_a + dx;
        // Build the interpolant with the analytic derivatives.
        double max_err = 0.0;
        if (n_derivs_per_knot == 2) {
            utils::hermite_interp_1d_t<2> H(
                x_a, x_b,
                {analytic_f::f(x_a), analytic_f::fp(x_a)},
                {analytic_f::f(x_b), analytic_f::fp(x_b)});
            // Sample the interior of the cell.
            for (int i = 1; i < 20; ++i) {
                double const x = x_a + (i / 20.0) * dx;
                double const e = std::abs(H.evaluate(x) - analytic_f::f(x));
                if (e > max_err) max_err = e;
            }
        } else {
            utils::hermite_interp_1d_t<3> H(
                x_a, x_b,
                {analytic_f::f(x_a), analytic_f::fp(x_a), analytic_f::fpp(x_a)},
                {analytic_f::f(x_b), analytic_f::fp(x_b), analytic_f::fpp(x_b)});
            for (int i = 1; i < 20; ++i) {
                double const x = x_a + (i / 20.0) * dx;
                double const e = std::abs(H.evaluate(x) - analytic_f::f(x));
                if (e > max_err) max_err = e;
            }
        }
        return max_err;
    };

    SECTION("cubic: error ~ dx^4 as dx -> 0") {
        double const e_coarse = max_error(2, 0.4);
        double const e_fine   = max_error(2, 0.2);
        // Halving dx should cut error by ~16x for O(dx^4).
        double const ratio = e_coarse / e_fine;
        REQUIRE(ratio > 12.0);
        REQUIRE(ratio < 20.0);
    }

    SECTION("quintic: error ~ dx^6 as dx -> 0") {
        double const e_coarse = max_error(3, 0.4);
        double const e_fine   = max_error(3, 0.2);
        // Halving dx should cut error by ~64x for O(dx^6).
        double const ratio = e_coarse / e_fine;
        REQUIRE(ratio > 50.0);
        REQUIRE(ratio < 80.0);
    }
}

//---------------------------------------------------------------------
// Integrate-over-cell returns the exact integral of the interpolating
// polynomial, which for a cubic is 4th-order accurate; test against the
// integral of sin(2x) on a small cell.
//---------------------------------------------------------------------
TEST_CASE("integral_over_cell matches analytic integral on cubic polynomial",
          "[hermite_interp]") {
    // Use a cubic polynomial so the cubic Hermite integrates exactly.
    // p(x) = 2 - x + 0.5 x^3, P(x) = 2x - x^2/2 + x^4/8.
    auto f  = [](double x) { return 2.0 - x + 0.5 * x * x * x; };
    auto fp = [](double x) { return -1.0 + 1.5 * x * x; };
    auto F  = [](double x) { return 2.0 * x - 0.5 * x * x + 0.125 * x * x * x * x; };

    double const x_a = -1.0;
    double const x_b =  1.5;
    double const expected = F(x_b) - F(x_a);

    utils::hermite_interp_1d_t<2> H(x_a, x_b,
                                    {f(x_a), fp(x_a)},
                                    {f(x_b), fp(x_b)});
    REQUIRE_THAT(H.integral_over_cell(), WithinAbs(expected, kTightTol));
}

//---------------------------------------------------------------------
// Runtime-order derivative matches compile-time-order derivative.
//---------------------------------------------------------------------
TEST_CASE("Runtime and compile-time derivative evaluators agree",
          "[hermite_interp]") {
    // Quintic on a known polynomial.
    auto f   = [](double x) { return 1.0 - 0.7*x + 1.2*x*x - 0.5*x*x*x + 0.2*x*x*x*x - 0.1*x*x*x*x*x; };
    auto fp  = [](double x) { return -0.7 + 2.4*x - 1.5*x*x + 0.8*x*x*x - 0.5*x*x*x*x; };
    auto fpp = [](double x) { return 2.4 - 3.0*x + 2.4*x*x - 2.0*x*x*x; };

    utils::hermite_interp_1d_t<3> H(-0.5, 1.5,
                                    {f(-0.5), fp(-0.5), fpp(-0.5)},
                                    {f(1.5),  fp(1.5),  fpp(1.5)});

    for (double x : {-0.3, 0.0, 0.5, 1.2}) {
        REQUIRE_THAT(H.template evaluate_derivative<0>(x),
                     WithinAbs(H.evaluate_derivative_rt(x, 0), kLooseTol));
        REQUIRE_THAT(H.template evaluate_derivative<1>(x),
                     WithinAbs(H.evaluate_derivative_rt(x, 1), kLooseTol));
        REQUIRE_THAT(H.template evaluate_derivative<2>(x),
                     WithinAbs(H.evaluate_derivative_rt(x, 2), kLooseTol));
        REQUIRE_THAT(H.template evaluate_derivative<3>(x),
                     WithinAbs(H.evaluate_derivative_rt(x, 3), kLooseTol));
    }
}
