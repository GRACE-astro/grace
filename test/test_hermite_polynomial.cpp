#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace/utils/hermite_polynomial.hh>

#include <array>
#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kTightTol = 1e-14;
constexpr double kLooseTol = 1e-10;

}  // namespace

TEST_CASE("Default construction yields the zero polynomial", "[hermite_polynomial]") {
    utils::polynomial_t<3> p;
    REQUIRE(p.coeffs[0] == 0.0);
    REQUIRE(p.coeffs[1] == 0.0);
    REQUIRE(p.coeffs[2] == 0.0);
    REQUIRE(p.coeffs[3] == 0.0);

    // Evaluates to zero everywhere.
    REQUIRE_THAT(p.evaluate(0.0),  WithinAbs(0.0, kTightTol));
    REQUIRE_THAT(p.evaluate(1.7),  WithinAbs(0.0, kTightTol));
    REQUIRE_THAT(p.evaluate(-3.2), WithinAbs(0.0, kTightTol));
}

TEST_CASE("Horner evaluation matches a known polynomial", "[hermite_polynomial]") {
    //  p(x) = 2 + 3*x - 1.5*x^2 + 0.5*x^3
    utils::polynomial_t<3> p(std::array<double, 4>{2.0, 3.0, -1.5, 0.5});

    auto reference = [](double x) {
        return 2.0 + 3.0 * x - 1.5 * x * x + 0.5 * x * x * x;
    };

    for (double x : {-2.0, -0.5, 0.0, 0.3, 1.0, 2.7}) {
        REQUIRE_THAT(p.evaluate(x), WithinAbs(reference(x), kTightTol));
    }
}

TEST_CASE("Derivative lowers the degree and scales coefficients correctly",
          "[hermite_polynomial]") {
    //  p(x) = 2 + 3*x - 1.5*x^2 + 0.5*x^3
    //  p'(x) = 3 - 3*x + 1.5*x^2
    utils::polynomial_t<3> p(std::array<double, 4>{2.0, 3.0, -1.5, 0.5});

    auto dp = p.derivative();
    static_assert(decltype(dp)::degree == 2,
                  "Derivative of a cubic is a quadratic");

    REQUIRE_THAT(dp.coeffs[0], WithinAbs( 3.0, kTightTol));
    REQUIRE_THAT(dp.coeffs[1], WithinAbs(-3.0, kTightTol));
    REQUIRE_THAT(dp.coeffs[2], WithinAbs( 1.5, kTightTol));

    // Spot-check values.
    auto d_reference = [](double x) { return 3.0 - 3.0 * x + 1.5 * x * x; };
    for (double x : {-1.4, 0.0, 0.75, 3.1}) {
        REQUIRE_THAT(dp.evaluate(x), WithinAbs(d_reference(x), kTightTol));
    }
}

TEST_CASE("Derivative of a constant is the zero polynomial", "[hermite_polynomial]") {
    utils::polynomial_t<0> p(std::array<double, 1>{4.2});
    auto dp = p.derivative();
    static_assert(decltype(dp)::degree == 0,
                  "Derivative of a degree-0 polynomial stays degree 0 (by API)");
    REQUIRE_THAT(dp.coeffs[0], WithinAbs(0.0, kTightTol));
    REQUIRE_THAT(dp.evaluate(17.3), WithinAbs(0.0, kTightTol));
}

TEST_CASE("Integral raises degree and is the exact inverse of derivative (mod constant)",
          "[hermite_polynomial]") {
    //  p(x) = 3 - 3*x + 1.5*x^2
    //  integral I with I(0)=0:  I(x) = 3*x - 1.5*x^2 + 0.5*x^3
    utils::polynomial_t<2> p(std::array<double, 3>{3.0, -3.0, 1.5});
    auto I = p.integral();
    static_assert(decltype(I)::degree == 3,
                  "Integral of a quadratic is a cubic");

    REQUIRE_THAT(I.coeffs[0], WithinAbs( 0.0, kTightTol));  // I(0) == 0 by construction
    REQUIRE_THAT(I.coeffs[1], WithinAbs( 3.0, kTightTol));
    REQUIRE_THAT(I.coeffs[2], WithinAbs(-1.5, kTightTol));
    REQUIRE_THAT(I.coeffs[3], WithinAbs( 0.5, kTightTol));

    // Differentiating the integral recovers the original polynomial.
    auto dI = I.derivative();
    for (int i = 0; i < 3; ++i) {
        REQUIRE_THAT(dI.coeffs[i], WithinAbs(p.coeffs[i], kTightTol));
    }
}

TEST_CASE("Addition and subtraction are coefficient-wise", "[hermite_polynomial]") {
    utils::polynomial_t<2> a(std::array<double, 3>{1.0,  2.0, 3.0});
    utils::polynomial_t<2> b(std::array<double, 3>{4.0, -1.0, 0.5});

    auto s = a + b;
    REQUIRE_THAT(s.coeffs[0], WithinAbs(5.0, kTightTol));
    REQUIRE_THAT(s.coeffs[1], WithinAbs(1.0, kTightTol));
    REQUIRE_THAT(s.coeffs[2], WithinAbs(3.5, kTightTol));

    auto d = a - b;
    REQUIRE_THAT(d.coeffs[0], WithinAbs(-3.0, kTightTol));
    REQUIRE_THAT(d.coeffs[1], WithinAbs( 3.0, kTightTol));
    REQUIRE_THAT(d.coeffs[2], WithinAbs( 2.5, kTightTol));
}

TEST_CASE("Scalar multiplication distributes over coefficients", "[hermite_polynomial]") {
    utils::polynomial_t<2> p(std::array<double, 3>{1.5, -2.0, 4.0});

    auto q1 = 2.0 * p;
    auto q2 = p * 2.0;
    for (int i = 0; i < 3; ++i) {
        REQUIRE_THAT(q1.coeffs[i], WithinAbs(2.0 * p.coeffs[i], kTightTol));
        REQUIRE_THAT(q2.coeffs[i], WithinAbs(2.0 * p.coeffs[i], kTightTol));
    }
}

TEST_CASE("Polynomial-by-polynomial product has summed degree", "[hermite_polynomial]") {
    //  a(x) = 1 + 2x
    //  b(x) = 3 - x + x^2
    //  a*b  = 3 + 5x + 0*x^2 + 2*x^3
    utils::polynomial_t<1> a(std::array<double, 2>{1.0, 2.0});
    utils::polynomial_t<2> b(std::array<double, 3>{3.0, -1.0, 1.0});

    auto c = a * b;
    static_assert(decltype(c)::degree == 3,
                  "Product degree should be deg(a) + deg(b)");

    REQUIRE_THAT(c.coeffs[0], WithinAbs(3.0, kTightTol));
    REQUIRE_THAT(c.coeffs[1], WithinAbs(5.0, kTightTol));
    REQUIRE_THAT(c.coeffs[2], WithinAbs(0.0, kTightTol));
    REQUIRE_THAT(c.coeffs[3], WithinAbs(2.0, kTightTol));

    // Evaluate at several points and compare to the reference expansion.
    auto ref = [](double x) {
        return (1.0 + 2.0 * x) * (3.0 - x + x * x);
    };
    for (double x : {-1.5, 0.0, 0.4, 2.0}) {
        REQUIRE_THAT(c.evaluate(x), WithinAbs(ref(x), kLooseTol));
    }
}

TEST_CASE("Derivative chain rule holds for a product (via direct expansion)",
          "[hermite_polynomial]") {
    //  a(x) = 1 + 2x              ->  a'(x) = 2
    //  b(x) = 3 - x + x^2         ->  b'(x) = -1 + 2x
    //  (a*b)' = a'*b + a*b'
    utils::polynomial_t<1> a(std::array<double, 2>{1.0, 2.0});
    utils::polynomial_t<2> b(std::array<double, 3>{3.0, -1.0, 1.0});

    auto ab  = a * b;
    auto dab = ab.derivative();

    auto ap = a.derivative();
    auto bp = b.derivative();
    auto rhs = ap * b + a * bp;

    // rhs has degree 2 (max of 0+2 and 1+1); dab has degree 2 too.
    static_assert(decltype(dab)::degree == decltype(rhs)::degree,
                  "Leibniz rule degrees must match");
    for (int i = 0; i <= decltype(dab)::degree; ++i) {
        REQUIRE_THAT(dab.coeffs[i], WithinAbs(rhs.coeffs[i], kLooseTol));
    }
}
