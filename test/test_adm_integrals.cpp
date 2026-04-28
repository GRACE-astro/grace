/**
 * @file test_adm_integrals.cpp
 * @brief Validation of the ADM-mass surface-integral diagnostic against
 *        the analytic single Schwarzschild puncture.
 *
 *        The analytic puncture initial data (puncture_id_t with M=1) sets
 *
 *            psi = 1 + M / (2 r),     gamma_ij = psi^4 delta_ij,    K_ij = 0.
 *
 *        Inserting this into the ADM-mass surface integral
 *
 *            M_ADM(r) = (1/16 pi) oint (d_j gamma_ij - d_i gamma_jj) n^i dA
 *
 *        yields the closed-form result on a sphere of radius r centred on
 *        the puncture:
 *
 *            M_ADM(r) = M * (1 + M/(2 r))^3.
 *
 *        Convergence to M as r -> infinity is the well-known asymptotic
 *        statement; at finite r, the integral has an exact analytic value
 *        that we compare against.
 *
 *        This test is gated on GRACE_ENABLE_Z4C_METRIC because the
 *        diagnostic itself only does anything when Z4c is built in.
 *
 * @date 2026-04-28
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace_config.h>

#ifdef GRACE_ENABLE_Z4C_METRIC

#include <grace/IO/diagnostics/adm_integrals.hh>
#include <grace/IO/spherical_surfaces.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <cmath>
#include <vector>


TEST_CASE("ADM-mass integral on analytic Schwarzschild puncture",
          "[adm_integrals][puncture]")
{
    using namespace grace;

    // The analytic puncture in puncture_id_t hardcodes M = 1.
    constexpr double M_punct = 1.0;

    adm_integrals diag;
    auto const sphere_indices = diag.sphere_indices_view();
    REQUIRE(sphere_indices.size() == 3);

    auto const values = diag.compute();
    REQUIRE(values.size() == sphere_indices.size());

    auto& spheres = spherical_surface_manager::get();

    for (size_t i = 0; i < sphere_indices.size(); ++i) {
        auto const sidx     = sphere_indices[i];
        auto const& detector = spheres.get(sidx);
        double const r       = detector.radius;

        // Analytic finite-r value.
        double const psi      = 1.0 + 0.5 * M_punct / r;
        double const expected = M_punct * psi * psi * psi;

        // Tolerance: the FD truncation error of the in-kernel central FD
        // on chi (= psi^{-2} for GRACE's Z4c convention gamma = gtilde/chi^2)
        // scales as ~ |chi'''| dx^2 / r^k. At r=8 with our smallest dx ~ 1
        // the relative error on the gradient is a few %, plus quadrature
        // error from the 31-point sphere sampling. Use 5% as a generous
        // bound that still rejects an order-of-magnitude regression.
        double const rel_tol = 0.05;
        double const abs_tol = std::abs(expected) * rel_tol;

        INFO("Detector " << detector.name << "  r = " << r
             << "  expected = " << expected
             << "  got = " << values[i]
             << "  abs_tol = " << abs_tol);

        REQUIRE(std::isfinite(values[i]));
        REQUIRE_THAT(values[i],
                     Catch::Matchers::WithinAbs(expected, abs_tol));
    }

    // Sanity: the integral must converge toward M as r grows. Find the
    // smallest- and largest-radius detectors and check ordering.
    double r_min = std::numeric_limits<double>::infinity();
    double r_max = 0.0;
    double M_at_rmin = 0.0;
    double M_at_rmax = 0.0;
    for (size_t i = 0; i < sphere_indices.size(); ++i) {
        auto const& detector = spheres.get(sphere_indices[i]);
        if (detector.radius < r_min) { r_min = detector.radius; M_at_rmin = values[i]; }
        if (detector.radius > r_max) { r_max = detector.radius; M_at_rmax = values[i]; }
    }
    INFO("M_ADM(r=" << r_min << ") = " << M_at_rmin
         << "   M_ADM(r=" << r_max << ") = " << M_at_rmax
         << "   M_punct = " << M_punct);
    // M_ADM(r) is a strictly decreasing function of r (toward M from above).
    REQUIRE(M_at_rmin > M_at_rmax);
    REQUIRE(M_at_rmax > M_punct - 1e-3);
}

#endif  // GRACE_ENABLE_Z4C_METRIC
