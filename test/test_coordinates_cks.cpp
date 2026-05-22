/**
 * @file test_coordinates_cks.cpp
 * @brief Sanity tests for the Cartesian Kerr-Schild branch of the coordinate
 *        system (`coordinate_system.is_kerr_schild = true`).
 *
 * The standard Cartesian-coordinate tests in `test_coordinates.cpp` cover
 * the spin-independent code path.  The methods that *do* depend on the
 * spin are:
 *
 *   - `cart_to_sph(x, y, z)`  — CKS Cartesian → KS spherical (r, θ, φ_KS).
 *   - `sph_to_cart(r, θ, φ)`  — KS spherical → CKS Cartesian.
 *
 * Both functions use the **KS spherical** azimuth convention, defined by
 *     x + iy = (r + i·a) · sin(θ) · e^{iφ_KS},     z = r · cos(θ).
 * Consequently the two are exact algebraic inverses: `sph_to_cart(cart_to_sph(p)) ≡ p`.
 *
 * Initial-data routines that need **Boyer-Lindquist** φ should implement
 * the BL conversion locally (the proper shift is `∫(a/Δ) dr`, not
 * `a·r/Δ`).  The plain `cart_to_sph` API stays clean and inverse-consistent.
 *
 * The Kerr-Schild r is defined implicitly via
 *     r⁴ − (R² − a²)·r² − a²·z² = 0,    R² = x² + y² + z² ,
 * so for a≠0 the spherical-radius scalar `rad = sqrt(R²)` is NOT equal
 * to `r`.  The polar angle satisfies cos(θ) = z/r (note: z/r, NOT z/R).
 *
 * Bound to `configs/basic_config_cks.yaml` — same grid as `basic_config`
 * plus `coordinate_system: { is_kerr_schild: true, bh_spin: 0.5 }`.
 *
 * Each SECTION is multi-rank safe: every rank holds the same coord-system
 * singleton, all numeric checks use only the singleton, no MPI reductions.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>

#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>

#include <array>
#include <cmath>

namespace {
// Helper: call the device coord-system on a single (x,y,z) point via a
// 1-element parallel_for so we can exercise the device CKS path even if
// the build's default execution space is GPU.
struct cks_xyz_to_rtp {
    grace::cartesian_device_coordinate_system_impl_t dcs ;
    Kokkos::View<double[3]> in_xyz ;
    Kokkos::View<double[3]> out_rtp ;
    KOKKOS_FUNCTION
    void operator()(int) const {
        double xyz[3] = {in_xyz(0), in_xyz(1), in_xyz(2)} ;
        double rtp[3] ;
        dcs.cart_to_sph(xyz, rtp) ;
        out_rtp(0) = rtp[0] ;
        out_rtp(1) = rtp[1] ;
        out_rtp(2) = rtp[2] ;
    }
} ;
struct cks_rtp_to_xyz {
    grace::cartesian_device_coordinate_system_impl_t dcs ;
    Kokkos::View<double[3]> in_rtp ;
    Kokkos::View<double[3]> out_xyz ;
    KOKKOS_FUNCTION
    void operator()(int) const {
        double rtp[3] = {in_rtp(0), in_rtp(1), in_rtp(2)} ;
        double xyz[3] ;
        dcs.sph_to_cart(rtp, xyz) ;
        out_xyz(0) = xyz[0] ;
        out_xyz(1) = xyz[1] ;
        out_xyz(2) = xyz[2] ;
    }
} ;
} // namespace

TEST_CASE("cartesian Kerr-Schild coordinates", "[coords][cks]")
{
    using namespace grace ;
    using namespace Kokkos ;

    auto& cs  = coordinate_system::get() ;
    auto dcs  = cs.get_device_coord_system() ;

    SECTION("coord-system advertises CKS + the requested spin")
    {
        REQUIRE( cs.get_is_cks() == true ) ;
        REQUIRE_THAT( cs.get_bh_spin(),
                      Catch::Matchers::WithinRel(0.5, 1e-15) ) ;
        REQUIRE( dcs.is_cks == true ) ;
        REQUIRE( dcs.bh_spin == cs.get_bh_spin() ) ;
    }

    SECTION("Kerr-Schild r reduces to ρ = sqrt(x²+y²+z²) on the z-axis")
    {
        // Along the symmetry axis (x=y=0), the Kerr-Schild r equation
        //     r⁴ − (z² − a²)·r² − a²·z² = 0
        // factors as (r² − z²)(r² + a²) = 0, so r = |z| irrespective of a.
        double xyz[3] = {0.0, 0.0, 5.0} ;
        double rtp[3] ;
        dcs.cart_to_sph(xyz, rtp) ;
        REQUIRE_THAT( rtp[0], Catch::Matchers::WithinAbs(5.0, 1e-12) ) ;
        REQUIRE_THAT( rtp[1], Catch::Matchers::WithinAbs(0.0, 1e-12) ) ;
    }

    SECTION("Kerr-Schild r in the equatorial plane: r² = R² − a²")
    {
        // z = 0  →  r⁴ − (R²−a²)·r² = 0  →  r² = R² − a² (the non-trivial root),
        // provided R² > a².  Pick R² = 10·a² to stay well-defined.
        double const a = cs.get_bh_spin() ;
        double const R = std::sqrt(10.0) * a ;
        double xyz[3] = {R, 0.0, 0.0} ;
        double rtp[3] ;
        dcs.cart_to_sph(xyz, rtp) ;
        double const r_expected = std::sqrt(R*R - a*a) ;
        REQUIRE_THAT( rtp[0], Catch::Matchers::WithinRel(r_expected, 1e-12) ) ;
        REQUIRE_THAT( rtp[1], Catch::Matchers::WithinRel(M_PI/2, 1e-12) ) ;
    }

    SECTION("cos(θ) = z/r (not z/R) — distinct from flat-space spherical")
    {
        // Point off-axis where R ≠ r so the distinction matters.
        double const a = cs.get_bh_spin() ;
        double xyz[3] = {3.0, 0.0, 4.0} ;
        double rtp[3] ;
        dcs.cart_to_sph(xyz, rtp) ;
        REQUIRE_THAT( std::cos(rtp[1]),
                      Catch::Matchers::WithinAbs(xyz[2] / rtp[0], 1e-12) ) ;
        // Sanity: R = 5, so for a = 0.5 we have r² = (25 − 0.25)/2 + sqrt(((25 − 0.25)/2)² + 0.25·16)
        // → r² ≈ 12.46 → r ≈ 3.53.  rtp[0] should be in that ballpark.
        double const R2 = 9.0 + 16.0 ;
        double const r2_expected =
              (R2 - a*a)/2
            + std::sqrt( (R2 - a*a)*(R2 - a*a)/4 + a*a*xyz[2]*xyz[2] ) ;
        REQUIRE_THAT( rtp[0]*rtp[0],
                      Catch::Matchers::WithinRel(r2_expected, 1e-12) ) ;
    }

    SECTION("sph_to_cart ∘ cart_to_sph is the identity (round-trip)")
    {
        // Both functions use KS-spherical azimuth, so the composition is
        // the identity to FP floor for any (x, y, z) with R > a (so the
        // KS r-equation has a real positive root).  Five test points
        // sample various geometric configurations.
        struct point { double x, y, z ; } ;
        std::array<point, 5> pts = {
          point{ 5.0,  0.0, 0.0},   // equatorial, on +x axis
          point{ 3.0, -2.0, 1.5},   // generic off-axis
          point{-4.0,  4.0, 6.0},   // upper hemisphere
          point{ 0.7,  0.3, 0.5},   // small but non-degenerate
          point{12.0,  5.0, 7.0},   // far-field
        } ;
        for (auto const& p : pts) {
            INFO("input  (" << p.x << ", " << p.y << ", " << p.z << ")") ;
            double xyz[3]  = {p.x, p.y, p.z} ;
            double rtp[3] ;
            dcs.cart_to_sph(xyz, rtp) ;
            double xyz2[3] ;
            dcs.sph_to_cart(rtp, xyz2) ;
            INFO("recovered (" << xyz2[0] << ", " << xyz2[1] << ", " << xyz2[2] << ")") ;
            REQUIRE_THAT( xyz2[0], Catch::Matchers::WithinAbs(p.x, 1e-12) ) ;
            REQUIRE_THAT( xyz2[1], Catch::Matchers::WithinAbs(p.y, 1e-12) ) ;
            REQUIRE_THAT( xyz2[2], Catch::Matchers::WithinAbs(p.z, 1e-12) ) ;
        }
    }

    SECTION("sph_to_cart on z-axis: independent of spin, equals (0,0,r·cos θ)")
    {
        // The forward formula  x = (r·cos φ − a·sin φ)·sin θ  vanishes on
        // sin θ = 0 regardless of (a, φ), so points on the z-axis are
        // spin-independent.  A useful structural check on the forward
        // formula that doesn't depend on KS vs BL azimuth conventions.
        double rtp[3] = {5.0, 0.0, 1.23} ;   // θ = 0, arbitrary φ
        double xyz[3] ;
        dcs.sph_to_cart(rtp, xyz) ;
        REQUIRE_THAT( xyz[0], Catch::Matchers::WithinAbs(0.0, 1e-14) ) ;
        REQUIRE_THAT( xyz[1], Catch::Matchers::WithinAbs(0.0, 1e-14) ) ;
        REQUIRE_THAT( xyz[2], Catch::Matchers::WithinAbs(5.0, 1e-14) ) ;
    }

    SECTION("sph_to_cart at equator: x²+y² = (r²+a²)·sin²θ")
    {
        // Modulus of the KS algebraic relation |x+iy|² = (r²+a²)·sin²θ —
        // independent of φ.  Holds at any θ but easiest to check on the
        // equator where sin θ = 1.
        double const a = cs.get_bh_spin() ;
        double const r = 4.0 ;
        for (double phi_test : {0.0, 0.7, 2.1, -1.4}) {
            INFO("phi = " << phi_test) ;
            double rtp[3] = {r, M_PI/2, phi_test} ;
            double xyz[3] ;
            dcs.sph_to_cart(rtp, xyz) ;
            double const rho2 = xyz[0]*xyz[0] + xyz[1]*xyz[1] ;
            REQUIRE_THAT( rho2,
                          Catch::Matchers::WithinRel(r*r + a*a, 1e-12) ) ;
            REQUIRE_THAT( xyz[2], Catch::Matchers::WithinAbs(0.0, 1e-14) ) ;
        }
    }

    SECTION("device kernel call agrees bit-exact with the host coord-system")
    {
        // Same inputs as above, this time routing the call through a
        // Kokkos::parallel_for so the device path is exercised even when
        // the default execution space is GPU.  Result must match the host
        // call to FP floor.
        double const a = cs.get_bh_spin() ;
        (void)a ;
        double const xin[3] = {3.0, -2.0, 1.5} ;

        View<double[3]> in_xyz_d("in_xyz_d") ;
        View<double[3]> out_rtp_d("out_rtp_d") ;
        auto in_xyz_h = create_mirror_view(in_xyz_d) ;
        in_xyz_h(0) = xin[0] ; in_xyz_h(1) = xin[1] ; in_xyz_h(2) = xin[2] ;
        deep_copy(in_xyz_d, in_xyz_h) ;

        parallel_for("cks_cart_to_sph_device", 1,
                     cks_xyz_to_rtp{dcs, in_xyz_d, out_rtp_d}) ;
        Kokkos::fence() ;

        auto out_rtp_h = create_mirror_view(out_rtp_d) ;
        deep_copy(out_rtp_h, out_rtp_d) ;

        double rtp_host[3] ;
        dcs.cart_to_sph(const_cast<double*>(xin), rtp_host) ;
        for (int c = 0 ; c < 3 ; ++c) {
            INFO("component " << c) ;
            REQUIRE_THAT( out_rtp_h(c),
                          Catch::Matchers::WithinAbs(rtp_host[c], 1e-14) ) ;
        }
    }
}
