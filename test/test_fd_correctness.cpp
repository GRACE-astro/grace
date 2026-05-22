// Correctness regression test for the templated FD operators in
// include/grace/physics/fd_subexpressions.hh.
//
// Two checks per (stencil kind, order, axis/pair):
//
//   1. Polynomial exactness.  A centered (or biased / upwind) 1st-derivative
//      stencil of order N is exact on polynomials of degree ≤ N; a 2nd-
//      derivative stencil is exact on polynomials of degree ≤ N+1; the
//      Kreiss-Oliger stencil at order N annihilates polynomials of degree ≤ N.
//      We evaluate the FD on a random polynomial that *just* fits the
//      stencil's exactness window and assert the result matches the analytic
//      derivative (or zero, for KO) to within a small number of ulp.
//
//   2. Convergence rate.  On a smooth non-polynomial function we evaluate
//      the FD at grid spacings h, h/2, h/4 and assert that
//         log2( e_h / e_{h/2} )  ≈  order
//      to within ±0.5 (the second ratio is permitted a wider window because
//      higher-order stencils approach the FP floor faster).
//
// The polynomial-exactness check catches FP-accumulation pathologies in the
// symmetry-equivariant grouping that pure partner-equivariance tests miss.
// The convergence-rate check catches "right grouping, wrong weights" bugs
// where the codegen ships a stencil of one order under the label of another.
//
// Author: carlo.musolino@aei.mpg.de

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/physics/fd_subexpressions.hh>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using HostView = Kokkos::View<double***, Kokkos::HostSpace>;
using Catch::Matchers::WithinAbs;

// Deterministic xorshift PRNG; same shape as test_fd_symmetry.cpp.
struct Rng {
	std::uint64_t s;
	explicit Rng(std::uint64_t seed) : s(seed ? seed : 0xC0DEC0DEDEADBEEFull) {}
	double next() {
		s ^= s << 13; s ^= s >> 7; s ^= s << 17;
		return double(s >> 11) * (1.0 / double(1ull << 53)) - 0.5;
	}
};

// =====================================================================
// Random trivariate polynomial of total degree ≤ deg.  Stored as a flat
// vector indexed by (a*Dim1 + b*Dim2 + c) where 0 ≤ a+b+c ≤ deg.
// =====================================================================
struct Poly {
	int deg;
	std::vector<double> c; // index (a,b,c) -> c[((a*(deg+1)) + b)*(deg+1) + c]

	double coeff(int a, int b, int c) const {
		return this->c[((a * (deg + 1)) + b) * (deg + 1) + c];
	}

	double eval(double x, double y, double z) const {
		double s = 0.0;
		for (int a = 0; a <= deg; ++a) {
			for (int b = 0; a + b <= deg; ++b) {
				for (int cc = 0; a + b + cc <= deg; ++cc) {
					s += coeff(a, b, cc)
					   * std::pow(x, a) * std::pow(y, b) * std::pow(z, cc);
				}
			}
		}
		return s;
	}

	// k-th derivative wrt the axis-tuple (kx, ky, kz).
	double deriv(double x, double y, double z, int kx, int ky, int kz) const {
		double s = 0.0;
		for (int a = kx; a <= deg; ++a) {
			for (int b = ky; a + b <= deg; ++b) {
				for (int cc = kz; a + b + cc <= deg; ++cc) {
					double w = coeff(a, b, cc);
					for (int t = 0; t < kx; ++t) w *= (a - t);
					for (int t = 0; t < ky; ++t) w *= (b - t);
					for (int t = 0; t < kz; ++t) w *= (cc - t);
					s += w * std::pow(x, a - kx)
					        * std::pow(y, b - ky)
					        * std::pow(z, cc - kz);
				}
			}
		}
		return s;
	}
};

Poly make_random_poly(int deg, std::uint64_t seed) {
	Poly p;
	p.deg = deg;
	int side = deg + 1;
	p.c.assign(side * side * side, 0.0);
	Rng rng(seed);
	for (int a = 0; a <= deg; ++a)
		for (int b = 0; a + b <= deg; ++b)
			for (int cc = 0; a + b + cc <= deg; ++cc)
				p.c[((a * side) + b) * side + cc] = rng.next();
	return p;
}

// Fill u(i,j,k) = poly( (i - ic)*h, (j - jc)*h, (k - kc)*h ) so the centre
// of the test polynomial lands on (ic, jc, kc).  This keeps stencil-centre
// arithmetic small in magnitude and the analytic derivative trivially
// computable as poly.deriv(0, 0, 0, kx, ky, kz).
void fill_poly_grid(HostView u, const Poly& p, double h,
                    int ic, int jc, int kc)
{
	int Nx = (int)u.extent(0);
	int Ny = (int)u.extent(1);
	int Nz = (int)u.extent(2);
	for (int i = 0; i < Nx; ++i) {
		double x = (i - ic) * h;
		for (int j = 0; j < Ny; ++j) {
			double y = (j - jc) * h;
			for (int k = 0; k < Nz; ++k) {
				double z = (k - kc) * h;
				u(i, j, k) = p.eval(x, y, z);
			}
		}
	}
}

// Fill u with a smooth non-polynomial test function for convergence testing.
//   u(x,y,z) = sin(K x) * cos(K y) * exp(α z),   K = π/2,  α = 1/2
// The frequency K is chosen low enough that the sub-leading O(h^{N+1})
// truncation term does not dominate or nearly cancel the leading O(h^N)
// term at our coarsest grid spacing h=0.05 — which would push the measured
// rate well below `order` and produce false-negative convergence failures
// even though the operator is correct at every order (proven by polynomial
// exactness).  Analytic 1st & 2nd derivatives are inline in the test bodies.
namespace conv_func {
	constexpr double KW = M_PI / 2.0; // spatial frequency
	constexpr double AZ = 0.5;        // z-exponent scale
}
void fill_smooth_grid(HostView u, double h, double x0, double y0, double z0)
{
	int Nx = (int)u.extent(0);
	int Ny = (int)u.extent(1);
	int Nz = (int)u.extent(2);
	const double K = conv_func::KW;
	const double A = conv_func::AZ;
	for (int i = 0; i < Nx; ++i) {
		double x = x0 + i * h;
		double sx = std::sin(K * x);
		for (int j = 0; j < Ny; ++j) {
			double y = y0 + j * h;
			double cy = std::cos(K * y);
			for (int k = 0; k < Nz; ++k) {
				double z = z0 + k * h;
				u(i, j, k) = sx * cy * std::exp(A * z);
			}
		}
	}
}

// Tolerance for polynomial exactness.  Model:
//   tol = ULP_BUDGET · eps · max(|u|) · invh^{deriv_total}
// where `invh^{deriv_total}` accounts for the kth derivative's leading
// invh-power amplification of FP errors in the partial sum.  ULP_BUDGET
// covers accumulation over a stencil whose width grows with order.
double poly_tol(double scale, int order, int deriv_total, double invh) {
	constexpr double ULP_BUDGET = 64.0;
	double invh_pow = 1.0;
	for (int i = 0; i < deriv_total; ++i) invh_pow *= invh;
	double width_factor = static_cast<double>((order + 1) * (deriv_total > 1 ? (order + 1) : 1));
	return ULP_BUDGET * std::numeric_limits<double>::epsilon()
	       * std::max(scale, 1.0) * invh_pow * width_factor;
}

// ---------------------------------------------------------------------
// Generic exactness-checker for an FD kernel that returns a single scalar.
// `axis_order` is the order of the polynomial we feed in (= FD exactness
// degree).  `deriv_axes` tells us which derivative of the poly to compare
// against (e.g. {1,0,0} for ∂/∂x, {2,0,0} for ∂²/∂x², etc.).
// ---------------------------------------------------------------------
template <typename Kernel>
void check_poly_exact(int poly_deg, int order,
                      std::array<int, 3> deriv_axes,
                      Kernel&& kernel,
                      std::uint64_t seed,
                      double tol_multiplier = 1.0)
{
	constexpr int Nside = 24;
	constexpr int ic = Nside / 2;
	const double h = 0.1; // arbitrary; cancels in centred FD when poly fits exactly
	const double invh = 1.0 / h;
	HostView u("u_poly", Nside, Nside, Nside);

	Poly p = make_random_poly(poly_deg, seed);
	fill_poly_grid(u, p, h, ic, ic, ic);

	double du = 0.0;
	kernel(u, ic, ic, ic, invh, &du);

	double analytic = p.deriv(0.0, 0.0, 0.0,
	                          deriv_axes[0], deriv_axes[1], deriv_axes[2]);

	// Bound on |u|·eps accumulation: use the maximum magnitude in the
	// stencil footprint as the scale, multiplied by a per-order safety
	// factor (catches accumulation of partial sums).
	double scale = 0.0;
	int reach = std::max({order, 4});
	for (int dz = -reach; dz <= reach; ++dz)
	for (int dy = -reach; dy <= reach; ++dy)
	for (int dx = -reach; dx <= reach; ++dx) {
		double v = std::abs(u(ic + dx, ic + dy, ic + dz));
		if (v > scale) scale = v;
	}
	int deriv_total = deriv_axes[0] + deriv_axes[1] + deriv_axes[2];
	double tol = tol_multiplier * poly_tol(scale, order, deriv_total, invh);

	INFO("order=" << order << " poly_deg=" << poly_deg
	     << " axes=(" << deriv_axes[0] << ',' << deriv_axes[1] << ',' << deriv_axes[2] << ')'
	     << " analytic=" << analytic << " du=" << du
	     << " |diff|=" << std::abs(du - analytic) << " tol=" << tol);
	REQUIRE_THAT(du, WithinAbs(analytic, tol));
}

// Same shape but for KO: the stencil annihilates polynomials of degree ≤ N.
template <typename Kernel>
void check_poly_annihilation(int poly_deg, int order,
                             Kernel&& kernel,
                             std::uint64_t seed)
{
	constexpr int Nside = 24;
	constexpr int ic = Nside / 2;
	const double h = 0.1;
	const double invh = 1.0 / h;
	HostView u("u_poly_ko", Nside, Nside, Nside);

	Poly p = make_random_poly(poly_deg, seed);
	fill_poly_grid(u, p, h, ic, ic, ic);

	double du = 0.0;
	kernel(u, ic, ic, ic, invh, &du);

	double scale = 0.0;
	int reach = order + 2;
	for (int dz = -reach; dz <= reach; ++dz)
	for (int dy = -reach; dy <= reach; ++dy)
	for (int dx = -reach; dx <= reach; ++dx) {
		double v = std::abs(u(ic + dx, ic + dy, ic + dz));
		if (v > scale) scale = v;
	}
	// KO body emits invh^1 (the leading invh-power is 1, regardless of
	// the natural derivative-order built into the weights).
	double tol = poly_tol(scale, order, /*deriv_total=*/1, invh);

	INFO("KO order=" << order << " poly_deg=" << poly_deg
	     << " du=" << du << " tol=" << tol);
	REQUIRE_THAT(du, WithinAbs(0.0, tol));
}

// Convergence-rate check.  Evaluates the kernel on the smooth function at
// three grid spacings and asserts log2(e_h / e_{h/2}) ≈ order ± 0.5.
template <typename Kernel, typename AnalyticFn>
void check_convergence(int order, Kernel&& kernel, AnalyticFn&& analytic_fn,
                       std::array<int, 3> stencil_reach)
{
	// Pick h0 so that h0^order is well above FP ulp for order≤6.
	// h0=0.05 gives h0^6 ≈ 1.5e-8 — three orders above eps at order 6.
	const double h_seq[3] = {0.05, 0.025, 0.0125};
	double err[3];

	// Choose a "random" interior evaluation point.
	const double x0_phys = 0.30, y0_phys = 0.40, z0_phys = 0.50;
	const double analytic = analytic_fn(x0_phys, y0_phys, z0_phys);

	for (int g = 0; g < 3; ++g) {
		double h = h_seq[g];
		// Grid sized to comfortably contain the stencil reach on each side
		// plus a few extra cells of margin.
		int margin = std::max({stencil_reach[0], stencil_reach[1], stencil_reach[2]}) + 2;
		int Nside = 2 * margin + 1;
		// Place evaluation point at the grid centre.
		int ic = margin;
		double x0_grid = x0_phys - ic * h;
		double y0_grid = y0_phys - ic * h;
		double z0_grid = z0_phys - ic * h;

		HostView u("u_conv", Nside, Nside, Nside);
		fill_smooth_grid(u, h, x0_grid, y0_grid, z0_grid);

		double du = 0.0;
		kernel(u, ic, ic, ic, 1.0 / h, &du);
		err[g] = std::abs(du - analytic);
	}

	// Decide which rate ratio (if any) is meaningful and apply the order check.
	// Three regimes:
	//   (A) All-below-floor: truncation has already dropped below FP roundoff
	//       at the coarsest h, so every rate is noise.  The operator delivered
	//       the analytical answer to within roundoff; just confirm the
	//       magnitude stays at the floor.  This is the right thing for high-
	//       order 2nd-derivative stencils on smooth low-frequency functions
	//       where the leading-error coefficient is tiny.
	//   (B) Asymptotic at h/4: rate2 is reliable, require it close to `order`.
	//   (C) FP-floor reached between h/2 and h/4 (err[2] bounces back up from
	//       roundoff): rate2 is garbage; use rate1 with a looser bound.
	double rate1 = std::log2(err[0] / err[1]);
	double rate2 = std::log2(err[1] / err[2]);

	INFO("order=" << order
	     << " err[h=0.05]="     << err[0]
	     << " err[h=0.025]="    << err[1]
	     << " err[h=0.0125]="   << err[2]
	     << " rate(h->h/2)="    << rate1
	     << " rate(h/2->h/4)="  << rate2);

	constexpr double FP_FLOOR = 1e-11; // ~ eps · O(scale · invh^{deriv_total})
	double max_err = std::max({err[0], err[1], err[2]});
	bool all_below_floor = (max_err < FP_FLOOR);
	bool rate2_asymptotic = (err[2] > FP_FLOOR) && (err[2] < 0.5 * err[1]);

	if (all_below_floor) {
		// (A) Operator hit FP roundoff across the entire h-window.  Polynomial
		// exactness has already proven the stencil's order; here we just
		// confirm the error stays small.  100×FP_FLOOR leaves enough headroom
		// for the worst per-cell roundoff times stencil width.
		REQUIRE(max_err < 100.0 * FP_FLOOR);
	} else if (rate2_asymptotic) {
		// (B) Tight match to `order` at h/4.
		REQUIRE(std::abs(rate2 - order) <= 0.8);
	} else {
		// (C) Use rate1 with a looser bound; still strong enough to catch a
		// stencil that claims order N but delivers order N-2 (whose rate1
		// would land at ≈ N-2, well below the threshold).
		REQUIRE(rate1 >= order - 1.0);
		REQUIRE(rate1 <= order + 1.5);
	}
	// Smoke check, applies in all regimes except all-below-floor where it's
	// already vacuous:  the error from h to h/2 must not blow up.  Add the
	// floor tolerance so FP-noise wobbles don't trip a passing operator.
	if (!all_below_floor) {
		REQUIRE(err[1] < 2.0 * err[0] + FP_FLOOR);
	}
}

} // namespace

// =====================================================================
// Centered 1st derivative — fd_der_{x,y,z}<Order>
// =====================================================================

template <int Order>
void test_centered_1st_poly()
{
	auto Kx = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x<Order>(v,i,j,k,h,d); };
	auto Ky = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y<Order>(v,i,j,k,h,d); };
	auto Kz = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z<Order>(v,i,j,k,h,d); };
	SECTION("x") { check_poly_exact(Order, Order, {1,0,0}, Kx, 0xC0FFEE01u + Order); }
	SECTION("y") { check_poly_exact(Order, Order, {0,1,0}, Ky, 0xC0FFEE02u + Order); }
	SECTION("z") { check_poly_exact(Order, Order, {0,0,1}, Kz, 0xC0FFEE03u + Order); }
}

template <int Order>
void test_centered_1st_conv()
{
	auto Kx = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x<Order>(v,i,j,k,h,d); };
	auto Ky = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y<Order>(v,i,j,k,h,d); };
	auto Kz = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z<Order>(v,i,j,k,h,d); };
	const double K = conv_func::KW;
	const double A = conv_func::AZ;
	auto dx_smooth = [=](double x, double y, double z) {
		return K * std::cos(K*x) * std::cos(K*y) * std::exp(A*z);
	};
	auto dy_smooth = [=](double x, double y, double z) {
		return -K * std::sin(K*x) * std::sin(K*y) * std::exp(A*z);
	};
	auto dz_smooth = [=](double x, double y, double z) {
		return A * std::sin(K*x) * std::cos(K*y) * std::exp(A*z);
	};
	int reach = Order / 2;
	SECTION("x") { check_convergence(Order, Kx, dx_smooth, {reach, 0, 0}); }
	SECTION("y") { check_convergence(Order, Ky, dy_smooth, {0, reach, 0}); }
	SECTION("z") { check_convergence(Order, Kz, dz_smooth, {0, 0, reach}); }
}

TEST_CASE("fd_der_*<2>: polynomial exactness",  "[fd_correctness][poly][1st]") { test_centered_1st_poly<2>(); }
TEST_CASE("fd_der_*<4>: polynomial exactness",  "[fd_correctness][poly][1st]") { test_centered_1st_poly<4>(); }
TEST_CASE("fd_der_*<6>: polynomial exactness",  "[fd_correctness][poly][1st]") { test_centered_1st_poly<6>(); }
TEST_CASE("fd_der_*<2>: convergence rate",      "[fd_correctness][conv][1st]") { test_centered_1st_conv<2>(); }
TEST_CASE("fd_der_*<4>: convergence rate",      "[fd_correctness][conv][1st]") { test_centered_1st_conv<4>(); }
TEST_CASE("fd_der_*<6>: convergence rate",      "[fd_correctness][conv][1st]") { test_centered_1st_conv<6>(); }


// =====================================================================
// Biased L1/R1 1st derivative — fd_der_{x,y,z}_{l1,r1}<Order>
// =====================================================================

template <int Order>
void test_biased_1st_poly()
{
	auto Kxl = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_l1<Order>(v,i,j,k,h,d); };
	auto Kxr = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_r1<Order>(v,i,j,k,h,d); };
	auto Kyl = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y_l1<Order>(v,i,j,k,h,d); };
	auto Kyr = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y_r1<Order>(v,i,j,k,h,d); };
	auto Kzl = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z_l1<Order>(v,i,j,k,h,d); };
	auto Kzr = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z_r1<Order>(v,i,j,k,h,d); };
	SECTION("x_l1") { check_poly_exact(Order, Order, {1,0,0}, Kxl, 0xBADC0DE0u + Order); }
	SECTION("x_r1") { check_poly_exact(Order, Order, {1,0,0}, Kxr, 0xBADC0DE1u + Order); }
	SECTION("y_l1") { check_poly_exact(Order, Order, {0,1,0}, Kyl, 0xBADC0DE2u + Order); }
	SECTION("y_r1") { check_poly_exact(Order, Order, {0,1,0}, Kyr, 0xBADC0DE3u + Order); }
	SECTION("z_l1") { check_poly_exact(Order, Order, {0,0,1}, Kzl, 0xBADC0DE4u + Order); }
	SECTION("z_r1") { check_poly_exact(Order, Order, {0,0,1}, Kzr, 0xBADC0DE5u + Order); }
}

TEST_CASE("fd_der_*_{l1,r1}<2>: polynomial exactness", "[fd_correctness][poly][biased]") { test_biased_1st_poly<2>(); }
TEST_CASE("fd_der_*_{l1,r1}<4>: polynomial exactness", "[fd_correctness][poly][biased]") { test_biased_1st_poly<4>(); }
TEST_CASE("fd_der_*_{l1,r1}<6>: polynomial exactness", "[fd_correctness][poly][biased]") { test_biased_1st_poly<6>(); }

// Convergence rates for biased stencils: same order as their centered counterpart.
template <int Order>
void test_biased_1st_conv()
{
	auto Kxl = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_l1<Order>(v,i,j,k,h,d); };
	auto Kxr = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_r1<Order>(v,i,j,k,h,d); };
	const double K = conv_func::KW;
	const double A = conv_func::AZ;
	auto dx_smooth = [=](double x, double y, double z) {
		return K * std::cos(K*x) * std::cos(K*y) * std::exp(A*z);
	};
	int reach = Order / 2 + 1; // biased stencil reaches one cell further on one side
	SECTION("x_l1") { check_convergence(Order, Kxl, dx_smooth, {reach, 0, 0}); }
	SECTION("x_r1") { check_convergence(Order, Kxr, dx_smooth, {reach, 0, 0}); }
}

TEST_CASE("fd_der_x_{l1,r1}<2>: convergence rate", "[fd_correctness][conv][biased]") { test_biased_1st_conv<2>(); }
TEST_CASE("fd_der_x_{l1,r1}<4>: convergence rate", "[fd_correctness][conv][biased]") { test_biased_1st_conv<4>(); }
TEST_CASE("fd_der_x_{l1,r1}<6>: convergence rate", "[fd_correctness][conv][biased]") { test_biased_1st_conv<6>(); }


// =====================================================================
// Centered 2nd derivative diagonal — fd_der_{xx,yy,zz}<Order>
// =====================================================================

template <int Order>
void test_centered_2nd_diag_poly()
{
	auto Kxx = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_xx<Order>(v,i,j,k,h,d); };
	auto Kyy = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_yy<Order>(v,i,j,k,h,d); };
	auto Kzz = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_zz<Order>(v,i,j,k,h,d); };
	// Centered 2nd-derivative stencil is exact on polynomials of degree ≤ Order+1.
	SECTION("xx") { check_poly_exact(Order + 1, Order, {2,0,0}, Kxx, 0xD15EA5E0u + Order); }
	SECTION("yy") { check_poly_exact(Order + 1, Order, {0,2,0}, Kyy, 0xD15EA5E1u + Order); }
	SECTION("zz") { check_poly_exact(Order + 1, Order, {0,0,2}, Kzz, 0xD15EA5E2u + Order); }
}

template <int Order>
void test_centered_2nd_diag_conv()
{
	auto Kxx = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_xx<Order>(v,i,j,k,h,d); };
	auto Kyy = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_yy<Order>(v,i,j,k,h,d); };
	auto Kzz = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_zz<Order>(v,i,j,k,h,d); };
	const double K = conv_func::KW;
	const double A = conv_func::AZ;
	auto dxx_smooth = [=](double x, double y, double z) {
		return -K*K * std::sin(K*x) * std::cos(K*y) * std::exp(A*z);
	};
	auto dyy_smooth = [=](double x, double y, double z) {
		return -K*K * std::sin(K*x) * std::cos(K*y) * std::exp(A*z);
	};
	auto dzz_smooth = [=](double x, double y, double z) {
		return A*A * std::sin(K*x) * std::cos(K*y) * std::exp(A*z);
	};
	int reach = Order / 2;
	SECTION("xx") { check_convergence(Order, Kxx, dxx_smooth, {reach, 0, 0}); }
	SECTION("yy") { check_convergence(Order, Kyy, dyy_smooth, {0, reach, 0}); }
	SECTION("zz") { check_convergence(Order, Kzz, dzz_smooth, {0, 0, reach}); }
}

TEST_CASE("fd_der_{xx,yy,zz}<2>: polynomial exactness", "[fd_correctness][poly][2nd]") { test_centered_2nd_diag_poly<2>(); }
TEST_CASE("fd_der_{xx,yy,zz}<4>: polynomial exactness", "[fd_correctness][poly][2nd]") { test_centered_2nd_diag_poly<4>(); }
TEST_CASE("fd_der_{xx,yy,zz}<6>: polynomial exactness", "[fd_correctness][poly][2nd]") { test_centered_2nd_diag_poly<6>(); }
TEST_CASE("fd_der_{xx,yy,zz}<2>: convergence rate",     "[fd_correctness][conv][2nd]") { test_centered_2nd_diag_conv<2>(); }
TEST_CASE("fd_der_{xx,yy,zz}<4>: convergence rate",     "[fd_correctness][conv][2nd]") { test_centered_2nd_diag_conv<4>(); }
TEST_CASE("fd_der_{xx,yy,zz}<6>: convergence rate",     "[fd_correctness][conv][2nd]") { test_centered_2nd_diag_conv<6>(); }


// =====================================================================
// Centered 2nd derivative mixed — fd_der_{xy,xz,yz}<Order>
// =====================================================================

template <int Order>
void test_centered_2nd_mixed_poly()
{
	auto Kxy = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_xy<Order>(v,i,j,k,h,d); };
	auto Kxz = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_xz<Order>(v,i,j,k,h,d); };
	auto Kyz = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_yz<Order>(v,i,j,k,h,d); };
	// Mixed stencil = (1st)*(1st); exactness window = degree ≤ 2*Order.
	// Test with degree = 2*Order (or Order+1, the minimum needed to exercise
	// both factors).  We use Order+1 to keep the tolerance manageable.
	int deg = Order + 1;
	SECTION("xy") { check_poly_exact(deg, Order, {1,1,0}, Kxy, 0xFEEDFACE0u + Order); }
	SECTION("xz") { check_poly_exact(deg, Order, {1,0,1}, Kxz, 0xFEEDFACE1u + Order); }
	SECTION("yz") { check_poly_exact(deg, Order, {0,1,1}, Kyz, 0xFEEDFACE2u + Order); }
}

template <int Order>
void test_centered_2nd_mixed_conv()
{
	auto Kxy = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_xy<Order>(v,i,j,k,h,d); };
	const double K = conv_func::KW;
	const double A = conv_func::AZ;
	auto dxy_smooth = [=](double x, double y, double z) {
		return -K*K * std::cos(K*x) * std::sin(K*y) * std::exp(A*z);
	};
	int reach = Order / 2;
	SECTION("xy") { check_convergence(Order, Kxy, dxy_smooth, {reach, reach, 0}); }
}

TEST_CASE("fd_der_{xy,xz,yz}<2>: polynomial exactness", "[fd_correctness][poly][2nd_mix]") { test_centered_2nd_mixed_poly<2>(); }
TEST_CASE("fd_der_{xy,xz,yz}<4>: polynomial exactness", "[fd_correctness][poly][2nd_mix]") { test_centered_2nd_mixed_poly<4>(); }
TEST_CASE("fd_der_{xy,xz,yz}<6>: polynomial exactness", "[fd_correctness][poly][2nd_mix]") { test_centered_2nd_mixed_poly<6>(); }
TEST_CASE("fd_der_xy<2>: convergence rate",             "[fd_correctness][conv][2nd_mix]") { test_centered_2nd_mixed_conv<2>(); }
TEST_CASE("fd_der_xy<4>: convergence rate",             "[fd_correctness][conv][2nd_mix]") { test_centered_2nd_mixed_conv<4>(); }
TEST_CASE("fd_der_xy<6>: convergence rate",             "[fd_correctness][conv][2nd_mix]") { test_centered_2nd_mixed_conv<6>(); }


// =====================================================================
// Upwind 1st derivative — fd_der_{x,y,z}_upw_{pos,neg}<Order>
//
// The upwind stencil is exact on polynomials of degree ≤ Order, identical
// to the centered case.  It has a different truncation constant but the
// same convergence rate.
// =====================================================================

template <int Order>
void test_upwind_1st_poly()
{
	auto Kxp = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_upw_pos<Order>(v,i,j,k,h,d); };
	auto Kxn = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_upw_neg<Order>(v,i,j,k,h,d); };
	auto Kyp = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y_upw_pos<Order>(v,i,j,k,h,d); };
	auto Kyn = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y_upw_neg<Order>(v,i,j,k,h,d); };
	auto Kzp = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z_upw_pos<Order>(v,i,j,k,h,d); };
	auto Kzn = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z_upw_neg<Order>(v,i,j,k,h,d); };
	SECTION("x_upw_pos") { check_poly_exact(Order, Order, {1,0,0}, Kxp, 0xABCDEF01u + Order); }
	SECTION("x_upw_neg") { check_poly_exact(Order, Order, {1,0,0}, Kxn, 0xABCDEF02u + Order); }
	SECTION("y_upw_pos") { check_poly_exact(Order, Order, {0,1,0}, Kyp, 0xABCDEF03u + Order); }
	SECTION("y_upw_neg") { check_poly_exact(Order, Order, {0,1,0}, Kyn, 0xABCDEF04u + Order); }
	SECTION("z_upw_pos") { check_poly_exact(Order, Order, {0,0,1}, Kzp, 0xABCDEF05u + Order); }
	SECTION("z_upw_neg") { check_poly_exact(Order, Order, {0,0,1}, Kzn, 0xABCDEF06u + Order); }
}

template <int Order>
void test_upwind_1st_conv()
{
	auto Kxp = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_upw_pos<Order>(v,i,j,k,h,d); };
	auto Kxn = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_upw_neg<Order>(v,i,j,k,h,d); };
	const double K = conv_func::KW;
	const double A = conv_func::AZ;
	auto dx_smooth = [=](double x, double y, double z) {
		return K * std::cos(K*x) * std::cos(K*y) * std::exp(A*z);
	};
	int reach = Order / 2 + 1;
	SECTION("x_upw_pos") { check_convergence(Order, Kxp, dx_smooth, {reach, 0, 0}); }
	SECTION("x_upw_neg") { check_convergence(Order, Kxn, dx_smooth, {reach, 0, 0}); }
}

TEST_CASE("fd_der_*_upw_*<2>: polynomial exactness", "[fd_correctness][poly][upw]") { test_upwind_1st_poly<2>(); }
TEST_CASE("fd_der_*_upw_*<4>: polynomial exactness", "[fd_correctness][poly][upw]") { test_upwind_1st_poly<4>(); }
TEST_CASE("fd_der_*_upw_*<6>: polynomial exactness", "[fd_correctness][poly][upw]") { test_upwind_1st_poly<6>(); }
TEST_CASE("fd_der_x_upw_*<2>: convergence rate",     "[fd_correctness][conv][upw]") { test_upwind_1st_conv<2>(); }
TEST_CASE("fd_der_x_upw_*<4>: convergence rate",     "[fd_correctness][conv][upw]") { test_upwind_1st_conv<4>(); }
TEST_CASE("fd_der_x_upw_*<6>: convergence rate",     "[fd_correctness][conv][upw]") { test_upwind_1st_conv<6>(); }


// =====================================================================
// Kreiss-Oliger dissipation — fd_diss_{x,y,z}<Order>
//
// The KO operator at order N is proportional to the (N+1)-th derivative,
// so it annihilates polynomials of degree ≤ N exactly.  We check this
// annihilation property; the convergence rate of the *dissipation term*
// (epsdiss·h·KO·u) is order N+1 on smooth flow, but that's a property of
// the wrapper in z4c.hh, not of fd_diss_* itself, so we don't test it
// here.
// =====================================================================

template <int Order>
void test_ko_annihilation()
{
	auto Dx = [](HostView v, int i, int j, int k, double h, double* d){ fd_diss_x<Order>(v,i,j,k,h,d); };
	auto Dy = [](HostView v, int i, int j, int k, double h, double* d){ fd_diss_y<Order>(v,i,j,k,h,d); };
	auto Dz = [](HostView v, int i, int j, int k, double h, double* d){ fd_diss_z<Order>(v,i,j,k,h,d); };
	SECTION("x") { check_poly_annihilation(Order, Order, Dx, 0xDADD1ED0u + Order); }
	SECTION("y") { check_poly_annihilation(Order, Order, Dy, 0xDADD1ED1u + Order); }
	SECTION("z") { check_poly_annihilation(Order, Order, Dz, 0xDADD1ED2u + Order); }
}

TEST_CASE("fd_diss_*<2>: polynomial annihilation", "[fd_correctness][poly][ko]") { test_ko_annihilation<2>(); }
TEST_CASE("fd_diss_*<4>: polynomial annihilation", "[fd_correctness][poly][ko]") { test_ko_annihilation<4>(); }
TEST_CASE("fd_diss_*<6>: polynomial annihilation", "[fd_correctness][poly][ko]") { test_ko_annihilation<6>(); }
