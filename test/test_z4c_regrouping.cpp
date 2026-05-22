// Equivalence guardrail for the partner-equivariant rewrites of the Z4c
// β-chain subexpressions.
//
// For each function pair (original in z4c_subexpressions.hh, candidate
// rewrite in z4c_subexpressions_regrouped.hh) this test feeds randomized
// but physically-plausible inputs and asserts that the two outputs agree
// to within a numerically-justified tolerance.  The tolerance is chosen
// loose enough that any IEEE-permissible reordering of additions passes,
// but tight enough that a dropped term, miscoefficient, or changed sign
// fails.
//
// While the "regrouped" implementations are still delegations, this test
// passes trivially.  Each time a function is rewritten, the test catches
// any accidental math change before we trust it in the audit.
//
// Tolerance derivation:  a flat sum of N IEEE doubles has worst-case
// rounding bounded by ~(N-1)·ulp·M, where M is the max-magnitude monomial
// in the sum.  Two distinct evaluation trees of the same flat sum can
// differ by at most twice that bound.  For the Γ̃-RHS, N ≈ 30 and M ~ O(1)
// in our test ensemble, giving a worst-case ~6·10⁻¹⁴.  We use
// rel_tol = 1·10⁻¹², which is two orders of magnitude looser than the
// IEEE bound (safety margin) yet ~3 orders tighter than the symmetry
// audit's working precision — any real bug shows up far above this floor.
//
// Author: carlo.musolino@aei.mpg.de

#include <catch2/catch_test_macros.hpp>
#include <grace/physics/z4c_subexpressions.hh>
#include <grace/physics/z4c_subexpressions_regrouped.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int    N_TRIALS = 256;
constexpr double REL_TOL  = 1.0e-12;
constexpr double ABS_FLOOR = 1.0e-15; // guards near-zero outputs

struct Rng {
	std::uint64_t s;
	explicit Rng(std::uint64_t seed)
		: s(seed ? seed : 0xC0DEC0DEDEADBEEFull) {}
	double uniform(double lo, double hi) {
		s ^= s << 13; s ^= s >> 7; s ^= s << 17;
		double u = double(s >> 11) * (1.0 / double(1ull << 53));
		return lo + (hi - lo) * u;
	}
};

// γ̃_{ij} as a small perturbation of δ_{ij}: positive-definite, eigenvalues
// near 1, off-diagonals near 0.  Voigt: [xx, xy, xz, yy, yz, zz].
void fill_gtdd(double g[6], Rng& rng) {
	g[0] = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
	g[1] = 0.05       * rng.uniform(-1.0, 1.0);
	g[2] = 0.05       * rng.uniform(-1.0, 1.0);
	g[3] = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
	g[4] = 0.05       * rng.uniform(-1.0, 1.0);
	g[5] = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
}

// Invert a symmetric 3x3 matrix in Voigt layout.
void invert_sym3(const double g[6], double gi[6]) {
	const double a = g[0], b = g[1], c = g[2];
	const double d = g[3], e = g[4], f = g[5];
	const double det = a*(d*f - e*e) - b*(b*f - c*e) + c*(b*e - c*d);
	const double inv_det = 1.0 / det;
	gi[0] =  (d*f - e*e) * inv_det;
	gi[1] = -(b*f - c*e) * inv_det;
	gi[2] =  (b*e - c*d) * inv_det;
	gi[3] =  (a*f - c*c) * inv_det;
	gi[4] = -(a*e - b*c) * inv_det;
	gi[5] =  (a*d - b*b) * inv_det;
}

template <int N>
void fill_uniform(double arr[N], Rng& rng, double scale) {
	for (int i = 0; i < N; ++i) arr[i] = scale * rng.uniform(-1.0, 1.0);
}

bool close(double a, double b) {
	const double m = std::max({std::abs(a), std::abs(b), ABS_FLOOR});
	return std::abs(a - b) <= REL_TOL * m;
}

} // namespace

// =====================================================================
// z4c_get_contracted_Christoffel
// =====================================================================
TEST_CASE("z4c_get_contracted_Christoffel: regrouped agrees with original",
          "[z4c_regrouping]")
{
	Rng rng(0xC011u);
	int max_violation_trial = -1;
	double worst_rel = 0.0;

	for (int trial = 0; trial < N_TRIALS; ++trial) {
		double gtdd[6], gtuu[6];
		fill_gtdd(gtdd, rng);
		invert_sym3(gtdd, gtuu);

		double Gammatudd[18];
		fill_uniform<18>(Gammatudd, rng, 0.5);

		double GammatDu_orig[3] = {0.0, 0.0, 0.0};
		double GammatDu_new [3] = {0.0, 0.0, 0.0};
		z4c_get_contracted_Christoffel          (gtuu, Gammatudd, &GammatDu_orig);
		z4c_get_contracted_Christoffel_regrouped(gtuu, Gammatudd, &GammatDu_new );

		for (int i = 0; i < 3; ++i) {
			const double m = std::max({std::abs(GammatDu_orig[i]),
			                           std::abs(GammatDu_new[i]),
			                           ABS_FLOOR});
			const double r = std::abs(GammatDu_orig[i] - GammatDu_new[i]) / m;
			if (r > worst_rel) { worst_rel = r; max_violation_trial = trial; }
			INFO("trial=" << trial << " i=" << i
			     << "  orig=" << GammatDu_orig[i]
			     << "  new=" << GammatDu_new[i]
			     << "  rel=" << r);
			REQUIRE(close(GammatDu_orig[i], GammatDu_new[i]));
		}
	}

	WARN("z4c_get_contracted_Christoffel worst rel-error: " << worst_rel
	     << " (trial " << max_violation_trial << ", tol=" << REL_TOL << ")");
}

// =====================================================================
// z4c_get_Gammatilde_rhs
// =====================================================================
TEST_CASE("z4c_get_Gammatilde_rhs: regrouped agrees with original",
          "[z4c_regrouping]")
{
	Rng rng(0xC012u);
	int max_violation_trial = -1;
	double worst_rel = 0.0;

	for (int trial = 0; trial < N_TRIALS; ++trial) {
		// TOV-like state: lapse near 1, W near 1, all dynamical fields
		// small-to-moderate.
		const double alp    = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
		const double W      = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
		const double kappa1 = 0.05 * rng.uniform( 0.0, 1.0);

		double gtdd[6], gtuu[6];
		fill_gtdd(gtdd, rng);
		invert_sym3(gtdd, gtuu);

		double Gammatu     [3]  ;  fill_uniform<3 >(Gammatu     , rng, 0.05);
		double Si          [3]  ;  fill_uniform<3 >(Si          , rng, 0.01);
		double Atuu        [6]  ;  fill_uniform<6 >(Atuu        , rng, 0.05);
		double Gammatudd   [18] ;  fill_uniform<18>(Gammatudd   , rng, 0.2 );
		double GammatDu    [3]  ;  fill_uniform<3 >(GammatDu    , rng, 0.1 );
		double dbetau_dx   [9]  ;  fill_uniform<9 >(dbetau_dx   , rng, 0.05);
		double dKhat_dx    [3]  ;  fill_uniform<3 >(dKhat_dx    , rng, 0.05);
		double dW_dx       [3]  ;  fill_uniform<3 >(dW_dx       , rng, 0.05);
		double dalp_dx     [3]  ;  fill_uniform<3 >(dalp_dx     , rng, 0.05);
		double dtheta_dx   [3]  ;  fill_uniform<3 >(dtheta_dx   , rng, 0.01);
		double ddbetau_dx2 [18] ;  fill_uniform<18>(ddbetau_dx2 , rng, 0.05);

		double dGammatu_dt_orig[3] = {0.0, 0.0, 0.0};
		double dGammatu_dt_new [3] = {0.0, 0.0, 0.0};

		z4c_get_Gammatilde_rhs(
			alp, W, Gammatu, Si, kappa1, gtuu, Atuu, Gammatudd, GammatDu,
			dbetau_dx, dKhat_dx, dW_dx, dalp_dx, dtheta_dx, ddbetau_dx2,
			&dGammatu_dt_orig);
		z4c_get_Gammatilde_rhs_regrouped(
			alp, W, Gammatu, Si, kappa1, gtuu, Atuu, Gammatudd, GammatDu,
			dbetau_dx, dKhat_dx, dW_dx, dalp_dx, dtheta_dx, ddbetau_dx2,
			&dGammatu_dt_new);

		for (int i = 0; i < 3; ++i) {
			const double m = std::max({std::abs(dGammatu_dt_orig[i]),
			                           std::abs(dGammatu_dt_new [i]),
			                           ABS_FLOOR});
			const double r = std::abs(dGammatu_dt_orig[i]
			                          - dGammatu_dt_new[i]) / m;
			if (r > worst_rel) { worst_rel = r; max_violation_trial = trial; }
			INFO("trial=" << trial << " i=" << i
			     << "  orig=" << dGammatu_dt_orig[i]
			     << "  new=" << dGammatu_dt_new[i]
			     << "  rel=" << r);
			REQUIRE(close(dGammatu_dt_orig[i], dGammatu_dt_new[i]));
		}
	}

	WARN("z4c_get_Gammatilde_rhs worst rel-error: " << worst_rel
	     << " (trial " << max_violation_trial << ", tol=" << REL_TOL << ")");
}
