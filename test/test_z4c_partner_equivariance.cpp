// Bit-exact partner-equivariance test for the Z4c β-chain subexpressions.
//
// For each function under test we build a random input set and a "partner"
// input set obtained by sign-flipping each input array element by its parity
// under the chosen discrete-symmetry transform (here: π_z = rotation π about
// the z-axis, i.e. x→-x, y→-y, z→+z).  We then call the function on both,
// and assert that the output at the partner equals the per-component output
// parity times the original output, **bit-exactly**.
//
// Why bit-exact:  IEEE multiplication by -1 is a sign-bit flip with zero
// rounding cost, and IEEE addition is commutative.  Therefore a sum
//     T  =  M_1 + M_2 + ... + M_N
// in which every monomial individually has parity π under the partner
// substitution satisfies T(partner) = π · T(original) bit-exactly,
// irrespective of associativity.  A failure of this test on covariant
// tensor code means *some monomial does not have the parity it should* —
// either a math bug in the codegen or a compile-flag issue allowing the
// compiler to reorder additions.
//
// π_z parities used here:
//   scalars (α, W, κ_1)                             : +1
//   vectors with upper index    (Γ̃^i, Γ̃D^i)        : (-1, -1, +1)
//   covectors with lower index  (S_i, ∂_iα, …)      : (-1, -1, +1)
//   symmetric 2-tensors γ̃^{ab}, Ã^{ab}              : (xx,xy,xz,yy,yz,zz)
//                                                    = (+1,+1,-1,+1,-1,+1)
//   Γ̃^k_{ab} stored as 6·k + sym(ab):
//     upper x,y (parity -1): per-sym (-1,-1,+1,-1,+1,-1)
//     upper z   (parity +1): per-sym (+1,+1,-1,+1,-1,+1)
//   ∂_j β^i  stored as dbetau_dx[i + 3·j]
//      = parity(β^i) × parity(x_j)
//   ∂_j ∂_k β^i stored as ddbetau_dx2[3·sym(jk) + i]
//      = parity(x_j) × parity(x_k) × parity(β^i)
//
// Author: carlo.musolino@aei.mpg.de

#include <catch2/catch_test_macros.hpp>
#include <grace/physics/z4c_subexpressions.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

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

void fill_gtdd(double g[6], Rng& rng) {
	g[0] = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
	g[1] = 0.05       * rng.uniform(-1.0, 1.0);
	g[2] = 0.05       * rng.uniform(-1.0, 1.0);
	g[3] = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
	g[4] = 0.05       * rng.uniform(-1.0, 1.0);
	g[5] = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
}

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

template <int N>
void apply_parity(double dst[N], const double src[N], const int par[N]) {
	for (int i = 0; i < N; ++i)
		dst[i] = (par[i] > 0) ? src[i] : -src[i];
}

// ============= π_z parity tables =============

const int gtuu_par      [6]  = { +1, +1, -1, +1, -1, +1 };
const int Atuu_par      [6]  = { +1, +1, -1, +1, -1, +1 };
const int Gammatu_par   [3]  = { -1, -1, +1 };
const int GammatDu_par  [3]  = { -1, -1, +1 };
const int Si_par        [3]  = { -1, -1, +1 };
const int dKhat_par     [3]  = { -1, -1, +1 };
const int dW_par        [3]  = { -1, -1, +1 };
const int dalp_par      [3]  = { -1, -1, +1 };
const int dtheta_par    [3]  = { -1, -1, +1 };
// Γ̃^k_{ab}: layout 6·k + sym(ab), with sym(ab) ∈ [xx,xy,xz,yy,yz,zz]
const int Gammatudd_par [18] = {
	-1, -1, +1, -1, +1, -1,   // upper x
	-1, -1, +1, -1, +1, -1,   // upper y
	+1, +1, -1, +1, -1, +1    // upper z
};
// ∂_j β^i: layout i + 3·j ; parity = parity(β^i) × parity(x_j)
const int dbetau_par    [9]  = {
	+1, +1, -1,   // ∂_x β^{x,y,z}
	+1, +1, -1,   // ∂_y β^{x,y,z}
	-1, -1, +1    // ∂_z β^{x,y,z}
};
// ∂_j ∂_k β^i: layout 3·sym(jk) + i ;
// pair parities for jk in [xx,xy,xz,yy,yz,zz] = (+,+,-,+,-,+) ;
// vector parities for i in [x,y,z] = (-,-,+)
const int ddbetau_par   [18] = {
	-1, -1, +1,   // sym xx (+) × i [x,y,z]
	-1, -1, +1,   // sym xy (+)
	+1, +1, -1,   // sym xz (-)
	-1, -1, +1,   // sym yy (+)
	+1, +1, -1,   // sym yz (-)
	-1, -1, +1    // sym zz (+)
};

// Output parities under π_z: Γ̃D^i and Γ̃-RHS^i both have the parity of Γ̃^i.
const int (&out_par)[3] = GammatDu_par;

// Symmetric 2-tensor (lower-lower or upper-upper) parities under π_z,
// Voigt order [xx, xy, xz, yy, yz, zz].  Same as gtuu/Atuu/sij/W2Rdd/etc.
const int sym2_par[6] = { +1, +1, -1, +1, -1, +1 };

// Polar 3-vector parity table under π_z (used for Bdriver, dGammatu_dt).
const int polar_vec_par[3] = { -1, -1, +1 };

} // namespace

// =====================================================================
// z4c_get_contracted_Christoffel  —  GammatDu^i = Γ̃^i_{ab} γ̃^{ab}
// =====================================================================
TEST_CASE("z4c_get_contracted_Christoffel: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	constexpr int N_TRIALS = 256;
	Rng rng(0xCC11u);

	double worst_abs[3] = { 0.0, 0.0, 0.0 };
	int    worst_trial[3] = { -1, -1, -1 };

	for (int trial = 0; trial < N_TRIALS; ++trial) {
		double gtdd[6]; fill_gtdd(gtdd, rng);
		double gtuu[6]; invert_sym3(gtdd, gtuu);
		double Gammatudd[18]; fill_uniform<18>(Gammatudd, rng, 0.5);

		double GammatDu_orig[3] = { 0.0, 0.0, 0.0 };
		z4c_get_contracted_Christoffel(gtuu, Gammatudd, &GammatDu_orig);

		double gtuu_p     [6];  apply_parity<6 >(gtuu_p,      gtuu,      gtuu_par     );
		double Gammatudd_p[18]; apply_parity<18>(Gammatudd_p, Gammatudd, Gammatudd_par);

		double GammatDu_part[3] = { 0.0, 0.0, 0.0 };
		z4c_get_contracted_Christoffel(gtuu_p, Gammatudd_p, &GammatDu_part);

		for (int i = 0; i < 3; ++i) {
			const double expected = (out_par[i] > 0) ? GammatDu_orig[i]
			                                         : -GammatDu_orig[i];
			const double diff = GammatDu_part[i] - expected;
			if (std::abs(diff) >= worst_abs[i]) {
				worst_abs[i]   = std::abs(diff);
				worst_trial[i] = trial;
			}
			INFO("trial=" << trial << " i=" << i
			     << "  orig="     << GammatDu_orig[i]
			     << "  partner="  << GammatDu_part[i]
			     << "  expected=" << expected
			     << "  diff="     << diff);
			REQUIRE(GammatDu_part[i] == expected); // bit-exact
		}
	}

	for (int i = 0; i < 3; ++i)
		WARN("contracted_Christoffel  i=" << i
		     << "  worst |Δ|=" << worst_abs[i]
		     << "  (trial "    << worst_trial[i] << ")");
}

// =====================================================================
// z4c_get_Gammatilde_rhs  —  dGammatu_dt^i
// =====================================================================
TEST_CASE("z4c_get_Gammatilde_rhs: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	constexpr int N_TRIALS = 256;
	Rng rng(0xCC12u);

	double worst_abs[3] = { 0.0, 0.0, 0.0 };
	int    worst_trial[3] = { -1, -1, -1 };

	for (int trial = 0; trial < N_TRIALS; ++trial) {
		// Scalars are parity-+ under π_z; identical at the partner.
		const double alp    = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
		const double W      = 1.0  + 0.1  * rng.uniform(-1.0, 1.0);
		const double kappa1 = 0.05 * rng.uniform( 0.0, 1.0);

		double gtdd[6]; fill_gtdd(gtdd, rng);
		double gtuu[6]; invert_sym3(gtdd, gtuu);

		double Gammatu     [3]  ; fill_uniform<3 >(Gammatu     , rng, 0.05);
		double Si          [3]  ; fill_uniform<3 >(Si          , rng, 0.01);
		double Atuu        [6]  ; fill_uniform<6 >(Atuu        , rng, 0.05);
		double Gammatudd   [18] ; fill_uniform<18>(Gammatudd   , rng, 0.2 );
		double GammatDu    [3]  ; fill_uniform<3 >(GammatDu    , rng, 0.1 );
		double dbetau_dx   [9]  ; fill_uniform<9 >(dbetau_dx   , rng, 0.05);
		double dKhat_dx    [3]  ; fill_uniform<3 >(dKhat_dx    , rng, 0.05);
		double dW_dx       [3]  ; fill_uniform<3 >(dW_dx       , rng, 0.05);
		double dalp_dx     [3]  ; fill_uniform<3 >(dalp_dx     , rng, 0.05);
		double dtheta_dx   [3]  ; fill_uniform<3 >(dtheta_dx   , rng, 0.01);
		double ddbetau_dx2 [18] ; fill_uniform<18>(ddbetau_dx2 , rng, 0.05);

		double dGammatu_dt_orig[3] = { 0.0, 0.0, 0.0 };
		z4c_get_Gammatilde_rhs(
			alp, W, Gammatu, Si, kappa1, gtuu, Atuu, Gammatudd, GammatDu,
			dbetau_dx, dKhat_dx, dW_dx, dalp_dx, dtheta_dx, ddbetau_dx2,
			&dGammatu_dt_orig);

		double Gammatu_p    [3]  ; apply_parity<3 >(Gammatu_p    , Gammatu    , Gammatu_par   );
		double Si_p         [3]  ; apply_parity<3 >(Si_p         , Si         , Si_par        );
		double gtuu_p       [6]  ; apply_parity<6 >(gtuu_p       , gtuu       , gtuu_par      );
		double Atuu_p       [6]  ; apply_parity<6 >(Atuu_p       , Atuu       , Atuu_par      );
		double Gammatudd_p  [18] ; apply_parity<18>(Gammatudd_p  , Gammatudd  , Gammatudd_par );
		double GammatDu_p   [3]  ; apply_parity<3 >(GammatDu_p   , GammatDu   , GammatDu_par  );
		double dbetau_dx_p  [9]  ; apply_parity<9 >(dbetau_dx_p  , dbetau_dx  , dbetau_par    );
		double dKhat_dx_p   [3]  ; apply_parity<3 >(dKhat_dx_p   , dKhat_dx   , dKhat_par     );
		double dW_dx_p      [3]  ; apply_parity<3 >(dW_dx_p      , dW_dx      , dW_par        );
		double dalp_dx_p    [3]  ; apply_parity<3 >(dalp_dx_p    , dalp_dx    , dalp_par      );
		double dtheta_dx_p  [3]  ; apply_parity<3 >(dtheta_dx_p  , dtheta_dx  , dtheta_par    );
		double ddbetau_dx2_p[18] ; apply_parity<18>(ddbetau_dx2_p, ddbetau_dx2, ddbetau_par   );

		double dGammatu_dt_part[3] = { 0.0, 0.0, 0.0 };
		z4c_get_Gammatilde_rhs(
			alp, W, Gammatu_p, Si_p, kappa1, gtuu_p, Atuu_p,
			Gammatudd_p, GammatDu_p,
			dbetau_dx_p, dKhat_dx_p, dW_dx_p, dalp_dx_p,
			dtheta_dx_p, ddbetau_dx2_p,
			&dGammatu_dt_part);

		for (int i = 0; i < 3; ++i) {
			const double expected = (out_par[i] > 0) ? dGammatu_dt_orig[i]
			                                         : -dGammatu_dt_orig[i];
			const double diff = dGammatu_dt_part[i] - expected;
			if (std::abs(diff) >= worst_abs[i]) {
				worst_abs[i]   = std::abs(diff);
				worst_trial[i] = trial;
			}
			INFO("trial=" << trial << " i=" << i
			     << "  orig="     << dGammatu_dt_orig[i]
			     << "  partner="  << dGammatu_dt_part[i]
			     << "  expected=" << expected
			     << "  diff="     << diff);
			REQUIRE(dGammatu_dt_part[i] == expected); // bit-exact
		}
	}

	for (int i = 0; i < 3; ++i)
		WARN("Gammatilde_rhs  i=" << i
		     << "  worst |Δ|=" << worst_abs[i]
		     << "  (trial "    << worst_trial[i] << ")");
}

// =====================================================================
// z4c_get_chi_rhs   →  dW (scalar +)
// All inputs are scalars or parity-flipped vectors; trivial but locked.
// =====================================================================
TEST_CASE("z4c_get_chi_rhs: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	constexpr int N_TRIALS = 128;
	Rng rng(0xCC21u);
	double worst = 0.0;

	for (int trial = 0; trial < N_TRIALS; ++trial) {
		const double alp   = 1.0 + 0.1*rng.uniform(-1.0, 1.0);
		const double W     = 1.0 + 0.1*rng.uniform(-1.0, 1.0);
		const double theta = 0.01*rng.uniform(-1.0, 1.0);
		const double Khat  = 0.05*rng.uniform(-1.0, 1.0);
		double dbetau_dx[9]; fill_uniform<9>(dbetau_dx, rng, 0.05);

		double dW_o = 0.0, dW_p = 0.0;
		z4c_get_chi_rhs(alp, W, theta, Khat, dbetau_dx, &dW_o);

		double dbetau_p[9]; apply_parity<9>(dbetau_p, dbetau_dx, dbetau_par);
		z4c_get_chi_rhs(alp, W, theta, Khat, dbetau_p, &dW_p);

		INFO("trial=" << trial << " dW_o=" << dW_o << " dW_p=" << dW_p);
		REQUIRE(dW_p == dW_o); // scalar parity +
		worst = std::max(worst, std::abs(dW_p - dW_o));
	}
	WARN("chi_rhs worst |Δ|=" << worst);
}

// =====================================================================
// z4c_get_gtdd_rhs  →  dgtdd_dt (sym2, parities +,+,-,+,-,+)
// =====================================================================
TEST_CASE("z4c_get_gtdd_rhs: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	constexpr int N_TRIALS = 128;
	Rng rng(0xCC22u);
	double worst[6] = {0.0,0.0,0.0,0.0,0.0,0.0};

	for (int trial = 0; trial < N_TRIALS; ++trial) {
		const double alp = 1.0 + 0.1*rng.uniform(-1.0, 1.0);
		double gtdd[6]; fill_gtdd(gtdd, rng);
		double Atdd[6]; fill_uniform<6>(Atdd, rng, 0.05);
		double dbetau_dx[9]; fill_uniform<9>(dbetau_dx, rng, 0.05);

		double dgtdd_o[6] = {0,0,0,0,0,0};
		z4c_get_gtdd_rhs(gtdd, Atdd, alp, dbetau_dx, &dgtdd_o);

		double gtdd_p[6];   apply_parity<6>(gtdd_p,   gtdd,   sym2_par);
		double Atdd_p[6];   apply_parity<6>(Atdd_p,   Atdd,   sym2_par);
		double dbetau_p[9]; apply_parity<9>(dbetau_p, dbetau_dx, dbetau_par);

		double dgtdd_p[6] = {0,0,0,0,0,0};
		z4c_get_gtdd_rhs(gtdd_p, Atdd_p, alp, dbetau_p, &dgtdd_p);

		for (int k = 0; k < 6; ++k) {
			const double expected = (sym2_par[k] > 0) ? dgtdd_o[k] : -dgtdd_o[k];
			INFO("trial=" << trial << " k=" << k
			     << " o=" << dgtdd_o[k] << " p=" << dgtdd_p[k]);
			REQUIRE(dgtdd_p[k] == expected);
			worst[k] = std::max(worst[k], std::abs(dgtdd_p[k] - expected));
		}
	}
	for (int k = 0; k < 6; ++k)
		WARN("gtdd_rhs k=" << k << "  worst |Δ|=" << worst[k]);
}

// =====================================================================
// z4c_get_Khat_rhs  →  scalar +
// All inputs scalar +; trivially equivariant but worth pinning.
// =====================================================================
TEST_CASE("z4c_get_Khat_rhs: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	constexpr int N_TRIALS = 128;
	Rng rng(0xCC23u);
	double worst = 0.0;

	for (int trial = 0; trial < N_TRIALS; ++trial) {
		const double alp     = 1.0 + 0.1*rng.uniform(-1.0, 1.0);
		const double theta   = 0.01*rng.uniform(-1.0, 1.0);
		const double Ktr     = 0.05*rng.uniform(-1.0, 1.0);
		const double S       = 0.02*rng.uniform(-1.0, 1.0);
		const double rho     = 0.02*rng.uniform( 0.0, 1.0);
		const double kappa1  = 0.05*rng.uniform( 0.0, 1.0);
		const double kappa2  = 0.05*rng.uniform( 0.0, 1.0);
		const double Asqr    = 0.01*rng.uniform( 0.0, 1.0);
		const double DiDialp = 0.01*rng.uniform(-1.0, 1.0);

		double dKhat_o = 0.0, dKhat_p = 0.0;
		z4c_get_Khat_rhs(alp, theta, Ktr, S, rho, kappa1, kappa2, Asqr,
		                 DiDialp, &dKhat_o);
		z4c_get_Khat_rhs(alp, theta, Ktr, S, rho, kappa1, kappa2, Asqr,
		                 DiDialp, &dKhat_p);
		REQUIRE(dKhat_p == dKhat_o);
		worst = std::max(worst, std::abs(dKhat_p - dKhat_o));
	}
	WARN("Khat_rhs worst |Δ|=" << worst);
}

// =====================================================================
// z4c_get_theta_rhs  →  scalar +
// All inputs scalar +.
// =====================================================================
TEST_CASE("z4c_get_theta_rhs: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	constexpr int N_TRIALS = 128;
	Rng rng(0xCC24u);
	double worst = 0.0;

	for (int trial = 0; trial < N_TRIALS; ++trial) {
		const double alp              = 1.0 + 0.1*rng.uniform(-1.0, 1.0);
		const double theta            = 0.01*rng.uniform(-1.0, 1.0);
		const double Khat             = 0.05*rng.uniform(-1.0, 1.0);
		const double rho              = 0.02*rng.uniform( 0.0, 1.0);
		const double kappa1           = 0.05*rng.uniform( 0.0, 1.0);
		const double kappa2           = 0.05*rng.uniform( 0.0, 1.0);
		const double theta_damp_fact  = 0.5*rng.uniform( 0.5, 1.5);
		const double Asqr             = 0.01*rng.uniform( 0.0, 1.0);
		const double Rtrace           = 0.05*rng.uniform(-1.0, 1.0);

		double dt_o = 0.0, dt_p = 0.0;
		z4c_get_theta_rhs(alp, theta, Khat, rho, kappa1, kappa2,
		                  theta_damp_fact, Asqr, Rtrace, &dt_o);
		z4c_get_theta_rhs(alp, theta, Khat, rho, kappa1, kappa2,
		                  theta_damp_fact, Asqr, Rtrace, &dt_p);
		REQUIRE(dt_p == dt_o);
		worst = std::max(worst, std::abs(dt_p - dt_o));
	}
	WARN("theta_rhs worst |Δ|=" << worst);
}

// =====================================================================
// z4c_get_Atdd_rhs  →  dAtdd_dt (sym2, parities +,+,-,+,-,+)
// The big one: 13 input arrays plus scalars, ~30 terms per component.
// =====================================================================
TEST_CASE("z4c_get_Atdd_rhs: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	constexpr int N_TRIALS = 256;
	Rng rng(0xCC25u);
	double worst[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
	int    worst_trial[6] = {-1,-1,-1,-1,-1,-1};

	for (int trial = 0; trial < N_TRIALS; ++trial) {
		const double alp     = 1.0 + 0.1*rng.uniform(-1.0, 1.0);
		const double W       = 1.0 + 0.1*rng.uniform(-1.0, 1.0);
		const double Ktr     = 0.05*rng.uniform(-1.0, 1.0);
		const double S       = 0.02*rng.uniform(-1.0, 1.0);
		const double DiDialp = 0.01*rng.uniform(-1.0, 1.0);
		const double Rtrace  = 0.05*rng.uniform(-1.0, 1.0);

		double gtdd[6]; fill_gtdd(gtdd, rng);
		double gtuu[6]; invert_sym3(gtdd, gtuu);
		double Atdd      [6]; fill_uniform<6>(Atdd      , rng, 0.05);
		double Sij       [6]; fill_uniform<6>(Sij       , rng, 0.02);
		double W2DiDjalp [6]; fill_uniform<6>(W2DiDjalp , rng, 0.05);
		double W2Rdd     [6]; fill_uniform<6>(W2Rdd     , rng, 0.05);
		double dbetau_dx [9]; fill_uniform<9>(dbetau_dx , rng, 0.05);

		double dAtdd_o[6] = {0,0,0,0,0,0};
		z4c_get_Atdd_rhs(gtdd, Atdd, alp, W, Ktr, S, Sij, gtuu, W2DiDjalp,
		                 DiDialp, W2Rdd, Rtrace, dbetau_dx, &dAtdd_o);

		double gtdd_p     [6]; apply_parity<6>(gtdd_p     , gtdd     , sym2_par);
		double gtuu_p     [6]; apply_parity<6>(gtuu_p     , gtuu     , sym2_par);
		double Atdd_p     [6]; apply_parity<6>(Atdd_p     , Atdd     , sym2_par);
		double Sij_p      [6]; apply_parity<6>(Sij_p      , Sij      , sym2_par);
		double W2DiDjalp_p[6]; apply_parity<6>(W2DiDjalp_p, W2DiDjalp, sym2_par);
		double W2Rdd_p    [6]; apply_parity<6>(W2Rdd_p    , W2Rdd    , sym2_par);
		double dbetau_p   [9]; apply_parity<9>(dbetau_p   , dbetau_dx, dbetau_par);

		double dAtdd_p[6] = {0,0,0,0,0,0};
		z4c_get_Atdd_rhs(gtdd_p, Atdd_p, alp, W, Ktr, S, Sij_p, gtuu_p,
		                 W2DiDjalp_p, DiDialp, W2Rdd_p, Rtrace, dbetau_p,
		                 &dAtdd_p);

		for (int k = 0; k < 6; ++k) {
			const double expected = (sym2_par[k] > 0) ? dAtdd_o[k]
			                                          : -dAtdd_o[k];
			const double diff = dAtdd_p[k] - expected;
			if (std::abs(diff) >= worst[k]) {
				worst[k] = std::abs(diff);
				worst_trial[k] = trial;
			}
			INFO("trial=" << trial << " k=" << k
			     << "  o=" << dAtdd_o[k] << " p=" << dAtdd_p[k]
			     << "  diff=" << diff);
			REQUIRE(dAtdd_p[k] == expected);
		}
	}
	for (int k = 0; k < 6; ++k)
		WARN("Atdd_rhs k=" << k
		     << "  worst |Δ|=" << worst[k]
		     << "  (trial " << worst_trial[k] << ")");
}

// =====================================================================
// z4c_get_alpha_rhs  →  scalar +
// =====================================================================
TEST_CASE("z4c_get_alpha_rhs: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	Rng rng(0xCC26u);
	for (int trial = 0; trial < 64; ++trial) {
		const double alp  = 1.0 + 0.1*rng.uniform(-1.0, 1.0);
		const double Khat = 0.05*rng.uniform(-1.0, 1.0);
		double d_o = 0.0, d_p = 0.0;
		z4c_get_alpha_rhs(alp, Khat, &d_o);
		z4c_get_alpha_rhs(alp, Khat, &d_p);
		REQUIRE(d_p == d_o);
	}
}

// =====================================================================
// z4c_get_beta_rhs   →  dbeta_dt (polar, -, -, +)
// =====================================================================
TEST_CASE("z4c_get_beta_rhs: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	Rng rng(0xCC27u);
	for (int trial = 0; trial < 64; ++trial) {
		double Bdriver[3]; fill_uniform<3>(Bdriver, rng, 0.05);
		double dbeta_o[3] = {0,0,0};
		z4c_get_beta_rhs(Bdriver, &dbeta_o);

		double Bdriver_p[3]; apply_parity<3>(Bdriver_p, Bdriver, polar_vec_par);
		double dbeta_p[3] = {0,0,0};
		z4c_get_beta_rhs(Bdriver_p, &dbeta_p);

		for (int i = 0; i < 3; ++i) {
			const double expected = (polar_vec_par[i] > 0) ? dbeta_o[i]
			                                               : -dbeta_o[i];
			REQUIRE(dbeta_p[i] == expected);
		}
	}
}

// =====================================================================
// z4c_get_Bdriver_rhs  →  dBd_dt (polar, -, -, +)
// =====================================================================
TEST_CASE("z4c_get_Bdriver_rhs: bit-exact equivariance under pirot_z",
          "[z4c_partner_equivariance][pirot_z]")
{
	Rng rng(0xCC28u);
	for (int trial = 0; trial < 64; ++trial) {
		const double eta = 1.4;
		double Bdriver[3];      fill_uniform<3>(Bdriver,      rng, 0.05);
		double dGammatu_dt[3];  fill_uniform<3>(dGammatu_dt,  rng, 0.05);
		double dBd_o[3] = {0,0,0};
		z4c_get_Bdriver_rhs(Bdriver, eta, dGammatu_dt, &dBd_o);

		double Bdriver_p[3];    apply_parity<3>(Bdriver_p,    Bdriver,    polar_vec_par);
		double dGammatu_p[3];   apply_parity<3>(dGammatu_p,   dGammatu_dt, polar_vec_par);
		double dBd_p[3] = {0,0,0};
		z4c_get_Bdriver_rhs(Bdriver_p, eta, dGammatu_p, &dBd_p);

		for (int i = 0; i < 3; ++i) {
			const double expected = (polar_vec_par[i] > 0) ? dBd_o[i]
			                                               : -dBd_o[i];
			REQUIRE(dBd_p[i] == expected);
		}
	}
}
