// Drop-in replacements for selected functions in z4c_subexpressions.hh,
// rewritten so the partial-sum order is partner-equivariant under discrete
// symmetry transforms (mirror, π-rotation).
//
// Each function in this header MUST be mathematically identical to the
// original SymPy-codegen'd version in z4c_subexpressions.hh — only the
// associativity of additions is changed.  The accompanying unit test
// `test/test_z4c_regrouping.cpp` randomizes inputs and asserts
// |regrouped - original| < TOL × max(|result|, floor), with TOL chosen
// loose enough to swallow any IEEE-permissible reordering and tight enough
// to catch a dropped/added/miscoefficient term.
//
// When a function below still delegates to the original, the test passes
// trivially.  Replace the body with the actual regrouped expression and
// rerun the test before committing.
//
// Author: carlo.musolino@aei.mpg.de

#ifndef GRACE_Z4C_SUBEXPR_REGROUPED_HH
#define GRACE_Z4C_SUBEXPR_REGROUPED_HH

#include <grace/physics/z4c_subexpressions.hh>

// ---------------------------------------------------------------------------
// z4c_get_contracted_Christoffel_regrouped
//
// GammatDu^i = Γ̃^i_{jk} γ̃^{jk}  — a 6-term sum per component, currently
// emitted as a flat lex-ordered sum (line 133–135 of z4c_subexpressions.hh).
// Partner-equivariance leak target.
// ---------------------------------------------------------------------------
static void KOKKOS_INLINE_FUNCTION
z4c_get_contracted_Christoffel_regrouped(
	const double gtuu[6],
	const double Gammatudd[18],
	double (*GammatDu)[3]
)
{
	// TODO: replace with regrouped expression once unit test is in place.
	z4c_get_contracted_Christoffel(gtuu, Gammatudd, GammatDu);
}

// ---------------------------------------------------------------------------
// z4c_get_Gammatilde_rhs_regrouped
//
// dGammatu_dt^i: ~30-term flat sum per component, lines 635–637 of
// z4c_subexpressions.hh.  This is the single largest exposed surface in
// the β chain and the most plausible source of the residual β-drift.
// ---------------------------------------------------------------------------
static void KOKKOS_INLINE_FUNCTION
z4c_get_Gammatilde_rhs_regrouped(
	double alp,
	double W,
	const double Gammatu[3],
	const double Si[3],
	double kappa1,
	const double gtuu[6],
	const double Atuu[6],
	const double Gammatudd[18],
	const double GammatDu[3],
	const double dbetau_dx[9],
	const double dKhat_dx[3],
	const double dW_dx[3],
	const double dalp_dx[3],
	const double dtheta_dx[3],
	const double ddbetau_dx2[18],
	double (*dGammatu_dt)[3]
)
{
	// TODO: replace with regrouped expression once unit test is in place.
	z4c_get_Gammatilde_rhs(
		alp, W, Gammatu, Si, kappa1, gtuu, Atuu, Gammatudd, GammatDu,
		dbetau_dx, dKhat_dx, dW_dx, dalp_dx, dtheta_dx, ddbetau_dx2,
		dGammatu_dt);
}

#endif // GRACE_Z4C_SUBEXPR_REGROUPED_HH
