// Regression guard for the FD-stencil partial-sum groupings in
// include/grace/physics/fd_subexpressions.hh.
//
// The header was rewritten so that each kernel evaluates its stencil as a
// sum of antisymmetric/symmetric *pairs* (u(i+a) ± u(i-a)), making the
// partner-cell evaluation bit-exactly equal to ±(original) under IEEE-754
// double arithmetic.  This file asserts that property for every patched
// kernel.  If a future edit reorders terms in a way that breaks
// partial-sum commutativity under mirror, REQUIRE(du_p == ±du) trips.
//
// Author: carlo.musolino@aei.mpg.de

#include <catch2/catch_test_macros.hpp>
#include <grace/physics/fd_subexpressions.hh>
#include <Kokkos_Core.hpp>
#include <cstdint>

namespace {

constexpr int N = 20; // even, leaves ≥4-cell margin both halves for upw9 stencil
using HostView = Kokkos::View<double***, Kokkos::HostSpace>;

// Deterministic xorshift PRNG.  Avoids std::mt19937 to keep the test
// reproducible across stdlib versions.
struct Rng {
	std::uint64_t s;
	explicit Rng(std::uint64_t seed) : s(seed ? seed : 0xC0DEC0DEDEADBEEFull) {}
	double next() {
		s ^= s << 13; s ^= s >> 7; s ^= s << 17;
		return double(s >> 11) * (1.0 / double(1ull << 53)) - 0.5;
	}
};

// Build u with a chosen parity under reflection through the cell-pair plane
// midway between index N/2-1 and N/2 along `axis`.  Partner of i is N-1-i.
// sign = +1 → u(N-1-i,…) = +u(i,…)   (symmetric input)
// sign = -1 → u(N-1-i,…) = -u(i,…)   (antisymmetric input)
HostView make_parity(int axis, double sign, std::uint64_t seed) {
	HostView u("u", N, N, N);
	Rng rng(seed);
	for (int k = 0; k < N; ++k)
		for (int j = 0; j < N; ++j)
			for (int i = 0; i < N; ++i)
				u(i, j, k) = rng.next();
	// Overwrite the upper-half slice along `axis` with ±(lower half).
	if (axis == 0) {
		for (int k = 0; k < N; ++k)
			for (int j = 0; j < N; ++j)
				for (int i = 0; i < N / 2; ++i)
					u(N - 1 - i, j, k) = sign * u(i, j, k);
	} else if (axis == 1) {
		for (int k = 0; k < N; ++k)
			for (int j = 0; j < N / 2; ++j)
				for (int i = 0; i < N; ++i)
					u(i, N - 1 - j, k) = sign * u(i, j, k);
	} else {
		for (int k = 0; k < N / 2; ++k)
			for (int j = 0; j < N; ++j)
				for (int i = 0; i < N; ++i)
					u(i, j, N - 1 - k) = sign * u(i, j, k);
	}
	return u;
}

// Sweep sample points (i,j,k) in the "lower half" along `axis`, and at each
// pair with its mirror partner, evaluate `kernel` and require bit-exact
// equality du_partner == expected_sign * du_orig.  `stencil_w` is the
// maximum index reach of the kernel on either side of the central cell;
// `other_w` is the reach in the other two axes (≥ 0; >0 only for mixed
// derivatives).
template <typename Kernel>
void check_partner_equivariance(HostView u, int axis, int stencil_w,
                                int other_w, double expected_sign,
                                Kernel&& kernel)
{
	constexpr double invh = 1.0; // value irrelevant; we test bit-exactness
	int axis_lo = stencil_w;
	int axis_hi = N / 2 - 1;             // last index in the lower half
	int other_lo = other_w;
	int other_hi = N - 1 - other_w;
	// Two off-axis samples is enough to exercise interaction of the stencil
	// with non-trivial neighbours, while keeping the test cheap.
	int step = (other_hi - other_lo) > 0 ? (other_hi - other_lo) : 1;
	for (int s = axis_lo; s <= axis_hi; ++s) {
		for (int t = other_lo; t <= other_hi; t += step) {
			for (int v = other_lo; v <= other_hi; v += step) {
				int i, j, k, i_p, j_p, k_p;
				if (axis == 0) {
					i = s;             j = t;             k = v;
					i_p = N - 1 - s;   j_p = t;           k_p = v;
				} else if (axis == 1) {
					i = t;             j = s;             k = v;
					i_p = t;           j_p = N - 1 - s;   k_p = v;
				} else {
					i = t;             j = v;             k = s;
					i_p = t;           j_p = v;           k_p = N - 1 - s;
				}
				double du = 0.0, du_p = 0.0;
				kernel(u, i, j, k, invh, &du);
				kernel(u, i_p, j_p, k_p, invh, &du_p);
				INFO("axis=" << axis
				     << " ijk=("  << i  << ',' << j  << ',' << k  << ')'
				     << " part=(" << i_p << ',' << j_p << ',' << k_p << ')'
				     << " du="    << du << " du_p=" << du_p);
				REQUIRE(du_p == expected_sign * du);
			}
		}
	}
}

// Variant that uses a different kernel at the partner cell.  Used for the
// upwind operators, where D_upw_neg(partner) ↔ D_upw_pos(original) under
// mirror with a sign that depends on input parity:
//   sym input, antisym deriv : D_upw_neg(partner) == -D_upw_pos(orig)
//   antisym input, sym deriv : D_upw_neg(partner) == +D_upw_pos(orig)
template <typename KernelOrig, typename KernelPartner>
void check_partner_equivariance_swap(HostView u, int axis, int stencil_w,
                                     double expected_sign,
                                     KernelOrig&& kernel_orig,
                                     KernelPartner&& kernel_partner)
{
	constexpr double invh = 1.0;
	int axis_lo = stencil_w;
	int axis_hi = N / 2 - 1;
	int other_lo = stencil_w;
	int other_hi = N - 1 - stencil_w;
	int step = (other_hi - other_lo) > 0 ? (other_hi - other_lo) : 1;
	for (int s = axis_lo; s <= axis_hi; ++s) {
		for (int t = other_lo; t <= other_hi; t += step) {
			for (int v = other_lo; v <= other_hi; v += step) {
				int i, j, k, i_p, j_p, k_p;
				if (axis == 0) {
					i = s;             j = t;             k = v;
					i_p = N - 1 - s;   j_p = t;           k_p = v;
				} else if (axis == 1) {
					i = t;             j = s;             k = v;
					i_p = t;           j_p = N - 1 - s;   k_p = v;
				} else {
					i = t;             j = v;             k = s;
					i_p = t;           j_p = v;           k_p = N - 1 - s;
				}
				double du = 0.0, du_p = 0.0;
				kernel_orig   (u, i,   j,   k,   invh, &du);
				kernel_partner(u, i_p, j_p, k_p, invh, &du_p);
				INFO("axis=" << axis
				     << " ijk=("  << i   << ',' << j   << ',' << k   << ')'
				     << " part=(" << i_p << ',' << j_p << ',' << k_p << ')'
				     << " du="    << du << " du_p=" << du_p);
				REQUIRE(du_p == expected_sign * du);
			}
		}
	}
}

} // namespace

// =====================================================================
// First derivatives: 6th-order centered (fd_der_{x,y,z})
//   sym input  → antisym derivative → du_p == -du
//   antisym in → sym derivative     → du_p == +du
// =====================================================================
TEST_CASE("fd_der_*: 6th-order centered first derivative", "[fd_symmetry][fd1]")
{
	auto K_x = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x<6>(v,i,j,k,h,d); };
	auto K_y = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y<6>(v,i,j,k,h,d); };
	auto K_z = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z<6>(v,i,j,k,h,d); };

	SECTION("fd_der_x, sym")    { check_partner_equivariance(make_parity(0,+1.0,0xA001), 0, 3, 0, -1.0, K_x); }
	SECTION("fd_der_x, antisym"){ check_partner_equivariance(make_parity(0,-1.0,0xA002), 0, 3, 0, +1.0, K_x); }
	SECTION("fd_der_y, sym")    { check_partner_equivariance(make_parity(1,+1.0,0xA003), 1, 3, 0, -1.0, K_y); }
	SECTION("fd_der_y, antisym"){ check_partner_equivariance(make_parity(1,-1.0,0xA004), 1, 3, 0, +1.0, K_y); }
	SECTION("fd_der_z, sym")    { check_partner_equivariance(make_parity(2,+1.0,0xA005), 2, 3, 0, -1.0, K_z); }
	SECTION("fd_der_z, antisym"){ check_partner_equivariance(make_parity(2,-1.0,0xA006), 2, 3, 0, +1.0, K_z); }
}

// =====================================================================
// First derivatives: 4th-order centered (fd_der_4_{x,y,z})
// =====================================================================
TEST_CASE("fd_der_4_*: 4th-order centered first derivative", "[fd_symmetry][fd1]")
{
	auto K_x = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x<4>(v,i,j,k,h,d); };
	auto K_y = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y<4>(v,i,j,k,h,d); };
	auto K_z = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z<4>(v,i,j,k,h,d); };

	SECTION("fd_der_4_x, sym")    { check_partner_equivariance(make_parity(0,+1.0,0xB001), 0, 2, 0, -1.0, K_x); }
	SECTION("fd_der_4_x, antisym"){ check_partner_equivariance(make_parity(0,-1.0,0xB002), 0, 2, 0, +1.0, K_x); }
	SECTION("fd_der_4_y, sym")    { check_partner_equivariance(make_parity(1,+1.0,0xB003), 1, 2, 0, -1.0, K_y); }
	SECTION("fd_der_4_y, antisym"){ check_partner_equivariance(make_parity(1,-1.0,0xB004), 1, 2, 0, +1.0, K_y); }
	SECTION("fd_der_4_z, sym")    { check_partner_equivariance(make_parity(2,+1.0,0xB005), 2, 2, 0, -1.0, K_z); }
	SECTION("fd_der_4_z, antisym"){ check_partner_equivariance(make_parity(2,-1.0,0xB006), 2, 2, 0, +1.0, K_z); }
}

// =====================================================================
// First derivatives: 2nd-order centered (fd_der_2_{x,y,z}) — already in
// antisym-pair form (u(i+1) - u(i-1))/(2h); included as a sanity check.
// =====================================================================
TEST_CASE("fd_der_2_*: 2nd-order centered first derivative", "[fd_symmetry][fd1]")
{
	auto K_x = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x<2>(v,i,j,k,h,d); };
	auto K_y = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y<2>(v,i,j,k,h,d); };
	auto K_z = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z<2>(v,i,j,k,h,d); };

	SECTION("fd_der_2_x, sym")    { check_partner_equivariance(make_parity(0,+1.0,0xC001), 0, 1, 0, -1.0, K_x); }
	SECTION("fd_der_2_x, antisym"){ check_partner_equivariance(make_parity(0,-1.0,0xC002), 0, 1, 0, +1.0, K_x); }
	SECTION("fd_der_2_y, sym")    { check_partner_equivariance(make_parity(1,+1.0,0xC003), 1, 1, 0, -1.0, K_y); }
	SECTION("fd_der_2_y, antisym"){ check_partner_equivariance(make_parity(1,-1.0,0xC004), 1, 1, 0, +1.0, K_y); }
	SECTION("fd_der_2_z, sym")    { check_partner_equivariance(make_parity(2,+1.0,0xC005), 2, 1, 0, -1.0, K_z); }
	SECTION("fd_der_2_z, antisym"){ check_partner_equivariance(make_parity(2,-1.0,0xC006), 2, 1, 0, +1.0, K_z); }
}

// =====================================================================
// Pure second derivatives (fd_der_{xx,yy,zz})
//   sym input  → sym deriv   → du_p == +du
//   antisym in → antisym out → du_p == -du
// =====================================================================
TEST_CASE("fd_der_{xx,yy,zz}: pure 2nd derivative", "[fd_symmetry][fd2]")
{
	auto Kxx = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_xx<6>(v,i,j,k,h,d); };
	auto Kyy = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_yy<6>(v,i,j,k,h,d); };
	auto Kzz = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_zz<6>(v,i,j,k,h,d); };

	SECTION("xx, sym")    { check_partner_equivariance(make_parity(0,+1.0,0xD001), 0, 3, 0, +1.0, Kxx); }
	SECTION("xx, antisym"){ check_partner_equivariance(make_parity(0,-1.0,0xD002), 0, 3, 0, -1.0, Kxx); }
	SECTION("yy, sym")    { check_partner_equivariance(make_parity(1,+1.0,0xD003), 1, 3, 0, +1.0, Kyy); }
	SECTION("yy, antisym"){ check_partner_equivariance(make_parity(1,-1.0,0xD004), 1, 3, 0, -1.0, Kyy); }
	SECTION("zz, sym")    { check_partner_equivariance(make_parity(2,+1.0,0xD005), 2, 3, 0, +1.0, Kzz); }
	SECTION("zz, antisym"){ check_partner_equivariance(make_parity(2,-1.0,0xD006), 2, 3, 0, -1.0, Kzz); }
}

// =====================================================================
// Mixed second derivatives (fd_der_{xy,xz,yz})
// Test each kernel under a mirror of one of its two axes; assume
// u is sym/antisym only in that flipped axis (free in the other).
//   For mirror along the flipped axis with sym input  : du_p == -du
//   For mirror along the flipped axis with antisym in : du_p == +du
// =====================================================================
TEST_CASE("fd_der_{xy,xz,yz}: mixed 2nd derivative", "[fd_symmetry][fd2mix]")
{
	auto Kxy = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_xy<6>(v,i,j,k,h,d); };
	auto Kxz = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_xz<6>(v,i,j,k,h,d); };
	auto Kyz = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_yz<6>(v,i,j,k,h,d); };

	// fd_der_xy under x-mirror.
	SECTION("xy, sym-in-x")    { check_partner_equivariance(make_parity(0,+1.0,0xE001), 0, 3, 3, -1.0, Kxy); }
	SECTION("xy, antisym-in-x"){ check_partner_equivariance(make_parity(0,-1.0,0xE002), 0, 3, 3, +1.0, Kxy); }
	// fd_der_xy under y-mirror.
	SECTION("xy, sym-in-y")    { check_partner_equivariance(make_parity(1,+1.0,0xE003), 1, 3, 3, -1.0, Kxy); }
	SECTION("xy, antisym-in-y"){ check_partner_equivariance(make_parity(1,-1.0,0xE004), 1, 3, 3, +1.0, Kxy); }
	// fd_der_xz under x-mirror.
	SECTION("xz, sym-in-x")    { check_partner_equivariance(make_parity(0,+1.0,0xE005), 0, 3, 3, -1.0, Kxz); }
	SECTION("xz, antisym-in-x"){ check_partner_equivariance(make_parity(0,-1.0,0xE006), 0, 3, 3, +1.0, Kxz); }
	// fd_der_xz under z-mirror.
	SECTION("xz, sym-in-z")    { check_partner_equivariance(make_parity(2,+1.0,0xE007), 2, 3, 3, -1.0, Kxz); }
	SECTION("xz, antisym-in-z"){ check_partner_equivariance(make_parity(2,-1.0,0xE008), 2, 3, 3, +1.0, Kxz); }
	// fd_der_yz under y-mirror.
	SECTION("yz, sym-in-y")    { check_partner_equivariance(make_parity(1,+1.0,0xE009), 1, 3, 3, -1.0, Kyz); }
	SECTION("yz, antisym-in-y"){ check_partner_equivariance(make_parity(1,-1.0,0xE00A), 1, 3, 3, +1.0, Kyz); }
	// fd_der_yz under z-mirror.
	SECTION("yz, sym-in-z")    { check_partner_equivariance(make_parity(2,+1.0,0xE00B), 2, 3, 3, -1.0, Kyz); }
	SECTION("yz, antisym-in-z"){ check_partner_equivariance(make_parity(2,-1.0,0xE00C), 2, 3, 3, +1.0, Kyz); }
}

// =====================================================================
// KO dissipation operator (fd_diss_{x,y,z}) — 8th-order, behaves like an
// even-order derivative under reflection.
//   sym input  → sym output  → du_p == +du
//   antisym in → antisym out → du_p == -du
// =====================================================================
TEST_CASE("fd_diss_*: KO dissipation operator", "[fd_symmetry][fdko]")
{
	auto Dx = [](HostView v, int i, int j, int k, double h, double* d){ fd_diss_x<6>(v,i,j,k,h,d); };
	auto Dy = [](HostView v, int i, int j, int k, double h, double* d){ fd_diss_y<6>(v,i,j,k,h,d); };
	auto Dz = [](HostView v, int i, int j, int k, double h, double* d){ fd_diss_z<6>(v,i,j,k,h,d); };

	SECTION("fd_diss_x, sym")    { check_partner_equivariance(make_parity(0,+1.0,0xF001), 0, 4, 0, +1.0, Dx); }
	SECTION("fd_diss_x, antisym"){ check_partner_equivariance(make_parity(0,-1.0,0xF002), 0, 4, 0, -1.0, Dx); }
	SECTION("fd_diss_y, sym")    { check_partner_equivariance(make_parity(1,+1.0,0xF003), 1, 4, 0, +1.0, Dy); }
	SECTION("fd_diss_y, antisym"){ check_partner_equivariance(make_parity(1,-1.0,0xF004), 1, 4, 0, -1.0, Dy); }
	SECTION("fd_diss_z, sym")    { check_partner_equivariance(make_parity(2,+1.0,0xF005), 2, 4, 0, +1.0, Dz); }
	SECTION("fd_diss_z, antisym"){ check_partner_equivariance(make_parity(2,-1.0,0xF006), 2, 4, 0, -1.0, Dz); }
}

// =====================================================================
// Upwind first derivatives (fd_der_{x,y,z}_upw_{pos,neg}).
//
// The stencil is not mirror-symmetric: D_upw_neg uses (-3..+4) effectively
// while D_upw_pos uses (-4..+3).  Under partner mapping the *kernel*
// itself swaps: D_upw_neg(partner) corresponds to D_upw_pos(orig), with a
// sign set by input parity.
//
//   sym input :   D_upw_neg(partner) == -D_upw_pos(orig)
//                 D_upw_pos(partner) == -D_upw_neg(orig)
//   antisym in:   D_upw_neg(partner) == +D_upw_pos(orig)
//                 D_upw_pos(partner) == +D_upw_neg(orig)
// =====================================================================
TEST_CASE("fd_der_*_upw_*: upwind first derivative cross-pair", "[fd_symmetry][fd1upw]")
{
	auto Pxn = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_upw_neg<6>(v,i,j,k,h,d); };
	auto Pxp = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_x_upw_pos<6>(v,i,j,k,h,d); };
	auto Pyn = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y_upw_neg<6>(v,i,j,k,h,d); };
	auto Pyp = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_y_upw_pos<6>(v,i,j,k,h,d); };
	auto Pzn = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z_upw_neg<6>(v,i,j,k,h,d); };
	auto Pzp = [](HostView v, int i, int j, int k, double h, double* d){ fd_der_z_upw_pos<6>(v,i,j,k,h,d); };

	// x-axis kernels under x-mirror.
	SECTION("x_upw: sym, neg(partner) = -pos(orig)") {
		check_partner_equivariance_swap(make_parity(0,+1.0,0x1101), 0, 4, -1.0, Pxp, Pxn);
	}
	SECTION("x_upw: sym, pos(partner) = -neg(orig)") {
		check_partner_equivariance_swap(make_parity(0,+1.0,0x1102), 0, 4, -1.0, Pxn, Pxp);
	}
	SECTION("x_upw: antisym, neg(partner) = +pos(orig)") {
		check_partner_equivariance_swap(make_parity(0,-1.0,0x1103), 0, 4, +1.0, Pxp, Pxn);
	}
	SECTION("x_upw: antisym, pos(partner) = +neg(orig)") {
		check_partner_equivariance_swap(make_parity(0,-1.0,0x1104), 0, 4, +1.0, Pxn, Pxp);
	}

	// y-axis kernels under y-mirror.
	SECTION("y_upw: sym, neg(partner) = -pos(orig)") {
		check_partner_equivariance_swap(make_parity(1,+1.0,0x1201), 1, 4, -1.0, Pyp, Pyn);
	}
	SECTION("y_upw: antisym, neg(partner) = +pos(orig)") {
		check_partner_equivariance_swap(make_parity(1,-1.0,0x1202), 1, 4, +1.0, Pyp, Pyn);
	}

	// z-axis kernels under z-mirror.
	SECTION("z_upw: sym, neg(partner) = -pos(orig)") {
		check_partner_equivariance_swap(make_parity(2,+1.0,0x1301), 2, 4, -1.0, Pzp, Pzn);
	}
	SECTION("z_upw: antisym, neg(partner) = +pos(orig)") {
		check_partner_equivariance_swap(make_parity(2,-1.0,0x1302), 2, 4, +1.0, Pzp, Pzn);
	}
}
