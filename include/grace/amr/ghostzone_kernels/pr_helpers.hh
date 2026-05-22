/**
 * @file high_order_pr_helpers.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Helpers for 4th order prolongation and restriction
 * @date 2026-01-02
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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
 * 
 */

#ifndef GRACE_AMR_GHOSTZONE_KERNELS_HIGH_ORDER_PR_HELPERS_HH
#define GRACE_AMR_GHOSTZONE_KERNELS_HIGH_ORDER_PR_HELPERS_HH

#include <vector>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace {

//-----------------------------------------------------------------------------
// Pair-symmetric 1D contraction.
//
// For an N-point stencil with coefficients c[0..N-1] and operands u[0..N-1],
// returns sum_{d} c[d] * u[d] but accumulated so that mirror-partner index
// pairs (0, N-1), (1, N-2), … are summed together before the partial sums
// are combined.  This is what makes the result bit-equivariant under each
// axis's discrete mirror: at the mirror partner cell, the stencil is read
// with reversed index, but the inner pair sums commute exactly under IEEE
// because the two summands literally swap.  Naive left-to-right accumulation
// instead orders the same N products differently between the two cells,
// seeding ~1 ulp drift per call.  Supported widths: N=4 (order-3 Lagrange)
// and N=5 (order-4 Lagrange).
//-----------------------------------------------------------------------------
template <int N>
KOKKOS_INLINE_FUNCTION double
contract1d_pairsym(double const c[N], double const u[N])
{
    if constexpr (N == 5) {
        double const s_outer = c[0]*u[0] + c[4]*u[4];
        double const s_inner = c[1]*u[1] + c[3]*u[3];
        double const s_ctr   = c[2]*u[2];
        return (s_outer + s_inner) + s_ctr;
    } else if constexpr (N == 4) {
        double const s_outer = c[0]*u[0] + c[3]*u[3];
        double const s_inner = c[1]*u[1] + c[2]*u[2];
        return s_outer + s_inner;
    } else {
        static_assert(N == 4 || N == 5,
            "contract1d_pairsym only supports order-3 (N=4) or order-4 (N=5)");
        return 0.0;
    }
}

template< typename lim_t >
struct slope_limited_prolong_op {
	static constexpr int low_cell_flag = -1 ; 
	static constexpr int up_cell_flag  =  1 ; 

	template<typename view_t> 
    double KOKKOS_INLINE_FUNCTION
    operator() (view_t u, int i, int j, int k, int bi, int bj, int bk) const 
	{
		lim_t limiter{} ; 
		double eta ; 
        double slopeR ; 
        double slopeL ; 
        double u_fine{0.};

		double const u0 = u(i,j,k) ; 
		eta = bi*0.25 ; 
		slopeL = u0-u(i-1,j,k) ; 
		slopeR = u(i+1,j,k)-u0 ; 
		u_fine += eta * limiter(slopeL,slopeR) ; 

		eta = bj*0.25 ; 
		slopeL = u0-u(i,j-1,k) ; 
		slopeR = u(i,j+1,k)-u0 ; 
		u_fine += eta * limiter(slopeL,slopeR) ; 

		eta = bk*0.25 ; 
		slopeL = u0-u(i,j,k-1) ; 
		slopeR = u(i,j,k+1)-u0 ; 
		u_fine += eta * limiter(slopeL,slopeR) ;

		return u0 + u_fine ; 
	}
} ; 

template< size_t order >
struct lagrange_prolong_op {

	static constexpr int low_cell_flag = 0 ; 
	static constexpr int up_cell_flag  = 1 ; 

    readonly_view_t<double> coeffs ; //!< Interp coefficients

	lagrange_prolong_op(
		Kokkos::View<double*, grace::default_space> _coeffs
	) : coeffs(_coeffs) {} 

    template<typename view_t>
    double KOKKOS_INLINE_FUNCTION
    operator() (view_t u, int i, int j, int k, int bi, int bj, int bk) const
    {
        constexpr int N = order + 1 ;

        // for bi == 0 i.e. cell at x_c - dx/4
        // we need to pick i-2 i-1 i i+1
        // whereas for bi==1 i-1 i i+1 i+2
        int ci,cj,ck ;
        if constexpr(order==3){
            ci = bi - 2 ;
            cj = bj - 2 ;
            ck = bk - 2 ;
        } else if constexpr(order==4) {
            ci=cj=ck=-2;
        }

        const size_t oi = N*bi ;
        const size_t oj = N*bj ;
        const size_t ok = N*bk ;

        // Hoist 1D coefficient slices into registers.  The two mirror-partner
        // stencils (bi=0 vs bi=1) are exact bit-reverses of each other in the
        // coeff table; combined with the pair-symmetric 1D accumulator below,
        // this gives bit-equivariant prolongation under each axis's discrete
        // mirror.  Naive left-to-right summation drifted ~1 ulp per call.
        double cx[N], cy[N], cz[N] ;
        #pragma unroll
        for (int d = 0; d < N; ++d) {
            cx[d] = coeffs(oi+d) ;
            cy[d] = coeffs(oj+d) ;
            cz[d] = coeffs(ok+d) ;
        }

        // Three nested pair-symmetric 1D contractions (N^3 fmas, same cost as
        // before).  Sx[dj][dk] = contract(cx, u(i+ci+0..N-1, j+cj+dj, k+ck+dk));
        // Sy[dk]     = contract(cy, Sx[0..N-1][dk]);
        // result     = contract(cz, Sy[0..N-1]).
        double Sx[N][N];
        #pragma unroll
        for (int dk = 0; dk < N; ++dk) {
            #pragma unroll
            for (int dj = 0; dj < N; ++dj) {
                double u_line[N];
                #pragma unroll
                for (int di = 0; di < N; ++di) {
                    u_line[di] = u(i+ci+di, j+cj+dj, k+ck+dk);
                }
                Sx[dj][dk] = contract1d_pairsym<N>(cx, u_line);
            }
        }
        double Sy[N];
        #pragma unroll
        for (int dk = 0; dk < N; ++dk) {
            double s_line[N];
            #pragma unroll
            for (int dj = 0; dj < N; ++dj) {
                s_line[dj] = Sx[dj][dk];
            }
            Sy[dk] = contract1d_pairsym<N>(cy, s_line);
        }
        return contract1d_pairsym<N>(cz, Sy);
    }

} ;

struct second_order_restrict_op {
	template<typename view_t>
    double KOKKOS_INLINE_FUNCTION
    operator() (view_t u, int i, int j, int k) const
	{
		// Pair-symmetric 8-corner average.  Pairs of fine children that are
		// mirror partners under each of π_x, π_y, π_z are summed before being
		// combined, so the result is bit-equivariant under every axis flip.
		// Naive linear dd=0..7 accumulation drifted ~1 ulp between mirror cells.
		double const q00 = u(i  ,j  ,k  ) + u(i+1,j  ,k  );  // π_x pair, (dj,dk)=(0,0)
		double const q10 = u(i  ,j+1,k  ) + u(i+1,j+1,k  );  // π_x pair, (1,0)
		double const q01 = u(i  ,j  ,k+1) + u(i+1,j  ,k+1);  // π_x pair, (0,1)
		double const q11 = u(i  ,j+1,k+1) + u(i+1,j+1,k+1);  // π_x pair, (1,1)
		double const r0  = q00 + q10;                        // π_y pair, dk=0
		double const r1  = q01 + q11;                        // π_y pair, dk=1
		return 0.125 * (r0 + r1);                            // π_z pair
	}
	// Half-flag-aware overload: the volume average is intrinsically symmetric
	// (equal weights), so the flags are ignored. Exists so callers can pass the
	// flags uniformly across all restrict op types.
	template<typename view_t>
    double KOKKOS_INLINE_FUNCTION
    operator() (view_t u, int i, int j, int k, int, int, int) const
	{
		return (*this)(u, i, j, k);
	}
} ;

template< size_t order >
struct lagrange_restrict_op {

    enum stencil_repr_t : int {
        L2=0, L1, CENTER, R1, R2
    }  ;

    readonly_view_t<double> coeffs ; //!< Interp coefficients
    int nx,ny,nz,ngz; //!< Number of cells and ghostzones

	lagrange_restrict_op(
		Kokkos::View<double*, grace::default_space> _coeffs,
		int _nx, int _ny, int _nz, int _ngz 
	) : coeffs(_coeffs), nx(_nx), ny(_ny), nz(_nz), ngz(_ngz) {}

    /**
     * @brief Returns 4th order accurate interpolation at coarse cell center
     * @param u Fine data view
     * @param i Index of fine cell x_0-h/4 where x_0 is target point
     * @param j Index of fine cell y_0-h/4 where y_0 is target point
     * @param k Index of fine cell z_0-h/4 where z_0 is target point
     * @param half_x 1 if the target coarse cell lies in the UPPER half of the
     *               coarse parent quad along x (mirrors stencil bias); 0 otherwise
     * @param half_y same convention along y
     * @param half_z same convention along z
     *
     * The interior 5-pt Lagrange stencil is intrinsically biased (the target
     * x_0 doesn't sit on a fine cell center). Using the same biased stencil
     * everywhere produces a systematic O(h_f^5) left/right asymmetry: each
     * coarse cell pulls samples from one fixed direction. To restore the
     * mirror symmetry of a coarse parent quad, the lower half uses the
     * right-biased CENTER stencil (offsets {-1.5,...,+2.5}h_f) and the upper
     * half uses the left-biased L1 stencil (offsets {-2.5,...,+1.5}h_f).
     * Together the two halves are mirror images about the parent center.
     */
    template<typename view_t>
    double KOKKOS_INLINE_FUNCTION
    operator() (view_t u, int i, int j, int k,
                int half_x = 0, int half_y = 0, int half_z = 0) const
    {
        constexpr int N = order + 1;

        // Per-axis stencil selection (CENTER/L1/L2/R1/R2 depending on edge
        // proximity and half-flag). The two stencils picked by the two
        // mirror-partner halves are exact bit-reverses of each other in the
        // coeff table, so combined with the pair-symmetric 1D accumulator
        // below the restriction is bit-equivariant under each axis's mirror.
        int const ox = compute_offset(i,nx,half_x);
        int const oy = compute_offset(j,ny,half_y);
        int const oz = compute_offset(k,nz,half_z);
        int cx_off, cy_off, cz_off;
        if constexpr (order==3) {
            cx_off = ox-2;  cy_off = oy-2;  cz_off = oz-2;
        } else if constexpr (order==4) {
            cx_off = ox-3;  cy_off = oy-3;  cz_off = oz-3;
        }

        // Hoist the three 1D coefficient slices into registers.
        double cx[N], cy[N], cz[N];
        #pragma unroll
        for (int d = 0; d < N; ++d) {
            cx[d] = coeffs(N*ox + d);
            cy[d] = coeffs(N*oy + d);
            cz[d] = coeffs(N*oz + d);
        }

        // Three nested pair-symmetric 1D contractions, same structure as
        // lagrange_prolong_op's rewrite.  Sx[dj][dk] = contract(cx, u_line);
        // Sy[dk] = contract(cy, Sx[*][dk]); result = contract(cz, Sy).
        double Sx[N][N];
        #pragma unroll
        for (int dk = 0; dk < N; ++dk) {
            #pragma unroll
            for (int dj = 0; dj < N; ++dj) {
                double u_line[N];
                #pragma unroll
                for (int di = 0; di < N; ++di) {
                    u_line[di] = u(i+di+cx_off, j+dj+cy_off, k+dk+cz_off);
                }
                Sx[dj][dk] = contract1d_pairsym<N>(cx, u_line);
            }
        }
        double Sy[N];
        #pragma unroll
        for (int dk = 0; dk < N; ++dk) {
            double s_line[N];
            #pragma unroll
            for (int dj = 0; dj < N; ++dj) {
                s_line[dj] = Sx[dj][dk];
            }
            Sy[dk] = contract1d_pairsym<N>(cy, s_line);
        }
        return contract1d_pairsym<N>(cz, Sy);
    }


    KOKKOS_INLINE_FUNCTION
    int compute_offset(int pos, int n, int half = 0) const {
        int lb = pos - ngz;
        int ub = n + ngz - pos - 1;
        if constexpr (order==3) {
            if (lb == 0) return 2;
            if (ub == 1) return 0;
            return 1 ;
        } else if constexpr (order==4) {
            // boundary shifts: mirror-symmetric pair, always applied at edges
            if (ub==1) return 0 ;   // L2 (cx=-3, all-left): rightmost interior cell
            if (lb==0) return 3 ;   // R2 (cx=0,  all-right): leftmost interior cell
            if (ub==2) return 1 ;   // L1 (cx=-2): one cell from right edge
            // interior: flip bias based on which half of the parent the target sits in
            //   lower half → CENTER (cx=-1, offsets {-1.5..+2.5}h_f), right-biased
            //   upper half → L1     (cx=-2, offsets {-2.5..+1.5}h_f), left-biased
            // mirror about the parent-quad midplane; preserves L↔R symmetry.
            return half ? 1 : 2 ;
        }
    }
} ; 





} /* namespace grace */

#endif /*GRACE_AMR_GHOSTZONE_KERNELS_PROLONG_HELPERS_HH*/