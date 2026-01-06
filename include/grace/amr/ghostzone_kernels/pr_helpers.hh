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
 * Copyright (C) 2023 Carlo Musolino
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


        // for bi == 0 i.e. cell at x_c - dx/4 
        // we need to pick i-2 i-1 i i+1
        // whereas for bi==1 i-1 i i+1 i+2 
        int ci = bi - 2 ; 
        int cj = bj - 2 ;
        int ck = bk - 2 ; 

        size_t oi = (order+1)*bi ; 
        size_t oj = (order+1)*bj ; 
        size_t ok = (order+1)*bk ; 

        double res = 0 ; 
        #pragma unroll 16
        for (int dd = 0; dd < (order+1)*(order+1)*(order+1); ++dd) {
            int di = dd & 3;
            int dj = (dd >> 2) & 3;
            int dk = (dd >> 4);

            double coeff = coeffs(oi+di) * coeffs(oj+dj) * coeffs(ok+dk) ; 

            res += coeff * u(i+di+ci,j+dj+cj,k+dk+ck) ; 
        }
        return res ; 
    }

} ; 

struct second_order_restrict_op {
	template<typename view_t> 
    double KOKKOS_INLINE_FUNCTION
    operator() (view_t u, int i, int j, int k) const 
	{
		double res{0.0} ; 
		#pragma unroll 8 
		for( int dd=0; dd<8; ++dd) {
			int di = dd&1 ; 
			int dj = (dd>>1)&1;
			int dk = (dd>>2)&1;
			res += u(i+di,j+dj,k+dk) ; 
		}
		return 0.125*res ;
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
     */
    template<typename view_t> 
    double KOKKOS_INLINE_FUNCTION
    operator() (view_t u, int i, int j, int k) const
    {
        // first we need to determine the stencil 
        int ox = compute_offset(i,nx) ;
        int oy = compute_offset(j,ny) ;
        int oz = compute_offset(k,nz) ;

        int cx = ox-2;
        int cy = oy-2;
        int cz = oz-2;

        int off_i = (order+1)*ox ; 
        int off_j = (order+1)*oy ; 
        int off_k = (order+1)*oz ; 

        double res=0; 
        // maybe too much unrolling..
        #pragma unroll 16 
        for ( int dd=0; dd<(order+1)*(order+1)*(order+1); ++dd) {
            int di = dd & 3;          // dd % 4
            int dj = (dd >> 2) & 3;   // (dd / 4) % 4
            int dk = (dd >> 4);       // dd / 16

            double coeff = coeffs(off_i+di) * coeffs(off_j+dj) * coeffs(off_k+dk) ; 
            res += coeff * u(i+di+cx,j+dj+cy,k+dk+cz) ; 
        }

        return res ; 
    }


    KOKKOS_INLINE_FUNCTION
    int compute_offset(int pos, int n) const {
        int lb = pos - ngz;
        int ub = n + ngz - pos - 1;
        if constexpr (order==3) {
            if (lb == 0) return 2;
            if (ub == 1) return 0;
            return 1 ;
        } 
    }
} ; 





} /* namespace grace */

#endif /*GRACE_AMR_GHOSTZONE_KERNELS_PROLONG_HELPERS_HH*/