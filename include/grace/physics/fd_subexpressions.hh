
/****************************************************************************/
/*                      FD helpers, SymPy generated                         */
/****************************************************************************/
#ifndef GRACE_FD_SUBEXPR_HH
#define GRACE_FD_SUBEXPR_HH

#include <Kokkos_Core.hpp>

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_x_l1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((2.0/5.0)*u(i+1,j,k) - 1.0/30.0*u(i+2,j,k) + (7.0/12.0)*u(i,j,k) - 4.0/3.0*u(i-1,j,k) + (1.0/2.0)*u(i-2,j,k) - 2.0/15.0*u(i-3,j,k) + (1.0/60.0)*u(i-4,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_x(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((3.0/4.0)*u(i+1,j,k) - 3.0/20.0*u(i+2,j,k) + (1.0/60.0)*u(i+3,j,k) - 3.0/4.0*u(i-1,j,k) + (3.0/20.0)*u(i-2,j,k) - 1.0/60.0*u(i-3,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_x_r1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((4.0/3.0)*u(i+1,j,k) - 1.0/2.0*u(i+2,j,k) + (2.0/15.0)*u(i+3,j,k) - 1.0/60.0*u(i+4,j,k) - 7.0/12.0*u(i,j,k) - 2.0/5.0*u(i-1,j,k) + (1.0/30.0)*u(i-2,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_y_l1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((2.0/5.0)*u(i,j+1,k) - 1.0/30.0*u(i,j+2,k) + (7.0/12.0)*u(i,j,k) - 4.0/3.0*u(i,j-1,k) + (1.0/2.0)*u(i,j-2,k) - 2.0/15.0*u(i,j-3,k) + (1.0/60.0)*u(i,j-4,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_y(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((3.0/4.0)*u(i,j+1,k) - 3.0/20.0*u(i,j+2,k) + (1.0/60.0)*u(i,j+3,k) - 3.0/4.0*u(i,j-1,k) + (3.0/20.0)*u(i,j-2,k) - 1.0/60.0*u(i,j-3,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_y_r1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((4.0/3.0)*u(i,j+1,k) - 1.0/2.0*u(i,j+2,k) + (2.0/15.0)*u(i,j+3,k) - 1.0/60.0*u(i,j+4,k) - 7.0/12.0*u(i,j,k) - 2.0/5.0*u(i,j-1,k) + (1.0/30.0)*u(i,j-2,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_z_l1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((7.0/12.0)*u(i,j,k) + (2.0/5.0)*u(i,j,k+1) - 1.0/30.0*u(i,j,k+2) - 4.0/3.0*u(i,j,k-1) + (1.0/2.0)*u(i,j,k-2) - 2.0/15.0*u(i,j,k-3) + (1.0/60.0)*u(i,j,k-4));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_z(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((3.0/4.0)*u(i,j,k+1) - 3.0/20.0*u(i,j,k+2) + (1.0/60.0)*u(i,j,k+3) - 3.0/4.0*u(i,j,k-1) + (3.0/20.0)*u(i,j,k-2) - 1.0/60.0*u(i,j,k-3));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_z_r1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(-7.0/12.0*u(i,j,k) + (4.0/3.0)*u(i,j,k+1) - 1.0/2.0*u(i,j,k+2) + (2.0/15.0)*u(i,j,k+3) - 1.0/60.0*u(i,j,k+4) - 2.0/5.0*u(i,j,k-1) + (1.0/30.0)*u(i,j,k-2));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_4_x(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((2.0/3.0)*u(i+1,j,k) - 1.0/12.0*u(i+2,j,k) - 2.0/3.0*u(i-1,j,k) + (1.0/12.0)*u(i-2,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_4_y(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((2.0/3.0)*u(i,j+1,k) - 1.0/12.0*u(i,j+2,k) - 2.0/3.0*u(i,j-1,k) + (1.0/12.0)*u(i,j-2,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_4_z(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((2.0/3.0)*u(i,j,k+1) - 1.0/12.0*u(i,j,k+2) - 2.0/3.0*u(i,j,k-1) + (1.0/12.0)*u(i,j,k-2));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_2_x_l1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((3.0/2.0)*u(i,j,k) - 2*u(i-1,j,k) + (1.0/2.0)*u(i-2,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_2_x(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((1.0/2.0)*u(i+1,j,k) - 1.0/2.0*u(i-1,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_2_x_r1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(2*u(i+1,j,k) - 1.0/2.0*u(i+2,j,k) - 3.0/2.0*u(i,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_2_y_l1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((3.0/2.0)*u(i,j,k) - 2*u(i,j-1,k) + (1.0/2.0)*u(i,j-2,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_2_y(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((1.0/2.0)*u(i,j+1,k) - 1.0/2.0*u(i,j-1,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_2_y_r1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(2*u(i,j+1,k) - 1.0/2.0*u(i,j+2,k) - 3.0/2.0*u(i,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_2_z_l1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((3.0/2.0)*u(i,j,k) - 2*u(i,j,k-1) + (1.0/2.0)*u(i,j,k-2));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_2_z(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((1.0/2.0)*u(i,j,k+1) - 1.0/2.0*u(i,j,k-1));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_2_z_r1(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(-3.0/2.0*u(i,j,k) + 2*u(i,j,k+1) - 1.0/2.0*u(i,j,k+2));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_xx(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = ((invh)*(invh))*((3.0/2.0)*u(i+1,j,k) - 3.0/20.0*u(i+2,j,k) + (1.0/90.0)*u(i+3,j,k) - 49.0/18.0*u(i,j,k) + (3.0/2.0)*u(i-1,j,k) - 3.0/20.0*u(i-2,j,k) + (1.0/90.0)*u(i-3,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_yy(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = ((invh)*(invh))*((3.0/2.0)*u(i,j+1,k) - 3.0/20.0*u(i,j+2,k) + (1.0/90.0)*u(i,j+3,k) - 49.0/18.0*u(i,j,k) + (3.0/2.0)*u(i,j-1,k) - 3.0/20.0*u(i,j-2,k) + (1.0/90.0)*u(i,j-3,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_zz(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = ((invh)*(invh))*(-49.0/18.0*u(i,j,k) + (3.0/2.0)*u(i,j,k+1) - 3.0/20.0*u(i,j,k+2) + (1.0/90.0)*u(i,j,k+3) + (3.0/2.0)*u(i,j,k-1) - 3.0/20.0*u(i,j,k-2) + (1.0/90.0)*u(i,j,k-3));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_xy(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = ((invh)*(invh))*((9.0/16.0)*u(i+1,j+1,k) - 9.0/80.0*u(i+1,j+2,k) + (1.0/80.0)*u(i+1,j+3,k) - 9.0/16.0*u(i+1,j-1,k) + (9.0/80.0)*u(i+1,j-2,k) - 1.0/80.0*u(i+1,j-3,k) - 9.0/80.0*u(i+2,j+1,k) + (9.0/400.0)*u(i+2,j+2,k) - 1.0/400.0*u(i+2,j+3,k) + (9.0/80.0)*u(i+2,j-1,k) - 9.0/400.0*u(i+2,j-2,k) + (1.0/400.0)*u(i+2,j-3,k) + (1.0/80.0)*u(i+3,j+1,k) - 1.0/400.0*u(i+3,j+2,k) + (1.0/3600.0)*u(i+3,j+3,k) - 1.0/80.0*u(i+3,j-1,k) + (1.0/400.0)*u(i+3,j-2,k) - 1.0/3600.0*u(i+3,j-3,k) - 9.0/16.0*u(i-1,j+1,k) + (9.0/80.0)*u(i-1,j+2,k) - 1.0/80.0*u(i-1,j+3,k) + (9.0/16.0)*u(i-1,j-1,k) - 9.0/80.0*u(i-1,j-2,k) + (1.0/80.0)*u(i-1,j-3,k) + (9.0/80.0)*u(i-2,j+1,k) - 9.0/400.0*u(i-2,j+2,k) + (1.0/400.0)*u(i-2,j+3,k) - 9.0/80.0*u(i-2,j-1,k) + (9.0/400.0)*u(i-2,j-2,k) - 1.0/400.0*u(i-2,j-3,k) - 1.0/80.0*u(i-3,j+1,k) + (1.0/400.0)*u(i-3,j+2,k) - 1.0/3600.0*u(i-3,j+3,k) + (1.0/80.0)*u(i-3,j-1,k) - 1.0/400.0*u(i-3,j-2,k) + (1.0/3600.0)*u(i-3,j-3,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_xz(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = ((invh)*(invh))*((9.0/16.0)*u(i+1,j,k+1) - 9.0/80.0*u(i+1,j,k+2) + (1.0/80.0)*u(i+1,j,k+3) - 9.0/16.0*u(i+1,j,k-1) + (9.0/80.0)*u(i+1,j,k-2) - 1.0/80.0*u(i+1,j,k-3) - 9.0/80.0*u(i+2,j,k+1) + (9.0/400.0)*u(i+2,j,k+2) - 1.0/400.0*u(i+2,j,k+3) + (9.0/80.0)*u(i+2,j,k-1) - 9.0/400.0*u(i+2,j,k-2) + (1.0/400.0)*u(i+2,j,k-3) + (1.0/80.0)*u(i+3,j,k+1) - 1.0/400.0*u(i+3,j,k+2) + (1.0/3600.0)*u(i+3,j,k+3) - 1.0/80.0*u(i+3,j,k-1) + (1.0/400.0)*u(i+3,j,k-2) - 1.0/3600.0*u(i+3,j,k-3) - 9.0/16.0*u(i-1,j,k+1) + (9.0/80.0)*u(i-1,j,k+2) - 1.0/80.0*u(i-1,j,k+3) + (9.0/16.0)*u(i-1,j,k-1) - 9.0/80.0*u(i-1,j,k-2) + (1.0/80.0)*u(i-1,j,k-3) + (9.0/80.0)*u(i-2,j,k+1) - 9.0/400.0*u(i-2,j,k+2) + (1.0/400.0)*u(i-2,j,k+3) - 9.0/80.0*u(i-2,j,k-1) + (9.0/400.0)*u(i-2,j,k-2) - 1.0/400.0*u(i-2,j,k-3) - 1.0/80.0*u(i-3,j,k+1) + (1.0/400.0)*u(i-3,j,k+2) - 1.0/3600.0*u(i-3,j,k+3) + (1.0/80.0)*u(i-3,j,k-1) - 1.0/400.0*u(i-3,j,k-2) + (1.0/3600.0)*u(i-3,j,k-3));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_yz(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = ((invh)*(invh))*((9.0/16.0)*u(i,j+1,k+1) - 9.0/80.0*u(i,j+1,k+2) + (1.0/80.0)*u(i,j+1,k+3) - 9.0/16.0*u(i,j+1,k-1) + (9.0/80.0)*u(i,j+1,k-2) - 1.0/80.0*u(i,j+1,k-3) - 9.0/80.0*u(i,j+2,k+1) + (9.0/400.0)*u(i,j+2,k+2) - 1.0/400.0*u(i,j+2,k+3) + (9.0/80.0)*u(i,j+2,k-1) - 9.0/400.0*u(i,j+2,k-2) + (1.0/400.0)*u(i,j+2,k-3) + (1.0/80.0)*u(i,j+3,k+1) - 1.0/400.0*u(i,j+3,k+2) + (1.0/3600.0)*u(i,j+3,k+3) - 1.0/80.0*u(i,j+3,k-1) + (1.0/400.0)*u(i,j+3,k-2) - 1.0/3600.0*u(i,j+3,k-3) - 9.0/16.0*u(i,j-1,k+1) + (9.0/80.0)*u(i,j-1,k+2) - 1.0/80.0*u(i,j-1,k+3) + (9.0/16.0)*u(i,j-1,k-1) - 9.0/80.0*u(i,j-1,k-2) + (1.0/80.0)*u(i,j-1,k-3) + (9.0/80.0)*u(i,j-2,k+1) - 9.0/400.0*u(i,j-2,k+2) + (1.0/400.0)*u(i,j-2,k+3) - 9.0/80.0*u(i,j-2,k-1) + (9.0/400.0)*u(i,j-2,k-2) - 1.0/400.0*u(i,j-2,k-3) - 1.0/80.0*u(i,j-3,k+1) + (1.0/400.0)*u(i,j-3,k+2) - 1.0/3600.0*u(i,j-3,k+3) + (1.0/80.0)*u(i,j-3,k-1) - 1.0/400.0*u(i,j-3,k-2) + (1.0/3600.0)*u(i,j-3,k-3));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_x_upw_neg(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((2.0/5.0)*u(i+1,j,k) - 1.0/30.0*u(i+2,j,k) + (7.0/12.0)*u(i,j,k) - 4.0/3.0*u(i-1,j,k) + (1.0/2.0)*u(i-2,j,k) - 2.0/15.0*u(i-3,j,k) + (1.0/60.0)*u(i-4,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_x_upw_pos(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((4.0/3.0)*u(i+1,j,k) - 1.0/2.0*u(i+2,j,k) + (2.0/15.0)*u(i+3,j,k) - 1.0/60.0*u(i+4,j,k) - 7.0/12.0*u(i,j,k) - 2.0/5.0*u(i-1,j,k) + (1.0/30.0)*u(i-2,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_y_upw_neg(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((2.0/5.0)*u(i,j+1,k) - 1.0/30.0*u(i,j+2,k) + (7.0/12.0)*u(i,j,k) - 4.0/3.0*u(i,j-1,k) + (1.0/2.0)*u(i,j-2,k) - 2.0/15.0*u(i,j-3,k) + (1.0/60.0)*u(i,j-4,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_y_upw_pos(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((4.0/3.0)*u(i,j+1,k) - 1.0/2.0*u(i,j+2,k) + (2.0/15.0)*u(i,j+3,k) - 1.0/60.0*u(i,j+4,k) - 7.0/12.0*u(i,j,k) - 2.0/5.0*u(i,j-1,k) + (1.0/30.0)*u(i,j-2,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_z_upw_neg(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((7.0/12.0)*u(i,j,k) + (2.0/5.0)*u(i,j,k+1) - 1.0/30.0*u(i,j,k+2) - 4.0/3.0*u(i,j,k-1) + (1.0/2.0)*u(i,j,k-2) - 2.0/15.0*u(i,j,k-3) + (1.0/60.0)*u(i,j,k-4));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_z_upw_pos(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(-7.0/12.0*u(i,j,k) + (4.0/3.0)*u(i,j,k+1) - 1.0/2.0*u(i,j,k+2) + (2.0/15.0)*u(i,j,k+3) - 1.0/60.0*u(i,j,k+4) - 2.0/5.0*u(i,j,k-1) + (1.0/30.0)*u(i,j,k-2));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_diss_x(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(-56*u(i+1,j,k) + 28*u(i+2,j,k) - 8*u(i+3,j,k) + u(i+4,j,k) + 70*u(i,j,k) - 56*u(i-1,j,k) + 28*u(i-2,j,k) - 8*u(i-3,j,k) + u(i-4,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_diss_y(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(-56*u(i,j+1,k) + 28*u(i,j+2,k) - 8*u(i,j+3,k) + u(i,j+4,k) + 70*u(i,j,k) - 56*u(i,j-1,k) + 28*u(i,j-2,k) - 8*u(i,j-3,k) + u(i,j-4,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_diss_z(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(70*u(i,j,k) - 56*u(i,j,k+1) + 28*u(i,j,k+2) - 8*u(i,j,k+3) + u(i,j,k+4) - 56*u(i,j,k-1) + 28*u(i,j,k-2) - 8*u(i,j,k-3) + u(i,j,k-4));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_scalar_4(view_t state, int i, int j, int k, int iv, int q, double d[3], double invh)
{
	using namespace Kokkos;
	auto u = subview(state,ALL(),ALL(),ALL(),iv,q);
	fd_der_4_x(u,i,j,k,invh,&(d[0]));
	fd_der_4_y(u,i,j,k,invh,&(d[1]));
	fd_der_4_z(u,i,j,k,invh,&(d[2]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_vector_4(view_t state, int i, int j, int k, int iv, int q, double d[9], double invh)
{
	using namespace Kokkos;
	auto ux = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	fd_der_4_x(ux,i,j,k,invh,&(d[0]));
	fd_der_4_x(uy,i,j,k,invh,&(d[1]));
	fd_der_4_x(uz,i,j,k,invh,&(d[2]));
	fd_der_4_y(ux,i,j,k,invh,&(d[3]));
	fd_der_4_y(uy,i,j,k,invh,&(d[4]));
	fd_der_4_y(uz,i,j,k,invh,&(d[5]));
	fd_der_4_z(ux,i,j,k,invh,&(d[6]));
	fd_der_4_z(uy,i,j,k,invh,&(d[7]));
	fd_der_4_z(uz,i,j,k,invh,&(d[8]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_tensor_4(view_t state, int i, int j, int k, int iv, int q, double d[18], double invh)
{
	using namespace Kokkos;
	auto uxx = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uxy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uxz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	auto uyy = subview(state,ALL(),ALL(),ALL(),iv+3,q);
	auto uyz = subview(state,ALL(),ALL(),ALL(),iv+4,q);
	auto uzz = subview(state,ALL(),ALL(),ALL(),iv+5,q);
	fd_der_4_x(uxx,i,j,k,invh,&(d[0]));
	fd_der_4_x(uxy,i,j,k,invh,&(d[1]));
	fd_der_4_x(uxz,i,j,k,invh,&(d[2]));
	fd_der_4_x(uyy,i,j,k,invh,&(d[3]));
	fd_der_4_x(uyz,i,j,k,invh,&(d[4]));
	fd_der_4_x(uzz,i,j,k,invh,&(d[5]));
	fd_der_4_y(uxx,i,j,k,invh,&(d[6]));
	fd_der_4_y(uxy,i,j,k,invh,&(d[7]));
	fd_der_4_y(uxz,i,j,k,invh,&(d[8]));
	fd_der_4_y(uyy,i,j,k,invh,&(d[9]));
	fd_der_4_y(uyz,i,j,k,invh,&(d[10]));
	fd_der_4_y(uzz,i,j,k,invh,&(d[11]));
	fd_der_4_z(uxx,i,j,k,invh,&(d[12]));
	fd_der_4_z(uxy,i,j,k,invh,&(d[13]));
	fd_der_4_z(uxz,i,j,k,invh,&(d[14]));
	fd_der_4_z(uyy,i,j,k,invh,&(d[15]));
	fd_der_4_z(uyz,i,j,k,invh,&(d[16]));
	fd_der_4_z(uzz,i,j,k,invh,&(d[17]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_scalar(view_t state, int i, int j, int k, int iv, int q, double d[3], double invh)
{
	using namespace Kokkos;
	auto u = subview(state,ALL(),ALL(),ALL(),iv,q);
	fd_der_x(u,i,j,k,invh,&(d[0]));
	fd_der_y(u,i,j,k,invh,&(d[1]));
	fd_der_z(u,i,j,k,invh,&(d[2]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_vector(view_t state, int i, int j, int k, int iv, int q, double d[9], double invh)
{
	using namespace Kokkos;
	auto ux = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	fd_der_x(ux,i,j,k,invh,&(d[0]));
	fd_der_x(uy,i,j,k,invh,&(d[1]));
	fd_der_x(uz,i,j,k,invh,&(d[2]));
	fd_der_y(ux,i,j,k,invh,&(d[3]));
	fd_der_y(uy,i,j,k,invh,&(d[4]));
	fd_der_y(uz,i,j,k,invh,&(d[5]));
	fd_der_z(ux,i,j,k,invh,&(d[6]));
	fd_der_z(uy,i,j,k,invh,&(d[7]));
	fd_der_z(uz,i,j,k,invh,&(d[8]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_tensor(view_t state, int i, int j, int k, int iv, int q, double d[18], double invh)
{
	using namespace Kokkos;
	auto uxx = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uxy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uxz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	auto uyy = subview(state,ALL(),ALL(),ALL(),iv+3,q);
	auto uyz = subview(state,ALL(),ALL(),ALL(),iv+4,q);
	auto uzz = subview(state,ALL(),ALL(),ALL(),iv+5,q);
	fd_der_x(uxx,i,j,k,invh,&(d[0]));
	fd_der_x(uxy,i,j,k,invh,&(d[1]));
	fd_der_x(uxz,i,j,k,invh,&(d[2]));
	fd_der_x(uyy,i,j,k,invh,&(d[3]));
	fd_der_x(uyz,i,j,k,invh,&(d[4]));
	fd_der_x(uzz,i,j,k,invh,&(d[5]));
	fd_der_y(uxx,i,j,k,invh,&(d[6]));
	fd_der_y(uxy,i,j,k,invh,&(d[7]));
	fd_der_y(uxz,i,j,k,invh,&(d[8]));
	fd_der_y(uyy,i,j,k,invh,&(d[9]));
	fd_der_y(uyz,i,j,k,invh,&(d[10]));
	fd_der_y(uzz,i,j,k,invh,&(d[11]));
	fd_der_z(uxx,i,j,k,invh,&(d[12]));
	fd_der_z(uxy,i,j,k,invh,&(d[13]));
	fd_der_z(uxz,i,j,k,invh,&(d[14]));
	fd_der_z(uyy,i,j,k,invh,&(d[15]));
	fd_der_z(uyz,i,j,k,invh,&(d[16]));
	fd_der_z(uzz,i,j,k,invh,&(d[17]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_scalar_upw(view_t state, int i, int j, int k, int iv, int q, double * d, double v[3], double invh)
{
	using namespace Kokkos;
	auto u = subview(state,ALL(),ALL(),ALL(),iv,q);
	(*d) = 0 ;
	double dl;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(u,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(u,i,j,k,invh,&(dl));
	}
	(*d) += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(u,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(u,i,j,k,invh,&(dl));
	}
	(*d) += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(u,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(u,i,j,k,invh,&(dl));
	}
	(*d) += dl * v[2];
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_vector_upw(view_t state, int i, int j, int k, int iv, int q, double d[3], double v[3], double invh)
{
	using namespace Kokkos;
	auto ux = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	double dl;
	d[0] = 0;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(ux,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(ux,i,j,k,invh,&(dl));
	}
	d[0] += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(ux,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(ux,i,j,k,invh,&(dl));
	}
	d[0] += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(ux,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(ux,i,j,k,invh,&(dl));
	}
	d[0] += dl * v[2];
	d[1] = 0;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(uy,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uy,i,j,k,invh,&(dl));
	}
	d[1] += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(uy,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uy,i,j,k,invh,&(dl));
	}
	d[1] += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(uy,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uy,i,j,k,invh,&(dl));
	}
	d[1] += dl * v[2];
	d[2] = 0;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(uz,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uz,i,j,k,invh,&(dl));
	}
	d[2] += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(uz,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uz,i,j,k,invh,&(dl));
	}
	d[2] += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(uz,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uz,i,j,k,invh,&(dl));
	}
	d[2] += dl * v[2];
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_tensor_upw(view_t state, int i, int j, int k, int iv, int q, double d[6], double v[3], double invh)
{
	using namespace Kokkos;
	auto uxx = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uxy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uxz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	auto uyy = subview(state,ALL(),ALL(),ALL(),iv+3,q);
	auto uyz = subview(state,ALL(),ALL(),ALL(),iv+4,q);
	auto uzz = subview(state,ALL(),ALL(),ALL(),iv+5,q);
	double dl;
	d[0] = 0;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(uxx,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uxx,i,j,k,invh,&(dl));
	}
	d[0] += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(uxx,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uxx,i,j,k,invh,&(dl));
	}
	d[0] += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(uxx,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uxx,i,j,k,invh,&(dl));
	}
	d[0] += dl * v[2];
	d[1] = 0;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(uxy,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uxy,i,j,k,invh,&(dl));
	}
	d[1] += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(uxy,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uxy,i,j,k,invh,&(dl));
	}
	d[1] += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(uxy,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uxy,i,j,k,invh,&(dl));
	}
	d[1] += dl * v[2];
	d[2] = 0;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(uxz,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uxz,i,j,k,invh,&(dl));
	}
	d[2] += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(uxz,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uxz,i,j,k,invh,&(dl));
	}
	d[2] += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(uxz,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uxz,i,j,k,invh,&(dl));
	}
	d[2] += dl * v[2];
	d[3] = 0;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(uyy,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uyy,i,j,k,invh,&(dl));
	}
	d[3] += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(uyy,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uyy,i,j,k,invh,&(dl));
	}
	d[3] += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(uyy,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uyy,i,j,k,invh,&(dl));
	}
	d[3] += dl * v[2];
	d[4] = 0;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(uyz,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uyz,i,j,k,invh,&(dl));
	}
	d[4] += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(uyz,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uyz,i,j,k,invh,&(dl));
	}
	d[4] += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(uyz,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uyz,i,j,k,invh,&(dl));
	}
	d[4] += dl * v[2];
	d[5] = 0;
	dl=0.0;
	if (v[0]>0){
		fd_der_x_upw_pos(uzz,i,j,k,invh,&(dl));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uzz,i,j,k,invh,&(dl));
	}
	d[5] += dl * v[0];
	dl=0.0;
	if (v[1]>0){
		fd_der_y_upw_pos(uzz,i,j,k,invh,&(dl));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uzz,i,j,k,invh,&(dl));
	}
	d[5] += dl * v[1];
	dl=0.0;
	if (v[2]>0){
		fd_der_z_upw_pos(uzz,i,j,k,invh,&(dl));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uzz,i,j,k,invh,&(dl));
	}
	d[5] += dl * v[2];
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_second_deriv_scalar(view_t state, int i, int j, int k, int iv, int q, double d[6], double invh)
{
	using namespace Kokkos;
	auto u = subview(state,ALL(),ALL(),ALL(),iv,q);
	fd_der_xx(u,i,j,k,invh,&(d[0]));
	fd_der_xy(u,i,j,k,invh,&(d[1]));
	fd_der_xz(u,i,j,k,invh,&(d[2]));
	fd_der_yy(u,i,j,k,invh,&(d[3]));
	fd_der_yz(u,i,j,k,invh,&(d[4]));
	fd_der_zz(u,i,j,k,invh,&(d[5]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_second_deriv_vector(view_t state, int i, int j, int k, int iv, int q, double d[18], double invh)
{
	using namespace Kokkos;
	auto ux = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	fd_der_xx(ux,i,j,k,invh,&(d[0]));
	fd_der_xx(uy,i,j,k,invh,&(d[1]));
	fd_der_xx(uz,i,j,k,invh,&(d[2]));
	fd_der_xy(ux,i,j,k,invh,&(d[3]));
	fd_der_xy(uy,i,j,k,invh,&(d[4]));
	fd_der_xy(uz,i,j,k,invh,&(d[5]));
	fd_der_xz(ux,i,j,k,invh,&(d[6]));
	fd_der_xz(uy,i,j,k,invh,&(d[7]));
	fd_der_xz(uz,i,j,k,invh,&(d[8]));
	fd_der_yy(ux,i,j,k,invh,&(d[9]));
	fd_der_yy(uy,i,j,k,invh,&(d[10]));
	fd_der_yy(uz,i,j,k,invh,&(d[11]));
	fd_der_yz(ux,i,j,k,invh,&(d[12]));
	fd_der_yz(uy,i,j,k,invh,&(d[13]));
	fd_der_yz(uz,i,j,k,invh,&(d[14]));
	fd_der_zz(ux,i,j,k,invh,&(d[15]));
	fd_der_zz(uy,i,j,k,invh,&(d[16]));
	fd_der_zz(uz,i,j,k,invh,&(d[17]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_second_deriv_tensor(view_t state, int i, int j, int k, int iv, int q, double d[36], double invh)
{
	using namespace Kokkos;
	auto uxx = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uxy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uxz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	auto uyy = subview(state,ALL(),ALL(),ALL(),iv+3,q);
	auto uyz = subview(state,ALL(),ALL(),ALL(),iv+4,q);
	auto uzz = subview(state,ALL(),ALL(),ALL(),iv+5,q);
	fd_der_xx(uxx,i,j,k,invh,&(d[0]));
	fd_der_xx(uxy,i,j,k,invh,&(d[1]));
	fd_der_xx(uxz,i,j,k,invh,&(d[2]));
	fd_der_xx(uyy,i,j,k,invh,&(d[3]));
	fd_der_xx(uyz,i,j,k,invh,&(d[4]));
	fd_der_xx(uzz,i,j,k,invh,&(d[5]));
	fd_der_xy(uxx,i,j,k,invh,&(d[6]));
	fd_der_xy(uxy,i,j,k,invh,&(d[7]));
	fd_der_xy(uxz,i,j,k,invh,&(d[8]));
	fd_der_xy(uyy,i,j,k,invh,&(d[9]));
	fd_der_xy(uyz,i,j,k,invh,&(d[10]));
	fd_der_xy(uzz,i,j,k,invh,&(d[11]));
	fd_der_xz(uxx,i,j,k,invh,&(d[12]));
	fd_der_xz(uxy,i,j,k,invh,&(d[13]));
	fd_der_xz(uxz,i,j,k,invh,&(d[14]));
	fd_der_xz(uyy,i,j,k,invh,&(d[15]));
	fd_der_xz(uyz,i,j,k,invh,&(d[16]));
	fd_der_xz(uzz,i,j,k,invh,&(d[17]));
	fd_der_yy(uxx,i,j,k,invh,&(d[18]));
	fd_der_yy(uxy,i,j,k,invh,&(d[19]));
	fd_der_yy(uxz,i,j,k,invh,&(d[20]));
	fd_der_yy(uyy,i,j,k,invh,&(d[21]));
	fd_der_yy(uyz,i,j,k,invh,&(d[22]));
	fd_der_yy(uzz,i,j,k,invh,&(d[23]));
	fd_der_yz(uxx,i,j,k,invh,&(d[24]));
	fd_der_yz(uxy,i,j,k,invh,&(d[25]));
	fd_der_yz(uxz,i,j,k,invh,&(d[26]));
	fd_der_yz(uyy,i,j,k,invh,&(d[27]));
	fd_der_yz(uyz,i,j,k,invh,&(d[28]));
	fd_der_yz(uzz,i,j,k,invh,&(d[29]));
	fd_der_zz(uxx,i,j,k,invh,&(d[30]));
	fd_der_zz(uxy,i,j,k,invh,&(d[31]));
	fd_der_zz(uxz,i,j,k,invh,&(d[32]));
	fd_der_zz(uyy,i,j,k,invh,&(d[33]));
	fd_der_zz(uyz,i,j,k,invh,&(d[34]));
	fd_der_zz(uzz,i,j,k,invh,&(d[35]));
}
#endif 
