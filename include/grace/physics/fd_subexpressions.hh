
/****************************************************************************/
/*                      FD helpers, SymPy generated                         */
/****************************************************************************/
#ifndef GRACE_FD_SUBEXPR_HH
#define GRACE_FD_SUBEXPR_HH

#include <Kokkos_Core.hpp>

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_x_l2(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((25.0/12.0)*u(i,j,k) - 4*u(i-1,j,k) + 3*u(i-2,j,k) - 4.0/3.0*u(i-3,j,k) + (1.0/4.0)*u(i-4,j,k));
}

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
	*du = (1.0/12.0)*invh*(3*u(i+1,j,k) + 10*u(i,j,k) - 18*u(i-1,j,k) + 6*u(i-2,j,k) - u(i-3,j,k));
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
	*du = (1.0/12.0)*invh*(8*u(i+1,j,k) - u(i+2,j,k) - 8*u(i-1,j,k) + u(i-2,j,k));
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
	*du = (1.0/12.0)*invh*(18*u(i+1,j,k) - 6*u(i+2,j,k) + u(i+3,j,k) - 10*u(i,j,k) - 3*u(i-1,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_x_r2(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(4*u(i+1,j,k) - 3*u(i+2,j,k) + (4.0/3.0)*u(i+3,j,k) - 1.0/4.0*u(i+4,j,k) - 25.0/12.0*u(i,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_y_l2(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((25.0/12.0)*u(i,j,k) - 4*u(i,j-1,k) + 3*u(i,j-2,k) - 4.0/3.0*u(i,j-3,k) + (1.0/4.0)*u(i,j-4,k));
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
	*du = (1.0/12.0)*invh*(3*u(i,j+1,k) + 10*u(i,j,k) - 18*u(i,j-1,k) + 6*u(i,j-2,k) - u(i,j-3,k));
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
	*du = (1.0/12.0)*invh*(8*u(i,j+1,k) - u(i,j+2,k) - 8*u(i,j-1,k) + u(i,j-2,k));
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
	*du = (1.0/12.0)*invh*(18*u(i,j+1,k) - 6*u(i,j+2,k) + u(i,j+3,k) - 10*u(i,j,k) - 3*u(i,j-1,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_y_r2(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(4*u(i,j+1,k) - 3*u(i,j+2,k) + (4.0/3.0)*u(i,j+3,k) - 1.0/4.0*u(i,j+4,k) - 25.0/12.0*u(i,j,k));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_z_l2(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*((25.0/12.0)*u(i,j,k) - 4*u(i,j,k-1) + 3*u(i,j,k-2) - 4.0/3.0*u(i,j,k-3) + (1.0/4.0)*u(i,j,k-4));
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
	*du = (1.0/12.0)*invh*(10*u(i,j,k) + 3*u(i,j,k+1) - 18*u(i,j,k-1) + 6*u(i,j,k-2) - u(i,j,k-3));
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
	*du = (1.0/12.0)*invh*(8*u(i,j,k+1) - u(i,j,k+2) - 8*u(i,j,k-1) + u(i,j,k-2));
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
	*du = (1.0/12.0)*invh*(-10*u(i,j,k) + 18*u(i,j,k+1) - 6*u(i,j,k+2) + u(i,j,k+3) - 3*u(i,j,k-1));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fd_der_z_r2(
	view_t u,
	int i,
	int j,
	int k,
	double invh,
	double * __restrict__ du
)
{
	*du = invh*(-25.0/12.0*u(i,j,k) + 4*u(i,j,k+1) - 3*u(i,j,k+2) + (4.0/3.0)*u(i,j,k+3) - 1.0/4.0*u(i,j,k+4));
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
	*du = (1.0/2.0)*invh*(u(i+1,j,k) - u(i-1,j,k));
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
	*du = (1.0/2.0)*invh*(u(i,j+1,k) - u(i,j-1,k));
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
	*du = (1.0/2.0)*invh*(u(i,j,k+1) - u(i,j,k-1));
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
	*du = (1.0/12.0)*((invh)*(invh))*(16*u(i+1,j,k) - u(i+2,j,k) - 30*u(i,j,k) + 16*u(i-1,j,k) - u(i-2,j,k));
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
	*du = (1.0/12.0)*((invh)*(invh))*(16*u(i,j+1,k) - u(i,j+2,k) - 30*u(i,j,k) + 16*u(i,j-1,k) - u(i,j-2,k));
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
	*du = (1.0/12.0)*((invh)*(invh))*(-30*u(i,j,k) + 16*u(i,j,k+1) - u(i,j,k+2) + 16*u(i,j,k-1) - u(i,j,k-2));
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
	*du = (1.0/144.0)*((invh)*(invh))*(64*u(i+1,j+1,k) - 8*u(i+1,j+2,k) - 64*u(i+1,j-1,k) + 8*u(i+1,j-2,k) - 8*u(i+2,j+1,k) + u(i+2,j+2,k) + 8*u(i+2,j-1,k) - u(i+2,j-2,k) - 64*u(i-1,j+1,k) + 8*u(i-1,j+2,k) + 64*u(i-1,j-1,k) - 8*u(i-1,j-2,k) + 8*u(i-2,j+1,k) - u(i-2,j+2,k) - 8*u(i-2,j-1,k) + u(i-2,j-2,k));
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
	*du = (1.0/144.0)*((invh)*(invh))*(64*u(i+1,j,k+1) - 8*u(i+1,j,k+2) - 64*u(i+1,j,k-1) + 8*u(i+1,j,k-2) - 8*u(i+2,j,k+1) + u(i+2,j,k+2) + 8*u(i+2,j,k-1) - u(i+2,j,k-2) - 64*u(i-1,j,k+1) + 8*u(i-1,j,k+2) + 64*u(i-1,j,k-1) - 8*u(i-1,j,k-2) + 8*u(i-2,j,k+1) - u(i-2,j,k+2) - 8*u(i-2,j,k-1) + u(i-2,j,k-2));
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
	*du = (1.0/144.0)*((invh)*(invh))*(64*u(i,j+1,k+1) - 8*u(i,j+1,k+2) - 64*u(i,j+1,k-1) + 8*u(i,j+1,k-2) - 8*u(i,j+2,k+1) + u(i,j+2,k+2) + 8*u(i,j+2,k-1) - u(i,j+2,k-2) - 64*u(i,j-1,k+1) + 8*u(i,j-1,k+2) + 64*u(i,j-1,k-1) - 8*u(i,j-1,k-2) + 8*u(i,j-2,k+1) - u(i,j-2,k+2) - 8*u(i,j-2,k-1) + u(i,j-2,k-2));
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
	*du = (1.0/12.0)*invh*(3*u(i+1,j,k) + 10*u(i,j,k) - 18*u(i-1,j,k) + 6*u(i-2,j,k) - u(i-3,j,k));
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
	*du = (1.0/12.0)*invh*(18*u(i+1,j,k) - 6*u(i+2,j,k) + u(i+3,j,k) - 10*u(i,j,k) - 3*u(i-1,j,k));
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
	*du = (1.0/12.0)*invh*(3*u(i,j+1,k) + 10*u(i,j,k) - 18*u(i,j-1,k) + 6*u(i,j-2,k) - u(i,j-3,k));
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
	*du = (1.0/12.0)*invh*(18*u(i,j+1,k) - 6*u(i,j+2,k) + u(i,j+3,k) - 10*u(i,j,k) - 3*u(i,j-1,k));
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
	*du = (1.0/12.0)*invh*(10*u(i,j,k) + 3*u(i,j,k+1) - 18*u(i,j,k-1) + 6*u(i,j,k-2) - u(i,j,k-3));
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
	*du = (1.0/12.0)*invh*(-10*u(i,j,k) + 18*u(i,j,k+1) - 6*u(i,j,k+2) + u(i,j,k+3) - 3*u(i,j,k-1));
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
fill_deriv_scalar_upw(view_t state, int i, int j, int k, int iv, int q, double d[3], double v[3], double invh)
{
	using namespace Kokkos;
	auto u = subview(state,ALL(),ALL(),ALL(),iv,q);
	if (v[0]>0){
		fd_der_x_upw_pos(u,i,j,k,invh,&(d[0]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(u,i,j,k,invh,&(d[0]));
	}else{
		fd_der_x(u,i,j,k,invh,&(d[0]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(u,i,j,k,invh,&(d[1]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(u,i,j,k,invh,&(d[1]));
	}else{
		fd_der_y(u,i,j,k,invh,&(d[1]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(u,i,j,k,invh,&(d[2]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(u,i,j,k,invh,&(d[2]));
	}else{
		fd_der_z(u,i,j,k,invh,&(d[2]));
	}
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_vector_upw(view_t state, int i, int j, int k, int iv, int q, double d[9], double v[3], double invh)
{
	using namespace Kokkos;
	auto ux = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	if (v[0]>0){
		fd_der_x_upw_pos(ux,i,j,k,invh,&(d[0]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(ux,i,j,k,invh,&(d[0]));
	}else{
		fd_der_x(ux,i,j,k,invh,&(d[0]));
	}
	if (v[0]>0){
		fd_der_x_upw_pos(uy,i,j,k,invh,&(d[1]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uy,i,j,k,invh,&(d[1]));
	}else{
		fd_der_x(uy,i,j,k,invh,&(d[1]));
	}
	if (v[0]>0){
		fd_der_x_upw_pos(uz,i,j,k,invh,&(d[2]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uz,i,j,k,invh,&(d[2]));
	}else{
		fd_der_x(uz,i,j,k,invh,&(d[2]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(ux,i,j,k,invh,&(d[3]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(ux,i,j,k,invh,&(d[3]));
	}else{
		fd_der_y(ux,i,j,k,invh,&(d[3]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(uy,i,j,k,invh,&(d[4]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uy,i,j,k,invh,&(d[4]));
	}else{
		fd_der_y(uy,i,j,k,invh,&(d[4]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(uz,i,j,k,invh,&(d[5]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uz,i,j,k,invh,&(d[5]));
	}else{
		fd_der_y(uz,i,j,k,invh,&(d[5]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(ux,i,j,k,invh,&(d[6]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(ux,i,j,k,invh,&(d[6]));
	}else{
		fd_der_z(ux,i,j,k,invh,&(d[6]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(uy,i,j,k,invh,&(d[7]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uy,i,j,k,invh,&(d[7]));
	}else{
		fd_der_z(uy,i,j,k,invh,&(d[7]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(uz,i,j,k,invh,&(d[8]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uz,i,j,k,invh,&(d[8]));
	}else{
		fd_der_z(uz,i,j,k,invh,&(d[8]));
	}
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_tensor_upw(view_t state, int i, int j, int k, int iv, int q, double d[18], double v[3], double invh)
{
	using namespace Kokkos;
	auto uxx = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uxy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uxz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	auto uyy = subview(state,ALL(),ALL(),ALL(),iv+3,q);
	auto uyz = subview(state,ALL(),ALL(),ALL(),iv+4,q);
	auto uzz = subview(state,ALL(),ALL(),ALL(),iv+5,q);
	if (v[0]>0){
		fd_der_x_upw_pos(uxx,i,j,k,invh,&(d[0]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uxx,i,j,k,invh,&(d[0]));
	}else{
		fd_der_x(uxx,i,j,k,invh,&(d[0]));
	}
	if (v[0]>0){
		fd_der_x_upw_pos(uxy,i,j,k,invh,&(d[1]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uxy,i,j,k,invh,&(d[1]));
	}else{
		fd_der_x(uxy,i,j,k,invh,&(d[1]));
	}
	if (v[0]>0){
		fd_der_x_upw_pos(uxz,i,j,k,invh,&(d[2]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uxz,i,j,k,invh,&(d[2]));
	}else{
		fd_der_x(uxz,i,j,k,invh,&(d[2]));
	}
	if (v[0]>0){
		fd_der_x_upw_pos(uyy,i,j,k,invh,&(d[3]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uyy,i,j,k,invh,&(d[3]));
	}else{
		fd_der_x(uyy,i,j,k,invh,&(d[3]));
	}
	if (v[0]>0){
		fd_der_x_upw_pos(uyz,i,j,k,invh,&(d[4]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uyz,i,j,k,invh,&(d[4]));
	}else{
		fd_der_x(uyz,i,j,k,invh,&(d[4]));
	}
	if (v[0]>0){
		fd_der_x_upw_pos(uzz,i,j,k,invh,&(d[5]));
	}else if (v[0]<0){
		fd_der_x_upw_neg(uzz,i,j,k,invh,&(d[5]));
	}else{
		fd_der_x(uzz,i,j,k,invh,&(d[5]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(uxx,i,j,k,invh,&(d[6]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uxx,i,j,k,invh,&(d[6]));
	}else{
		fd_der_y(uxx,i,j,k,invh,&(d[6]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(uxy,i,j,k,invh,&(d[7]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uxy,i,j,k,invh,&(d[7]));
	}else{
		fd_der_y(uxy,i,j,k,invh,&(d[7]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(uxz,i,j,k,invh,&(d[8]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uxz,i,j,k,invh,&(d[8]));
	}else{
		fd_der_y(uxz,i,j,k,invh,&(d[8]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(uyy,i,j,k,invh,&(d[9]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uyy,i,j,k,invh,&(d[9]));
	}else{
		fd_der_y(uyy,i,j,k,invh,&(d[9]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(uyz,i,j,k,invh,&(d[10]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uyz,i,j,k,invh,&(d[10]));
	}else{
		fd_der_y(uyz,i,j,k,invh,&(d[10]));
	}
	if (v[1]>0){
		fd_der_y_upw_pos(uzz,i,j,k,invh,&(d[11]));
	}else if (v[1]<0){
		fd_der_y_upw_neg(uzz,i,j,k,invh,&(d[11]));
	}else{
		fd_der_y(uzz,i,j,k,invh,&(d[11]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(uxx,i,j,k,invh,&(d[12]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uxx,i,j,k,invh,&(d[12]));
	}else{
		fd_der_z(uxx,i,j,k,invh,&(d[12]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(uxy,i,j,k,invh,&(d[13]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uxy,i,j,k,invh,&(d[13]));
	}else{
		fd_der_z(uxy,i,j,k,invh,&(d[13]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(uxz,i,j,k,invh,&(d[14]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uxz,i,j,k,invh,&(d[14]));
	}else{
		fd_der_z(uxz,i,j,k,invh,&(d[14]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(uyy,i,j,k,invh,&(d[15]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uyy,i,j,k,invh,&(d[15]));
	}else{
		fd_der_z(uyy,i,j,k,invh,&(d[15]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(uyz,i,j,k,invh,&(d[16]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uyz,i,j,k,invh,&(d[16]));
	}else{
		fd_der_z(uyz,i,j,k,invh,&(d[16]));
	}
	if (v[2]>0){
		fd_der_z_upw_pos(uzz,i,j,k,invh,&(d[17]));
	}else if (v[2]<0){
		fd_der_z_upw_neg(uzz,i,j,k,invh,&(d[17]));
	}else{
		fd_der_z(uzz,i,j,k,invh,&(d[17]));
	}
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
