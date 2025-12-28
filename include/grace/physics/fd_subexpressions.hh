
/****************************************************************************/
/*                      FD helpers, SymPy generated                         */
/****************************************************************************/
#ifndef GRACE_FD_SUBEXPR_HH
#define GRACE_FD_SUBEXPR_HH

#include <Kokkos_Core.hpp>

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_x_l2(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = ((25.0/12.0)*u(i,j,k) - 4*u(i-1,j,k) + 3*u(i-2,j,k) - 4.0/3.0*u(i-3,j,k) + (1.0/4.0)*u(i-4,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_x_l1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(3*u(i+1,j,k) + 10*u(i,j,k) - 18*u(i-1,j,k) + 6*u(i-2,j,k) - u(i-3,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_x(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(8*u(i+1,j,k) - u(i+2,j,k) - 8*u(i-1,j,k) + u(i-2,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_x_r1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(18*u(i+1,j,k) - 6*u(i+2,j,k) + u(i+3,j,k) - 10*u(i,j,k) - 3*u(i-1,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_x_r2(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (4*u(i+1,j,k) - 3*u(i+2,j,k) + (4.0/3.0)*u(i+3,j,k) - 1.0/4.0*u(i+4,j,k) - 25.0/12.0*u(i,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_y_l2(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = ((25.0/12.0)*u(i,j,k) - 4*u(i,j-1,k) + 3*u(i,j-2,k) - 4.0/3.0*u(i,j-3,k) + (1.0/4.0)*u(i,j-4,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_y_l1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(3*u(i,j+1,k) + 10*u(i,j,k) - 18*u(i,j-1,k) + 6*u(i,j-2,k) - u(i,j-3,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_y(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(8*u(i,j+1,k) - u(i,j+2,k) - 8*u(i,j-1,k) + u(i,j-2,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_y_r1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(18*u(i,j+1,k) - 6*u(i,j+2,k) + u(i,j+3,k) - 10*u(i,j,k) - 3*u(i,j-1,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_y_r2(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (4*u(i,j+1,k) - 3*u(i,j+2,k) + (4.0/3.0)*u(i,j+3,k) - 1.0/4.0*u(i,j+4,k) - 25.0/12.0*u(i,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_z_l2(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = ((25.0/12.0)*u(i,j,k) - 4*u(i,j,k-1) + 3*u(i,j,k-2) - 4.0/3.0*u(i,j,k-3) + (1.0/4.0)*u(i,j,k-4))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_z_l1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(10*u(i,j,k) + 3*u(i,j,k+1) - 18*u(i,j,k-1) + 6*u(i,j,k-2) - u(i,j,k-3))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_z(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(8*u(i,j,k+1) - u(i,j,k+2) - 8*u(i,j,k-1) + u(i,j,k-2))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_z_r1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(-10*u(i,j,k) + 18*u(i,j,k+1) - 6*u(i,j,k+2) + u(i,j,k+3) - 3*u(i,j,k-1))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_z_r2(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (-25.0/12.0*u(i,j,k) + 4*u(i,j,k+1) - 3*u(i,j,k+2) + (4.0/3.0)*u(i,j,k+3) - 1.0/4.0*u(i,j,k+4))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_2_x_l1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = ((3.0/2.0)*u(i,j,k) - 2*u(i-1,j,k) + (1.0/2.0)*u(i-2,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_2_x(
	double h,
	view_t u,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/2.0)*(u(i+1,j,k) - u(i-1,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_2_x_r1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (2*u(i+1,j,k) - 1.0/2.0*u(i+2,j,k) - 3.0/2.0*u(i,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_2_y_l1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = ((3.0/2.0)*u(i,j,k) - 2*u(i,j-1,k) + (1.0/2.0)*u(i,j-2,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_2_y(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/2.0)*(u(i,j+1,k) - u(i,j-1,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_2_y_r1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (2*u(i,j+1,k) - 1.0/2.0*u(i,j+2,k) - 3.0/2.0*u(i,j,k))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_2_z_l1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = ((3.0/2.0)*u(i,j,k) - 2*u(i,j,k-1) + (1.0/2.0)*u(i,j,k-2))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_2_z(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/2.0)*(u(i,j,k+1) - u(i,j,k-1))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_2_z_r1(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (-3.0/2.0*u(i,j,k) + 2*u(i,j,k+1) - 1.0/2.0*u(i,j,k+2))/h;
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_xx(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(16*u(i+1,j,k) - u(i+2,j,k) - 30*u(i,j,k) + 16*u(i-1,j,k) - u(i-2,j,k))/((h)*(h));
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_yy(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(16*u(i,j+1,k) - u(i,j+2,k) - 30*u(i,j,k) + 16*u(i,j-1,k) - u(i,j-2,k))/((h)*(h));
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_zz(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/12.0)*(-30*u(i,j,k) + 16*u(i,j,k+1) - u(i,j,k+2) + 16*u(i,j,k-1) - u(i,j,k-2))/((h)*(h));
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_xy(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/144.0)*(64*u(i+1,j+1,k) - 8*u(i+1,j+2,k) - 64*u(i+1,j-1,k) + 8*u(i+1,j-2,k) - 8*u(i+2,j+1,k) + u(i+2,j+2,k) + 8*u(i+2,j-1,k) - u(i+2,j-2,k) - 64*u(i-1,j+1,k) + 8*u(i-1,j+2,k) + 64*u(i-1,j-1,k) - 8*u(i-1,j-2,k) + 8*u(i-2,j+1,k) - u(i-2,j+2,k) - 8*u(i-2,j-1,k) + u(i-2,j-2,k))/((h)*(h));
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_xz(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/144.0)*(64*u(i+1,j,k+1) - 8*u(i+1,j,k+2) - 64*u(i+1,j,k-1) + 8*u(i+1,j,k-2) - 8*u(i+2,j,k+1) + u(i+2,j,k+2) + 8*u(i+2,j,k-1) - u(i+2,j,k-2) - 64*u(i-1,j,k+1) + 8*u(i-1,j,k+2) + 64*u(i-1,j,k-1) - 8*u(i-1,j,k-2) + 8*u(i-2,j,k+1) - u(i-2,j,k+2) - 8*u(i-2,j,k-1) + u(i-2,j,k-2))/((h)*(h));
}

template< typename view_t>
static void KOKKOS_INLINE_FUNCTION
fd_der_yz(
	view_t u,
	double h,
	int i,
	int j,
	int k,
	double * __restrict__ du
)
{
	*du = (1.0/144.0)*(64*u(i,j+1,k+1) - 8*u(i,j+1,k+2) - 64*u(i,j+1,k-1) + 8*u(i,j+1,k-2) - 8*u(i,j+2,k+1) + u(i,j+2,k+2) + 8*u(i,j+2,k-1) - u(i,j+2,k-2) - 64*u(i,j-1,k+1) + 8*u(i,j-1,k+2) + 64*u(i,j-1,k-1) - 8*u(i,j-1,k-2) + 8*u(i,j-2,k+1) - u(i,j-2,k+2) - 8*u(i,j-2,k-1) + u(i,j-2,k-2))/((h)*(h));
}

template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_scalar(view_t state, int i, int j, int k, int iv, int q, double d[3], double h)
{
	using namespace Kokkos;
	auto u = subview(state,ALL(),ALL(),ALL(),iv,q);
	fd_der_x(u,h,i,j,k,&(d[0]));
	fd_der_y(u,h,i,j,k,&(d[1]));
	fd_der_z(u,h,i,j,k,&(d[2]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_vector(view_t state, int i, int j, int k, int iv, int q, double d[9], double h)
{
	using namespace Kokkos;
	auto ux = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	fd_der_x(ux,h,i,j,k,&(d[0]));
	fd_der_x(uy,h,i,j,k,&(d[1]));
	fd_der_x(uz,h,i,j,k,&(d[2]));
	fd_der_y(ux,h,i,j,k,&(d[3]));
	fd_der_y(uy,h,i,j,k,&(d[4]));
	fd_der_y(uz,h,i,j,k,&(d[5]));
	fd_der_z(ux,h,i,j,k,&(d[6]));
	fd_der_z(uy,h,i,j,k,&(d[7]));
	fd_der_z(uz,h,i,j,k,&(d[8]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_tensor(view_t state, int i, int j, int k, int iv, int q, double d[18], double h)
{
	using namespace Kokkos;
	auto uxx = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uxy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uxz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	auto uyy = subview(state,ALL(),ALL(),ALL(),iv+3,q);
	auto uyz = subview(state,ALL(),ALL(),ALL(),iv+4,q);
	auto uzz = subview(state,ALL(),ALL(),ALL(),iv+5,q);
	fd_der_x(uxx,h,i,j,k,&(d[0]));
	fd_der_x(uxy,h,i,j,k,&(d[1]));
	fd_der_x(uxz,h,i,j,k,&(d[2]));
	fd_der_x(uyy,h,i,j,k,&(d[3]));
	fd_der_x(uyz,h,i,j,k,&(d[4]));
	fd_der_x(uzz,h,i,j,k,&(d[5]));
	fd_der_y(uxx,h,i,j,k,&(d[6]));
	fd_der_y(uxy,h,i,j,k,&(d[7]));
	fd_der_y(uxz,h,i,j,k,&(d[8]));
	fd_der_y(uyy,h,i,j,k,&(d[9]));
	fd_der_y(uyz,h,i,j,k,&(d[10]));
	fd_der_y(uzz,h,i,j,k,&(d[11]));
	fd_der_z(uxx,h,i,j,k,&(d[12]));
	fd_der_z(uxy,h,i,j,k,&(d[13]));
	fd_der_z(uxz,h,i,j,k,&(d[14]));
	fd_der_z(uyy,h,i,j,k,&(d[15]));
	fd_der_z(uyz,h,i,j,k,&(d[16]));
	fd_der_z(uzz,h,i,j,k,&(d[17]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_scalar_upw(view_t state, int i, int j, int k, int iv, int q, double d[3], double v[3], double h)
{
	using namespace Kokkos;
	auto u = subview(state,ALL(),ALL(),ALL(),iv,q);
	if (v[0]>0){
		fd_der_x_r1(u,h,i,j,k,&(d[0]));
	}else if (v[0]<0){
		fd_der_x_l1(u,h,i,j,k,&(d[0]));
	}else{
		fd_der_x(u,h,i,j,k,&(d[0]));
	}
	if (v[1]>0){
		fd_der_y_r1(u,h,i,j,k,&(d[1]));
	}else if (v[1]<0){
		fd_der_y_l1(u,h,i,j,k,&(d[1]));
	}else{
		fd_der_y(u,h,i,j,k,&(d[1]));
	}
	if (v[2]>0){
		fd_der_z_r1(u,h,i,j,k,&(d[2]));
	}else if (v[2]<0){
		fd_der_z_l1(u,h,i,j,k,&(d[2]));
	}else{
		fd_der_z(u,h,i,j,k,&(d[2]));
	}
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_vector_upw(view_t state, int i, int j, int k, int iv, int q, double d[9], double v[3], double h)
{
	using namespace Kokkos;
	auto ux = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	if (v[0]>0){
		fd_der_x_r1(ux,h,i,j,k,&(d[0]));
	}else if (v[0]<0){
		fd_der_x_l1(ux,h,i,j,k,&(d[0]));
	}else{
		fd_der_x(ux,h,i,j,k,&(d[0]));
	}
	if (v[0]>0){
		fd_der_x_r1(uy,h,i,j,k,&(d[1]));
	}else if (v[0]<0){
		fd_der_x_l1(uy,h,i,j,k,&(d[1]));
	}else{
		fd_der_x(uy,h,i,j,k,&(d[1]));
	}
	if (v[0]>0){
		fd_der_x_r1(uz,h,i,j,k,&(d[2]));
	}else if (v[0]<0){
		fd_der_x_l1(uz,h,i,j,k,&(d[2]));
	}else{
		fd_der_x(uz,h,i,j,k,&(d[2]));
	}
	if (v[1]>0){
		fd_der_y_r1(ux,h,i,j,k,&(d[3]));
	}else if (v[1]<0){
		fd_der_y_l1(ux,h,i,j,k,&(d[3]));
	}else{
		fd_der_y(ux,h,i,j,k,&(d[3]));
	}
	if (v[1]>0){
		fd_der_y_r1(uy,h,i,j,k,&(d[4]));
	}else if (v[1]<0){
		fd_der_y_l1(uy,h,i,j,k,&(d[4]));
	}else{
		fd_der_y(uy,h,i,j,k,&(d[4]));
	}
	if (v[1]>0){
		fd_der_y_r1(uz,h,i,j,k,&(d[5]));
	}else if (v[1]<0){
		fd_der_y_l1(uz,h,i,j,k,&(d[5]));
	}else{
		fd_der_y(uz,h,i,j,k,&(d[5]));
	}
	if (v[2]>0){
		fd_der_z_r1(ux,h,i,j,k,&(d[6]));
	}else if (v[2]<0){
		fd_der_z_l1(ux,h,i,j,k,&(d[6]));
	}else{
		fd_der_z(ux,h,i,j,k,&(d[6]));
	}
	if (v[2]>0){
		fd_der_z_r1(uy,h,i,j,k,&(d[7]));
	}else if (v[2]<0){
		fd_der_z_l1(uy,h,i,j,k,&(d[7]));
	}else{
		fd_der_z(uy,h,i,j,k,&(d[7]));
	}
	if (v[2]>0){
		fd_der_z_r1(uz,h,i,j,k,&(d[8]));
	}else if (v[2]<0){
		fd_der_z_l1(uz,h,i,j,k,&(d[8]));
	}else{
		fd_der_z(uz,h,i,j,k,&(d[8]));
	}
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_deriv_tensor_upw(view_t state, int i, int j, int k, int iv, int q, double d[18], double v[3], double h)
{
	using namespace Kokkos;
	auto uxx = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uxy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uxz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	auto uyy = subview(state,ALL(),ALL(),ALL(),iv+3,q);
	auto uyz = subview(state,ALL(),ALL(),ALL(),iv+4,q);
	auto uzz = subview(state,ALL(),ALL(),ALL(),iv+5,q);
	if (v[0]>0){
		fd_der_x_r1(uxx,h,i,j,k,&(d[0]));
	}else if (v[0]<0){
		fd_der_x_l1(uxx,h,i,j,k,&(d[0]));
	}else{
		fd_der_x(uxx,h,i,j,k,&(d[0]));
	}
	if (v[0]>0){
		fd_der_x_r1(uxy,h,i,j,k,&(d[1]));
	}else if (v[0]<0){
		fd_der_x_l1(uxy,h,i,j,k,&(d[1]));
	}else{
		fd_der_x(uxy,h,i,j,k,&(d[1]));
	}
	if (v[0]>0){
		fd_der_x_r1(uxz,h,i,j,k,&(d[2]));
	}else if (v[0]<0){
		fd_der_x_l1(uxz,h,i,j,k,&(d[2]));
	}else{
		fd_der_x(uxz,h,i,j,k,&(d[2]));
	}
	if (v[0]>0){
		fd_der_x_r1(uyy,h,i,j,k,&(d[3]));
	}else if (v[0]<0){
		fd_der_x_l1(uyy,h,i,j,k,&(d[3]));
	}else{
		fd_der_x(uyy,h,i,j,k,&(d[3]));
	}
	if (v[0]>0){
		fd_der_x_r1(uyz,h,i,j,k,&(d[4]));
	}else if (v[0]<0){
		fd_der_x_l1(uyz,h,i,j,k,&(d[4]));
	}else{
		fd_der_x(uyz,h,i,j,k,&(d[4]));
	}
	if (v[0]>0){
		fd_der_x_r1(uzz,h,i,j,k,&(d[5]));
	}else if (v[0]<0){
		fd_der_x_l1(uzz,h,i,j,k,&(d[5]));
	}else{
		fd_der_x(uzz,h,i,j,k,&(d[5]));
	}
	if (v[1]>0){
		fd_der_y_r1(uxx,h,i,j,k,&(d[6]));
	}else if (v[1]<0){
		fd_der_y_l1(uxx,h,i,j,k,&(d[6]));
	}else{
		fd_der_y(uxx,h,i,j,k,&(d[6]));
	}
	if (v[1]>0){
		fd_der_y_r1(uxy,h,i,j,k,&(d[7]));
	}else if (v[1]<0){
		fd_der_y_l1(uxy,h,i,j,k,&(d[7]));
	}else{
		fd_der_y(uxy,h,i,j,k,&(d[7]));
	}
	if (v[1]>0){
		fd_der_y_r1(uxz,h,i,j,k,&(d[8]));
	}else if (v[1]<0){
		fd_der_y_l1(uxz,h,i,j,k,&(d[8]));
	}else{
		fd_der_y(uxz,h,i,j,k,&(d[8]));
	}
	if (v[1]>0){
		fd_der_y_r1(uyy,h,i,j,k,&(d[9]));
	}else if (v[1]<0){
		fd_der_y_l1(uyy,h,i,j,k,&(d[9]));
	}else{
		fd_der_y(uyy,h,i,j,k,&(d[9]));
	}
	if (v[1]>0){
		fd_der_y_r1(uyz,h,i,j,k,&(d[10]));
	}else if (v[1]<0){
		fd_der_y_l1(uyz,h,i,j,k,&(d[10]));
	}else{
		fd_der_y(uyz,h,i,j,k,&(d[10]));
	}
	if (v[1]>0){
		fd_der_y_r1(uzz,h,i,j,k,&(d[11]));
	}else if (v[1]<0){
		fd_der_y_l1(uzz,h,i,j,k,&(d[11]));
	}else{
		fd_der_y(uzz,h,i,j,k,&(d[11]));
	}
	if (v[2]>0){
		fd_der_z_r1(uxx,h,i,j,k,&(d[12]));
	}else if (v[2]<0){
		fd_der_z_l1(uxx,h,i,j,k,&(d[12]));
	}else{
		fd_der_z(uxx,h,i,j,k,&(d[12]));
	}
	if (v[2]>0){
		fd_der_z_r1(uxy,h,i,j,k,&(d[13]));
	}else if (v[2]<0){
		fd_der_z_l1(uxy,h,i,j,k,&(d[13]));
	}else{
		fd_der_z(uxy,h,i,j,k,&(d[13]));
	}
	if (v[2]>0){
		fd_der_z_r1(uxz,h,i,j,k,&(d[14]));
	}else if (v[2]<0){
		fd_der_z_l1(uxz,h,i,j,k,&(d[14]));
	}else{
		fd_der_z(uxz,h,i,j,k,&(d[14]));
	}
	if (v[2]>0){
		fd_der_z_r1(uyy,h,i,j,k,&(d[15]));
	}else if (v[2]<0){
		fd_der_z_l1(uyy,h,i,j,k,&(d[15]));
	}else{
		fd_der_z(uyy,h,i,j,k,&(d[15]));
	}
	if (v[2]>0){
		fd_der_z_r1(uyz,h,i,j,k,&(d[16]));
	}else if (v[2]<0){
		fd_der_z_l1(uyz,h,i,j,k,&(d[16]));
	}else{
		fd_der_z(uyz,h,i,j,k,&(d[16]));
	}
	if (v[2]>0){
		fd_der_z_r1(uzz,h,i,j,k,&(d[17]));
	}else if (v[2]<0){
		fd_der_z_l1(uzz,h,i,j,k,&(d[17]));
	}else{
		fd_der_z(uzz,h,i,j,k,&(d[17]));
	}
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_second_deriv_scalar(view_t state, int i, int j, int k, int iv, int q, double d[6], double h)
{
	using namespace Kokkos;
	auto u = subview(state,ALL(),ALL(),ALL(),iv,q);
	fd_der_xx(u,h,i,j,k,&(d[0]));
	fd_der_xy(u,h,i,j,k,&(d[1]));
	fd_der_xz(u,h,i,j,k,&(d[2]));
	fd_der_yy(u,h,i,j,k,&(d[3]));
	fd_der_yz(u,h,i,j,k,&(d[4]));
	fd_der_zz(u,h,i,j,k,&(d[5]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_second_deriv_vector(view_t state, int i, int j, int k, int iv, int q, double d[18], double h)
{
	using namespace Kokkos;
	auto ux = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	fd_der_xx(ux,h,i,j,k,&(d[0]));
	fd_der_xx(uy,h,i,j,k,&(d[1]));
	fd_der_xx(uz,h,i,j,k,&(d[2]));
	fd_der_xy(ux,h,i,j,k,&(d[3]));
	fd_der_xy(uy,h,i,j,k,&(d[4]));
	fd_der_xy(uz,h,i,j,k,&(d[5]));
	fd_der_xz(ux,h,i,j,k,&(d[6]));
	fd_der_xz(uy,h,i,j,k,&(d[7]));
	fd_der_xz(uz,h,i,j,k,&(d[8]));
	fd_der_yy(ux,h,i,j,k,&(d[9]));
	fd_der_yy(uy,h,i,j,k,&(d[10]));
	fd_der_yy(uz,h,i,j,k,&(d[11]));
	fd_der_yz(ux,h,i,j,k,&(d[12]));
	fd_der_yz(uy,h,i,j,k,&(d[13]));
	fd_der_yz(uz,h,i,j,k,&(d[14]));
	fd_der_zz(ux,h,i,j,k,&(d[15]));
	fd_der_zz(uy,h,i,j,k,&(d[16]));
	fd_der_zz(uz,h,i,j,k,&(d[17]));
}
template< typename view_t >
static void KOKKOS_INLINE_FUNCTION
fill_second_deriv_tensor(view_t state, int i, int j, int k, int iv, int q, double d[36], double h)
{
	using namespace Kokkos;
	auto uxx = subview(state,ALL(),ALL(),ALL(),iv,q)  ;
	auto uxy = subview(state,ALL(),ALL(),ALL(),iv+1,q);
	auto uxz = subview(state,ALL(),ALL(),ALL(),iv+2,q);
	auto uyy = subview(state,ALL(),ALL(),ALL(),iv+3,q);
	auto uyz = subview(state,ALL(),ALL(),ALL(),iv+4,q);
	auto uzz = subview(state,ALL(),ALL(),ALL(),iv+5,q);
	fd_der_xx(uxx,h,i,j,k,&(d[0]));
	fd_der_xx(uxy,h,i,j,k,&(d[1]));
	fd_der_xx(uxz,h,i,j,k,&(d[2]));
	fd_der_xx(uyy,h,i,j,k,&(d[3]));
	fd_der_xx(uyz,h,i,j,k,&(d[4]));
	fd_der_xx(uzz,h,i,j,k,&(d[5]));
	fd_der_xy(uxx,h,i,j,k,&(d[6]));
	fd_der_xy(uxy,h,i,j,k,&(d[7]));
	fd_der_xy(uxz,h,i,j,k,&(d[8]));
	fd_der_xy(uyy,h,i,j,k,&(d[9]));
	fd_der_xy(uyz,h,i,j,k,&(d[10]));
	fd_der_xy(uzz,h,i,j,k,&(d[11]));
	fd_der_xz(uxx,h,i,j,k,&(d[12]));
	fd_der_xz(uxy,h,i,j,k,&(d[13]));
	fd_der_xz(uxz,h,i,j,k,&(d[14]));
	fd_der_xz(uyy,h,i,j,k,&(d[15]));
	fd_der_xz(uyz,h,i,j,k,&(d[16]));
	fd_der_xz(uzz,h,i,j,k,&(d[17]));
	fd_der_yy(uxx,h,i,j,k,&(d[18]));
	fd_der_yy(uxy,h,i,j,k,&(d[19]));
	fd_der_yy(uxz,h,i,j,k,&(d[20]));
	fd_der_yy(uyy,h,i,j,k,&(d[21]));
	fd_der_yy(uyz,h,i,j,k,&(d[22]));
	fd_der_yy(uzz,h,i,j,k,&(d[23]));
	fd_der_yz(uxx,h,i,j,k,&(d[24]));
	fd_der_yz(uxy,h,i,j,k,&(d[25]));
	fd_der_yz(uxz,h,i,j,k,&(d[26]));
	fd_der_yz(uyy,h,i,j,k,&(d[27]));
	fd_der_yz(uyz,h,i,j,k,&(d[28]));
	fd_der_yz(uzz,h,i,j,k,&(d[29]));
	fd_der_zz(uxx,h,i,j,k,&(d[30]));
	fd_der_zz(uxy,h,i,j,k,&(d[31]));
	fd_der_zz(uxz,h,i,j,k,&(d[32]));
	fd_der_zz(uyy,h,i,j,k,&(d[33]));
	fd_der_zz(uyz,h,i,j,k,&(d[34]));
	fd_der_zz(uzz,h,i,j,k,&(d[35]));
}
#endif 
