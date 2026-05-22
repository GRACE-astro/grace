
/****************************************************************************/
/*                     Tetrad helpers, SymPy generated                      */
/****************************************************************************/
#ifndef GRACE_TETRAD_SUBEXPR_HH
#define GRACE_TETRAD_SUBEXPR_HH

#include <Kokkos_Core.hpp>

static void KOKKOS_INLINE_FUNCTION
get_tetrad(
	double alp,
	const double betau[3],
	int idir,
	int jdir,
	int kdir,
	const double gammaUU[3][3],
	const double gammadd[3][3],
	const double betad[3],
	double (*etU)[4],
	double (*eiU)[4],
	double (*ejU)[4],
	double (*ekU)[4],
	double (*etd)[4],
	double (*eid)[4],
	double (*ejd)[4],
	double (*ekd)[4]
)
{
	double x0 = 1/(alp);
	double x1 = (1.0/sqrt(gammaUU[idir][idir]));
	double x2 = gammadd[jdir][jdir]*gammadd[kdir][kdir] - ((gammadd[jdir][kdir])*(gammadd[jdir][kdir]));
	double x3 = (1.0/sqrt(gammadd[kdir][kdir]*x2));
	double x4 = ((kdir == 0) ? (1) : (0));
	double x5 = ((jdir == 0) ? (1) : (0));
	double x6 = ((kdir == 1) ? (1) : (0));
	double x7 = ((jdir == 1) ? (1) : (0));
	double x8 = ((kdir == 2) ? (1) : (0));
	double x9 = ((jdir == 2) ? (1) : (0));
	double x10 = (1.0/sqrt(gammadd[kdir][kdir]));
	double x11 = ((idir == 0) ? (1) : (0));
	double x12 = ((idir == 1) ? (1) : (0));
	double x13 = ((idir == 2) ? (1) : (0));
	double x14 = gammadd[idir][jdir]*gammadd[kdir][kdir] - gammadd[idir][kdir]*gammadd[jdir][kdir];
	(*etU)[0] = x0;
	(*etU)[1] = -betau[0]*x0;
	(*etU)[2] = -betau[1]*x0;
	(*etU)[3] = -betau[2]*x0;
	(*eiU)[0] = 0;
	(*eiU)[1] = gammaUU[idir][0]*x1;
	(*eiU)[2] = gammaUU[idir][1]*x1;
	(*eiU)[3] = gammaUU[idir][2]*x1;
	(*ejU)[0] = 0;
	(*ejU)[1] = -x3*(gammadd[jdir][kdir]*x4 - gammadd[kdir][kdir]*x5);
	(*ejU)[2] = -x3*(gammadd[jdir][kdir]*x6 - gammadd[kdir][kdir]*x7);
	(*ejU)[3] = -x3*(gammadd[jdir][kdir]*x8 - gammadd[kdir][kdir]*x9);
	(*ekU)[0] = 0;
	(*ekU)[1] = x10*x4;
	(*ekU)[2] = x10*x6;
	(*ekU)[3] = x10*x8;
	(*etd)[0] = -alp;
	(*etd)[1] = 0;
	(*etd)[2] = 0;
	(*etd)[3] = 0;
	(*eid)[0] = betau[idir]*x1;
	(*eid)[1] = x1*x11;
	(*eid)[2] = x1*x12;
	(*eid)[3] = x1*x13;
	(*ejd)[0] = x3*(betad[jdir]*gammadd[kdir][kdir] - betad[kdir]*gammadd[jdir][kdir]);
	(*ejd)[1] = x3*(x11*x14 + x2*x5);
	(*ejd)[2] = x3*(x12*x14 + x2*x7);
	(*ejd)[3] = x3*(x13*x14 + x2*x9);
	(*ekd)[0] = betad[kdir]*x10;
	(*ekd)[1] = gammadd[kdir][0]*x10;
	(*ekd)[2] = gammadd[kdir][1]*x10;
	(*ekd)[3] = gammadd[kdir][2]*x10;
}

static void KOKKOS_INLINE_FUNCTION
transform_primitives_to_tetrad_frame(
	double alp,
	const double betau[3],
	const double gdd[6],
	const double etd[4],
	const double exd[4],
	const double eyd[4],
	const double ezd[4],
	const double zvec[3],
	const double Bvec[3],
	double (*uhat)[3],
	double (*vhat)[3],
	double (*Bhat)[3]
)
{
	double x0 = sqrt(zvec[0]*(gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2]) + zvec[1]*(gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2]) + zvec[2]*(gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2]) + 1);
	double x1 = alp/x0;
	double x2 = -betau[0] + x1*zvec[0];
	double x3 = -betau[1] + x1*zvec[1];
	double x4 = -betau[2] + x1*zvec[2];
	double x5 = exd[0] + exd[1]*x2 + exd[2]*x3 + exd[3]*x4;
	double x6 = x0/alp;
	double x7 = eyd[0] + eyd[1]*x2 + eyd[2]*x3 + eyd[3]*x4;
	double x8 = ezd[0] + ezd[1]*x2 + ezd[2]*x3 + ezd[3]*x4;
	double x9 = 1/(etd[0] + etd[1]*x2 + etd[2]*x3 + etd[3]*x4);
	(*uhat)[0] = x5*x6;
	(*uhat)[1] = x6*x7;
	(*uhat)[2] = x6*x8;
	(*vhat)[0] = -x5*x9;
	(*vhat)[1] = -x7*x9;
	(*vhat)[2] = -x8*x9;
	(*Bhat)[0] = Bvec[0]*exd[1] + Bvec[1]*exd[2] + Bvec[2]*exd[3];
	(*Bhat)[1] = Bvec[0]*eyd[1] + Bvec[1]*eyd[2] + Bvec[2]*eyd[3];
	(*Bhat)[2] = Bvec[0]*ezd[1] + Bvec[1]*ezd[2] + Bvec[2]*ezd[3];
}

static void KOKKOS_INLINE_FUNCTION
get_interface_velocity(
	double alp,
	const double betau[3],
	int idir,
	const double gammaUU[3][3],
	double * __restrict__ lambda_interface
)
{
	*lambda_interface = betau[idir]/(alp*sqrt(gammaUU[idir][idir]));
}

static void KOKKOS_INLINE_FUNCTION
transform_fluxes_to_grid_frame(
	double alp,
	int idir,
	const double exd[4],
	const double eyd[4],
	const double ezd[4],
	double dens,
	double tau,
	double fdens,
	double ftau,
	const double fJhat[3],
	const double Jhat[3],
	const double eUU[4][4],
	double * __restrict__ fdens_eul,
	double (*fJhat_eul)[3],
	double * __restrict__ ftau_eul
)
{
	double x0 = eUU[idir+1][idir+1]*fJhat[0];
	double x1 = eUU[idir+1][idir+1]*fJhat[1];
	double x2 = eUU[idir+1][idir+1]*fJhat[2];
	*fdens_eul = alp*(dens*eUU[0][idir+1] + eUU[idir+1][idir+1]*fdens);
	(*fJhat_eul)[0] = alp*(eUU[0][idir+1]*(Jhat[0]*exd[1] + Jhat[1]*eyd[1] + Jhat[2]*ezd[1]) + exd[1]*x0 + eyd[1]*x1 + ezd[1]*x2);
	(*fJhat_eul)[1] = alp*(eUU[0][idir+1]*(Jhat[0]*exd[2] + Jhat[1]*eyd[2] + Jhat[2]*ezd[2]) + exd[2]*x0 + eyd[2]*x1 + ezd[2]*x2);
	(*fJhat_eul)[2] = alp*(eUU[0][idir+1]*(Jhat[0]*exd[3] + Jhat[1]*eyd[3] + Jhat[2]*ezd[3]) + exd[3]*x0 + eyd[3]*x1 + ezd[3]*x2);
	*ftau_eul = alp*(eUU[0][idir+1]*tau + eUU[idir+1][idir+1]*ftau);
}

#endif 
