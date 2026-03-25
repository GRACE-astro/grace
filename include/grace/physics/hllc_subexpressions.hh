
/****************************************************************************/
/*                     HLLC helpers, SymPy generated                        */
/****************************************************************************/
#ifndef GRACE_HLLC_SUBEXPR_HH
#define GRACE_HLLC_SUBEXPR_HH

#include <Kokkos_Core.hpp>

static void KOKKOS_INLINE_FUNCTION
hllc_get_W_vel(
	const double gdd[6],
	const double zvec[3],
	double * __restrict__ W,
	double (*velu)[3]
)
{
	double x0 = 1/(W);
	*W = sqrt(zvec[0]*(gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2]) + zvec[1]*(gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2]) + zvec[2]*(gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2]) + 1);
	(*velu)[0] = x0*zvec[0];
	(*velu)[1] = x0*zvec[1];
	(*velu)[2] = x0*zvec[2];
}

static void KOKKOS_INLINE_FUNCTION
hllc_get_tetrad(
	int idir,
	int jdir,
	int kdir,
	const double gammaUU[3][3],
	const double gammadd[3][3],
	const double betaU[3],
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
	double x0 = (1.0/sqrt(gammaUU[idir][idir]));
	double x1 = gammadd[jdir][jdir]*gammadd[kdir][kdir] - ((gammadd[jdir][kdir])*(gammadd[jdir][kdir]));
	double x2 = (1.0/sqrt(gammadd[kdir][kdir]*x1));
	double x3 = (1.0/sqrt(gammadd[kdir][kdir]));
	(*etU)[0] = etU[0];
	(*etU)[1] = etU[1];
	(*etU)[2] = etU[2];
	(*etU)[3] = etU[3];
	(*eiU)[0] = 0;
	(*eiU)[1] = gammaUU[idir][0]*x0;
	(*eiU)[2] = gammaUU[idir][1]*x0;
	(*eiU)[3] = gammaUU[idir][2]*x0;
	(*ejU)[0] = 0;
	(*ejU)[1] = 0;
	(*ejU)[2] = gammadd[kdir][kdir]*x2;
	(*ejU)[3] = -gammadd[jdir][kdir]*x2;
	(*ekU)[0] = 0;
	(*ekU)[1] = 0;
	(*ekU)[2] = 0;
	(*ekU)[3] = x3;
	(*etd)[0] = etd[0];
	(*etd)[1] = etd[1];
	(*etd)[2] = etd[2];
	(*etd)[3] = etd[3];
	(*eid)[0] = betaU[idir]*x0;
	(*eid)[1] = x0*((idir == 0) ? (
   1
)
: (
   0
));
	(*eid)[2] = x0*((idir == 1) ? (
   1
)
: (
   0
));
	(*eid)[3] = x0*((idir == 2) ? (
   1
)
: (
   0
));
	(*ejd)[0] = x2*(betad[jdir]*gammadd[kdir][kdir] - betad[kdir]*gammadd[jdir][kdir]);
	(*ejd)[1] = x2*(gammadd[idir][jdir]*gammadd[kdir][kdir] - gammadd[idir][kdir]*gammadd[jdir][kdir]);
	(*ejd)[2] = x1*x2;
	(*ejd)[3] = 0;
	(*ekd)[0] = betad[kdir]*x3;
	(*ekd)[1] = gammadd[kdir][0]*x3;
	(*ekd)[2] = gammadd[kdir][1]*x3;
	(*ekd)[3] = gammadd[kdir][2]*x3;
}

static void KOKKOS_INLINE_FUNCTION
hllc_transform_vectors(
	double alp,
	const double betau[3],
	const double Bvec[3],
	const double zvec[3],
	double W,
	const double etd[4],
	const double exd[4],
	const double eyd[4],
	const double ezd[4],
	double (*uhat)[4],
	double (*vhat)[3],
	double (*Bhat)[3]
)
{
	double x0 = alp/W;
	double x1 = betau[0] - x0*zvec[0];
	double x2 = betau[1] - x0*zvec[1];
	double x3 = betau[2] - x0*zvec[2];
	double x4 = -etd[0] + etd[1]*x1 + etd[2]*x2 + etd[3]*x3;
	double x5 = W/alp;
	double x6 = -exd[0] + exd[1]*x1 + exd[2]*x2 + exd[3]*x3;
	double x7 = -eyd[0] + eyd[1]*x1 + eyd[2]*x2 + eyd[3]*x3;
	double x8 = -ezd[0] + ezd[1]*x1 + ezd[2]*x2 + ezd[3]*x3;
	double x9 = 1/(x4);
	(*uhat)[0] = -x4*x5;
	(*uhat)[1] = -x5*x6;
	(*uhat)[2] = -x5*x7;
	(*uhat)[3] = -x5*x8;
	(*vhat)[0] = x6*x9;
	(*vhat)[1] = x7*x9;
	(*vhat)[2] = x8*x9;
	(*Bhat)[0] = Bvec[0]*exd[1] + Bvec[1]*exd[2] + Bvec[2]*exd[3];
	(*Bhat)[1] = Bvec[0]*eyd[1] + Bvec[1]*eyd[2] + Bvec[2]*eyd[3];
	(*Bhat)[2] = Bvec[0]*ezd[1] + Bvec[1]*ezd[2] + Bvec[2]*ezd[3];
}

static void KOKKOS_INLINE_FUNCTION
hllc_get_state_and_fluxes(
	double rho,
	double press,
	double eps,
	double W,
	int idir,
	const double vhat[3],
	const double uhat[4],
	const double Bhat[3],
	double * __restrict__ dens,
	double (*stilde)[3],
	double * __restrict__ tau,
	double * __restrict__ fdens,
	double (*fstilde)[3],
	double * __restrict__ ftau
)
{
	double x0 = W*rho;
	double x1 = press/rho;
	double x2 = eps + x1 + 1;
	double x3 = x0*x2;
	double x4 = 1/(W);
	double x5 = Bhat[0]*uhat[1] + Bhat[1]*uhat[2] + Bhat[2]*uhat[3];
	double x6 = ((Bhat[0])*(Bhat[0])) + ((Bhat[1])*(Bhat[1])) + ((Bhat[2])*(Bhat[2]));
	double x7 = uhat[1]*x3 - x4*(Bhat[0]*x5 - uhat[1]*x6);
	double x8 = uhat[2]*x3 - x4*(Bhat[1]*x5 - uhat[2]*x6);
	double x9 = uhat[3]*x3 - x4*(Bhat[2]*x5 - uhat[3]*x6);
	double x10 = ((x5)*(x5));
	double x11 = ((W)*(W));
	double x12 = x0*(W*x2 - x1*x4) - x0 - x10 + (1.0/2.0)*(x10 + x6)*(2*x11 - 1)/x11;
	*dens = x0;
	(*stilde)[0] = x7;
	(*stilde)[1] = x8;
	(*stilde)[2] = x9;
	*tau = x12;
	*fdens = vhat[idir]*x0;
	(*fstilde)[0] = press*((idir == 0) ? (
   1
)
: (
   0
)) + vhat[idir]*x7;
	(*fstilde)[1] = press*((idir == 1) ? (
   1
)
: (
   0
)) + vhat[idir]*x8;
	(*fstilde)[2] = press*((idir == 2) ? (
   1
)
: (
   0
)) + vhat[idir]*x9;
	*ftau = vhat[idir]*(press + x12);
}

static void KOKKOS_INLINE_FUNCTION
hllc_get_wavespeeds(
	double rho,
	double press,
	double eps,
	double cs2,
	double W,
	int idir,
	const double vhat[3],
	const double uhat[4],
	const double Bhat[3],
	double * __restrict__ cm,
	double * __restrict__ cp
)
{
	double x0 = ((vhat[0])*(vhat[0])) + ((vhat[1])*(vhat[1])) + ((vhat[2])*(vhat[2]));
	double x1 = (((Bhat[0])*(Bhat[0])) + ((Bhat[1])*(Bhat[1])) + ((Bhat[2])*(Bhat[2])) + ((Bhat[0]*uhat[1] + Bhat[1]*uhat[2] + Bhat[2]*uhat[3])*(Bhat[0]*uhat[1] + Bhat[1]*uhat[2] + Bhat[2]*uhat[3])))/((W)*(W));
	double x2 = x1/(eps*rho + press + rho + x1);
	double x3 = cs2*(1 - x2) + x2;
	double x4 = x0*x3 - 1;
	double x5 = 1/(x4);
	double x6 = x3 - 1;
	double x7 = vhat[idir]*x6;
	double x8 = sqrt(x3)*sqrt(fmax(0, (x0 - 1)*(-((vhat[idir])*(vhat[idir]))*x6 + x4)));
	*cm = x5*(x7 - x8);
	*cp = x5*(x7 + x8);
}

static void KOKKOS_INLINE_FUNCTION
hllc_get_contact_speed(
	int idir,
	const double FstildeHLL[3],
	const double stildeHLL[3],
	double FtauHLL,
	double FdensHLL,
	double tauHLL,
	double densHLL,
	double * __restrict__ lambda_c
)
{
	double x0 = FdensHLL + FtauHLL;
	double x1 = FstildeHLL[idir] + densHLL + tauHLL;
	*lambda_c = (1.0/2.0)*(x1 - sqrt(fmax(0, -4*stildeHLL[idir]*x0 + ((x1)*(x1)))))/x0;
}

static void KOKKOS_INLINE_FUNCTION
hllc_get_interface_velocity(
	double alp,
	int idir,
	const double gammaUU[3][3],
	const double betaU[3],
	double * __restrict__ lambda_interface
)
{
	*lambda_interface = betaU[idir]/(alp*sqrt(gammaUU[idir][idir]));
}

static void KOKKOS_INLINE_FUNCTION
hllc_get_central_state_and_fluxes(
	int idir,
	const double FstildeHLL[3],
	double FtauHLL,
	double FdensHLL,
	double lambda_c,
	double lambda,
	double densLR,
	double tauLR,
	const double stildeLR[3],
	const double vhatLR[3],
	double pressLR,
	double FdensLR,
	double FtauLR,
	const double FstildeLR[3],
	double * __restrict__ dens_c,
	double (*stilde_c)[3],
	double * __restrict__ tau_c,
	double * __restrict__ fdens_c,
	double (*fstilde_c)[3],
	double * __restrict__ ftau_c
)
{
	double x0 = lambda - vhatLR[idir];
	double x1 = 1/(lambda - lambda_c);
	double x2 = x0*x1;
	double x3 = lambda_c*(FdensHLL + FtauHLL);
	double x4 = -FstildeHLL[idir] + pressLR + x3;
	double x5 = x1*(stildeLR[0]*x0 - x4*((idir == 0) ? (
   1
)
: (
   0
)));
	double x6 = x1*(stildeLR[1]*x0 - x4*((idir == 1) ? (
   1
)
: (
   0
)));
	double x7 = x1*(stildeLR[2]*x0 - x4*((idir == 2) ? (
   1
)
: (
   0
)));
	double x8 = x1*(lambda_c*(FstildeHLL[idir] - x3) - pressLR*vhatLR[idir] + tauLR*x0);
	*dens_c = densLR*x2;
	(*stilde_c)[0] = x5;
	(*stilde_c)[1] = x6;
	(*stilde_c)[2] = x7;
	*tau_c = x8;
	*fdens_c = FdensLR - densLR*lambda*(1 - x2);
	(*fstilde_c)[0] = FstildeLR[0] - lambda*(stildeLR[0] - x5);
	(*fstilde_c)[1] = FstildeLR[1] - lambda*(stildeLR[1] - x6);
	(*fstilde_c)[2] = FstildeLR[2] - lambda*(stildeLR[2] - x7);
	*ftau_c = FtauLR - lambda*(tauLR - x8);
}

static void KOKKOS_INLINE_FUNCTION
hllc_transform_fluxes_to_grid_frame(
	double alp,
	int idir,
	const double eUU[4][4],
	const double edd[4][4],
	double dens_int,
	double fdens_int,
	const double stilde_int[3],
	const double fstilde_int[3],
	double tau_int,
	double ftau_int,
	double * __restrict__ fdens,
	double (*fstilde)[3],
	double * __restrict__ ftau
)
{
	double x0 = eUU[0][idir+1]*stilde_int[0] + eUU[idir+1][idir+1]*fstilde_int[0];
	double x1 = eUU[0][idir+1]*stilde_int[1] + eUU[idir+1][idir+1]*fstilde_int[1];
	double x2 = eUU[0][idir+1]*stilde_int[2] + eUU[idir+1][idir+1]*fstilde_int[2];
	*fdens = alp*(dens_int*eUU[0][idir+1] + eUU[idir+1][idir+1]*fdens_int);
	(*fstilde)[0] = alp*(edd[1][1]*x0 + edd[2][1]*x1 + edd[3][1]*x2);
	(*fstilde)[1] = alp*(edd[1][2]*x0 + edd[2][2]*x1 + edd[3][2]*x2);
	(*fstilde)[2] = alp*(edd[1][3]*x0 + edd[2][3]*x1 + edd[3][3]*x2);
	*ftau = alp*(eUU[0][idir+1]*tau_int + eUU[idir+1][idir+1]*ftau_int);
}

#endif 
