
/****************************************************************************/
/*                     GRMHD helpers, SymPy generated                       */
/****************************************************************************/
#ifndef GRACE_GRMHD_SUBEXPR_HH
#define GRACE_GRMHD_SUBEXPR_HH

#include <Kokkos_Core.hpp>

static void KOKKOS_INLINE_FUNCTION
grmhd_get_W(
	const double gdd[6],
	const double zvec[3],
	double * __restrict__ W
)
{
	*W = sqrt(zvec[0]*(gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2]) + zvec[1]*(gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2]) + zvec[2]*(gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2]) + 1);
}

static void KOKKOS_INLINE_FUNCTION
grmhd_get_smallbu_smallb2(
	const double betau[3],
	const double gdd[6],
	const double Bvec[3],
	const double zvec[3],
	double W,
	double alp,
	double (*smallb)[4],
	double * __restrict__ smallb2
)
{
	double x0 = Bvec[0]*(gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2]) + Bvec[1]*(gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2]) + Bvec[2]*(gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2]);
	double x1 = x0/alp;
	double x2 = 1/(W);
	double x3 = alp*x2;
	double x4 = W*x1;
	(*smallb)[0] = x1;
	(*smallb)[1] = x2*(Bvec[0] - x4*(betau[0] - x3*zvec[0]));
	(*smallb)[2] = x2*(Bvec[1] - x4*(betau[1] - x3*zvec[1]));
	(*smallb)[3] = x2*(Bvec[2] - x4*(betau[2] - x3*zvec[2]));
	*smallb2 = (Bvec[0]*(Bvec[0]*gdd[0] + Bvec[1]*gdd[1] + Bvec[2]*gdd[2]) + Bvec[1]*(Bvec[0]*gdd[1] + Bvec[1]*gdd[3] + Bvec[2]*gdd[4]) + Bvec[2]*(Bvec[0]*gdd[2] + Bvec[1]*gdd[4] + Bvec[2]*gdd[5]) + ((x0)*(x0)))/((W)*(W));
}

static void KOKKOS_INLINE_FUNCTION
grmhd_get_vtildeu(
	const double betau[3],
	double W,
	const double zvec[3],
	double alp,
	double (*vtilde)[3]
)
{
	double x0 = 1/(W);
	(*vtilde)[0] = alp*x0*zvec[0] - betau[0];
	(*vtilde)[1] = alp*x0*zvec[1] - betau[1];
	(*vtilde)[2] = alp*x0*zvec[2] - betau[2];
}

static void KOKKOS_INLINE_FUNCTION
grmhd_get_cm_cp(
	double cs2,
	const double vtildeu[3],
	double b2,
	const double betau[3],
	double W,
	double eps,
	double rho,
	double guuDD,
	double alp,
	double press,
	int idir,
	double * __restrict__ cm,
	double * __restrict__ cp
)
{
	double x0 = ((alp)*(alp));
	double x1 = 1/(x0);
	double x2 = b2/(b2 + eps*rho + press + rho);
	double x3 = x2 - 1;
	double x4 = -cs2*x3;
	double x5 = x2 + x4;
	double x6 = ((W)*(W));
	double x7 = x3 + x4;
	double x8 = -x6*x7;
	double x9 = sqrt(fmax(0, 4*x1*(x1*((betau[idir]*x5 - vtildeu[idir]*x8)*(betau[idir]*x5 - vtildeu[idir]*x8)) - (x5 + x8)*(((vtildeu[idir])*(vtildeu[idir]))*x1*x8 - x5*(-((betau[idir])*(betau[idir]))*x1 + guuDD)))));
	double x10 = cs2*x3;
	double x11 = 2*x1;
	double x12 = x6*x7;
	double x13 = betau[idir]*x11*(-x10 + x2) + vtildeu[idir]*x11*x12;
	double x14 = (1.0/2.0)*x0/(x10 + x12 - x2);
	double x15 = x14*(x13 + x9);
	double x16 = x14*(x13 - x9);
	*cm = fmin(x15, x16);
	*cp = fmax(x15, x16);
}

static void KOKKOS_INLINE_FUNCTION
grmhd_get_fluxes(
	double W,
	double rho,
	const double smallbu[4],
	double b2,
	double alp,
	double eps,
	double press,
	const double betau[3],
	const double zvec[3],
	const double gdd[6],
	double s,
	const double vtildeu[3],
	int idir,
	double * __restrict__ dens,
	double * __restrict__ tau,
	double (*stilde)[3],
	double * __restrict__ entstar,
	double * __restrict__ fD,
	double * __restrict__ ftau,
	double (*fstilde)[3],
	double * __restrict__ fentstar
)
{
	double x0 = W*rho;
	double x1 = ((alp)*(alp));
	double x2 = ((W)*(W));
	double x3 = b2 + eps*rho + press + rho;
	double x4 = betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2];
	double x5 = gdd[0]*smallbu[1] + gdd[1]*smallbu[2] + gdd[2]*smallbu[3] + smallbu[0]*x4;
	double x6 = alp/W;
	double x7 = betau[0] - x6*zvec[0];
	double x8 = betau[1] - x6*zvec[1];
	double x9 = betau[2] - x6*zvec[2];
	double x10 = 1/(x1);
	double x11 = x2*x3;
	double x12 = x10*x11;
	double x13 = betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4];
	double x14 = gdd[1]*smallbu[1] + gdd[3]*smallbu[2] + gdd[4]*smallbu[3] + smallbu[0]*x13;
	double x15 = betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5];
	double x16 = gdd[2]*smallbu[1] + gdd[4]*smallbu[2] + gdd[5]*smallbu[3] + smallbu[0]*x15;
	double x17 = s*x0;
	double x18 = vtildeu[idir]*x0;
	double x19 = (1.0/2.0)*b2 + press;
	double x20 = vtildeu[idir]*x11;
	double x21 = x10*x20;
	*dens = x0;
	*tau = -1.0/2.0*b2 - press - ((smallbu[0])*(smallbu[0]))*x1 - x0 + x2*x3;
	(*stilde)[0] = -alp*(smallbu[0]*x5 + x12*(gdd[0]*x7 + gdd[1]*x8 + gdd[2]*x9 - x4));
	(*stilde)[1] = -alp*(smallbu[0]*x14 + x12*(gdd[1]*x7 + gdd[3]*x8 + gdd[4]*x9 - x13));
	(*stilde)[2] = -alp*(smallbu[0]*x16 + x12*(gdd[2]*x7 + gdd[4]*x8 + gdd[5]*x9 - x15));
	*entstar = x17;
	*fD = x18;
	*ftau = betau[idir]*x19 - smallbu[0]*smallbu[1+idir]*x1 - x18 + x20;
	(*fstilde)[0] = alp*(-smallbu[1+idir]*x5 + x19*((idir == 0) ? (
   1
)
: (
   0
)) + x21*(gdd[0]*vtildeu[0] + gdd[1]*vtildeu[1] + gdd[2]*vtildeu[2] + x4));
	(*fstilde)[1] = alp*(-smallbu[1+idir]*x14 + x19*((idir == 1) ? (
   1
)
: (
   0
)) + x21*(gdd[1]*vtildeu[0] + gdd[3]*vtildeu[1] + gdd[4]*vtildeu[2] + x13));
	(*fstilde)[2] = alp*(-smallbu[1+idir]*x16 + x19*((idir == 2) ? (
   1
)
: (
   0
)) + x21*(gdd[2]*vtildeu[0] + gdd[4]*vtildeu[1] + gdd[5]*vtildeu[2] + x15));
	*fentstar = vtildeu[idir]*x17;
}

static void KOKKOS_INLINE_FUNCTION
grmhd_get_geom_sources(
	const double betau[3],
	const double zvec[3],
	const double Kdd[6],
	const double dalpha_dx[3],
	const double guu[6],
	const double Bvec[3],
	const double gdd[6],
	double eps,
	double alp,
	double W,
	double rho,
	double press,
	const double dgdd_dx[18],
	const double dbetau_dx[9],
	double * __restrict__ dtau,
	double (*dstilde)[3]
)
{
	double x0 = Bvec[0]*(gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2]) + Bvec[1]*(gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2]) + Bvec[2]*(gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2]);
	double x1 = ((x0)*(x0));
	double x2 = ((W)*(W));
	double x3 = 1/(x2);
	double x4 = x3*(Bvec[0]*(Bvec[0]*gdd[0] + Bvec[1]*gdd[1] + Bvec[2]*gdd[2]) + Bvec[1]*(Bvec[0]*gdd[1] + Bvec[1]*gdd[3] + Bvec[2]*gdd[4]) + Bvec[2]*(Bvec[0]*gdd[2] + Bvec[1]*gdd[4] + Bvec[2]*gdd[5]) + x1);
	double x5 = eps*rho + press + rho + x4;
	double x6 = 2*x2*x5;
	double x7 = 2*press + x4;
	double x8 = 2*x1 - x6 + x7;
	double x9 = 1/(alp);
	double x10 = betau[0]*x9;
	double x11 = 1/(W);
	double x12 = alp*x11;
	double x13 = betau[0] - x12*zvec[0];
	double x14 = x13*x9;
	double x15 = W*x0;
	double x16 = Bvec[0] - x14*x15;
	double x17 = 2*x0*x11;
	double x18 = -x10*x7 + x14*x6 + x16*x17;
	double x19 = betau[1]*x9;
	double x20 = betau[1] - x12*zvec[1];
	double x21 = x15*x9;
	double x22 = Bvec[1] - x20*x21;
	double x23 = x6*x9;
	double x24 = x17*x22 - x19*x7 + x20*x23;
	double x25 = betau[2]*x9;
	double x26 = betau[2] - x12*zvec[2];
	double x27 = Bvec[2] - x21*x26;
	double x28 = x17*x27 + x23*x26 - x25*x7;
	double x29 = 2*x3;
	double x30 = ((x16)*(x16))*x29;
	double x31 = 1/(((alp)*(alp)));
	double x32 = ((betau[0])*(betau[0]))*x31;
	double x33 = guu[0] - x32;
	double x34 = ((x13)*(x13));
	double x35 = 2*x18;
	double x36 = (1.0/2.0)*alp;
	double x37 = ((x22)*(x22))*x29;
	double x38 = ((betau[1])*(betau[1]))*x31;
	double x39 = guu[3] - x38;
	double x40 = ((x20)*(x20));
	double x41 = 2*x24;
	double x42 = ((x27)*(x27))*x29;
	double x43 = ((betau[2])*(betau[2]))*x31;
	double x44 = guu[5] - x43;
	double x45 = ((x26)*(x26));
	double x46 = 2*x28;
	double x47 = betau[0]*x31;
	double x48 = betau[1]*x47;
	double x49 = guu[1] - x48;
	double x50 = x16*x29;
	double x51 = x22*x50;
	double x52 = betau[2]*x47;
	double x53 = guu[2] - x52;
	double x54 = x27*x50;
	double x55 = betau[1]*betau[2]*x31;
	double x56 = guu[4] - x55;
	double x57 = x22*x27*x29;
	double x58 = x31*x6;
	double x59 = alp*(-x30 + x33*x7 + x34*x58);
	double x60 = alp*(-x37 + x39*x7 + x40*x58);
	double x61 = alp*(-x42 + x44*x7 + x45*x58);
	double x62 = betau[0]*dgdd_dx[0] + betau[1]*dgdd_dx[1] + betau[2]*dgdd_dx[2] + dbetau_dx[0]*gdd[0] + dbetau_dx[1]*gdd[1] + dbetau_dx[2]*gdd[2];
	double x63 = betau[0]*dgdd_dx[1] + betau[1]*dgdd_dx[3] + betau[2]*dgdd_dx[4] + dbetau_dx[0]*gdd[1] + dbetau_dx[1]*gdd[3] + dbetau_dx[2]*gdd[4];
	double x64 = betau[0]*dgdd_dx[2] + betau[1]*dgdd_dx[4] + betau[2]*dgdd_dx[5] + dbetau_dx[0]*gdd[2] + dbetau_dx[1]*gdd[4] + dbetau_dx[2]*gdd[5];
	double x65 = x13*x58;
	double x66 = 2*alp;
	double x67 = x66*(x20*x65 + x49*x7 - x51);
	double x68 = x66*(x26*x65 + x53*x7 - x54);
	double x69 = x66*(x20*x26*x58 + x56*x7 - x57);
	double x70 = betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2];
	double x71 = betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4];
	double x72 = betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5];
	double x73 = x8*x9;
	double x74 = betau[0]*dgdd_dx[6] + betau[1]*dgdd_dx[7] + betau[2]*dgdd_dx[8] + dbetau_dx[3]*gdd[0] + dbetau_dx[4]*gdd[1] + dbetau_dx[5]*gdd[2];
	double x75 = betau[0]*dgdd_dx[7] + betau[1]*dgdd_dx[9] + betau[2]*dgdd_dx[10] + dbetau_dx[3]*gdd[1] + dbetau_dx[4]*gdd[3] + dbetau_dx[5]*gdd[4];
	double x76 = betau[0]*dgdd_dx[8] + betau[1]*dgdd_dx[10] + betau[2]*dgdd_dx[11] + dbetau_dx[3]*gdd[2] + dbetau_dx[4]*gdd[4] + dbetau_dx[5]*gdd[5];
	double x77 = betau[0]*dgdd_dx[12] + betau[1]*dgdd_dx[13] + betau[2]*dgdd_dx[14] + dbetau_dx[6]*gdd[0] + dbetau_dx[7]*gdd[1] + dbetau_dx[8]*gdd[2];
	double x78 = betau[0]*dgdd_dx[13] + betau[1]*dgdd_dx[15] + betau[2]*dgdd_dx[16] + dbetau_dx[6]*gdd[1] + dbetau_dx[7]*gdd[3] + dbetau_dx[8]*gdd[4];
	double x79 = betau[0]*dgdd_dx[14] + betau[1]*dgdd_dx[16] + betau[2]*dgdd_dx[17] + dbetau_dx[6]*gdd[2] + dbetau_dx[7]*gdd[4] + dbetau_dx[8]*gdd[5];
	*dtau = Kdd[0]*x36*(-x10*x35 + 2*x2*x31*x34*x5 - x30 - x32*x8 + x33*x7) + Kdd[1]*alp*(2*x13*x2*x20*x31*x5 - x19*x35 - x48*x8 + x49*x7 - x51) + Kdd[2]*alp*(2*x13*x2*x26*x31*x5 - x25*x35 - x52*x8 + x53*x7 - x54) + Kdd[3]*x36*(-x19*x41 + 2*x2*x31*x40*x5 - x37 - x38*x8 + x39*x7) + Kdd[4]*alp*(2*x2*x20*x26*x31*x5 - x25*x41 - x55*x8 + x56*x7 - x57) + Kdd[5]*x36*(2*x2*x31*x45*x5 - x25*x46 - x42 - x43*x8 + x44*x7) + (1.0/2.0)*dalpha_dx[0]*(x10*x8 + x18) + (1.0/2.0)*dalpha_dx[1]*(x19*x8 + x24) + (1.0/2.0)*dalpha_dx[2]*(x25*x8 + x28);
	(*dstilde)[0] = (1.0/4.0)*dgdd_dx[0]*x59 + (1.0/4.0)*dgdd_dx[1]*x67 + (1.0/4.0)*dgdd_dx[2]*x68 + (1.0/4.0)*dgdd_dx[3]*x60 + (1.0/4.0)*dgdd_dx[4]*x69 + (1.0/4.0)*dgdd_dx[5]*x61 - 1.0/4.0*x35*x62 - 1.0/4.0*x41*x63 - 1.0/4.0*x46*x64 - 1.0/4.0*x73*(betau[0]*x62 + betau[1]*x63 + betau[2]*x64 - dalpha_dx[0]*x66 + dbetau_dx[0]*x70 + dbetau_dx[1]*x71 + dbetau_dx[2]*x72);
	(*dstilde)[1] = (1.0/4.0)*dgdd_dx[10]*x69 + (1.0/4.0)*dgdd_dx[11]*x61 + (1.0/4.0)*dgdd_dx[6]*x59 + (1.0/4.0)*dgdd_dx[7]*x67 + (1.0/4.0)*dgdd_dx[8]*x68 + (1.0/4.0)*dgdd_dx[9]*x60 - 1.0/4.0*x35*x74 - 1.0/4.0*x41*x75 - 1.0/4.0*x46*x76 - 1.0/4.0*x73*(betau[0]*x74 + betau[1]*x75 + betau[2]*x76 - dalpha_dx[1]*x66 + dbetau_dx[3]*x70 + dbetau_dx[4]*x71 + dbetau_dx[5]*x72);
	(*dstilde)[2] = (1.0/4.0)*dgdd_dx[12]*x59 + (1.0/4.0)*dgdd_dx[13]*x67 + (1.0/4.0)*dgdd_dx[14]*x68 + (1.0/4.0)*dgdd_dx[15]*x60 + (1.0/4.0)*dgdd_dx[16]*x69 + (1.0/4.0)*dgdd_dx[17]*x61 - 1.0/4.0*x35*x77 - 1.0/4.0*x41*x78 - 1.0/4.0*x46*x79 - 1.0/4.0*x73*(betau[0]*x77 + betau[1]*x78 + betau[2]*x79 - dalpha_dx[2]*x66 + dbetau_dx[6]*x70 + dbetau_dx[7]*x71 + dbetau_dx[8]*x72);
}

static void KOKKOS_INLINE_FUNCTION
grmhd_get_conserved(
	double W,
	double rho,
	const double smallbu[4],
	double b2,
	double alp,
	double eps,
	double press,
	const double betau[3],
	const double zvec[3],
	const double gdd[6],
	double s,
	double * __restrict__ dens,
	double * __restrict__ tau,
	double (*stilde)[3],
	double * __restrict__ entstar
)
{
	double x0 = W*rho;
	double x1 = ((alp)*(alp));
	double x2 = ((W)*(W));
	double x3 = b2 + eps*rho + press + rho;
	double x4 = betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2];
	double x5 = alp/W;
	double x6 = betau[0] - x5*zvec[0];
	double x7 = betau[1] - x5*zvec[1];
	double x8 = betau[2] - x5*zvec[2];
	double x9 = x2*x3/x1;
	double x10 = betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4];
	double x11 = betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5];
	*dens = x0;
	*tau = -1.0/2.0*b2 - press - ((smallbu[0])*(smallbu[0]))*x1 - x0 + x2*x3;
	(*stilde)[0] = -alp*(smallbu[0]*(gdd[0]*smallbu[1] + gdd[1]*smallbu[2] + gdd[2]*smallbu[3] + smallbu[0]*x4) + x9*(gdd[0]*x6 + gdd[1]*x7 + gdd[2]*x8 - x4));
	(*stilde)[1] = -alp*(smallbu[0]*(gdd[1]*smallbu[1] + gdd[3]*smallbu[2] + gdd[4]*smallbu[3] + smallbu[0]*x10) + x9*(gdd[1]*x6 + gdd[3]*x7 + gdd[4]*x8 - x10));
	(*stilde)[2] = -alp*(smallbu[0]*(gdd[2]*smallbu[1] + gdd[4]*smallbu[2] + gdd[5]*smallbu[3] + smallbu[0]*x11) + x9*(gdd[2]*x6 + gdd[4]*x7 + gdd[5]*x8 - x11));
	*entstar = s*x0;
}

static void KOKKOS_INLINE_FUNCTION
grmhd_get_Tupmunu(
	const double betau[3],
	const double zvec[3],
	const double guu[6],
	const double Bvec[3],
	const double gdd[6],
	double eps,
	double alp,
	double W,
	double rho,
	double press,
	double (*Tuu)[10]
)
{
	double x0 = 1/(((alp)*(alp)));
	double x1 = Bvec[0]*(gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2]) + Bvec[1]*(gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2]) + Bvec[2]*(gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2]);
	double x2 = ((x1)*(x1));
	double x3 = ((W)*(W));
	double x4 = 1/(x3);
	double x5 = x4*(Bvec[0]*(Bvec[0]*gdd[0] + Bvec[1]*gdd[1] + Bvec[2]*gdd[2]) + Bvec[1]*(Bvec[0]*gdd[1] + Bvec[1]*gdd[3] + Bvec[2]*gdd[4]) + Bvec[2]*(Bvec[0]*gdd[2] + Bvec[1]*gdd[4] + Bvec[2]*gdd[5]) + x2);
	double x6 = eps*rho + press + rho + x5;
	double x7 = 1/(alp);
	double x8 = 2*press + x5;
	double x9 = 1/(W);
	double x10 = alp*x9;
	double x11 = betau[0] - x10*zvec[0];
	double x12 = x11*x7;
	double x13 = W*x1;
	double x14 = Bvec[0] - x12*x13;
	double x15 = x1*x9;
	double x16 = x3*x6;
	double x17 = x7*((1.0/2.0)*betau[0]*x7*x8 - x12*x16 - x14*x15);
	double x18 = betau[1] - x10*zvec[1];
	double x19 = x18*x7;
	double x20 = Bvec[1] - x13*x19;
	double x21 = x7*((1.0/2.0)*betau[1]*x7*x8 - x15*x20 - x16*x19);
	double x22 = betau[2] - x10*zvec[2];
	double x23 = x22*x7;
	double x24 = Bvec[2] - x13*x23;
	double x25 = x7*((1.0/2.0)*betau[2]*x7*x8 - x15*x24 - x16*x23);
	double x26 = (1.0/2.0)*x8;
	double x27 = x0*x16;
	double x28 = betau[0]*x0;
	double x29 = x14*x4;
	double x30 = x11*x27;
	double x31 = x18*x30 - x20*x29 + x26*(-betau[1]*x28 + guu[1]);
	double x32 = x22*x30 - x24*x29 + x26*(-betau[2]*x28 + guu[2]);
	double x33 = x18*x22*x27 - x20*x24*x4 + x26*(-betau[1]*betau[2]*x0 + guu[4]);
	(*Tuu)[0] = x0*(-press - x2 + x3*x6 - 1.0/2.0*x5);
	(*Tuu)[1] = x17;
	(*Tuu)[2] = x21;
	(*Tuu)[3] = x25;
	(*Tuu)[4] = ((x11)*(x11))*x27 - ((x14)*(x14))*x4 + x26*(-((betau[0])*(betau[0]))*x0 + guu[0]);
	(*Tuu)[5] = x31;
	(*Tuu)[6] = x32;
	(*Tuu)[7] = ((x18)*(x18))*x27 - ((x20)*(x20))*x4 + x26*(-((betau[1])*(betau[1]))*x0 + guu[3]);
	(*Tuu)[8] = x33;
	(*Tuu)[9] = ((x22)*(x22))*x27 - ((x24)*(x24))*x4 + x26*(-((betau[2])*(betau[2]))*x0 + guu[5]);
}

#endif 
