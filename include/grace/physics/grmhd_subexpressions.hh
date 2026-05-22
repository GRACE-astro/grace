
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
	double alp,
	const double betau[3],
	const double gdd[6],
	const double Bvec[3],
	const double zvec[3],
	double W,
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
	double alp,
	const double betau[3],
	const double zvec[3],
	double W,
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
	double alp,
	const double betau[3],
	double rho,
	double press,
	double eps,
	double cs2,
	double W,
	double b2,
	const double vtildeu[3],
	double guuDD,
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
	double alp,
	const double betau[3],
	const double gdd[6],
	double rho,
	double press,
	double eps,
	const double zvec[3],
	double s,
	double W,
	double b2,
	const double smallbu[4],
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
	(*fstilde)[0] = alp*(-smallbu[1+idir]*x5 + x19*((idir == 0) ? (1) : (0)) + x21*(gdd[0]*vtildeu[0] + gdd[1]*vtildeu[1] + gdd[2]*vtildeu[2] + x4));
	(*fstilde)[1] = alp*(-smallbu[1+idir]*x14 + x19*((idir == 1) ? (1) : (0)) + x21*(gdd[1]*vtildeu[0] + gdd[3]*vtildeu[1] + gdd[4]*vtildeu[2] + x13));
	(*fstilde)[2] = alp*(-smallbu[1+idir]*x16 + x19*((idir == 2) ? (1) : (0)) + x21*(gdd[2]*vtildeu[0] + gdd[4]*vtildeu[1] + gdd[5]*vtildeu[2] + x15));
	*fentstar = vtildeu[idir]*x17;
}

static void KOKKOS_INLINE_FUNCTION
grmhd_get_geom_sources(
	double alp,
	const double betau[3],
	const double gdd[6],
	const double guu[6],
	const double Kdd[6],
	const double dalp_dx[3],
	const double dgdd_dx[18],
	const double dbetau_dx[9],
	double rho,
	double press,
	double eps,
	const double Bvec[3],
	const double zvec[3],
	double W,
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
	double x32 = ((betau[0])*(betau[0]));
	double x33 = x31*x32;
	double x34 = guu[0] - x33;
	double x35 = ((x13)*(x13));
	double x36 = 2*x18;
	double x37 = (1.0/2.0)*alp;
	double x38 = ((x22)*(x22))*x29;
	double x39 = ((betau[1])*(betau[1]));
	double x40 = x31*x39;
	double x41 = guu[3] - x40;
	double x42 = ((x20)*(x20));
	double x43 = 2*x24;
	double x44 = ((x27)*(x27))*x29;
	double x45 = ((betau[2])*(betau[2]));
	double x46 = x31*x45;
	double x47 = guu[5] - x46;
	double x48 = ((x26)*(x26));
	double x49 = 2*x28;
	double x50 = betau[0]*x31;
	double x51 = betau[1]*x50;
	double x52 = guu[1] - x51;
	double x53 = x16*x29;
	double x54 = x22*x53;
	double x55 = betau[2]*x50;
	double x56 = guu[2] - x55;
	double x57 = x27*x53;
	double x58 = betau[1]*betau[2]*x31;
	double x59 = guu[4] - x58;
	double x60 = x22*x27*x29;
	double x61 = x31*x6;
	double x62 = alp*(-x30 + x34*x7 + x35*x61);
	double x63 = alp*(-x38 + x41*x7 + x42*x61);
	double x64 = alp*(-x44 + x47*x7 + x48*x61);
	double x65 = 2*alp;
	double x66 = betau[1]*dgdd_dx[1];
	double x67 = 2*betau[0];
	double x68 = betau[2]*dgdd_dx[2];
	double x69 = dbetau_dx[0]*gdd[0];
	double x70 = dbetau_dx[1]*gdd[1];
	double x71 = dbetau_dx[2]*gdd[2];
	double x72 = betau[2]*dgdd_dx[4];
	double x73 = 2*betau[1];
	double x74 = dbetau_dx[0]*gdd[1];
	double x75 = dbetau_dx[1]*gdd[3];
	double x76 = dbetau_dx[2]*gdd[4];
	double x77 = dbetau_dx[0]*gdd[2];
	double x78 = 2*betau[2];
	double x79 = dbetau_dx[1]*gdd[4];
	double x80 = dbetau_dx[2]*gdd[5];
	double x81 = x8*x9;
	double x82 = x13*x61;
	double x83 = x65*(x20*x82 + x52*x7 - x54);
	double x84 = x65*(x26*x82 + x56*x7 - x57);
	double x85 = x65*(x20*x26*x61 + x59*x7 - x60);
	double x86 = betau[1]*dgdd_dx[7];
	double x87 = betau[2]*dgdd_dx[8];
	double x88 = dbetau_dx[3]*gdd[0];
	double x89 = dbetau_dx[4]*gdd[1];
	double x90 = dbetau_dx[5]*gdd[2];
	double x91 = betau[2]*dgdd_dx[10];
	double x92 = dbetau_dx[3]*gdd[1];
	double x93 = dbetau_dx[4]*gdd[3];
	double x94 = dbetau_dx[5]*gdd[4];
	double x95 = dbetau_dx[3]*gdd[2];
	double x96 = dbetau_dx[4]*gdd[4];
	double x97 = dbetau_dx[5]*gdd[5];
	double x98 = betau[1]*dgdd_dx[13];
	double x99 = betau[2]*dgdd_dx[14];
	double x100 = dbetau_dx[6]*gdd[0];
	double x101 = dbetau_dx[7]*gdd[1];
	double x102 = dbetau_dx[8]*gdd[2];
	double x103 = betau[2]*dgdd_dx[16];
	double x104 = dbetau_dx[6]*gdd[1];
	double x105 = dbetau_dx[7]*gdd[3];
	double x106 = dbetau_dx[8]*gdd[4];
	double x107 = dbetau_dx[6]*gdd[2];
	double x108 = dbetau_dx[7]*gdd[4];
	double x109 = dbetau_dx[8]*gdd[5];
	*dtau = Kdd[0]*x37*(-x10*x36 + 2*x2*x31*x35*x5 - x30 - x33*x8 + x34*x7) + Kdd[1]*alp*(-x10*x24 + 2*x13*x2*x20*x31*x5 - x18*x19 - x51*x8 + x52*x7 - x54) + Kdd[2]*alp*(-x10*x28 + 2*x13*x2*x26*x31*x5 - x18*x25 - x55*x8 + x56*x7 - x57) + Kdd[3]*x37*(-x19*x43 + 2*x2*x31*x42*x5 - x38 - x40*x8 + x41*x7) + Kdd[4]*alp*(-x19*x28 + 2*x2*x20*x26*x31*x5 - x24*x25 - x58*x8 + x59*x7 - x60) + Kdd[5]*x37*(2*x2*x31*x48*x5 - x25*x49 - x44 - x46*x8 + x47*x7) + (1.0/2.0)*dalp_dx[0]*(x10*x8 + x18) + (1.0/2.0)*dalp_dx[1]*(x19*x8 + x24) + (1.0/2.0)*dalp_dx[2]*(x25*x8 + x28);
	(*dstilde)[0] = (1.0/4.0)*dgdd_dx[0]*x62 + (1.0/4.0)*dgdd_dx[1]*x83 + (1.0/4.0)*dgdd_dx[2]*x84 + (1.0/4.0)*dgdd_dx[3]*x63 + (1.0/4.0)*dgdd_dx[4]*x85 + (1.0/4.0)*dgdd_dx[5]*x64 - 1.0/4.0*x36*(betau[0]*dgdd_dx[0] + x66 + x68 + x69 + x70 + x71) - 1.0/4.0*x43*(betau[0]*dgdd_dx[1] + betau[1]*dgdd_dx[3] + x72 + x74 + x75 + x76) - 1.0/4.0*x49*(betau[0]*dgdd_dx[2] + betau[1]*dgdd_dx[4] + betau[2]*dgdd_dx[5] + x77 + x79 + x80) - 1.0/4.0*x81*(-dalp_dx[0]*x65 + dgdd_dx[0]*x32 + dgdd_dx[3]*x39 + dgdd_dx[5]*x45 + x66*x67 + x67*x68 + x67*x69 + x67*x70 + x67*x71 + x72*x73 + x73*x74 + x73*x75 + x73*x76 + x77*x78 + x78*x79 + x78*x80);
	(*dstilde)[1] = (1.0/4.0)*dgdd_dx[10]*x85 + (1.0/4.0)*dgdd_dx[11]*x64 + (1.0/4.0)*dgdd_dx[6]*x62 + (1.0/4.0)*dgdd_dx[7]*x83 + (1.0/4.0)*dgdd_dx[8]*x84 + (1.0/4.0)*dgdd_dx[9]*x63 - 1.0/4.0*x36*(betau[0]*dgdd_dx[6] + x86 + x87 + x88 + x89 + x90) - 1.0/4.0*x43*(betau[0]*dgdd_dx[7] + betau[1]*dgdd_dx[9] + x91 + x92 + x93 + x94) - 1.0/4.0*x49*(betau[0]*dgdd_dx[8] + betau[1]*dgdd_dx[10] + betau[2]*dgdd_dx[11] + x95 + x96 + x97) - 1.0/4.0*x81*(-dalp_dx[1]*x65 + dgdd_dx[11]*x45 + dgdd_dx[6]*x32 + dgdd_dx[9]*x39 + x67*x86 + x67*x87 + x67*x88 + x67*x89 + x67*x90 + x73*x91 + x73*x92 + x73*x93 + x73*x94 + x78*x95 + x78*x96 + x78*x97);
	(*dstilde)[2] = (1.0/4.0)*dgdd_dx[12]*x62 + (1.0/4.0)*dgdd_dx[13]*x83 + (1.0/4.0)*dgdd_dx[14]*x84 + (1.0/4.0)*dgdd_dx[15]*x63 + (1.0/4.0)*dgdd_dx[16]*x85 + (1.0/4.0)*dgdd_dx[17]*x64 - 1.0/4.0*x36*(betau[0]*dgdd_dx[12] + x100 + x101 + x102 + x98 + x99) - 1.0/4.0*x43*(betau[0]*dgdd_dx[13] + betau[1]*dgdd_dx[15] + x103 + x104 + x105 + x106) - 1.0/4.0*x49*(betau[0]*dgdd_dx[14] + betau[1]*dgdd_dx[16] + betau[2]*dgdd_dx[17] + x107 + x108 + x109) - 1.0/4.0*x81*(-dalp_dx[2]*x65 + dgdd_dx[12]*x32 + dgdd_dx[15]*x39 + dgdd_dx[17]*x45 + x100*x67 + x101*x67 + x102*x67 + x103*x73 + x104*x73 + x105*x73 + x106*x73 + x107*x78 + x108*x78 + x109*x78 + x67*x98 + x67*x99);
}

static void KOKKOS_INLINE_FUNCTION
grmhd_get_conserved(
	double alp,
	const double betau[3],
	const double gdd[6],
	double rho,
	double press,
	double eps,
	const double zvec[3],
	double s,
	double W,
	double b2,
	const double smallbu[4],
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
	double alp,
	const double betau[3],
	const double gdd[6],
	const double guu[6],
	double rho,
	double press,
	double eps,
	const double Bvec[3],
	const double zvec[3],
	double W,
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
