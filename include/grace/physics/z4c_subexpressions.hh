
/****************************************************************************/
/*                       Z4C helpers, SymPy generated                       */
/****************************************************************************/
#ifndef GRACE_Z4C_SUBEXPR_HH
#define GRACE_Z4C_SUBEXPR_HH

#include <Kokkos_Core.hpp>

static void KOKKOS_INLINE_FUNCTION
z4c_get_det_conf_metric(
	const double gtdd[6],
	double * __restrict__ detg
)
{
	*detg = gtdd[0]*gtdd[3]*gtdd[5] - gtdd[0]*((gtdd[4])*(gtdd[4])) - ((gtdd[1])*(gtdd[1]))*gtdd[5] + 2*gtdd[1]*gtdd[2]*gtdd[4] - ((gtdd[2])*(gtdd[2]))*gtdd[3];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_inverse_conf_metric(
	const double gtdd[6],
	double detg,
	double (*gtuu)[6]
)
{
	double x0 = 1/(detg);
	double x1 = -x0*(gtdd[1]*gtdd[5] - gtdd[2]*gtdd[4]);
	double x2 = x0*(gtdd[1]*gtdd[4] - gtdd[2]*gtdd[3]);
	double x3 = -x0*(gtdd[0]*gtdd[4] - gtdd[1]*gtdd[2]);
	(*gtuu)[0] = x0*(gtdd[3]*gtdd[5] - ((gtdd[4])*(gtdd[4])));
	(*gtuu)[1] = x1;
	(*gtuu)[2] = x2;
	(*gtuu)[3] = x0*(gtdd[0]*gtdd[5] - ((gtdd[2])*(gtdd[2])));
	(*gtuu)[4] = x3;
	(*gtuu)[5] = x0*(gtdd[0]*gtdd[3] - ((gtdd[1])*(gtdd[1])));
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Atuu(
	const double Atdd[6],
	const double gtuu[6],
	double (*Atuu)[6]
)
{
	double x0 = 2*Atdd[4];
	(*Atuu)[0] = Atdd[0]*((gtuu[0])*(gtuu[0])) + 2*Atdd[1]*gtuu[0]*gtuu[1] + 2*Atdd[2]*gtuu[0]*gtuu[2] + Atdd[3]*((gtuu[1])*(gtuu[1])) + Atdd[5]*((gtuu[2])*(gtuu[2])) + gtuu[1]*gtuu[2]*x0;
	(*Atuu)[1] = Atdd[0]*gtuu[0]*gtuu[1] + Atdd[1]*gtuu[0]*gtuu[3] + Atdd[1]*((gtuu[1])*(gtuu[1])) + Atdd[2]*gtuu[0]*gtuu[4] + Atdd[2]*gtuu[1]*gtuu[2] + Atdd[3]*gtuu[1]*gtuu[3] + Atdd[4]*gtuu[1]*gtuu[4] + Atdd[4]*gtuu[2]*gtuu[3] + Atdd[5]*gtuu[2]*gtuu[4];
	(*Atuu)[2] = Atdd[0]*gtuu[0]*gtuu[2] + Atdd[1]*gtuu[0]*gtuu[4] + Atdd[1]*gtuu[1]*gtuu[2] + Atdd[2]*gtuu[0]*gtuu[5] + Atdd[2]*((gtuu[2])*(gtuu[2])) + Atdd[3]*gtuu[1]*gtuu[4] + Atdd[4]*gtuu[1]*gtuu[5] + Atdd[4]*gtuu[2]*gtuu[4] + Atdd[5]*gtuu[2]*gtuu[5];
	(*Atuu)[3] = Atdd[0]*((gtuu[1])*(gtuu[1])) + 2*Atdd[1]*gtuu[1]*gtuu[3] + 2*Atdd[2]*gtuu[1]*gtuu[4] + Atdd[3]*((gtuu[3])*(gtuu[3])) + Atdd[5]*((gtuu[4])*(gtuu[4])) + gtuu[3]*gtuu[4]*x0;
	(*Atuu)[4] = Atdd[0]*gtuu[1]*gtuu[2] + Atdd[1]*gtuu[1]*gtuu[4] + Atdd[1]*gtuu[2]*gtuu[3] + Atdd[2]*gtuu[1]*gtuu[5] + Atdd[2]*gtuu[2]*gtuu[4] + Atdd[3]*gtuu[3]*gtuu[4] + Atdd[4]*gtuu[3]*gtuu[5] + Atdd[4]*((gtuu[4])*(gtuu[4])) + Atdd[5]*gtuu[4]*gtuu[5];
	(*Atuu)[5] = Atdd[0]*((gtuu[2])*(gtuu[2])) + 2*Atdd[1]*gtuu[2]*gtuu[4] + 2*Atdd[2]*gtuu[2]*gtuu[5] + Atdd[3]*((gtuu[4])*(gtuu[4])) + 2*Atdd[4]*gtuu[4]*gtuu[5] + Atdd[5]*((gtuu[5])*(gtuu[5]));
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Asqr(
	const double Atdd[6],
	const double Atuu[6],
	double * __restrict__ Asqr
)
{
	*Asqr = Atdd[0]*Atuu[0] + 2*Atdd[1]*Atuu[1] + 2*Atdd[2]*Atuu[2] + Atdd[3]*Atuu[3] + 2*Atdd[4]*Atuu[4] + Atdd[5]*Atuu[5];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_first_Christoffel(
	const double dgtdd_dx[18],
	double (*Gammatddd)[18]
)
{
	double x0 = (1.0/2.0)*dgtdd_dx[6];
	double x1 = (1.0/2.0)*dgtdd_dx[12];
	double x2 = (1.0/2.0)*dgtdd_dx[3];
	double x3 = (1.0/2.0)*dgtdd_dx[5];
	double x4 = (1.0/2.0)*dgtdd_dx[15];
	double x5 = (1.0/2.0)*dgtdd_dx[11];
	(*Gammatddd)[0] = (1.0/2.0)*dgtdd_dx[0];
	(*Gammatddd)[1] = x0;
	(*Gammatddd)[2] = x1;
	(*Gammatddd)[3] = dgtdd_dx[7] - x2;
	(*Gammatddd)[4] = (1.0/2.0)*(dgtdd_dx[13] - dgtdd_dx[4] + dgtdd_dx[8]);
	(*Gammatddd)[5] = dgtdd_dx[14] - x3;
	(*Gammatddd)[6] = dgtdd_dx[1] - x0;
	(*Gammatddd)[7] = x2;
	(*Gammatddd)[8] = (1.0/2.0)*(dgtdd_dx[13] + dgtdd_dx[4] - dgtdd_dx[8]);
	(*Gammatddd)[9] = (1.0/2.0)*dgtdd_dx[9];
	(*Gammatddd)[10] = x4;
	(*Gammatddd)[11] = dgtdd_dx[16] - x5;
	(*Gammatddd)[12] = dgtdd_dx[2] - x1;
	(*Gammatddd)[13] = (1.0/2.0)*(-dgtdd_dx[13] + dgtdd_dx[4] + dgtdd_dx[8]);
	(*Gammatddd)[14] = x3;
	(*Gammatddd)[15] = dgtdd_dx[10] - x4;
	(*Gammatddd)[16] = x5;
	(*Gammatddd)[17] = (1.0/2.0)*dgtdd_dx[17];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_second_Christoffel(
	const double gtuu[6],
	const double Gammatddd[18],
	double (*Gammatudd)[18]
)
{
	(*Gammatudd)[0] = Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1];
	(*Gammatudd)[1] = Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1];
	(*Gammatudd)[2] = Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1];
	(*Gammatudd)[3] = Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1];
	(*Gammatudd)[4] = Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0];
	(*Gammatudd)[5] = Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0];
	(*Gammatudd)[6] = Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3];
	(*Gammatudd)[7] = Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3];
	(*Gammatudd)[8] = Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3];
	(*Gammatudd)[9] = Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3];
	(*Gammatudd)[10] = Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1];
	(*Gammatudd)[11] = Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1];
	(*Gammatudd)[12] = Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4];
	(*Gammatudd)[13] = Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4];
	(*Gammatudd)[14] = Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4];
	(*Gammatudd)[15] = Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4];
	(*Gammatudd)[16] = Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2];
	(*Gammatudd)[17] = Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_contracted_Christoffel(
	const double gtuu[6],
	const double Gammatudd[18],
	double (*GammatDu)[3]
)
{
	double x0 = 2*gtuu[1];
	double x1 = 2*gtuu[2];
	double x2 = 2*gtuu[4];
	(*GammatDu)[0] = Gammatudd[0]*gtuu[0] + Gammatudd[1]*x0 + Gammatudd[2]*x1 + Gammatudd[3]*gtuu[3] + Gammatudd[4]*x2 + Gammatudd[5]*gtuu[5];
	(*GammatDu)[1] = Gammatudd[10]*x2 + Gammatudd[11]*gtuu[5] + Gammatudd[6]*gtuu[0] + Gammatudd[7]*x0 + Gammatudd[8]*x1 + Gammatudd[9]*gtuu[3];
	(*GammatDu)[2] = Gammatudd[12]*gtuu[0] + Gammatudd[13]*x0 + Gammatudd[14]*x1 + Gammatudd[15]*gtuu[3] + Gammatudd[16]*x2 + Gammatudd[17]*gtuu[5];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_DiDjalp(
	const double gtdd[6],
	double W,
	const double gtuu[6],
	const double Gammatudd[18],
	const double dW_dx[3],
	const double dalp_dx[3],
	const double ddalp_dx2[6],
	double (*W2DiDjalp)[6]
)
{
	double x0 = dW_dx[0]*dalp_dx[0];
	double x1 = dW_dx[0]*dalp_dx[1];
	double x2 = dW_dx[0]*dalp_dx[2];
	double x3 = dW_dx[1]*dalp_dx[0];
	double x4 = dW_dx[1]*dalp_dx[1];
	double x5 = dW_dx[1]*dalp_dx[2];
	double x6 = dW_dx[2]*dalp_dx[0];
	double x7 = dW_dx[2]*dalp_dx[1];
	double x8 = dW_dx[2]*dalp_dx[2];
	(*W2DiDjalp)[0] = -W*(W*(Gammatudd[0]*dalp_dx[0] + Gammatudd[12]*dalp_dx[2] + Gammatudd[6]*dalp_dx[1] - ddalp_dx2[0]) + gtdd[0]*gtuu[0]*x0 + gtdd[0]*gtuu[1]*x1 + gtdd[0]*gtuu[1]*x3 + gtdd[0]*gtuu[2]*x2 + gtdd[0]*gtuu[2]*x6 + gtdd[0]*gtuu[3]*x4 + gtdd[0]*gtuu[4]*x5 + gtdd[0]*gtuu[4]*x7 + gtdd[0]*gtuu[5]*x8 - 2*x0);
	(*W2DiDjalp)[1] = -W*(W*(Gammatudd[13]*dalp_dx[2] + Gammatudd[1]*dalp_dx[0] + Gammatudd[7]*dalp_dx[1] - ddalp_dx2[1]) + gtdd[1]*gtuu[0]*x0 + gtdd[1]*gtuu[1]*x1 + gtdd[1]*gtuu[1]*x3 + gtdd[1]*gtuu[2]*x2 + gtdd[1]*gtuu[2]*x6 + gtdd[1]*gtuu[3]*x4 + gtdd[1]*gtuu[4]*x5 + gtdd[1]*gtuu[4]*x7 + gtdd[1]*gtuu[5]*x8 - x1 - x3);
	(*W2DiDjalp)[2] = -W*(W*(Gammatudd[14]*dalp_dx[2] + Gammatudd[2]*dalp_dx[0] + Gammatudd[8]*dalp_dx[1] - ddalp_dx2[2]) + gtdd[2]*gtuu[0]*x0 + gtdd[2]*gtuu[1]*x1 + gtdd[2]*gtuu[1]*x3 + gtdd[2]*gtuu[2]*x2 + gtdd[2]*gtuu[2]*x6 + gtdd[2]*gtuu[3]*x4 + gtdd[2]*gtuu[4]*x5 + gtdd[2]*gtuu[4]*x7 + gtdd[2]*gtuu[5]*x8 - x2 - x6);
	(*W2DiDjalp)[3] = -W*(W*(Gammatudd[15]*dalp_dx[2] + Gammatudd[3]*dalp_dx[0] + Gammatudd[9]*dalp_dx[1] - ddalp_dx2[3]) + gtdd[3]*gtuu[0]*x0 + gtdd[3]*gtuu[1]*x1 + gtdd[3]*gtuu[1]*x3 + gtdd[3]*gtuu[2]*x2 + gtdd[3]*gtuu[2]*x6 + gtdd[3]*gtuu[3]*x4 + gtdd[3]*gtuu[4]*x5 + gtdd[3]*gtuu[4]*x7 + gtdd[3]*gtuu[5]*x8 - 2*x4);
	(*W2DiDjalp)[4] = -W*(W*(Gammatudd[10]*dalp_dx[1] + Gammatudd[16]*dalp_dx[2] + Gammatudd[4]*dalp_dx[0] - ddalp_dx2[4]) + gtdd[4]*gtuu[0]*x0 + gtdd[4]*gtuu[1]*x1 + gtdd[4]*gtuu[1]*x3 + gtdd[4]*gtuu[2]*x2 + gtdd[4]*gtuu[2]*x6 + gtdd[4]*gtuu[3]*x4 + gtdd[4]*gtuu[4]*x5 + gtdd[4]*gtuu[4]*x7 + gtdd[4]*gtuu[5]*x8 - x5 - x7);
	(*W2DiDjalp)[5] = -W*(W*(Gammatudd[11]*dalp_dx[1] + Gammatudd[17]*dalp_dx[2] + Gammatudd[5]*dalp_dx[0] - ddalp_dx2[5]) + gtdd[5]*gtuu[0]*x0 + gtdd[5]*gtuu[1]*x1 + gtdd[5]*gtuu[1]*x3 + gtdd[5]*gtuu[2]*x2 + gtdd[5]*gtuu[2]*x6 + gtdd[5]*gtuu[3]*x4 + gtdd[5]*gtuu[4]*x5 + gtdd[5]*gtuu[4]*x7 + gtdd[5]*gtuu[5]*x8 - 2*x8);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_DiDialp(
	const double gtuu[6],
	const double W2DiDjalp[6],
	double * __restrict__ DiDialp
)
{
	*DiDialp = W2DiDjalp[0]*gtuu[0] + 2*W2DiDjalp[1]*gtuu[1] + 2*W2DiDjalp[2]*gtuu[2] + W2DiDjalp[3]*gtuu[3] + 2*W2DiDjalp[4]*gtuu[4] + W2DiDjalp[5]*gtuu[5];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci(
	const double gtdd[6],
	double W,
	const double gtuu[6],
	const double Gammatddd[18],
	const double Gammatudd[18],
	const double GammatDu[3],
	const double dGammatu_dx[9],
	const double ddgtdd_dx2[36],
	double (*W2Rtdd)[6]
)
{
	double x0 = ((W)*(W));
	double x1 = (1.0/2.0)*gtuu[0];
	double x2 = (1.0/2.0)*gtuu[3];
	double x3 = (1.0/2.0)*gtuu[5];
	double x4 = 3*gtuu[0];
	double x5 = 3*gtuu[3];
	double x6 = 3*gtuu[5];
	double x7 = 2*Gammatddd[2];
	double x8 = 2*Gammatddd[4];
	double x9 = 2*Gammatddd[5];
	double x10 = 2*Gammatddd[1];
	double x11 = 2*Gammatddd[3];
	double x12 = Gammatddd[0]*Gammatudd[1];
	double x13 = Gammatddd[1]*Gammatudd[0];
	double x14 = Gammatddd[13]*Gammatudd[12];
	double x15 = Gammatddd[7]*Gammatudd[6];
	double x16 = Gammatddd[0]*Gammatudd[2];
	double x17 = Gammatddd[2]*Gammatudd[0];
	double x18 = Gammatddd[14]*Gammatudd[12];
	double x19 = Gammatddd[8]*Gammatudd[6];
	double x20 = Gammatddd[13]*Gammatudd[14];
	double x21 = Gammatddd[14]*Gammatudd[13];
	double x22 = Gammatddd[1]*Gammatudd[2];
	double x23 = Gammatddd[2]*Gammatudd[1];
	double x24 = Gammatddd[8]*Gammatudd[7];
	double x25 = Gammatddd[7]*Gammatudd[8];
	double x26 = (1.0/2.0)*gtdd[1];
	double x27 = (1.0/2.0)*dGammatu_dx[1];
	double x28 = (1.0/2.0)*dGammatu_dx[2];
	double x29 = (1.0/2.0)*gtdd[0];
	double x30 = (1.0/2.0)*gtdd[2];
	double x31 = (1.0/2.0)*GammatDu[0];
	double x32 = (1.0/2.0)*GammatDu[1];
	double x33 = (1.0/2.0)*GammatDu[2];
	double x34 = Gammatddd[9]*Gammatudd[7];
	double x35 = 2*x34;
	double x36 = Gammatddd[4]*Gammatudd[10];
	double x37 = Gammatddd[10]*Gammatudd[8];
	double x38 = Gammatddd[7]*Gammatudd[7];
	double x39 = Gammatddd[9]*Gammatudd[6] + x38;
	double x40 = Gammatddd[6]*Gammatudd[2];
	double x41 = Gammatddd[0]*Gammatudd[4] + Gammatddd[4]*Gammatudd[0];
	double x42 = Gammatddd[1]*Gammatudd[10];
	double x43 = Gammatddd[10]*Gammatudd[6];
	double x44 = x25 + x43;
	double x45 = Gammatddd[8]*Gammatudd[14];
	double x46 = Gammatddd[16]*Gammatudd[12];
	double x47 = Gammatddd[2]*Gammatudd[16] + x46;
	double x48 = x22 + x23;
	double x49 = Gammatddd[10]*Gammatudd[13];
	double x50 = Gammatddd[15]*Gammatudd[13];
	double x51 = Gammatddd[1]*Gammatudd[3];
	double x52 = Gammatddd[3]*Gammatudd[1];
	double x53 = Gammatddd[7]*Gammatudd[1];
	double x54 = Gammatddd[4]*Gammatudd[16];
	double x55 = Gammatddd[16]*Gammatudd[13];
	double x56 = Gammatddd[10]*Gammatudd[14] + x55;
	double x57 = Gammatddd[3]*Gammatudd[10];
	double x58 = Gammatddd[10]*Gammatudd[7];
	double x59 = Gammatddd[9]*Gammatudd[8];
	double x60 = x58 + x59;
	double x61 = Gammatddd[11]*Gammatudd[13];
	double x62 = Gammatddd[7]*Gammatudd[2];
	double x63 = Gammatddd[1]*Gammatudd[4];
	double x64 = Gammatddd[4]*Gammatudd[1];
	double x65 = x63 + x64;
	double x66 = Gammatddd[2]*Gammatudd[3];
	double x67 = Gammatddd[8]*Gammatudd[1];
	double x68 = Gammatddd[11]*Gammatudd[14];
	double x69 = Gammatddd[16]*Gammatudd[14];
	double x70 = Gammatddd[5]*Gammatudd[16] + x69;
	double x71 = Gammatddd[8]*Gammatudd[2];
	double x72 = Gammatddd[2]*Gammatudd[4];
	double x73 = Gammatddd[4]*Gammatudd[2];
	double x74 = x72 + x73;
	double x75 = Gammatddd[17]*Gammatudd[14];
	double x76 = 2*x75;
	double x77 = Gammatddd[12]*Gammatudd[1];
	double x78 = Gammatddd[13]*Gammatudd[7];
	double x79 = Gammatddd[14]*Gammatudd[14];
	double x80 = Gammatddd[17]*Gammatudd[12] + x79;
	double x81 = Gammatddd[15]*Gammatudd[7];
	double x82 = Gammatddd[13]*Gammatudd[1];
	double x83 = Gammatddd[16]*Gammatudd[7] + x37;
	double x84 = Gammatddd[11]*Gammatudd[7];
	double x85 = Gammatddd[15]*Gammatudd[8];
	double x86 = Gammatddd[13]*Gammatudd[2];
	double x87 = Gammatddd[5]*Gammatudd[1];
	double x88 = Gammatddd[14]*Gammatudd[1];
	double x89 = Gammatddd[17]*Gammatudd[13];
	double x90 = Gammatddd[11]*Gammatudd[8];
	double x91 = Gammatddd[16]*Gammatudd[8];
	double x92 = Gammatddd[14]*Gammatudd[2];
	double x93 = Gammatddd[2]*Gammatudd[5];
	double x94 = Gammatddd[5]*Gammatudd[2];
	double x95 = 2*Gammatddd[8];
	double x96 = 2*Gammatddd[10];
	double x97 = 2*Gammatddd[11];
	double x98 = 2*Gammatddd[6];
	double x99 = 2*Gammatddd[7];
	double x100 = Gammatddd[7]*Gammatudd[9];
	double x101 = Gammatddd[7]*Gammatudd[10];
	double x102 = Gammatddd[16]*Gammatudd[15];
	double x103 = Gammatddd[10]*Gammatudd[9];
	double x104 = Gammatddd[9]*Gammatudd[10];
	double x105 = Gammatddd[4]*Gammatudd[3];
	double x106 = (1.0/2.0)*gtdd[4];
	double x107 = Gammatddd[17]*Gammatudd[16];
	double x108 = 2*x107;
	double x109 = Gammatddd[14]*Gammatudd[15];
	double x110 = Gammatddd[8]*Gammatudd[10];
	double x111 = Gammatddd[14]*Gammatudd[16];
	double x112 = x111 + x89;
	double x113 = Gammatddd[13]*Gammatudd[10];
	double x114 = Gammatddd[12]*Gammatudd[4];
	double x115 = Gammatddd[16]*Gammatudd[16];
	double x116 = Gammatddd[17]*Gammatudd[15] + x115;
	double x117 = Gammatddd[10]*Gammatudd[11];
	double x118 = Gammatddd[11]*Gammatudd[10];
	double x119 = Gammatddd[16]*Gammatudd[10];
	double x120 = Gammatddd[14]*Gammatudd[4];
	double x121 = Gammatddd[5]*Gammatudd[4];
	double x122 = 2*Gammatddd[15];
	double x123 = 2*Gammatddd[12];
	double x124 = 2*Gammatddd[13];
	double x125 = Gammatddd[14]*Gammatudd[17];
	double x126 = Gammatddd[16]*Gammatudd[17];
	(*W2Rtdd)[0] += x0*(GammatDu[0]*Gammatddd[0] + GammatDu[1]*Gammatddd[1] + GammatDu[2]*Gammatddd[2] + Gammatddd[0]*Gammatudd[0]*x4 + Gammatddd[1]*Gammatudd[1]*x5 + Gammatddd[2]*Gammatudd[2]*x6 + Gammatudd[12]*gtuu[0]*(Gammatddd[12] + x7) + Gammatudd[13]*gtuu[3]*(Gammatddd[13] + x8) + Gammatudd[14]*gtuu[5]*(Gammatddd[14] + x9) + Gammatudd[6]*gtuu[0]*(Gammatddd[6] + x10) + Gammatudd[7]*gtuu[3]*(Gammatddd[7] + x11) + Gammatudd[8]*gtuu[5]*(Gammatddd[8] + x8) + dGammatu_dx[0]*gtdd[0] + dGammatu_dx[1]*gtdd[1] + dGammatu_dx[2]*gtdd[2] - ddgtdd_dx2[0]*x1 - ddgtdd_dx2[12]*gtuu[2] - ddgtdd_dx2[18]*x2 - ddgtdd_dx2[24]*gtuu[4] - ddgtdd_dx2[30]*x3 - ddgtdd_dx2[6]*gtuu[1] + gtuu[1]*(x12 + 2*x13) + gtuu[1]*(2*x12 + x13) + gtuu[1]*(Gammatddd[12]*Gammatudd[13] + Gammatudd[12]*x8) + gtuu[1]*(Gammatddd[6]*Gammatudd[7] + Gammatudd[6]*x11) + gtuu[1]*(Gammatudd[13]*x7 + x14) + gtuu[1]*(Gammatudd[7]*x10 + x15) + gtuu[2]*(x16 + 2*x17) + gtuu[2]*(2*x16 + x17) + gtuu[2]*(Gammatddd[12]*Gammatudd[14] + Gammatudd[12]*x9) + gtuu[2]*(Gammatddd[6]*Gammatudd[8] + Gammatudd[6]*x8) + gtuu[2]*(Gammatudd[14]*x7 + x18) + gtuu[2]*(Gammatudd[8]*x10 + x19) + gtuu[4]*(x22 + 2*x23) + gtuu[4]*(2*x22 + x23) + gtuu[4]*(Gammatudd[13]*x9 + x20) + gtuu[4]*(Gammatudd[14]*x8 + x21) + gtuu[4]*(Gammatudd[7]*x8 + x25) + gtuu[4]*(Gammatudd[8]*x11 + x24));
	(*W2Rtdd)[1] += x0*(dGammatu_dx[0]*x26 + dGammatu_dx[3]*x29 + dGammatu_dx[4]*x26 + dGammatu_dx[5]*x30 - ddgtdd_dx2[13]*gtuu[2] - ddgtdd_dx2[19]*x2 - ddgtdd_dx2[1]*x1 - ddgtdd_dx2[25]*gtuu[4] - ddgtdd_dx2[31]*x3 - ddgtdd_dx2[7]*gtuu[1] + gtdd[3]*x27 + gtdd[4]*x28 + gtuu[0]*(Gammatddd[1]*Gammatudd[7] + 2*x15) + gtuu[0]*(Gammatddd[2]*Gammatudd[13] + Gammatddd[8]*Gammatudd[12] + x14) + gtuu[0]*(Gammatddd[6]*Gammatudd[0] + x12 + x13) + gtuu[1]*(Gammatddd[1]*Gammatudd[9] + x39) + gtuu[1]*(Gammatddd[3]*Gammatudd[7] + x39) + gtuu[1]*(Gammatddd[7]*Gammatudd[0] + Gammatudd[1]*x10) + gtuu[1]*(Gammatddd[0]*Gammatudd[3] + Gammatddd[3]*Gammatudd[0] + Gammatddd[6]*Gammatudd[1]) + gtuu[1]*(Gammatddd[10]*Gammatudd[12] + Gammatddd[13]*Gammatudd[13] + Gammatddd[4]*Gammatudd[13]) + gtuu[1]*(Gammatddd[15]*Gammatudd[12] + Gammatddd[2]*Gammatudd[15] + Gammatddd[8]*Gammatudd[13]) + gtuu[2]*(x40 + x41) + gtuu[2]*(x42 + x44) + gtuu[2]*(x45 + x47) + gtuu[2]*(Gammatddd[4]*Gammatudd[7] + x44) + gtuu[2]*(Gammatddd[8]*Gammatudd[0] + x48) + gtuu[2]*(Gammatddd[11]*Gammatudd[12] + Gammatddd[5]*Gammatudd[13] + x20) + gtuu[3]*(Gammatddd[3]*Gammatudd[9] + x35) + gtuu[3]*(x51 + x52 + x53) + gtuu[3]*(Gammatddd[4]*Gammatudd[15] + x49 + x50) + gtuu[4]*(x54 + x56) + gtuu[4]*(x57 + x60) + gtuu[4]*(x62 + x65) + gtuu[4]*(Gammatddd[4]*Gammatudd[9] + x60) + gtuu[4]*(Gammatddd[15]*Gammatudd[14] + Gammatddd[5]*Gammatudd[15] + x61) + gtuu[4]*(Gammatddd[3]*Gammatudd[2] + x66 + x67) + gtuu[5]*(x36 + 2*x37) + gtuu[5]*(x68 + x70) + gtuu[5]*(x71 + x74) + x31*(Gammatddd[1] + Gammatddd[6]) + x32*(Gammatddd[3] + Gammatddd[7]) + x33*(Gammatddd[4] + Gammatddd[8]));
	(*W2Rtdd)[2] += x0*(dGammatu_dx[0]*x30 + dGammatu_dx[6]*x29 + dGammatu_dx[7]*x26 + dGammatu_dx[8]*x30 - ddgtdd_dx2[14]*gtuu[2] - ddgtdd_dx2[20]*x2 - ddgtdd_dx2[26]*gtuu[4] - ddgtdd_dx2[2]*x1 - ddgtdd_dx2[32]*x3 - ddgtdd_dx2[8]*gtuu[1] + gtdd[4]*x27 + gtdd[5]*x28 + gtuu[0]*(Gammatddd[2]*Gammatudd[14] + 2*x18) + gtuu[0]*(Gammatddd[12]*Gammatudd[0] + x16 + x17) + gtuu[0]*(Gammatddd[13]*Gammatudd[6] + Gammatddd[1]*Gammatudd[8] + x19) + gtuu[1]*(x21 + x47) + gtuu[1]*(x41 + x77) + gtuu[1]*(Gammatddd[13]*Gammatudd[0] + x48) + gtuu[1]*(x42 + x43 + x78) + gtuu[1]*(Gammatddd[15]*Gammatudd[6] + Gammatddd[3]*Gammatudd[8] + x24) + gtuu[1]*(Gammatddd[4]*Gammatudd[14] + x21 + x46) + gtuu[2]*(Gammatddd[14]*Gammatudd[0] + Gammatudd[2]*x7) + gtuu[2]*(Gammatddd[2]*Gammatudd[17] + x80) + gtuu[2]*(Gammatddd[5]*Gammatudd[14] + x80) + gtuu[2]*(Gammatddd[0]*Gammatudd[5] + Gammatddd[12]*Gammatudd[2] + Gammatddd[5]*Gammatudd[0]) + gtuu[2]*(Gammatddd[11]*Gammatudd[6] + Gammatddd[13]*Gammatudd[8] + Gammatddd[1]*Gammatudd[11]) + gtuu[2]*(Gammatddd[16]*Gammatudd[6] + Gammatddd[4]*Gammatudd[8] + Gammatddd[8]*Gammatudd[8]) + gtuu[3]*(x54 + 2*x55) + gtuu[3]*(x65 + x82) + gtuu[3]*(x57 + x58 + x81) + gtuu[4]*(x36 + x83) + gtuu[4]*(x70 + x89) + gtuu[4]*(x74 + x88) + gtuu[4]*(Gammatddd[1]*Gammatudd[5] + x86 + x87) + gtuu[4]*(Gammatddd[3]*Gammatudd[11] + x84 + x85) + gtuu[4]*(Gammatddd[4]*Gammatudd[17] + x69 + x89) + gtuu[5]*(Gammatddd[5]*Gammatudd[17] + x76) + gtuu[5]*(x92 + x93 + x94) + gtuu[5]*(Gammatddd[4]*Gammatudd[11] + x90 + x91) + x31*(Gammatddd[12] + Gammatddd[2]) + x32*(Gammatddd[13] + Gammatddd[4]) + x33*(Gammatddd[14] + Gammatddd[5]));
	(*W2Rtdd)[3] += x0*(GammatDu[0]*Gammatddd[7] + GammatDu[1]*Gammatddd[9] + GammatDu[2]*Gammatddd[10] + Gammatddd[10]*Gammatudd[10]*x6 + Gammatddd[9]*Gammatudd[9]*x5 + Gammatudd[13]*gtuu[0]*(Gammatddd[13] + x95) + Gammatudd[15]*gtuu[3]*(Gammatddd[15] + x96) + Gammatudd[16]*gtuu[5]*(Gammatddd[16] + x97) + Gammatudd[1]*gtuu[0]*(Gammatddd[1] + x98) + Gammatudd[3]*gtuu[3]*(Gammatddd[3] + x99) + Gammatudd[4]*gtuu[5]*(Gammatddd[4] + x95) + dGammatu_dx[3]*gtdd[1] + dGammatu_dx[4]*gtdd[3] + dGammatu_dx[5]*gtdd[4] - ddgtdd_dx2[15]*gtuu[2] - ddgtdd_dx2[21]*x2 - ddgtdd_dx2[27]*gtuu[4] - ddgtdd_dx2[33]*x3 - ddgtdd_dx2[3]*x1 - ddgtdd_dx2[9]*gtuu[1] + gtuu[1]*(x100 + x35) + gtuu[1]*(2*x100 + x34) + gtuu[1]*(x51 + 2*x53) + gtuu[1]*(Gammatddd[13]*Gammatudd[15] + 2*x49) + gtuu[1]*(Gammatudd[15]*x95 + x50) + gtuu[1]*(Gammatudd[3]*x98 + x52) + gtuu[2]*(x101 + 2*x58) + gtuu[2]*(2*x101 + x58) + gtuu[2]*(x63 + 2*x67) + gtuu[2]*(Gammatddd[13]*Gammatudd[16] + 2*x61) + gtuu[2]*(Gammatudd[16]*x95 + x55) + gtuu[2]*(Gammatudd[4]*x98 + x64) + gtuu[4]*(x103 + 2*x104) + gtuu[4]*(2*x103 + x104) + gtuu[4]*(Gammatddd[15]*Gammatudd[16] + Gammatudd[15]*x97) + gtuu[4]*(Gammatddd[3]*Gammatudd[4] + Gammatudd[3]*x95) + gtuu[4]*(Gammatudd[16]*x96 + x102) + gtuu[4]*(Gammatudd[4]*x99 + x105) + x38*x4);
	(*W2Rtdd)[4] += x0*(dGammatu_dx[3]*x30 + dGammatu_dx[4]*x106 + (1.0/2.0)*dGammatu_dx[5]*gtdd[5] + dGammatu_dx[6]*x26 + (1.0/2.0)*dGammatu_dx[7]*gtdd[3] + dGammatu_dx[8]*x106 - ddgtdd_dx2[10]*gtuu[1] - ddgtdd_dx2[16]*gtuu[2] - ddgtdd_dx2[22]*x2 - ddgtdd_dx2[28]*gtuu[4] - ddgtdd_dx2[34]*x3 - ddgtdd_dx2[4]*x1 + gtuu[0]*(2*x21 + x45) + gtuu[0]*(x23 + x40 + x77) + gtuu[0]*(x24 + x25 + x78) + gtuu[1]*(x109 + x56) + gtuu[1]*(x62 + x66 + x82) + gtuu[1]*(Gammatddd[12]*Gammatudd[3] + Gammatddd[6]*Gammatudd[4] + x64) + gtuu[1]*(Gammatddd[13]*Gammatudd[9] + x101 + x58) + gtuu[1]*(Gammatddd[8]*Gammatudd[16] + x109 + x55) + gtuu[1]*(Gammatddd[8]*Gammatudd[9] + x59 + x81) + gtuu[2]*(x110 + x83) + gtuu[2]*(x112 + x68) + gtuu[2]*(Gammatddd[8]*Gammatudd[17] + x112) + gtuu[2]*(x71 + x72 + x88) + gtuu[2]*(Gammatddd[6]*Gammatudd[5] + x114 + x87) + gtuu[2]*(Gammatddd[7]*Gammatudd[11] + x113 + x84) + gtuu[3]*(Gammatddd[10]*Gammatudd[16] + 2*x102) + gtuu[3]*(Gammatddd[13]*Gammatudd[3] + Gammatddd[7]*Gammatudd[4] + x105) + gtuu[3]*(Gammatddd[15]*Gammatudd[9] + x103 + x104) + gtuu[4]*(Gammatddd[10]*Gammatudd[17] + x116) + gtuu[4]*(Gammatddd[11]*Gammatudd[16] + x116) + gtuu[4]*(Gammatddd[16]*Gammatudd[9] + Gammatudd[10]*x96) + gtuu[4]*(Gammatddd[11]*Gammatudd[9] + Gammatddd[15]*Gammatudd[10] + Gammatddd[9]*Gammatudd[11]) + gtuu[4]*(Gammatddd[13]*Gammatudd[4] + Gammatddd[5]*Gammatudd[3] + Gammatddd[7]*Gammatudd[5]) + gtuu[4]*(Gammatddd[14]*Gammatudd[3] + Gammatddd[4]*Gammatudd[4] + Gammatddd[8]*Gammatudd[4]) + gtuu[5]*(Gammatddd[11]*Gammatudd[17] + x108) + gtuu[5]*(x117 + x118 + x119) + gtuu[5]*(Gammatddd[8]*Gammatudd[5] + x120 + x121) + x31*(Gammatddd[13] + Gammatddd[8]) + x32*(Gammatddd[10] + Gammatddd[15]) + x33*(Gammatddd[11] + Gammatddd[16]));
	(*W2Rtdd)[5] += x0*(GammatDu[0]*Gammatddd[14] + GammatDu[1]*Gammatddd[16] + GammatDu[2]*Gammatddd[17] + Gammatddd[17]*Gammatudd[17]*x6 + Gammatudd[10]*gtuu[3]*(Gammatddd[10] + x122) + Gammatudd[11]*gtuu[5]*(Gammatddd[11] + 2*Gammatddd[16]) + Gammatudd[2]*gtuu[0]*(Gammatddd[2] + x123) + Gammatudd[4]*gtuu[3]*(Gammatddd[4] + x124) + Gammatudd[5]*gtuu[5]*(2*Gammatddd[14] + Gammatddd[5]) + Gammatudd[8]*gtuu[0]*(Gammatddd[8] + x124) + dGammatu_dx[6]*gtdd[2] + dGammatu_dx[7]*gtdd[4] + dGammatu_dx[8]*gtdd[5] - ddgtdd_dx2[11]*gtuu[1] - ddgtdd_dx2[17]*gtuu[2] - ddgtdd_dx2[23]*x2 - ddgtdd_dx2[29]*gtuu[4] - ddgtdd_dx2[35]*x3 - ddgtdd_dx2[5]*x1 + gtuu[1]*(x110 + 2*x85) + gtuu[1]*(x111 + 2*x69) + gtuu[1]*(2*x111 + x69) + gtuu[1]*(2*x113 + x37) + gtuu[1]*(2*x114 + x73) + gtuu[1]*(x72 + 2*x86) + gtuu[2]*(x125 + x76) + gtuu[2]*(2*x125 + x75) + gtuu[2]*(2*x92 + x93) + gtuu[2]*(Gammatddd[8]*Gammatudd[11] + 2*x91) + gtuu[2]*(Gammatudd[11]*x124 + x90) + gtuu[2]*(Gammatudd[5]*x123 + x94) + gtuu[4]*(x107 + 2*x126) + gtuu[4]*(x108 + x126) + gtuu[4]*(x117 + 2*x119) + gtuu[4]*(Gammatddd[4]*Gammatudd[5] + 2*x120) + gtuu[4]*(Gammatudd[11]*x122 + x118) + gtuu[4]*(Gammatudd[5]*x124 + x121) + x115*x5 + x4*x79);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci_wave(
	double W,
	const double gtuu[6],
	const double ddgtdd_dx2[36],
	double (*W2Rtdd_wave)[6]
)
{
	double x0 = ((W)*(W));
	double x1 = (1.0/2.0)*gtuu[0];
	double x2 = (1.0/2.0)*gtuu[3];
	double x3 = (1.0/2.0)*gtuu[5];
	(*W2Rtdd_wave)[0] += -x0*(ddgtdd_dx2[0]*x1 + ddgtdd_dx2[12]*gtuu[2] + ddgtdd_dx2[18]*x2 + ddgtdd_dx2[24]*gtuu[4] + ddgtdd_dx2[30]*x3 + ddgtdd_dx2[6]*gtuu[1]);
	(*W2Rtdd_wave)[1] += -x0*(ddgtdd_dx2[13]*gtuu[2] + ddgtdd_dx2[19]*x2 + ddgtdd_dx2[1]*x1 + ddgtdd_dx2[25]*gtuu[4] + ddgtdd_dx2[31]*x3 + ddgtdd_dx2[7]*gtuu[1]);
	(*W2Rtdd_wave)[2] += -x0*(ddgtdd_dx2[14]*gtuu[2] + ddgtdd_dx2[20]*x2 + ddgtdd_dx2[26]*gtuu[4] + ddgtdd_dx2[2]*x1 + ddgtdd_dx2[32]*x3 + ddgtdd_dx2[8]*gtuu[1]);
	(*W2Rtdd_wave)[3] += -x0*(ddgtdd_dx2[15]*gtuu[2] + ddgtdd_dx2[21]*x2 + ddgtdd_dx2[27]*gtuu[4] + ddgtdd_dx2[33]*x3 + ddgtdd_dx2[3]*x1 + ddgtdd_dx2[9]*gtuu[1]);
	(*W2Rtdd_wave)[4] += -x0*(ddgtdd_dx2[10]*gtuu[1] + ddgtdd_dx2[16]*gtuu[2] + ddgtdd_dx2[22]*x2 + ddgtdd_dx2[28]*gtuu[4] + ddgtdd_dx2[34]*x3 + ddgtdd_dx2[4]*x1);
	(*W2Rtdd_wave)[5] += -x0*(ddgtdd_dx2[11]*gtuu[1] + ddgtdd_dx2[17]*gtuu[2] + ddgtdd_dx2[23]*x2 + ddgtdd_dx2[29]*gtuu[4] + ddgtdd_dx2[35]*x3 + ddgtdd_dx2[5]*x1);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci_dgamma(
	const double gtdd[6],
	double W,
	const double Gammatddd[18],
	const double GammatDu[3],
	const double dGammatu_dx[9],
	double (*W2Rtdd_dg)[6]
)
{
	double x0 = ((W)*(W));
	double x1 = (1.0/2.0)*x0;
	(*W2Rtdd_dg)[0] += x0*(GammatDu[0]*Gammatddd[0] + GammatDu[1]*Gammatddd[1] + GammatDu[2]*Gammatddd[2] + dGammatu_dx[0]*gtdd[0] + dGammatu_dx[1]*gtdd[1] + dGammatu_dx[2]*gtdd[2]);
	(*W2Rtdd_dg)[1] += x1*(GammatDu[0]*(Gammatddd[1] + Gammatddd[6]) + GammatDu[1]*(Gammatddd[3] + Gammatddd[7]) + GammatDu[2]*(Gammatddd[4] + Gammatddd[8]) + dGammatu_dx[0]*gtdd[1] + dGammatu_dx[1]*gtdd[3] + dGammatu_dx[2]*gtdd[4] + dGammatu_dx[3]*gtdd[0] + dGammatu_dx[4]*gtdd[1] + dGammatu_dx[5]*gtdd[2]);
	(*W2Rtdd_dg)[2] += x1*(GammatDu[0]*(Gammatddd[12] + Gammatddd[2]) + GammatDu[1]*(Gammatddd[13] + Gammatddd[4]) + GammatDu[2]*(Gammatddd[14] + Gammatddd[5]) + dGammatu_dx[0]*gtdd[2] + dGammatu_dx[1]*gtdd[4] + dGammatu_dx[2]*gtdd[5] + dGammatu_dx[6]*gtdd[0] + dGammatu_dx[7]*gtdd[1] + dGammatu_dx[8]*gtdd[2]);
	(*W2Rtdd_dg)[3] += x0*(GammatDu[0]*Gammatddd[7] + GammatDu[1]*Gammatddd[9] + GammatDu[2]*Gammatddd[10] + dGammatu_dx[3]*gtdd[1] + dGammatu_dx[4]*gtdd[3] + dGammatu_dx[5]*gtdd[4]);
	(*W2Rtdd_dg)[4] += x1*(GammatDu[0]*(Gammatddd[13] + Gammatddd[8]) + GammatDu[1]*(Gammatddd[10] + Gammatddd[15]) + GammatDu[2]*(Gammatddd[11] + Gammatddd[16]) + dGammatu_dx[3]*gtdd[2] + dGammatu_dx[4]*gtdd[4] + dGammatu_dx[5]*gtdd[5] + dGammatu_dx[6]*gtdd[1] + dGammatu_dx[7]*gtdd[3] + dGammatu_dx[8]*gtdd[4]);
	(*W2Rtdd_dg)[5] += x0*(GammatDu[0]*Gammatddd[14] + GammatDu[1]*Gammatddd[16] + GammatDu[2]*Gammatddd[17] + dGammatu_dx[6]*gtdd[2] + dGammatu_dx[7]*gtdd[4] + dGammatu_dx[8]*gtdd[5]);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci_gammagamma(
	double W,
	const double gtuu[6],
	const double Gammatddd[18],
	const double Gammatudd[18],
	double (*W2Rtdd_gg)[6]
)
{
	double x0 = ((W)*(W));
	double x1 = 3*gtuu[0];
	double x2 = 3*gtuu[3];
	double x3 = 3*gtuu[5];
	double x4 = 2*Gammatddd[2];
	double x5 = 2*Gammatddd[4];
	double x6 = 2*Gammatddd[5];
	double x7 = 2*Gammatddd[1];
	double x8 = 2*Gammatddd[3];
	double x9 = Gammatddd[0]*Gammatudd[1];
	double x10 = Gammatddd[1]*Gammatudd[0];
	double x11 = Gammatddd[13]*Gammatudd[12];
	double x12 = Gammatddd[7]*Gammatudd[6];
	double x13 = Gammatddd[0]*Gammatudd[2];
	double x14 = Gammatddd[2]*Gammatudd[0];
	double x15 = Gammatddd[14]*Gammatudd[12];
	double x16 = Gammatddd[8]*Gammatudd[6];
	double x17 = Gammatddd[13]*Gammatudd[14];
	double x18 = Gammatddd[14]*Gammatudd[13];
	double x19 = Gammatddd[1]*Gammatudd[2];
	double x20 = Gammatddd[2]*Gammatudd[1];
	double x21 = Gammatddd[8]*Gammatudd[7];
	double x22 = Gammatddd[7]*Gammatudd[8];
	double x23 = Gammatddd[9]*Gammatudd[7];
	double x24 = 2*x23;
	double x25 = Gammatddd[4]*Gammatudd[10];
	double x26 = Gammatddd[10]*Gammatudd[8];
	double x27 = Gammatddd[7]*Gammatudd[7];
	double x28 = Gammatddd[9]*Gammatudd[6] + x27;
	double x29 = Gammatddd[6]*Gammatudd[2];
	double x30 = Gammatddd[0]*Gammatudd[4] + Gammatddd[4]*Gammatudd[0];
	double x31 = Gammatddd[1]*Gammatudd[10];
	double x32 = Gammatddd[10]*Gammatudd[6];
	double x33 = x22 + x32;
	double x34 = Gammatddd[8]*Gammatudd[14];
	double x35 = Gammatddd[16]*Gammatudd[12];
	double x36 = Gammatddd[2]*Gammatudd[16] + x35;
	double x37 = x19 + x20;
	double x38 = Gammatddd[10]*Gammatudd[13];
	double x39 = Gammatddd[15]*Gammatudd[13];
	double x40 = Gammatddd[1]*Gammatudd[3];
	double x41 = Gammatddd[3]*Gammatudd[1];
	double x42 = Gammatddd[7]*Gammatudd[1];
	double x43 = Gammatddd[4]*Gammatudd[16];
	double x44 = Gammatddd[16]*Gammatudd[13];
	double x45 = Gammatddd[10]*Gammatudd[14] + x44;
	double x46 = Gammatddd[3]*Gammatudd[10];
	double x47 = Gammatddd[10]*Gammatudd[7];
	double x48 = Gammatddd[9]*Gammatudd[8];
	double x49 = x47 + x48;
	double x50 = Gammatddd[11]*Gammatudd[13];
	double x51 = Gammatddd[7]*Gammatudd[2];
	double x52 = Gammatddd[1]*Gammatudd[4];
	double x53 = Gammatddd[4]*Gammatudd[1];
	double x54 = x52 + x53;
	double x55 = Gammatddd[2]*Gammatudd[3];
	double x56 = Gammatddd[8]*Gammatudd[1];
	double x57 = Gammatddd[11]*Gammatudd[14];
	double x58 = Gammatddd[16]*Gammatudd[14];
	double x59 = Gammatddd[5]*Gammatudd[16] + x58;
	double x60 = Gammatddd[8]*Gammatudd[2];
	double x61 = Gammatddd[2]*Gammatudd[4];
	double x62 = Gammatddd[4]*Gammatudd[2];
	double x63 = x61 + x62;
	double x64 = Gammatddd[17]*Gammatudd[14];
	double x65 = 2*x64;
	double x66 = Gammatddd[12]*Gammatudd[1];
	double x67 = Gammatddd[13]*Gammatudd[7];
	double x68 = Gammatddd[14]*Gammatudd[14];
	double x69 = Gammatddd[17]*Gammatudd[12] + x68;
	double x70 = Gammatddd[15]*Gammatudd[7];
	double x71 = Gammatddd[13]*Gammatudd[1];
	double x72 = Gammatddd[16]*Gammatudd[7] + x26;
	double x73 = Gammatddd[11]*Gammatudd[7];
	double x74 = Gammatddd[15]*Gammatudd[8];
	double x75 = Gammatddd[13]*Gammatudd[2];
	double x76 = Gammatddd[5]*Gammatudd[1];
	double x77 = Gammatddd[14]*Gammatudd[1];
	double x78 = Gammatddd[17]*Gammatudd[13];
	double x79 = Gammatddd[11]*Gammatudd[8];
	double x80 = Gammatddd[16]*Gammatudd[8];
	double x81 = Gammatddd[14]*Gammatudd[2];
	double x82 = Gammatddd[2]*Gammatudd[5];
	double x83 = Gammatddd[5]*Gammatudd[2];
	double x84 = 2*Gammatddd[8];
	double x85 = 2*Gammatddd[10];
	double x86 = 2*Gammatddd[11];
	double x87 = 2*Gammatddd[6];
	double x88 = 2*Gammatddd[7];
	double x89 = Gammatddd[7]*Gammatudd[9];
	double x90 = Gammatddd[7]*Gammatudd[10];
	double x91 = Gammatddd[16]*Gammatudd[15];
	double x92 = Gammatddd[10]*Gammatudd[9];
	double x93 = Gammatddd[9]*Gammatudd[10];
	double x94 = Gammatddd[4]*Gammatudd[3];
	double x95 = Gammatddd[17]*Gammatudd[16];
	double x96 = 2*x95;
	double x97 = Gammatddd[14]*Gammatudd[15];
	double x98 = Gammatddd[8]*Gammatudd[10];
	double x99 = Gammatddd[14]*Gammatudd[16];
	double x100 = x78 + x99;
	double x101 = Gammatddd[13]*Gammatudd[10];
	double x102 = Gammatddd[12]*Gammatudd[4];
	double x103 = Gammatddd[16]*Gammatudd[16];
	double x104 = Gammatddd[17]*Gammatudd[15] + x103;
	double x105 = Gammatddd[10]*Gammatudd[11];
	double x106 = Gammatddd[11]*Gammatudd[10];
	double x107 = Gammatddd[16]*Gammatudd[10];
	double x108 = Gammatddd[14]*Gammatudd[4];
	double x109 = Gammatddd[5]*Gammatudd[4];
	double x110 = 2*Gammatddd[15];
	double x111 = 2*Gammatddd[12];
	double x112 = 2*Gammatddd[13];
	double x113 = Gammatddd[14]*Gammatudd[17];
	double x114 = Gammatddd[16]*Gammatudd[17];
	(*W2Rtdd_gg)[0] += x0*(Gammatddd[0]*Gammatudd[0]*x1 + Gammatddd[1]*Gammatudd[1]*x2 + Gammatddd[2]*Gammatudd[2]*x3 + Gammatudd[12]*gtuu[0]*(Gammatddd[12] + x4) + Gammatudd[13]*gtuu[3]*(Gammatddd[13] + x5) + Gammatudd[14]*gtuu[5]*(Gammatddd[14] + x6) + Gammatudd[6]*gtuu[0]*(Gammatddd[6] + x7) + Gammatudd[7]*gtuu[3]*(Gammatddd[7] + x8) + Gammatudd[8]*gtuu[5]*(Gammatddd[8] + x5) + gtuu[1]*(x10 + 2*x9) + gtuu[1]*(2*x10 + x9) + gtuu[1]*(Gammatddd[12]*Gammatudd[13] + Gammatudd[12]*x5) + gtuu[1]*(Gammatddd[6]*Gammatudd[7] + Gammatudd[6]*x8) + gtuu[1]*(Gammatudd[13]*x4 + x11) + gtuu[1]*(Gammatudd[7]*x7 + x12) + gtuu[2]*(x13 + 2*x14) + gtuu[2]*(2*x13 + x14) + gtuu[2]*(Gammatddd[12]*Gammatudd[14] + Gammatudd[12]*x6) + gtuu[2]*(Gammatddd[6]*Gammatudd[8] + Gammatudd[6]*x5) + gtuu[2]*(Gammatudd[14]*x4 + x15) + gtuu[2]*(Gammatudd[8]*x7 + x16) + gtuu[4]*(x19 + 2*x20) + gtuu[4]*(2*x19 + x20) + gtuu[4]*(Gammatudd[13]*x6 + x17) + gtuu[4]*(Gammatudd[14]*x5 + x18) + gtuu[4]*(Gammatudd[7]*x5 + x22) + gtuu[4]*(Gammatudd[8]*x8 + x21));
	(*W2Rtdd_gg)[1] += x0*(gtuu[0]*(Gammatddd[1]*Gammatudd[7] + 2*x12) + gtuu[0]*(Gammatddd[2]*Gammatudd[13] + Gammatddd[8]*Gammatudd[12] + x11) + gtuu[0]*(Gammatddd[6]*Gammatudd[0] + x10 + x9) + gtuu[1]*(Gammatddd[1]*Gammatudd[9] + x28) + gtuu[1]*(Gammatddd[3]*Gammatudd[7] + x28) + gtuu[1]*(Gammatddd[7]*Gammatudd[0] + Gammatudd[1]*x7) + gtuu[1]*(Gammatddd[0]*Gammatudd[3] + Gammatddd[3]*Gammatudd[0] + Gammatddd[6]*Gammatudd[1]) + gtuu[1]*(Gammatddd[10]*Gammatudd[12] + Gammatddd[13]*Gammatudd[13] + Gammatddd[4]*Gammatudd[13]) + gtuu[1]*(Gammatddd[15]*Gammatudd[12] + Gammatddd[2]*Gammatudd[15] + Gammatddd[8]*Gammatudd[13]) + gtuu[2]*(x29 + x30) + gtuu[2]*(x31 + x33) + gtuu[2]*(x34 + x36) + gtuu[2]*(Gammatddd[4]*Gammatudd[7] + x33) + gtuu[2]*(Gammatddd[8]*Gammatudd[0] + x37) + gtuu[2]*(Gammatddd[11]*Gammatudd[12] + Gammatddd[5]*Gammatudd[13] + x17) + gtuu[3]*(Gammatddd[3]*Gammatudd[9] + x24) + gtuu[3]*(x40 + x41 + x42) + gtuu[3]*(Gammatddd[4]*Gammatudd[15] + x38 + x39) + gtuu[4]*(x43 + x45) + gtuu[4]*(x46 + x49) + gtuu[4]*(x51 + x54) + gtuu[4]*(Gammatddd[4]*Gammatudd[9] + x49) + gtuu[4]*(Gammatddd[15]*Gammatudd[14] + Gammatddd[5]*Gammatudd[15] + x50) + gtuu[4]*(Gammatddd[3]*Gammatudd[2] + x55 + x56) + gtuu[5]*(x25 + 2*x26) + gtuu[5]*(x57 + x59) + gtuu[5]*(x60 + x63));
	(*W2Rtdd_gg)[2] += x0*(gtuu[0]*(Gammatddd[2]*Gammatudd[14] + 2*x15) + gtuu[0]*(Gammatddd[12]*Gammatudd[0] + x13 + x14) + gtuu[0]*(Gammatddd[13]*Gammatudd[6] + Gammatddd[1]*Gammatudd[8] + x16) + gtuu[1]*(x18 + x36) + gtuu[1]*(x30 + x66) + gtuu[1]*(Gammatddd[13]*Gammatudd[0] + x37) + gtuu[1]*(x31 + x32 + x67) + gtuu[1]*(Gammatddd[15]*Gammatudd[6] + Gammatddd[3]*Gammatudd[8] + x21) + gtuu[1]*(Gammatddd[4]*Gammatudd[14] + x18 + x35) + gtuu[2]*(Gammatddd[14]*Gammatudd[0] + Gammatudd[2]*x4) + gtuu[2]*(Gammatddd[2]*Gammatudd[17] + x69) + gtuu[2]*(Gammatddd[5]*Gammatudd[14] + x69) + gtuu[2]*(Gammatddd[0]*Gammatudd[5] + Gammatddd[12]*Gammatudd[2] + Gammatddd[5]*Gammatudd[0]) + gtuu[2]*(Gammatddd[11]*Gammatudd[6] + Gammatddd[13]*Gammatudd[8] + Gammatddd[1]*Gammatudd[11]) + gtuu[2]*(Gammatddd[16]*Gammatudd[6] + Gammatddd[4]*Gammatudd[8] + Gammatddd[8]*Gammatudd[8]) + gtuu[3]*(x43 + 2*x44) + gtuu[3]*(x54 + x71) + gtuu[3]*(x46 + x47 + x70) + gtuu[4]*(x25 + x72) + gtuu[4]*(x59 + x78) + gtuu[4]*(x63 + x77) + gtuu[4]*(Gammatddd[1]*Gammatudd[5] + x75 + x76) + gtuu[4]*(Gammatddd[3]*Gammatudd[11] + x73 + x74) + gtuu[4]*(Gammatddd[4]*Gammatudd[17] + x58 + x78) + gtuu[5]*(Gammatddd[5]*Gammatudd[17] + x65) + gtuu[5]*(x81 + x82 + x83) + gtuu[5]*(Gammatddd[4]*Gammatudd[11] + x79 + x80));
	(*W2Rtdd_gg)[3] += x0*(Gammatddd[10]*Gammatudd[10]*x3 + Gammatddd[9]*Gammatudd[9]*x2 + Gammatudd[13]*gtuu[0]*(Gammatddd[13] + x84) + Gammatudd[15]*gtuu[3]*(Gammatddd[15] + x85) + Gammatudd[16]*gtuu[5]*(Gammatddd[16] + x86) + Gammatudd[1]*gtuu[0]*(Gammatddd[1] + x87) + Gammatudd[3]*gtuu[3]*(Gammatddd[3] + x88) + Gammatudd[4]*gtuu[5]*(Gammatddd[4] + x84) + gtuu[1]*(x23 + 2*x89) + gtuu[1]*(x24 + x89) + gtuu[1]*(x40 + 2*x42) + gtuu[1]*(Gammatddd[13]*Gammatudd[15] + 2*x38) + gtuu[1]*(Gammatudd[15]*x84 + x39) + gtuu[1]*(Gammatudd[3]*x87 + x41) + gtuu[2]*(x47 + 2*x90) + gtuu[2]*(2*x47 + x90) + gtuu[2]*(x52 + 2*x56) + gtuu[2]*(Gammatddd[13]*Gammatudd[16] + 2*x50) + gtuu[2]*(Gammatudd[16]*x84 + x44) + gtuu[2]*(Gammatudd[4]*x87 + x53) + gtuu[4]*(x92 + 2*x93) + gtuu[4]*(2*x92 + x93) + gtuu[4]*(Gammatddd[15]*Gammatudd[16] + Gammatudd[15]*x86) + gtuu[4]*(Gammatddd[3]*Gammatudd[4] + Gammatudd[3]*x84) + gtuu[4]*(Gammatudd[16]*x85 + x91) + gtuu[4]*(Gammatudd[4]*x88 + x94) + x1*x27);
	(*W2Rtdd_gg)[4] += x0*(gtuu[0]*(2*x18 + x34) + gtuu[0]*(x20 + x29 + x66) + gtuu[0]*(x21 + x22 + x67) + gtuu[1]*(x45 + x97) + gtuu[1]*(x51 + x55 + x71) + gtuu[1]*(Gammatddd[12]*Gammatudd[3] + Gammatddd[6]*Gammatudd[4] + x53) + gtuu[1]*(Gammatddd[13]*Gammatudd[9] + x47 + x90) + gtuu[1]*(Gammatddd[8]*Gammatudd[16] + x44 + x97) + gtuu[1]*(Gammatddd[8]*Gammatudd[9] + x48 + x70) + gtuu[2]*(x100 + x57) + gtuu[2]*(x72 + x98) + gtuu[2]*(Gammatddd[8]*Gammatudd[17] + x100) + gtuu[2]*(x60 + x61 + x77) + gtuu[2]*(Gammatddd[6]*Gammatudd[5] + x102 + x76) + gtuu[2]*(Gammatddd[7]*Gammatudd[11] + x101 + x73) + gtuu[3]*(Gammatddd[10]*Gammatudd[16] + 2*x91) + gtuu[3]*(Gammatddd[13]*Gammatudd[3] + Gammatddd[7]*Gammatudd[4] + x94) + gtuu[3]*(Gammatddd[15]*Gammatudd[9] + x92 + x93) + gtuu[4]*(Gammatddd[10]*Gammatudd[17] + x104) + gtuu[4]*(Gammatddd[11]*Gammatudd[16] + x104) + gtuu[4]*(Gammatddd[16]*Gammatudd[9] + Gammatudd[10]*x85) + gtuu[4]*(Gammatddd[11]*Gammatudd[9] + Gammatddd[15]*Gammatudd[10] + Gammatddd[9]*Gammatudd[11]) + gtuu[4]*(Gammatddd[13]*Gammatudd[4] + Gammatddd[5]*Gammatudd[3] + Gammatddd[7]*Gammatudd[5]) + gtuu[4]*(Gammatddd[14]*Gammatudd[3] + Gammatddd[4]*Gammatudd[4] + Gammatddd[8]*Gammatudd[4]) + gtuu[5]*(Gammatddd[11]*Gammatudd[17] + x96) + gtuu[5]*(x105 + x106 + x107) + gtuu[5]*(Gammatddd[8]*Gammatudd[5] + x108 + x109));
	(*W2Rtdd_gg)[5] += x0*(Gammatddd[17]*Gammatudd[17]*x3 + Gammatudd[10]*gtuu[3]*(Gammatddd[10] + x110) + Gammatudd[11]*gtuu[5]*(Gammatddd[11] + 2*Gammatddd[16]) + Gammatudd[2]*gtuu[0]*(Gammatddd[2] + x111) + Gammatudd[4]*gtuu[3]*(Gammatddd[4] + x112) + Gammatudd[5]*gtuu[5]*(2*Gammatddd[14] + Gammatddd[5]) + Gammatudd[8]*gtuu[0]*(Gammatddd[8] + x112) + gtuu[1]*(2*x101 + x26) + gtuu[1]*(2*x102 + x62) + gtuu[1]*(x58 + 2*x99) + gtuu[1]*(2*x58 + x99) + gtuu[1]*(x61 + 2*x75) + gtuu[1]*(2*x74 + x98) + gtuu[2]*(x113 + x65) + gtuu[2]*(2*x113 + x64) + gtuu[2]*(2*x81 + x82) + gtuu[2]*(Gammatddd[8]*Gammatudd[11] + 2*x80) + gtuu[2]*(Gammatudd[11]*x112 + x79) + gtuu[2]*(Gammatudd[5]*x111 + x83) + gtuu[4]*(x105 + 2*x107) + gtuu[4]*(x114 + x96) + gtuu[4]*(2*x114 + x95) + gtuu[4]*(Gammatddd[4]*Gammatudd[5] + 2*x108) + gtuu[4]*(Gammatudd[11]*x110 + x106) + gtuu[4]*(Gammatudd[5]*x112 + x109) + x1*x68 + x103*x2);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci_conf(
	const double gtdd[6],
	double W,
	const double gtuu[6],
	const double Gammatudd[18],
	const double dW_dx[3],
	const double ddW_dx2[6],
	double (*W2Rphi)[6]
)
{
	double x0 = W*(Gammatudd[0]*dW_dx[0] + Gammatudd[12]*dW_dx[2] + Gammatudd[6]*dW_dx[1] - ddW_dx2[0]);
	double x1 = 2*dW_dx[0];
	double x2 = W*(Gammatudd[13]*dW_dx[2] + Gammatudd[1]*dW_dx[0] + Gammatudd[7]*dW_dx[1] - ddW_dx2[1]);
	double x3 = W*(Gammatudd[14]*dW_dx[2] + Gammatudd[2]*dW_dx[0] + Gammatudd[8]*dW_dx[1] - ddW_dx2[2]);
	double x4 = W*(Gammatudd[15]*dW_dx[2] + Gammatudd[3]*dW_dx[0] + Gammatudd[9]*dW_dx[1] - ddW_dx2[3]);
	double x5 = W*(Gammatudd[10]*dW_dx[1] + Gammatudd[16]*dW_dx[2] + Gammatudd[4]*dW_dx[0] - ddW_dx2[4]);
	double x6 = W*(Gammatudd[11]*dW_dx[1] + Gammatudd[17]*dW_dx[2] + Gammatudd[5]*dW_dx[0] - ddW_dx2[5]);
	double x7 = gtuu[0]*(2*((dW_dx[0])*(dW_dx[0])) + x0) + 2*gtuu[1]*(dW_dx[1]*x1 + x2) + 2*gtuu[2]*(dW_dx[2]*x1 + x3) + gtuu[3]*(2*((dW_dx[1])*(dW_dx[1])) + x4) + 2*gtuu[4]*(2*dW_dx[1]*dW_dx[2] + x5) + gtuu[5]*(2*((dW_dx[2])*(dW_dx[2])) + x6);
	(*W2Rphi)[0] += -gtdd[0]*x7 - x0;
	(*W2Rphi)[1] += -gtdd[1]*x7 - x2;
	(*W2Rphi)[2] += -gtdd[2]*x7 - x3;
	(*W2Rphi)[3] += -gtdd[3]*x7 - x4;
	(*W2Rphi)[4] += -gtdd[4]*x7 - x5;
	(*W2Rphi)[5] += -gtdd[5]*x7 - x6;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci_trace(
	const double gtuu[6],
	const double W2Rdd[6],
	double * __restrict__ Rtrace
)
{
	*Rtrace = W2Rdd[0]*gtuu[0] + 2*W2Rdd[1]*gtuu[1] + 2*W2Rdd[2]*gtuu[2] + W2Rdd[3]*gtuu[3] + 2*W2Rdd[4]*gtuu[4] + W2Rdd[5]*gtuu[5];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_chi_rhs(
	double alp,
	double W,
	double theta,
	double Khat,
	const double dbetau_dx[9],
	double * __restrict__ dW
)
{
	*dW = (1.0/3.0)*W*(alp*(Khat + 2*theta) - dbetau_dx[0] - dbetau_dx[4] - dbetau_dx[8]);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_gtdd_rhs(
	const double gtdd[6],
	const double Atdd[6],
	double alp,
	const double dbetau_dx[9],
	double (*dgtdd_dt)[6]
)
{
	double x0 = (1.0/3.0)*gtdd[0];
	double x1 = 2*alp;
	double x2 = (1.0/3.0)*gtdd[1];
	double x3 = (1.0/3.0)*gtdd[2];
	double x4 = (1.0/3.0)*gtdd[3];
	double x5 = (1.0/3.0)*gtdd[4];
	double x6 = (1.0/3.0)*gtdd[5];
	(*dgtdd_dt)[0] = -2*Atdd[0]*alp + (4.0/3.0)*dbetau_dx[0]*gtdd[0] + 2*dbetau_dx[1]*gtdd[1] + 2*dbetau_dx[2]*gtdd[2] - 2*dbetau_dx[4]*x0 - 2*dbetau_dx[8]*x0;
	(*dgtdd_dt)[1] = -Atdd[1]*x1 + dbetau_dx[0]*x2 + dbetau_dx[1]*gtdd[3] + dbetau_dx[2]*gtdd[4] + dbetau_dx[3]*gtdd[0] + dbetau_dx[4]*x2 + dbetau_dx[5]*gtdd[2] - 2.0/3.0*dbetau_dx[8]*gtdd[1];
	(*dgtdd_dt)[2] = -Atdd[2]*x1 + dbetau_dx[0]*x3 + dbetau_dx[1]*gtdd[4] + dbetau_dx[2]*gtdd[5] - 2.0/3.0*dbetau_dx[4]*gtdd[2] + dbetau_dx[6]*gtdd[0] + dbetau_dx[7]*gtdd[1] + dbetau_dx[8]*x3;
	(*dgtdd_dt)[3] = -2*Atdd[3]*alp - 2*dbetau_dx[0]*x4 + 2*dbetau_dx[3]*gtdd[1] + (4.0/3.0)*dbetau_dx[4]*gtdd[3] + 2*dbetau_dx[5]*gtdd[4] - 2*dbetau_dx[8]*x4;
	(*dgtdd_dt)[4] = -Atdd[4]*x1 - 2.0/3.0*dbetau_dx[0]*gtdd[4] + dbetau_dx[3]*gtdd[2] + dbetau_dx[4]*x5 + dbetau_dx[5]*gtdd[5] + dbetau_dx[6]*gtdd[1] + dbetau_dx[7]*gtdd[3] + dbetau_dx[8]*x5;
	(*dgtdd_dt)[5] = -2*Atdd[5]*alp - 2*dbetau_dx[0]*x6 - 2*dbetau_dx[4]*x6 + 2*dbetau_dx[6]*gtdd[2] + 2*dbetau_dx[7]*gtdd[4] + (4.0/3.0)*dbetau_dx[8]*gtdd[5];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Khat_rhs(
	double alp,
	double theta,
	double Ktr,
	double S,
	double rho,
	double kappa1,
	double kappa2,
	double Asqr,
	double DiDialp,
	double * __restrict__ dKhat_dt
)
{
	*dKhat_dt = -DiDialp - alp*kappa1*theta*(kappa2 - 1) + (1.0/3.0)*alp*(3*Asqr + ((Ktr)*(Ktr))) + 4*M_PI*alp*(S + rho);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Gammatilde_rhs(
	double alp,
	double W,
	const double Gammatu[3],
	const double Si[3],
	double kappa1,
	const double gtuu[6],
	const double Atuu[6],
	const double Gammatudd[18],
	const double GammatDu[3],
	const double dbetau_dx[9],
	const double dKhat_dx[3],
	const double dW_dx[3],
	const double dalp_dx[3],
	const double dtheta_dx[3],
	const double ddbetau_dx2[18],
	double (*dGammatu_dt)[3]
)
{
	double x0 = 2*dalp_dx[0];
	double x1 = 2*dalp_dx[1];
	double x2 = 2*dalp_dx[2];
	double x3 = (2.0/3.0)*GammatDu[0];
	double x4 = 2*alp;
	double x5 = Atuu[0]*x4;
	double x6 = 4*alp;
	double x7 = Atuu[1]*x6;
	double x8 = Atuu[2]*x6;
	double x9 = Atuu[3]*x4;
	double x10 = Atuu[4]*x6;
	double x11 = Atuu[5]*x4;
	double x12 = 16*M_PI*alp;
	double x13 = Si[0]*x12;
	double x14 = Si[1]*x12;
	double x15 = Si[2]*x12;
	double x16 = 6*alp/W;
	double x17 = dW_dx[0]*x16;
	double x18 = dW_dx[1]*x16;
	double x19 = dW_dx[2]*x16;
	double x20 = (2.0/3.0)*alp;
	double x21 = x20*(2*dKhat_dx[0] + dtheta_dx[0]);
	double x22 = x20*(2*dKhat_dx[1] + dtheta_dx[1]);
	double x23 = x20*(2*dKhat_dx[2] + dtheta_dx[2]);
	double x24 = kappa1*x4;
	double x25 = (2.0/3.0)*GammatDu[1];
	double x26 = (2.0/3.0)*GammatDu[2];
	(*dGammatu_dt)[0] = -Atuu[0]*x0 - Atuu[0]*x17 - Atuu[1]*x1 - Atuu[1]*x18 - Atuu[2]*x19 - Atuu[2]*x2 - 1.0/3.0*GammatDu[0]*dbetau_dx[0] - GammatDu[1]*dbetau_dx[3] - GammatDu[2]*dbetau_dx[6] + Gammatudd[0]*x5 + Gammatudd[1]*x7 + Gammatudd[2]*x8 + Gammatudd[3]*x9 + Gammatudd[4]*x10 + Gammatudd[5]*x11 + dbetau_dx[4]*x3 + dbetau_dx[8]*x3 + (4.0/3.0)*ddbetau_dx2[0]*gtuu[0] + (1.0/3.0)*ddbetau_dx2[10]*gtuu[1] + 2*ddbetau_dx2[12]*gtuu[4] + (1.0/3.0)*ddbetau_dx2[13]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[14]*gtuu[1] + ddbetau_dx2[15]*gtuu[5] + (1.0/3.0)*ddbetau_dx2[17]*gtuu[2] + (7.0/3.0)*ddbetau_dx2[3]*gtuu[1] + (1.0/3.0)*ddbetau_dx2[4]*gtuu[0] + (7.0/3.0)*ddbetau_dx2[6]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[8]*gtuu[0] + ddbetau_dx2[9]*gtuu[3] - gtuu[0]*x13 - gtuu[0]*x21 - gtuu[1]*x14 - gtuu[1]*x22 - gtuu[2]*x15 - gtuu[2]*x23 + x24*(GammatDu[0] - Gammatu[0]);
	(*dGammatu_dt)[1] = -Atuu[1]*x0 - Atuu[1]*x17 - Atuu[3]*x1 - Atuu[3]*x18 - Atuu[4]*x19 - Atuu[4]*x2 - GammatDu[0]*dbetau_dx[1] - 1.0/3.0*GammatDu[1]*dbetau_dx[4] - GammatDu[2]*dbetau_dx[7] + Gammatudd[10]*x10 + Gammatudd[11]*x11 + Gammatudd[6]*x5 + Gammatudd[7]*x7 + Gammatudd[8]*x8 + Gammatudd[9]*x9 + dbetau_dx[0]*x25 + dbetau_dx[8]*x25 + (1.0/3.0)*ddbetau_dx2[0]*gtuu[1] + (4.0/3.0)*ddbetau_dx2[10]*gtuu[3] + (7.0/3.0)*ddbetau_dx2[13]*gtuu[4] + (1.0/3.0)*ddbetau_dx2[14]*gtuu[3] + ddbetau_dx2[16]*gtuu[5] + (1.0/3.0)*ddbetau_dx2[17]*gtuu[4] + ddbetau_dx2[1]*gtuu[0] + (1.0/3.0)*ddbetau_dx2[3]*gtuu[3] + (7.0/3.0)*ddbetau_dx2[4]*gtuu[1] + (1.0/3.0)*ddbetau_dx2[6]*gtuu[4] + 2*ddbetau_dx2[7]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[8]*gtuu[1] - gtuu[1]*x13 - gtuu[1]*x21 - gtuu[3]*x14 - gtuu[3]*x22 - gtuu[4]*x15 - gtuu[4]*x23 + x24*(GammatDu[1] - Gammatu[1]);
	(*dGammatu_dt)[2] = -Atuu[2]*x0 - Atuu[2]*x17 - Atuu[4]*x1 - Atuu[4]*x18 - Atuu[5]*x19 - Atuu[5]*x2 - GammatDu[0]*dbetau_dx[2] - GammatDu[1]*dbetau_dx[5] - 1.0/3.0*GammatDu[2]*dbetau_dx[8] + Gammatudd[12]*x5 + Gammatudd[13]*x7 + Gammatudd[14]*x8 + Gammatudd[15]*x9 + Gammatudd[16]*x10 + Gammatudd[17]*x11 + dbetau_dx[0]*x26 + dbetau_dx[4]*x26 + (1.0/3.0)*ddbetau_dx2[0]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[10]*gtuu[4] + ddbetau_dx2[11]*gtuu[3] + (1.0/3.0)*ddbetau_dx2[13]*gtuu[5] + (7.0/3.0)*ddbetau_dx2[14]*gtuu[4] + (4.0/3.0)*ddbetau_dx2[17]*gtuu[5] + ddbetau_dx2[2]*gtuu[0] + (1.0/3.0)*ddbetau_dx2[3]*gtuu[4] + (1.0/3.0)*ddbetau_dx2[4]*gtuu[2] + 2*ddbetau_dx2[5]*gtuu[1] + (1.0/3.0)*ddbetau_dx2[6]*gtuu[5] + (7.0/3.0)*ddbetau_dx2[8]*gtuu[2] - gtuu[2]*x13 - gtuu[2]*x21 - gtuu[4]*x14 - gtuu[4]*x22 - gtuu[5]*x15 - gtuu[5]*x23 + x24*(GammatDu[2] - Gammatu[2]);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_theta_rhs(
	double alp,
	double theta,
	double Khat,
	double rho,
	double kappa1,
	double kappa2,
	double theta_damp_fact,
	double Asqr,
	double Rtrace,
	double * __restrict__ dtheta_dt
)
{
	*dtheta_dt = -alp*theta_damp_fact*((1.0/2.0)*Asqr - 1.0/2.0*Rtrace + kappa1*theta*(kappa2 + 2) + 8*M_PI*rho - 1.0/3.0*((Khat + 2*theta)*(Khat + 2*theta)));
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Atdd_rhs(
	const double gtdd[6],
	const double Atdd[6],
	double alp,
	double W,
	double Ktr,
	double S,
	const double Sij[6],
	const double gtuu[6],
	const double W2DiDjalp[6],
	double DiDialp,
	const double W2Rdd[6],
	double Rtrace,
	const double dbetau_dx[9],
	double (*dAtdd_dt)[6]
)
{
	double x0 = (2.0/3.0)*Atdd[0];
	double x1 = 8*M_PI;
	double x2 = ((W)*(W))*x1;
	double x3 = DiDialp - alp*(Rtrace - S*x1);
	double x4 = Atdd[1]*gtuu[1];
	double x5 = Atdd[2]*gtuu[2];
	double x6 = alp*(Atdd[0]*gtuu[0] + x4 + x5);
	double x7 = 2*Atdd[1];
	double x8 = alp*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]);
	double x9 = 2*Atdd[2];
	double x10 = alp*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]);
	double x11 = (1.0/3.0)*Atdd[1];
	double x12 = (2.0/3.0)*dbetau_dx[8];
	double x13 = Ktr*alp;
	double x14 = (1.0/3.0)*x3;
	double x15 = 2*Atdd[3];
	double x16 = 2*Atdd[4];
	double x17 = (1.0/3.0)*Atdd[2];
	double x18 = (2.0/3.0)*dbetau_dx[4];
	double x19 = 2*Atdd[5];
	double x20 = (2.0/3.0)*dbetau_dx[0];
	double x21 = alp*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]);
	double x22 = Atdd[4]*gtuu[4];
	double x23 = alp*(Atdd[3]*gtuu[3] + x22 + x4);
	double x24 = alp*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]);
	double x25 = (1.0/3.0)*Atdd[4];
	(*dAtdd_dt)[0] = Atdd[0]*Ktr*alp + (4.0/3.0)*Atdd[0]*dbetau_dx[0] - 2*Atdd[0]*x6 + 2*Atdd[1]*dbetau_dx[1] + 2*Atdd[2]*dbetau_dx[2] - W2DiDjalp[0] - alp*(Sij[0]*x2 - W2Rdd[0]) - dbetau_dx[4]*x0 - dbetau_dx[8]*x0 + (1.0/3.0)*gtdd[0]*x3 - x10*x9 - x7*x8;
	(*dAtdd_dt)[1] = Atdd[0]*dbetau_dx[3] - Atdd[1]*x12 + Atdd[1]*x13 + Atdd[2]*dbetau_dx[5] + Atdd[3]*dbetau_dx[1] + Atdd[4]*dbetau_dx[2] - W2DiDjalp[1] - alp*(Sij[1]*x2 - W2Rdd[1]) + dbetau_dx[0]*x11 + dbetau_dx[4]*x11 + gtdd[1]*x14 - x10*x16 - x15*x8 - x6*x7;
	(*dAtdd_dt)[2] = Atdd[0]*dbetau_dx[6] + Atdd[1]*dbetau_dx[7] + Atdd[2]*x13 - Atdd[2]*x18 + Atdd[4]*dbetau_dx[1] + Atdd[5]*dbetau_dx[2] - W2DiDjalp[2] - alp*(Sij[2]*x2 - W2Rdd[2]) + dbetau_dx[0]*x17 + dbetau_dx[8]*x17 + gtdd[2]*x14 - x10*x19 - x16*x8 - x6*x9;
	(*dAtdd_dt)[3] = 2*Atdd[1]*dbetau_dx[3] + Atdd[3]*Ktr*alp + (4.0/3.0)*Atdd[3]*dbetau_dx[4] - Atdd[3]*x12 - Atdd[3]*x20 + 2*Atdd[4]*dbetau_dx[5] - W2DiDjalp[3] - alp*(Sij[3]*x2 - W2Rdd[3]) + (1.0/3.0)*gtdd[3]*x3 - x15*x23 - x16*x24 - x21*x7;
	(*dAtdd_dt)[4] = Atdd[1]*dbetau_dx[6] + Atdd[2]*dbetau_dx[3] + Atdd[3]*dbetau_dx[7] + Atdd[4]*x13 - Atdd[4]*x20 + Atdd[5]*dbetau_dx[5] - W2DiDjalp[4] - alp*(Sij[4]*x2 - W2Rdd[4]) + dbetau_dx[4]*x25 + dbetau_dx[8]*x25 + gtdd[4]*x14 - x16*x23 - x19*x24 - x21*x9;
	(*dAtdd_dt)[5] = 2*Atdd[2]*dbetau_dx[6] + 2*Atdd[4]*dbetau_dx[7] + Atdd[5]*Ktr*alp + (4.0/3.0)*Atdd[5]*dbetau_dx[8] - Atdd[5]*x18 - Atdd[5]*x20 - W2DiDjalp[5] - alp*x16*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4]) - alp*x19*(Atdd[5]*gtuu[5] + x22 + x5) - alp*x9*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2]) - alp*(Sij[5]*x2 - W2Rdd[5]) + (1.0/3.0)*gtdd[5]*x3;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_alpha_rhs(
	double alp,
	double Khat,
	double * __restrict__ dalpha_dt
)
{
	*dalpha_dt = -2*Khat*alp;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_beta_rhs(
	const double Bdriver[3],
	double (*dbeta_dt)[3]
)
{
	(*dbeta_dt)[0] = (3.0/4.0)*Bdriver[0];
	(*dbeta_dt)[1] = (3.0/4.0)*Bdriver[1];
	(*dbeta_dt)[2] = (3.0/4.0)*Bdriver[2];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Bdriver_rhs(
	const double Bdriver[3],
	double eta,
	const double dGammatu_dt[3],
	double (*dBd_dt)[3]
)
{
	(*dBd_dt)[0] = -Bdriver[0]*eta + dGammatu_dt[0];
	(*dBd_dt)[1] = -Bdriver[1]*eta + dGammatu_dt[1];
	(*dBd_dt)[2] = -Bdriver[2]*eta + dGammatu_dt[2];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_constraints(
	const double Atdd[6],
	double W,
	double theta,
	double Khat,
	double rho,
	const double Si[3],
	const double gtuu[6],
	const double Atuu[6],
	double Asqr,
	const double Gammatudd[18],
	const double GammatDu[3],
	double Rtrace,
	const double dgtdd_dx[18],
	const double dAtdd_dx[18],
	const double dKhat_dx[3],
	const double dW_dx[3],
	const double dtheta_dx[3],
	double * __restrict__ H,
	double (*M)[3]
)
{
	double x0 = Atuu[0]*dgtdd_dx[0];
	double x1 = Atuu[0]*dgtdd_dx[1];
	double x2 = Atuu[0]*dgtdd_dx[2];
	double x3 = Atuu[1]*gtuu[0];
	double x4 = Atuu[1]*gtuu[1];
	double x5 = Atuu[1]*gtuu[2];
	double x6 = Atuu[2]*gtuu[0];
	double x7 = Atuu[2]*gtuu[1];
	double x8 = Atuu[2]*gtuu[2];
	double x9 = Atuu[3]*dgtdd_dx[10];
	double x10 = Atuu[3]*dgtdd_dx[7];
	double x11 = Atuu[3]*dgtdd_dx[9];
	double x12 = Atuu[4]*gtuu[1];
	double x13 = Atuu[4]*gtuu[2];
	double x14 = Atuu[4]*gtuu[0];
	double x15 = Atuu[5]*dgtdd_dx[14];
	double x16 = Atuu[5]*dgtdd_dx[16];
	double x17 = Atuu[5]*dgtdd_dx[17];
	double x18 = 8*M_PI;
	double x19 = Si[0]*x18;
	double x20 = Si[1]*x18;
	double x21 = Si[2]*x18;
	double x22 = ((gtuu[2])*(gtuu[2]));
	double x23 = ((gtuu[1])*(gtuu[1]));
	double x24 = 3/W;
	double x25 = dW_dx[0]*x24;
	double x26 = dW_dx[1]*x24;
	double x27 = dW_dx[2]*x24;
	double x28 = (2.0/3.0)*(dKhat_dx[0] + 2*dtheta_dx[0]);
	double x29 = (2.0/3.0)*(dKhat_dx[1] + 2*dtheta_dx[1]);
	double x30 = (2.0/3.0)*(dKhat_dx[2] + 2*dtheta_dx[2]);
	double x31 = Atdd[1]*gtuu[1];
	double x32 = Atdd[2]*gtuu[2];
	double x33 = Atuu[1]*gtuu[3];
	double x34 = Atuu[1]*gtuu[4];
	double x35 = Atuu[2]*gtuu[3];
	double x36 = Atuu[2]*gtuu[4];
	double x37 = Atuu[4]*gtuu[3];
	double x38 = Atuu[4]*gtuu[4];
	double x39 = ((gtuu[4])*(gtuu[4]));
	double x40 = Atdd[4]*gtuu[4];
	double x41 = Atuu[1]*gtuu[5];
	double x42 = Atuu[2]*gtuu[5];
	double x43 = Atuu[4]*gtuu[5];
	*H = -Asqr + Rtrace - 16*M_PI*rho + (2.0/3.0)*((Khat + 2*theta)*(Khat + 2*theta));
	(*M)[0] = Atuu[0]*Gammatudd[0] - Atuu[0]*x25 + 2*Atuu[1]*Gammatudd[1] - Atuu[1]*x26 + 2*Atuu[2]*Gammatudd[2] - Atuu[2]*x27 + Atuu[3]*Gammatudd[3] + 2*Atuu[4]*Gammatudd[4] + Atuu[5]*Gammatudd[5] - GammatDu[0]*(Atdd[0]*gtuu[0] + x31 + x32) - GammatDu[1]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) - GammatDu[2]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2]) + dAtdd_dx[0]*((gtuu[0])*(gtuu[0])) + dAtdd_dx[10]*gtuu[1]*gtuu[4] + dAtdd_dx[10]*gtuu[2]*gtuu[3] + dAtdd_dx[11]*gtuu[2]*gtuu[4] + dAtdd_dx[12]*gtuu[0]*gtuu[2] + dAtdd_dx[13]*gtuu[0]*gtuu[4] + dAtdd_dx[13]*gtuu[1]*gtuu[2] + dAtdd_dx[14]*gtuu[0]*gtuu[5] + dAtdd_dx[14]*x22 + dAtdd_dx[15]*gtuu[1]*gtuu[4] + dAtdd_dx[16]*gtuu[1]*gtuu[5] + dAtdd_dx[16]*gtuu[2]*gtuu[4] + dAtdd_dx[17]*gtuu[2]*gtuu[5] + 2*dAtdd_dx[1]*gtuu[0]*gtuu[1] + 2*dAtdd_dx[2]*gtuu[0]*gtuu[2] + dAtdd_dx[3]*x23 + 2*dAtdd_dx[4]*gtuu[1]*gtuu[2] + dAtdd_dx[5]*x22 + dAtdd_dx[6]*gtuu[0]*gtuu[1] + dAtdd_dx[7]*gtuu[0]*gtuu[3] + dAtdd_dx[7]*x23 + dAtdd_dx[8]*gtuu[0]*gtuu[4] + dAtdd_dx[8]*gtuu[1]*gtuu[2] + dAtdd_dx[9]*gtuu[1]*gtuu[3] - dgtdd_dx[10]*x12 - dgtdd_dx[11]*x13 - dgtdd_dx[12]*x6 - dgtdd_dx[13]*x14 - dgtdd_dx[13]*x7 - dgtdd_dx[14]*x8 - dgtdd_dx[15]*x12 - dgtdd_dx[16]*x13 - dgtdd_dx[1]*x3 - dgtdd_dx[2]*x6 - dgtdd_dx[3]*x4 - dgtdd_dx[4]*x5 - dgtdd_dx[4]*x7 - dgtdd_dx[5]*x8 - dgtdd_dx[6]*x3 - dgtdd_dx[7]*x4 - dgtdd_dx[8]*x14 - dgtdd_dx[8]*x5 - gtuu[0]*x0 - gtuu[0]*x10 - gtuu[0]*x15 - gtuu[0]*x19 - gtuu[0]*x28 - gtuu[1]*x1 - gtuu[1]*x11 - gtuu[1]*x16 - gtuu[1]*x20 - gtuu[1]*x29 - gtuu[2]*x17 - gtuu[2]*x2 - gtuu[2]*x21 - gtuu[2]*x30 - gtuu[2]*x9;
	(*M)[1] = Atuu[0]*Gammatudd[6] + 2*Atuu[1]*Gammatudd[7] - Atuu[1]*x25 + 2*Atuu[2]*Gammatudd[8] + Atuu[3]*Gammatudd[9] - Atuu[3]*x26 + 2*Atuu[4]*Gammatudd[10] - Atuu[4]*x27 + Atuu[5]*Gammatudd[11] - GammatDu[0]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) - GammatDu[1]*(Atdd[3]*gtuu[3] + x31 + x40) - GammatDu[2]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4]) + dAtdd_dx[0]*gtuu[0]*gtuu[1] + 2*dAtdd_dx[10]*gtuu[3]*gtuu[4] + dAtdd_dx[11]*x39 + dAtdd_dx[12]*gtuu[1]*gtuu[2] + dAtdd_dx[13]*gtuu[1]*gtuu[4] + dAtdd_dx[13]*gtuu[2]*gtuu[3] + dAtdd_dx[14]*gtuu[1]*gtuu[5] + dAtdd_dx[14]*gtuu[2]*gtuu[4] + dAtdd_dx[15]*gtuu[3]*gtuu[4] + dAtdd_dx[16]*gtuu[3]*gtuu[5] + dAtdd_dx[16]*x39 + dAtdd_dx[17]*gtuu[4]*gtuu[5] + dAtdd_dx[1]*gtuu[0]*gtuu[3] + dAtdd_dx[1]*x23 + dAtdd_dx[2]*gtuu[0]*gtuu[4] + dAtdd_dx[2]*gtuu[1]*gtuu[2] + dAtdd_dx[3]*gtuu[1]*gtuu[3] + dAtdd_dx[4]*gtuu[1]*gtuu[4] + dAtdd_dx[4]*gtuu[2]*gtuu[3] + dAtdd_dx[5]*gtuu[2]*gtuu[4] + dAtdd_dx[6]*x23 + 2*dAtdd_dx[7]*gtuu[1]*gtuu[3] + 2*dAtdd_dx[8]*gtuu[1]*gtuu[4] + dAtdd_dx[9]*((gtuu[3])*(gtuu[3])) - dgtdd_dx[10]*x37 - dgtdd_dx[11]*x38 - dgtdd_dx[12]*x7 - dgtdd_dx[13]*x12 - dgtdd_dx[13]*x35 - dgtdd_dx[14]*x36 - dgtdd_dx[15]*x37 - dgtdd_dx[16]*x38 - dgtdd_dx[1]*x4 - dgtdd_dx[2]*x7 - dgtdd_dx[3]*x33 - dgtdd_dx[4]*x34 - dgtdd_dx[4]*x35 - dgtdd_dx[5]*x36 - dgtdd_dx[6]*x4 - dgtdd_dx[7]*x33 - dgtdd_dx[8]*x12 - dgtdd_dx[8]*x34 - gtuu[1]*x0 - gtuu[1]*x10 - gtuu[1]*x15 - gtuu[1]*x19 - gtuu[1]*x28 - gtuu[3]*x1 - gtuu[3]*x11 - gtuu[3]*x16 - gtuu[3]*x20 - gtuu[3]*x29 - gtuu[4]*x17 - gtuu[4]*x2 - gtuu[4]*x21 - gtuu[4]*x30 - gtuu[4]*x9;
	(*M)[2] = Atuu[0]*Gammatudd[12] + 2*Atuu[1]*Gammatudd[13] + 2*Atuu[2]*Gammatudd[14] - Atuu[2]*x25 + Atuu[3]*Gammatudd[15] + 2*Atuu[4]*Gammatudd[16] - Atuu[4]*x26 + Atuu[5]*Gammatudd[17] - Atuu[5]*x27 - GammatDu[0]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) - GammatDu[1]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) - GammatDu[2]*(Atdd[5]*gtuu[5] + x32 + x40) + dAtdd_dx[0]*gtuu[0]*gtuu[2] + dAtdd_dx[10]*gtuu[3]*gtuu[5] + dAtdd_dx[10]*x39 + dAtdd_dx[11]*gtuu[4]*gtuu[5] + dAtdd_dx[12]*x22 + 2*dAtdd_dx[13]*gtuu[2]*gtuu[4] + 2*dAtdd_dx[14]*gtuu[2]*gtuu[5] + dAtdd_dx[15]*x39 + 2*dAtdd_dx[16]*gtuu[4]*gtuu[5] + dAtdd_dx[17]*((gtuu[5])*(gtuu[5])) + dAtdd_dx[1]*gtuu[0]*gtuu[4] + dAtdd_dx[1]*gtuu[1]*gtuu[2] + dAtdd_dx[2]*gtuu[0]*gtuu[5] + dAtdd_dx[2]*x22 + dAtdd_dx[3]*gtuu[1]*gtuu[4] + dAtdd_dx[4]*gtuu[1]*gtuu[5] + dAtdd_dx[4]*gtuu[2]*gtuu[4] + dAtdd_dx[5]*gtuu[2]*gtuu[5] + dAtdd_dx[6]*gtuu[1]*gtuu[2] + dAtdd_dx[7]*gtuu[1]*gtuu[4] + dAtdd_dx[7]*gtuu[2]*gtuu[3] + dAtdd_dx[8]*gtuu[1]*gtuu[5] + dAtdd_dx[8]*gtuu[2]*gtuu[4] + dAtdd_dx[9]*gtuu[3]*gtuu[4] - dgtdd_dx[10]*x38 - dgtdd_dx[11]*x43 - dgtdd_dx[12]*x8 - dgtdd_dx[13]*x13 - dgtdd_dx[13]*x36 - dgtdd_dx[14]*x42 - dgtdd_dx[15]*x38 - dgtdd_dx[16]*x43 - dgtdd_dx[1]*x5 - dgtdd_dx[2]*x8 - dgtdd_dx[3]*x34 - dgtdd_dx[4]*x36 - dgtdd_dx[4]*x41 - dgtdd_dx[5]*x42 - dgtdd_dx[6]*x5 - dgtdd_dx[7]*x34 - dgtdd_dx[8]*x13 - dgtdd_dx[8]*x41 - gtuu[2]*x0 - gtuu[2]*x10 - gtuu[2]*x15 - gtuu[2]*x19 - gtuu[2]*x28 - gtuu[4]*x1 - gtuu[4]*x11 - gtuu[4]*x16 - gtuu[4]*x20 - gtuu[4]*x29 - gtuu[5]*x17 - gtuu[5]*x2 - gtuu[5]*x21 - gtuu[5]*x30 - gtuu[5]*x9;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_matter_sources(
	const double gtdd[6],
	const double betau[3],
	double alp,
	double W,
	const double gtuu[6],
	const double zvec[3],
	const double Bvec[3],
	double rho0,
	double press,
	double eps,
	double * __restrict__ rho_adm,
	double * __restrict__ S_adm_trace,
	double (*Sd_adm)[3],
	double (*Sdd_adm)[6]
)
{
	double x0 = ((alp)*(alp));
	double x1 = 1/(x0);
	double x2 = 1/(((W)*(W)));
	double x3 = betau[0]*gtdd[0] + betau[1]*gtdd[1] + betau[2]*gtdd[2];
	double x4 = x2*x3;
	double x5 = betau[0]*gtdd[1] + betau[1]*gtdd[3] + betau[2]*gtdd[4];
	double x6 = x2*x5;
	double x7 = betau[0]*gtdd[2] + betau[1]*gtdd[4] + betau[2]*gtdd[5];
	double x8 = x2*x7;
	double x9 = betau[0]*x4 + betau[1]*x6 + betau[2]*x8 - x0;
	double x10 = gtdd[0]*zvec[0] + gtdd[1]*zvec[1] + gtdd[2]*zvec[2];
	double x11 = gtdd[1]*zvec[0] + gtdd[3]*zvec[1] + gtdd[4]*zvec[2];
	double x12 = gtdd[2]*zvec[0] + gtdd[4]*zvec[1] + gtdd[5]*zvec[2];
	double x13 = x10*x2*zvec[0] + x11*x2*zvec[1] + x12*x2*zvec[2] + 1;
	double x14 = Bvec[0]*x10 + Bvec[1]*x11 + Bvec[2]*x12;
	double x15 = x2*(Bvec[0]*(Bvec[0]*gtdd[0] + Bvec[1]*gtdd[1] + Bvec[2]*gtdd[2]) + Bvec[1]*(Bvec[0]*gtdd[1] + Bvec[1]*gtdd[3] + Bvec[2]*gtdd[4]) + Bvec[2]*(Bvec[0]*gtdd[2] + Bvec[1]*gtdd[4] + Bvec[2]*gtdd[5]) + ((x14)*(x14))*x2)/x13;
	double x16 = 2*press + x15;
	double x17 = sqrt(x13);
	double x18 = 1/(x17);
	double x19 = alp*x18;
	double x20 = -betau[0] + x19*zvec[0];
	double x21 = -betau[1] + x19*zvec[1];
	double x22 = -betau[2] + x19*zvec[2];
	double x23 = x20*x4 + x21*x6 + x22*x8 + x9;
	double x24 = x1*x13*(eps*rho0 + press + rho0 + x15);
	double x25 = 1/(alp);
	double x26 = x14*x25;
	double x27 = x17*x2*x26;
	double x28 = x18*(Bvec[0] + x20*x27);
	double x29 = x18*(Bvec[1] + x21*x27);
	double x30 = x18*(Bvec[2] + x22*x27);
	double x31 = x26*x9 + x28*x3 + x29*x5 + x30*x7;
	double x32 = gtdd[0]*x16;
	double x33 = gtdd[0]*x20 + gtdd[1]*x21 + gtdd[2]*x22 + x3;
	double x34 = x2*x24;
	double x35 = ((x33)*(x33))*x34;
	double x36 = gtdd[0]*x28 + gtdd[1]*x29 + gtdd[2]*x30 + x26*x4;
	double x37 = x2*((x36)*(x36));
	double x38 = (1.0/2.0)*x32 + x35 - x37;
	double x39 = gtdd[3]*x16;
	double x40 = gtdd[1]*x20 + gtdd[3]*x21 + gtdd[4]*x22 + x5;
	double x41 = x34*((x40)*(x40));
	double x42 = gtdd[1]*x28 + gtdd[3]*x29 + gtdd[4]*x30 + x26*x6;
	double x43 = x2*((x42)*(x42));
	double x44 = (1.0/2.0)*x39 + x41 - x43;
	double x45 = gtdd[5]*x16;
	double x46 = gtdd[2]*x20 + gtdd[4]*x21 + gtdd[5]*x22 + x7;
	double x47 = x34*((x46)*(x46));
	double x48 = gtdd[2]*x28 + gtdd[4]*x29 + gtdd[5]*x30 + x26*x8;
	double x49 = x2*((x48)*(x48));
	double x50 = (1.0/2.0)*x45 + x47 - x49;
	double x51 = gtdd[1]*x16;
	double x52 = x33*x34;
	double x53 = x40*x52;
	double x54 = x2*x36;
	double x55 = x42*x54;
	double x56 = x51 + 2*x53 - 2*x55;
	double x57 = betau[0]*x2;
	double x58 = gtdd[2]*x16;
	double x59 = x46*x52;
	double x60 = x48*x54;
	double x61 = x58 + 2*x59 - 2*x60;
	double x62 = gtdd[4]*x16;
	double x63 = x34*x40*x46;
	double x64 = x2*x42*x48;
	double x65 = x62 + 2*x63 - 2*x64;
	double x66 = betau[1]*x2;
	double x67 = x16*x3;
	double x68 = x23*x24;
	double x69 = x33*x68;
	double x70 = x2*x31;
	double x71 = x36*x70;
	double x72 = x16*x5;
	double x73 = x40*x68;
	double x74 = x42*x70;
	double x75 = x16*x7;
	double x76 = x46*x68;
	double x77 = x48*x70;
	double x78 = (1.0/2.0)*x56;
	double x79 = (1.0/2.0)*betau[2];
	double x80 = x2*x25;
	double x81 = x2*((1.0/2.0)*x51 + x53 - x55);
	double x82 = x2*((1.0/2.0)*x58 + x59 - x60);
	double x83 = x2*((1.0/2.0)*x62 + x63 - x64);
	*rho_adm = x1*(((betau[0])*(betau[0]))*x2*x38 + ((betau[1])*(betau[1]))*x2*x44 + betau[1]*x56*x57 + ((betau[2])*(betau[2]))*x2*x50 - betau[2]*x2*(x75 + 2*x76 - 2*x77) + betau[2]*x57*x61 + betau[2]*x65*x66 + (1.0/2.0)*x16*x9 + ((x23)*(x23))*x24 - x57*(x67 + 2*x69 - 2*x71) - x66*(x72 + 2*x73 - 2*x74) - ((x31)*(x31))/((W)*(W)*(W)*(W)));
	*S_adm_trace = gtuu[0]*x38 + gtuu[1]*x56 + gtuu[2]*x61 + gtuu[3]*x44 + gtuu[4]*x65 + gtuu[5]*x50;
	(*Sd_adm)[0] = x80*(betau[0]*x38 + betau[1]*x78 + x61*x79 - 1.0/2.0*x67 - x69 + x71);
	(*Sd_adm)[1] = x80*(betau[0]*x78 + betau[1]*x44 + x65*x79 - 1.0/2.0*x72 - x73 + x74);
	(*Sd_adm)[2] = x80*((1.0/2.0)*betau[0]*x61 + (1.0/2.0)*betau[1]*x65 + betau[2]*x50 - 1.0/2.0*x75 - x76 + x77);
	(*Sdd_adm)[0] = x2*((1.0/2.0)*x32 + x35 - x37);
	(*Sdd_adm)[1] = x81;
	(*Sdd_adm)[2] = x82;
	(*Sdd_adm)[3] = x2*((1.0/2.0)*x39 + x41 - x43);
	(*Sdd_adm)[4] = x83;
	(*Sdd_adm)[5] = x2*((1.0/2.0)*x45 + x47 - x49);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_gammatilde_initial(
	const double gtuu[6],
	const double dgtdd_dx[18],
	double (*Gammatu_id)[3]
)
{
	double x0 = gtuu[1]*gtuu[4];
	double x1 = gtuu[2]*gtuu[3];
	double x2 = gtuu[2]*gtuu[4];
	double x3 = dgtdd_dx[12]*gtuu[2];
	double x4 = gtuu[0]*gtuu[4];
	double x5 = gtuu[1]*gtuu[2];
	double x6 = dgtdd_dx[14]*gtuu[5];
	double x7 = dgtdd_dx[16]*gtuu[5];
	double x8 = dgtdd_dx[17]*gtuu[5];
	double x9 = gtuu[0]*gtuu[1];
	double x10 = gtuu[0]*gtuu[3];
	double x11 = gtuu[1]*gtuu[3];
	double x12 = ((gtuu[2])*(gtuu[2]));
	double x13 = ((gtuu[1])*(gtuu[1]));
	double x14 = gtuu[0]*gtuu[2];
	double x15 = gtuu[3]*gtuu[4];
	double x16 = ((gtuu[4])*(gtuu[4]));
	double x17 = gtuu[1]*gtuu[5];
	(*Gammatu_id)[0] = dgtdd_dx[0]*((gtuu[0])*(gtuu[0])) + dgtdd_dx[10]*x0 + dgtdd_dx[10]*x1 + dgtdd_dx[11]*x2 + dgtdd_dx[13]*x4 + dgtdd_dx[13]*x5 + dgtdd_dx[14]*x12 + dgtdd_dx[15]*x0 + dgtdd_dx[16]*x2 + 2*dgtdd_dx[1]*x9 + 2*dgtdd_dx[2]*x14 + dgtdd_dx[3]*x13 + 2*dgtdd_dx[4]*x5 + dgtdd_dx[5]*x12 + dgtdd_dx[6]*x9 + dgtdd_dx[7]*x10 + dgtdd_dx[7]*x13 + dgtdd_dx[8]*x4 + dgtdd_dx[8]*x5 + dgtdd_dx[9]*x11 + gtuu[0]*x3 + gtuu[0]*x6 + gtuu[1]*x7 + gtuu[2]*x8;
	(*Gammatu_id)[1] = dgtdd_dx[0]*x9 + 2*dgtdd_dx[10]*x15 + dgtdd_dx[11]*x16 + dgtdd_dx[13]*x0 + dgtdd_dx[13]*x1 + dgtdd_dx[14]*x2 + dgtdd_dx[15]*x15 + dgtdd_dx[16]*x16 + dgtdd_dx[1]*x10 + dgtdd_dx[1]*x13 + dgtdd_dx[2]*x4 + dgtdd_dx[2]*x5 + dgtdd_dx[3]*x11 + dgtdd_dx[4]*x0 + dgtdd_dx[4]*x1 + dgtdd_dx[5]*x2 + dgtdd_dx[6]*x13 + 2*dgtdd_dx[7]*x11 + 2*dgtdd_dx[8]*x0 + dgtdd_dx[9]*((gtuu[3])*(gtuu[3])) + gtuu[1]*x3 + gtuu[1]*x6 + gtuu[3]*x7 + gtuu[4]*x8;
	(*Gammatu_id)[2] = dgtdd_dx[0]*x14 + dgtdd_dx[10]*gtuu[3]*gtuu[5] + dgtdd_dx[10]*x16 + dgtdd_dx[11]*gtuu[4]*gtuu[5] + dgtdd_dx[12]*x12 + 2*dgtdd_dx[13]*x2 + dgtdd_dx[15]*x16 + dgtdd_dx[17]*((gtuu[5])*(gtuu[5])) + dgtdd_dx[1]*x4 + dgtdd_dx[1]*x5 + dgtdd_dx[2]*gtuu[0]*gtuu[5] + dgtdd_dx[2]*x12 + dgtdd_dx[3]*x0 + dgtdd_dx[4]*x17 + dgtdd_dx[4]*x2 + dgtdd_dx[5]*gtuu[2]*gtuu[5] + dgtdd_dx[6]*x5 + dgtdd_dx[7]*x0 + dgtdd_dx[7]*x1 + dgtdd_dx[8]*x17 + dgtdd_dx[8]*x2 + dgtdd_dx[9]*x15 + 2*gtuu[2]*x6 + 2*gtuu[4]*x7;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_psi4(
	const double gtdd[6],
	const double Atdd[6],
	double W,
	double theta,
	double Khat,
	const double gtuu[6],
	const double dgtdd_dx[18],
	const double dAtdd_dx[18],
	const double dKhat_dx[3],
	const double dW_dx[3],
	const double dtheta_dx[3],
	const double ddgtdd_dx2[36],
	const double ddW_dx2[6],
	const double xyz[3],
	double * __restrict__ Psi4Re,
	double * __restrict__ Psi4Im
)
{
	double x0 = ((xyz[0])*(xyz[0]));
	double x1 = ((xyz[1])*(xyz[1]));
	double x2 = ((xyz[2])*(xyz[2]));
	double x3 = sqrt(x0 + x1 + x2);
	double x4 = ((W)*(W));
	double x5 = 1/(x4);
	double x6 = gtdd[0]*x5;
	double x7 = gtdd[3]*x5;
	double x8 = xyz[0]*xyz[1];
	double x9 = 2*gtdd[1];
	double x10 = x5*x9;
	double x11 = x0*x7 + x1*x6 - x10*x8;
	double x12 = 1/(x11);
	double x13 = ((gtdd[4])*(gtdd[4]));
	double x14 = ((gtdd[2])*(gtdd[2]));
	double x15 = ((gtdd[1])*(gtdd[1]));
	double x16 = sqrt((gtdd[0]*gtdd[3]*gtdd[5] - gtdd[0]*x13 + 2*gtdd[1]*gtdd[2]*gtdd[4] - gtdd[3]*x14 - gtdd[5]*x15)/((W)*(W)*(W)*(W)*(W)*(W)));
	double x17 = x16*x4;
	double x18 = gtuu[2]*x17;
	double x19 = gtuu[0]*x4;
	double x20 = xyz[0]*xyz[2];
	double x21 = gtuu[1]*x17;
	double x22 = xyz[1]*xyz[2];
	double x23 = -x0*x18 - x1*x18 + x16*x19*x20 + x21*x22;
	double x24 = (1.0/sqrt(x11));
	double x25 = x24*xyz[1];
	double x26 = x25*x6;
	double x27 = gtuu[4]*x17;
	double x28 = x0*x27;
	double x29 = x1*x27;
	double x30 = gtuu[3]*x4;
	double x31 = x16*x22*x30 + x20*x21 - x28 - x29;
	double x32 = gtdd[1]*x5;
	double x33 = x31*x32;
	double x34 = gtuu[5]*x4;
	double x35 = x16*x34;
	double x36 = x0*x35;
	double x37 = x1*x35;
	double x38 = x18*x20 + x22*x27 - x36 - x37;
	double x39 = x38*x5;
	double x40 = gtdd[2]*x25;
	double x41 = gtdd[1]*x23*x24*x5*xyz[0] + gtdd[3]*x24*x31*x5*xyz[0] + gtdd[4]*x24*x38*x5*xyz[0] - x23*x26 - x25*x33 - x39*x40;
	double x42 = x5*xyz[2];
	double x43 = gtdd[1]*x0*x24*x5 + gtdd[3]*x24*x5*xyz[0]*xyz[1] + gtdd[4]*x24*x5*xyz[0]*xyz[2] - x1*x24*x32 - x26*xyz[0] - x40*x42;
	double x44 = x25*x43 + xyz[0];
	double x45 = ((x44)*(x44));
	double x46 = x24*xyz[0];
	double x47 = -x43*x46 + xyz[1];
	double x48 = ((x47)*(x47));
	double x49 = gtdd[5]*x5;
	double x50 = x44*x47;
	double x51 = 2*x42;
	double x52 = gtdd[2]*x44*x51 + gtdd[4]*x47*x51 + x10*x50 + x2*x49 + x45*x6 + x48*x7;
	double x53 = (1.0/sqrt(x52));
	double x54 = x44*x53;
	double x55 = x47*x53;
	double x56 = x53*xyz[2];
	double x57 = x5*x56;
	double x58 = x53*(gtdd[2]*x23*x57 + gtdd[2]*x39*x54 + gtdd[4]*x31*x57 + gtdd[4]*x39*x55 + x23*x32*x55 + x23*x54*x6 + x31*x55*x7 + x33*x54 + x38*x49*x56);
	double x59 = x23 + x25*x41 - x44*x58;
	double x60 = ((x59)*(x59));
	double x61 = gtuu[1]*x16*x4*xyz[0]*xyz[2] + gtuu[3]*x16*x4*xyz[1]*xyz[2] - x28 - x29 - x41*x46 - x47*x58;
	double x62 = ((x61)*(x61));
	double x63 = gtuu[2]*x16*x4*xyz[0]*xyz[2] + gtuu[4]*x16*x4*xyz[1]*xyz[2] - x36 - x37 - x58*xyz[2];
	double x64 = ((x63)*(x63));
	double x65 = x59*x61;
	double x66 = 2*x63;
	double x67 = x5*x66;
	double x68 = gtdd[2]*x59*x67 + gtdd[4]*x61*x67 + x10*x65 + x49*x64 + x6*x60 + x62*x7;
	double x69 = 1/(x68);
	double x70 = -x1*x12 + x60*x69;
	double x71 = (1.0/3.0)*Khat + (2.0/3.0)*theta;
	double x72 = gtdd[0]*x71;
	double x73 = Atdd[0] + x72;
	double x74 = (1.0/4.0)*x5;
	double x75 = gtuu[0]*x74;
	double x76 = gtdd[1]*x71;
	double x77 = Atdd[1] + x76;
	double x78 = ((x77)*(x77));
	double x79 = gtuu[3]*x74;
	double x80 = 1/(W);
	double x81 = 2*dW_dx[0];
	double x82 = x80*x81;
	double x83 = dgtdd_dx[0] - gtdd[0]*x82;
	double x84 = (1.0/2.0)*x83;
	double x85 = gtuu[2]*x84;
	double x86 = 2*dW_dx[1];
	double x87 = x80*x86;
	double x88 = dgtdd_dx[6] - gtdd[0]*x87;
	double x89 = (1.0/2.0)*x5;
	double x90 = x4*(x5*(dgtdd_dx[1] - gtdd[1]*x82) - x88*x89);
	double x91 = 2*dW_dx[2];
	double x92 = x80*x91;
	double x93 = dgtdd_dx[12] - gtdd[0]*x92;
	double x94 = x4*(x5*(dgtdd_dx[2] - gtdd[2]*x82) - x89*x93);
	double x95 = gtuu[4]*x90 + gtuu[5]*x94 + x85;
	double x96 = dgtdd_dx[11] - gtdd[5]*x87;
	double x97 = gtuu[4]*x96;
	double x98 = x95*x97;
	double x99 = gtuu[1]*x84;
	double x100 = gtuu[3]*x90 + gtuu[4]*x94 + x99;
	double x101 = dgtdd_dx[15] - gtdd[3]*x92;
	double x102 = gtuu[4]*x101;
	double x103 = x100*x102;
	double x104 = gtdd[2]*x71;
	double x105 = Atdd[2] + x104;
	double x106 = ((x105)*(x105));
	double x107 = gtuu[5]*x74;
	double x108 = dgtdd_dx[3] - gtdd[3]*x82;
	double x109 = gtuu[1]*x108;
	double x110 = (1.0/8.0)*x100;
	double x111 = gtuu[1]*x88;
	double x112 = gtuu[0]*x84;
	double x113 = gtuu[1]*x90;
	double x114 = gtuu[2]*x94;
	double x115 = x112 + x113 + x114;
	double x116 = (1.0/8.0)*x115;
	double x117 = gtuu[2]*x93;
	double x118 = dgtdd_dx[5] - gtdd[5]*x82;
	double x119 = gtuu[2]*x118;
	double x120 = (1.0/8.0)*x95;
	double x121 = dgtdd_dx[9] - gtdd[3]*x87;
	double x122 = x110*x121;
	double x123 = dgtdd_dx[17] - gtdd[5]*x92;
	double x124 = gtuu[5]*x123;
	double x125 = x74*(Khat + 2*theta);
	double x126 = (1.0/8.0)*gtuu[1];
	double x127 = 1/(((W)*(W)*(W)));
	double x128 = 2*dgtdd_dx[0];
	double x129 = 4*x127;
	double x130 = dW_dx[0]*x129;
	double x131 = ddW_dx2[0]*x127;
	double x132 = 2*x127;
	double x133 = gtdd[0]*x132;
	double x134 = 1/(((W)*(W)*(W)*(W)));
	double x135 = 6*x134;
	double x136 = ((dW_dx[0])*(dW_dx[0]))*x135;
	double x137 = gtdd[0]*x135;
	double x138 = dW_dx[0]*x137;
	double x139 = -dW_dx[1]*x138 + ddW_dx2[1]*x133 + ddgtdd_dx2[1]*x5 - ddgtdd_dx2[6]*x5 - dgtdd_dx[1]*x130 + gtdd[1]*x136 + x127*(dW_dx[1]*x128 + dgtdd_dx[6]*x81) - x131*x9;
	double x140 = x139*x4;
	double x141 = -x139;
	double x142 = x126*x4;
	double x143 = (1.0/2.0)*gtuu[0];
	double x144 = x143*x88;
	double x145 = (1.0/2.0)*gtuu[1];
	double x146 = x108*x145;
	double x147 = x89*(dgtdd_dx[4] - gtdd[4]*x82);
	double x148 = x89*(dgtdd_dx[8] - gtdd[2]*x87);
	double x149 = x89*(dgtdd_dx[13] - gtdd[1]*x92);
	double x150 = x4*(x147 + x148 - x149);
	double x151 = gtuu[2]*x150;
	double x152 = x144 + x146 + x151;
	double x153 = x152*x83;
	double x154 = (1.0/8.0)*gtuu[2];
	double x155 = 2*x131;
	double x156 = dW_dx[2]*x138 - ddW_dx2[2]*x133 + ddgtdd_dx2[12]*x5 - ddgtdd_dx2[2]*x5 + dgtdd_dx[2]*x130 - gtdd[2]*x136 + gtdd[2]*x155 - x127*(dW_dx[2]*x128 + dgtdd_dx[12]*x81);
	double x157 = x156*x4;
	double x158 = -x156;
	double x159 = x154*x4;
	double x160 = x143*x93;
	double x161 = (1.0/2.0)*x118;
	double x162 = gtuu[2]*x161;
	double x163 = x4*(x147 - x148 + x149);
	double x164 = gtuu[1]*x163;
	double x165 = x160 + x162 + x164;
	double x166 = x154*x165;
	double x167 = x127*(dgtdd_dx[1]*x86 + dgtdd_dx[7]*x81);
	double x168 = gtdd[1]*x129;
	double x169 = dW_dx[1]*x129;
	double x170 = ((dW_dx[1])*(dW_dx[1]));
	double x171 = -ddW_dx2[3]*x133 + ddgtdd_dx2[18]*x5 - dgtdd_dx[6]*x169 + x137*x170;
	double x172 = ddgtdd_dx2[3]*x5 - dgtdd_dx[3]*x130 + gtdd[3]*x136 - gtdd[3]*x155;
	double x173 = 12*dW_dx[0]*dW_dx[1]*gtdd[1]*x134 - ddW_dx2[1]*x168 + 2*ddgtdd_dx2[7]*x5 - 2*x167 - x171 - x172;
	double x174 = (1.0/8.0)*x173;
	double x175 = (1.0/8.0)*gtuu[3];
	double x176 = x145*x88;
	double x177 = (1.0/2.0)*x108;
	double x178 = gtuu[3]*x177;
	double x179 = gtuu[4]*x150;
	double x180 = x176 + x178 + x179;
	double x181 = x108*x180;
	double x182 = x152*x88;
	double x183 = x127*(dgtdd_dx[13]*x81 + dgtdd_dx[1]*x91);
	double x184 = ddgtdd_dx2[13]*x5;
	double x185 = dW_dx[0]*dW_dx[2];
	double x186 = 12*x134;
	double x187 = ddgtdd_dx2[4]*x5;
	double x188 = gtdd[4]*x136;
	double x189 = gtdd[4]*x155;
	double x190 = dgtdd_dx[4]*x130;
	double x191 = -x187 - x188 + x189 + x190;
	double x192 = x127*(dgtdd_dx[12]*x86 + dgtdd_dx[6]*x91);
	double x193 = ddgtdd_dx2[24]*x5;
	double x194 = ddW_dx2[4]*x133;
	double x195 = dW_dx[1]*dW_dx[2];
	double x196 = x137*x195;
	double x197 = x192 - x193 + x194 - x196;
	double x198 = -ddW_dx2[2]*x168 + gtdd[1]*x185*x186 - 2*x183 + 2*x184 + x191 + x197;
	double x199 = (1.0/8.0)*gtuu[4];
	double x200 = x199*x4;
	double x201 = x127*(dgtdd_dx[2]*x86 + dgtdd_dx[8]*x81);
	double x202 = gtdd[2]*x129;
	double x203 = -x192 + x193 - x194 + x196;
	double x204 = x187 + x188 - x189 - x190;
	double x205 = 12*dW_dx[0]*dW_dx[1]*gtdd[2]*x134 - ddW_dx2[1]*x202 + 2*ddgtdd_dx2[8]*x5 - 2*x201 - x203 - x204;
	double x206 = x152*x93;
	double x207 = x145*x93;
	double x208 = gtuu[4]*x161;
	double x209 = gtuu[3]*x163 + x207 + x208;
	double x210 = x108*x209;
	double x211 = (1.0/2.0)*gtuu[2];
	double x212 = x211*x88;
	double x213 = gtuu[4]*x177;
	double x214 = gtuu[5]*x150 + x212 + x213;
	double x215 = x118*x214;
	double x216 = x165*x88;
	double x217 = dgtdd_dx[14]*x81 + dgtdd_dx[2]*x91;
	double x218 = ddgtdd_dx2[14]*x5;
	double x219 = ((dW_dx[2])*(dW_dx[2]));
	double x220 = dW_dx[2]*x129;
	double x221 = ddW_dx2[5]*x133 - ddgtdd_dx2[30]*x5 + dgtdd_dx[12]*x220 - x137*x219;
	double x222 = -ddgtdd_dx2[5]*x5 + dgtdd_dx[5]*x130 - gtdd[5]*x136 + gtdd[5]*x155;
	double x223 = -ddW_dx2[2]*x202 + gtdd[2]*x185*x186 - 2*x127*x217 + 2*x218 + x221 + x222;
	double x224 = (1.0/8.0)*x34;
	double x225 = (1.0/8.0)*gtuu[5];
	double x226 = x165*x93;
	double x227 = x211*x93;
	double x228 = gtuu[5]*x161;
	double x229 = gtuu[4]*x163;
	double x230 = x227 + x228 + x229;
	double x231 = x225*x230;
	double x232 = x73*x89;
	double x233 = x105*x89;
	double x234 = gtuu[2]*x233;
	double x235 = x4*(-x147 + x148 + x149);
	double x236 = (1.0/2.0)*gtuu[4];
	double x237 = gtuu[1]*x150;
	double x238 = x237*x95;
	double x239 = gtuu[2]*x163;
	double x240 = x100*x239;
	double x241 = x4*(-x101*x89 + x5*(dgtdd_dx[10] - gtdd[4]*x87));
	double x242 = gtuu[3]*x241;
	double x243 = x242*x95;
	double x244 = x4*(-x108*x89 + x5*(dgtdd_dx[7] - gtdd[1]*x87));
	double x245 = gtuu[3]*x244;
	double x246 = x115*x245;
	double x247 = x4*(-x118*x89 + x5*(dgtdd_dx[14] - gtdd[2]*x92));
	double x248 = gtuu[5]*x247;
	double x249 = x115*x248;
	double x250 = x4*(x5*(dgtdd_dx[16] - gtdd[4]*x92) - x89*x96);
	double x251 = gtuu[5]*x250;
	double x252 = x100*x251;
	double x253 = x113*x180;
	double x254 = gtuu[1]*x214;
	double x255 = x254*x94;
	double x256 = gtuu[2]*x209;
	double x257 = x256*x90;
	double x258 = x114*x230;
	double x259 = x150*x214;
	double x260 = gtuu[3]*x259;
	double x261 = x180*x229;
	double x262 = x179*x230;
	double x263 = gtuu[5]*x209;
	double x264 = x163*x263;
	double x265 = -gtuu[1]*x232*x77 - gtuu[3]*x122 - gtuu[4]*x233*x77 - 1.0/4.0*x103 - x106*x107 - x109*x110 - x111*x116 - x115*x235*x236 - x116*x117 + x118*x231 - x119*x120 - x120*x124 + x125*x73 + x126*x140 + x126*x153 + x141*x142 + x154*x157 + x158*x159 + x166*x83 + x174*x30 + x175*x181 + x175*x182 + x198*x200 + x199*x206 + x199*x210 + x199*x215 + x199*x216 + x200*x205 + x223*x224 + x225*x226 - x234*x73 - 1.0/4.0*x238 - 1.0/4.0*x240 - 1.0/4.0*x243 - 1.0/4.0*x246 - 1.0/4.0*x249 - 1.0/4.0*x252 + (1.0/4.0)*x253 + (1.0/4.0)*x255 + (1.0/4.0)*x257 + (1.0/4.0)*x258 + (1.0/4.0)*x260 + (1.0/4.0)*x261 + (1.0/4.0)*x262 + (1.0/4.0)*x264 - ((x73)*(x73))*x75 - x78*x79 - 1.0/4.0*x98;
	double x266 = -x0*x12 + x62*x69;
	double x267 = x121*x145;
	double x268 = gtuu[0]*x244 + gtuu[2]*x241 + x267;
	double x269 = x117*x268;
	double x270 = (1.0/2.0)*x121;
	double x271 = gtuu[4]*x270;
	double x272 = gtuu[2]*x244 + gtuu[5]*x241 + x271;
	double x273 = x119*x272;
	double x274 = gtdd[3]*x71;
	double x275 = Atdd[3] + x274;
	double x276 = gtdd[4]*x71;
	double x277 = Atdd[4] + x276;
	double x278 = ((x277)*(x277));
	double x279 = (1.0/8.0)*x268;
	double x280 = gtuu[0]*x83;
	double x281 = gtuu[3]*x270;
	double x282 = gtuu[1]*x244;
	double x283 = gtuu[4]*x241;
	double x284 = x281 + x282 + x283;
	double x285 = (1.0/8.0)*x284;
	double x286 = (1.0/8.0)*x272;
	double x287 = (1.0/8.0)*gtuu[0];
	double x288 = x127*x9;
	double x289 = gtdd[3]*x132;
	double x290 = x135*x170;
	double x291 = dW_dx[0]*dW_dx[1];
	double x292 = gtdd[3]*x135;
	double x293 = ddW_dx2[1]*x289 - ddW_dx2[3]*x288 + ddgtdd_dx2[19]*x5 - ddgtdd_dx2[9]*x5 - dgtdd_dx[7]*x169 + gtdd[1]*x290 + x127*(dgtdd_dx[3]*x86 + dgtdd_dx[9]*x81) - x291*x292;
	double x294 = -x293;
	double x295 = x126*x180;
	double x296 = x127*(dgtdd_dx[10]*x81 + dgtdd_dx[4]*x86);
	double x297 = ddgtdd_dx2[10]*x5;
	double x298 = gtdd[4]*x129;
	double x299 = x127*(dgtdd_dx[15]*x81 + dgtdd_dx[3]*x91);
	double x300 = ddgtdd_dx2[15]*x5;
	double x301 = ddW_dx2[2]*x289;
	double x302 = x185*x292;
	double x303 = x299 - x300 + x301 - x302;
	double x304 = ddgtdd_dx2[20]*x5;
	double x305 = gtdd[2]*x290;
	double x306 = ddW_dx2[3]*x132;
	double x307 = gtdd[2]*x306;
	double x308 = dgtdd_dx[8]*x169;
	double x309 = -x304 - x305 + x307 + x308;
	double x310 = -ddW_dx2[1]*x298 + gtdd[4]*x186*x291 - 2*x296 + 2*x297 + x303 + x309;
	double x311 = x127*(dgtdd_dx[13]*x86 + dgtdd_dx[7]*x91);
	double x312 = x304 + x305 - x307 - x308;
	double x313 = -x299 + x300 - x301 + x302;
	double x314 = 12*dW_dx[1]*dW_dx[2]*gtdd[1]*x134 - ddW_dx2[4]*x168 + 2*ddgtdd_dx2[25]*x5 - 2*x311 - x312 - x313;
	double x315 = x214*x96;
	double x316 = x101*x180;
	double x317 = (1.0/2.0)*x101;
	double x318 = gtuu[3]*x317;
	double x319 = (1.0/2.0)*x96;
	double x320 = gtuu[4]*x319;
	double x321 = gtuu[1]*x235;
	double x322 = x318 + x320 + x321;
	double x323 = x101*x145;
	double x324 = x211*x96;
	double x325 = gtuu[0]*x235 + x323 + x324;
	double x326 = ddW_dx2[4]*x289 + ddgtdd_dx2[22]*x5 - ddgtdd_dx2[27]*x5 - dgtdd_dx[10]*x169 + gtdd[4]*x290 - gtdd[4]*x306 + x127*(dgtdd_dx[15]*x86 + dgtdd_dx[9]*x91) - x195*x292;
	double x327 = -x326;
	double x328 = x121*x199;
	double x329 = ddgtdd_dx2[33]*x5;
	double x330 = dgtdd_dx[10]*x91 + dgtdd_dx[16]*x86;
	double x331 = x127*x330;
	double x332 = dgtdd_dx[15]*x220;
	double x333 = ddW_dx2[5]*x289;
	double x334 = x135*x219;
	double x335 = gtdd[3]*x334;
	double x336 = ddgtdd_dx2[23]*x5 - dgtdd_dx[11]*x169 + gtdd[5]*x290 - gtdd[5]*x306;
	double x337 = 12*dW_dx[1]*dW_dx[2]*gtdd[4]*x134 - ddW_dx2[4]*x298 + 2*ddgtdd_dx2[28]*x5 - x329 - 2*x331 + x332 + x333 - x335 - x336;
	double x338 = gtuu[4]*x317;
	double x339 = gtuu[5]*x319;
	double x340 = gtuu[2]*x235;
	double x341 = x338 + x339 + x340;
	double x342 = x341*x96;
	double x343 = x101*x225;
	double x344 = x77*x89;
	double x345 = x277*x89;
	double x346 = x163*x211;
	double x347 = gtuu[4]*x345;
	double x348 = gtuu[0]*x90;
	double x349 = x284*x348;
	double x350 = gtuu[0]*x94;
	double x351 = x272*x350;
	double x352 = x237*x272;
	double x353 = gtuu[4]*x235;
	double x354 = x268*x353;
	double x355 = x248*x268;
	double x356 = x251*x284;
	double x357 = gtuu[0]*x259;
	double x358 = x241*x254;
	double x359 = x152*x282;
	double x360 = x152*x340;
	double x361 = x151*x341;
	double x362 = x283*x341;
	double x363 = gtuu[4]*x244;
	double x364 = x325*x363;
	double x365 = gtuu[5]*x235;
	double x366 = x325*x365;
	double x367 = -gtuu[1]*x275*x344 - gtuu[2]*x345*x77 - x102*x285 - x107*x278 + x108*x154*x322 - x109*x285 - x111*x279 + x121*x295 - x124*x286 + x125*x275 + x142*x293 + x142*x294 + x154*x315 + x154*x316 + x154*x325*x88 + x159*x310 + x159*x314 + x174*x19 + x181*x287 + x182*x287 + x200*x326 + x200*x327 + x224*x337 + x225*x342 - 1.0/4.0*x269 - 1.0/4.0*x273 - ((x275)*(x275))*x79 - x275*x347 - x279*x280 - x284*x346 - x286*x97 + x322*x328 + x322*x343 - 1.0/4.0*x349 - 1.0/4.0*x351 - 1.0/4.0*x352 - 1.0/4.0*x354 - 1.0/4.0*x355 - 1.0/4.0*x356 + (1.0/4.0)*x357 + (1.0/4.0)*x358 + (1.0/4.0)*x359 + (1.0/4.0)*x360 + (1.0/4.0)*x361 + (1.0/4.0)*x362 + (1.0/4.0)*x364 + (1.0/4.0)*x366 - x75*x78;
	double x368 = x12*x8 + x65*x69;
	double x369 = gtuu[1]*x74;
	double x370 = x117*x152;
	double x371 = x119*x214;
	double x372 = (1.0/8.0)*x180;
	double x373 = (1.0/8.0)*x214;
	double x374 = gtuu[0]*x116;
	double x375 = ddW_dx2[1]*x288 - ddgtdd_dx2[7]*x5 - gtdd[1]*x135*x291 + x167;
	double x376 = x171 + x375;
	double x377 = x172 + x375;
	double x378 = ddgtdd_dx2[8]*x5;
	double x379 = ddW_dx2[1]*x132;
	double x380 = gtdd[2]*x379;
	double x381 = x135*x291;
	double x382 = gtdd[2]*x381;
	double x383 = x201 - x378 + x380 - x382;
	double x384 = x203 + x383;
	double x385 = gtdd[1]*x135;
	double x386 = -ddW_dx2[2]*x288 - x183 + x184 + x185*x385;
	double x387 = x191 + x386;
	double x388 = -x387;
	double x389 = x120*x96;
	double x390 = x101*x110;
	double x391 = -gtdd[4]*x379 + gtdd[4]*x381 - x296 + x297;
	double x392 = x303 + x391;
	double x393 = -x392;
	double x394 = ddgtdd_dx2[25]*x5;
	double x395 = ddW_dx2[4]*x288;
	double x396 = x195*x385;
	double x397 = x311 - x394 + x395 - x396;
	double x398 = x312 + x397;
	double x399 = x127*(dgtdd_dx[16]*x81 + dgtdd_dx[4]*x91);
	double x400 = ddgtdd_dx2[16]*x5;
	double x401 = ddW_dx2[2]*x132;
	double x402 = gtdd[4]*x401;
	double x403 = x135*x185;
	double x404 = gtdd[4]*x403;
	double x405 = ddgtdd_dx2[11]*x5 - gtdd[5]*x379 + gtdd[5]*x381 - x127*(dgtdd_dx[11]*x81 + dgtdd_dx[5]*x86);
	double x406 = x399 - x400 + x402 - x404 + x405;
	double x407 = x127*(dgtdd_dx[14]*x86 + dgtdd_dx[8]*x91);
	double x408 = ddgtdd_dx2[26]*x5;
	double x409 = gtdd[2]*x132;
	double x410 = ddW_dx2[4]*x409;
	double x411 = x135*x195;
	double x412 = gtdd[2]*x411;
	double x413 = x407 - x408 + x410 - x412;
	double x414 = ddgtdd_dx2[31]*x5;
	double x415 = dgtdd_dx[13]*x220;
	double x416 = ddW_dx2[5]*x288;
	double x417 = gtdd[1]*x334;
	double x418 = x414 - x415 - x416 + x417;
	double x419 = -x406 - x413 - x418;
	double x420 = x73*x75;
	double x421 = x180*x348;
	double x422 = x214*x350;
	double x423 = x369*x73;
	double x424 = x150*x254;
	double x425 = gtuu[2]*x74;
	double x426 = x425*x73;
	double x427 = x105*x77;
	double x428 = x77*x79;
	double x429 = gtuu[4]*x74;
	double x430 = x429*x77;
	double x431 = x105*x429;
	double x432 = x152*x353;
	double x433 = x105*x107;
	double x434 = x152*x248;
	double x435 = x180*x251;
	double x436 = gtuu[0]*x150*x95;
	double x437 = gtuu[1]*x241*x95;
	double x438 = x115*x282;
	double x439 = x115*x340;
	double x440 = x151*x230;
	double x441 = x230*x283;
	double x442 = x165*x363;
	double x443 = x165*x365;
	double x444 = 2*gtuu[0]*x108*x110 + 2*gtuu[1]*x122 + 2*gtuu[2]*x389 + 2*gtuu[2]*x390 - 2*x102*x372 - 2*x109*x372 - 1.0/4.0*x111*x152 - 2*x124*x373 + 2*x125*x77 + 2*x142*x376 + 2*x142*x377 - 2*x153*x287 + 2*x154*x210 + 2*x159*x384 + 2*x159*x388 + 2*x166*x88 - 2*x180*x346 + 2*x200*x393 + 2*x200*x398 + 2*x209*x328 + 2*x209*x343 + 2*x224*x419 + 2*x231*x96 - 2*x275*x423 - 2*x275*x428 - 2*x275*x431 - 2*x277*x426 - 2*x277*x430 - 2*x277*x433 - 2*x369*x78 - 1.0/2.0*x370 - 1.0/2.0*x371 - 2*x373*x97 + 2*x374*x88 - 2*x420*x77 - 1.0/2.0*x421 - 1.0/2.0*x422 - 1.0/2.0*x424 - 2*x425*x427 - 1.0/2.0*x432 - 1.0/2.0*x434 - 1.0/2.0*x435 + (1.0/2.0)*x436 + (1.0/2.0)*x437 + (1.0/2.0)*x438 + (1.0/2.0)*x439 + (1.0/2.0)*x440 + (1.0/2.0)*x441 + (1.0/2.0)*x442 + (1.0/2.0)*x443;
	double x445 = (1.0/2.0)*x123;
	double x446 = gtuu[4]*x445;
	double x447 = gtuu[1]*x247 + gtuu[3]*x250 + x446;
	double x448 = x109*x447;
	double x449 = x123*x211;
	double x450 = gtuu[0]*x247 + gtuu[1]*x250 + x449;
	double x451 = x111*x450;
	double x452 = gtdd[5]*x71;
	double x453 = Atdd[5] + x452;
	double x454 = (1.0/8.0)*x450;
	double x455 = gtuu[5]*x445;
	double x456 = gtuu[2]*x247;
	double x457 = gtuu[4]*x250;
	double x458 = x455 + x456 + x457;
	double x459 = (1.0/8.0)*x458;
	double x460 = x121*x175;
	double x461 = (1.0/8.0)*x102;
	double x462 = (1.0/8.0)*x19;
	double x463 = x118*x230;
	double x464 = x405 + x418;
	double x465 = 12*dW_dx[1]*dW_dx[2]*gtdd[2]*x134 - ddW_dx2[4]*x202 + 2*ddgtdd_dx2[26]*x5 - 2*x407 - x464;
	double x466 = 12*dW_dx[0]*dW_dx[2]*gtdd[4]*x134 - ddW_dx2[2]*x298 + 2*ddgtdd_dx2[16]*x5 - 2*x399 - x464;
	double x467 = x230*x96;
	double x468 = x126*x93;
	double x469 = x118*x341;
	double x470 = ddW_dx2[5]*x409 + ddgtdd_dx2[17]*x5 - ddgtdd_dx2[32]*x5 + dgtdd_dx[14]*x220 - gtdd[2]*x334 - gtdd[5]*x401 + gtdd[5]*x403 - x127*(dgtdd_dx[17]*x81 + dgtdd_dx[5]*x91);
	double x471 = -x470;
	double x472 = x123*x154;
	double x473 = x30*x337;
	double x474 = x101*x175;
	double x475 = ddW_dx2[4]*x132;
	double x476 = ddW_dx2[5]*gtdd[4]*x132 + ddgtdd_dx2[29]*x5 - ddgtdd_dx2[34]*x5 + dgtdd_dx[16]*x220 - gtdd[4]*x334 + gtdd[5]*x411 - gtdd[5]*x475 - x127*(dgtdd_dx[11]*x91 + dgtdd_dx[17]*x86);
	double x477 = -x476;
	double x478 = x123*x199;
	double x479 = x145*x150;
	double x480 = x348*x447;
	double x481 = x350*x458;
	double x482 = x239*x447;
	double x483 = x242*x458;
	double x484 = x245*x450;
	double x485 = x353*x450;
	double x486 = gtuu[0]*x163;
	double x487 = x209*x486;
	double x488 = x165*x321;
	double x489 = x164*x322;
	double x490 = x165*x456;
	double x491 = gtuu[2]*x250;
	double x492 = x209*x491;
	double x493 = gtuu[3]*x235;
	double x494 = x325*x493;
	double x495 = gtuu[4]*x247;
	double x496 = x325*x495;
	double x497 = x322*x457;
	double x498 = x64*x69;
	double x499 = (1.0/4.0)*x134;
	double x500 = -x197 - x386;
	double x501 = x145*x4;
	double x502 = x204 + x383;
	double x503 = -2*ddW_dx2[2]*gtdd[2]*x127 + gtdd[2]*x403 - x127*x217 + x218;
	double x504 = -x221 - x503;
	double x505 = x211*x4;
	double x506 = -x222 - x503;
	double x507 = x309 - x311 + x392 + x394 - x395 + x396;
	double x508 = (1.0/2.0)*x30;
	double x509 = gtuu[3]*x319;
	double x510 = x236*x4;
	double x511 = -x414 + x415 + x416 - x417;
	double x512 = x407 - x408 + x410 - x412 - x511;
	double x513 = x109*x209;
	double x514 = x111*x165;
	double x515 = x100*x486;
	double x516 = x115*x321;
	double x517 = x164*x180;
	double x518 = x115*x456;
	double x519 = x100*x491;
	double x520 = x152*x493;
	double x521 = x152*x495;
	double x522 = x180*x457;
	double x523 = x209*x348;
	double x524 = x230*x350;
	double x525 = x209*x239;
	double x526 = x230*x242;
	double x527 = x165*x245;
	double x528 = x165*x353;
	double x529 = 2*x237;
	double x530 = x100*x323 - x112*x165 + x115*x160 + x118*x143*x95 + x145*x215 + x145*x95*x96 + x152*x207 - x162*x230 - x165*x227 + x180*x318 - x209*x281 - x209*x338 + x214*x446 + x214*x509 - x230*x320 - x230*x529 + x406*x510 + x449*x95 + x500*x501 + x501*x502 + x504*x505 + x505*x506 + x507*x508 + x510*x512 - x513 - x514 + x515 + x516 + x517 + x518 + x519 + x520 + x521 + x522 - x523 - x524 - x525 - x526 - x527 - x528;
	double x531 = gtdd[2]*x89;
	double x532 = x143*x4;
	double x533 = -x112*x450 + x143*x463 + x145*x467 + x145*x469 + x160*x165 - x162*x458 + x207*x325 + x209*x323 + x223*x532 - x227*x450 + x230*x449 - x281*x447 + x318*x322 - x320*x458 - x338*x447 + x341*x446 + x341*x509 - x448 - x451 - x458*x529 + x465*x501 + x466*x501 + x470*x505 + x471*x505 + (1.0/2.0)*x473 + x476*x510 + x477*x510 - x480 - x481 - x482 - x483 - x484 - x485 + x487 + x488 + x489 + x490 + x492 + x494 + x496 + x497;
	double x534 = (1.0/4.0)*x533;
	double x535 = (1.0/2.0)*x34;
	double x536 = (1.0/2.0)*gtuu[3]*x182 + (1.0/2.0)*gtuu[5]*x226 - x100*x146 - x100*x281 - x103 - x115*x176 - x115*x227 - 2*x115*x353 + x140*x145 + x141*x501 + x152*x99 + x157*x211 + x158*x505 - x162*x95 + x165*x85 + x173*x508 + x178*x180 + x198*x510 + x205*x510 + x206*x236 + x208*x214 + x209*x213 + x216*x236 + x223*x535 + x228*x230 - x238 - x240 - x243 - x246 - x249 - x252 + x253 + x255 + x257 + x258 + x260 + x261 + x262 + x264 - x455*x95 - x98;
	double x537 = (1.0/4.0)*x49;
	double x538 = x453*x499;
	double x539 = 2*x239;
	double x540 = gtuu[2]*x177*x322 + gtuu[5]*x317*x322 - x112*x268 + x143*x181 + x144*x152 - x146*x284 + x173*x532 - x176*x268 + x180*x267 + x211*x316 + x212*x325 + x214*x324 - x269 + x271*x322 - x272*x320 - x272*x455 - x273 - x284*x338 - x284*x539 + x293*x501 + x294*x501 + x310*x505 + x314*x505 + x326*x510 + x327*x510 + x337*x535 + x339*x341 - x349 - x351 - x352 - x354 - x355 - x356 + x357 + x358 + x359 + x360 + x361 + x362 + x364 + x366;
	double x541 = x100*x101*x211 + x100*x108*x143 + x100*x267 - x112*x152 + x115*x144 - x146*x180 - x152*x176 + x165*x212 + x177*x256 - x180*x338 - x180*x539 + x209*x271 - x214*x320 - x214*x455 + x230*x339 + x263*x317 + x324*x95 - x370 - x371 + x376*x501 + x377*x501 + x384*x505 + x388*x505 + x393*x510 + x398*x510 + x419*x535 - x421 - x422 - x424 - x432 - x434 - x435 + x436 + x437 + x438 + x439 + x440 + x441 + x442 + x443;
	double x542 = 2*x4;
	double x543 = x197 - x201 + x378 - x380 + x382 + x387;
	double x544 = x313 + x397;
	double x545 = -x309 - x391;
	double x546 = x118*x272;
	double x547 = x405 + x413;
	double x548 = x399 - x400 + x402 - x404 - x511;
	double x549 = ddgtdd_dx2[28]*x5;
	double x550 = gtdd[4]*x411;
	double x551 = gtdd[4]*x475 + x331 + x336 - x549 - x550;
	double x552 = 2*ddW_dx2[4]*gtdd[4]*x127 + x127*x330 + x329 - x332 - x333 + x335 - x549 - x550;
	double x553 = x109*x322;
	double x554 = x111*x325;
	double x555 = x180*x486;
	double x556 = x152*x321;
	double x557 = x164*x284;
	double x558 = x152*x456;
	double x559 = x180*x491;
	double x560 = x268*x493;
	double x561 = x268*x495;
	double x562 = x284*x457;
	double x563 = x322*x348;
	double x564 = x341*x350;
	double x565 = x239*x322;
	double x566 = x242*x341;
	double x567 = x245*x325;
	double x568 = x325*x353;
	double x569 = -x112*x325 + x143*x215 + x145*x315 + x145*x546 + x152*x160 - x162*x341 + x180*x323 + x207*x268 + x214*x449 - x227*x325 + x272*x446 + x272*x509 - x281*x322 + x284*x318 - x320*x341 - x322*x338 - x341*x529 + x501*x544 + x501*x545 + x505*x547 + x505*x548 + x510*x551 + x510*x552 + x532*x543 - x553 - x554 + x555 + x556 + x557 + x558 + x559 + x560 + x561 + x562 - x563 - x564 - x565 - x566 - x567 - x568;
	double x570 = gtuu[1]*x541*x542 + gtuu[2]*x530*x542 + gtuu[4]*x542*x569 + x19*x536 + x30*x540 + x34*x533;
	double x571 = (1.0/8.0)*x134*x570;
	double x572 = -gtdd[0]*gtdd[5]*x571 - x106*x499 + x14*x571 - x530*x531 + x534*x6 + x536*x537 + x538*x73;
	double x573 = 1/(x52);
	double x574 = x2*x573;
	double x575 = gtdd[4]*x89;
	double x576 = gtdd[3]*x571;
	double x577 = -gtdd[5]*x576 + x13*x571 + x275*x538 - x278*x499 + x534*x7 + x537*x540 - x569*x575;
	double x578 = x574*x577;
	double x579 = gtdd[1]*x89;
	double x580 = (1.0/4.0)*x6;
	double x581 = -gtdd[0]*x576 + x15*x571 + x275*x499*x73 - x499*x78 + (1.0/4.0)*x536*x7 + x540*x580 - x541*x579;
	double x582 = x573*x581;
	double x583 = x45*x582;
	double x584 = x71*x89;
	double x585 = (1.0/3.0)*dKhat_dx[2] + (2.0/3.0)*dtheta_dx[2];
	double x586 = 2*Atdd[2] + 2*x104;
	double x587 = (1.0/2.0)*x127;
	double x588 = dW_dx[0]*x587;
	double x589 = x453*x89;
	double x590 = (1.0/3.0)*dKhat_dx[0] + (2.0/3.0)*dtheta_dx[0];
	double x591 = 2*Atdd[0] + 2*x72;
	double x592 = dW_dx[2]*x587;
	double x593 = dAtdd_dx[12]*x89 - dAtdd_dx[2]*x89 + dgtdd_dx[12]*x584 - dgtdd_dx[2]*x584 + x100*x345 + x115*x233 - x165*x232 - x209*x344 - x230*x233 - x531*x590 + (1.0/2.0)*x585*x6 + x586*x588 + x589*x95 - x591*x592;
	double x594 = x56*x593;
	double x595 = (1.0/3.0)*dKhat_dx[1] + (2.0/3.0)*dtheta_dx[1];
	double x596 = 2*Atdd[3] + 2*x274;
	double x597 = x275*x89;
	double x598 = 2*Atdd[4] + 2*x276;
	double x599 = dAtdd_dx[10]*x89 - 1.0/2.0*dAtdd_dx[15]*x5 - 1.0/2.0*dW_dx[1]*x127*x598 + dgtdd_dx[10]*x584 - 1.0/2.0*dgtdd_dx[15]*x5*x71 - 1.0/2.0*gtdd[3]*x5*x585 - 1.0/2.0*x105*x268*x5 - 1.0/2.0*x272*x453*x5 - 1.0/2.0*x277*x284*x5 + x322*x597 + x325*x344 + x341*x345 + x575*x595 + x592*x596;
	double x600 = -x599;
	double x601 = dAtdd_dx[4]*x89;
	double x602 = x588*x598;
	double x603 = dgtdd_dx[4]*x584;
	double x604 = x575*x590;
	double x605 = x232*x325;
	double x606 = x322*x344;
	double x607 = x233*x341;
	double x608 = 2*Atdd[1] + 2*x76;
	double x609 = dAtdd_dx[13]*x89 + dgtdd_dx[13]*x584 + x152*x233 + x180*x345 + x214*x589 + x579*x585 - x592*x608;
	double x610 = -x601 + x602 - x603 - x604 - x605 - x606 - x607 + x609;
	double x611 = x368*x56;
	double x612 = dW_dx[1]*x587;
	double x613 = -dAtdd_dx[8]*x89 - dgtdd_dx[8]*x584 - x165*x344 - x209*x597 - x230*x345 - x531*x595 + x586*x612;
	double x614 = x609 + x613;
	double x615 = dAtdd_dx[3]*x89 - dAtdd_dx[7]*x89 + dgtdd_dx[3]*x584 - dgtdd_dx[7]*x584 - x152*x344 - x180*x597 - x214*x345 + x232*x268 + x233*x272 + x284*x344 - x579*x595 - x588*x596 + (1.0/2.0)*x590*x7 + x608*x612;
	double x616 = (1.0/2.0)*x595;
	double x617 = dAtdd_dx[1]*x89 - dAtdd_dx[6]*x89 + dgtdd_dx[1]*x584 - dgtdd_dx[6]*x584 - x100*x597 - x115*x344 + x152*x232 + x180*x344 + x214*x233 - x345*x95 + x579*x590 - x588*x608 + x591*x612 - x6*x616;
	double x618 = -x55*x617;
	double x619 = -x615;
	double x620 = 2*x368;
	double x621 = gtdd[4]*x74;
	double x622 = x277*x499;
	double x623 = gtdd[1]*x571;
	double x624 = gtdd[2]*gtdd[4]*x571 - gtdd[2]*x569*x74 - gtdd[5]*x623 - x105*x622 + x32*x534 - x530*x621 + x537*x541 + x538*x77;
	double x625 = x574*x624;
	double x626 = (1.0/8.0)*x165;
	double x627 = (1.0/8.0)*x230;
	double x628 = gtuu[0]*x118*x120 + gtuu[1]*x389 + gtuu[1]*x390 + gtuu[2]*x120*x123 + x105*x125 - x105*x420 - x106*x425 - x117*x626 - x119*x627 + x126*x206 + x126*x215 + x142*x500 + x142*x502 + x159*x504 + x159*x506 + x175*x315 + x175*x316 + x200*x406 + x200*x512 - x209*x460 - x209*x461 + x214*x478 - x230*x479 - x277*x423 - x277*x428 - x277*x431 - x280*x626 + (1.0/8.0)*x30*x507 - x369*x427 + x374*x93 - x426*x453 - x430*x453 - x433*x453 - 1.0/4.0*x513 - 1.0/4.0*x514 + (1.0/4.0)*x515 + (1.0/4.0)*x516 + (1.0/4.0)*x517 + (1.0/4.0)*x518 + (1.0/4.0)*x519 + (1.0/4.0)*x520 + (1.0/4.0)*x521 + (1.0/4.0)*x522 - 1.0/4.0*x523 - 1.0/4.0*x524 - 1.0/4.0*x525 - 1.0/4.0*x526 - 1.0/4.0*x527 - 1.0/4.0*x528 - x627*x97;
	double x629 = x66*x69;
	double x630 = x59*x629;
	double x631 = (1.0/8.0)*x325;
	double x632 = (1.0/8.0)*x341;
	double x633 = x101*x295 - x105*x275*x369 - x105*x277*x425 - x107*x277*x453 - x117*x631 - x119*x632 + x125*x277 + x126*x315 + x126*x546 + x142*x544 + x142*x545 + x159*x547 + x159*x548 + x175*x272*x96 + x200*x551 + x200*x552 + x206*x287 + x214*x472 + x215*x287 + x268*x468 + x272*x478 - x275*x277*x79 - x275*x429*x453 - x277*x369*x77 - x278*x429 - x280*x631 + x284*x474 - x322*x460 - x322*x461 - x341*x479 - x425*x453*x77 - x427*x75 + x462*x543 - 1.0/4.0*x553 - 1.0/4.0*x554 + (1.0/4.0)*x555 + (1.0/4.0)*x556 + (1.0/4.0)*x557 + (1.0/4.0)*x558 + (1.0/4.0)*x559 + (1.0/4.0)*x560 + (1.0/4.0)*x561 + (1.0/4.0)*x562 - 1.0/4.0*x563 - 1.0/4.0*x564 - 1.0/4.0*x565 - 1.0/4.0*x566 - 1.0/4.0*x567 - 1.0/4.0*x568 - x632*x97;
	double x634 = x61*x629;
	double x635 = x498*x573;
	double x636 = 2*Atdd[5] + 2*x452;
	double x637 = dAtdd_dx[14]*x89 - 1.0/2.0*dAtdd_dx[5]*x5 - 1.0/2.0*dW_dx[2]*x127*x586 + dgtdd_dx[14]*x584 - 1.0/2.0*dgtdd_dx[5]*x5*x71 - 1.0/2.0*gtdd[5]*x5*x590 - 1.0/2.0*x105*x458*x5 + x165*x233 + x209*x345 + x230*x589 - 1.0/2.0*x447*x5*x77 - 1.0/2.0*x450*x5*x73 + x531*x585 + x588*x636;
	double x638 = dAtdd_dx[11]*x89 - dAtdd_dx[16]*x89 + dgtdd_dx[11]*x584 - dgtdd_dx[16]*x584 - x233*x325 - x322*x345 - x341*x589 + x344*x450 + x345*x458 + x447*x597 + x49*x616 - x575*x585 + x592*x598 - x612*x636;
	double x639 = -1.0/8.0*gtdd[1]*gtdd[4]*x134*x570 - 1.0/4.0*gtdd[2]*x5*x540 + gtdd[2]*x576 - 1.0/4.0*gtdd[3]*x5*x530 - 1.0/4.0*x105*x134*x275 + (1.0/4.0)*x32*x569 + x541*x621 + x622*x77;
	double x640 = -x639;
	double x641 = x573*xyz[2];
	double x642 = x44*x641;
	double x643 = x640*x642;
	double x644 = -1.0/8.0*gtdd[0]*gtdd[4]*x134*x570 - 1.0/4.0*gtdd[1]*x5*x530 - 1.0/4.0*gtdd[2]*x5*x541 + gtdd[2]*x623 - 1.0/4.0*x105*x134*x77 + x536*x621 + x569*x580 + x622*x73;
	double x645 = -x644;
	double x646 = x642*x645;
	double x647 = x47*x641;
	double x648 = x639*x647;
	double x649 = x50*x573;
	double x650 = -x581*x649;
	double x651 = x63*x69;
	double x652 = x56*x651;
	double x653 = -x638;
	double x654 = -x593;
	double x655 = x54*x651;
	double x656 = -x610;
	double x657 = x61*x655;
	double x658 = x601 - x602 + x603 + x604 + x605 + x606 + x607 + x613;
	double x659 = -x614;
	double x660 = x55*x651;
	double x661 = x59*x660;
	double x662 = -x658;
	double x663 = x45*x573*x644;
	double x664 = -x572;
	double x665 = -x624;
	double x666 = x642*x665;
	double x667 = -x577*x647;
	double x668 = x639*x649;
	double x669 = (1.0/sqrt(x68));
	double x670 = x46*x669;
	double x671 = x25*x669;
	double x672 = x59*x670 - x61*x671;
	double x673 = 2*x672;
	double x674 = x66*x670;
	double x675 = x61*x670;
	double x676 = 2*x675;
	double x677 = x63*x671;
	double x678 = x55*x677;
	double x679 = 2*x59*x671;
	*Psi4Re = x3*(x265*x70 + x266*x367 - x266*x54*x615 - x266*x56*x600 + x266*x578 + x266*x583 + 2*x266*x643 + x368*x444 - x368*x54*x617 - x368*x55*x619 + x45*x572*x635 + x48*x573*x630*x640 + x48*x577*x635 + x48*x582*x70 + x498*x54*x637 - x498*x55*x638 + x498*(-gtuu[1]*x233*x277 + x101*x126*x209 - x106*x75 - x107*((x453)*(x453)) - x117*x454 - x119*x459 + x125*x453 + x126*x467 + x126*x469 + x142*x465 + x142*x466 + x159*x470 + x159*x471 + x175*x342 + x200*x476 + x200*x477 + x223*x462 + x226*x287 + x230*x472 - x234*x453 - x278*x79 - x280*x454 + x287*x463 + x322*x474 + x325*x468 + x341*x478 - x347*x453 - x447*x460 - x447*x461 - 1.0/4.0*x448 - 1.0/4.0*x451 - x458*x479 - x459*x97 + (1.0/8.0)*x473 - 1.0/4.0*x480 - 1.0/4.0*x481 - 1.0/4.0*x482 - 1.0/4.0*x483 - 1.0/4.0*x484 - 1.0/4.0*x485 + (1.0/4.0)*x487 + (1.0/4.0)*x488 + (1.0/4.0)*x489 + (1.0/4.0)*x490 + (1.0/4.0)*x492 + (1.0/4.0)*x494 + (1.0/4.0)*x496 + (1.0/4.0)*x497) + 2*x50*x624*x635 + x572*x574*x70 - x59*x637*x652 - x59*x654*x655 - x594*x70 - x599*x61*x660 - x61*x652*x653 - x610*x611 - x611*x614 - x618*x70 + x620*x625 + x620*x646 + x620*x648 + x620*x650 + x628*x630 + x630*x642*x664 + x630*x645*x649 + x630*x647*x665 + x633*x634 + x634*x663 + x634*x666 + x634*x667 + x634*x668 + 2*x644*x647*x70 - x656*x657 - x657*x658 - x659*x661 - x661*x662);
	*Psi4Im = x3*(2*x2*x24*x572*x573*x59*x669*xyz[1] + 2*x24*x265*x59*x669*xyz[1] + 2*x24*x44*x47*x573*x63*x645*x669*xyz[1] + 2*x24*x44*x53*x61*x615*x669*xyz[0] + x24*x44*x53*x63*x656*x669*xyz[0] + x24*x44*x53*x63*x658*x669*xyz[0] + 2*x24*x44*x573*x63*x664*x669*xyz[1]*xyz[2] + x24*x47*x53*x599*x63*x669*xyz[0] + 4*x24*x47*x573*x59*x644*x669*xyz[1]*xyz[2] + 2*x24*x47*x573*x63*x665*x669*xyz[1]*xyz[2] + 2*x24*x48*x573*x581*x59*x669*xyz[1] + 2*x24*x48*x573*x63*x640*x669*xyz[1] + 2*x24*x53*x600*x61*x669*xyz[0]*xyz[2] + x24*x53*x63*x653*x669*xyz[0]*xyz[2] + 2*x24*x628*x63*x669*xyz[1] - x367*x676 + x44*x53*x617*x672 - x444*x672 + x47*x53*x619*x672 + x53*x610*x672*xyz[2] + x53*x614*x672*xyz[2] - x54*x654*x677 - x56*x637*x677 - x578*x676 - x583*x676 - x594*x679 - x618*x679 - x625*x673 - x633*x674 - 4*x643*x675 - x646*x673 - x648*x673 - x650*x673 - x659*x678 - x662*x678 - x663*x674 - x666*x674 - x667*x674 - x668*x674);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_adaptive_eta(
	double W,
	double eta,
	const double gtuu[6],
	const double dW_dx[3],
	double apar,
	double bpar,
	double epstiny,
	double * __restrict__ eta_adaptive
)
{
	double x0 = 2*dW_dx[0];
	*eta_adaptive = (1.0/2.0)*eta*sqrt(((dW_dx[0])*(dW_dx[0]))*gtuu[0] + ((dW_dx[1])*(dW_dx[1]))*gtuu[3] + 2*dW_dx[1]*dW_dx[2]*gtuu[4] + dW_dx[1]*gtuu[1]*x0 + ((dW_dx[2])*(dW_dx[2]))*gtuu[5] + dW_dx[2]*gtuu[2]*x0)/(epstiny + pow(1 - pow(W, apar), bpar));
}

#endif 
