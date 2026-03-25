
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
	double dW_dx_upwind,
	double * __restrict__ dW
)
{
	double x0 = (1.0/3.0)*W;
	*dW = (1.0/3.0)*W*alp*(Khat + 2*theta) + dW_dx_upwind - dbetau_dx[0]*x0 - dbetau_dx[4]*x0 - dbetau_dx[8]*x0;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_gtdd_rhs(
	const double gtdd[6],
	const double Atdd[6],
	double alp,
	const double dgtdd_dx_upwind[6],
	const double dbetau_dx[9],
	double (*dgtdd_dt)[6]
)
{
	double x0 = 2*alp;
	double x1 = 2*gtdd[1];
	double x2 = 2*gtdd[2];
	double x3 = (2.0/3.0)*gtdd[0];
	double x4 = (1.0/3.0)*gtdd[1];
	double x5 = (2.0/3.0)*dbetau_dx[8];
	double x6 = (1.0/3.0)*gtdd[2];
	double x7 = (2.0/3.0)*dbetau_dx[4];
	double x8 = (2.0/3.0)*dbetau_dx[0];
	double x9 = 2*gtdd[4];
	double x10 = (1.0/3.0)*gtdd[4];
	(*dgtdd_dt)[0] = -Atdd[0]*x0 + (4.0/3.0)*dbetau_dx[0]*gtdd[0] + dbetau_dx[1]*x1 + dbetau_dx[2]*x2 - dbetau_dx[4]*x3 - dbetau_dx[8]*x3 + dgtdd_dx_upwind[0];
	(*dgtdd_dt)[1] = -Atdd[1]*x0 + dbetau_dx[0]*x4 + dbetau_dx[1]*gtdd[3] + dbetau_dx[2]*gtdd[4] + dbetau_dx[3]*gtdd[0] + dbetau_dx[4]*x4 + dbetau_dx[5]*gtdd[2] + dgtdd_dx_upwind[1] - gtdd[1]*x5;
	(*dgtdd_dt)[2] = -Atdd[2]*x0 + dbetau_dx[0]*x6 + dbetau_dx[1]*gtdd[4] + dbetau_dx[2]*gtdd[5] + dbetau_dx[6]*gtdd[0] + dbetau_dx[7]*gtdd[1] + dbetau_dx[8]*x6 + dgtdd_dx_upwind[2] - gtdd[2]*x7;
	(*dgtdd_dt)[3] = -Atdd[3]*x0 + dbetau_dx[3]*x1 + (4.0/3.0)*dbetau_dx[4]*gtdd[3] + dbetau_dx[5]*x9 + dgtdd_dx_upwind[3] - gtdd[3]*x5 - gtdd[3]*x8;
	(*dgtdd_dt)[4] = -Atdd[4]*x0 + dbetau_dx[3]*gtdd[2] + dbetau_dx[4]*x10 + dbetau_dx[5]*gtdd[5] + dbetau_dx[6]*gtdd[1] + dbetau_dx[7]*gtdd[3] + dbetau_dx[8]*x10 + dgtdd_dx_upwind[4] - gtdd[4]*x8;
	(*dgtdd_dt)[5] = -Atdd[5]*x0 + dbetau_dx[6]*x2 + dbetau_dx[7]*x9 + (4.0/3.0)*dbetau_dx[8]*gtdd[5] + dgtdd_dx_upwind[5] - gtdd[5]*x7 - gtdd[5]*x8;
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
	double dKhat_dx_upwind,
	double * __restrict__ dKhat_dt
)
{
	*dKhat_dt = -DiDialp - alp*kappa1*theta*(kappa2 - 1) + (1.0/3.0)*alp*(3*Asqr + ((Ktr)*(Ktr))) + 4*M_PI*alp*(S + rho) + dKhat_dx_upwind;
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
	const double dGammatu_dx_upwind[3],
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
	(*dGammatu_dt)[0] = -Atuu[0]*x0 - Atuu[0]*x17 - Atuu[1]*x1 - Atuu[1]*x18 - Atuu[2]*x19 - Atuu[2]*x2 - 1.0/3.0*GammatDu[0]*dbetau_dx[0] - GammatDu[1]*dbetau_dx[3] - GammatDu[2]*dbetau_dx[6] + Gammatudd[0]*x5 + Gammatudd[1]*x7 + Gammatudd[2]*x8 + Gammatudd[3]*x9 + Gammatudd[4]*x10 + Gammatudd[5]*x11 + dGammatu_dx_upwind[0] + dbetau_dx[4]*x3 + dbetau_dx[8]*x3 + (4.0/3.0)*ddbetau_dx2[0]*gtuu[0] + (1.0/3.0)*ddbetau_dx2[10]*gtuu[1] + 2*ddbetau_dx2[12]*gtuu[4] + (1.0/3.0)*ddbetau_dx2[13]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[14]*gtuu[1] + ddbetau_dx2[15]*gtuu[5] + (1.0/3.0)*ddbetau_dx2[17]*gtuu[2] + (7.0/3.0)*ddbetau_dx2[3]*gtuu[1] + (1.0/3.0)*ddbetau_dx2[4]*gtuu[0] + (7.0/3.0)*ddbetau_dx2[6]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[8]*gtuu[0] + ddbetau_dx2[9]*gtuu[3] - gtuu[0]*x13 - gtuu[0]*x21 - gtuu[1]*x14 - gtuu[1]*x22 - gtuu[2]*x15 - gtuu[2]*x23 + x24*(GammatDu[0] - Gammatu[0]);
	(*dGammatu_dt)[1] = -Atuu[1]*x0 - Atuu[1]*x17 - Atuu[3]*x1 - Atuu[3]*x18 - Atuu[4]*x19 - Atuu[4]*x2 - GammatDu[0]*dbetau_dx[1] - 1.0/3.0*GammatDu[1]*dbetau_dx[4] - GammatDu[2]*dbetau_dx[7] + Gammatudd[10]*x10 + Gammatudd[11]*x11 + Gammatudd[6]*x5 + Gammatudd[7]*x7 + Gammatudd[8]*x8 + Gammatudd[9]*x9 + dGammatu_dx_upwind[1] + dbetau_dx[0]*x25 + dbetau_dx[8]*x25 + (1.0/3.0)*ddbetau_dx2[0]*gtuu[1] + (4.0/3.0)*ddbetau_dx2[10]*gtuu[3] + (7.0/3.0)*ddbetau_dx2[13]*gtuu[4] + (1.0/3.0)*ddbetau_dx2[14]*gtuu[3] + ddbetau_dx2[16]*gtuu[5] + (1.0/3.0)*ddbetau_dx2[17]*gtuu[4] + ddbetau_dx2[1]*gtuu[0] + (1.0/3.0)*ddbetau_dx2[3]*gtuu[3] + (7.0/3.0)*ddbetau_dx2[4]*gtuu[1] + (1.0/3.0)*ddbetau_dx2[6]*gtuu[4] + 2*ddbetau_dx2[7]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[8]*gtuu[1] - gtuu[1]*x13 - gtuu[1]*x21 - gtuu[3]*x14 - gtuu[3]*x22 - gtuu[4]*x15 - gtuu[4]*x23 + x24*(GammatDu[1] - Gammatu[1]);
	(*dGammatu_dt)[2] = -Atuu[2]*x0 - Atuu[2]*x17 - Atuu[4]*x1 - Atuu[4]*x18 - Atuu[5]*x19 - Atuu[5]*x2 - GammatDu[0]*dbetau_dx[2] - GammatDu[1]*dbetau_dx[5] - 1.0/3.0*GammatDu[2]*dbetau_dx[8] + Gammatudd[12]*x5 + Gammatudd[13]*x7 + Gammatudd[14]*x8 + Gammatudd[15]*x9 + Gammatudd[16]*x10 + Gammatudd[17]*x11 + dGammatu_dx_upwind[2] + dbetau_dx[0]*x26 + dbetau_dx[4]*x26 + (1.0/3.0)*ddbetau_dx2[0]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[10]*gtuu[4] + ddbetau_dx2[11]*gtuu[3] + (1.0/3.0)*ddbetau_dx2[13]*gtuu[5] + (7.0/3.0)*ddbetau_dx2[14]*gtuu[4] + (4.0/3.0)*ddbetau_dx2[17]*gtuu[5] + ddbetau_dx2[2]*gtuu[0] + (1.0/3.0)*ddbetau_dx2[3]*gtuu[4] + (1.0/3.0)*ddbetau_dx2[4]*gtuu[2] + 2*ddbetau_dx2[5]*gtuu[1] + (1.0/3.0)*ddbetau_dx2[6]*gtuu[5] + (7.0/3.0)*ddbetau_dx2[8]*gtuu[2] - gtuu[2]*x13 - gtuu[2]*x21 - gtuu[4]*x14 - gtuu[4]*x22 - gtuu[5]*x15 - gtuu[5]*x23 + x24*(GammatDu[2] - Gammatu[2]);
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
	double dtheta_dx_upwind,
	double * __restrict__ dtheta_dt
)
{
	*dtheta_dt = -1.0/6.0*alp*theta_damp_fact*(3*Asqr - 3*Rtrace + 6*kappa1*theta*(kappa2 + 2) + 48*M_PI*rho - 2*((Khat + 2*theta)*(Khat + 2*theta))) + dtheta_dx_upwind;
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
	const double dAtdd_dx_upwind[6],
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
	(*dAtdd_dt)[0] = Atdd[0]*Ktr*alp + (4.0/3.0)*Atdd[0]*dbetau_dx[0] - 2*Atdd[0]*x6 + 2*Atdd[1]*dbetau_dx[1] + 2*Atdd[2]*dbetau_dx[2] - W2DiDjalp[0] - alp*(Sij[0]*x2 - W2Rdd[0]) + dAtdd_dx_upwind[0] - dbetau_dx[4]*x0 - dbetau_dx[8]*x0 + (1.0/3.0)*gtdd[0]*x3 - x10*x9 - x7*x8;
	(*dAtdd_dt)[1] = Atdd[0]*dbetau_dx[3] - Atdd[1]*x12 + Atdd[1]*x13 + Atdd[2]*dbetau_dx[5] + Atdd[3]*dbetau_dx[1] + Atdd[4]*dbetau_dx[2] - W2DiDjalp[1] - alp*(Sij[1]*x2 - W2Rdd[1]) + dAtdd_dx_upwind[1] + dbetau_dx[0]*x11 + dbetau_dx[4]*x11 + gtdd[1]*x14 - x10*x16 - x15*x8 - x6*x7;
	(*dAtdd_dt)[2] = Atdd[0]*dbetau_dx[6] + Atdd[1]*dbetau_dx[7] + Atdd[2]*x13 - Atdd[2]*x18 + Atdd[4]*dbetau_dx[1] + Atdd[5]*dbetau_dx[2] - W2DiDjalp[2] - alp*(Sij[2]*x2 - W2Rdd[2]) + dAtdd_dx_upwind[2] + dbetau_dx[0]*x17 + dbetau_dx[8]*x17 + gtdd[2]*x14 - x10*x19 - x16*x8 - x6*x9;
	(*dAtdd_dt)[3] = 2*Atdd[1]*dbetau_dx[3] + Atdd[3]*Ktr*alp + (4.0/3.0)*Atdd[3]*dbetau_dx[4] - Atdd[3]*x12 - Atdd[3]*x20 + 2*Atdd[4]*dbetau_dx[5] - W2DiDjalp[3] - alp*(Sij[3]*x2 - W2Rdd[3]) + dAtdd_dx_upwind[3] + (1.0/3.0)*gtdd[3]*x3 - x15*x23 - x16*x24 - x21*x7;
	(*dAtdd_dt)[4] = Atdd[1]*dbetau_dx[6] + Atdd[2]*dbetau_dx[3] + Atdd[3]*dbetau_dx[7] + Atdd[4]*x13 - Atdd[4]*x20 + Atdd[5]*dbetau_dx[5] - W2DiDjalp[4] - alp*(Sij[4]*x2 - W2Rdd[4]) + dAtdd_dx_upwind[4] + dbetau_dx[4]*x25 + dbetau_dx[8]*x25 + gtdd[4]*x14 - x16*x23 - x19*x24 - x21*x9;
	(*dAtdd_dt)[5] = 2*Atdd[2]*dbetau_dx[6] + 2*Atdd[4]*dbetau_dx[7] + Atdd[5]*Ktr*alp + (4.0/3.0)*Atdd[5]*dbetau_dx[8] - Atdd[5]*x18 - Atdd[5]*x20 - W2DiDjalp[5] - alp*x16*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4]) - alp*x19*(Atdd[5]*gtuu[5] + x22 + x5) - alp*x9*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2]) - alp*(Sij[5]*x2 - W2Rdd[5]) + dAtdd_dx_upwind[5] + (1.0/3.0)*gtdd[5]*x3;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_alpha_rhs(
	double alp,
	double Khat,
	double dalp_dx_upwind,
	double * __restrict__ dalpha_dt
)
{
	*dalpha_dt = -2*Khat*alp + dalp_dx_upwind;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_beta_rhs(
	const double Bdriver[3],
	const double dbetau_dx_upwind[3],
	double (*dbeta_dt)[3]
)
{
	(*dbeta_dt)[0] = (3.0/4.0)*Bdriver[0] + dbetau_dx_upwind[0];
	(*dbeta_dt)[1] = (3.0/4.0)*Bdriver[1] + dbetau_dx_upwind[1];
	(*dbeta_dt)[2] = (3.0/4.0)*Bdriver[2] + dbetau_dx_upwind[2];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Bdriver_rhs(
	const double Bdriver[3],
	double eta,
	const double dGammatu_dt[3],
	const double dBdriver_dx_upwind[3],
	const double dGammatu_dx_upwind[3],
	double (*dBd_dt)[3]
)
{
	(*dBd_dt)[0] = -Bdriver[0]*eta + dBdriver_dx_upwind[0] + dGammatu_dt[0] - dGammatu_dx_upwind[0];
	(*dBd_dt)[1] = -Bdriver[1]*eta + dBdriver_dx_upwind[1] + dGammatu_dt[1] - dGammatu_dx_upwind[1];
	(*dBd_dt)[2] = -Bdriver[2]*eta + dBdriver_dx_upwind[2] + dGammatu_dt[2] - dGammatu_dx_upwind[2];
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
	double x0 = ((xyz[2])*(xyz[2]));
	double x1 = ((xyz[0])*(xyz[0]));
	double x2 = ((xyz[1])*(xyz[1]));
	double x3 = x1 + x2;
	double x4 = sqrt(x0 + x3);
	double x5 = ((W)*(W));
	double x6 = 1/(x5);
	double x7 = gtdd[0]*x6;
	double x8 = gtdd[3]*x6;
	double x9 = xyz[0]*xyz[1];
	double x10 = gtdd[1]*x6;
	double x11 = 2*x10;
	double x12 = x1*x8 - x11*x9 + x2*x7;
	double x13 = 1/(x12);
	double x14 = xyz[0]*xyz[2];
	double x15 = (1.0/sqrt(x12));
	double x16 = x10*x15*x2;
	double x17 = x15*xyz[1];
	double x18 = -x3;
	double x19 = x18*x6;
	double x20 = gtdd[2]*x19;
	double x21 = gtdd[1]*x1*x15*x6*xyz[2] + gtdd[3]*x15*x6*xyz[0]*xyz[1]*xyz[2] + gtdd[4]*x15*x18*x6*xyz[0] - x14*x17*x7 - x16*xyz[2] - x17*x20;
	double x22 = xyz[1]*xyz[2];
	double x23 = gtdd[1]*x1*x15*x6 - gtdd[2]*x15*x22*x6 + gtdd[3]*x15*x6*xyz[0]*xyz[1] + gtdd[4]*x15*x6*xyz[0]*xyz[2] - x16 - x17*x7*xyz[0];
	double x24 = x17*x23 + xyz[0];
	double x25 = ((x24)*(x24));
	double x26 = x15*xyz[0];
	double x27 = -x23*x26 + xyz[1];
	double x28 = ((x27)*(x27));
	double x29 = gtdd[5]*x6;
	double x30 = x24*x27;
	double x31 = gtdd[2]*x6;
	double x32 = 2*xyz[2];
	double x33 = x24*x32;
	double x34 = gtdd[4]*x6;
	double x35 = x27*x32;
	double x36 = x0*x29 + x11*x30 + x25*x7 + x28*x8 + x31*x33 + x34*x35;
	double x37 = (1.0/sqrt(x36));
	double x38 = x0*x37;
	double x39 = x24*x37;
	double x40 = x27*x37;
	double x41 = x37*xyz[2];
	double x42 = x37*(gtdd[4]*x19*x40 + x10*x14*x40 + x10*x22*x39 + x14*x39*x7 + x18*x29*x41 + x20*x39 + x22*x40*x8 + x31*x38*xyz[0] + x34*x38*xyz[1]);
	double x43 = x14 + x17*x21 - x24*x42;
	double x44 = ((x43)*(x43));
	double x45 = -x21*x26 - x27*x42 + xyz[1]*xyz[2];
	double x46 = ((x45)*(x45));
	double x47 = -x3 - x42*xyz[2];
	double x48 = ((x47)*(x47));
	double x49 = x43*x45;
	double x50 = 2*x47;
	double x51 = x43*x50;
	double x52 = x45*x50;
	double x53 = x11*x49 + x29*x48 + x31*x51 + x34*x52 + x44*x7 + x46*x8;
	double x54 = 1/(x53);
	double x55 = -x13*x2 + x44*x54;
	double x56 = (1.0/3.0)*Khat + (2.0/3.0)*theta;
	double x57 = gtdd[0]*x56;
	double x58 = Atdd[0] + x57;
	double x59 = (1.0/4.0)*x6;
	double x60 = gtuu[0]*x59;
	double x61 = gtdd[1]*x56;
	double x62 = Atdd[1] + x61;
	double x63 = ((x62)*(x62));
	double x64 = gtuu[3]*x59;
	double x65 = 1/(W);
	double x66 = 2*dW_dx[0];
	double x67 = x65*x66;
	double x68 = dgtdd_dx[0] - gtdd[0]*x67;
	double x69 = (1.0/2.0)*x68;
	double x70 = gtuu[2]*x69;
	double x71 = 2*dW_dx[1];
	double x72 = x65*x71;
	double x73 = dgtdd_dx[6] - gtdd[0]*x72;
	double x74 = (1.0/2.0)*x6;
	double x75 = x5*(x6*(dgtdd_dx[1] - gtdd[1]*x67) - x73*x74);
	double x76 = 2*dW_dx[2];
	double x77 = x65*x76;
	double x78 = dgtdd_dx[12] - gtdd[0]*x77;
	double x79 = x5*(x6*(dgtdd_dx[2] - gtdd[2]*x67) - x74*x78);
	double x80 = gtuu[4]*x75 + gtuu[5]*x79 + x70;
	double x81 = dgtdd_dx[11] - gtdd[5]*x72;
	double x82 = gtuu[4]*x81;
	double x83 = x80*x82;
	double x84 = gtuu[1]*x69;
	double x85 = gtuu[3]*x75 + gtuu[4]*x79 + x84;
	double x86 = dgtdd_dx[15] - gtdd[3]*x77;
	double x87 = gtuu[4]*x86;
	double x88 = x85*x87;
	double x89 = gtdd[2]*x56;
	double x90 = Atdd[2] + x89;
	double x91 = ((x90)*(x90));
	double x92 = gtuu[5]*x59;
	double x93 = dgtdd_dx[3] - gtdd[3]*x67;
	double x94 = gtuu[1]*x93;
	double x95 = (1.0/8.0)*x85;
	double x96 = gtuu[1]*x73;
	double x97 = gtuu[0]*x69;
	double x98 = gtuu[1]*x75;
	double x99 = gtuu[2]*x79;
	double x100 = x97 + x98 + x99;
	double x101 = (1.0/8.0)*x100;
	double x102 = gtuu[2]*x78;
	double x103 = dgtdd_dx[5] - gtdd[5]*x67;
	double x104 = gtuu[2]*x103;
	double x105 = (1.0/8.0)*x80;
	double x106 = dgtdd_dx[9] - gtdd[3]*x72;
	double x107 = x106*x95;
	double x108 = dgtdd_dx[17] - gtdd[5]*x77;
	double x109 = gtuu[5]*x108;
	double x110 = x59*(Khat + 2*theta);
	double x111 = (1.0/8.0)*gtuu[1];
	double x112 = 1/(((W)*(W)*(W)));
	double x113 = 2*dgtdd_dx[0];
	double x114 = 4*x112;
	double x115 = dW_dx[0]*x114;
	double x116 = 2*x112;
	double x117 = ddW_dx2[0]*x116;
	double x118 = gtdd[0]*x116;
	double x119 = 1/(((W)*(W)*(W)*(W)));
	double x120 = 6*x119;
	double x121 = ((dW_dx[0])*(dW_dx[0]))*x120;
	double x122 = gtdd[0]*x120;
	double x123 = dW_dx[0]*x122;
	double x124 = -dW_dx[1]*x123 + ddW_dx2[1]*x118 + ddgtdd_dx2[1]*x6 - ddgtdd_dx2[6]*x6 - dgtdd_dx[1]*x115 - gtdd[1]*x117 + gtdd[1]*x121 + x112*(dW_dx[1]*x113 + dgtdd_dx[6]*x66);
	double x125 = x124*x5;
	double x126 = -x124;
	double x127 = x111*x5;
	double x128 = (1.0/2.0)*gtuu[0];
	double x129 = x128*x73;
	double x130 = (1.0/2.0)*gtuu[1];
	double x131 = x130*x93;
	double x132 = x74*(dgtdd_dx[4] - gtdd[4]*x67);
	double x133 = x74*(dgtdd_dx[8] - gtdd[2]*x72);
	double x134 = x74*(dgtdd_dx[13] - gtdd[1]*x77);
	double x135 = x5*(x132 + x133 - x134);
	double x136 = gtuu[2]*x135;
	double x137 = x129 + x131 + x136;
	double x138 = x137*x68;
	double x139 = (1.0/8.0)*gtuu[2];
	double x140 = dW_dx[2]*x123 - ddW_dx2[2]*x118 + ddgtdd_dx2[12]*x6 - ddgtdd_dx2[2]*x6 + dgtdd_dx[2]*x115 + gtdd[2]*x117 - gtdd[2]*x121 - x112*(dW_dx[2]*x113 + dgtdd_dx[12]*x66);
	double x141 = x140*x5;
	double x142 = -x140;
	double x143 = x139*x5;
	double x144 = x128*x78;
	double x145 = (1.0/2.0)*x103;
	double x146 = gtuu[2]*x145;
	double x147 = x5*(x132 - x133 + x134);
	double x148 = gtuu[1]*x147;
	double x149 = x144 + x146 + x148;
	double x150 = x139*x149;
	double x151 = gtuu[3]*x5;
	double x152 = x112*(dgtdd_dx[1]*x71 + dgtdd_dx[7]*x66);
	double x153 = gtdd[1]*x114;
	double x154 = dW_dx[1]*x114;
	double x155 = ((dW_dx[1])*(dW_dx[1]));
	double x156 = -ddW_dx2[3]*x118 + ddgtdd_dx2[18]*x6 - dgtdd_dx[6]*x154 + x122*x155;
	double x157 = ddgtdd_dx2[3]*x6 - dgtdd_dx[3]*x115 - gtdd[3]*x117 + gtdd[3]*x121;
	double x158 = 12*dW_dx[0]*dW_dx[1]*gtdd[1]*x119 - ddW_dx2[1]*x153 + 2*ddgtdd_dx2[7]*x6 - 2*x152 - x156 - x157;
	double x159 = (1.0/8.0)*x158;
	double x160 = (1.0/8.0)*gtuu[3];
	double x161 = x130*x73;
	double x162 = (1.0/2.0)*x93;
	double x163 = gtuu[3]*x162;
	double x164 = gtuu[4]*x135;
	double x165 = x161 + x163 + x164;
	double x166 = x165*x93;
	double x167 = x137*x73;
	double x168 = x112*(dgtdd_dx[13]*x66 + dgtdd_dx[1]*x76);
	double x169 = ddgtdd_dx2[13]*x6;
	double x170 = dW_dx[0]*dW_dx[2];
	double x171 = 12*x119;
	double x172 = ddgtdd_dx2[4]*x6;
	double x173 = gtdd[4]*x121;
	double x174 = gtdd[4]*x117;
	double x175 = dgtdd_dx[4]*x115;
	double x176 = -x172 - x173 + x174 + x175;
	double x177 = x112*(dgtdd_dx[12]*x71 + dgtdd_dx[6]*x76);
	double x178 = ddgtdd_dx2[24]*x6;
	double x179 = ddW_dx2[4]*x118;
	double x180 = dW_dx[1]*dW_dx[2];
	double x181 = x122*x180;
	double x182 = x177 - x178 + x179 - x181;
	double x183 = -ddW_dx2[2]*x153 + gtdd[1]*x170*x171 - 2*x168 + 2*x169 + x176 + x182;
	double x184 = (1.0/8.0)*gtuu[4];
	double x185 = x184*x5;
	double x186 = x112*(dgtdd_dx[2]*x71 + dgtdd_dx[8]*x66);
	double x187 = gtdd[2]*x114;
	double x188 = -x177 + x178 - x179 + x181;
	double x189 = x172 + x173 - x174 - x175;
	double x190 = 12*dW_dx[0]*dW_dx[1]*gtdd[2]*x119 - ddW_dx2[1]*x187 + 2*ddgtdd_dx2[8]*x6 - 2*x186 - x188 - x189;
	double x191 = x137*x78;
	double x192 = x130*x78;
	double x193 = gtuu[4]*x145;
	double x194 = gtuu[3]*x147 + x192 + x193;
	double x195 = x194*x93;
	double x196 = (1.0/2.0)*gtuu[2];
	double x197 = x196*x73;
	double x198 = gtuu[4]*x162;
	double x199 = gtuu[5]*x135 + x197 + x198;
	double x200 = x103*x199;
	double x201 = x149*x73;
	double x202 = dgtdd_dx[14]*x66 + dgtdd_dx[2]*x76;
	double x203 = ddgtdd_dx2[14]*x6;
	double x204 = ((dW_dx[2])*(dW_dx[2]));
	double x205 = dW_dx[2]*x114;
	double x206 = ddW_dx2[5]*x118 - ddgtdd_dx2[30]*x6 + dgtdd_dx[12]*x205 - x122*x204;
	double x207 = -ddgtdd_dx2[5]*x6 + dgtdd_dx[5]*x115 + gtdd[5]*x117 - gtdd[5]*x121;
	double x208 = -ddW_dx2[2]*x187 + gtdd[2]*x170*x171 - 2*x112*x202 + 2*x203 + x206 + x207;
	double x209 = gtuu[5]*x5;
	double x210 = (1.0/8.0)*x209;
	double x211 = (1.0/8.0)*gtuu[5];
	double x212 = x149*x78;
	double x213 = x196*x78;
	double x214 = gtuu[5]*x145;
	double x215 = gtuu[4]*x147;
	double x216 = x213 + x214 + x215;
	double x217 = x211*x216;
	double x218 = x58*x74;
	double x219 = x74*x90;
	double x220 = gtuu[2]*x219;
	double x221 = x5*(-x132 + x133 + x134);
	double x222 = (1.0/2.0)*gtuu[4];
	double x223 = gtuu[1]*x135;
	double x224 = x223*x80;
	double x225 = gtuu[2]*x147;
	double x226 = x225*x85;
	double x227 = x5*(x6*(dgtdd_dx[10] - gtdd[4]*x72) - x74*x86);
	double x228 = gtuu[3]*x227;
	double x229 = x228*x80;
	double x230 = x5*(x6*(dgtdd_dx[7] - gtdd[1]*x72) - x74*x93);
	double x231 = gtuu[3]*x230;
	double x232 = x100*x231;
	double x233 = x5*(-x103*x74 + x6*(dgtdd_dx[14] - gtdd[2]*x77));
	double x234 = gtuu[5]*x233;
	double x235 = x100*x234;
	double x236 = x5*(x6*(dgtdd_dx[16] - gtdd[4]*x77) - x74*x81);
	double x237 = gtuu[5]*x236;
	double x238 = x237*x85;
	double x239 = x165*x98;
	double x240 = gtuu[1]*x199;
	double x241 = x240*x79;
	double x242 = gtuu[2]*x194;
	double x243 = x242*x75;
	double x244 = x216*x99;
	double x245 = x135*x199;
	double x246 = gtuu[3]*x245;
	double x247 = x165*x215;
	double x248 = x164*x216;
	double x249 = gtuu[5]*x194;
	double x250 = x147*x249;
	double x251 = -gtuu[1]*x218*x62 - gtuu[3]*x107 - gtuu[4]*x219*x62 - x100*x221*x222 - x101*x102 - x101*x96 + x103*x217 - x104*x105 - x105*x109 + x110*x58 + x111*x125 + x111*x138 + x126*x127 + x139*x141 + x142*x143 + x150*x68 + x151*x159 + x160*x166 + x160*x167 + x183*x185 + x184*x191 + x184*x195 + x184*x200 + x184*x201 + x185*x190 + x208*x210 + x211*x212 - x220*x58 - 1.0/4.0*x224 - 1.0/4.0*x226 - 1.0/4.0*x229 - 1.0/4.0*x232 - 1.0/4.0*x235 - 1.0/4.0*x238 + (1.0/4.0)*x239 + (1.0/4.0)*x241 + (1.0/4.0)*x243 + (1.0/4.0)*x244 + (1.0/4.0)*x246 + (1.0/4.0)*x247 + (1.0/4.0)*x248 + (1.0/4.0)*x250 - ((x58)*(x58))*x60 - x63*x64 - 1.0/4.0*x83 - 1.0/4.0*x88 - x91*x92 - x94*x95;
	double x252 = -x1*x13 + x46*x54;
	double x253 = x106*x130;
	double x254 = gtuu[0]*x230 + gtuu[2]*x227 + x253;
	double x255 = x102*x254;
	double x256 = (1.0/2.0)*x106;
	double x257 = gtuu[4]*x256;
	double x258 = gtuu[2]*x230 + gtuu[5]*x227 + x257;
	double x259 = x104*x258;
	double x260 = gtdd[3]*x56;
	double x261 = Atdd[3] + x260;
	double x262 = gtdd[4]*x56;
	double x263 = Atdd[4] + x262;
	double x264 = ((x263)*(x263));
	double x265 = (1.0/8.0)*x254;
	double x266 = gtuu[0]*x68;
	double x267 = gtuu[3]*x256;
	double x268 = gtuu[1]*x230;
	double x269 = gtuu[4]*x227;
	double x270 = x267 + x268 + x269;
	double x271 = (1.0/8.0)*x270;
	double x272 = (1.0/8.0)*x258;
	double x273 = gtuu[0]*x5;
	double x274 = (1.0/8.0)*gtuu[0];
	double x275 = ddW_dx2[3]*x116;
	double x276 = gtdd[3]*x116;
	double x277 = x120*x155;
	double x278 = dW_dx[0]*dW_dx[1];
	double x279 = gtdd[3]*x120;
	double x280 = ddW_dx2[1]*x276 + ddgtdd_dx2[19]*x6 - ddgtdd_dx2[9]*x6 - dgtdd_dx[7]*x154 - gtdd[1]*x275 + gtdd[1]*x277 + x112*(dgtdd_dx[3]*x71 + dgtdd_dx[9]*x66) - x278*x279;
	double x281 = -x280;
	double x282 = x111*x165;
	double x283 = x112*(dgtdd_dx[10]*x66 + dgtdd_dx[4]*x71);
	double x284 = ddgtdd_dx2[10]*x6;
	double x285 = gtdd[4]*x114;
	double x286 = x112*(dgtdd_dx[15]*x66 + dgtdd_dx[3]*x76);
	double x287 = ddgtdd_dx2[15]*x6;
	double x288 = ddW_dx2[2]*x276;
	double x289 = x170*x279;
	double x290 = x286 - x287 + x288 - x289;
	double x291 = ddgtdd_dx2[20]*x6;
	double x292 = gtdd[2]*x277;
	double x293 = gtdd[2]*x275;
	double x294 = dgtdd_dx[8]*x154;
	double x295 = -x291 - x292 + x293 + x294;
	double x296 = -ddW_dx2[1]*x285 + gtdd[4]*x171*x278 - 2*x283 + 2*x284 + x290 + x295;
	double x297 = x112*(dgtdd_dx[13]*x71 + dgtdd_dx[7]*x76);
	double x298 = x291 + x292 - x293 - x294;
	double x299 = -x286 + x287 - x288 + x289;
	double x300 = 12*dW_dx[1]*dW_dx[2]*gtdd[1]*x119 - ddW_dx2[4]*x153 + 2*ddgtdd_dx2[25]*x6 - 2*x297 - x298 - x299;
	double x301 = x199*x81;
	double x302 = x165*x86;
	double x303 = (1.0/2.0)*x86;
	double x304 = gtuu[3]*x303;
	double x305 = (1.0/2.0)*x81;
	double x306 = gtuu[4]*x305;
	double x307 = gtuu[1]*x221;
	double x308 = x304 + x306 + x307;
	double x309 = x130*x86;
	double x310 = x196*x81;
	double x311 = gtuu[0]*x221 + x309 + x310;
	double x312 = ddW_dx2[4]*x276 + ddgtdd_dx2[22]*x6 - ddgtdd_dx2[27]*x6 - dgtdd_dx[10]*x154 - gtdd[4]*x275 + gtdd[4]*x277 + x112*(dgtdd_dx[15]*x71 + dgtdd_dx[9]*x76) - x180*x279;
	double x313 = -x312;
	double x314 = x106*x184;
	double x315 = ddgtdd_dx2[33]*x6;
	double x316 = dgtdd_dx[10]*x76 + dgtdd_dx[16]*x71;
	double x317 = x112*x316;
	double x318 = dgtdd_dx[15]*x205;
	double x319 = ddW_dx2[5]*x276;
	double x320 = x120*x204;
	double x321 = gtdd[3]*x320;
	double x322 = ddgtdd_dx2[23]*x6 - dgtdd_dx[11]*x154 - gtdd[5]*x275 + gtdd[5]*x277;
	double x323 = 12*dW_dx[1]*dW_dx[2]*gtdd[4]*x119 - ddW_dx2[4]*x285 + 2*ddgtdd_dx2[28]*x6 - x315 - 2*x317 + x318 + x319 - x321 - x322;
	double x324 = gtuu[4]*x303;
	double x325 = gtuu[5]*x305;
	double x326 = gtuu[2]*x221;
	double x327 = x324 + x325 + x326;
	double x328 = x327*x81;
	double x329 = x211*x86;
	double x330 = x62*x74;
	double x331 = x263*x74;
	double x332 = x147*x196;
	double x333 = gtuu[4]*x331;
	double x334 = gtuu[0]*x75;
	double x335 = x270*x334;
	double x336 = gtuu[0]*x79;
	double x337 = x258*x336;
	double x338 = x223*x258;
	double x339 = gtuu[4]*x221;
	double x340 = x254*x339;
	double x341 = x234*x254;
	double x342 = x237*x270;
	double x343 = gtuu[0]*x245;
	double x344 = x227*x240;
	double x345 = x137*x268;
	double x346 = x137*x326;
	double x347 = x136*x327;
	double x348 = x269*x327;
	double x349 = gtuu[4]*x230;
	double x350 = x311*x349;
	double x351 = gtuu[5]*x221;
	double x352 = x311*x351;
	double x353 = -gtuu[1]*x261*x330 - gtuu[2]*x331*x62 + x106*x282 - x109*x272 + x110*x261 + x127*x280 + x127*x281 + x139*x301 + x139*x302 + x139*x308*x93 + x139*x311*x73 + x143*x296 + x143*x300 + x159*x273 + x166*x274 + x167*x274 + x185*x312 + x185*x313 + x210*x323 + x211*x328 - 1.0/4.0*x255 - 1.0/4.0*x259 - ((x261)*(x261))*x64 - x261*x333 - x264*x92 - x265*x266 - x265*x96 - x270*x332 - x271*x87 - x271*x94 - x272*x82 + x308*x314 + x308*x329 - 1.0/4.0*x335 - 1.0/4.0*x337 - 1.0/4.0*x338 - 1.0/4.0*x340 - 1.0/4.0*x341 - 1.0/4.0*x342 + (1.0/4.0)*x343 + (1.0/4.0)*x344 + (1.0/4.0)*x345 + (1.0/4.0)*x346 + (1.0/4.0)*x347 + (1.0/4.0)*x348 + (1.0/4.0)*x350 + (1.0/4.0)*x352 - x60*x63;
	double x354 = x13*x9 + x49*x54;
	double x355 = gtuu[1]*x59;
	double x356 = x102*x137;
	double x357 = x104*x199;
	double x358 = (1.0/8.0)*x165;
	double x359 = (1.0/8.0)*x199;
	double x360 = gtuu[0]*x101;
	double x361 = ddW_dx2[1]*x116;
	double x362 = -ddgtdd_dx2[7]*x6 - gtdd[1]*x120*x278 + gtdd[1]*x361 + x152;
	double x363 = x156 + x362;
	double x364 = x157 + x362;
	double x365 = ddgtdd_dx2[8]*x6;
	double x366 = gtdd[2]*x361;
	double x367 = x120*x278;
	double x368 = gtdd[2]*x367;
	double x369 = x186 - x365 + x366 - x368;
	double x370 = x188 + x369;
	double x371 = gtdd[1]*x116;
	double x372 = gtdd[1]*x120;
	double x373 = -ddW_dx2[2]*x371 - x168 + x169 + x170*x372;
	double x374 = x176 + x373;
	double x375 = -x374;
	double x376 = x105*x81;
	double x377 = x86*x95;
	double x378 = -gtdd[4]*x361 + gtdd[4]*x367 - x283 + x284;
	double x379 = x290 + x378;
	double x380 = -x379;
	double x381 = ddgtdd_dx2[25]*x6;
	double x382 = ddW_dx2[4]*x371;
	double x383 = x180*x372;
	double x384 = x297 - x381 + x382 - x383;
	double x385 = x298 + x384;
	double x386 = x112*(dgtdd_dx[16]*x66 + dgtdd_dx[4]*x76);
	double x387 = ddgtdd_dx2[16]*x6;
	double x388 = ddW_dx2[2]*x116;
	double x389 = gtdd[4]*x388;
	double x390 = x120*x170;
	double x391 = gtdd[4]*x390;
	double x392 = ddgtdd_dx2[11]*x6 - gtdd[5]*x361 + gtdd[5]*x367 - x112*(dgtdd_dx[11]*x66 + dgtdd_dx[5]*x71);
	double x393 = x386 - x387 + x389 - x391 + x392;
	double x394 = x112*(dgtdd_dx[14]*x71 + dgtdd_dx[8]*x76);
	double x395 = ddgtdd_dx2[26]*x6;
	double x396 = gtdd[2]*x116;
	double x397 = ddW_dx2[4]*x396;
	double x398 = x120*x180;
	double x399 = gtdd[2]*x398;
	double x400 = x394 - x395 + x397 - x399;
	double x401 = ddgtdd_dx2[31]*x6;
	double x402 = dgtdd_dx[13]*x205;
	double x403 = ddW_dx2[5]*x371;
	double x404 = gtdd[1]*x320;
	double x405 = x401 - x402 - x403 + x404;
	double x406 = -x393 - x400 - x405;
	double x407 = x58*x60;
	double x408 = x165*x334;
	double x409 = x199*x336;
	double x410 = x355*x58;
	double x411 = x135*x240;
	double x412 = gtuu[2]*x59;
	double x413 = x412*x58;
	double x414 = x62*x90;
	double x415 = x62*x64;
	double x416 = gtuu[4]*x59;
	double x417 = x416*x62;
	double x418 = x416*x90;
	double x419 = x137*x339;
	double x420 = x90*x92;
	double x421 = x137*x234;
	double x422 = x165*x237;
	double x423 = gtuu[0]*x135*x80;
	double x424 = gtuu[1]*x227*x80;
	double x425 = x100*x268;
	double x426 = x100*x326;
	double x427 = x136*x216;
	double x428 = x216*x269;
	double x429 = x149*x349;
	double x430 = x149*x351;
	double x431 = 2*gtuu[0]*x93*x95 + 2*gtuu[1]*x107 + 2*gtuu[2]*x376 + 2*gtuu[2]*x377 - 2*x109*x359 + 2*x110*x62 + 2*x127*x363 + 2*x127*x364 - 1.0/4.0*x137*x96 - 2*x138*x274 + 2*x139*x195 + 2*x143*x370 + 2*x143*x375 + 2*x150*x73 - 2*x165*x332 + 2*x185*x380 + 2*x185*x385 + 2*x194*x314 + 2*x194*x329 + 2*x210*x406 + 2*x217*x81 - 2*x261*x410 - 2*x261*x415 - 2*x261*x418 - 2*x263*x413 - 2*x263*x417 - 2*x263*x420 - 2*x355*x63 - 1.0/2.0*x356 - 1.0/2.0*x357 - 2*x358*x87 - 2*x358*x94 - 2*x359*x82 + 2*x360*x73 - 2*x407*x62 - 1.0/2.0*x408 - 1.0/2.0*x409 - 1.0/2.0*x411 - 2*x412*x414 - 1.0/2.0*x419 - 1.0/2.0*x421 - 1.0/2.0*x422 + (1.0/2.0)*x423 + (1.0/2.0)*x424 + (1.0/2.0)*x425 + (1.0/2.0)*x426 + (1.0/2.0)*x427 + (1.0/2.0)*x428 + (1.0/2.0)*x429 + (1.0/2.0)*x430;
	double x432 = (1.0/2.0)*x108;
	double x433 = gtuu[4]*x432;
	double x434 = gtuu[1]*x233 + gtuu[3]*x236 + x433;
	double x435 = x434*x94;
	double x436 = x108*x196;
	double x437 = gtuu[0]*x233 + gtuu[1]*x236 + x436;
	double x438 = x437*x96;
	double x439 = gtdd[5]*x56;
	double x440 = Atdd[5] + x439;
	double x441 = (1.0/8.0)*x437;
	double x442 = gtuu[5]*x432;
	double x443 = gtuu[2]*x233;
	double x444 = gtuu[4]*x236;
	double x445 = x442 + x443 + x444;
	double x446 = (1.0/8.0)*x445;
	double x447 = x106*x160;
	double x448 = (1.0/8.0)*x87;
	double x449 = (1.0/8.0)*x273;
	double x450 = x103*x216;
	double x451 = x392 + x405;
	double x452 = 12*dW_dx[1]*dW_dx[2]*gtdd[2]*x119 - ddW_dx2[4]*x187 + 2*ddgtdd_dx2[26]*x6 - 2*x394 - x451;
	double x453 = 12*dW_dx[0]*dW_dx[2]*gtdd[4]*x119 - ddW_dx2[2]*x285 + 2*ddgtdd_dx2[16]*x6 - 2*x386 - x451;
	double x454 = x216*x81;
	double x455 = x111*x78;
	double x456 = x103*x327;
	double x457 = ddW_dx2[5]*x396 + ddgtdd_dx2[17]*x6 - ddgtdd_dx2[32]*x6 + dgtdd_dx[14]*x205 - gtdd[2]*x320 - gtdd[5]*x388 + gtdd[5]*x390 - x112*(dgtdd_dx[17]*x66 + dgtdd_dx[5]*x76);
	double x458 = -x457;
	double x459 = x108*x139;
	double x460 = x151*x323;
	double x461 = x160*x86;
	double x462 = ddW_dx2[4]*x116;
	double x463 = ddW_dx2[5]*gtdd[4]*x116 + ddgtdd_dx2[29]*x6 - ddgtdd_dx2[34]*x6 + dgtdd_dx[16]*x205 - gtdd[4]*x320 + gtdd[5]*x398 - gtdd[5]*x462 - x112*(dgtdd_dx[11]*x76 + dgtdd_dx[17]*x71);
	double x464 = -x463;
	double x465 = x108*x184;
	double x466 = x130*x135;
	double x467 = x334*x434;
	double x468 = x336*x445;
	double x469 = x225*x434;
	double x470 = x228*x445;
	double x471 = x231*x437;
	double x472 = x339*x437;
	double x473 = gtuu[0]*x147;
	double x474 = x194*x473;
	double x475 = x149*x307;
	double x476 = x148*x308;
	double x477 = x149*x443;
	double x478 = gtuu[2]*x236;
	double x479 = x194*x478;
	double x480 = gtuu[3]*x221;
	double x481 = x311*x480;
	double x482 = gtuu[4]*x233;
	double x483 = x311*x482;
	double x484 = x308*x444;
	double x485 = x48*x54;
	double x486 = (1.0/4.0)*x119;
	double x487 = -x182 - x373;
	double x488 = x130*x5;
	double x489 = x189 + x369;
	double x490 = -2*ddW_dx2[2]*gtdd[2]*x112 + gtdd[2]*x390 - x112*x202 + x203;
	double x491 = -x206 - x490;
	double x492 = x196*x5;
	double x493 = -x207 - x490;
	double x494 = x295 - x297 + x379 + x381 - x382 + x383;
	double x495 = (1.0/2.0)*x151;
	double x496 = gtuu[3]*x305;
	double x497 = x222*x5;
	double x498 = -x401 + x402 + x403 - x404;
	double x499 = x394 - x395 + x397 - x399 - x498;
	double x500 = x194*x94;
	double x501 = x149*x96;
	double x502 = x473*x85;
	double x503 = x100*x307;
	double x504 = x148*x165;
	double x505 = x100*x443;
	double x506 = x478*x85;
	double x507 = x137*x480;
	double x508 = x137*x482;
	double x509 = x165*x444;
	double x510 = x194*x334;
	double x511 = x216*x336;
	double x512 = x194*x225;
	double x513 = x216*x228;
	double x514 = x149*x231;
	double x515 = x149*x339;
	double x516 = 2*x223;
	double x517 = x100*x144 + x103*x128*x80 + x130*x200 + x130*x80*x81 + x137*x192 - x146*x216 - x149*x213 - x149*x97 + x165*x304 - x194*x267 - x194*x324 + x199*x433 + x199*x496 - x216*x306 - x216*x516 + x309*x85 + x393*x497 + x436*x80 + x487*x488 + x488*x489 + x491*x492 + x492*x493 + x494*x495 + x497*x499 - x500 - x501 + x502 + x503 + x504 + x505 + x506 + x507 + x508 + x509 - x510 - x511 - x512 - x513 - x514 - x515;
	double x518 = gtdd[2]*x74;
	double x519 = x128*x5;
	double x520 = x128*x450 + x130*x454 + x130*x456 + x144*x149 - x146*x445 + x192*x311 + x194*x309 + x208*x519 - x213*x437 + x216*x436 - x267*x434 + x304*x308 - x306*x445 - x324*x434 + x327*x433 + x327*x496 - x435 - x437*x97 - x438 - x445*x516 + x452*x488 + x453*x488 + x457*x492 + x458*x492 + (1.0/2.0)*x460 + x463*x497 + x464*x497 - x467 - x468 - x469 - x470 - x471 - x472 + x474 + x475 + x476 + x477 + x479 + x481 + x483 + x484;
	double x521 = (1.0/4.0)*x520;
	double x522 = (1.0/2.0)*x209;
	double x523 = (1.0/2.0)*gtuu[3]*x167 + (1.0/2.0)*gtuu[5]*x212 - x100*x161 - x100*x213 - 2*x100*x339 + x125*x130 + x126*x488 - x131*x85 + x137*x84 + x141*x196 + x142*x492 - x146*x80 + x149*x70 + x158*x495 + x163*x165 + x183*x497 + x190*x497 + x191*x222 + x193*x199 + x194*x198 + x201*x222 + x208*x522 + x214*x216 - x224 - x226 - x229 - x232 - x235 - x238 + x239 + x241 + x243 + x244 + x246 + x247 + x248 + x250 - x267*x85 - x442*x80 - x83 - x88;
	double x524 = (1.0/4.0)*x29;
	double x525 = x440*x486;
	double x526 = 2*x225;
	double x527 = gtuu[2]*x162*x308 + gtuu[5]*x303*x308 + x128*x166 + x129*x137 - x131*x270 + x158*x519 - x161*x254 + x165*x253 + x196*x302 + x197*x311 + x199*x310 - x254*x97 - x255 + x257*x308 - x258*x306 - x258*x442 - x259 - x270*x324 - x270*x526 + x280*x488 + x281*x488 + x296*x492 + x300*x492 + x312*x497 + x313*x497 + x323*x522 + x325*x327 - x335 - x337 - x338 - x340 - x341 - x342 + x343 + x344 + x345 + x346 + x347 + x348 + x350 + x352;
	double x528 = x100*x129 + x128*x85*x93 - x131*x165 - x137*x161 - x137*x97 + x149*x197 + x162*x242 - x165*x324 - x165*x526 + x194*x257 + x196*x85*x86 - x199*x306 - x199*x442 + x216*x325 + x249*x303 + x253*x85 + x310*x80 - x356 - x357 + x363*x488 + x364*x488 + x370*x492 + x375*x492 + x380*x497 + x385*x497 + x406*x522 - x408 - x409 - x411 - x419 - x421 - x422 + x423 + x424 + x425 + x426 + x427 + x428 + x429 + x430;
	double x529 = 2*x5;
	double x530 = x182 - x186 + x365 - x366 + x368 + x374;
	double x531 = x299 + x384;
	double x532 = -x295 - x378;
	double x533 = x103*x258;
	double x534 = x392 + x400;
	double x535 = x386 - x387 + x389 - x391 - x498;
	double x536 = ddgtdd_dx2[28]*x6;
	double x537 = gtdd[4]*x398;
	double x538 = gtdd[4]*x462 + x317 + x322 - x536 - x537;
	double x539 = 2*ddW_dx2[4]*gtdd[4]*x112 + x112*x316 + x315 - x318 - x319 + x321 - x536 - x537;
	double x540 = x308*x94;
	double x541 = x311*x96;
	double x542 = x165*x473;
	double x543 = x137*x307;
	double x544 = x148*x270;
	double x545 = x137*x443;
	double x546 = x165*x478;
	double x547 = x254*x480;
	double x548 = x254*x482;
	double x549 = x270*x444;
	double x550 = x308*x334;
	double x551 = x327*x336;
	double x552 = x225*x308;
	double x553 = x228*x327;
	double x554 = x231*x311;
	double x555 = x311*x339;
	double x556 = x128*x200 + x130*x301 + x130*x533 + x137*x144 - x146*x327 + x165*x309 + x192*x254 + x199*x436 - x213*x311 + x258*x433 + x258*x496 - x267*x308 + x270*x304 - x306*x327 - x308*x324 - x311*x97 - x327*x516 + x488*x531 + x488*x532 + x492*x534 + x492*x535 + x497*x538 + x497*x539 + x519*x530 - x540 - x541 + x542 + x543 + x544 + x545 + x546 + x547 + x548 + x549 - x550 - x551 - x552 - x553 - x554 - x555;
	double x557 = gtuu[1]*x528*x529 + gtuu[2]*x517*x529 + gtuu[4]*x529*x556 + x151*x527 + x209*x520 + x273*x523;
	double x558 = (1.0/8.0)*x119*x557;
	double x559 = gtdd[5]*x558;
	double x560 = -gtdd[0]*x559 + ((gtdd[2])*(gtdd[2]))*x558 - x486*x91 - x517*x518 + x521*x7 + x523*x524 + x525*x58;
	double x561 = 1/(x36);
	double x562 = x0*x561;
	double x563 = x560*x562;
	double x564 = gtdd[4]*x74;
	double x565 = -gtdd[3]*x559 + ((gtdd[4])*(gtdd[4]))*x558 + x261*x525 - x264*x486 + x521*x8 + x524*x527 - x556*x564;
	double x566 = x562*x565;
	double x567 = gtdd[1]*x74;
	double x568 = (1.0/4.0)*x7;
	double x569 = gtdd[3]*x558;
	double x570 = -gtdd[0]*x569 + ((gtdd[1])*(gtdd[1]))*x558 + x261*x486*x58 - x486*x63 + (1.0/4.0)*x523*x8 + x527*x568 - x528*x567;
	double x571 = x561*x570;
	double x572 = x25*x571;
	double x573 = x28*x571;
	double x574 = x56*x74;
	double x575 = (1.0/3.0)*dKhat_dx[2] + (2.0/3.0)*dtheta_dx[2];
	double x576 = 2*Atdd[2] + 2*x89;
	double x577 = (1.0/2.0)*x112;
	double x578 = dW_dx[0]*x577;
	double x579 = x440*x74;
	double x580 = (1.0/3.0)*dKhat_dx[0] + (2.0/3.0)*dtheta_dx[0];
	double x581 = x580*x74;
	double x582 = 2*Atdd[0] + 2*x57;
	double x583 = dW_dx[2]*x577;
	double x584 = dAtdd_dx[12]*x74 - dAtdd_dx[2]*x74 + dgtdd_dx[12]*x574 - dgtdd_dx[2]*x574 - gtdd[2]*x581 + x100*x219 - x149*x218 - x194*x330 - x216*x219 + x331*x85 + (1.0/2.0)*x575*x7 + x576*x578 + x579*x80 - x582*x583;
	double x585 = (1.0/3.0)*dKhat_dx[1] + (2.0/3.0)*dtheta_dx[1];
	double x586 = x585*x74;
	double x587 = 2*Atdd[3] + 2*x260;
	double x588 = x261*x74;
	double x589 = 2*Atdd[4] + 2*x262;
	double x590 = dAtdd_dx[10]*x74 - 1.0/2.0*dAtdd_dx[15]*x6 - 1.0/2.0*dW_dx[1]*x112*x589 + dgtdd_dx[10]*x574 - 1.0/2.0*dgtdd_dx[15]*x56*x6 - 1.0/2.0*gtdd[3]*x575*x6 + gtdd[4]*x586 - 1.0/2.0*x254*x6*x90 - 1.0/2.0*x258*x440*x6 - 1.0/2.0*x263*x270*x6 + x308*x588 + x311*x330 + x327*x331 + x583*x587;
	double x591 = -x590;
	double x592 = dAtdd_dx[4]*x74;
	double x593 = x578*x589;
	double x594 = dgtdd_dx[4]*x574;
	double x595 = gtdd[4]*x581;
	double x596 = x218*x311;
	double x597 = x308*x330;
	double x598 = x219*x327;
	double x599 = 2*Atdd[1] + 2*x61;
	double x600 = dAtdd_dx[13]*x74 + dgtdd_dx[13]*x574 + x137*x219 + x165*x331 + x199*x579 + x567*x575 - x583*x599;
	double x601 = -x592 + x593 - x594 - x595 - x596 - x597 - x598 + x600;
	double x602 = dW_dx[1]*x577;
	double x603 = -dAtdd_dx[8]*x74 - dgtdd_dx[8]*x574 - gtdd[2]*x586 - x149*x330 - x194*x588 - x216*x331 + x576*x602;
	double x604 = x600 + x603;
	double x605 = dAtdd_dx[3]*x74 - dAtdd_dx[7]*x74 + dgtdd_dx[3]*x574 - dgtdd_dx[7]*x574 - gtdd[1]*x586 - x137*x330 - x165*x588 - x199*x331 + x218*x254 + x219*x258 + x270*x330 - x578*x587 + (1.0/2.0)*x580*x8 + x599*x602;
	double x606 = (1.0/2.0)*x585;
	double x607 = dAtdd_dx[1]*x74 - dAtdd_dx[6]*x74 + dgtdd_dx[1]*x574 - dgtdd_dx[6]*x574 + gtdd[1]*x581 - x100*x330 + x137*x218 + x165*x330 + x199*x219 - x331*x80 - x578*x599 + x582*x602 - x588*x85 - x606*x7;
	double x608 = -x607;
	double x609 = -x605;
	double x610 = 2*x354;
	double x611 = gtdd[4]*x59;
	double x612 = x263*x486;
	double x613 = -gtdd[1]*x559 + gtdd[2]*gtdd[4]*x558 - gtdd[2]*x556*x59 + x10*x521 - x517*x611 + x524*x528 + x525*x62 - x612*x90;
	double x614 = x562*x613;
	double x615 = (1.0/8.0)*x149;
	double x616 = (1.0/8.0)*x216;
	double x617 = gtuu[0]*x103*x105 + gtuu[1]*x376 + gtuu[1]*x377 + gtuu[2]*x105*x108 - x102*x615 - x104*x616 + x110*x90 + x111*x191 + x111*x200 + x127*x487 + x127*x489 + x143*x491 + x143*x493 + (1.0/8.0)*x151*x494 + x160*x301 + x160*x302 + x185*x393 + x185*x499 - x194*x447 - x194*x448 + x199*x465 - x216*x466 - x263*x410 - x263*x415 - x263*x418 - x266*x615 - x355*x414 + x360*x78 - x407*x90 - x412*x91 - x413*x440 - x417*x440 - x420*x440 - 1.0/4.0*x500 - 1.0/4.0*x501 + (1.0/4.0)*x502 + (1.0/4.0)*x503 + (1.0/4.0)*x504 + (1.0/4.0)*x505 + (1.0/4.0)*x506 + (1.0/4.0)*x507 + (1.0/4.0)*x508 + (1.0/4.0)*x509 - 1.0/4.0*x510 - 1.0/4.0*x511 - 1.0/4.0*x512 - 1.0/4.0*x513 - 1.0/4.0*x514 - 1.0/4.0*x515 - x616*x82;
	double x618 = x51*x54;
	double x619 = (1.0/8.0)*x311;
	double x620 = (1.0/8.0)*x327;
	double x621 = -x102*x619 - x104*x620 + x110*x263 + x111*x301 + x111*x533 + x127*x531 + x127*x532 + x143*x534 + x143*x535 + x160*x258*x81 + x185*x538 + x185*x539 + x191*x274 + x199*x459 + x200*x274 + x254*x455 + x258*x465 - x261*x263*x64 - x261*x355*x90 - x261*x416*x440 - x263*x355*x62 - x263*x412*x90 - x263*x440*x92 - x264*x416 - x266*x619 + x270*x461 + x282*x86 - x308*x447 - x308*x448 - x327*x466 - x412*x440*x62 - x414*x60 + x449*x530 - 1.0/4.0*x540 - 1.0/4.0*x541 + (1.0/4.0)*x542 + (1.0/4.0)*x543 + (1.0/4.0)*x544 + (1.0/4.0)*x545 + (1.0/4.0)*x546 + (1.0/4.0)*x547 + (1.0/4.0)*x548 + (1.0/4.0)*x549 - 1.0/4.0*x550 - 1.0/4.0*x551 - 1.0/4.0*x552 - 1.0/4.0*x553 - 1.0/4.0*x554 - 1.0/4.0*x555 - x620*x82;
	double x622 = x52*x54;
	double x623 = x485*x561;
	double x624 = 2*Atdd[5] + 2*x439;
	double x625 = dAtdd_dx[14]*x74 - 1.0/2.0*dAtdd_dx[5]*x6 - 1.0/2.0*dW_dx[2]*x112*x576 + dgtdd_dx[14]*x574 - 1.0/2.0*dgtdd_dx[5]*x56*x6 - 1.0/2.0*gtdd[5]*x580*x6 + x149*x219 + x194*x331 + x216*x579 - 1.0/2.0*x434*x6*x62 - 1.0/2.0*x437*x58*x6 - 1.0/2.0*x445*x6*x90 + x518*x575 + x578*x624;
	double x626 = dAtdd_dx[11]*x74 - dAtdd_dx[16]*x74 + dgtdd_dx[11]*x574 - dgtdd_dx[16]*x574 - x219*x311 + x29*x606 - x308*x331 - x327*x579 + x330*x437 + x331*x445 + x434*x588 - x564*x575 + x583*x589 - x602*x624;
	double x627 = -1.0/8.0*gtdd[1]*gtdd[4]*x119*x557 - 1.0/4.0*gtdd[2]*x527*x6 + gtdd[2]*x569 - 1.0/4.0*gtdd[3]*x517*x6 + (1.0/4.0)*x10*x556 - 1.0/4.0*x119*x261*x90 + x528*x611 + x612*x62;
	double x628 = -x627;
	double x629 = x33*x561;
	double x630 = -1.0/8.0*gtdd[0]*gtdd[4]*x119*x557 + gtdd[1]*gtdd[2]*x558 - 1.0/4.0*gtdd[1]*x517*x6 - 1.0/4.0*gtdd[2]*x528*x6 - 1.0/4.0*x119*x62*x90 + x523*x611 + x556*x568 + x58*x612;
	double x631 = -x630;
	double x632 = x629*x631;
	double x633 = x35*x561;
	double x634 = x627*x633;
	double x635 = x30*x561;
	double x636 = -x570*x635;
	double x637 = -x626;
	double x638 = -x584;
	double x639 = -x601;
	double x640 = x592 - x593 + x594 + x595 + x596 + x597 + x598 + x603;
	double x641 = -x604;
	double x642 = -x640;
	double x643 = x561*x630;
	double x644 = x25*x643;
	double x645 = x561*x628;
	double x646 = x28*x645;
	double x647 = -x560;
	double x648 = x47*x54;
	double x649 = x629*x648;
	double x650 = -x613;
	double x651 = x633*x648;
	double x652 = -x565;
	double x653 = x631*x635;
	double x654 = x627*x635;
	double x655 = (1.0/sqrt(x53));
	double x656 = x43*x655;
	double x657 = x17*x655;
	double x658 = x26*x656 - x45*x657;
	double x659 = x41*x658;
	double x660 = 2*x658;
	double x661 = x50*x657;
	double x662 = 2*x656;
	double x663 = x17*x662;
	double x664 = x26*x655;
	double x665 = x50*x664;
	double x666 = 2*x45;
	double x667 = x664*x666;
	double x668 = x15*x22;
	double x669 = x37*x655;
	double x670 = x47*x669;
	double x671 = x47*x657;
	double x672 = x40*x671;
	double x673 = x14*x15;
	double x674 = x47*x664;
	double x675 = x39*x674;
	double x676 = x27*x668;
	double x677 = x24*x655;
	double x678 = x50*x561;
	double x679 = x677*x678;
	double x680 = x655*x678;
	*Psi4Re = x4*(x24*x252*x37*x605 + x24*x354*x37*x607 + x24*x37*x43*x47*x54*x638 + x24*x37*x45*x47*x54*x639 + x24*x37*x45*x47*x54*x640 - x24*x37*x48*x54*x625 - x25*x560*x623 - x251*x55 - x252*x353 + x252*x37*x591*xyz[2] - x252*x566 - x252*x572 - x252*x628*x629 + x27*x354*x37*x609 + x27*x37*x43*x47*x54*x641 + x27*x37*x43*x47*x54*x642 + x27*x37*x45*x47*x54*x590 + x27*x37*x48*x54*x626 + x27*x37*x55*x608 - x28*x565*x623 - 2*x30*x613*x623 + x354*x37*x601*xyz[2] + x354*x37*x604*xyz[2] - x354*x431 - x354*x632 - x354*x634 + x37*x43*x47*x54*x625*xyz[2] + x37*x45*x47*x54*x637*xyz[2] + x37*x55*x584*xyz[2] - x43*x647*x649 - x43*x650*x651 - x45*x649*x650 - x45*x651*x652 - x485*(-gtuu[1]*x219*x263 - x102*x441 - x104*x446 + x110*x440 + x111*x194*x86 + x111*x454 + x111*x456 + x127*x452 + x127*x453 + x143*x457 + x143*x458 + x160*x328 + x185*x463 + x185*x464 + x208*x449 + x212*x274 + x216*x459 - x220*x440 - x264*x64 - x266*x441 + x274*x450 + x308*x461 + x311*x455 + x327*x465 - x333*x440 - x434*x447 - x434*x448 - 1.0/4.0*x435 - 1.0/4.0*x438 - ((x440)*(x440))*x92 - x445*x466 - x446*x82 + (1.0/8.0)*x460 - 1.0/4.0*x467 - 1.0/4.0*x468 - 1.0/4.0*x469 - 1.0/4.0*x470 - 1.0/4.0*x471 - 1.0/4.0*x472 + (1.0/4.0)*x474 + (1.0/4.0)*x475 + (1.0/4.0)*x476 + (1.0/4.0)*x477 + (1.0/4.0)*x479 + (1.0/4.0)*x481 + (1.0/4.0)*x483 + (1.0/4.0)*x484 - x60*x91) - x55*x563 - x55*x573 - x55*x630*x633 - x610*x614 - x610*x636 - x617*x618 - x618*x646 - x618*x653 - x621*x622 - x622*x644 - x622*x654);
	*Psi4Im = x4*(-x251*x663 + x27*x652*x673*x680 + x353*x667 + x37*x584*x662*x668 - x39*x605*x667 - x39*x607*x658 + x39*x638*x671 - x40*x590*x674 + x40*x608*x663 - x40*x609*x658 + x431*x658 + 4*x45*x645*x673*x677 - x563*x663 + x566*x667 + x572*x667 - x573*x663 - x591*x666*x669*x673 - x601*x659 - x604*x659 + x614*x660 - x617*x661 + x621*x665 + x625*x668*x670 + x632*x658 + x634*x658 + x636*x660 - x637*x670*x673 - x639*x675 - x640*x675 + x641*x672 + x642*x672 - 4*x643*x656*x676 + x644*x665 - x646*x661 - x647*x668*x679 + x650*x673*x679 - x650*x676*x680 - x653*x661 + x654*x665);
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
