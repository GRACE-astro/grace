
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
	const double Atdd[6],
	double alp,
	double theta,
	double Ktr,
	double S,
	double rho,
	double kappa1,
	double kappa2,
	const double Atuu[6],
	double DiDialp,
	double dKhat_dx_upwind,
	double * __restrict__ dKhat_dt
)
{
	*dKhat_dt = -DiDialp - alp*kappa1*theta*(kappa2 - 1) + 4*M_PI*alp*(S + rho) + (1.0/3.0)*alp*(3*Atdd[0]*Atuu[0] + 6*Atdd[1]*Atuu[1] + 6*Atdd[2]*Atuu[2] + 3*Atdd[3]*Atuu[3] + 6*Atdd[4]*Atuu[4] + 3*Atdd[5]*Atuu[5] + ((Ktr)*(Ktr))) + dKhat_dx_upwind;
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
	const double Atdd[6],
	double alp,
	double theta,
	double Khat,
	double rho,
	double kappa1,
	double kappa2,
	double theta_damp_fact,
	const double Atuu[6],
	double Rtrace,
	double dtheta_dx_upwind,
	double * __restrict__ dtheta_dt
)
{
	*dtheta_dt = -1.0/6.0*alp*theta_damp_fact*(3*Atdd[0]*Atuu[0] + 6*Atdd[1]*Atuu[1] + 6*Atdd[2]*Atuu[2] + 3*Atdd[3]*Atuu[3] + 6*Atdd[4]*Atuu[4] + 3*Atdd[5]*Atuu[5] - 3*Rtrace + 6*kappa1*theta*(kappa2 + 2) + 48*M_PI*rho - 2*((Khat + 2*theta)*(Khat + 2*theta))) + dtheta_dx_upwind;
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
	*H = -Atdd[0]*Atuu[0] - 2*Atdd[1]*Atuu[1] - 2*Atdd[2]*Atuu[2] - Atdd[3]*Atuu[3] - 2*Atdd[4]*Atuu[4] - Atdd[5]*Atuu[5] + Rtrace - 16*M_PI*rho + (2.0/3.0)*((Khat + 2*theta)*(Khat + 2*theta));
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
	double x65 = gtdd[2]*x56;
	double x66 = Atdd[2] + x65;
	double x67 = ((x66)*(x66));
	double x68 = gtuu[5]*x59;
	double x69 = Khat + 2*theta;
	double x70 = x59*x69;
	double x71 = 1/(((W)*(W)*(W)));
	double x72 = 2*dgtdd_dx[0];
	double x73 = 2*dW_dx[0];
	double x74 = 4*x71;
	double x75 = dW_dx[0]*x74;
	double x76 = 2*x71;
	double x77 = ddW_dx2[0]*x76;
	double x78 = gtdd[0]*x76;
	double x79 = 1/(((W)*(W)*(W)*(W)));
	double x80 = 6*x79;
	double x81 = ((dW_dx[0])*(dW_dx[0]))*x80;
	double x82 = gtdd[0]*x80;
	double x83 = dW_dx[0]*x82;
	double x84 = -dW_dx[1]*x83 + ddW_dx2[1]*x78 + ddgtdd_dx2[1]*x6 - ddgtdd_dx2[6]*x6 - dgtdd_dx[1]*x75 - gtdd[1]*x77 + gtdd[1]*x81 + x71*(dW_dx[1]*x72 + dgtdd_dx[6]*x73);
	double x85 = (1.0/8.0)*x5;
	double x86 = gtuu[1]*x85;
	double x87 = -x84;
	double x88 = dW_dx[2]*x83 - ddW_dx2[2]*x78 + ddgtdd_dx2[12]*x6 - ddgtdd_dx2[2]*x6 + dgtdd_dx[2]*x75 + gtdd[2]*x77 - gtdd[2]*x81 - x71*(dW_dx[2]*x72 + dgtdd_dx[12]*x73);
	double x89 = gtuu[2]*x85;
	double x90 = -x88;
	double x91 = gtuu[3]*x5;
	double x92 = 2*dW_dx[1];
	double x93 = x71*(dgtdd_dx[1]*x92 + dgtdd_dx[7]*x73);
	double x94 = gtdd[1]*x74;
	double x95 = dW_dx[1]*x74;
	double x96 = ((dW_dx[1])*(dW_dx[1]));
	double x97 = -ddW_dx2[3]*x78 + ddgtdd_dx2[18]*x6 - dgtdd_dx[6]*x95 + x82*x96;
	double x98 = ddgtdd_dx2[3]*x6 - dgtdd_dx[3]*x75 - gtdd[3]*x77 + gtdd[3]*x81;
	double x99 = 12*dW_dx[0]*dW_dx[1]*gtdd[1]*x79 - ddW_dx2[1]*x94 + 2*ddgtdd_dx2[7]*x6 - 2*x93 - x97 - x98;
	double x100 = (1.0/8.0)*x99;
	double x101 = 2*dW_dx[2];
	double x102 = x71*(dgtdd_dx[13]*x73 + dgtdd_dx[1]*x101);
	double x103 = ddgtdd_dx2[13]*x6;
	double x104 = dW_dx[0]*dW_dx[2];
	double x105 = 12*x79;
	double x106 = ddgtdd_dx2[4]*x6;
	double x107 = gtdd[4]*x81;
	double x108 = gtdd[4]*x77;
	double x109 = dgtdd_dx[4]*x75;
	double x110 = -x106 - x107 + x108 + x109;
	double x111 = x71*(dgtdd_dx[12]*x92 + dgtdd_dx[6]*x101);
	double x112 = ddgtdd_dx2[24]*x6;
	double x113 = ddW_dx2[4]*x78;
	double x114 = dW_dx[1]*dW_dx[2];
	double x115 = x114*x82;
	double x116 = x111 - x112 + x113 - x115;
	double x117 = -ddW_dx2[2]*x94 + gtdd[1]*x104*x105 - 2*x102 + 2*x103 + x110 + x116;
	double x118 = gtuu[4]*x85;
	double x119 = x71*(dgtdd_dx[2]*x92 + dgtdd_dx[8]*x73);
	double x120 = gtdd[2]*x74;
	double x121 = -x111 + x112 - x113 + x115;
	double x122 = x106 + x107 - x108 - x109;
	double x123 = 12*dW_dx[0]*dW_dx[1]*gtdd[2]*x79 - ddW_dx2[1]*x120 + 2*ddgtdd_dx2[8]*x6 - 2*x119 - x121 - x122;
	double x124 = dgtdd_dx[14]*x73 + dgtdd_dx[2]*x101;
	double x125 = ddgtdd_dx2[14]*x6;
	double x126 = ((dW_dx[2])*(dW_dx[2]));
	double x127 = dW_dx[2]*x74;
	double x128 = ddW_dx2[5]*x78 - ddgtdd_dx2[30]*x6 + dgtdd_dx[12]*x127 - x126*x82;
	double x129 = -ddgtdd_dx2[5]*x6 + dgtdd_dx[5]*x75 + gtdd[5]*x77 - gtdd[5]*x81;
	double x130 = -ddW_dx2[2]*x120 + gtdd[2]*x104*x105 - 2*x124*x71 + 2*x125 + x128 + x129;
	double x131 = gtuu[5]*x5;
	double x132 = (1.0/8.0)*x131;
	double x133 = (1.0/2.0)*x6;
	double x134 = x133*x58;
	double x135 = gtuu[1]*x62;
	double x136 = gtuu[2]*x66;
	double x137 = x62*x66;
	double x138 = gtuu[4]*x133;
	double x139 = x100*x91 + x117*x118 + x118*x123 + x130*x132 - x134*x135 - x134*x136 - x137*x138 - ((x58)*(x58))*x60 + x58*x70 - x63*x64 - x67*x68 + x84*x86 + x86*x87 + x88*x89 + x89*x90;
	double x140 = -x1*x13 + x46*x54;
	double x141 = gtdd[3]*x56;
	double x142 = Atdd[3] + x141;
	double x143 = gtdd[4]*x56;
	double x144 = Atdd[4] + x143;
	double x145 = ((x144)*(x144));
	double x146 = gtuu[0]*x5;
	double x147 = ddW_dx2[3]*x76;
	double x148 = gtdd[3]*x76;
	double x149 = x80*x96;
	double x150 = dW_dx[0]*dW_dx[1];
	double x151 = gtdd[3]*x80;
	double x152 = ddW_dx2[1]*x148 + ddgtdd_dx2[19]*x6 - ddgtdd_dx2[9]*x6 - dgtdd_dx[7]*x95 - gtdd[1]*x147 + gtdd[1]*x149 - x150*x151 + x71*(dgtdd_dx[3]*x92 + dgtdd_dx[9]*x73);
	double x153 = -x152;
	double x154 = x71*(dgtdd_dx[10]*x73 + dgtdd_dx[4]*x92);
	double x155 = ddgtdd_dx2[10]*x6;
	double x156 = gtdd[4]*x74;
	double x157 = x71*(dgtdd_dx[15]*x73 + dgtdd_dx[3]*x101);
	double x158 = ddgtdd_dx2[15]*x6;
	double x159 = ddW_dx2[2]*x148;
	double x160 = x104*x151;
	double x161 = x157 - x158 + x159 - x160;
	double x162 = ddgtdd_dx2[20]*x6;
	double x163 = gtdd[2]*x149;
	double x164 = gtdd[2]*x147;
	double x165 = dgtdd_dx[8]*x95;
	double x166 = -x162 - x163 + x164 + x165;
	double x167 = -ddW_dx2[1]*x156 + gtdd[4]*x105*x150 - 2*x154 + 2*x155 + x161 + x166;
	double x168 = x71*(dgtdd_dx[13]*x92 + dgtdd_dx[7]*x101);
	double x169 = x162 + x163 - x164 - x165;
	double x170 = -x157 + x158 - x159 + x160;
	double x171 = 12*dW_dx[1]*dW_dx[2]*gtdd[1]*x79 - ddW_dx2[4]*x94 + 2*ddgtdd_dx2[25]*x6 - 2*x168 - x169 - x170;
	double x172 = ddW_dx2[4]*x148 + ddgtdd_dx2[22]*x6 - ddgtdd_dx2[27]*x6 - dgtdd_dx[10]*x95 - gtdd[4]*x147 + gtdd[4]*x149 - x114*x151 + x71*(dgtdd_dx[15]*x92 + dgtdd_dx[9]*x101);
	double x173 = -x172;
	double x174 = ddgtdd_dx2[33]*x6;
	double x175 = dgtdd_dx[10]*x101 + dgtdd_dx[16]*x92;
	double x176 = x175*x71;
	double x177 = dgtdd_dx[15]*x127;
	double x178 = ddW_dx2[5]*x148;
	double x179 = x126*x80;
	double x180 = gtdd[3]*x179;
	double x181 = ddgtdd_dx2[23]*x6 - dgtdd_dx[11]*x95 - gtdd[5]*x147 + gtdd[5]*x149;
	double x182 = 12*dW_dx[1]*dW_dx[2]*gtdd[4]*x79 - ddW_dx2[4]*x156 + 2*ddgtdd_dx2[28]*x6 - x174 - 2*x176 + x177 + x178 - x180 - x181;
	double x183 = x144*x62;
	double x184 = x138*x144;
	double x185 = -gtuu[2]*x133*x183 + x100*x146 + x118*x172 + x118*x173 + x132*x182 - x133*x135*x142 - ((x142)*(x142))*x64 - x142*x184 + x142*x70 - x145*x68 + x152*x86 + x153*x86 + x167*x89 + x171*x89 - x60*x63;
	double x186 = x13*x9 + x49*x54;
	double x187 = ddW_dx2[1]*x76;
	double x188 = -ddgtdd_dx2[7]*x6 - gtdd[1]*x150*x80 + gtdd[1]*x187 + x93;
	double x189 = x188 + x97;
	double x190 = x188 + x98;
	double x191 = ddgtdd_dx2[8]*x6;
	double x192 = gtdd[2]*x187;
	double x193 = x150*x80;
	double x194 = gtdd[2]*x193;
	double x195 = x119 - x191 + x192 - x194;
	double x196 = x121 + x195;
	double x197 = gtdd[1]*x76;
	double x198 = gtdd[1]*x80;
	double x199 = -ddW_dx2[2]*x197 - x102 + x103 + x104*x198;
	double x200 = x110 + x199;
	double x201 = -x200;
	double x202 = -gtdd[4]*x187 + gtdd[4]*x193 - x154 + x155;
	double x203 = x161 + x202;
	double x204 = -x203;
	double x205 = ddgtdd_dx2[25]*x6;
	double x206 = ddW_dx2[4]*x197;
	double x207 = x114*x198;
	double x208 = x168 - x205 + x206 - x207;
	double x209 = x169 + x208;
	double x210 = x71*(dgtdd_dx[16]*x73 + dgtdd_dx[4]*x101);
	double x211 = ddgtdd_dx2[16]*x6;
	double x212 = ddW_dx2[2]*x76;
	double x213 = gtdd[4]*x212;
	double x214 = x104*x80;
	double x215 = gtdd[4]*x214;
	double x216 = ddgtdd_dx2[11]*x6 - gtdd[5]*x187 + gtdd[5]*x193 - x71*(dgtdd_dx[11]*x73 + dgtdd_dx[5]*x92);
	double x217 = x210 - x211 + x213 - x215 + x216;
	double x218 = x71*(dgtdd_dx[14]*x92 + dgtdd_dx[8]*x101);
	double x219 = ddgtdd_dx2[26]*x6;
	double x220 = gtdd[2]*x76;
	double x221 = ddW_dx2[4]*x220;
	double x222 = x114*x80;
	double x223 = gtdd[2]*x222;
	double x224 = x218 - x219 + x221 - x223;
	double x225 = ddgtdd_dx2[31]*x6;
	double x226 = dgtdd_dx[13]*x127;
	double x227 = ddW_dx2[5]*x197;
	double x228 = gtdd[1]*x179;
	double x229 = x225 - x226 - x227 + x228;
	double x230 = -x217 - x224 - x229;
	double x231 = gtuu[1]*x59;
	double x232 = x58*x60;
	double x233 = x231*x58;
	double x234 = gtuu[2]*x59;
	double x235 = x234*x58;
	double x236 = x142*x64;
	double x237 = gtuu[4]*x59;
	double x238 = x237*x66;
	double x239 = x144*x66;
	double x240 = (1.0/4.0)*gtuu[1]*x189*x5 + (1.0/4.0)*gtuu[1]*x190*x5 + (1.0/4.0)*gtuu[2]*x196*x5 + (1.0/4.0)*gtuu[2]*x201*x5 + (1.0/4.0)*gtuu[4]*x204*x5 + (1.0/4.0)*gtuu[4]*x209*x5 + (1.0/4.0)*gtuu[5]*x230*x5 - 2*x137*x234 - 2*x142*x233 - 2*x142*x238 - 2*x144*x235 - 2*x183*x237 - 2*x231*x63 - 2*x232*x62 - 2*x236*x62 - 2*x239*x68 + (1.0/2.0)*x6*x62*x69;
	double x241 = gtdd[5]*x56;
	double x242 = Atdd[5] + x241;
	double x243 = x130*x146;
	double x244 = x216 + x229;
	double x245 = 12*dW_dx[1]*dW_dx[2]*gtdd[2]*x79 - ddW_dx2[4]*x120 + 2*ddgtdd_dx2[26]*x6 - 2*x218 - x244;
	double x246 = 12*dW_dx[0]*dW_dx[2]*gtdd[4]*x79 - ddW_dx2[2]*x156 + 2*ddgtdd_dx2[16]*x6 - 2*x210 - x244;
	double x247 = ddW_dx2[5]*x220 + ddgtdd_dx2[17]*x6 - ddgtdd_dx2[32]*x6 + dgtdd_dx[14]*x127 - gtdd[2]*x179 - gtdd[5]*x212 + gtdd[5]*x214 - x71*(dgtdd_dx[17]*x73 + dgtdd_dx[5]*x101);
	double x248 = -x247;
	double x249 = x182*x91;
	double x250 = ddW_dx2[4]*x76;
	double x251 = ddW_dx2[5]*gtdd[4]*x76 + ddgtdd_dx2[29]*x6 - ddgtdd_dx2[34]*x6 + dgtdd_dx[16]*x127 - gtdd[4]*x179 + gtdd[5]*x222 - gtdd[5]*x250 - x71*(dgtdd_dx[11]*x101 + dgtdd_dx[17]*x92);
	double x252 = -x251;
	double x253 = x48*x54;
	double x254 = (1.0/4.0)*x79;
	double x255 = -x116 - x199;
	double x256 = (1.0/2.0)*x5;
	double x257 = gtuu[1]*x256;
	double x258 = x122 + x195;
	double x259 = -2*ddW_dx2[2]*gtdd[2]*x71 + gtdd[2]*x214 - x124*x71 + x125;
	double x260 = -x128 - x259;
	double x261 = gtuu[2]*x256;
	double x262 = -x129 - x259;
	double x263 = x166 - x168 + x203 + x205 - x206 + x207;
	double x264 = (1.0/2.0)*x91;
	double x265 = gtuu[4]*x256;
	double x266 = -x225 + x226 + x227 - x228;
	double x267 = x218 - x219 + x221 - x223 - x266;
	double x268 = x217*x265 + x255*x257 + x257*x258 + x260*x261 + x261*x262 + x263*x264 + x265*x267;
	double x269 = gtdd[2]*x133;
	double x270 = (1.0/2.0)*x243 + x245*x257 + x246*x257 + x247*x261 + x248*x261 + (1.0/2.0)*x249 + x251*x265 + x252*x265;
	double x271 = (1.0/4.0)*x270;
	double x272 = (1.0/2.0)*x131;
	double x273 = x117*x265 + x123*x265 + x130*x272 + x257*x84 + x257*x87 + x261*x88 + x261*x90 + x264*x99;
	double x274 = (1.0/4.0)*x29;
	double x275 = x242*x254;
	double x276 = (1.0/2.0)*x146;
	double x277 = x152*x257 + x153*x257 + x167*x261 + x171*x261 + x172*x265 + x173*x265 + x182*x272 + x276*x99;
	double x278 = x189*x257 + x190*x257 + x196*x261 + x201*x261 + x204*x265 + x209*x265 + x230*x272;
	double x279 = 2*x5;
	double x280 = x116 - x119 + x191 - x192 + x194 + x200;
	double x281 = x170 + x208;
	double x282 = -x166 - x202;
	double x283 = x216 + x224;
	double x284 = x210 - x211 + x213 - x215 - x266;
	double x285 = ddgtdd_dx2[28]*x6;
	double x286 = gtdd[4]*x222;
	double x287 = gtdd[4]*x250 + x176 + x181 - x285 - x286;
	double x288 = 2*ddW_dx2[4]*gtdd[4]*x71 + x174 + x175*x71 - x177 - x178 + x180 - x285 - x286;
	double x289 = x257*x281 + x257*x282 + x261*x283 + x261*x284 + x265*x287 + x265*x288 + x276*x280;
	double x290 = gtuu[1]*x278*x279 + gtuu[2]*x268*x279 + gtuu[4]*x279*x289 + x131*x270 + x146*x273 + x277*x91;
	double x291 = (1.0/8.0)*x290*x79;
	double x292 = gtdd[5]*x291;
	double x293 = -gtdd[0]*x292 + ((gtdd[2])*(gtdd[2]))*x291 - x254*x67 - x268*x269 + x271*x7 + x273*x274 + x275*x58;
	double x294 = 1/(x36);
	double x295 = x0*x294;
	double x296 = x293*x295;
	double x297 = gtdd[4]*x133;
	double x298 = -gtdd[3]*x292 + ((gtdd[4])*(gtdd[4]))*x291 + x142*x275 - x145*x254 + x271*x8 + x274*x277 - x289*x297;
	double x299 = x295*x298;
	double x300 = gtdd[1]*x133;
	double x301 = (1.0/4.0)*x7;
	double x302 = gtdd[3]*x291;
	double x303 = -gtdd[0]*x302 + ((gtdd[1])*(gtdd[1]))*x291 + x142*x254*x58 - x254*x63 + (1.0/4.0)*x273*x8 + x277*x301 - x278*x300;
	double x304 = x294*x303;
	double x305 = x25*x304;
	double x306 = x28*x304;
	double x307 = x133*x56;
	double x308 = (1.0/3.0)*dKhat_dx[2] + (2.0/3.0)*dtheta_dx[2];
	double x309 = 2*Atdd[2] + 2*x65;
	double x310 = (1.0/2.0)*x71;
	double x311 = dW_dx[0]*x310;
	double x312 = (1.0/3.0)*dKhat_dx[0] + (2.0/3.0)*dtheta_dx[0];
	double x313 = x133*x312;
	double x314 = 2*Atdd[0] + 2*x57;
	double x315 = dW_dx[2]*x310;
	double x316 = dAtdd_dx[12]*x133 - dAtdd_dx[2]*x133 + dgtdd_dx[12]*x307 - dgtdd_dx[2]*x307 - gtdd[2]*x313 + (1.0/2.0)*x308*x7 + x309*x311 - x314*x315;
	double x317 = (1.0/3.0)*dKhat_dx[1] + (2.0/3.0)*dtheta_dx[1];
	double x318 = x133*x317;
	double x319 = 2*Atdd[3] + 2*x141;
	double x320 = 2*Atdd[4] + 2*x143;
	double x321 = dAtdd_dx[10]*x133 - 1.0/2.0*dAtdd_dx[15]*x6 - 1.0/2.0*dW_dx[1]*x320*x71 + dgtdd_dx[10]*x307 - 1.0/2.0*dgtdd_dx[15]*x56*x6 - 1.0/2.0*gtdd[3]*x308*x6 + gtdd[4]*x318 + x315*x319;
	double x322 = -x321;
	double x323 = dAtdd_dx[4]*x133;
	double x324 = x311*x320;
	double x325 = dgtdd_dx[4]*x307;
	double x326 = gtdd[4]*x313;
	double x327 = 2*Atdd[1] + 2*x61;
	double x328 = dAtdd_dx[13]*x133 + dgtdd_dx[13]*x307 + x300*x308 - x315*x327;
	double x329 = -x323 + x324 - x325 - x326 + x328;
	double x330 = dW_dx[1]*x310;
	double x331 = -dAtdd_dx[8]*x133 - dgtdd_dx[8]*x307 - gtdd[2]*x318 + x309*x330;
	double x332 = x328 + x331;
	double x333 = dAtdd_dx[3]*x133 - dAtdd_dx[7]*x133 + dgtdd_dx[3]*x307 - dgtdd_dx[7]*x307 - gtdd[1]*x318 - x311*x319 + (1.0/2.0)*x312*x8 + x327*x330;
	double x334 = (1.0/2.0)*x317;
	double x335 = dAtdd_dx[1]*x133 - dAtdd_dx[6]*x133 + dgtdd_dx[1]*x307 - dgtdd_dx[6]*x307 + gtdd[1]*x313 - x311*x327 + x314*x330 - x334*x7;
	double x336 = -x335;
	double x337 = -x333;
	double x338 = 2*x186;
	double x339 = gtdd[4]*x59;
	double x340 = x144*x254;
	double x341 = -gtdd[1]*x292 + gtdd[2]*gtdd[4]*x291 - gtdd[2]*x289*x59 + x10*x271 - x268*x339 + x274*x278 + x275*x62 - x340*x66;
	double x342 = x295*x341;
	double x343 = x242*x62;
	double x344 = x242*x68;
	double x345 = (1.0/8.0)*gtuu[1]*x255*x5 + (1.0/8.0)*gtuu[1]*x258*x5 + (1.0/8.0)*gtuu[2]*x260*x5 + (1.0/8.0)*gtuu[2]*x262*x5 + (1.0/8.0)*gtuu[3]*x263*x5 + (1.0/8.0)*gtuu[4]*x217*x5 + (1.0/8.0)*gtuu[4]*x267*x5 - x137*x231 - x144*x233 - x144*x238 - x183*x64 - x232*x66 - x234*x67 - x235*x242 - x237*x343 - x344*x66 + (1.0/4.0)*x6*x66*x69;
	double x346 = x51*x54;
	double x347 = (1.0/8.0)*gtuu[0]*x280*x5 + (1.0/8.0)*gtuu[1]*x281*x5 + (1.0/8.0)*gtuu[1]*x282*x5 + (1.0/8.0)*gtuu[2]*x283*x5 + (1.0/8.0)*gtuu[2]*x284*x5 + (1.0/8.0)*gtuu[4]*x287*x5 + (1.0/8.0)*gtuu[4]*x288*x5 - x137*x60 - x142*x231*x66 - x142*x237*x242 - x144*x236 - x144*x344 + (1.0/4.0)*x144*x6*x69 - x145*x237 - x183*x231 - x234*x239 - x234*x343;
	double x348 = x52*x54;
	double x349 = x253*x294;
	double x350 = 2*Atdd[5] + 2*x241;
	double x351 = dAtdd_dx[14]*x133 - 1.0/2.0*dAtdd_dx[5]*x6 - 1.0/2.0*dW_dx[2]*x309*x71 + dgtdd_dx[14]*x307 - 1.0/2.0*dgtdd_dx[5]*x56*x6 - 1.0/2.0*gtdd[5]*x312*x6 + x269*x308 + x311*x350;
	double x352 = dAtdd_dx[11]*x133 - dAtdd_dx[16]*x133 + dgtdd_dx[11]*x307 - dgtdd_dx[16]*x307 + x29*x334 - x297*x308 + x315*x320 - x330*x350;
	double x353 = -1.0/8.0*gtdd[1]*gtdd[4]*x290*x79 - 1.0/4.0*gtdd[2]*x277*x6 + gtdd[2]*x302 - 1.0/4.0*gtdd[3]*x268*x6 + (1.0/4.0)*x10*x289 - 1.0/4.0*x142*x66*x79 + x278*x339 + x340*x62;
	double x354 = -x353;
	double x355 = x294*x33;
	double x356 = -1.0/8.0*gtdd[0]*gtdd[4]*x290*x79 + gtdd[1]*gtdd[2]*x291 - 1.0/4.0*gtdd[1]*x268*x6 - 1.0/4.0*gtdd[2]*x278*x6 + x273*x339 + x289*x301 + x340*x58 - 1.0/4.0*x62*x66*x79;
	double x357 = -x356;
	double x358 = x355*x357;
	double x359 = x294*x35;
	double x360 = x353*x359;
	double x361 = x294*x30;
	double x362 = -x303*x361;
	double x363 = -x352;
	double x364 = -x316;
	double x365 = -x329;
	double x366 = x323 - x324 + x325 + x326 + x331;
	double x367 = -x332;
	double x368 = -x366;
	double x369 = x294*x356;
	double x370 = x25*x369;
	double x371 = x294*x354;
	double x372 = x28*x371;
	double x373 = -x293;
	double x374 = x47*x54;
	double x375 = x355*x374;
	double x376 = -x341;
	double x377 = x359*x374;
	double x378 = -x298;
	double x379 = x357*x361;
	double x380 = x353*x361;
	double x381 = (1.0/sqrt(x53));
	double x382 = x381*x43;
	double x383 = x17*x381;
	double x384 = x26*x382 - x383*x45;
	double x385 = x384*x41;
	double x386 = 2*x384;
	double x387 = x383*x50;
	double x388 = 2*x382;
	double x389 = x17*x388;
	double x390 = x26*x381;
	double x391 = x390*x50;
	double x392 = 2*x45;
	double x393 = x390*x392;
	double x394 = x15*x22;
	double x395 = x37*x381;
	double x396 = x395*x47;
	double x397 = x383*x47;
	double x398 = x397*x40;
	double x399 = x14*x15;
	double x400 = x390*x47;
	double x401 = x39*x400;
	double x402 = x27*x394;
	double x403 = x24*x381;
	double x404 = x294*x50;
	double x405 = x403*x404;
	double x406 = x381*x404;
	*Psi4Re = x4*(-x139*x55 - x140*x185 + x140*x24*x333*x37 - x140*x299 - x140*x305 + x140*x322*x37*xyz[2] - x140*x354*x355 + x186*x24*x335*x37 - x186*x240 + x186*x27*x337*x37 + x186*x329*x37*xyz[2] + x186*x332*x37*xyz[2] - x186*x358 - x186*x360 - x24*x351*x37*x48*x54 + x24*x364*x37*x43*x47*x54 + x24*x365*x37*x45*x47*x54 + x24*x366*x37*x45*x47*x54 - x25*x293*x349 - x253*(-gtuu[1]*x133*x239 + x118*x251 + x118*x252 - x133*x136*x242 - x145*x64 - x184*x242 - ((x242)*(x242))*x68 + x242*x70 + (1.0/8.0)*x243 + x245*x86 + x246*x86 + x247*x89 + x248*x89 + (1.0/8.0)*x249 - x60*x67) + x27*x321*x37*x45*x47*x54 + x27*x336*x37*x55 + x27*x352*x37*x48*x54 + x27*x367*x37*x43*x47*x54 + x27*x368*x37*x43*x47*x54 - x28*x298*x349 - x296*x55 - 2*x30*x341*x349 - x306*x55 + x316*x37*x55*xyz[2] - x338*x342 - x338*x362 - x345*x346 - x346*x372 - x346*x379 - x347*x348 - x348*x370 - x348*x380 + x351*x37*x43*x47*x54*xyz[2] - x356*x359*x55 + x363*x37*x45*x47*x54*xyz[2] - x373*x375*x43 - x375*x376*x45 - x376*x377*x43 - x377*x378*x45);
	*Psi4Im = x4*(-x139*x389 + x185*x393 + x240*x384 + x27*x378*x399*x406 - x296*x389 + x299*x393 + x305*x393 - x306*x389 + x316*x37*x388*x394 - x321*x40*x400 - x322*x392*x395*x399 - x329*x385 - x332*x385 - x333*x39*x393 - x335*x384*x39 + x336*x389*x40 - x337*x384*x40 + x342*x386 - x345*x387 + x347*x391 + x351*x394*x396 + x358*x384 + x360*x384 + x362*x386 - x363*x396*x399 + x364*x39*x397 - x365*x401 - x366*x401 + x367*x398 + x368*x398 - 4*x369*x382*x402 + x370*x391 + 4*x371*x399*x403*x45 - x372*x387 - x373*x394*x405 + x376*x399*x405 - x376*x402*x406 - x379*x387 + x380*x391);
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
