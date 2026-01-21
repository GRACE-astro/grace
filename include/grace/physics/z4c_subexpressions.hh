
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
z4c_get_Christoffel(
	const double dgtdd_dx[18],
	double (*Gammatddd)[18]
)
{
	double x0 = (1.0/2.0)*dgtdd_dx[6];
	double x1 = (1.0/2.0)*dgtdd_dx[12];
	double x2 = (1.0/2.0)*(dgtdd_dx[13] - dgtdd_dx[4] + dgtdd_dx[8]);
	double x3 = (1.0/2.0)*dgtdd_dx[15];
	(*Gammatddd)[0] = (1.0/2.0)*dgtdd_dx[0];
	(*Gammatddd)[1] = x0;
	(*Gammatddd)[2] = x1;
	(*Gammatddd)[3] = 0;
	(*Gammatddd)[4] = 0;
	(*Gammatddd)[5] = 0;
	(*Gammatddd)[6] = x0;
	(*Gammatddd)[7] = -1.0/2.0*dgtdd_dx[3] + dgtdd_dx[7];
	(*Gammatddd)[8] = x2;
	(*Gammatddd)[9] = (1.0/2.0)*dgtdd_dx[9];
	(*Gammatddd)[10] = x3;
	(*Gammatddd)[11] = 0;
	(*Gammatddd)[12] = x1;
	(*Gammatddd)[13] = x2;
	(*Gammatddd)[14] = dgtdd_dx[14] - 1.0/2.0*dgtdd_dx[5];
	(*Gammatddd)[15] = x3;
	(*Gammatddd)[16] = -1.0/2.0*dgtdd_dx[11] + dgtdd_dx[16];
	(*Gammatddd)[17] = (1.0/2.0)*dgtdd_dx[17];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_DiDjalp(
	const double gtdd[6],
	double W,
	const double gtuu[6],
	const double Gammatddd[18],
	const double dW_dx[3],
	const double dalp_dx[3],
	const double ddalp_dx2[6],
	double (*DiDjalp)[6]
)
{
	double x0 = 1/(W);
	double x1 = gtdd[0]*x0;
	double x2 = dW_dx[0]*dalp_dx[0];
	double x3 = dW_dx[0]*dalp_dx[1];
	double x4 = dW_dx[0]*dalp_dx[2];
	double x5 = dW_dx[1]*dalp_dx[0];
	double x6 = dW_dx[1]*dalp_dx[1];
	double x7 = dW_dx[1]*dalp_dx[2];
	double x8 = dW_dx[2]*dalp_dx[0];
	double x9 = dW_dx[2]*dalp_dx[1];
	double x10 = dW_dx[2]*dalp_dx[2];
	double x11 = -ddalp_dx2[1];
	double x12 = gtdd[1]*x0;
	double x13 = -x0*(x3 + x5);
	double x14 = -ddalp_dx2[2];
	double x15 = gtdd[2]*x0;
	double x16 = -x0*(x4 + x8);
	double x17 = gtdd[3]*x0;
	double x18 = -ddalp_dx2[4];
	double x19 = gtdd[4]*x0;
	double x20 = -x0*(x7 + x9);
	double x21 = gtdd[5]*x0;
	(*DiDjalp)[0] = 2*dW_dx[0]*dalp_dx[0]*x0 - dalp_dx[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) - dalp_dx[1]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) - dalp_dx[2]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + ddalp_dx2[0] - gtuu[0]*x1*x2 - gtuu[1]*x1*x3 - gtuu[1]*x1*x5 - gtuu[2]*x1*x4 - gtuu[2]*x1*x8 - gtuu[3]*x1*x6 - gtuu[4]*x1*x7 - gtuu[4]*x1*x9 - gtuu[5]*x1*x10;
	(*DiDjalp)[1] = -dalp_dx[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) - dalp_dx[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) - dalp_dx[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) - gtuu[0]*x12*x2 - gtuu[1]*x12*x3 - gtuu[1]*x12*x5 - gtuu[2]*x12*x4 - gtuu[2]*x12*x8 - gtuu[3]*x12*x6 - gtuu[4]*x12*x7 - gtuu[4]*x12*x9 - gtuu[5]*x10*x12 - x11 - x13;
	(*DiDjalp)[2] = -dalp_dx[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) - dalp_dx[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) - dalp_dx[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) - gtuu[0]*x15*x2 - gtuu[1]*x15*x3 - gtuu[1]*x15*x5 - gtuu[2]*x15*x4 - gtuu[2]*x15*x8 - gtuu[3]*x15*x6 - gtuu[4]*x15*x7 - gtuu[4]*x15*x9 - gtuu[5]*x10*x15 - x14 - x16;
	(*DiDjalp)[3] = 2*dW_dx[1]*dalp_dx[1]*x0 - dalp_dx[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) - dalp_dx[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) - dalp_dx[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + ddalp_dx2[3] - gtuu[0]*x17*x2 - gtuu[1]*x17*x3 - gtuu[1]*x17*x5 - gtuu[2]*x17*x4 - gtuu[2]*x17*x8 - gtuu[3]*x17*x6 - gtuu[4]*x17*x7 - gtuu[4]*x17*x9 - gtuu[5]*x10*x17;
	(*DiDjalp)[4] = -dalp_dx[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) - dalp_dx[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) - dalp_dx[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) - gtuu[0]*x19*x2 - gtuu[1]*x19*x3 - gtuu[1]*x19*x5 - gtuu[2]*x19*x4 - gtuu[2]*x19*x8 - gtuu[3]*x19*x6 - gtuu[4]*x19*x7 - gtuu[4]*x19*x9 - gtuu[5]*x10*x19 - x18 - x20;
	(*DiDjalp)[5] = 2*dW_dx[2]*dalp_dx[2]*x0 - dalp_dx[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) - dalp_dx[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) - dalp_dx[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + ddalp_dx2[5] - gtuu[0]*x2*x21 - gtuu[1]*x21*x3 - gtuu[1]*x21*x5 - gtuu[2]*x21*x4 - gtuu[2]*x21*x8 - gtuu[3]*x21*x6 - gtuu[4]*x21*x7 - gtuu[4]*x21*x9 - gtuu[5]*x10*x21;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_DiDialp(
	double W,
	const double gtuu[6],
	const double DiDjalp[6],
	double * __restrict__ DiDialp
)
{
	*DiDialp = ((W)*(W))*(DiDjalp[0]*gtuu[0] + 2*DiDjalp[1]*gtuu[1] + 2*DiDjalp[2]*gtuu[2] + DiDjalp[3]*gtuu[3] + 2*DiDjalp[4]*gtuu[4] + DiDjalp[5]*gtuu[5]);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci1(
	const double gtuu[6],
	const double ddgtdd_dx2[36],
	double (*Rw)[6]
)
{
	double x0 = (1.0/2.0)*gtuu[0];
	double x1 = (1.0/2.0)*gtuu[3];
	double x2 = (1.0/2.0)*gtuu[5];
	double x3 = -ddgtdd_dx2[13]*gtuu[2] - ddgtdd_dx2[19]*x1 - ddgtdd_dx2[1]*x0 - ddgtdd_dx2[25]*gtuu[4] - ddgtdd_dx2[31]*x2 - ddgtdd_dx2[7]*gtuu[1];
	double x4 = -ddgtdd_dx2[14]*gtuu[2] - ddgtdd_dx2[20]*x1 - ddgtdd_dx2[26]*gtuu[4] - ddgtdd_dx2[2]*x0 - ddgtdd_dx2[32]*x2 - ddgtdd_dx2[8]*gtuu[1];
	double x5 = -ddgtdd_dx2[10]*gtuu[1] - ddgtdd_dx2[16]*gtuu[2] - ddgtdd_dx2[22]*x1 - ddgtdd_dx2[28]*gtuu[4] - ddgtdd_dx2[34]*x2 - ddgtdd_dx2[4]*x0;
	(*Rw)[0] += -ddgtdd_dx2[0]*x0 - ddgtdd_dx2[12]*gtuu[2] - ddgtdd_dx2[18]*x1 - ddgtdd_dx2[24]*gtuu[4] - ddgtdd_dx2[30]*x2 - ddgtdd_dx2[6]*gtuu[1];
	(*Rw)[1] += x3;
	(*Rw)[2] += x4;
	(*Rw)[3] += -ddgtdd_dx2[15]*gtuu[2] - ddgtdd_dx2[21]*x1 - ddgtdd_dx2[27]*gtuu[4] - ddgtdd_dx2[33]*x2 - ddgtdd_dx2[3]*x0 - ddgtdd_dx2[9]*gtuu[1];
	(*Rw)[4] += x5;
	(*Rw)[5] += -ddgtdd_dx2[11]*gtuu[1] - ddgtdd_dx2[17]*gtuu[2] - ddgtdd_dx2[23]*x1 - ddgtdd_dx2[29]*gtuu[4] - ddgtdd_dx2[35]*x2 - ddgtdd_dx2[5]*x0;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci2(
	const double gtdd[6],
	const double gtuu[6],
	const double Gammatddd[18],
	const double dGammatu_dx[9],
	double (*Rgd)[6]
)
{
	double x0 = 2*gtuu[1];
	double x1 = 2*gtuu[2];
	double x2 = 2*gtuu[4];
	double x3 = gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + x0*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + x1*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + x2*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]);
	double x4 = gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + x0*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + x1*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + x2*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]);
	double x5 = gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x0*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x1*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x2*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]);
	double x6 = (1.0/2.0)*(Gammatddd[1]*x3 + Gammatddd[3]*x4 + Gammatddd[4]*x5 + Gammatddd[6]*x3 + Gammatddd[7]*x4 + Gammatddd[8]*x5 + dGammatu_dx[0]*gtdd[1] + dGammatu_dx[1]*gtdd[3] + dGammatu_dx[2]*gtdd[4] + dGammatu_dx[3]*gtdd[0] + dGammatu_dx[4]*gtdd[1] + dGammatu_dx[5]*gtdd[2]);
	double x7 = (1.0/2.0)*(Gammatddd[12]*x3 + Gammatddd[13]*x4 + Gammatddd[14]*x5 + Gammatddd[2]*x3 + Gammatddd[4]*x4 + Gammatddd[5]*x5 + dGammatu_dx[0]*gtdd[2] + dGammatu_dx[1]*gtdd[4] + dGammatu_dx[2]*gtdd[5] + dGammatu_dx[6]*gtdd[0] + dGammatu_dx[7]*gtdd[1] + dGammatu_dx[8]*gtdd[2]);
	double x8 = (1.0/2.0)*(Gammatddd[10]*x4 + Gammatddd[11]*x5 + Gammatddd[13]*x3 + Gammatddd[15]*x4 + Gammatddd[16]*x5 + Gammatddd[8]*x3 + dGammatu_dx[3]*gtdd[2] + dGammatu_dx[4]*gtdd[4] + dGammatu_dx[5]*gtdd[5] + dGammatu_dx[6]*gtdd[1] + dGammatu_dx[7]*gtdd[3] + dGammatu_dx[8]*gtdd[4]);
	(*Rgd)[0] += Gammatddd[0]*x3 + Gammatddd[1]*x4 + Gammatddd[2]*x5 + dGammatu_dx[0]*gtdd[0] + dGammatu_dx[1]*gtdd[1] + dGammatu_dx[2]*gtdd[2];
	(*Rgd)[1] += x6;
	(*Rgd)[2] += x7;
	(*Rgd)[3] += Gammatddd[10]*x5 + Gammatddd[7]*x3 + Gammatddd[9]*x4 + dGammatu_dx[3]*gtdd[1] + dGammatu_dx[4]*gtdd[3] + dGammatu_dx[5]*gtdd[4];
	(*Rgd)[4] += x8;
	(*Rgd)[5] += Gammatddd[14]*x3 + Gammatddd[16]*x4 + Gammatddd[17]*x5 + dGammatu_dx[6]*gtdd[2] + dGammatu_dx[7]*gtdd[4] + dGammatu_dx[8]*gtdd[5];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci3(
	const double gtuu[6],
	const double Gammatddd[18],
	double (*Rgg)[6]
)
{
	double x0 = 3*gtuu[3];
	double x1 = 3*gtuu[5];
	double x2 = 3*gtuu[0];
	(*Rgg)[0] += 3*Gammatddd[0]*gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + Gammatddd[1]*x0*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[2]*x1*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + gtuu[0]*(Gammatddd[12] + 2*Gammatddd[2])*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + gtuu[0]*(2*Gammatddd[1] + Gammatddd[6])*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + gtuu[1]*(Gammatddd[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + 2*Gammatddd[1]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[1]*(2*Gammatddd[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[1]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[1]*(Gammatddd[12]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + 2*Gammatddd[4]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4])) + gtuu[1]*(Gammatddd[13]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + 2*Gammatddd[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4])) + gtuu[1]*(2*Gammatddd[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[7]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3])) + gtuu[1]*(2*Gammatddd[3]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + Gammatddd[6]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3])) + gtuu[2]*(Gammatddd[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + 2*Gammatddd[2]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[2]*(2*Gammatddd[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[2]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[2]*(Gammatddd[12]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + 2*Gammatddd[5]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4])) + gtuu[2]*(Gammatddd[14]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + 2*Gammatddd[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[2]*(2*Gammatddd[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[8]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3])) + gtuu[2]*(2*Gammatddd[4]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + Gammatddd[6]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])) + gtuu[3]*(Gammatddd[13] + 2*Gammatddd[4])*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + gtuu[3]*(2*Gammatddd[3] + Gammatddd[7])*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + gtuu[4]*(Gammatddd[13]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + 2*Gammatddd[5]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4])) + gtuu[4]*(Gammatddd[14]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + 2*Gammatddd[4]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[4]*(Gammatddd[1]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + 2*Gammatddd[2]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[4]*(2*Gammatddd[1]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[2]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[4]*(2*Gammatddd[3]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[8]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3])) + gtuu[4]*(2*Gammatddd[4]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[7]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])) + gtuu[5]*(Gammatddd[14] + 2*Gammatddd[5])*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + gtuu[5]*(2*Gammatddd[4] + Gammatddd[8])*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]);
	(*Rgg)[1] += gtuu[0]*(Gammatddd[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + 2*Gammatddd[7]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3])) + gtuu[0]*(Gammatddd[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[1]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + Gammatddd[6]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[0]*(Gammatddd[13]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[8]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4])) + gtuu[1]*(2*Gammatddd[1]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[7]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[1]*(Gammatddd[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + Gammatddd[3]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + Gammatddd[6]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[1]*(Gammatddd[10]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[13]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[4]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4])) + gtuu[1]*(Gammatddd[15]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + Gammatddd[8]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4])) + gtuu[1]*(Gammatddd[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + Gammatddd[7]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[9]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3])) + gtuu[1]*(Gammatddd[3]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[7]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[9]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3])) + gtuu[2]*(Gammatddd[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[4]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + Gammatddd[6]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])) + gtuu[2]*(Gammatddd[10]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + Gammatddd[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + Gammatddd[7]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])) + gtuu[2]*(Gammatddd[10]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + Gammatddd[4]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[7]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])) + gtuu[2]*(Gammatddd[11]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[13]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[5]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4])) + gtuu[2]*(Gammatddd[16]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + Gammatddd[8]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[2]*(Gammatddd[1]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[2]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[8]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[3]*(Gammatddd[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + 2*Gammatddd[9]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3])) + gtuu[3]*(Gammatddd[10]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[15]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[4]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4])) + gtuu[3]*(Gammatddd[1]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + Gammatddd[3]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[7]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[4]*(Gammatddd[10]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[3]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + Gammatddd[9]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])) + gtuu[4]*(Gammatddd[10]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[4]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + Gammatddd[9]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])) + gtuu[4]*(Gammatddd[10]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[16]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[4]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[4]*(Gammatddd[11]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[15]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[5]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4])) + gtuu[4]*(Gammatddd[1]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[4]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[7]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])) + gtuu[4]*(Gammatddd[2]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + Gammatddd[3]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[8]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[5]*(2*Gammatddd[10]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[4]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[5]*(Gammatddd[11]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[16]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[5]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[5]*(Gammatddd[2]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[4]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[8]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]));
	(*Rgg)[2] += gtuu[0]*(2*Gammatddd[14]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[0]*(Gammatddd[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[12]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + Gammatddd[2]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[0]*(Gammatddd[13]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + Gammatddd[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[8]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3])) + gtuu[1]*(Gammatddd[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[12]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[4]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[1]*(Gammatddd[10]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + Gammatddd[13]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[1]*(Gammatddd[13]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + Gammatddd[1]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[2]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[1]*(Gammatddd[14]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[16]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[1]*(Gammatddd[14]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[16]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[4]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[1]*(Gammatddd[15]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + Gammatddd[3]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[8]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3])) + gtuu[2]*(Gammatddd[14]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + 2*Gammatddd[2]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])) + gtuu[2]*(Gammatddd[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + Gammatddd[12]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[5]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])) + gtuu[2]*(Gammatddd[11]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + Gammatddd[13]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + gtuu[2]*(Gammatddd[14]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[17]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) + gtuu[2]*(Gammatddd[14]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[17]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + Gammatddd[5]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[2]*(Gammatddd[16]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + Gammatddd[4]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[8]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])) + gtuu[3]*(2*Gammatddd[16]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[4]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[3]*(Gammatddd[10]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[15]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[3]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[3]*(Gammatddd[13]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[1]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[4]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[4]*(Gammatddd[10]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[16]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[4]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[4]*(Gammatddd[11]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[15]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[3]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + gtuu[4]*(Gammatddd[13]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[1]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + Gammatddd[5]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[4]*(Gammatddd[14]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[2]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[4]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])) + gtuu[4]*(Gammatddd[16]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[17]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[4]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) + gtuu[4]*(Gammatddd[16]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[17]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[5]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[5]*(2*Gammatddd[17]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) + gtuu[5]*(Gammatddd[11]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[16]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[4]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + gtuu[5]*(Gammatddd[14]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[2]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + Gammatddd[5]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]));
	(*Rgg)[3] += Gammatddd[10]*x1*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + Gammatddd[7]*x2*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + 3*Gammatddd[9]*gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + gtuu[0]*(Gammatddd[13] + 2*Gammatddd[8])*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + gtuu[0]*(Gammatddd[1] + 2*Gammatddd[6])*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + gtuu[1]*(2*Gammatddd[10]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[13]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4])) + gtuu[1]*(Gammatddd[15]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + 2*Gammatddd[8]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4])) + gtuu[1]*(Gammatddd[1]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + 2*Gammatddd[7]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[1]*(Gammatddd[3]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + 2*Gammatddd[6]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1])) + gtuu[1]*(Gammatddd[7]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + 2*Gammatddd[9]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3])) + gtuu[1]*(2*Gammatddd[7]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + Gammatddd[9]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3])) + gtuu[2]*(Gammatddd[10]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + 2*Gammatddd[7]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[2]*(2*Gammatddd[10]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[7]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[2]*(2*Gammatddd[11]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[13]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[2]*(Gammatddd[16]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + 2*Gammatddd[8]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[2]*(Gammatddd[1]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + 2*Gammatddd[8]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])) + gtuu[2]*(Gammatddd[4]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + 2*Gammatddd[6]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])) + gtuu[3]*(2*Gammatddd[10] + Gammatddd[15])*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + gtuu[3]*(Gammatddd[3] + 2*Gammatddd[7])*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + gtuu[4]*(2*Gammatddd[10]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + Gammatddd[16]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4])) + gtuu[4]*(Gammatddd[10]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + 2*Gammatddd[9]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[4]*(2*Gammatddd[10]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + Gammatddd[9]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[4]*(2*Gammatddd[11]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + Gammatddd[15]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[4]*(Gammatddd[3]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + 2*Gammatddd[8]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1])) + gtuu[4]*(Gammatddd[4]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + 2*Gammatddd[7]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])) + gtuu[5]*(2*Gammatddd[11] + Gammatddd[16])*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + gtuu[5]*(Gammatddd[4] + 2*Gammatddd[8])*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]);
	(*Rgg)[4] += gtuu[0]*(2*Gammatddd[14]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[8]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[0]*(Gammatddd[12]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[2]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[6]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])) + gtuu[0]*(Gammatddd[13]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[7]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[8]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3])) + gtuu[1]*(Gammatddd[10]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[13]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + Gammatddd[7]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[1]*(Gammatddd[10]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[14]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + Gammatddd[16]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4])) + gtuu[1]*(Gammatddd[12]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + Gammatddd[4]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[6]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])) + gtuu[1]*(Gammatddd[13]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[2]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + Gammatddd[7]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])) + gtuu[1]*(Gammatddd[14]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + Gammatddd[16]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[8]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[1]*(Gammatddd[15]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[8]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + Gammatddd[9]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])) + gtuu[2]*(Gammatddd[10]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[16]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[8]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[2]*(Gammatddd[11]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + Gammatddd[13]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + Gammatddd[7]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + gtuu[2]*(Gammatddd[11]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[14]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + Gammatddd[17]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4])) + gtuu[2]*(Gammatddd[12]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[5]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[6]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) + gtuu[2]*(Gammatddd[14]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + Gammatddd[17]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + Gammatddd[8]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) + gtuu[2]*(Gammatddd[14]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + Gammatddd[2]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[8]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])) + gtuu[3]*(Gammatddd[10]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + 2*Gammatddd[16]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4])) + gtuu[3]*(Gammatddd[10]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + Gammatddd[15]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + Gammatddd[9]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[3]*(Gammatddd[13]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + Gammatddd[4]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + Gammatddd[7]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])) + gtuu[4]*(2*Gammatddd[10]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + Gammatddd[16]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3])) + gtuu[4]*(Gammatddd[10]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + Gammatddd[16]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + Gammatddd[17]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4])) + gtuu[4]*(Gammatddd[11]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + Gammatddd[16]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + Gammatddd[17]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4])) + gtuu[4]*(Gammatddd[11]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + Gammatddd[15]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + Gammatddd[9]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + gtuu[4]*(Gammatddd[13]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[5]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + Gammatddd[7]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) + gtuu[4]*(Gammatddd[14]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + Gammatddd[4]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[8]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])) + gtuu[5]*(Gammatddd[11]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + 2*Gammatddd[17]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[5]*(Gammatddd[10]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + Gammatddd[11]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + Gammatddd[16]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[5]*(Gammatddd[14]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[5]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[8]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]));
	(*Rgg)[5] += Gammatddd[14]*x2*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + Gammatddd[16]*x0*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + 3*Gammatddd[17]*gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + gtuu[0]*(2*Gammatddd[12] + Gammatddd[2])*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + gtuu[0]*(2*Gammatddd[13] + Gammatddd[8])*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + gtuu[1]*(Gammatddd[10]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + 2*Gammatddd[13]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[1]*(2*Gammatddd[12]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[4]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])) + gtuu[1]*(2*Gammatddd[13]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[2]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])) + gtuu[1]*(Gammatddd[14]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + 2*Gammatddd[16]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[1]*(2*Gammatddd[14]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + Gammatddd[16]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[1]*(2*Gammatddd[15]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[8]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[2]*(Gammatddd[11]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + 2*Gammatddd[13]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + gtuu[2]*(2*Gammatddd[12]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + Gammatddd[5]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])) + gtuu[2]*(Gammatddd[14]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + 2*Gammatddd[17]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[2]*(2*Gammatddd[14]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + Gammatddd[17]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])) + gtuu[2]*(2*Gammatddd[14]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + Gammatddd[2]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) + gtuu[2]*(2*Gammatddd[16]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + Gammatddd[8]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + gtuu[3]*(Gammatddd[10] + 2*Gammatddd[15])*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + gtuu[3]*(2*Gammatddd[13] + Gammatddd[4])*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + gtuu[4]*(Gammatddd[10]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + 2*Gammatddd[16]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) + gtuu[4]*(Gammatddd[11]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + 2*Gammatddd[15]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + gtuu[4]*(2*Gammatddd[13]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + Gammatddd[5]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])) + gtuu[4]*(2*Gammatddd[14]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + Gammatddd[4]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) + gtuu[4]*(Gammatddd[16]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + 2*Gammatddd[17]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[4]*(2*Gammatddd[16]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + Gammatddd[17]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) + gtuu[5]*(Gammatddd[11] + 2*Gammatddd[16])*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + gtuu[5]*(2*Gammatddd[14] + Gammatddd[5])*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci4(
	const double gtdd[6],
	double W,
	const double gtuu[6],
	const double Gammatddd[18],
	const double dW_dx[3],
	const double ddW_dx2[6],
	double (*Rphi)[6]
)
{
	double x0 = 1/(W);
	double x1 = 2*x0;
	double x2 = ((dW_dx[0])*(dW_dx[0]))*x1;
	double x3 = -ddW_dx2[0];
	double x4 = ((dW_dx[1])*(dW_dx[1]))*x1;
	double x5 = -ddW_dx2[3];
	double x6 = ((dW_dx[2])*(dW_dx[2]))*x1;
	double x7 = -ddW_dx2[5];
	double x8 = 2*gtdd[0];
	double x9 = dW_dx[0]*x1;
	double x10 = dW_dx[1]*x9;
	double x11 = -ddW_dx2[1];
	double x12 = dW_dx[2]*x9;
	double x13 = -ddW_dx2[2];
	double x14 = dW_dx[1]*dW_dx[2]*x1;
	double x15 = -ddW_dx2[4];
	double x16 = 2*gtdd[1];
	double x17 = 2*gtdd[2];
	double x18 = 2*gtdd[3];
	double x19 = 2*gtdd[4];
	double x20 = 2*gtdd[5];
	(*Rphi)[0] += x0*(-dW_dx[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) - dW_dx[1]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) - dW_dx[2]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) - gtdd[0]*gtuu[0]*(dW_dx[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + dW_dx[1]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + dW_dx[2]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + x2 + x3) - gtdd[0]*gtuu[3]*(dW_dx[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + dW_dx[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + dW_dx[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + x4 + x5) - gtdd[0]*gtuu[5]*(dW_dx[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + dW_dx[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + dW_dx[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x6 + x7) - gtuu[1]*x8*(dW_dx[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + dW_dx[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + dW_dx[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x10 + x11) - gtuu[2]*x8*(dW_dx[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + dW_dx[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + dW_dx[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x12 + x13) - gtuu[4]*x8*(dW_dx[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + dW_dx[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + dW_dx[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + x14 + x15) - x3);
	(*Rphi)[1] += x0*(-dW_dx[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) - dW_dx[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) - dW_dx[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) - gtdd[1]*gtuu[0]*(dW_dx[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + dW_dx[1]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + dW_dx[2]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + x2 + x3) - gtdd[1]*gtuu[3]*(dW_dx[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + dW_dx[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + dW_dx[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + x4 + x5) - gtdd[1]*gtuu[5]*(dW_dx[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + dW_dx[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + dW_dx[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x6 + x7) - gtuu[1]*x16*(dW_dx[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + dW_dx[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + dW_dx[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x10 + x11) - gtuu[2]*x16*(dW_dx[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + dW_dx[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + dW_dx[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x12 + x13) - gtuu[4]*x16*(dW_dx[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + dW_dx[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + dW_dx[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + x14 + x15) - x11);
	(*Rphi)[2] += x0*(-dW_dx[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) - dW_dx[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) - dW_dx[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) - gtdd[2]*gtuu[0]*(dW_dx[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + dW_dx[1]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + dW_dx[2]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + x2 + x3) - gtdd[2]*gtuu[3]*(dW_dx[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + dW_dx[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + dW_dx[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + x4 + x5) - gtdd[2]*gtuu[5]*(dW_dx[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + dW_dx[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + dW_dx[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x6 + x7) - gtuu[1]*x17*(dW_dx[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + dW_dx[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + dW_dx[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x10 + x11) - gtuu[2]*x17*(dW_dx[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + dW_dx[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + dW_dx[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x12 + x13) - gtuu[4]*x17*(dW_dx[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + dW_dx[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + dW_dx[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + x14 + x15) - x13);
	(*Rphi)[3] += x0*(-dW_dx[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) - dW_dx[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) - dW_dx[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) - gtdd[3]*gtuu[0]*(dW_dx[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + dW_dx[1]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + dW_dx[2]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + x2 + x3) - gtdd[3]*gtuu[3]*(dW_dx[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + dW_dx[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + dW_dx[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + x4 + x5) - gtdd[3]*gtuu[5]*(dW_dx[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + dW_dx[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + dW_dx[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x6 + x7) - gtuu[1]*x18*(dW_dx[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + dW_dx[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + dW_dx[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x10 + x11) - gtuu[2]*x18*(dW_dx[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + dW_dx[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + dW_dx[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x12 + x13) - gtuu[4]*x18*(dW_dx[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + dW_dx[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + dW_dx[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + x14 + x15) - x5);
	(*Rphi)[4] += x0*(-dW_dx[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) - dW_dx[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) - dW_dx[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) - gtdd[4]*gtuu[0]*(dW_dx[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + dW_dx[1]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + dW_dx[2]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + x2 + x3) - gtdd[4]*gtuu[3]*(dW_dx[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + dW_dx[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + dW_dx[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + x4 + x5) - gtdd[4]*gtuu[5]*(dW_dx[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + dW_dx[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + dW_dx[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x6 + x7) - gtuu[1]*x19*(dW_dx[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + dW_dx[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + dW_dx[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x10 + x11) - gtuu[2]*x19*(dW_dx[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + dW_dx[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + dW_dx[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x12 + x13) - gtuu[4]*x19*(dW_dx[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + dW_dx[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + dW_dx[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + x14 + x15) - x15);
	(*Rphi)[5] += x0*(-dW_dx[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) - dW_dx[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) - dW_dx[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) - gtdd[5]*gtuu[0]*(dW_dx[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + dW_dx[1]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + dW_dx[2]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + x2 + x3) - gtdd[5]*gtuu[3]*(dW_dx[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + dW_dx[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + dW_dx[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + x4 + x5) - gtdd[5]*gtuu[5]*(dW_dx[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + dW_dx[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + dW_dx[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x6 + x7) - gtuu[1]*x20*(dW_dx[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + dW_dx[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + dW_dx[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x10 + x11) - gtuu[2]*x20*(dW_dx[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + dW_dx[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + dW_dx[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x12 + x13) - gtuu[4]*x20*(dW_dx[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + dW_dx[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + dW_dx[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + x14 + x15) - x7);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Ricci_trace(
	double W,
	const double gtuu[6],
	const double Rdd[6],
	double * __restrict__ Rtrace
)
{
	*Rtrace = ((W)*(W))*(Rdd[0]*gtuu[0] + 2*Rdd[1]*gtuu[1] + 2*Rdd[2]*gtuu[2] + Rdd[3]*gtuu[3] + 2*Rdd[4]*gtuu[4] + Rdd[5]*gtuu[5]);
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_chi_rhs(
	const double betau[3],
	double alp,
	double W,
	double theta,
	double Khat,
	const double dbetau_dx[9],
	const double dW_dx_upwind[3],
	double * __restrict__ dW
)
{
	double x0 = (1.0/3.0)*W;
	*dW = alp*x0*(Khat + 2*theta) + betau[0]*dW_dx_upwind[0] + betau[1]*dW_dx_upwind[1] + betau[2]*dW_dx_upwind[2] - dbetau_dx[0]*x0 - dbetau_dx[4]*x0 - dbetau_dx[8]*x0;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_gtdd_rhs(
	const double gtdd[6],
	const double Atdd[6],
	const double betau[3],
	double alp,
	const double dgtdd_dx_upwind[18],
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
	double x6 = -Atdd[1]*x0 + betau[0]*dgtdd_dx_upwind[1] + betau[1]*dgtdd_dx_upwind[7] + betau[2]*dgtdd_dx_upwind[13] + dbetau_dx[0]*x4 + dbetau_dx[1]*gtdd[3] + dbetau_dx[2]*gtdd[4] + dbetau_dx[3]*gtdd[0] + dbetau_dx[4]*x4 + dbetau_dx[5]*gtdd[2] - gtdd[1]*x5;
	double x7 = (1.0/3.0)*gtdd[2];
	double x8 = (2.0/3.0)*dbetau_dx[4];
	double x9 = -Atdd[2]*x0 + betau[0]*dgtdd_dx_upwind[2] + betau[1]*dgtdd_dx_upwind[8] + betau[2]*dgtdd_dx_upwind[14] + dbetau_dx[0]*x7 + dbetau_dx[1]*gtdd[4] + dbetau_dx[2]*gtdd[5] + dbetau_dx[6]*gtdd[0] + dbetau_dx[7]*gtdd[1] + dbetau_dx[8]*x7 - gtdd[2]*x8;
	double x10 = (2.0/3.0)*dbetau_dx[0];
	double x11 = 2*gtdd[4];
	double x12 = (1.0/3.0)*gtdd[4];
	double x13 = -Atdd[4]*x0 + betau[0]*dgtdd_dx_upwind[4] + betau[1]*dgtdd_dx_upwind[10] + betau[2]*dgtdd_dx_upwind[16] + dbetau_dx[3]*gtdd[2] + dbetau_dx[4]*x12 + dbetau_dx[5]*gtdd[5] + dbetau_dx[6]*gtdd[1] + dbetau_dx[7]*gtdd[3] + dbetau_dx[8]*x12 - gtdd[4]*x10;
	(*dgtdd_dt)[0] = -Atdd[0]*x0 + betau[0]*dgtdd_dx_upwind[0] + betau[1]*dgtdd_dx_upwind[6] + betau[2]*dgtdd_dx_upwind[12] + (4.0/3.0)*dbetau_dx[0]*gtdd[0] + dbetau_dx[1]*x1 + dbetau_dx[2]*x2 - dbetau_dx[4]*x3 - dbetau_dx[8]*x3;
	(*dgtdd_dt)[1] = x6;
	(*dgtdd_dt)[2] = x9;
	(*dgtdd_dt)[3] = -Atdd[3]*x0 + betau[0]*dgtdd_dx_upwind[3] + betau[1]*dgtdd_dx_upwind[9] + betau[2]*dgtdd_dx_upwind[15] + dbetau_dx[3]*x1 + (4.0/3.0)*dbetau_dx[4]*gtdd[3] + dbetau_dx[5]*x11 - gtdd[3]*x10 - gtdd[3]*x5;
	(*dgtdd_dt)[4] = x13;
	(*dgtdd_dt)[5] = -Atdd[5]*x0 + betau[0]*dgtdd_dx_upwind[5] + betau[1]*dgtdd_dx_upwind[11] + betau[2]*dgtdd_dx_upwind[17] + dbetau_dx[6]*x2 + dbetau_dx[7]*x11 + (4.0/3.0)*dbetau_dx[8]*gtdd[5] - gtdd[5]*x10 - gtdd[5]*x8;
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Khat_rhs(
	const double Atdd[6],
	const double betau[3],
	double alp,
	double theta,
	double Ktr,
	double S,
	double rho,
	double kappa1,
	double kappa2,
	const double gtuu[6],
	double DiDialp,
	const double dKhat_dx_upwind[3],
	double * __restrict__ dKhat_dt
)
{
	double x0 = Atdd[1]*gtuu[1];
	double x1 = Atdd[2]*gtuu[2];
	double x2 = Atdd[0]*gtuu[0] + x0 + x1;
	double x3 = Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2];
	double x4 = Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2];
	double x5 = Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4];
	double x6 = Atdd[4]*gtuu[4];
	double x7 = Atdd[3]*gtuu[3] + x0 + x6;
	double x8 = Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4];
	double x9 = 3*Atdd[1];
	double x10 = Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5];
	double x11 = Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5];
	double x12 = Atdd[5]*gtuu[5] + x1 + x6;
	double x13 = 3*Atdd[2];
	double x14 = 3*Atdd[4];
	*dKhat_dt = -DiDialp - alp*kappa1*theta*(kappa2 - 1) + 4*M_PI*alp*(S + rho) + (1.0/3.0)*alp*(3*Atdd[0]*(gtuu[0]*x2 + gtuu[1]*x3 + gtuu[2]*x4) + 3*Atdd[3]*(gtuu[1]*x5 + gtuu[3]*x7 + gtuu[4]*x8) + 3*Atdd[5]*(gtuu[2]*x10 + gtuu[4]*x11 + gtuu[5]*x12) + ((Ktr)*(Ktr)) + x13*(gtuu[0]*x10 + gtuu[1]*x11 + gtuu[2]*x12) + x13*(gtuu[2]*x2 + gtuu[4]*x3 + gtuu[5]*x4) + x14*(gtuu[1]*x10 + gtuu[3]*x11 + gtuu[4]*x12) + x14*(gtuu[2]*x5 + gtuu[4]*x7 + gtuu[5]*x8) + x9*(gtuu[0]*x5 + gtuu[1]*x7 + gtuu[2]*x8) + x9*(gtuu[1]*x2 + gtuu[3]*x3 + gtuu[4]*x4)) + betau[0]*dKhat_dx_upwind[0] + betau[1]*dKhat_dx_upwind[1] + betau[2]*dKhat_dx_upwind[2];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Gammatilde_rhs(
	const double Atdd[6],
	const double betau[3],
	double alp,
	double W,
	const double Gammatu[3],
	const double Si[3],
	double kappa1,
	const double gtuu[6],
	const double Gammatddd[18],
	const double dbetau_dx[9],
	const double dGammatu_dx_upwind[9],
	const double dKhat_dx[3],
	const double dW_dx[3],
	const double dalp_dx[3],
	const double dtheta_dx[3],
	const double ddbetau_dx2[18],
	double (*dGammatu_dt)[3]
)
{
	double x0 = 16*M_PI*alp;
	double x1 = Si[0]*x0;
	double x2 = Si[1]*x0;
	double x3 = Si[2]*x0;
	double x4 = (2.0/3.0)*alp;
	double x5 = x4*(2*dKhat_dx[0] + dtheta_dx[0]);
	double x6 = x4*(2*dKhat_dx[1] + dtheta_dx[1]);
	double x7 = x4*(2*dKhat_dx[2] + dtheta_dx[2]);
	double x8 = 2*dalp_dx[0];
	double x9 = 2*dalp_dx[1];
	double x10 = 2*dalp_dx[2];
	double x11 = 6*alp/W;
	double x12 = dW_dx[0]*x11;
	double x13 = dW_dx[1]*x11;
	double x14 = dW_dx[2]*x11;
	double x15 = 2*alp;
	double x16 = kappa1*x15;
	(*dGammatu_dt)[0] = betau[0]*dGammatu_dx_upwind[0] + betau[1]*dGammatu_dx_upwind[3] + betau[2]*dGammatu_dx_upwind[6] - 1.0/3.0*dbetau_dx[0]*(gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) - dbetau_dx[3]*(gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + dbetau_dx[4]*((2.0/3.0)*gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + (4.0/3.0)*gtuu[1]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + (4.0/3.0)*gtuu[2]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + (2.0/3.0)*gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + (4.0/3.0)*gtuu[4]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + (2.0/3.0)*gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) - dbetau_dx[6]*(gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) + dbetau_dx[8]*((2.0/3.0)*gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + (4.0/3.0)*gtuu[1]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + (4.0/3.0)*gtuu[2]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + (2.0/3.0)*gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + (4.0/3.0)*gtuu[4]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + (2.0/3.0)*gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) + (4.0/3.0)*ddbetau_dx2[0]*gtuu[0] + (1.0/3.0)*ddbetau_dx2[10]*gtuu[1] + 2*ddbetau_dx2[12]*gtuu[4] + (1.0/3.0)*ddbetau_dx2[13]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[14]*gtuu[1] + ddbetau_dx2[15]*gtuu[5] + (1.0/3.0)*ddbetau_dx2[17]*gtuu[2] + (7.0/3.0)*ddbetau_dx2[3]*gtuu[1] + (1.0/3.0)*ddbetau_dx2[4]*gtuu[0] + (7.0/3.0)*ddbetau_dx2[6]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[8]*gtuu[0] + ddbetau_dx2[9]*gtuu[3] - gtuu[0]*x1 - gtuu[0]*x5 - gtuu[1]*x2 - gtuu[1]*x6 - gtuu[2]*x3 - gtuu[2]*x7 - x10*(gtuu[2]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[4]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[5]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) - x12*(gtuu[0]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[1]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[2]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) - x13*(gtuu[1]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[3]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[4]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) - x14*(gtuu[2]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[4]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[5]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1])*(gtuu[0]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[1]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[2]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])*(gtuu[1]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[3]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[4]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])*(gtuu[2]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[4]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[5]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x15*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])*(gtuu[2]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[4]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[5]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])*(gtuu[0]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[1]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[2]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x15*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1])*(gtuu[1]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[3]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[4]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])*(gtuu[0]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[1]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[2]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1])*(gtuu[2]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[4]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[5]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1])*(gtuu[1]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[3]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[4]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x16*(-Gammatu[0] + gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) - x8*(gtuu[0]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[1]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[2]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) - x9*(gtuu[1]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[3]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[4]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2]));
	(*dGammatu_dt)[1] = betau[0]*dGammatu_dx_upwind[1] + betau[1]*dGammatu_dx_upwind[4] + betau[2]*dGammatu_dx_upwind[7] + dbetau_dx[0]*((2.0/3.0)*gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + (4.0/3.0)*gtuu[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + (4.0/3.0)*gtuu[2]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + (2.0/3.0)*gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + (4.0/3.0)*gtuu[4]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + (2.0/3.0)*gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) - dbetau_dx[1]*(gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) - 1.0/3.0*dbetau_dx[4]*(gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) - dbetau_dx[7]*(gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) + dbetau_dx[8]*((2.0/3.0)*gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + (4.0/3.0)*gtuu[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + (4.0/3.0)*gtuu[2]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + (2.0/3.0)*gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + (4.0/3.0)*gtuu[4]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + (2.0/3.0)*gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) + (1.0/3.0)*ddbetau_dx2[0]*gtuu[1] + (4.0/3.0)*ddbetau_dx2[10]*gtuu[3] + (7.0/3.0)*ddbetau_dx2[13]*gtuu[4] + (1.0/3.0)*ddbetau_dx2[14]*gtuu[3] + ddbetau_dx2[16]*gtuu[5] + (1.0/3.0)*ddbetau_dx2[17]*gtuu[4] + ddbetau_dx2[1]*gtuu[0] + (1.0/3.0)*ddbetau_dx2[3]*gtuu[3] + (7.0/3.0)*ddbetau_dx2[4]*gtuu[1] + (1.0/3.0)*ddbetau_dx2[6]*gtuu[4] + 2*ddbetau_dx2[7]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[8]*gtuu[1] - gtuu[1]*x1 - gtuu[1]*x5 - gtuu[3]*x2 - gtuu[3]*x6 - gtuu[4]*x3 - gtuu[4]*x7 - x10*(gtuu[2]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[4]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[5]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) - x12*(gtuu[0]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[1]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[2]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) - x13*(gtuu[1]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[3]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[4]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) - x14*(gtuu[2]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[4]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[5]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x15*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3])*(gtuu[0]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[1]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[2]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])*(gtuu[1]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[3]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[4]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])*(gtuu[2]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[4]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[5]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x15*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])*(gtuu[2]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[4]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[5]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3])*(gtuu[0]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[1]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[2]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x15*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3])*(gtuu[1]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[3]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[4]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])*(gtuu[0]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[1]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[2]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3])*(gtuu[2]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[4]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[5]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3])*(gtuu[1]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[3]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[4]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x16*(-Gammatu[1] + gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) - x8*(gtuu[0]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[1]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[2]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) - x9*(gtuu[1]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[3]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[4]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4]));
	(*dGammatu_dt)[2] = betau[0]*dGammatu_dx_upwind[2] + betau[1]*dGammatu_dx_upwind[5] + betau[2]*dGammatu_dx_upwind[8] + dbetau_dx[0]*((2.0/3.0)*gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + (4.0/3.0)*gtuu[1]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + (4.0/3.0)*gtuu[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + (2.0/3.0)*gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + (4.0/3.0)*gtuu[4]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + (2.0/3.0)*gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) - dbetau_dx[2]*(gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0])) + dbetau_dx[4]*((2.0/3.0)*gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + (4.0/3.0)*gtuu[1]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + (4.0/3.0)*gtuu[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + (2.0/3.0)*gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + (4.0/3.0)*gtuu[4]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + (2.0/3.0)*gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) - dbetau_dx[5]*(gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1])) - 1.0/3.0*dbetau_dx[8]*(gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) + (1.0/3.0)*ddbetau_dx2[0]*gtuu[2] + (1.0/3.0)*ddbetau_dx2[10]*gtuu[4] + ddbetau_dx2[11]*gtuu[3] + (1.0/3.0)*ddbetau_dx2[13]*gtuu[5] + (7.0/3.0)*ddbetau_dx2[14]*gtuu[4] + (4.0/3.0)*ddbetau_dx2[17]*gtuu[5] + ddbetau_dx2[2]*gtuu[0] + (1.0/3.0)*ddbetau_dx2[3]*gtuu[4] + (1.0/3.0)*ddbetau_dx2[4]*gtuu[2] + 2*ddbetau_dx2[5]*gtuu[1] + (1.0/3.0)*ddbetau_dx2[6]*gtuu[5] + (7.0/3.0)*ddbetau_dx2[8]*gtuu[2] - gtuu[2]*x1 - gtuu[2]*x5 - gtuu[4]*x2 - gtuu[4]*x6 - gtuu[5]*x3 - gtuu[5]*x7 - x10*(gtuu[2]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[4]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[5]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) - x12*(gtuu[0]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[1]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[2]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) - x13*(gtuu[1]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[3]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[4]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) - x14*(gtuu[2]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[4]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[5]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4])*(gtuu[0]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[1]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[2]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])*(gtuu[1]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[3]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[4]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])*(gtuu[2]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[4]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[5]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x15*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])*(gtuu[2]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[4]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[5]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4])*(gtuu[0]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[1]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[2]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x15*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4])*(gtuu[1]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[3]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[4]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])*(gtuu[0]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[1]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[2]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) + x15*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4])*(gtuu[2]*(Atdd[0]*gtuu[0] + Atdd[1]*gtuu[1] + Atdd[2]*gtuu[2]) + gtuu[4]*(Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2]) + gtuu[5]*(Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2])) + x15*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4])*(gtuu[1]*(Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4]) + gtuu[3]*(Atdd[1]*gtuu[1] + Atdd[3]*gtuu[3] + Atdd[4]*gtuu[4]) + gtuu[4]*(Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4])) + x16*(-Gammatu[2] + gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + 2*gtuu[1]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + 2*gtuu[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + 2*gtuu[4]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2])) - x8*(gtuu[0]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[1]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[2]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5])) - x9*(gtuu[1]*(Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5]) + gtuu[3]*(Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5]) + gtuu[4]*(Atdd[2]*gtuu[2] + Atdd[4]*gtuu[4] + Atdd[5]*gtuu[5]));
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_theta_rhs(
	const double Atdd[6],
	const double betau[3],
	double alp,
	double theta,
	double Khat,
	double rho,
	double kappa1,
	double kappa2,
	const double gtuu[6],
	double Rtrace,
	const double dtheta_dx_upwind[3],
	double * __restrict__ dtheta_dt
)
{
	double x0 = Atdd[1]*gtuu[1];
	double x1 = Atdd[2]*gtuu[2];
	double x2 = Atdd[0]*gtuu[0] + x0 + x1;
	double x3 = Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + Atdd[4]*gtuu[2];
	double x4 = Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + Atdd[5]*gtuu[2];
	double x5 = Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + Atdd[2]*gtuu[4];
	double x6 = Atdd[4]*gtuu[4];
	double x7 = Atdd[3]*gtuu[3] + x0 + x6;
	double x8 = Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + Atdd[5]*gtuu[4];
	double x9 = 3*Atdd[1];
	double x10 = Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + Atdd[2]*gtuu[5];
	double x11 = Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + Atdd[4]*gtuu[5];
	double x12 = Atdd[5]*gtuu[5] + x1 + x6;
	double x13 = 3*Atdd[2];
	double x14 = 3*Atdd[4];
	*dtheta_dt = -alp*(kappa1*theta*(kappa2 + 2) + 8*M_PI*rho) - 1.0/6.0*alp*(3*Atdd[0]*(gtuu[0]*x2 + gtuu[1]*x3 + gtuu[2]*x4) + 3*Atdd[3]*(gtuu[1]*x5 + gtuu[3]*x7 + gtuu[4]*x8) + 3*Atdd[5]*(gtuu[2]*x10 + gtuu[4]*x11 + gtuu[5]*x12) - 3*Rtrace + x13*(gtuu[0]*x10 + gtuu[1]*x11 + gtuu[2]*x12) + x13*(gtuu[2]*x2 + gtuu[4]*x3 + gtuu[5]*x4) + x14*(gtuu[1]*x10 + gtuu[3]*x11 + gtuu[4]*x12) + x14*(gtuu[2]*x5 + gtuu[4]*x7 + gtuu[5]*x8) + x9*(gtuu[0]*x5 + gtuu[1]*x7 + gtuu[2]*x8) + x9*(gtuu[1]*x2 + gtuu[3]*x3 + gtuu[4]*x4) - 2*((Khat + 2*theta)*(Khat + 2*theta))) + betau[0]*dtheta_dx_upwind[0] + betau[1]*dtheta_dx_upwind[1] + betau[2]*dtheta_dx_upwind[2];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Atdd_rhs(
	const double gtdd[6],
	const double Atdd[6],
	const double betau[3],
	double alp,
	double W,
	double Ktr,
	double S,
	const double Sij[6],
	const double gtuu[6],
	const double DiDjalp[6],
	double DiDialp,
	const double Rdd[6],
	double Rtrace,
	const double dAtdd_dx_upwind[18],
	const double dbetau_dx[9],
	double (*dAtdd_dt)[6]
)
{
	double x0 = (2.0/3.0)*Atdd[0];
	double x1 = 2*Atdd[1];
	double x2 = 2*Atdd[2];
	double x3 = Ktr*alp;
	double x4 = alp*gtuu[2];
	double x5 = x2*x4;
	double x6 = Atdd[2]*x1;
	double x7 = alp*gtuu[4];
	double x8 = 2*alp;
	double x9 = ((Atdd[2])*(Atdd[2]))*x8;
	double x10 = 8*M_PI;
	double x11 = (1.0/3.0)*DiDialp - 1.0/3.0*alp*(Rtrace - S*x10);
	double x12 = ((W)*(W));
	double x13 = (1.0/3.0)*Atdd[1];
	double x14 = (2.0/3.0)*dbetau_dx[8];
	double x15 = x2*x7;
	double x16 = alp*gtuu[5];
	double x17 = x16*x2;
	double x18 = Atdd[0]*dbetau_dx[3] - Atdd[1]*x14 + Atdd[1]*x3 + Atdd[2]*dbetau_dx[5] + Atdd[3]*dbetau_dx[1] - Atdd[3]*x15 + Atdd[4]*dbetau_dx[2] - Atdd[4]*x17 + betau[0]*dAtdd_dx_upwind[1] + betau[1]*dAtdd_dx_upwind[7] + betau[2]*dAtdd_dx_upwind[13] + dbetau_dx[0]*x13 + dbetau_dx[4]*x13 + gtdd[1]*x11 - x12*(DiDjalp[1] - alp*(Rdd[1] - Sij[1]*x10)) - x4*x6;
	double x19 = (1.0/3.0)*Atdd[2];
	double x20 = (2.0/3.0)*dbetau_dx[4];
	double x21 = Atdd[0]*dbetau_dx[6] + Atdd[1]*dbetau_dx[7] - Atdd[2]*x20 + Atdd[2]*x3 + Atdd[4]*dbetau_dx[1] - Atdd[4]*x15 + Atdd[5]*dbetau_dx[2] - Atdd[5]*x17 + betau[0]*dAtdd_dx_upwind[2] + betau[1]*dAtdd_dx_upwind[8] + betau[2]*dAtdd_dx_upwind[14] + dbetau_dx[0]*x19 + dbetau_dx[8]*x19 + gtdd[2]*x11 - gtuu[2]*x9 - x12*(DiDjalp[2] - alp*(Rdd[2] - Sij[2]*x10));
	double x22 = (2.0/3.0)*dbetau_dx[0];
	double x23 = 2*Atdd[4];
	double x24 = x23*x7;
	double x25 = ((Atdd[4])*(Atdd[4]))*x8;
	double x26 = (1.0/3.0)*Atdd[4];
	double x27 = Atdd[1]*dbetau_dx[6] + Atdd[2]*dbetau_dx[3] + Atdd[3]*dbetau_dx[7] - Atdd[4]*x22 + Atdd[4]*x3 - Atdd[4]*x5 + Atdd[5]*dbetau_dx[5] - Atdd[5]*x16*x23 + betau[0]*dAtdd_dx_upwind[4] + betau[1]*dAtdd_dx_upwind[10] + betau[2]*dAtdd_dx_upwind[16] + dbetau_dx[4]*x26 + dbetau_dx[8]*x26 + gtdd[4]*x11 - gtuu[4]*x25 - x12*(DiDjalp[4] - alp*(Rdd[4] - Sij[4]*x10));
	(*dAtdd_dt)[0] = (4.0/3.0)*Atdd[0]*dbetau_dx[0] + Atdd[0]*x3 - Atdd[0]*x5 + betau[0]*dAtdd_dx_upwind[0] + betau[1]*dAtdd_dx_upwind[6] + betau[2]*dAtdd_dx_upwind[12] + dbetau_dx[1]*x1 + dbetau_dx[2]*x2 - dbetau_dx[4]*x0 - dbetau_dx[8]*x0 + gtdd[0]*x11 - gtuu[5]*x9 - x12*(DiDjalp[0] - alp*(Rdd[0] - Sij[0]*x10)) - x6*x7;
	(*dAtdd_dt)[1] = x18;
	(*dAtdd_dt)[2] = x21;
	(*dAtdd_dt)[3] = (4.0/3.0)*Atdd[3]*dbetau_dx[4] - Atdd[3]*x14 - Atdd[3]*x22 - Atdd[3]*x24 + Atdd[3]*x3 - Atdd[4]*x1*x4 + betau[0]*dAtdd_dx_upwind[3] + betau[1]*dAtdd_dx_upwind[9] + betau[2]*dAtdd_dx_upwind[15] + dbetau_dx[3]*x1 + dbetau_dx[5]*x23 + gtdd[3]*x11 - gtuu[5]*x25 - x12*(DiDjalp[3] - alp*(Rdd[3] - Sij[3]*x10));
	(*dAtdd_dt)[4] = x27;
	(*dAtdd_dt)[5] = -((Atdd[5])*(Atdd[5]))*gtuu[5]*x8 + (4.0/3.0)*Atdd[5]*dbetau_dx[8] - Atdd[5]*x20 - Atdd[5]*x22 - Atdd[5]*x24 + Atdd[5]*x3 - Atdd[5]*x5 + betau[0]*dAtdd_dx_upwind[5] + betau[1]*dAtdd_dx_upwind[11] + betau[2]*dAtdd_dx_upwind[17] + dbetau_dx[6]*x2 + dbetau_dx[7]*x23 + gtdd[5]*x11 - x12*(DiDjalp[5] - alp*(Rdd[5] - Sij[5]*x10));
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_alpha_rhs(
	const double betau[3],
	double alp,
	double Khat,
	const double dalp_dx_upwind[3],
	double * __restrict__ dalpha_dt
)
{
	*dalpha_dt = -2*Khat*alp + betau[0]*dalp_dx_upwind[0] + betau[1]*dalp_dx_upwind[1] + betau[2]*dalp_dx_upwind[2];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_beta_rhs(
	const double betau[3],
	const double Bdriver[3],
	const double dbetau_dx_upwind[9],
	double (*dbeta_dt)[3]
)
{
	(*dbeta_dt)[0] = (3.0/4.0)*Bdriver[0] + betau[0]*dbetau_dx_upwind[0] + betau[1]*dbetau_dx_upwind[3] + betau[2]*dbetau_dx_upwind[6];
	(*dbeta_dt)[1] = (3.0/4.0)*Bdriver[1] + betau[0]*dbetau_dx_upwind[1] + betau[1]*dbetau_dx_upwind[4] + betau[2]*dbetau_dx_upwind[7];
	(*dbeta_dt)[2] = (3.0/4.0)*Bdriver[2] + betau[0]*dbetau_dx_upwind[2] + betau[1]*dbetau_dx_upwind[5] + betau[2]*dbetau_dx_upwind[8];
}

static void KOKKOS_INLINE_FUNCTION
z4c_get_Bdriver_rhs(
	const double betau[3],
	const double Bdriver[3],
	double eta,
	const double dGammatu_dt[3],
	const double dBdriver_dx_upwind[9],
	const double dGammatu_dx_upwind[9],
	double (*dBd_dt)[3]
)
{
	(*dBd_dt)[0] = -Bdriver[0]*eta + betau[0]*(dBdriver_dx_upwind[0] - dGammatu_dx_upwind[0]) + betau[1]*(dBdriver_dx_upwind[3] - dGammatu_dx_upwind[3]) + betau[2]*(dBdriver_dx_upwind[6] - dGammatu_dx_upwind[6]) + dGammatu_dt[0];
	(*dBd_dt)[1] = -Bdriver[1]*eta + betau[0]*(dBdriver_dx_upwind[1] - dGammatu_dx_upwind[1]) + betau[1]*(dBdriver_dx_upwind[4] - dGammatu_dx_upwind[4]) + betau[2]*(dBdriver_dx_upwind[7] - dGammatu_dx_upwind[7]) + dGammatu_dt[1];
	(*dBd_dt)[2] = -Bdriver[2]*eta + betau[0]*(dBdriver_dx_upwind[2] - dGammatu_dx_upwind[2]) + betau[1]*(dBdriver_dx_upwind[5] - dGammatu_dx_upwind[5]) + betau[2]*(dBdriver_dx_upwind[8] - dGammatu_dx_upwind[8]) + dGammatu_dt[2];
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
	const double Gammatddd[18],
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
	double x0 = Atdd[1]*gtuu[1];
	double x1 = Atdd[2]*gtuu[2];
	double x2 = Atdd[0]*gtuu[0] + x0 + x1;
	double x3 = Atdd[4]*gtuu[2];
	double x4 = Atdd[1]*gtuu[0] + Atdd[3]*gtuu[1] + x3;
	double x5 = Atdd[5]*gtuu[2];
	double x6 = Atdd[2]*gtuu[0] + Atdd[4]*gtuu[1] + x5;
	double x7 = gtuu[0]*x2 + gtuu[1]*x4 + gtuu[2]*x6;
	double x8 = Atdd[2]*gtuu[4];
	double x9 = Atdd[0]*gtuu[1] + Atdd[1]*gtuu[3] + x8;
	double x10 = Atdd[4]*gtuu[4];
	double x11 = Atdd[3]*gtuu[3] + x0 + x10;
	double x12 = Atdd[5]*gtuu[4];
	double x13 = Atdd[2]*gtuu[1] + Atdd[4]*gtuu[3] + x12;
	double x14 = gtuu[0]*x9 + gtuu[1]*x11 + gtuu[2]*x13;
	double x15 = gtuu[1]*x2 + gtuu[3]*x4 + gtuu[4]*x6;
	double x16 = Atdd[2]*gtuu[5];
	double x17 = Atdd[0]*gtuu[2] + Atdd[1]*gtuu[4] + x16;
	double x18 = Atdd[4]*gtuu[5];
	double x19 = Atdd[1]*gtuu[2] + Atdd[3]*gtuu[4] + x18;
	double x20 = Atdd[5]*gtuu[5];
	double x21 = x1 + x10 + x20;
	double x22 = gtuu[0]*x17 + gtuu[1]*x19 + gtuu[2]*x21;
	double x23 = gtuu[2]*x2 + gtuu[4]*x4 + gtuu[5]*x6;
	double x24 = gtuu[1]*x9 + gtuu[3]*x11 + gtuu[4]*x13;
	double x25 = gtuu[1]*x17 + gtuu[3]*x19 + gtuu[4]*x21;
	double x26 = gtuu[2]*x9 + gtuu[4]*x11 + gtuu[5]*x13;
	double x27 = gtuu[2]*x17 + gtuu[4]*x19 + gtuu[5]*x21;
	double x28 = 8*M_PI;
	double x29 = Si[0]*x28;
	double x30 = Si[1]*x28;
	double x31 = Si[2]*x28;
	double x32 = ((gtuu[2])*(gtuu[2]));
	double x33 = ((gtuu[1])*(gtuu[1]));
	double x34 = (2.0/3.0)*(dKhat_dx[0] + 2*dtheta_dx[0]);
	double x35 = (2.0/3.0)*(dKhat_dx[1] + 2*dtheta_dx[1]);
	double x36 = (2.0/3.0)*(dKhat_dx[2] + 2*dtheta_dx[2]);
	double x37 = dgtdd_dx[0]*x7;
	double x38 = dgtdd_dx[10]*x26;
	double x39 = dgtdd_dx[10]*x24;
	double x40 = dgtdd_dx[11]*x26;
	double x41 = dgtdd_dx[12]*x22;
	double x42 = dgtdd_dx[13]*x25;
	double x43 = dgtdd_dx[13]*x22;
	double x44 = dgtdd_dx[14]*x27;
	double x45 = dgtdd_dx[14]*x22;
	double x46 = dgtdd_dx[15]*x25;
	double x47 = dgtdd_dx[16]*x27;
	double x48 = dgtdd_dx[16]*x25;
	double x49 = dgtdd_dx[17]*x27;
	double x50 = dgtdd_dx[1]*x15;
	double x51 = dgtdd_dx[1]*x7;
	double x52 = dgtdd_dx[2]*x23;
	double x53 = dgtdd_dx[2]*x7;
	double x54 = dgtdd_dx[3]*x15;
	double x55 = dgtdd_dx[4]*x23;
	double x56 = dgtdd_dx[4]*x15;
	double x57 = dgtdd_dx[5]*x23;
	double x58 = dgtdd_dx[6]*x14;
	double x59 = dgtdd_dx[7]*x24;
	double x60 = dgtdd_dx[7]*x14;
	double x61 = dgtdd_dx[8]*x26;
	double x62 = dgtdd_dx[8]*x14;
	double x63 = dgtdd_dx[9]*x24;
	double x64 = 3/W;
	double x65 = dW_dx[0]*x64;
	double x66 = dW_dx[1]*x64;
	double x67 = dW_dx[2]*x64;
	double x68 = 2*gtuu[1];
	double x69 = 2*gtuu[2];
	double x70 = 2*gtuu[4];
	double x71 = ((gtuu[4])*(gtuu[4]));
	*H = -Atdd[0]*x7 - Atdd[1]*x14 - Atdd[1]*x15 - Atdd[2]*x22 - Atdd[2]*x23 - Atdd[3]*x24 - Atdd[4]*x25 - Atdd[4]*x26 - Atdd[5]*x27 + Rtrace - 16*M_PI*rho + (2.0/3.0)*((Khat + 2*theta)*(Khat + 2*theta));
	(*M)[0] = dAtdd_dx[0]*((gtuu[0])*(gtuu[0])) + dAtdd_dx[10]*gtuu[1]*gtuu[4] + dAtdd_dx[10]*gtuu[2]*gtuu[3] + dAtdd_dx[11]*gtuu[2]*gtuu[4] + dAtdd_dx[12]*gtuu[0]*gtuu[2] + dAtdd_dx[13]*gtuu[0]*gtuu[4] + dAtdd_dx[13]*gtuu[1]*gtuu[2] + dAtdd_dx[14]*gtuu[0]*gtuu[5] + dAtdd_dx[14]*x32 + dAtdd_dx[15]*gtuu[1]*gtuu[4] + dAtdd_dx[16]*gtuu[1]*gtuu[5] + dAtdd_dx[16]*gtuu[2]*gtuu[4] + dAtdd_dx[17]*gtuu[2]*gtuu[5] + 2*dAtdd_dx[1]*gtuu[0]*gtuu[1] + 2*dAtdd_dx[2]*gtuu[0]*gtuu[2] + dAtdd_dx[3]*x33 + 2*dAtdd_dx[4]*gtuu[1]*gtuu[2] + dAtdd_dx[5]*x32 + dAtdd_dx[6]*gtuu[0]*gtuu[1] + dAtdd_dx[7]*gtuu[0]*gtuu[3] + dAtdd_dx[7]*x33 + dAtdd_dx[8]*gtuu[0]*gtuu[4] + dAtdd_dx[8]*gtuu[1]*gtuu[2] + dAtdd_dx[9]*gtuu[1]*gtuu[3] - gtuu[0]*x29 - gtuu[0]*x34 - gtuu[0]*x37 - gtuu[0]*x41 - gtuu[0]*x42 - gtuu[0]*x44 - gtuu[0]*x50 - gtuu[0]*x52 - gtuu[0]*x58 - gtuu[0]*x59 - gtuu[0]*x61 - gtuu[1]*x30 - gtuu[1]*x35 - gtuu[1]*x38 - gtuu[1]*x43 - gtuu[1]*x46 - gtuu[1]*x47 - gtuu[1]*x51 - gtuu[1]*x54 - gtuu[1]*x55 - gtuu[1]*x60 - gtuu[1]*x63 - gtuu[2]*x31 - gtuu[2]*x36 - gtuu[2]*x39 - gtuu[2]*x40 - gtuu[2]*x45 - gtuu[2]*x48 - gtuu[2]*x49 - gtuu[2]*x53 - gtuu[2]*x56 - gtuu[2]*x57 - gtuu[2]*x62 - x1*(gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + x68*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + x69*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + x70*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])) + x14*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) - x15*x66 + x15*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + x22*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) - x23*x67 + x23*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + x24*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + x25*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + x26*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) + x27*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) - x3*(gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + x68*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + x69*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + x70*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) - x5*(gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x68*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x69*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x70*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) - x65*x7 + x7*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]);
	(*M)[1] = dAtdd_dx[0]*gtuu[0]*gtuu[1] + 2*dAtdd_dx[10]*gtuu[3]*gtuu[4] + dAtdd_dx[11]*x71 + dAtdd_dx[12]*gtuu[1]*gtuu[2] + dAtdd_dx[13]*gtuu[1]*gtuu[4] + dAtdd_dx[13]*gtuu[2]*gtuu[3] + dAtdd_dx[14]*gtuu[1]*gtuu[5] + dAtdd_dx[14]*gtuu[2]*gtuu[4] + dAtdd_dx[15]*gtuu[3]*gtuu[4] + dAtdd_dx[16]*gtuu[3]*gtuu[5] + dAtdd_dx[16]*x71 + dAtdd_dx[17]*gtuu[4]*gtuu[5] + dAtdd_dx[1]*gtuu[0]*gtuu[3] + dAtdd_dx[1]*x33 + dAtdd_dx[2]*gtuu[0]*gtuu[4] + dAtdd_dx[2]*gtuu[1]*gtuu[2] + dAtdd_dx[3]*gtuu[1]*gtuu[3] + dAtdd_dx[4]*gtuu[1]*gtuu[4] + dAtdd_dx[4]*gtuu[2]*gtuu[3] + dAtdd_dx[5]*gtuu[2]*gtuu[4] + dAtdd_dx[6]*x33 + 2*dAtdd_dx[7]*gtuu[1]*gtuu[3] + 2*dAtdd_dx[8]*gtuu[1]*gtuu[4] + dAtdd_dx[9]*((gtuu[3])*(gtuu[3])) - gtuu[1]*x29 - gtuu[1]*x34 - gtuu[1]*x37 - gtuu[1]*x41 - gtuu[1]*x42 - gtuu[1]*x44 - gtuu[1]*x50 - gtuu[1]*x52 - gtuu[1]*x58 - gtuu[1]*x59 - gtuu[1]*x61 - gtuu[3]*x30 - gtuu[3]*x35 - gtuu[3]*x38 - gtuu[3]*x43 - gtuu[3]*x46 - gtuu[3]*x47 - gtuu[3]*x51 - gtuu[3]*x54 - gtuu[3]*x55 - gtuu[3]*x60 - gtuu[3]*x63 - gtuu[4]*x31 - gtuu[4]*x36 - gtuu[4]*x39 - gtuu[4]*x40 - gtuu[4]*x45 - gtuu[4]*x48 - gtuu[4]*x49 - gtuu[4]*x53 - gtuu[4]*x56 - gtuu[4]*x57 - gtuu[4]*x62 - x10*(gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + x68*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + x69*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + x70*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) - x12*(gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x68*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x69*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x70*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) - x14*x65 + x14*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + x15*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + x22*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + x23*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) - x24*x66 + x24*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + x25*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) - x26*x67 + x26*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) + x27*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + x7*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) - x8*(gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + x68*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + x69*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + x70*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]));
	(*M)[2] = dAtdd_dx[0]*gtuu[0]*gtuu[2] + dAtdd_dx[10]*gtuu[3]*gtuu[5] + dAtdd_dx[10]*x71 + dAtdd_dx[11]*gtuu[4]*gtuu[5] + dAtdd_dx[12]*x32 + 2*dAtdd_dx[13]*gtuu[2]*gtuu[4] + 2*dAtdd_dx[14]*gtuu[2]*gtuu[5] + dAtdd_dx[15]*x71 + 2*dAtdd_dx[16]*gtuu[4]*gtuu[5] + dAtdd_dx[17]*((gtuu[5])*(gtuu[5])) + dAtdd_dx[1]*gtuu[0]*gtuu[4] + dAtdd_dx[1]*gtuu[1]*gtuu[2] + dAtdd_dx[2]*gtuu[0]*gtuu[5] + dAtdd_dx[2]*x32 + dAtdd_dx[3]*gtuu[1]*gtuu[4] + dAtdd_dx[4]*gtuu[1]*gtuu[5] + dAtdd_dx[4]*gtuu[2]*gtuu[4] + dAtdd_dx[5]*gtuu[2]*gtuu[5] + dAtdd_dx[6]*gtuu[1]*gtuu[2] + dAtdd_dx[7]*gtuu[1]*gtuu[4] + dAtdd_dx[7]*gtuu[2]*gtuu[3] + dAtdd_dx[8]*gtuu[1]*gtuu[5] + dAtdd_dx[8]*gtuu[2]*gtuu[4] + dAtdd_dx[9]*gtuu[3]*gtuu[4] - gtuu[2]*x29 - gtuu[2]*x34 - gtuu[2]*x37 - gtuu[2]*x41 - gtuu[2]*x42 - gtuu[2]*x44 - gtuu[2]*x50 - gtuu[2]*x52 - gtuu[2]*x58 - gtuu[2]*x59 - gtuu[2]*x61 - gtuu[4]*x30 - gtuu[4]*x35 - gtuu[4]*x38 - gtuu[4]*x43 - gtuu[4]*x46 - gtuu[4]*x47 - gtuu[4]*x51 - gtuu[4]*x54 - gtuu[4]*x55 - gtuu[4]*x60 - gtuu[4]*x63 - gtuu[5]*x31 - gtuu[5]*x36 - gtuu[5]*x39 - gtuu[5]*x40 - gtuu[5]*x45 - gtuu[5]*x48 - gtuu[5]*x49 - gtuu[5]*x53 - gtuu[5]*x56 - gtuu[5]*x57 - gtuu[5]*x62 + x14*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x15*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) - x16*(gtuu[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) + gtuu[3]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) + gtuu[5]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) + x68*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) + x69*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) + x70*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0])) - x18*(gtuu[0]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) + gtuu[3]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) + gtuu[5]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) + x68*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) + x69*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) + x70*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1])) - x20*(gtuu[0]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + gtuu[3]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + gtuu[5]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x68*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) + x69*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x70*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2])) - x22*x65 + x22*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x23*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) + x24*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) - x25*x66 + x25*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) + x26*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) - x27*x67 + x27*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + x7*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]);
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
