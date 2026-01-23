
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
	double x1 = dW_dx[0]*x0;
	double x2 = dalp_dx[0]*x1;
	double x3 = dalp_dx[1]*x1;
	double x4 = dalp_dx[2]*x1;
	double x5 = dalp_dx[0]*x0;
	double x6 = dW_dx[1]*x5;
	double x7 = dW_dx[1]*x0;
	double x8 = dalp_dx[1]*x7;
	double x9 = dalp_dx[2]*x7;
	double x10 = dW_dx[2]*x5;
	double x11 = dW_dx[2]*x0;
	double x12 = dalp_dx[1]*x11;
	double x13 = dalp_dx[2]*x11;
	double x14 = -ddalp_dx2[1];
	double x15 = -dW_dx[0]*dalp_dx[1]*x0;
	double x16 = -dW_dx[1]*dalp_dx[0]*x0;
	double x17 = -ddalp_dx2[2];
	double x18 = -dW_dx[0]*dalp_dx[2]*x0;
	double x19 = -dW_dx[2]*dalp_dx[0]*x0;
	double x20 = -ddalp_dx2[4];
	double x21 = -dW_dx[1]*dalp_dx[2]*x0;
	double x22 = -dW_dx[2]*dalp_dx[1]*x0;
	(*DiDjalp)[0] = 2*dW_dx[0]*dalp_dx[0]*x0 - dalp_dx[0]*(Gammatddd[0]*gtuu[0] + Gammatddd[12]*gtuu[2] + Gammatddd[6]*gtuu[1]) - dalp_dx[1]*(Gammatddd[0]*gtuu[1] + Gammatddd[12]*gtuu[4] + Gammatddd[6]*gtuu[3]) - dalp_dx[2]*(Gammatddd[0]*gtuu[2] + Gammatddd[12]*gtuu[5] + Gammatddd[6]*gtuu[4]) + ddalp_dx2[0] - gtdd[0]*gtuu[0]*x2 - gtdd[0]*gtuu[1]*x3 - gtdd[0]*gtuu[1]*x6 - gtdd[0]*gtuu[2]*x10 - gtdd[0]*gtuu[2]*x4 - gtdd[0]*gtuu[3]*x8 - gtdd[0]*gtuu[4]*x12 - gtdd[0]*gtuu[4]*x9 - gtdd[0]*gtuu[5]*x13;
	(*DiDjalp)[1] = -dalp_dx[0]*(Gammatddd[13]*gtuu[2] + Gammatddd[1]*gtuu[0] + Gammatddd[7]*gtuu[1]) - dalp_dx[1]*(Gammatddd[13]*gtuu[4] + Gammatddd[1]*gtuu[1] + Gammatddd[7]*gtuu[3]) - dalp_dx[2]*(Gammatddd[13]*gtuu[5] + Gammatddd[1]*gtuu[2] + Gammatddd[7]*gtuu[4]) - gtdd[1]*gtuu[0]*x2 - gtdd[1]*gtuu[1]*x3 - gtdd[1]*gtuu[1]*x6 - gtdd[1]*gtuu[2]*x10 - gtdd[1]*gtuu[2]*x4 - gtdd[1]*gtuu[3]*x8 - gtdd[1]*gtuu[4]*x12 - gtdd[1]*gtuu[4]*x9 - gtdd[1]*gtuu[5]*x13 - x14 - x15 - x16;
	(*DiDjalp)[2] = -dalp_dx[0]*(Gammatddd[14]*gtuu[2] + Gammatddd[2]*gtuu[0] + Gammatddd[8]*gtuu[1]) - dalp_dx[1]*(Gammatddd[14]*gtuu[4] + Gammatddd[2]*gtuu[1] + Gammatddd[8]*gtuu[3]) - dalp_dx[2]*(Gammatddd[14]*gtuu[5] + Gammatddd[2]*gtuu[2] + Gammatddd[8]*gtuu[4]) - gtdd[2]*gtuu[0]*x2 - gtdd[2]*gtuu[1]*x3 - gtdd[2]*gtuu[1]*x6 - gtdd[2]*gtuu[2]*x10 - gtdd[2]*gtuu[2]*x4 - gtdd[2]*gtuu[3]*x8 - gtdd[2]*gtuu[4]*x12 - gtdd[2]*gtuu[4]*x9 - gtdd[2]*gtuu[5]*x13 - x17 - x18 - x19;
	(*DiDjalp)[3] = 2*dW_dx[1]*dalp_dx[1]*x0 - dalp_dx[0]*(Gammatddd[15]*gtuu[2] + Gammatddd[3]*gtuu[0] + Gammatddd[9]*gtuu[1]) - dalp_dx[1]*(Gammatddd[15]*gtuu[4] + Gammatddd[3]*gtuu[1] + Gammatddd[9]*gtuu[3]) - dalp_dx[2]*(Gammatddd[15]*gtuu[5] + Gammatddd[3]*gtuu[2] + Gammatddd[9]*gtuu[4]) + ddalp_dx2[3] - gtdd[3]*gtuu[0]*x2 - gtdd[3]*gtuu[1]*x3 - gtdd[3]*gtuu[1]*x6 - gtdd[3]*gtuu[2]*x10 - gtdd[3]*gtuu[2]*x4 - gtdd[3]*gtuu[3]*x8 - gtdd[3]*gtuu[4]*x12 - gtdd[3]*gtuu[4]*x9 - gtdd[3]*gtuu[5]*x13;
	(*DiDjalp)[4] = -dalp_dx[0]*(Gammatddd[10]*gtuu[1] + Gammatddd[16]*gtuu[2] + Gammatddd[4]*gtuu[0]) - dalp_dx[1]*(Gammatddd[10]*gtuu[3] + Gammatddd[16]*gtuu[4] + Gammatddd[4]*gtuu[1]) - dalp_dx[2]*(Gammatddd[10]*gtuu[4] + Gammatddd[16]*gtuu[5] + Gammatddd[4]*gtuu[2]) - gtdd[4]*gtuu[0]*x2 - gtdd[4]*gtuu[1]*x3 - gtdd[4]*gtuu[1]*x6 - gtdd[4]*gtuu[2]*x10 - gtdd[4]*gtuu[2]*x4 - gtdd[4]*gtuu[3]*x8 - gtdd[4]*gtuu[4]*x12 - gtdd[4]*gtuu[4]*x9 - gtdd[4]*gtuu[5]*x13 - x20 - x21 - x22;
	(*DiDjalp)[5] = 2*dW_dx[2]*dalp_dx[2]*x0 - dalp_dx[0]*(Gammatddd[11]*gtuu[1] + Gammatddd[17]*gtuu[2] + Gammatddd[5]*gtuu[0]) - dalp_dx[1]*(Gammatddd[11]*gtuu[3] + Gammatddd[17]*gtuu[4] + Gammatddd[5]*gtuu[1]) - dalp_dx[2]*(Gammatddd[11]*gtuu[4] + Gammatddd[17]*gtuu[5] + Gammatddd[5]*gtuu[2]) + ddalp_dx2[5] - gtdd[5]*gtuu[0]*x2 - gtdd[5]*gtuu[1]*x3 - gtdd[5]*gtuu[1]*x6 - gtdd[5]*gtuu[2]*x10 - gtdd[5]*gtuu[2]*x4 - gtdd[5]*gtuu[3]*x8 - gtdd[5]*gtuu[4]*x12 - gtdd[5]*gtuu[4]*x9 - gtdd[5]*gtuu[5]*x13;
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
	double theta_damp_fact,
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
	*dtheta_dt = -1.0/6.0*alp*theta_damp_fact*(3*Atdd[0]*(gtuu[0]*x2 + gtuu[1]*x3 + gtuu[2]*x4) + 3*Atdd[3]*(gtuu[1]*x5 + gtuu[3]*x7 + gtuu[4]*x8) + 3*Atdd[5]*(gtuu[2]*x10 + gtuu[4]*x11 + gtuu[5]*x12) - 3*Rtrace + 6*kappa1*theta*(kappa2 + 2) + 48*M_PI*rho + x13*(gtuu[0]*x10 + gtuu[1]*x11 + gtuu[2]*x12) + x13*(gtuu[2]*x2 + gtuu[4]*x3 + gtuu[5]*x4) + x14*(gtuu[1]*x10 + gtuu[3]*x11 + gtuu[4]*x12) + x14*(gtuu[2]*x5 + gtuu[4]*x7 + gtuu[5]*x8) + x9*(gtuu[0]*x5 + gtuu[1]*x7 + gtuu[2]*x8) + x9*(gtuu[1]*x2 + gtuu[3]*x3 + gtuu[4]*x4) - 2*((Khat + 2*theta)*(Khat + 2*theta))) + betau[0]*dtheta_dx_upwind[0] + betau[1]*dtheta_dx_upwind[1] + betau[2]*dtheta_dx_upwind[2];
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
	double x31 = x24*xyz[2];
	double x32 = gtdd[2]*x6;
	double x33 = 2*x32;
	double x34 = x27*xyz[2];
	double x35 = gtdd[4]*x6;
	double x36 = 2*x35;
	double x37 = x0*x29 + x11*x30 + x25*x7 + x28*x8 + x31*x33 + x34*x36;
	double x38 = (1.0/sqrt(x37));
	double x39 = x0*x38;
	double x40 = x24*x38;
	double x41 = x27*x38;
	double x42 = x38*xyz[2];
	double x43 = x38*(gtdd[4]*x19*x41 + x10*x14*x41 + x10*x22*x40 + x14*x40*x7 + x18*x29*x42 + x20*x40 + x22*x41*x8 + x32*x39*xyz[0] + x35*x39*xyz[1]);
	double x44 = x14 + x17*x21 - x24*x43;
	double x45 = ((x44)*(x44));
	double x46 = -x21*x26 - x27*x43 + xyz[1]*xyz[2];
	double x47 = ((x46)*(x46));
	double x48 = -x3 - x43*xyz[2];
	double x49 = ((x48)*(x48));
	double x50 = x46*x48;
	double x51 = x11*x44*x46 + x29*x49 + x33*x44*x48 + x36*x50 + x45*x7 + x47*x8;
	double x52 = 1/(x51);
	double x53 = -x13*x2 + x45*x52;
	double x54 = 1/(W);
	double x55 = 2*dW_dx[0];
	double x56 = x54*x55;
	double x57 = dgtdd_dx[5] - gtdd[5]*x56;
	double x58 = ((x57)*(x57));
	double x59 = ((gtuu[5])*(gtuu[5]));
	double x60 = (1.0/16.0)*x59;
	double x61 = (1.0/3.0)*Khat + (2.0/3.0)*theta;
	double x62 = gtdd[0]*x61;
	double x63 = Atdd[0] + x62;
	double x64 = (1.0/4.0)*x6;
	double x65 = gtuu[0]*x64;
	double x66 = gtdd[1]*x61;
	double x67 = Atdd[1] + x66;
	double x68 = ((x67)*(x67));
	double x69 = gtuu[3]*x64;
	double x70 = gtdd[2]*x61;
	double x71 = Atdd[2] + x70;
	double x72 = ((x71)*(x71));
	double x73 = gtuu[5]*x64;
	double x74 = x64*(Khat + 2*theta);
	double x75 = 1/(((W)*(W)*(W)));
	double x76 = 2*dgtdd_dx[0];
	double x77 = 4*x75;
	double x78 = dW_dx[0]*x77;
	double x79 = 2*x75;
	double x80 = ddW_dx2[0]*x79;
	double x81 = gtdd[0]*x79;
	double x82 = ((W)*(W)*(W)*(W));
	double x83 = 1/(x82);
	double x84 = 6*x83;
	double x85 = ((dW_dx[0])*(dW_dx[0]))*x84;
	double x86 = gtdd[0]*x84;
	double x87 = dW_dx[0]*x86;
	double x88 = -dW_dx[1]*x87 + ddW_dx2[1]*x81 + ddgtdd_dx2[1]*x6 - ddgtdd_dx2[6]*x6 - dgtdd_dx[1]*x78 - gtdd[1]*x80 + gtdd[1]*x85 + x75*(dW_dx[1]*x76 + dgtdd_dx[6]*x55);
	double x89 = gtuu[1]*x5;
	double x90 = (1.0/8.0)*x89;
	double x91 = -x88;
	double x92 = dW_dx[2]*x87 - ddW_dx2[2]*x81 + ddgtdd_dx2[12]*x6 - ddgtdd_dx2[2]*x6 + dgtdd_dx[2]*x78 + gtdd[2]*x80 - gtdd[2]*x85 - x75*(dW_dx[2]*x76 + dgtdd_dx[12]*x55);
	double x93 = gtuu[2]*x5;
	double x94 = (1.0/8.0)*x93;
	double x95 = -x92;
	double x96 = gtuu[3]*x5;
	double x97 = 2*dW_dx[1];
	double x98 = x75*(dgtdd_dx[1]*x97 + dgtdd_dx[7]*x55);
	double x99 = gtdd[1]*x77;
	double x100 = dW_dx[1]*x77;
	double x101 = ((dW_dx[1])*(dW_dx[1]));
	double x102 = -ddW_dx2[3]*x81 + ddgtdd_dx2[18]*x6 - dgtdd_dx[6]*x100 + x101*x86;
	double x103 = ddgtdd_dx2[3]*x6 - dgtdd_dx[3]*x78 - gtdd[3]*x80 + gtdd[3]*x85;
	double x104 = 12*dW_dx[0]*dW_dx[1]*gtdd[1]*x83 - ddW_dx2[1]*x99 + 2*ddgtdd_dx2[7]*x6 - x102 - x103 - 2*x98;
	double x105 = (1.0/8.0)*x104;
	double x106 = 2*dW_dx[2];
	double x107 = x75*(dgtdd_dx[13]*x55 + dgtdd_dx[1]*x106);
	double x108 = ddgtdd_dx2[13]*x6;
	double x109 = dW_dx[0]*dW_dx[2];
	double x110 = 12*x83;
	double x111 = ddgtdd_dx2[4]*x6;
	double x112 = gtdd[4]*x85;
	double x113 = gtdd[4]*x80;
	double x114 = dgtdd_dx[4]*x78;
	double x115 = -x111 - x112 + x113 + x114;
	double x116 = x75*(dgtdd_dx[12]*x97 + dgtdd_dx[6]*x106);
	double x117 = ddgtdd_dx2[24]*x6;
	double x118 = ddW_dx2[4]*x81;
	double x119 = dW_dx[1]*dW_dx[2];
	double x120 = x119*x86;
	double x121 = x116 - x117 + x118 - x120;
	double x122 = -ddW_dx2[2]*x99 + gtdd[1]*x109*x110 - 2*x107 + 2*x108 + x115 + x121;
	double x123 = gtuu[4]*x5;
	double x124 = (1.0/8.0)*x123;
	double x125 = x75*(dgtdd_dx[2]*x97 + dgtdd_dx[8]*x55);
	double x126 = gtdd[2]*x77;
	double x127 = -x116 + x117 - x118 + x120;
	double x128 = x111 + x112 - x113 - x114;
	double x129 = 12*dW_dx[0]*dW_dx[1]*gtdd[2]*x83 - ddW_dx2[1]*x126 + 2*ddgtdd_dx2[8]*x6 - 2*x125 - x127 - x128;
	double x130 = dgtdd_dx[14]*x55 + dgtdd_dx[2]*x106;
	double x131 = ddgtdd_dx2[14]*x6;
	double x132 = ((dW_dx[2])*(dW_dx[2]));
	double x133 = dW_dx[2]*x77;
	double x134 = ddW_dx2[5]*x81 - ddgtdd_dx2[30]*x6 + dgtdd_dx[12]*x133 - x132*x86;
	double x135 = -ddgtdd_dx2[5]*x6 + dgtdd_dx[5]*x78 + gtdd[5]*x80 - gtdd[5]*x85;
	double x136 = -ddW_dx2[2]*x126 + gtdd[2]*x109*x110 - 2*x130*x75 + 2*x131 + x134 + x135;
	double x137 = gtuu[5]*x5;
	double x138 = (1.0/8.0)*x137;
	double x139 = (1.0/16.0)*x57;
	double x140 = ((gtuu[2])*(gtuu[2]));
	double x141 = dgtdd_dx[0] - gtdd[0]*x56;
	double x142 = x140*x141;
	double x143 = dgtdd_dx[3] - gtdd[3]*x56;
	double x144 = ((gtuu[4])*(gtuu[4]));
	double x145 = x143*x144;
	double x146 = (1.0/2.0)*x6;
	double x147 = x146*x63;
	double x148 = gtuu[1]*x67;
	double x149 = gtuu[2]*x71;
	double x150 = gtuu[4]*x67;
	double x151 = x146*x71;
	double x152 = x106*x54;
	double x153 = dgtdd_dx[12] - gtdd[0]*x152;
	double x154 = -x146*x153 + x6*(dgtdd_dx[2] - gtdd[2]*x56);
	double x155 = dgtdd_dx[15] - gtdd[3]*x152;
	double x156 = x144*x155;
	double x157 = x156*x5;
	double x158 = x154*x157;
	double x159 = x140*x153;
	double x160 = x159*x5;
	double x161 = (1.0/8.0)*x154;
	double x162 = dgtdd_dx[17] - gtdd[5]*x152;
	double x163 = x162*x5*x59;
	double x164 = (1.0/4.0)*gtuu[5];
	double x165 = x146*(dgtdd_dx[4] - gtdd[4]*x56);
	double x166 = x54*x97;
	double x167 = x146*(dgtdd_dx[8] - gtdd[2]*x166);
	double x168 = x146*(dgtdd_dx[13] - gtdd[1]*x152);
	double x169 = x165 + x167 - x168;
	double x170 = ((x169)*(x169))*x82;
	double x171 = x164*x170;
	double x172 = x165 - x167 + x168;
	double x173 = x144*x82;
	double x174 = x169*x173;
	double x175 = x172*x174;
	double x176 = gtuu[2]*x139;
	double x177 = dgtdd_dx[6] - gtdd[0]*x166;
	double x178 = gtuu[4]*x177;
	double x179 = gtuu[5]*x153;
	double x180 = (1.0/2.0)*x154;
	double x181 = -x165 + x167 + x168;
	double x182 = gtuu[4]*x181;
	double x183 = gtuu[2]*x82;
	double x184 = x182*x183;
	double x185 = x154*x82;
	double x186 = gtuu[2]*x185;
	double x187 = -x143*x146 + x6*(dgtdd_dx[7] - gtdd[1]*x166);
	double x188 = gtuu[3]*x187;
	double x189 = x186*x188;
	double x190 = gtuu[4]*x172;
	double x191 = x186*x190;
	double x192 = -x146*x57 + x6*(dgtdd_dx[14] - gtdd[2]*x152);
	double x193 = x164*x192;
	double x194 = gtuu[3]*x185;
	double x195 = -x146*x155 + x6*(dgtdd_dx[10] - gtdd[4]*x166);
	double x196 = x164*x195;
	double x197 = dgtdd_dx[11] - gtdd[5]*x166;
	double x198 = x123*x154;
	double x199 = gtuu[4]*x164;
	double x200 = -x146*x197 + x6*(dgtdd_dx[16] - gtdd[4]*x152);
	double x201 = x185*x200;
	double x202 = gtuu[2]*x177;
	double x203 = x154*x90;
	double x204 = gtuu[4]*x143;
	double x205 = (1.0/8.0)*x96;
	double x206 = dgtdd_dx[9] - gtdd[3]*x166;
	double x207 = gtuu[4]*x206;
	double x208 = x205*x207;
	double x209 = x169*x82;
	double x210 = gtuu[1]*x209;
	double x211 = gtuu[4]*(-x146*x177 + x6*(dgtdd_dx[1] - gtdd[1]*x56));
	double x212 = x210*x211;
	double x213 = x123*x57;
	double x214 = x164*x169;
	double x215 = gtuu[2]*x141;
	double x216 = x169*x90;
	double x217 = gtuu[3]*x177;
	double x218 = x169*x94;
	double x219 = gtuu[4]*x153;
	double x220 = x57*x94;
	double x221 = x169*x204;
	double x222 = x124*x172;
	double x223 = gtuu[5]*x57;
	double x224 = gtuu[3]*x171 + x105*x96 + x122*x124 + x124*x129 + x136*x138 + x139*x142 + x139*x145 - x147*x148 - x147*x149 - x150*x151 - x154*x208 - 1.0/4.0*x158 - x160*x161 - x161*x163 - x164*x197*x198 + (1.0/4.0)*x175 + x176*x178 + x176*x179 - x180*x184 - x186*x193 - 1.0/4.0*x189 - 1.0/4.0*x191 - x194*x196 - x199*x201 - x202*x203 - x203*x204 + x205*x221 + x211*x220 + (1.0/4.0)*x212 + x213*x214 + x215*x216 + x217*x218 + x218*x219 + x222*x223 + x58*x60 - ((x63)*(x63))*x65 + x63*x74 - x68*x69 - x72*x73 + x88*x90 + x90*x91 + x92*x94 + x94*x95;
	double x225 = -x1*x13 + x47*x52;
	double x226 = ((x197)*(x197));
	double x227 = gtdd[3]*x61;
	double x228 = Atdd[3] + x227;
	double x229 = gtdd[4]*x61;
	double x230 = Atdd[4] + x229;
	double x231 = ((x230)*(x230));
	double x232 = gtuu[0]*x5;
	double x233 = ddW_dx2[3]*x79;
	double x234 = gtdd[3]*x79;
	double x235 = x101*x84;
	double x236 = dW_dx[0]*dW_dx[1];
	double x237 = gtdd[3]*x84;
	double x238 = ddW_dx2[1]*x234 + ddgtdd_dx2[19]*x6 - ddgtdd_dx2[9]*x6 - dgtdd_dx[7]*x100 - gtdd[1]*x233 + gtdd[1]*x235 - x236*x237 + x75*(dgtdd_dx[3]*x97 + dgtdd_dx[9]*x55);
	double x239 = -x238;
	double x240 = x75*(dgtdd_dx[10]*x55 + dgtdd_dx[4]*x97);
	double x241 = ddgtdd_dx2[10]*x6;
	double x242 = gtdd[4]*x77;
	double x243 = x75*(dgtdd_dx[15]*x55 + dgtdd_dx[3]*x106);
	double x244 = ddgtdd_dx2[15]*x6;
	double x245 = ddW_dx2[2]*x234;
	double x246 = x109*x237;
	double x247 = x243 - x244 + x245 - x246;
	double x248 = ddgtdd_dx2[20]*x6;
	double x249 = gtdd[2]*x235;
	double x250 = gtdd[2]*x233;
	double x251 = dgtdd_dx[8]*x100;
	double x252 = -x248 - x249 + x250 + x251;
	double x253 = -ddW_dx2[1]*x242 + gtdd[4]*x110*x236 - 2*x240 + 2*x241 + x247 + x252;
	double x254 = x75*(dgtdd_dx[13]*x97 + dgtdd_dx[7]*x106);
	double x255 = x248 + x249 - x250 - x251;
	double x256 = -x243 + x244 - x245 + x246;
	double x257 = 12*dW_dx[1]*dW_dx[2]*gtdd[1]*x83 - ddW_dx2[4]*x99 + 2*ddgtdd_dx2[25]*x6 - 2*x254 - x255 - x256;
	double x258 = ddW_dx2[4]*x234 + ddgtdd_dx2[22]*x6 - ddgtdd_dx2[27]*x6 - dgtdd_dx[10]*x100 - gtdd[4]*x233 + gtdd[4]*x235 - x119*x237 + x75*(dgtdd_dx[15]*x97 + dgtdd_dx[9]*x106);
	double x259 = -x258;
	double x260 = ddgtdd_dx2[33]*x6;
	double x261 = dgtdd_dx[10]*x106 + dgtdd_dx[16]*x97;
	double x262 = x261*x75;
	double x263 = dgtdd_dx[15]*x133;
	double x264 = ddW_dx2[5]*x234;
	double x265 = x132*x84;
	double x266 = gtdd[3]*x265;
	double x267 = ddgtdd_dx2[23]*x6 - dgtdd_dx[11]*x100 - gtdd[5]*x233 + gtdd[5]*x235;
	double x268 = 12*dW_dx[1]*dW_dx[2]*gtdd[4]*x83 - ddW_dx2[4]*x242 + 2*ddgtdd_dx2[28]*x6 - x260 - 2*x262 + x263 + x264 - x266 - x267;
	double x269 = (1.0/16.0)*x197;
	double x270 = x140*x177;
	double x271 = x144*x206;
	double x272 = x146*x228;
	double x273 = gtuu[2]*x67;
	double x274 = gtuu[4]*x230;
	double x275 = (1.0/4.0)*x159;
	double x276 = x275*x5;
	double x277 = (1.0/8.0)*x195;
	double x278 = x140*x82;
	double x279 = x169*x278;
	double x280 = x181*x279;
	double x281 = gtuu[4]*x269;
	double x282 = gtuu[2]*x143;
	double x283 = gtuu[5]*x155;
	double x284 = (1.0/2.0)*x195;
	double x285 = x183*x190;
	double x286 = x195*x82;
	double x287 = x211*x286;
	double x288 = gtuu[0]*x287;
	double x289 = gtuu[0]*x185;
	double x290 = gtuu[2]*x286;
	double x291 = x182*x290;
	double x292 = x57*x93;
	double x293 = (1.0/8.0)*x232;
	double x294 = x195*x215;
	double x295 = x195*x90;
	double x296 = gtuu[2]*x187;
	double x297 = x210*x296;
	double x298 = x197*x93;
	double x299 = x169*x293;
	double x300 = x197*x94;
	double x301 = gtuu[4]*x187;
	double x302 = gtuu[4]*x155;
	double x303 = gtuu[5]*x197;
	double x304 = x303*x94;
	double x305 = gtuu[0]*x171 + x105*x232 + x124*x258 + x124*x259 + x138*x268 - x146*x230*x273 - x148*x272 - x157*x277 - x163*x277 + x181*x304 - x193*x290 - x195*x276 - x196*x289 - x196*x292 - x199*x200*x286 - x202*x295 + x202*x299 - x204*x295 + x207*x216 + x214*x298 + x218*x302 + x221*x293 + x226*x60 - ((x228)*(x228))*x69 + x228*x74 - x231*x73 + x238*x90 + x239*x90 + x253*x94 + x257*x94 + x269*x270 + x269*x271 - x272*x274 + (1.0/4.0)*x280 + x281*x282 + x281*x283 - x284*x285 - 1.0/4.0*x288 - 1.0/4.0*x291 - x293*x294 + (1.0/4.0)*x297 + x300*x301 - x65*x68;
	double x306 = x44*x52;
	double x307 = x13*x9 + x306*x46;
	double x308 = x157*x169;
	double x309 = x160*x169;
	double x310 = x173*x195;
	double x311 = x172*x310;
	double x312 = gtuu[2]*x269;
	double x313 = (1.0/2.0)*x169;
	double x314 = gtuu[3]*x209;
	double x315 = x296*x314;
	double x316 = gtuu[2]*x209;
	double x317 = x190*x316;
	double x318 = gtuu[1]*x287;
	double x319 = x195*x94;
	double x320 = x195*x205;
	double x321 = gtuu[1]*x64;
	double x322 = ddW_dx2[1]*x79;
	double x323 = -ddgtdd_dx2[7]*x6 - gtdd[1]*x236*x84 + gtdd[1]*x322 + x98;
	double x324 = x102 + x323;
	double x325 = x103 + x323;
	double x326 = ddgtdd_dx2[8]*x6;
	double x327 = gtdd[2]*x322;
	double x328 = x236*x84;
	double x329 = gtdd[2]*x328;
	double x330 = x125 - x326 + x327 - x329;
	double x331 = x127 + x330;
	double x332 = gtdd[1]*x79;
	double x333 = gtdd[1]*x84;
	double x334 = -ddW_dx2[2]*x332 - x107 + x108 + x109*x333;
	double x335 = x115 + x334;
	double x336 = -x335;
	double x337 = -gtdd[4]*x322 + gtdd[4]*x328 - x240 + x241;
	double x338 = x247 + x337;
	double x339 = -x338;
	double x340 = ddgtdd_dx2[25]*x6;
	double x341 = ddW_dx2[4]*x332;
	double x342 = x119*x333;
	double x343 = x254 - x340 + x341 - x342;
	double x344 = x255 + x343;
	double x345 = x75*(dgtdd_dx[16]*x55 + dgtdd_dx[4]*x106);
	double x346 = ddgtdd_dx2[16]*x6;
	double x347 = ddW_dx2[2]*x79;
	double x348 = gtdd[4]*x347;
	double x349 = x109*x84;
	double x350 = gtdd[4]*x349;
	double x351 = ddgtdd_dx2[11]*x6 - gtdd[5]*x322 + gtdd[5]*x328 - x75*(dgtdd_dx[11]*x55 + dgtdd_dx[5]*x97);
	double x352 = x345 - x346 + x348 - x350 + x351;
	double x353 = x75*(dgtdd_dx[14]*x97 + dgtdd_dx[8]*x106);
	double x354 = ddgtdd_dx2[26]*x6;
	double x355 = gtdd[2]*x79;
	double x356 = ddW_dx2[4]*x355;
	double x357 = x119*x84;
	double x358 = gtdd[2]*x357;
	double x359 = x353 - x354 + x356 - x358;
	double x360 = ddgtdd_dx2[31]*x6;
	double x361 = dgtdd_dx[13]*x133;
	double x362 = ddW_dx2[5]*x332;
	double x363 = gtdd[1]*x265;
	double x364 = x360 - x361 - x362 + x363;
	double x365 = -x352 - x359 - x364;
	double x366 = x197*x57;
	double x367 = x63*x65;
	double x368 = x321*x63;
	double x369 = gtuu[2]*x63;
	double x370 = x369*x64;
	double x371 = x64*x67;
	double x372 = x67*x69;
	double x373 = gtuu[4]*x228;
	double x374 = x373*x64;
	double x375 = x71*x73;
	double x376 = x192*x209;
	double x377 = gtuu[2]*x376;
	double x378 = x200*x209;
	double x379 = gtuu[5]*x169;
	double x380 = x197*x379;
	double x381 = gtuu[1]*x185;
	double x382 = gtuu[5]*x195;
	double x383 = x382*x57;
	double x384 = -gtuu[1]*x171 + x124*x339 + x124*x344 - x124*x380 + x124*x383 + x138*x365 - x149*x371 + x154*x304 - 1.0/8.0*x163*x169 - x164*x377 + x196*x381 - x199*x378 - x202*x216 - x204*x216 - x220*x379 - x228*x368 - x228*x372 - x230*x370 - x230*x375 - x274*x371 - x321*x68 + x324*x90 + x325*x90 + x331*x94 + x336*x94 + x366*x60 - x367*x67 - x374*x71 + x67*x74;
	double x385 = x142*x269 + x145*x269 - x169*x208 + x178*x312 + x179*x312 - x184*x313 + x204*x320 + x211*x300 + x217*x319 + x219*x319 + x222*x303 + x294*x90 - 1.0/4.0*x308 - 1.0/8.0*x309 + (1.0/4.0)*x311 - 1.0/4.0*x315 - 1.0/4.0*x317 + (1.0/4.0)*x318 + x384;
	double x386 = x154*x278;
	double x387 = x181*x386;
	double x388 = gtuu[4]*x139;
	double x389 = gtuu[0]*x209;
	double x390 = x211*x389;
	double x391 = x182*x316;
	double x392 = x296*x381;
	double x393 = x154*x293;
	double x394 = x154*x94;
	double x395 = x181*x57;
	double x396 = gtuu[5]*x395;
	double x397 = x139*x270 + x139*x271 - x169*x276 + x176*x204 + x202*x393 + x203*x207 + x204*x393 - x215*x299 + x220*x301 + x283*x388 - x285*x313 + x302*x394 - 1.0/8.0*x308 + x384 + (1.0/4.0)*x387 - 1.0/4.0*x390 - 1.0/4.0*x391 + (1.0/4.0)*x392 + x396*x94;
	double x398 = gtdd[5]*x61;
	double x399 = Atdd[5] + x398;
	double x400 = (1.0/16.0)*x162;
	double x401 = x136*x232;
	double x402 = x351 + x364;
	double x403 = 12*dW_dx[1]*dW_dx[2]*gtdd[2]*x83 - ddW_dx2[4]*x126 + 2*ddgtdd_dx2[26]*x6 - 2*x353 - x402;
	double x404 = 12*dW_dx[0]*dW_dx[2]*gtdd[4]*x83 - ddW_dx2[2]*x242 + 2*ddgtdd_dx2[16]*x6 - 2*x345 - x402;
	double x405 = ddW_dx2[5]*x355 + ddgtdd_dx2[17]*x6 - ddgtdd_dx2[32]*x6 + dgtdd_dx[14]*x133 - gtdd[2]*x265 - gtdd[5]*x347 + gtdd[5]*x349 - x75*(dgtdd_dx[17]*x55 + dgtdd_dx[5]*x106);
	double x406 = -x405;
	double x407 = x268*x96;
	double x408 = ddW_dx2[4]*x79;
	double x409 = ddW_dx2[5]*gtdd[4]*x79 + ddgtdd_dx2[29]*x6 - ddgtdd_dx2[34]*x6 + dgtdd_dx[16]*x133 - gtdd[4]*x265 + gtdd[5]*x357 - gtdd[5]*x408 - x75*(dgtdd_dx[11]*x106 + dgtdd_dx[17]*x97);
	double x410 = -x409;
	double x411 = (1.0/16.0)*gtuu[5];
	double x412 = gtuu[0]*x58;
	double x413 = gtuu[3]*x226;
	double x414 = x146*x399;
	double x415 = (1.0/8.0)*gtuu[1];
	double x416 = x162*x415;
	double x417 = gtuu[0]*x215;
	double x418 = gtuu[3]*gtuu[4];
	double x419 = x206*x418;
	double x420 = gtuu[5]*x366;
	double x421 = (1.0/8.0)*x5;
	double x422 = x192*x57;
	double x423 = x140*x422;
	double x424 = x144*x197;
	double x425 = x200*x424;
	double x426 = gtuu[0]*x153;
	double x427 = gtuu[1]*x153;
	double x428 = gtuu[1]*x155;
	double x429 = gtuu[3]*x281;
	double x430 = x162*x89;
	double x431 = x162*x211;
	double x432 = gtuu[5]*x162;
	double x433 = x162*x94;
	double x434 = x162*x382;
	double x435 = x190*x57;
	double x436 = gtuu[2]*x181;
	double x437 = x436*x57;
	double x438 = x190*x197;
	double x439 = gtuu[3]*x181;
	double x440 = gtuu[4]*x192;
	double x441 = gtuu[4]*x200;
	double x442 = x49*x52;
	double x443 = (1.0/4.0)*x83;
	double x444 = x156*x57;
	double x445 = (1.0/4.0)*x162;
	double x446 = (1.0/2.0)*x5;
	double x447 = x172*x424;
	double x448 = (1.0/4.0)*x57;
	double x449 = gtuu[1]*x448;
	double x450 = (1.0/4.0)*x197;
	double x451 = gtuu[1]*x450;
	double x452 = gtuu[3]*x450;
	double x453 = gtuu[2]*x153;
	double x454 = gtuu[4]*x450;
	double x455 = gtuu[2]*x178;
	double x456 = x164*x453;
	double x457 = (1.0/2.0)*x89;
	double x458 = x197*x211;
	double x459 = (1.0/2.0)*x93;
	double x460 = (1.0/2.0)*x123;
	double x461 = x172*x460;
	double x462 = gtuu[5]*x422;
	double x463 = x200*x460;
	double x464 = -x121 - x334;
	double x465 = x128 + x330;
	double x466 = -2*ddW_dx2[2]*gtdd[2]*x75 + gtdd[2]*x349 - x130*x75 + x131;
	double x467 = -x134 - x466;
	double x468 = -x135 - x466;
	double x469 = x252 - x254 + x338 + x340 - x341 + x342;
	double x470 = (1.0/2.0)*x96;
	double x471 = -x360 + x361 + x362 - x363;
	double x472 = x353 - x354 + x356 - x358 - x471;
	double x473 = gtuu[2]*x58;
	double x474 = x180*x89;
	double x475 = x180*x432;
	double x476 = x313*x96;
	double x477 = x123*x313;
	double x478 = x313*x89;
	double x479 = x459*x57;
	double x480 = x284*x96;
	double x481 = -x164*x473 - x188*x479 - x199*x366 - x223*x478 - x223*x480 - x275*x57 + x303*x474 + x303*x476 + x352*x460 - x419*x448 + x432*x477 - x435*x459 + x457*x464 + x457*x465 + x459*x467 + x459*x468 + x460*x472 + x469*x470 + x475*x93;
	double x482 = x142*x445 + x145*x445 + x162*x456 - x182*x292 - x202*x449 + x202*x452 - x204*x449 + x204*x452 + x215*x451 - x223*x463 + x431*x459 + x432*x461 - 1.0/2.0*x444 + x445*x455 + x446*x447 + x453*x454 + x457*x458 - x459*x462 + x481;
	double x483 = gtdd[2]*x64;
	double x484 = x192*x386;
	double x485 = x174*x200;
	double x486 = (1.0/2.0)*gtuu[1];
	double x487 = x202*x57;
	double x488 = x204*x486;
	double x489 = x190*x289;
	double x490 = x381*x436;
	double x491 = x190*x210;
	double x492 = x314*x436;
	double x493 = gtuu[2]*gtuu[4];
	double x494 = x376*x493;
	double x495 = x201*x493;
	double x496 = x180*x232;
	double x497 = (1.0/2.0)*x232;
	double x498 = x211*x57;
	double x499 = -x182*x479 + x302*x474 + x302*x476 - x417*x448 - 1.0/4.0*x444 + x453*x478 + x453*x496 + x481 + x484 + x485 - x486*x487 - x488*x57 + x489 + x490 + x491 + x492 + x494 + x495 - x497*x498;
	double x500 = x202*x486;
	double x501 = gtuu[4]*x448;
	double x502 = x197*x459;
	double x503 = x162*x459;
	double x504 = gtuu[2]*x426*x448 + gtuu[2]*x427*x450 + x155*x418*x450 - x156*x445 - x162*x275 - x162*x488 - x162*x500 + x164*x412 + x164*x413 - x182*x503 - x188*x503 - x190*x503 - x232*x475 - x379*x430 + (1.0/2.0)*x401 + x403*x457 + x404*x457 + x405*x459 + x406*x459 + (1.0/2.0)*x407 + x409*x460 + x410*x460 - x417*x445 - x419*x445 + x420*x486 + x423*x446 + x425*x446 + x428*x501 - x431*x497 - x432*x480 + x435*x497 + x437*x457 + x438*x457 + x439*x502 + x440*x502 + x441*x479;
	double x505 = (1.0/4.0)*x504;
	double x506 = (1.0/4.0)*x59;
	double x507 = (1.0/2.0)*x137;
	double x508 = gtuu[5]*x170;
	double x509 = x313*x93;
	double x510 = gtuu[5]*x192;
	double x511 = gtuu[4]*gtuu[5];
	double x512 = gtuu[3]*x508 + x104*x470 + x122*x460 + x129*x460 + x136*x507 + x142*x448 + x145*x448 - x158 - x160*x180 - x163*x180 + x175 - x180*x207*x96 - 2*x182*x186 - x186*x510 - x189 - x191 - x194*x382 - x198*x303 - x201*x511 - x202*x474 - x204*x474 + x204*x476 + x211*x479 + x212 + x213*x379 + x215*x478 + x217*x509 + x219*x509 + x223*x461 + x448*x455 + x456*x57 + x457*x88 + x457*x91 + x459*x92 + x459*x95 + x506*x58;
	double x513 = (1.0/4.0)*x29;
	double x514 = x399*x443;
	double x515 = x215*x284;
	double x516 = x284*x93;
	double x517 = x180*x93;
	double x518 = -gtuu[1]*x508 + gtuu[5]*x213*x284 - gtuu[5]*x292*x313 - gtuu[5]*x377 - x163*x313 - x202*x478 - x204*x478 - x303*x477 + x303*x517 + x324*x457 + x325*x457 + x331*x459 + x336*x459 + x339*x460 + x344*x460 + x365*x507 + x366*x506 - x378*x511 + x381*x382;
	double x519 = x142*x450 + x145*x450 - x160*x313 + x197*x456 + x204*x480 - x207*x476 + x217*x516 + x219*x516 + x303*x461 - x308 + x311 - x315 - x317 + x318 - 2*x391 + x450*x455 + x458*x459 + x515*x89 + x518;
	double x520 = x155*x199;
	double x521 = x232*x313;
	double x522 = -x157*x313 + x202*x496 + x204*x496 + x207*x474 - x215*x521 + x270*x448 + x271*x448 + x282*x501 + x301*x479 + x302*x517 - x309 - 2*x317 + x387 - x390 - x391 + x392 + x396*x459 + x518 + x520*x57;
	double x523 = x181*x459;
	double x524 = x284*x89;
	double x525 = gtuu[0]*x508 + x104*x497 - x157*x284 - x160*x195 - x163*x284 - 2*x190*x290 + x197*x520 + x202*x521 - x202*x524 + x204*x521 - x204*x524 + x207*x478 + x226*x506 - x232*x515 + x238*x457 + x239*x457 + x253*x459 + x257*x459 + x258*x460 + x259*x460 + x268*x507 + x270*x450 + x271*x450 + x280 + x282*x454 - x288 - x289*x382 - x290*x510 - x291 - x292*x382 + x297 + x298*x379 + x301*x502 + x302*x509 + x303*x523 - x382*x441*x82;
	double x526 = x159*x197;
	double x527 = x140*x395;
	double x528 = gtuu[0]*x448;
	double x529 = gtuu[4]*x282;
	double x530 = x296*x57;
	double x531 = x121 - x125 + x326 - x327 + x329 + x335;
	double x532 = x256 + x343;
	double x533 = -x252 - x337;
	double x534 = x351 + x359;
	double x535 = x345 - x346 + x348 - x350 - x471;
	double x536 = ddgtdd_dx2[28]*x6;
	double x537 = gtdd[4]*x357;
	double x538 = gtdd[4]*x408 + x262 + x267 - x536 - x537;
	double x539 = 2*ddW_dx2[4]*gtdd[4]*x75 + x260 + x261*x75 - x263 - x264 + x266 - x536 - x537;
	double x540 = gtuu[4]*x226;
	double x541 = -gtuu[2]*x164*x366 + x123*x284*x432 - x156*x450 - x164*x540 - x182*x502 + x223*x521 + x223*x524 - x303*x478 - x303*x496 - x417*x450 + x432*x509 + x457*x532 + x457*x533 - x458*x497 + x459*x534 + x459*x535 + x460*x538 + x460*x539 + x497*x531;
	double x542 = x155*x448*x493 + x162*x520 - x190*x298 - x202*x451 + x202*x528 - x204*x451 + x204*x528 + x207*x449 + x270*x445 + x271*x445 + x301*x503 - x303*x463 + x432*x523 + x445*x529 + x446*x527 + x457*x530 - x502*x510 - 1.0/2.0*x526 + x541;
	double x543 = x192*x279;
	double x544 = x200*x310;
	double x545 = x190*x389;
	double x546 = x210*x436;
	double x547 = gtuu[1]*x190*x286;
	double x548 = gtuu[3]*x286*x436;
	double x549 = x192*x286*x493;
	double x550 = x378*x493;
	double x551 = -x188*x502 - x197*x275 - x197*x488 - x197*x500 + x302*x478 + x302*x480 - x419*x450 - x438*x459 + x453*x521 + x453*x524 + x541 + x543 + x544 + x545 + x546 + x547 + x548 + x549 + x550;
	double x552 = x123*x542 + x123*x551 + x137*x504 + x232*x512 + x482*x93 + x499*x93 + x519*x89 + x522*x89 + x525*x96;
	double x553 = (1.0/8.0)*x552*x83;
	double x554 = gtdd[5]*x553;
	double x555 = -gtdd[0]*x554 + ((gtdd[2])*(gtdd[2]))*x553 - x443*x72 - x482*x483 - x483*x499 + x505*x7 + x512*x513 + x514*x63;
	double x556 = 1/(x37);
	double x557 = x0*x556;
	double x558 = x555*x557;
	double x559 = gtdd[4]*x64;
	double x560 = -gtdd[3]*x554 + ((gtdd[4])*(gtdd[4]))*x553 + x228*x514 - x231*x443 + x505*x8 + x513*x525 - x542*x559 - x551*x559;
	double x561 = x557*x560;
	double x562 = x230*x443;
	double x563 = gtdd[2]*x553;
	double x564 = -gtdd[1]*x554 + gtdd[4]*x563 + x10*x505 + x514*x67 - x562*x71;
	double x565 = -x483*x542 - x499*x559 + x513*x522 + x564;
	double x566 = x307*x557;
	double x567 = -x482*x559 - x483*x551 + x513*x519 + x564;
	double x568 = (1.0/4.0)*x10;
	double x569 = (1.0/4.0)*x7;
	double x570 = -gtdd[0]*gtdd[3]*x553 + ((gtdd[1])*(gtdd[1]))*x553 + x228*x443*x63 - x443*x68 + (1.0/4.0)*x512*x8 - x519*x568 - x522*x568 + x525*x569;
	double x571 = x556*x570;
	double x572 = x25*x571;
	double x573 = x28*x571;
	double x574 = gtuu[1]*x177;
	double x575 = x139*x204;
	double x576 = x141*x312;
	double x577 = x124*x200;
	double x578 = x67*x71;
	double x579 = x150*x64;
	double x580 = x274*x64;
	double x581 = x206*x388;
	double x582 = x379*x57;
	double x583 = -gtuu[2]*x64*x72 - gtuu[3]*x581 + x124*x162*x379 + x124*x352 + x124*x472 - x139*x159 - x188*x220 - x190*x220 + x203*x303 + x205*x380 - x205*x383 + x205*x469 - x230*x368 - x230*x372 - x303*x388 - x321*x578 - x367*x71 - x370*x399 - x375*x399 + x394*x432 - x399*x579 - x411*x473 + x464*x90 + x465*x90 + x467*x94 + x468*x94 - x580*x71 - x582*x90 + x71*x74;
	double x584 = -gtuu[1]*x575 + gtuu[1]*x576 + gtuu[2]*x179*x400 + x142*x400 + x143*x429 + x145*x400 - x176*x574 - x182*x448*x93 + x217*x312 + x222*x432 - x223*x577 + x281*x453 + x400*x455 + x421*x447 + x431*x94 - 1.0/8.0*x444 + x458*x90 - x462*x94 + x583;
	double x585 = x306*x48;
	double x586 = x204*x415;
	double x587 = gtuu[0]*x176;
	double x588 = -x139*x156 - x141*x587 + x169*x205*x302 - x182*x220 + x203*x302 + x216*x453 - x293*x498 + x393*x453 - x415*x487 + (1.0/4.0)*x484 + (1.0/4.0)*x485 + (1.0/4.0)*x489 + (1.0/4.0)*x490 + (1.0/4.0)*x491 + (1.0/4.0)*x492 + (1.0/4.0)*x494 + (1.0/4.0)*x495 - x57*x586 + x583;
	double x589 = x273*x64;
	double x590 = x149*x64;
	double x591 = x230*x73;
	double x592 = -gtuu[0]*x576 - gtuu[4]*x231*x64 + x124*x434 + x124*x538 + x124*x539 - x156*x269 - x176*x303 - x182*x300 - x228*x230*x69 - x228*x321*x71 - x230*x321*x67 - x230*x590 + x230*x74 - x293*x458 + x293*x531 + x293*x582 - x303*x393 - x374*x399 + x379*x433 - x380*x90 + x383*x90 - x399*x589 - x399*x591 - x411*x540 + x532*x90 + x533*x90 + x534*x94 + x535*x94 - x578*x65;
	double x593 = gtuu[0]*x575 - gtuu[1]*x143*x281 + gtuu[1]*x581 + gtuu[4]*x283*x400 - x172*x454*x93 + x176*x302 + x177*x587 + x181*x432*x94 + x270*x400 + x271*x400 - x300*x510 + x301*x433 - x303*x577 - x312*x574 + x400*x529 + x421*x527 - 1.0/8.0*x526 + x530*x90 + x592;
	double x594 = x50*x52;
	double x595 = -x159*x269 - x188*x300 - x190*x300 - x197*x202*x415 - x197*x586 - x206*x429 + x216*x302 + x295*x453 + x299*x453 + x302*x320 + (1.0/4.0)*x543 + (1.0/4.0)*x544 + (1.0/4.0)*x545 + (1.0/4.0)*x546 + (1.0/4.0)*x547 + (1.0/4.0)*x548 + (1.0/4.0)*x549 + (1.0/4.0)*x550 + x592;
	double x596 = x146*x61;
	double x597 = (1.0/3.0)*dKhat_dx[2] + (2.0/3.0)*dtheta_dx[2];
	double x598 = (1.0/2.0)*x597;
	double x599 = gtuu[5]*x399;
	double x600 = 2*Atdd[2] + 2*x70;
	double x601 = (1.0/2.0)*x75;
	double x602 = dW_dx[0]*x601;
	double x603 = (1.0/3.0)*dKhat_dx[0] + (2.0/3.0)*dtheta_dx[0];
	double x604 = x146*x603;
	double x605 = 2*Atdd[0] + 2*x62;
	double x606 = dW_dx[2]*x601;
	double x607 = dAtdd_dx[12]*x146 - dAtdd_dx[2]*x146 + dgtdd_dx[12]*x596 - dgtdd_dx[2]*x596 - gtdd[2]*x604 + x149*x180 + x180*x274 + x180*x599 - x370*x57 - x375*x57 - x57*x579 + x598*x7 + x600*x602 - x605*x606;
	double x608 = 2*Atdd[4] + 2*x229;
	double x609 = dW_dx[1]*x601;
	double x610 = (1.0/3.0)*dKhat_dx[1] + (2.0/3.0)*dtheta_dx[1];
	double x611 = x146*x610;
	double x612 = 2*Atdd[3] + 2*x227;
	double x613 = -dAtdd_dx[10]*x146 + dAtdd_dx[15]*x146 - dgtdd_dx[10]*x596 + dgtdd_dx[15]*x596 - gtdd[4]*x611 + x149*x284 - x197*x374 - x197*x589 - x197*x591 + x274*x284 + x284*x599 + x598*x8 - x606*x612 + x608*x609;
	double x614 = dAtdd_dx[4]*x146;
	double x615 = x602*x608;
	double x616 = dgtdd_dx[4]*x596;
	double x617 = gtdd[4]*x604;
	double x618 = x197*x370;
	double x619 = x197*x579;
	double x620 = x197*x375;
	double x621 = x146*x597;
	double x622 = 2*Atdd[1] + 2*x66;
	double x623 = dAtdd_dx[13]*x146 + dgtdd_dx[13]*x596 + gtdd[1]*x621 + x149*x313 + x274*x313 + x313*x599 - x606*x622;
	double x624 = -x614 + x615 - x616 - x617 - x618 - x619 - x620 + x623;
	double x625 = -dAtdd_dx[8]*x146 - dgtdd_dx[8]*x596 - gtdd[2]*x611 - x374*x57 - x57*x589 - x57*x591 + x600*x609;
	double x626 = x623 + x625;
	double x627 = gtuu[5]*x71;
	double x628 = gtuu[5]*x230;
	double x629 = dAtdd_dx[3]*x146 - dAtdd_dx[7]*x146 + dgtdd_dx[3]*x596 - dgtdd_dx[7]*x596 - gtdd[1]*x611 + x150*x284 - x273*x313 + x284*x369 + x284*x627 - x313*x373 - x313*x628 - x602*x612 + (1.0/2.0)*x603*x8 + x609*x622;
	double x630 = (1.0/2.0)*x610;
	double x631 = dAtdd_dx[1]*x146 - dAtdd_dx[6]*x146 + dgtdd_dx[1]*x596 - dgtdd_dx[6]*x596 + gtdd[1]*x604 + x150*x313 - x180*x273 - x180*x373 - x180*x628 + x313*x369 + x313*x627 - x602*x622 + x605*x609 - x630*x7;
	double x632 = -x631;
	double x633 = -x629;
	double x634 = -1.0/8.0*gtdd[1]*gtdd[4]*x552*x83 - 1.0/4.0*gtdd[2]*x525*x6 + gtdd[3]*x563 - 1.0/4.0*x228*x71*x83 + x562*x67;
	double x635 = -1.0/4.0*gtdd[3]*x482*x6 + x519*x559 + x542*x568 + x634;
	double x636 = -x635;
	double x637 = x31*x556;
	double x638 = x225*x637;
	double x639 = -1.0/4.0*gtdd[3]*x499*x6 + x522*x559 + x551*x568 + x634;
	double x640 = -x639;
	double x641 = -1.0/8.0*gtdd[0]*gtdd[4]*x552*x83 + gtdd[1]*x563 + x512*x559 + x562*x63 - 1.0/4.0*x67*x71*x83;
	double x642 = -1.0/4.0*gtdd[1]*x482*x6 - 1.0/4.0*gtdd[2]*x522*x6 + x542*x569 + x641;
	double x643 = -x642;
	double x644 = x307*x637;
	double x645 = -1.0/4.0*gtdd[1]*x499*x6 - 1.0/4.0*gtdd[2]*x519*x6 + x551*x569 + x641;
	double x646 = -x645;
	double x647 = x34*x556;
	double x648 = x53*x647;
	double x649 = x307*x647;
	double x650 = x442*x556;
	double x651 = 2*Atdd[5] + 2*x398;
	double x652 = x399*x73;
	double x653 = dAtdd_dx[14]*x146 - 1.0/2.0*dAtdd_dx[5]*x6 - 1.0/2.0*dW_dx[2]*x600*x75 + dgtdd_dx[14]*x596 - 1.0/2.0*dgtdd_dx[5]*x6*x61 + gtdd[2]*x621 - 1.0/2.0*gtdd[5]*x6*x603 - 1.0/4.0*gtuu[2]*x162*x6*x63 - 1.0/4.0*gtuu[4]*x162*x6*x67 - 1.0/4.0*gtuu[5]*x162*x6*x71 + x57*x580 + x57*x590 + x57*x652 + x602*x651;
	double x654 = dAtdd_dx[11]*x146 - dAtdd_dx[16]*x146 + dgtdd_dx[11]*x596 - dgtdd_dx[16]*x596 - gtdd[4]*x621 + x162*x374 + x162*x589 + x162*x591 - x197*x580 - x197*x590 - x197*x652 + x29*x630 + x606*x608 - x609*x651;
	double x655 = x30*x556;
	double x656 = -2*x570*x655;
	double x657 = x25*x556;
	double x658 = x594*x657;
	double x659 = x28*x556;
	double x660 = x585*x659;
	double x661 = x30*x650;
	double x662 = -x654;
	double x663 = -x607;
	double x664 = -x624;
	double x665 = x614 - x615 + x616 + x617 + x618 + x619 + x620 + x625;
	double x666 = -x626;
	double x667 = -x665;
	double x668 = -x613;
	double x669 = -x567;
	double x670 = x594*x637;
	double x671 = -x565;
	double x672 = x585*x647;
	double x673 = x585*x655;
	double x674 = x594*x655;
	double x675 = -2*x555;
	double x676 = -2*x560;
	double x677 = (1.0/sqrt(x51));
	double x678 = x26*x677;
	double x679 = x17*x677;
	double x680 = x44*x678 - x46*x679;
	double x681 = x557*x680;
	double x682 = x42*x680;
	double x683 = x48*x678;
	double x684 = x637*x680;
	double x685 = x647*x680;
	double x686 = x48*x679;
	double x687 = 2*x44;
	double x688 = x679*x687;
	double x689 = 2*x46;
	double x690 = x678*x689;
	double x691 = x657*x683;
	double x692 = x15*x677;
	double x693 = x38*x692;
	double x694 = x48*x693;
	double x695 = x41*x686;
	double x696 = x40*x683;
	double x697 = x659*x686;
	double x698 = x14*x689;
	double x699 = x22*x687;
	double x700 = x556*x692;
	double x701 = x48*x700;
	double x702 = x669*x701;
	double x703 = x14*x24;
	double x704 = x671*x701;
	double x705 = x655*x683;
	double x706 = x22*x27;
	double x707 = x655*x686;
	double x708 = x27*x699*x700;
	double x709 = x24*x698*x700;
	*Psi4Re = x4*(-x224*x53 + x225*x24*x38*x629 - x225*x305 + x225*x38*x613*xyz[2] - x225*x561 - x225*x572 + x24*x307*x38*x631 + x24*x38*x44*x48*x52*x663 + x24*x38*x46*x48*x52*x664 + x24*x38*x46*x48*x52*x665 - x24*x38*x49*x52*x653 - x25*x555*x650 + x27*x307*x38*x633 + x27*x38*x44*x48*x52*x666 + x27*x38*x44*x48*x52*x667 + x27*x38*x46*x48*x52*x668 + x27*x38*x49*x52*x654 + x27*x38*x53*x632 - x28*x560*x650 + x307*x38*x624*xyz[2] + x307*x38*x626*xyz[2] - x307*x385 - x307*x397 - x307*x656 + x38*x44*x48*x52*x653*xyz[2] + x38*x46*x48*x52*x662*xyz[2] + x38*x53*x607*xyz[2] - x442*(-gtuu[1]*x151*x230 + x124*x409 + x124*x410 - x149*x414 + x155*x429 - x156*x400 - x159*x400 + x176*x426 - x182*x433 - x188*x433 - x190*x433 - x202*x416 - x204*x416 - x205*x434 - x214*x430 + x220*x441 - x231*x69 - x274*x414 - x293*x431 + x293*x435 + x300*x439 + x300*x440 + x312*x427 + x388*x428 - x393*x432 - ((x399)*(x399))*x73 + x399*x74 - x400*x417 - x400*x419 + (1.0/8.0)*x401 + x403*x90 + x404*x90 + x405*x94 + x406*x94 + (1.0/8.0)*x407 + x411*x412 + x411*x413 + x415*x420 + x421*x423 + x421*x425 + x437*x90 + x438*x90 - x65*x72) - x53*x558 - x53*x573 - x565*x566 - x565*x661 - x566*x567 - x567*x661 - x584*x585 - x585*x588 - x585*x637*x675 - x593*x594 - x594*x595 - x594*x647*x676 - x635*x649 - x635*x674 - x636*x638 - x636*x660 - x638*x640 - x639*x649 - x639*x674 - x640*x660 - x642*x648 - x642*x658 - x643*x644 - x643*x673 - x644*x646 - x645*x648 - x645*x658 - x646*x673 - x669*x670 - x669*x672 - x670*x671 - x671*x672);
	*Psi4Im = x4*(x14*x27*x676*x701 - x14*x662*x694 - x22*x24*x675*x701 + x22*x653*x694 - x224*x688 + x305*x690 + x385*x680 + x397*x680 - x40*x629*x690 - x40*x631*x680 + x40*x663*x686 + x41*x632*x688 - x41*x633*x680 - x41*x668*x683 - x558*x688 + x561*x690 + x565*x681 + x567*x681 + x572*x690 - x573*x688 - x584*x686 - x588*x686 + x593*x683 + x595*x683 + x607*x693*x699 - x613*x693*x698 - x624*x682 - x626*x682 + x635*x685 + x635*x705 - x636*x697 + x636*x709 + x639*x685 + x639*x705 - x640*x697 + x640*x709 + x642*x691 - x642*x708 + x643*x684 - x643*x707 + x645*x691 - x645*x708 + x646*x684 - x646*x707 + x656*x680 - x664*x696 - x665*x696 + x666*x695 + x667*x695 + x702*x703 - x702*x706 + x703*x704 - x704*x706);
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
