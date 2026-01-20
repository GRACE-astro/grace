
/****************************************************************************/
/*                        M1 helpers, SymPy generated                       */
/****************************************************************************/
#ifndef GRACE_M1_SUBEXPR_HH
#define GRACE_M1_SUBEXPR_HH

#include <Kokkos_Core.hpp>


static void KOKKOS_INLINE_FUNCTION
m1_closure_helpers(
	double W,
	double v2,
	double E,
	double vdotfh,
	double vdotF,
	double (*helpers)[12]
)
{
	double x0 = ((W)*(W));
	double x1 = E - 2*vdotF;
	double x2 = x0*x1;
	double x3 = E*((vdotfh)*(vdotfh));
	double x4 = 2*x0;
	double x5 = 1/(x4 + 1);
	double x6 = E*(x4 - 3);
	double x7 = (x0 - 1)*(-4*vdotF*x0 + x6);
	double x8 = x5*x7;
	double x9 = ((W)*(W)*(W));
	double x10 = x3*x9;
	(*helpers)[0] = x2;
	(*helpers)[1] = x0*x3;
	(*helpers)[2] = -x8;
	(*helpers)[3] = x1*x9;
	(*helpers)[4] = x10;
	(*helpers)[5] = -W*x5*(-vdotF*(x4 - 1) + x6 + x7);
	(*helpers)[6] = E*W*vdotfh;
	(*helpers)[7] = -W;
	(*helpers)[8] = W*v2;
	(*helpers)[9] = W*(-E + vdotF + x2);
	(*helpers)[10] = x10;
	(*helpers)[11] = -W*x8;
}

static void KOKKOS_INLINE_FUNCTION
m1_z_rootfind(
	double W,
	double dthin,
	double dthick,
	double v2,
	double E,
	double F,
	double vdotfh,
	double vdotF,
	double z,
	const double coeffs[12],
	double * __restrict__ out
)
{
	double x0 = W*coeffs[8];
	double x1 = ((F)*(F));
	double x2 = ((W)*(W));
	double x3 = ((coeffs[7])*(coeffs[7]));
	double x4 = 2*x3;
	double x5 = coeffs[0]*vdotF;
	double x6 = coeffs[1]*x0;
	double x7 = coeffs[1]*x3;
	double x8 = F*coeffs[6];
	double x9 = coeffs[1]*coeffs[7];
	double x10 = coeffs[5]*coeffs[8];
	double x11 = 2*dthick;
	double x12 = coeffs[0]*coeffs[1];
	double x13 = 2*x2;
	*out = -(((coeffs[0])*(coeffs[0]))*x0 + coeffs[1]*((dthin)*(dthin))*(E + 2*coeffs[1] + x6 - x7) - ((coeffs[9])*(coeffs[9])) + ((dthick)*(dthick))*(-((coeffs[11])*(coeffs[11])) + ((coeffs[5])*(coeffs[5]))*v2 + ((coeffs[8])*(coeffs[8]))*x1 + 2*coeffs[8]*vdotF*(W*(-E*(x4 - 3) + vdotF*(x13 - 1))/(x13 + 1) + coeffs[11])) + dthin*x11*(coeffs[11]*x9 + coeffs[1]*x10 + coeffs[5]*coeffs[6]*vdotfh + coeffs[8]*x8 + vdotF*x6) + 2*dthin*(coeffs[7]*x8 + coeffs[9]*x9 - vdotF*x7 + x0*x12 + x12) + x1*x2 + x11*(coeffs[0]*x10 - coeffs[11]*coeffs[9] + coeffs[5]*coeffs[7]*vdotF - v2*x1*x3 + x0*x5) - x4*x5 - ((z)*(z))*((coeffs[0] + coeffs[1]*dthin + coeffs[2]*dthick)*(coeffs[0] + coeffs[1]*dthin + coeffs[2]*dthick)))/((E)*(E));
}

static void KOKKOS_INLINE_FUNCTION
m1_J_Hd(
	double dthin,
	double dthick,
	double v2,
	const double Fd[3],
	const double vd[3],
	const double fd[3],
	const double coeffs[12],
	double * __restrict__ J,
	double (*Hd)[3]
)
{
	double x0 = coeffs[1]*dthin;
	double x1 = coeffs[6]*dthin;
	double x2 = coeffs[7]*(dthick*v2 - 1);
	double x3 = coeffs[0]*coeffs[7] - coeffs[5]*dthick + coeffs[7]*x0;
	*J = coeffs[0] + coeffs[2]*dthick + x0;
	(*Hd)[0] = Fd[0]*x2 - fd[0]*x1 + vd[0]*x3;
	(*Hd)[1] = Fd[1]*x2 - fd[1]*x1 + vd[1]*x3;
	(*Hd)[2] = Fd[2]*x2 - fd[2]*x1 + vd[2]*x3;
}

static void KOKKOS_INLINE_FUNCTION
m1_PUU(
	double W,
	double dthin,
	double dthick,
	double E,
	double F,
	const double Fu[3],
	const double vu[3],
	double vdotF,
	const double guu[6],
	double (*Puu)[3][3]
)
{
	double x0 = E*dthin/((F)*(F));
	double x1 = ((W)*(W));
	double x2 = 2*x1;
	double x3 = (E*(x2 - 1) - vdotF*x2)/(x2 + 1);
	double x4 = 4*x3;
	double x5 = x1*x4;
	double x6 = 1/(W);
	double x7 = W*vu[0];
	double x8 = W*(-E + vdotF + 3*x3);
	double x9 = Fu[0]*x6 + vu[0]*x8 - x4*x7;
	double x10 = Fu[0]*x0;
	double x11 = vu[0]*x5;
	double x12 = W*vu[1];
	double x13 = Fu[1]*x6 + vu[1]*x8 - x12*x4;
	double x14 = W*x9;
	double x15 = Fu[1]*x10 + dthick*(guu[1]*x3 + vu[1]*x11 + vu[1]*x14 + x13*x7);
	double x16 = W*vu[2];
	double x17 = Fu[2]*x6 + vu[2]*x8 - x16*x4;
	double x18 = Fu[2]*x10 + dthick*(guu[2]*x3 + vu[2]*x11 + vu[2]*x14 + x17*x7);
	double x19 = Fu[1]*Fu[2]*x0 + dthick*(guu[4]*x3 + vu[1]*vu[2]*x5 + x12*x17 + x13*x16);
	(*Puu)[0][0] = ((Fu[0])*(Fu[0]))*x0 + dthick*(guu[0]*x3 + ((vu[0])*(vu[0]))*x5 + 2*x7*x9);
	(*Puu)[0][1] = (*Puu)[1][0] = x15;
	(*Puu)[0][2] = (*Puu)[2][0] = x18;
	(*Puu)[1][1] = ((Fu[1])*(Fu[1]))*x0 + dthick*(guu[3]*x3 + ((vu[1])*(vu[1]))*x5 + 2*x12*x13);
	(*Puu)[1][2] = (*Puu)[2][1] = x19;
	(*Puu)[2][2] = ((Fu[2])*(Fu[2]))*x0 + dthick*(guu[5]*x3 + ((vu[2])*(vu[2]))*x5 + 2*x16*x17);
}

static void KOKKOS_INLINE_FUNCTION
m1_source(
	double W,
	double E,
	const double vd[3],
	double ka,
	double ks,
	double eta,
	double vdotF,
	double alp,
	double JJ,
	const double HHd[3],
	double * __restrict__ dE,
	double (*dF)[3]
)
{
	double x0 = ka + ks;
	double x1 = W*(JJ*ka - eta);
	*dE = W*alp*(JJ*ks + eta - x0*(E - vdotF));
	(*dF)[0] = -alp*(HHd[0]*x0 + vd[0]*x1);
	(*dF)[1] = -alp*(HHd[1]*x0 + vd[1]*x1);
	(*dF)[2] = -alp*(HHd[2]*x0 + vd[2]*x1);
}

static void KOKKOS_INLINE_FUNCTION
m1_jacobian(
	double W,
	double dthin,
	double dthick,
	double v2,
	double E,
	double F,
	const double vu[3],
	const double vd[3],
	const double fu[3],
	const double fd[3],
	double ka,
	double ks,
	double vdotfh,
	double alp,
	double (*J)[4][4]
)
{
	double x0 = ((W)*(W));
	double x1 = ((vdotfh)*(vdotfh));
	double x2 = dthin*x1;
	double x3 = x0 - 1;
	double x4 = 2*x0;
	double x5 = 1/(x4 + 1);
	double x6 = dthick*x5;
	double x7 = x6*(x4 - 3);
	double x8 = x0*x2 + x0 - x3*x7;
	double x9 = ka + ks;
	double x10 = W*alp;
	double x11 = 1/(F);
	double x12 = E*x11;
	double x13 = x12*x2;
	double x14 = fu[0]*x13;
	double x15 = dthin*vdotfh;
	double x16 = x12*x15;
	double x17 = x16 - 1;
	double x18 = x17 + 2*x3*x6;
	double x19 = -vu[0]*x18 + x14;
	double x20 = ks*x4;
	double x21 = fu[1]*x13;
	double x22 = -vu[1]*x18 + x21;
	double x23 = fu[2]*x13;
	double x24 = -vu[2]*x18 + x23;
	double x25 = ka*x8;
	double x26 = fd[0]*x15;
	double x27 = x0*(x2 - x7 + 1);
	double x28 = ka*x4;
	double x29 = vd[0]*x28;
	double x30 = x0*(dthick*(2*v2 + x5/x0) + 2*x16 - 2);
	double x31 = vd[0]*x30;
	double x32 = dthin*x12;
	double x33 = fd[0]*x32;
	double x34 = dthick*v2 + x17;
	double x35 = 2*x12;
	double x36 = x26*x35;
	double x37 = vd[0]*x4;
	double x38 = fd[1]*x15;
	double x39 = vd[1]*x28;
	double x40 = fd[1]*x32;
	double x41 = x35*x38;
	double x42 = vd[1]*x4;
	double x43 = vd[1]*x30;
	double x44 = fd[2]*x15;
	double x45 = vd[2]*x28;
	double x46 = fd[2]*x32;
	double x47 = x35*x44;
	double x48 = vd[2]*x4;
	double x49 = vd[2]*x30;
	(*J)[0][0] = -x10*(-ks*x8 + x9);
	(*J)[0][1] = x10*(vu[0]*x9 - x19*x20);
	(*J)[0][2] = x10*(vu[1]*x9 - x20*x22);
	(*J)[0][3] = x10*(vu[2]*x9 - x20*x24);
	(*J)[1][0] = -x10*(vd[0]*x25 - x9*(vd[0]*x27 + x26));
	(*J)[1][1] = x10*(x19*x29 - x9*(2*E*dthin*fd[0]*fu[0]*vdotfh*x11 + 2*E*dthin*fu[0]*vd[0]*x0*x1*x11 - vu[0]*x31 - vu[0]*x33 - x34));
	(*J)[1][2] = x10*(x22*x29 - x9*(fu[1]*x36 - vu[1]*x31 - vu[1]*x33 + x21*x37));
	(*J)[1][3] = x10*(x24*x29 - x9*(fu[2]*x36 - vu[2]*x31 - vu[2]*x33 + x23*x37));
	(*J)[2][0] = -x10*(vd[1]*x25 - x9*(vd[1]*x27 + x38));
	(*J)[2][1] = x10*(x19*x39 - x9*(fu[0]*x41 - vu[0]*x40 - vu[0]*x43 + x14*x42));
	(*J)[2][2] = x10*(x22*x39 - x9*(2*E*dthin*fd[1]*fu[1]*vdotfh*x11 + 2*E*dthin*fu[1]*vd[1]*x0*x1*x11 - vu[1]*x40 - vu[1]*x43 - x34));
	(*J)[2][3] = x10*(x24*x39 - x9*(fu[2]*x41 - vu[2]*x40 - vu[2]*x43 + x23*x42));
	(*J)[3][0] = -x10*(vd[2]*x25 - x9*(vd[2]*x27 + x44));
	(*J)[3][1] = x10*(x19*x45 - x9*(fu[0]*x47 - vu[0]*x46 - vu[0]*x49 + x14*x48));
	(*J)[3][2] = x10*(x22*x45 - x9*(fu[1]*x47 - vu[1]*x46 - vu[1]*x49 + x21*x48));
	(*J)[3][3] = x10*(x24*x45 - x9*(2*E*dthin*fd[2]*fu[2]*vdotfh*x11 + 2*E*dthin*fu[2]*vd[2]*x0*x1*x11 - vu[2]*x46 - vu[2]*x49 - x34));
}

static void KOKKOS_INLINE_FUNCTION
m1_fluid_to_lab_thick(
	double W,
	const double vd[3],
	double alp,
	const double betau[3],
	double JJ,
	const double HHd[3],
	double Ht,
	double * __restrict__ Et,
	double (*Fdt)[3]
)
{
	double x0 = (-HHd[0]*betau[0] - HHd[1]*betau[1] - HHd[2]*betau[2] + Ht)/alp;
	double x1 = (4.0/3.0)*JJ*W;
	*Et = (4.0/3.0)*JJ*((W)*(W)) - 1.0/3.0*JJ - 2*W*x0;
	(*Fdt)[0] = W*(HHd[0] - vd[0]*x0 + vd[0]*x1);
	(*Fdt)[1] = W*(HHd[1] - vd[1]*x0 + vd[1]*x1);
	(*Fdt)[2] = W*(HHd[2] - vd[2]*x0 + vd[2]*x1);
}

static void KOKKOS_INLINE_FUNCTION
m1_wavespeeds(
	double W,
	double dthin,
	double dthick,
	double F,
	double alp,
	double vDIR,
	double gammaDD,
	double betaDIR,
	double FDIR,
	double * __restrict__ cm,
	double * __restrict__ cp
)
{
	double x0 = alp*fabs(FDIR)/F;
	double x1 = -betaDIR + alp*vDIR/W;
	double x2 = 2*((W)*(W)) + 1;
	double x3 = 1/(x2);
	double x4 = 2*W*alp*vDIR;
	double x5 = sqrt(((alp)*(alp))*(gammaDD*x2 - 2*((vDIR)*(vDIR))));
	*cm = dthick*fmin(x1, -betaDIR + x3*(x4 - x5)) - dthin*(betaDIR + x0);
	*cp = dthick*fmax(x1, -betaDIR + x3*(x4 + x5)) - dthin*(betaDIR - x0);
}

static void KOKKOS_INLINE_FUNCTION
m1_geom_source_terms(
	double E,
	const double Fd[3],
	const double Fu[3],
	double alp,
	const double Kdd[6],
	const double dalpha[3],
	const double dgdd_dx[18],
	const double dbetau_dx[9],
	const double Puu[3][3],
	double * __restrict__ dE,
	double (*dF)[3]
)
{
	double x0 = Puu[0][0]*alp;
	double x1 = Puu[1][1]*alp;
	double x2 = Puu[2][2]*alp;
	double x3 = Puu[0][1]*alp;
	double x4 = Puu[0][2]*alp;
	double x5 = Puu[1][2]*alp;
	double x6 = (1.0/2.0)*x0;
	double x7 = (1.0/2.0)*x1;
	double x8 = (1.0/2.0)*x2;
	*dE = -Fu[0]*dalpha[0] - Fu[1]*dalpha[1] - Fu[2]*dalpha[2] + Kdd[0]*x0 + 2*Kdd[1]*x3 + 2*Kdd[2]*x4 + Kdd[3]*x1 + 2*Kdd[4]*x5 + Kdd[5]*x2;
	(*dF)[0] = -E*dalpha[0] + Fd[0]*dbetau_dx[0] + Fd[1]*dbetau_dx[1] + Fd[2]*dbetau_dx[2] + dgdd_dx[0]*x6 + dgdd_dx[1]*x3 + dgdd_dx[2]*x4 + dgdd_dx[3]*x7 + dgdd_dx[4]*x5 + dgdd_dx[5]*x8;
	(*dF)[1] = -E*dalpha[1] + Fd[0]*dbetau_dx[3] + Fd[1]*dbetau_dx[4] + Fd[2]*dbetau_dx[5] + dgdd_dx[10]*x5 + dgdd_dx[11]*x8 + dgdd_dx[6]*x6 + dgdd_dx[7]*x3 + dgdd_dx[8]*x4 + dgdd_dx[9]*x7;
	(*dF)[2] = -E*dalpha[2] + Fd[0]*dbetau_dx[6] + Fd[1]*dbetau_dx[7] + Fd[2]*dbetau_dx[8] + dgdd_dx[12]*x6 + dgdd_dx[13]*x3 + dgdd_dx[14]*x4 + dgdd_dx[15]*x7 + dgdd_dx[16]*x5 + dgdd_dx[17]*x8;
}

#endif 
