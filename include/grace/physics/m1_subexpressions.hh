/**
 * 
 */

#ifndef GRACE_PHYSICS_M1_SUBEXPR_HH
#define GRACE_PHYSICS_M1_SUBEXPR_HH

#include <grace_config.h>

#include <Kokkos_Core.hpp>

namespace grace {

static void KOKKOS_INLINE_FUNCTION
m1_closure_helpers(
    double v2, double W, double E, double vdotF, double vdotfh, double* out
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
    out[0] = x2;
    out[1] = x0*x3;
    out[2] = -x8;
    out[3] = x1*x9;
    out[4] = x10;
    out[5] = -W*x5*(-vdotF*(x4 - 1) + x6 + x7);
    out[6] = E*W*vdotfh;
    out[7] = -W;
    out[8] = W*v2;
    out[9] = W*(-E + vdotF + x2);
    out[10] = x10;
    out[11] = -W*x8;
}

static void KOKKOS_INLINE_FUNCTION 
m1_z_rootfind(
    double z, double dthin, double dthick,
    double v2, double vdotF, double vdotfh, double W, double F, double E, 
    double *coeffs, double *out
)
{
    *out = (-((F)*(F))*coeffs[7] - W*((coeffs[0])*(coeffs[0]))*coeffs[8] + 2*coeffs[0]*((coeffs[7])*(coeffs[7]))*vdotF + ((coeffs[9])*(coeffs[9])) - ((dthick)*(dthick))*(((F)*(F))*((coeffs[8])*(coeffs[8])) - ((coeffs[11])*(coeffs[11])) + ((coeffs[5])*(coeffs[5]))*v2 + coeffs[8]*vdotF*(2*W*(E*(3 - 2*((coeffs[7])*(coeffs[7]))) + vdotF*(2*((W)*(W)) - 1))/(2*((W)*(W)) + 1) + 2*coeffs[11])) - dthick*dthin*(2*F*coeffs[6]*coeffs[8] + 2*W*coeffs[1]*coeffs[8]*vdotF + 2*coeffs[11]*coeffs[1]*coeffs[7] + 2*coeffs[1]*coeffs[5]*coeffs[8] + 2*coeffs[5]*coeffs[6]*vdotfh) - dthick*(-2*((F)*(F))*((coeffs[7])*(coeffs[7]))*v2 + 2*W*coeffs[0]*coeffs[8]*vdotF + 2*coeffs[0]*coeffs[5]*coeffs[8] - 2*coeffs[11]*coeffs[9] + 2*coeffs[5]*coeffs[7]*vdotF) - ((dthin)*(dthin))*(E*coeffs[1] + W*((coeffs[1])*(coeffs[1]))*coeffs[8] - ((coeffs[1])*(coeffs[1]))*((coeffs[7])*(coeffs[7])) + 2*((coeffs[1])*(coeffs[1]))) - dthin*(2*F*coeffs[6]*coeffs[7] + 2*W*coeffs[0]*coeffs[1]*coeffs[8] + 2*coeffs[0]*coeffs[1] - 2*coeffs[1]*((coeffs[7])*(coeffs[7]))*vdotF + 2*coeffs[1]*coeffs[7]*coeffs[9]) + ((z)*(z))*((coeffs[0] + coeffs[1]*dthin + coeffs[2]*dthick)*(coeffs[0] + coeffs[1]*dthin + coeffs[2]*dthick)))/((E)*(E));
}

static void KOKKOS_INLINE_FUNCTION
m1_J(
    double dthin, double dthick, double *coeffs, double *out
)
{
    *out = coeffs[0] + coeffs[1]*dthin + coeffs[2]*dthick;
}

static void KOKKOS_INLINE_FUNCTION
m1_Hd(
    double dthin, double dthick, double v2, 
    double *Fd, double *fd, double *vd, double *coeffs,
    double *out
)
{
    double x0 = coeffs[6]*dthin;
    double x1 = coeffs[7]*(dthick*v2 - 1);
    double x2 = coeffs[0]*coeffs[7] + coeffs[1]*coeffs[7]*dthin - coeffs[5]*dthick;
    out[0] = Fd[0]*x1 - fd[0]*x0 + vd[0]*x2;
    out[1] = Fd[1]*x1 - fd[1]*x0 + vd[1]*x2;
    out[2] = Fd[2]*x1 - fd[2]*x0 + vd[2]*x2;
}

static void KOKKOS_INLINE_FUNCTION
m1_PUU(
    double dthin, double dthick,
    double vdotF, double vdotfh, double E, double F, double W,
    double *Fu, double *vu, double *guu, double (&out)[3][3]
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
    out[0][0] = ((Fu[0])*(Fu[0]))*x0 + dthick*(guu[0]*x3 + ((vu[0])*(vu[0]))*x5 + 2*x7*x9);
    out[0][1] = x15;
    out[0][2] = x18;
    out[1][0] = x15;
    out[1][1] = ((Fu[1])*(Fu[1]))*x0 + dthick*(guu[3]*x3 + ((vu[1])*(vu[1]))*x5 + 2*x12*x13);
    out[1][2] = x19;
    out[2][0] = x18;
    out[2][1] = x19;
    out[2][2] = ((Fu[2])*(Fu[2]))*x0 + dthick*(guu[5]*x3 + ((vu[2])*(vu[2]))*x5 + 2*x16*x17);
}

static void KOKKOS_INLINE_FUNCTION
m1_source(
    double W, double J, double E, double vdotF, 
    double alp, double ka, double ks, double eta,
    double *Hd, double *vd, double (&out) [4]
)
{
    double x0 = ka + ks;
    double x1 = W*(J*ka - eta);
    out[0] = W*alp*(J*ks + eta - x0*(E - vdotF));
    out[1] = -alp*(Hd[0]*x0 + vd[0]*x1);
    out[2] = -alp*(Hd[1]*x0 + vd[1]*x1);
    out[3] = -alp*(Hd[2]*x0 + vd[2]*x1);
}

static void KOKKOS_INLINE_FUNCTION
m1_jacobian(
    double dthin, double dthick, double W, double alp, double v2,
    double E, double F, double vdotfh, 
    double ka, double ks, double eta,
    double *vd, double *vu, 
    double *fd, double *fu,
    double (&out)[4][4] 
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
    out[0][0] = -x10*(-ks*x8 + x9);
    out[0][1] = x10*(vu[0]*x9 - x19*x20);
    out[0][2] = x10*(vu[1]*x9 - x20*x22);
    out[0][3] = x10*(vu[2]*x9 - x20*x24);
    out[1][0] = -x10*(vd[0]*x25 - x9*(vd[0]*x27 + x26));
    out[1][1] = x10*(x19*x29 - x9*(2*E*dthin*fd[0]*fu[0]*vdotfh*x11 + 2*E*dthin*fu[0]*vd[0]*x0*x1*x11 - vu[0]*x31 - vu[0]*x33 - x34));
    out[1][2] = x10*(x22*x29 - x9*(fu[1]*x36 - vu[1]*x31 - vu[1]*x33 + x21*x37));
    out[1][3] = x10*(x24*x29 - x9*(fu[2]*x36 - vu[2]*x31 - vu[2]*x33 + x23*x37));
    out[2][0] = -x10*(vd[1]*x25 - x9*(vd[1]*x27 + x38));
    out[2][1] = x10*(x19*x39 - x9*(fu[0]*x41 - vu[0]*x40 - vu[0]*x43 + x14*x42));
    out[2][2] = x10*(x22*x39 - x9*(2*E*dthin*fd[1]*fu[1]*vdotfh*x11 + 2*E*dthin*fu[1]*vd[1]*x0*x1*x11 - vu[1]*x40 - vu[1]*x43 - x34));
    out[2][3] = x10*(x24*x39 - x9*(fu[2]*x41 - vu[2]*x40 - vu[2]*x43 + x23*x42));
    out[3][0] = -x10*(vd[2]*x25 - x9*(vd[2]*x27 + x44));
    out[3][1] = x10*(x19*x45 - x9*(fu[0]*x47 - vu[0]*x46 - vu[0]*x49 + x14*x48));
    out[3][2] = x10*(x22*x45 - x9*(fu[1]*x47 - vu[1]*x46 - vu[1]*x49 + x21*x48));
    out[3][3] = x10*(x24*x45 - x9*(2*E*dthin*fd[2]*fu[2]*vdotfh*x11 + 2*E*dthin*fu[2]*vd[2]*x0*x1*x11 - vu[2]*x46 - vu[2]*x49 - x34));
}

static void KOKKOS_INLINE_FUNCTION
m1_fluid_to_lab_thick(
    double W, double Ht, double J, double alp, double *beta, double *vd, double *Hd, double (&out)[4]
)
{
    double x0 = (-Hd[0]*beta[0] - Hd[1]*beta[1] - Hd[2]*beta[2] + Ht)/alp;
    double x1 = (4.0/3.0)*J*W;
    out[0] = (4.0/3.0)*J*((W)*(W)) - 1.0/3.0*J - 2*W*x0;
    out[1] = W*(Hd[0] - vd[0]*x0 + vd[0]*x1);
    out[2] = W*(Hd[1] - vd[1]*x0 + vd[1]*x1);
    out[3] = W*(Hd[2] - vd[2]*x0 + vd[2]*x1);
}

static void KOKKOS_INLINE_FUNCTION
m1_wavespeeds(
    double dthin, double dthick, double alp, double F, double W, double betaDIR, double FDIR, double gammaDD, double vDIR, double *lp, double *lm
)
{
    double x0 = alp*fabs(FDIR)/F;
    double x1 = -betaDIR + alp*vDIR/W;
    double x2 = 2*((W)*(W)) + 1;
    double x3 = 1/(x2);
    double x4 = 2*W*alp*vDIR;
    double x5 = sqrt(((alp)*(alp))*(gammaDD*x2 - 2*((vDIR)*(vDIR))));
    *lm = dthick*fmin(x1, -betaDIR + x3*(x4 - x5)) - dthin*(betaDIR + x0);
    *lp = dthick*fmax(x1, -betaDIR + x3*(x4 + x5)) - dthin*(betaDIR - x0);
}

static void KOKKOS_FUNCTION 
m1_source_terms(
    double const gdd_x[3][3], double const gdd_y[3][3], double const gdd_z[3][3],
    double const betau_x[3], double const betau_y[3], double const betau_z[3],
    double const dalpha[3], double const betad[3], double const gdd[5], double const guu[5],
    double const alp, double const sqrtg,
    double const E,
    double const Fd[3],
    double const Fu[3],
    double const Puu[3][3],
    double (&out)[4]
)
{
    double x0 = 2*gdd[1];
    double x1 = 2*gdd[2];
    double x2 = 2*gdd_x[1][0] - gdd_y[0][0];
    double x3 = 2*gdd_x[2][0] - gdd_z[0][0];
    double x4 = 1.0*betad[0];
    double x5 = 1.0*betad[1];
    double x6 = 1.0*betad[2];
    double x7 = (1.0/2.0)*Puu[0][0];
    double x8 = gdd_x[0][1] - gdd_x[1][0] + gdd_y[0][0];
    double x9 = gdd_x[2][1] + gdd_y[2][0];
    double x10 = -gdd_z[1][0] + x9;
    double x11 = betau_x[0]*gdd[1] + betau_x[1]*gdd[3] + betau_x[2]*gdd[4] + betau_y[0]*gdd[0] + betau_y[1]*gdd[1] + betau_y[2]*gdd[2];
    double x12 = (1.0/2.0)*Puu[0][1];
    double x13 = gdd_x[1][1] - gdd_y[0][1] + gdd_y[1][0];
    double x14 = -gdd_z[0][1] + x9;
    double x15 = gdd_x[0][2] - gdd_x[2][0] + gdd_z[0][0];
    double x16 = gdd_x[1][2] + gdd_z[1][0];
    double x17 = -gdd_y[2][0] + x16;
    double x18 = betau_x[0]*gdd[2] + betau_x[1]*gdd[4] + betau_x[2]*gdd[5] + betau_z[0]*gdd[0] + betau_z[1]*gdd[1] + betau_z[2]*gdd[2];
    double x19 = (1.0/2.0)*Puu[0][2];
    double x20 = -gdd_y[0][2] + x16;
    double x21 = gdd_x[2][2] - gdd_z[0][2] + gdd_z[2][0];
    double x22 = 2*gdd[4];
    double x23 = -gdd_x[1][1] + 2*gdd_y[0][1];
    double x24 = 2*gdd_y[2][1] - gdd_z[1][1];
    double x25 = (1.0/2.0)*Puu[1][1];
    double x26 = gdd_y[0][2] + gdd_z[0][1];
    double x27 = -gdd_x[2][1] + x26;
    double x28 = gdd_y[1][2] - gdd_y[2][1] + gdd_z[1][1];
    double x29 = betau_y[0]*gdd[2] + betau_y[1]*gdd[4] + betau_y[2]*gdd[5] + betau_z[0]*gdd[1] + betau_z[1]*gdd[3] + betau_z[2]*gdd[4];
    double x30 = (1.0/2.0)*Puu[1][2];
    double x31 = -gdd_x[1][2] + x26;
    double x32 = gdd_y[2][2] - gdd_z[1][2] + gdd_z[2][1];
    double x33 = -gdd_x[2][2] + 2*gdd_z[0][2];
    double x34 = -gdd_y[2][2] + 2*gdd_z[1][2];
    double x35 = (1.0/2.0)*Puu[2][2];
    double x36 = alp*x7;
    double x37 = alp*x12;
    double x38 = alp*x19;
    double x39 = alp*x25;
    double x40 = alp*x30;
    double x41 = alp*x35;
    out[0] = sqrtg*(Fu[0]*dalpha[0] + Fu[1]*dalpha[1] + Fu[2]*dalpha[2] + x12*(x11 - x4*(gdd_x[1][1]*guu[1] + guu[0]*x8 + guu[2]*x10) - x5*(gdd_x[1][1]*guu[3] + guu[1]*x8 + guu[4]*x10) - x6*(gdd_x[1][1]*guu[4] + guu[2]*x8 + guu[5]*x10)) + x12*(x11 - x4*(gdd_y[0][0]*guu[0] + guu[1]*x13 + guu[2]*x14) - x5*(gdd_y[0][0]*guu[1] + guu[3]*x13 + guu[4]*x14) - x6*(gdd_y[0][0]*guu[2] + guu[4]*x13 + guu[5]*x14)) + x19*(x18 - x4*(gdd_x[2][2]*guu[2] + guu[0]*x15 + guu[1]*x17) - x5*(gdd_x[2][2]*guu[4] + guu[1]*x15 + guu[3]*x17) - x6*(gdd_x[2][2]*guu[5] + guu[2]*x15 + guu[4]*x17)) + x19*(x18 - x4*(gdd_z[0][0]*guu[0] + guu[1]*x20 + guu[2]*x21) - x5*(gdd_z[0][0]*guu[1] + guu[3]*x20 + guu[4]*x21) - x6*(gdd_z[0][0]*guu[2] + guu[4]*x20 + guu[5]*x21)) - x25*(-betau_y[0]*x0 - 2*betau_y[1]*gdd[3] - betau_y[2]*x22 + x4*(gdd_y[1][1]*guu[1] + guu[0]*x23 + guu[2]*x24) + x5*(gdd_y[1][1]*guu[3] + guu[1]*x23 + guu[4]*x24) + x6*(gdd_y[1][1]*guu[4] + guu[2]*x23 + guu[5]*x24)) + x30*(x29 - x4*(gdd_y[2][2]*guu[2] + guu[0]*x27 + guu[1]*x28) - x5*(gdd_y[2][2]*guu[4] + guu[1]*x27 + guu[3]*x28) - x6*(gdd_y[2][2]*guu[5] + guu[2]*x27 + guu[4]*x28)) + x30*(x29 - x4*(gdd_z[1][1]*guu[1] + guu[0]*x31 + guu[2]*x32) - x5*(gdd_z[1][1]*guu[3] + guu[1]*x31 + guu[4]*x32) - x6*(gdd_z[1][1]*guu[4] + guu[2]*x31 + guu[5]*x32)) - x35*(-betau_z[0]*x1 - betau_z[1]*x22 - 2*betau_z[2]*gdd[5] + x4*(gdd_z[2][2]*guu[2] + guu[0]*x33 + guu[1]*x34) + x5*(gdd_z[2][2]*guu[4] + guu[1]*x33 + guu[3]*x34) + x6*(gdd_z[2][2]*guu[5] + guu[2]*x33 + guu[4]*x34)) - x7*(-2*betau_x[0]*gdd[0] - betau_x[1]*x0 - betau_x[2]*x1 + x4*(gdd_x[0][0]*guu[0] + guu[1]*x2 + guu[2]*x3) + x5*(gdd_x[0][0]*guu[1] + guu[3]*x2 + guu[4]*x3) + x6*(gdd_x[0][0]*guu[2] + guu[4]*x2 + guu[5]*x3)));
    out[1] = sqrtg*(-E*dalpha[0] + Fd[0]*betau_x[0] + Fd[1]*betau_x[1] + Fd[2]*betau_x[2] + gdd_x[0][0]*x36 + gdd_x[0][1]*x37 + gdd_x[0][2]*x38 + gdd_x[1][0]*x37 + gdd_x[1][1]*x39 + gdd_x[1][2]*x40 + gdd_x[2][0]*x38 + gdd_x[2][1]*x40 + gdd_x[2][2]*x41);
    out[2] = sqrtg*(-E*dalpha[1] + Fd[0]*betau_y[0] + Fd[1]*betau_y[1] + Fd[2]*betau_y[2] + gdd_y[0][0]*x36 + gdd_y[0][1]*x37 + gdd_y[0][2]*x38 + gdd_y[1][0]*x37 + gdd_y[1][1]*x39 + gdd_y[1][2]*x40 + gdd_y[2][0]*x38 + gdd_y[2][1]*x40 + gdd_y[2][2]*x41);
    out[3] = sqrtg*(-E*dalpha[2] + Fd[0]*betau_z[0] + Fd[1]*betau_z[1] + Fd[2]*betau_z[2] + gdd_z[0][0]*x36 + gdd_z[0][1]*x37 + gdd_z[0][2]*x38 + gdd_z[1][0]*x37 + gdd_z[1][1]*x39 + gdd_z[1][2]*x40 + gdd_z[2][0]*x38 + gdd_z[2][1]*x40 + gdd_z[2][2]*x41);
}

}

#endif 