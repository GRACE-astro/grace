#include <grace_config.h>

#include <Kokkos_Core.hpp>

#include <grace/physics/m1_subexpressions.hh>

namespace grace {

void KOKKOS_INLINE_FUNCTION
m1_closure_helpers(
    double v2, double W, double E, double vdotF, double vdotfh, double* out
)
{
    double x0 = (W*W);
    double x1 = E - 2*vdotF;
    double x2 = E*(vdotfh*vdotfh);
    double x3 = 2*x0;
    double x4 = 1/(x3 + 1);
    double x5 = E*(x3 - 3);
    double x6 = (x0 - 1)*(-4*vdotF*x0 + x5);
    double x7 = (W*W*W);
    out[0] = x0*x1;
    out[1] = x0*x2;
    out[2] = -x4*x6;
    out[3] = x1*x7;
    out[4] = x2*x7;
    out[5] = -W*x4*(-vdotF*(x3 - 1) + x5 + x6);
    out[6] = E*W*vdotfh;
    out[7] = -W;
    out[8] = W*v2;
}

void KOKKOS_INLINE_FUNCTION 
m1_z_rootfind(
    double z, double dthin, double dthick,
    double v2, double E, 
    double *Fu, double *Fd, 
    double *vu, double *vd, 
    double *fu, double *fd,
    double *coeffs, double *out
)
{
    *out = ((z*z)*(coeffs[0] + coeffs[1]*dthin + coeffs[2]*dthick*coeffs[0] + coeffs[1]*dthin + coeffs[2]*dthick) - (-Fd[0]*(-coeffs[7]*dthick*v2 + coeffs[7]) - coeffs[6]*dthin*fd[0] + vd[0]*(coeffs[0]*coeffs[7] + coeffs[1]*coeffs[7]*dthin - coeffs[5]*dthick))*(-Fu[0]*(-coeffs[7]*dthick*v2 + coeffs[7]) - coeffs[6]*dthin*fu[0] + vu[0]*(coeffs[0]*coeffs[7] + coeffs[1]*coeffs[7]*dthin - coeffs[5]*dthick)) - (-Fd[1]*(-coeffs[7]*dthick*v2 + coeffs[7]) - coeffs[6]*dthin*fd[1] + vd[1]*(coeffs[0]*coeffs[7] + coeffs[1]*coeffs[7]*dthin - coeffs[5]*dthick))*(-Fu[1]*(-coeffs[7]*dthick*v2 + coeffs[7]) - coeffs[6]*dthin*fu[1] + vu[1]*(coeffs[0]*coeffs[7] + coeffs[1]*coeffs[7]*dthin - coeffs[5]*dthick)) - (-Fd[2]*(-coeffs[7]*dthick*v2 + coeffs[7]) - coeffs[6]*dthin*fd[2] + vd[2]*(coeffs[0]*coeffs[7] + coeffs[1]*coeffs[7]*dthin - coeffs[5]*dthick))*(-Fu[2]*(-coeffs[7]*dthick*v2 + coeffs[7]) - coeffs[6]*dthin*fu[2] + vu[2]*(coeffs[0]*coeffs[7] + coeffs[1]*coeffs[7]*dthin - coeffs[5]*dthick)))/(E*E);
}

void KOKKOS_INLINE_FUNCTION
m1_J(
    double dthin, double dthick, double *coeffs
)
{
    *out = coeffs[0] + coeffs[1]*dthin + coeffs[2]*dthick;
}

void KOKKOS_INLINE_FUNCTION
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

void KOKKOS_INLINE_FUNCTION
m1_PUU(
    double dthin, double dthick,
    double vdotF, double vdotfh, double E, double F, double W,
    double *Fu, double *vu,  double *Hu, double *guu, double (&out)[3][3]
)
{
    double x0 = E*dthin;
    double x1 = x0/(F*F);
    double x2 = Hu[0]*W;
    double x3 = (W*W);
    double x4 = 2*x3;
    double x5 = 4*x3;
    double x6 = -dthick*(x3 - 1)*(E*(x4 - 3) - vdotF*x5)/(x4 + 1) + (vdotfh*vdotfh)*x0*x3 + x3*(E - 2*vdotF);
    double x7 = x5*x6;
    double x8 = (1.0/3.0)*dthick;
    double x9 = Fu[0]*x1;
    double x10 = 3*x2;
    double x11 = Hu[1]*W;
    double x12 = 3*vu[0];
    double x13 = vu[0]*x7;
    double x14 = Fu[1]*x9 + x8*(guu[1]*x6 + vu[1]*x10 + vu[1]*x13 + x11*x12);
    double x15 = Hu[2]*W;
    double x16 = Fu[2]*x9 + x8*(guu[2]*x6 + vu[2]*x10 + vu[2]*x13 + x12*x15);
    double x17 = Fu[1]*Fu[2]*x1 + x8*(guu[4]*x6 + vu[1]*vu[2]*x7 + 3*vu[1]*x15 + 3*vu[2]*x11);
    out[0][0] = (Fu[0]*Fu[0])*x1 + x8*(guu[0]*x6 + (vu[0]*vu[0])*x7 + 6*vu[0]*x2);
    out[0][1] = x14;
    out[0][2] = x16;
    out[1][0] = x14;
    out[1][1] = (Fu[1]*Fu[1])*x1 + x8*(guu[3]*x6 + (vu[1]*vu[1])*x7 + 6*vu[1]*x11);
    out[1][2] = x17;
    out[2][0] = x16;
    out[2][1] = x17;
    out[2][2] = (Fu[2]*Fu[2])*x1 + x8*(guu[5]*x6 + (vu[2]*vu[2])*x7 + 6*vu[2]*x15);
}

void KOKKOS_INLINE_FUNCTION
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

void KOKKOS_INLINE_FUNCTION
m1_jacobian(
    double dthin, double dthick, double W,  
    double E, double F, double vdotfh, 
    double ka, double ks, double eta,
    double *vd, double *vu, 
    double *fd, double *fu,
    double (&out)[4][4] 
)
{
    double x0 = (W*W);
    double x1 = (vdotfh*vdotfh);
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

void KOKKOS_INLINE_FUNCTION
m1_fluid_to_lab_thick(
    double W, double Ht, double J, double alp, double *beta, double *vd, double *Hd, double (&out)[4]
)
{
    double x0 = (-Hd[0]*beta[0] - Hd[1]*beta[1] - Hd[2]*beta[2] + Ht)/alp;
    double x1 = (4.0/3.0)*J*W;
    out[0] = (4.0/3.0)*J*(W*W) - 1.0/3.0*J - 2*W*x0;
    out[1] = W*(Hd[0] - vd[0]*x0 + vd[0]*x1);
    out[2] = W*(Hd[1] - vd[1]*x0 + vd[1]*x1);
    out[3] = W*(Hd[2] - vd[2]*x0 + vd[2]*x1);
}

}