#ifndef GRACE_GRMHD_EXPR_HH
#define GRACE_GRMHD_EXPR_HH 

#include <Kokkos_Core.hpp>

namespace grace {

static void KOKKOS_INLINE_FUNCTION 
grmhd_source_terms(
    double const alp, double const sqrtg, double const * __restrict__ gdd, double const * __restrict__ guu, double const * __restrict__ betau 
    double const rho, double const eps, double const press, double const * __restrict__ zvec, double const * __restrict__ Bvec, 
    double const gdd_x[3][3], double const gdd_y[3][3], double const gdd_z[3][3],
    double const betau_x[3], double const betau_y[3], double const betau_z[3],
    double const dalpha[3], double *out
)
{
    double x0 = gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2];
    double x1 = x0*zvec[0];
    double x2 = gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2];
    double x3 = x2*zvec[1];
    double x4 = gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2];
    double x5 = x4*zvec[2];
    double x6 = x1 + x3 + x5 + 1;
    double x7 = sqrt(x6);
    double x8 = 1/(x7);
    double x9 = Bvec[0]*gdd[0] + Bvec[1]*gdd[1] + Bvec[2]*gdd[2];
    double x10 = Bvec[0]*gdd[1] + Bvec[1]*gdd[3] + Bvec[2]*gdd[4];
    double x11 = Bvec[0]*gdd[2] + Bvec[1]*gdd[4] + Bvec[2]*gdd[5];
    double x12 = Bvec[0]*x9 + Bvec[1]*x10 + Bvec[2]*x11;
    double x13 = Bvec[0]*x0 + Bvec[1]*x2 + Bvec[2]*x4;
    double x14 = x6*(press + rho*(eps + 1));
    double x15 = -Bvec[0]*x13 + x12*zvec[0] + x14*zvec[0];
    double x16 = -Bvec[1]*x13 + x12*zvec[1] + x14*zvec[1];
    double x17 = -Bvec[2]*x13 + x12*zvec[2] + x14*zvec[2];
    double x18 = (alp*alp);
    double x19 = 2*x6;
    double x20 = x18*x19;
    double x21 = x20*(Bvec[0] + x13*zvec[0]);
    double x22 = guu[0]*zvec[0] + guu[1]*zvec[1] + guu[2]*zvec[2];
    double x23 = alp*zvec[0] - betau[0]*x7;
    double x24 = alp*zvec[1] - betau[1]*x7;
    double x25 = alp*zvec[2] - betau[2]*x7;
    double x26 = press*x20 + x12*x18 + (Bvec[0]*(gdd[0]*x23 + gdd[1]*x24 + gdd[2]*x25 + x7*(betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2])) + Bvec[1]*(gdd[1]*x23 + gdd[3]*x24 + gdd[4]*x25 + x7*(betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4])) + Bvec[2]*(gdd[2]*x23 + gdd[4]*x24 + gdd[5]*x25 + x7*(betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5]))*Bvec[0]*(gdd[0]*x23 + gdd[1]*x24 + gdd[2]*x25 + x7*(betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2])) + Bvec[1]*(gdd[1]*x23 + gdd[3]*x24 + gdd[4]*x25 + x7*(betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4])) + Bvec[2]*(gdd[2]*x23 + gdd[4]*x24 + gdd[5]*x25 + x7*(betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5])));
    double x27 = x15*x26;
    double x28 = Bvec[0]*x21 - x22*x27;
    double x29 = (1.0/4.0)*x28;
    double x30 = gdd_x[0][0]*x29;
    double x31 = 1/((x6*x6));
    double x32 = x31/x18;
    double x33 = betau[0]*x32;
    double x34 = guu[1]*zvec[0] + guu[3]*zvec[1] + guu[4]*zvec[2];
    double x35 = Bvec[1]*x21 - x27*x34;
    double x36 = (1.0/4.0)*x33;
    double x37 = guu[2]*zvec[0] + guu[4]*zvec[1] + guu[5]*zvec[2];
    double x38 = Bvec[2]*x21 - x27*x37;
    double x39 = x20*(Bvec[1] + x13*zvec[1]);
    double x40 = x16*x26;
    double x41 = Bvec[0]*x39 - x22*x40;
    double x42 = Bvec[1]*x39 - x34*x40;
    double x43 = Bvec[2]*x39 - x37*x40;
    double x44 = x20*(Bvec[2] + x13*zvec[2]);
    double x45 = x17*x26;
    double x46 = Bvec[0]*x44 - x22*x45;
    double x47 = Bvec[1]*x44 - x34*x45;
    double x48 = Bvec[2]*x44 - x37*x45;
    double x49 = betau[1]*x32;
    double x50 = (1.0/4.0)*x49;
    double x51 = betau[2]*x32;
    double x52 = (1.0/4.0)*x51;
    double x53 = (1.0/2.0)*x32;
    double x54 = x0*x12 + x0*x14 - x13*x9;
    double x55 = -x10*x13 + x12*x2 + x14*x2;
    double x56 = -x11*x13 + x12*x4 + x14*x4;
    double x57 = (3.0/2.0)*(x12*(2*x1 + 2*x3 + 2*x5 + 1) - (x13*x13) - x19*(press - x14))/x6;
    double x58 = x31/alp;
    double x59 = (1.0/4.0)*x58;
    double x60 = x35*x59;
    double x61 = x38*x59;
    double x62 = x41*x59;
    double x63 = x42*x59;
    double x64 = x43*x59;
    double x65 = x46*x59;
    double x66 = x47*x59;
    double x67 = x48*x59;
    double x68 = x29*x58;
    out[0] = -betau_x[0]*x53*(gdd[0]*x28 + gdd[1]*x35 + gdd[2]*x38) - betau_x[1]*x53*(gdd[1]*x28 + gdd[3]*x35 + gdd[4]*x38) - betau_x[2]*x53*(gdd[2]*x28 + gdd[4]*x35 + gdd[5]*x38) - betau_y[0]*x53*(gdd[0]*x41 + gdd[1]*x42 + gdd[2]*x43) - betau_y[1]*x53*(gdd[1]*x41 + gdd[3]*x42 + gdd[4]*x43) - betau_y[2]*x53*(gdd[2]*x41 + gdd[4]*x42 + gdd[5]*x43) - betau_z[0]*x53*(gdd[0]*x46 + gdd[1]*x47 + gdd[2]*x48) - betau_z[1]*x53*(gdd[1]*x46 + gdd[3]*x47 + gdd[4]*x48) - betau_z[2]*x53*(gdd[2]*x46 + gdd[4]*x47 + gdd[5]*x48) - dalpha[0]*x15*x8 - dalpha[1]*x16*x8 - dalpha[2]*x17*x8 - gdd_x[0][1]*x35*x36 - gdd_x[0][2]*x36*x38 - gdd_x[1][0]*x36*x41 - gdd_x[1][1]*x36*x42 - gdd_x[1][2]*x36*x43 - gdd_x[2][0]*x36*x46 - gdd_x[2][1]*x36*x47 - gdd_x[2][2]*x36*x48 - gdd_y[0][0]*x29*x49 - gdd_y[0][1]*x35*x50 - gdd_y[0][2]*x38*x50 - gdd_y[1][0]*x41*x50 - gdd_y[1][1]*x42*x50 - gdd_y[1][2]*x43*x50 - gdd_y[2][0]*x46*x50 - gdd_y[2][1]*x47*x50 - gdd_y[2][2]*x48*x50 - gdd_z[0][0]*x29*x51 - gdd_z[0][1]*x35*x52 - gdd_z[0][2]*x38*x52 - gdd_z[1][0]*x41*x52 - gdd_z[1][1]*x42*x52 - gdd_z[1][2]*x43*x52 - gdd_z[2][0]*x46*x52 - gdd_z[2][1]*x47*x52 - gdd_z[2][2]*x48*x52 - x30*x33;
    out[1] = -dalpha[0]*x57 - gdd_x[0][1]*x60 - gdd_x[0][2]*x61 - gdd_x[1][0]*x62 - gdd_x[1][1]*x63 - gdd_x[1][2]*x64 - gdd_x[2][0]*x65 - gdd_x[2][1]*x66 - gdd_x[2][2]*x67 - x30*x58 + x54*x8*(betau_x[0]*gdd[0] + betau_x[1]*gdd[1] + betau_x[2]*gdd[2]) + x55*x8*(betau_x[0]*gdd[1] + betau_x[1]*gdd[3] + betau_x[2]*gdd[4]) + x56*x8*(betau_x[0]*gdd[2] + betau_x[1]*gdd[4] + betau_x[2]*gdd[5]);
    out[2] = -dalpha[1]*x57 - gdd_y[0][0]*x68 - gdd_y[0][1]*x60 - gdd_y[0][2]*x61 - gdd_y[1][0]*x62 - gdd_y[1][1]*x63 - gdd_y[1][2]*x64 - gdd_y[2][0]*x65 - gdd_y[2][1]*x66 - gdd_y[2][2]*x67 + x54*x8*(betau_y[0]*gdd[0] + betau_y[1]*gdd[1] + betau_y[2]*gdd[2]) + x55*x8*(betau_y[0]*gdd[1] + betau_y[1]*gdd[3] + betau_y[2]*gdd[4]) + x56*x8*(betau_y[0]*gdd[2] + betau_y[1]*gdd[4] + betau_y[2]*gdd[5]);
    out[3] = -dalpha[2]*x57 - gdd_z[0][0]*x68 - gdd_z[0][1]*x60 - gdd_z[0][2]*x61 - gdd_z[1][0]*x62 - gdd_z[1][1]*x63 - gdd_z[1][2]*x64 - gdd_z[2][0]*x65 - gdd_z[2][1]*x66 - gdd_z[2][2]*x67 + x54*x8*(betau_z[0]*gdd[0] + betau_z[1]*gdd[1] + betau_z[2]*gdd[2]) + x55*x8*(betau_z[0]*gdd[1] + betau_z[1]*gdd[3] + betau_z[2]*gdd[4]) + x56*x8*(betau_z[0]*gdd[2] + betau_z[1]*gdd[4] + betau_z[2]*gdd[5]);    
}


static void KOKKOS_INLINE_FUNCTION 
grmhd_x_flux(
    double const alp, double const sqrtg, double const * __restrict__ gdd, double const * __restrict__ guu, double const * __restrict__ betau,
    double const rho, double const eps, double const press, double const * __restrict__ zvec, double const * __restrict__ Bvec, 
    double * out 
)
{
    double x0 = gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2];
    double x1 = x0*zvec[0];
    double x2 = gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2];
    double x3 = x2*zvec[1];
    double x4 = gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2];
    double x5 = x4*zvec[2];
    double x6 = x1 + x3 + x5 + 1;
    double x7 = sqrt(x6);
    double x8 = 1/(x7);
    double x9 = alp*zvec[0];
    double x10 = rho*x7;
    double x11 = x10*(-betau[0] + x8*x9);
    double x12 = Bvec[0]*gdd[0] + Bvec[1]*gdd[1] + Bvec[2]*gdd[2];
    double x13 = Bvec[0]*gdd[1] + Bvec[1]*gdd[3] + Bvec[2]*gdd[4];
    double x14 = Bvec[0]*gdd[2] + Bvec[1]*gdd[4] + Bvec[2]*gdd[5];
    double x15 = Bvec[0]*x12 + Bvec[1]*x13 + Bvec[2]*x14;
    double x16 = Bvec[0]*x0 + Bvec[1]*x2 + Bvec[2]*x4;
    double x17 = press + rho*(eps + 1);
    double x18 = x17*x6;
    double x19 = -Bvec[0]*x16 + x15*zvec[0] + x18*zvec[0];
    double x20 = 2*x6;
    double x21 = betau[0]*x8;
    double x22 = (alp*alp);
    double x23 = x20*x22;
    double x24 = x23*(Bvec[0] + x16*zvec[0]);
    double x25 = -betau[0]*x7 + x9;
    double x26 = alp*zvec[1] - betau[1]*x7;
    double x27 = alp*zvec[2] - betau[2]*x7;
    double x28 = x19*(press*x23 + x15*x22 + (Bvec[0]*(gdd[0]*x25 + gdd[1]*x26 + gdd[2]*x27 + x7*(betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2])) + Bvec[1]*(gdd[1]*x25 + gdd[3]*x26 + gdd[4]*x27 + x7*(betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4])) + Bvec[2]*(gdd[2]*x25 + gdd[4]*x26 + gdd[5]*x27 + x7*(betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5]))*Bvec[0]*(gdd[0]*x25 + gdd[1]*x26 + gdd[2]*x27 + x7*(betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2])) + Bvec[1]*(gdd[1]*x25 + gdd[3]*x26 + gdd[4]*x27 + x7*(betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4])) + Bvec[2]*(gdd[2]*x25 + gdd[4]*x26 + gdd[5]*x27 + x7*(betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5]))));
    double x29 = Bvec[0]*x24 - x28*(guu[0]*zvec[0] + guu[1]*zvec[1] + guu[2]*zvec[2]);
    double x30 = Bvec[1]*x24 - x28*(guu[1]*zvec[0] + guu[3]*zvec[1] + guu[4]*zvec[2]);
    double x31 = Bvec[2]*x24 - x28*(guu[2]*zvec[0] + guu[4]*zvec[1] + guu[5]*zvec[2]);
    double x32 = (1.0/2.0)/(alp*(x6*x6));
    out[0] = x11;
    out[1] = -alp*(rho*zvec[0] - x19*x8) + (1.0/2.0)*betau[0]*(2*x10 - (x15*(2*x1 + 2*x3 + 2*x5 + 1) - (x16*x16) + x20*(-press + x17*x6))/x6);
    out[2] = -x21*(x0*x15 + x0*x18 - x12*x16) - x32*(gdd[0]*x29 + gdd[1]*x30 + gdd[2]*x31);
    out[3] = -x21*(-x13*x16 + x15*x2 + x18*x2) - x32*(gdd[1]*x29 + gdd[3]*x30 + gdd[4]*x31);
    out[4] = -x21*(-x14*x16 + x15*x4 + x18*x4) - x32*(gdd[2]*x29 + gdd[4]*x30 + gdd[5]*x31);
    out[5] = s*x11;
}

static void KOKKOS_INLINE_FUNCTION 
grmhd_y_flux(
    double const alp, double const sqrtg, double const * __restrict__ gdd, double const * __restrict__ guu, double const * __restrict__ betau,
    double const rho, double const eps, double const press, double const * __restrict__ zvec, double const * __restrict__ Bvec, 
    double * out 
)
{
    double x0 = gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2];
    double x1 = x0*zvec[0];
    double x2 = gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2];
    double x3 = x2*zvec[1];
    double x4 = gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2];
    double x5 = x4*zvec[2];
    double x6 = x1 + x3 + x5 + 1;
    double x7 = sqrt(x6);
    double x8 = 1/(x7);
    double x9 = alp*zvec[1];
    double x10 = rho*x7;
    double x11 = x10*(-betau[1] + x8*x9);
    double x12 = Bvec[0]*gdd[0] + Bvec[1]*gdd[1] + Bvec[2]*gdd[2];
    double x13 = Bvec[0]*gdd[1] + Bvec[1]*gdd[3] + Bvec[2]*gdd[4];
    double x14 = Bvec[0]*gdd[2] + Bvec[1]*gdd[4] + Bvec[2]*gdd[5];
    double x15 = Bvec[0]*x12 + Bvec[1]*x13 + Bvec[2]*x14;
    double x16 = Bvec[0]*x0 + Bvec[1]*x2 + Bvec[2]*x4;
    double x17 = press + rho*(eps + 1);
    double x18 = x17*x6;
    double x19 = -Bvec[1]*x16 + x15*zvec[1] + x18*zvec[1];
    double x20 = 2*x6;
    double x21 = betau[1]*x8;
    double x22 = (alp*alp);
    double x23 = x20*x22;
    double x24 = x23*(Bvec[1] + x16*zvec[1]);
    double x25 = alp*zvec[0] - betau[0]*x7;
    double x26 = -betau[1]*x7 + x9;
    double x27 = alp*zvec[2] - betau[2]*x7;
    double x28 = x19*(press*x23 + x15*x22 + (Bvec[0]*(gdd[0]*x25 + gdd[1]*x26 + gdd[2]*x27 + x7*(betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2])) + Bvec[1]*(gdd[1]*x25 + gdd[3]*x26 + gdd[4]*x27 + x7*(betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4])) + Bvec[2]*(gdd[2]*x25 + gdd[4]*x26 + gdd[5]*x27 + x7*(betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5]))*Bvec[0]*(gdd[0]*x25 + gdd[1]*x26 + gdd[2]*x27 + x7*(betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2])) + Bvec[1]*(gdd[1]*x25 + gdd[3]*x26 + gdd[4]*x27 + x7*(betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4])) + Bvec[2]*(gdd[2]*x25 + gdd[4]*x26 + gdd[5]*x27 + x7*(betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5]))));
    double x29 = Bvec[0]*x24 - x28*(guu[0]*zvec[0] + guu[1]*zvec[1] + guu[2]*zvec[2]);
    double x30 = Bvec[1]*x24 - x28*(guu[1]*zvec[0] + guu[3]*zvec[1] + guu[4]*zvec[2]);
    double x31 = Bvec[2]*x24 - x28*(guu[2]*zvec[0] + guu[4]*zvec[1] + guu[5]*zvec[2]);
    double x32 = (1.0/2.0)/(alp*(x6*x6));
    out[0] = x11;
    out[1] = -alp*(rho*zvec[1] - x19*x8) + (1.0/2.0)*betau[1]*(2*x10 - (x15*(2*x1 + 2*x3 + 2*x5 + 1) - (x16*x16) + x20*(-press + x17*x6))/x6);
    out[2] = -x21*(x0*x15 + x0*x18 - x12*x16) - x32*(gdd[0]*x29 + gdd[1]*x30 + gdd[2]*x31);
    out[3] = -x21*(-x13*x16 + x15*x2 + x18*x2) - x32*(gdd[1]*x29 + gdd[3]*x30 + gdd[4]*x31);
    out[4] = -x21*(-x14*x16 + x15*x4 + x18*x4) - x32*(gdd[2]*x29 + gdd[4]*x30 + gdd[5]*x31);
    out[5] = s*x11;
}

static void KOKKOS_INLINE_FUNCTION 
grmhd_z_flux(
    double const alp, double const sqrtg, double const * __restrict__ gdd, double const * __restrict__ guu, double const * __restrict__ betau,
    double const rho, double const eps, double const press, double const * __restrict__ zvec, double const * __restrict__ Bvec, 
    double * out 
)
{
    double x0 = gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2];
    double x1 = x0*zvec[0];
    double x2 = gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2];
    double x3 = x2*zvec[1];
    double x4 = gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2];
    double x5 = x4*zvec[2];
    double x6 = x1 + x3 + x5 + 1;
    double x7 = sqrt(x6);
    double x8 = 1/(x7);
    double x9 = alp*zvec[2];
    double x10 = rho*x7;
    double x11 = x10*(-betau[2] + x8*x9);
    double x12 = Bvec[0]*gdd[0] + Bvec[1]*gdd[1] + Bvec[2]*gdd[2];
    double x13 = Bvec[0]*gdd[1] + Bvec[1]*gdd[3] + Bvec[2]*gdd[4];
    double x14 = Bvec[0]*gdd[2] + Bvec[1]*gdd[4] + Bvec[2]*gdd[5];
    double x15 = Bvec[0]*x12 + Bvec[1]*x13 + Bvec[2]*x14;
    double x16 = Bvec[0]*x0 + Bvec[1]*x2 + Bvec[2]*x4;
    double x17 = press + rho*(eps + 1);
    double x18 = x17*x6;
    double x19 = -Bvec[2]*x16 + x15*zvec[2] + x18*zvec[2];
    double x20 = 2*x6;
    double x21 = betau[2]*x8;
    double x22 = (alp*alp);
    double x23 = x20*x22;
    double x24 = x23*(Bvec[2] + x16*zvec[2]);
    double x25 = alp*zvec[0] - betau[0]*x7;
    double x26 = alp*zvec[1] - betau[1]*x7;
    double x27 = -betau[2]*x7 + x9;
    double x28 = x19*(press*x23 + x15*x22 + (Bvec[0]*(gdd[0]*x25 + gdd[1]*x26 + gdd[2]*x27 + x7*(betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2])) + Bvec[1]*(gdd[1]*x25 + gdd[3]*x26 + gdd[4]*x27 + x7*(betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4])) + Bvec[2]*(gdd[2]*x25 + gdd[4]*x26 + gdd[5]*x27 + x7*(betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5]))*Bvec[0]*(gdd[0]*x25 + gdd[1]*x26 + gdd[2]*x27 + x7*(betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2])) + Bvec[1]*(gdd[1]*x25 + gdd[3]*x26 + gdd[4]*x27 + x7*(betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4])) + Bvec[2]*(gdd[2]*x25 + gdd[4]*x26 + gdd[5]*x27 + x7*(betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5]))));
    double x29 = Bvec[0]*x24 - x28*(guu[0]*zvec[0] + guu[1]*zvec[1] + guu[2]*zvec[2]);
    double x30 = Bvec[1]*x24 - x28*(guu[1]*zvec[0] + guu[3]*zvec[1] + guu[4]*zvec[2]);
    double x31 = Bvec[2]*x24 - x28*(guu[2]*zvec[0] + guu[4]*zvec[1] + guu[5]*zvec[2]);
    double x32 = (1.0/2.0)/(alp*(x6*x6));
    out[0] = x11;
    out[1] = -alp*(rho*zvec[2] - x19*x8) + (1.0/2.0)*betau[2]*(2*x10 - (x15*(2*x1 + 2*x3 + 2*x5 + 1) - (x16*x16) + x20*(-press + x17*x6))/x6);
    out[2] = -x21*(x0*x15 + x0*x18 - x12*x16) - x32*(gdd[0]*x29 + gdd[1]*x30 + gdd[2]*x31);
    out[3] = -x21*(-x13*x16 + x15*x2 + x18*x2) - x32*(gdd[1]*x29 + gdd[3]*x30 + gdd[4]*x31);
    out[4] = -x21*(-x14*x16 + x15*x4 + x18*x4) - x32*(gdd[2]*x29 + gdd[4]*x30 + gdd[5]*x31);
    out[5] = s*x11;
}

static void KOKKOS_ALWAYS_INLINE 
get_smallb(
    double const alp, double const sqrtg, double const * __restrict__ gdd, double const * __restrict__ guu, double const * __restrict__ betau,
    double const * __restrict__ zvec, double const * __restrict__ Bvec, double *out, double *b2
)
{
    double x0 = zvec[0]*(gdd[0]*zvec[0] + gdd[1]*zvec[1] + gdd[2]*zvec[2]) + zvec[1]*(gdd[1]*zvec[0] + gdd[3]*zvec[1] + gdd[4]*zvec[2]) + zvec[2]*(gdd[2]*zvec[0] + gdd[4]*zvec[1] + gdd[5]*zvec[2]) + 1;
    double x1 = sqrt(x0);
    double x2 = 1/((alp*alp));
    double x3 = 1/(x1);
    double x4 = alp*x3;
    double x5 = -betau[0] + x4*zvec[0];
    double x6 = -betau[1] + x4*zvec[1];
    double x7 = -betau[2] + x4*zvec[2];
    double x8 = betau[0]*gdd[0] + betau[1]*gdd[1] + betau[2]*gdd[2] + gdd[0]*x5 + gdd[1]*x6 + gdd[2]*x7;
    double x9 = betau[0]*gdd[1] + betau[1]*gdd[3] + betau[2]*gdd[4] + gdd[1]*x5 + gdd[3]*x6 + gdd[4]*x7;
    double x10 = betau[0]*gdd[2] + betau[1]*gdd[4] + betau[2]*gdd[5] + gdd[2]*x5 + gdd[4]*x6 + gdd[5]*x7;
    double x11 = Bvec[0]*x8 + Bvec[1]*x9 + Bvec[2]*x10;
    double x12 = x11*x2;
    double x13 = x0*x12;
    out[0] = x1*x12;
    out[1] = x3*(Bvec[0] + x13*x8);
    out[2] = x3*(Bvec[1] + x13*x9);
    out[3] = x3*(Bvec[2] + x10*x13);
    *b2 = (Bvec[0]*(Bvec[0]*gdd[0] + Bvec[1]*gdd[1] + Bvec[2]*gdd[2]) + Bvec[1]*(Bvec[0]*gdd[1] + Bvec[1]*gdd[3] + Bvec[2]*gdd[4]) + Bvec[2]*(Bvec[0]*gdd[2] + Bvec[1]*gdd[4] + Bvec[2]*gdd[5]) + x0*(x11*x11)*x2)/x0;
}
}


#endif 