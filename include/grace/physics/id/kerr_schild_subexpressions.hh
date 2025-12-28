
/****************************************************************************/
/*                  Kerr-Schild helpers, SymPy generated                    */
/****************************************************************************/
#ifndef GRACE_KERRSCHILD_ID_SUBEXPR_HH
#define GRACE_KERRSCHILD_ID_SUBEXPR_HH

#include <Kokkos_Core.hpp>

static void KOKKOS_INLINE_FUNCTION
kerr_schild_to_boyer_lindquist(
	const double xyz[3],
	double a,
	double reps,
	double * __restrict__ r,
	double * __restrict__ theta,
	double * __restrict__ phi
)
{
	double x0 = ((a)*(a));
	double x1 = ((xyz[2])*(xyz[2]));
	double x2 = -x0 + x1 + ((xyz[0])*(xyz[0])) + ((xyz[1])*(xyz[1]));
	double x3 = fmax(reps, (1.0/2.0)*M_SQRT2*sqrt(x2 + sqrt(4*x0*x1 + ((x2)*(x2)))));
	*r = x3;
	*theta = acos(xyz[2]/x3);
	*phi = -a*x3/(x0 + ((x3)*(x3)) - 2*x3) + atan2(-a*xyz[0] + x3*xyz[1], a*xyz[1] + x3*xyz[0]);
}

static void KOKKOS_INLINE_FUNCTION
kerr_schild_adm_metric(
	const double xyz[3],
	double a,
	double reps,
	double * __restrict__ gxx,
	double * __restrict__ gxy,
	double * __restrict__ gxz,
	double * __restrict__ gyy,
	double * __restrict__ gyz,
	double * __restrict__ gzz,
	double * __restrict__ gXX,
	double * __restrict__ gXY,
	double * __restrict__ gXZ,
	double * __restrict__ gYY,
	double * __restrict__ gYZ,
	double * __restrict__ gZZ,
	double * __restrict__ alp,
	double * __restrict__ betaX,
	double * __restrict__ betaY,
	double * __restrict__ betaZ,
	double * __restrict__ Kxx,
	double * __restrict__ Kxy,
	double * __restrict__ Kxz,
	double * __restrict__ Kyy,
	double * __restrict__ Kyz,
	double * __restrict__ Kzz
)
{
	double x0 = ((a)*(a));
	double x1 = M_SQRT2;
	double x2 = ((xyz[2])*(xyz[2]));
	double x3 = x0*x2;
	double x4 = ((xyz[0])*(xyz[0]));
	double x5 = ((xyz[1])*(xyz[1]));
	double x6 = x2 + x4 + x5;
	double x7 = -x0 + x6;
	double x8 = sqrt(4*x3 + ((x7)*(x7)));
	double x9 = sqrt(x7 + x8);
	double x10 = fmax(reps, (1.0/2.0)*x1*x9);
	double x11 = ((x10)*(x10));
	double x12 = x0 + x11;
	double x13 = 1/(((x12)*(x12)));
	double x14 = x10*xyz[0];
	double x15 = a*xyz[1] + x14;
	double x16 = ((x15)*(x15));
	double x17 = x13*x16;
	double x18 = ((x10)*(x10)*(x10));
	double x19 = ((x10)*(x10)*(x10)*(x10));
	double x20 = 1/(x19 + x3);
	double x21 = 2*x20;
	double x22 = x18*x21;
	double x23 = x17*x22;
	double x24 = x10*xyz[1];
	double x25 = a*xyz[0] - x24;
	double x26 = x13*x25;
	double x27 = x15*x22;
	double x28 = x26*x27;
	double x29 = 1/(x12);
	double x30 = x15*x29;
	double x31 = x11*x21;
	double x32 = x31*xyz[2];
	double x33 = x30*x32;
	double x34 = x13*((x25)*(x25));
	double x35 = x22*x34;
	double x36 = x25*x29;
	double x37 = x32*x36;
	double x38 = 2*x10;
	double x39 = x2*x20;
	double x40 = x38*x39;
	double x41 = x17 + x2/x11;
	double x42 = 1/(x22*(x34 + x41) + 1);
	double x43 = -x23*x42 + 1;
	double x44 = x28*x42;
	double x45 = -x35*x42 + 1;
	double x46 = x22 + 1;
	double x47 = sqrt(x46);
	double x48 = 1/(x46);
	double x49 = x22*x48;
	double x50 = 1/(x8);
	double x51 = x50*x7 + 1;
	double x52 = 1/(x9);
	double x53 = (((reps - 1.0/2.0*x1*x9 > 0) ? (
   0
)
: ((reps - 1.0/2.0*x1*x9 == 0) ? (
   1.0/2.0
)
: (
   1
))));
	double x54 = x1*x52*x53;
	double x55 = x51*x54;
	double x56 = x10*x29;
	double x57 = x55*xyz[0];
	double x58 = 3*x30;
	double x59 = -x25;
	double x60 = ((x59)*(x59));
	double x61 = x13*x60;
	double x62 = 1/(x22*(x41 + x61) + 1);
	double x63 = x40*x62 - 1;
	double x64 = -x63;
	double x65 = x19*x20;
	double x66 = 4*x65;
	double x67 = 1 - x66;
	double x68 = x64*x67;
	double x69 = x50*(x0 + x6) + 1;
	double x70 = x54*x69;
	double x71 = 2*x70;
	double x72 = 3*x70;
	double x73 = x15*x72;
	double x74 = 4*x11;
	double x75 = x30*x74;
	double x76 = x18*x70;
	double x77 = x0 + x76;
	double x78 = 4*x10;
	double x79 = x77*x78;
	double x80 = x15*x20;
	double x81 = x14*x71 - x70*x75 + x73 - x79*x80;
	double x82 = x21*x62;
	double x83 = x19*x82/((x12)*(x12)*(x12));
	double x84 = x16*x81*x83;
	double x85 = x22*x62;
	double x86 = 2*a;
	double x87 = -x1*x51*x52*x53*xyz[0]*xyz[1] + x86;
	double x88 = -x10*x87;
	double x89 = x10*(x57*xyz[1] + x86);
	double x90 = x55*xyz[1];
	double x91 = 2*x90;
	double x92 = x15*x91;
	double x93 = 2*x57;
	double x94 = x59*x93;
	double x95 = x57*x66;
	double x96 = x59*x95;
	double x97 = x11*x29;
	double x98 = x94*x97;
	double x99 = x11*x30;
	double x100 = x91*x99;
	double x101 = x66*x90;
	double x102 = x101*x15;
	double x103 = x59*x77;
	double x104 = x11*x20;
	double x105 = x104*x30;
	double x106 = x70*x97;
	double x107 = x106*xyz[0];
	double x108 = x13*x59;
	double x109 = 4*x15*x76;
	double x110 = x56*x73;
	double x111 = -4*x103*x105 + x107*x59 - x108*x109 + x110*x59 + x70*x99*xyz[1];
	double x112 = x100 + x102 + x111 + x88 - x89 - x92 + x94 - x96 - x98;
	double x113 = x108*x112;
	double x114 = x20*((xyz[2])*(xyz[2])*(xyz[2]));
	double x115 = x114*(x113*x85 - x57*x68 + x84);
	double x116 = x18*xyz[2];
	double x117 = x2*x57;
	double x118 = x67*x82;
	double x119 = -x23*x62 + 1;
	double x120 = x31*x62;
	double x121 = x17*(x113*x120 + x117*x118 - x119*x29*x81);
	double x122 = x118*x2*x55;
	double x123 = -x61*x85 + 1;
	double x124 = -x112*x123 + x122*x14*x59 + x59*x84;
	double x125 = -x11*x124*x13*x20*x25*xyz[2] + x115 + x116*x121*x20;
	double x126 = x104*x47;
	double x127 = x29*x74;
	double x128 = x127*x59;
	double x129 = -x128*x70 - x20*x59*x79 + x24*x71 + x59*x72;
	double x130 = x129*x60*x83;
	double x131 = -x100 - x102 + x111 - x88 + x89 + x92 - x94 + x96 + x98;
	double x132 = x13*x131;
	double x133 = x27*x62;
	double x134 = x114*(x130 + x132*x133 - x68*x90);
	double x135 = x13*(-x119*x131 + x122*x15*x24 + x130*x15);
	double x136 = x2*x90;
	double x137 = x118*x136 + x120*x132*x15 - x123*x129*x29;
	double x138 = x104*x135*x15*xyz[2] - x13*x137*x18*x20*x25*x59*xyz[2] + x134;
	double x139 = x2*x70;
	double x140 = x139 - x39*x79 + x78;
	double x141 = 2*x139*x97;
	double x142 = x139*x78;
	double x143 = 8*x39;
	double x144 = x143*x77;
	double x145 = -x109*x13*x2 + x117*x66 - x117 + x141*xyz[0] + x142*x30 - x144*x99 + x75;
	double x146 = x145*x30;
	double x147 = 4*x2*x76;
	double x148 = x29*x59;
	double x149 = x136*x66 - x136 + x141*xyz[1];
	double x150 = -x103*x143*x97 - x108*x147 + x128 + x142*x148 + x149;
	double x151 = x120*x148;
	double x152 = x10*x39;
	double x153 = x140*x2;
	double x154 = x108*x133;
	double x155 = x105*(-x119*x145 + x120*x153*x30 + x150*x154) - x11*x20*x25*x29*(-x123*x150 + x145*x154 + x151*x153) + x152*(x120*x146 - x140*x64 + x150*x151);
	double x156 = x20*x47;
	double x157 = (1.0/2.0)*x10*x156;
	double x158 = x104*x36;
	double x159 = x31*x42;
	double x160 = x11*x144*x36 - x127*x25 - x142*x36 + x147*x26 + x149;
	double x161 = x159*x36;
	*gxx = x23 + 1;
	*gxy = -x28;
	*gxz = x33;
	*gyy = x35 + 1;
	*gyz = -x37;
	*gzz = x40 + 1;
	*gXX = x43;
	*gXY = x44;
	*gXZ = -x33*x42;
	*gYY = x45;
	*gYZ = x37*x42;
	*gZZ = -x40*x42 + 1;
	*alp = 1/(x47);
	*betaX = x30*x49;
	*betaY = -x36*x49;
	*betaZ = x32*x48;
	*Kxx = -x126*(2*x1*x11*x13*x15*x51*x52*x53*xyz[0] + 4*x1*x15*x19*x20*x29*x51*x52*x53*xyz[0] - x125 - x56*(x38 + x4*x55) - x57*x58);
	*Kxy = -1.0/2.0*x126*(2*x1*x11*x13*x15*x51*x52*x53*xyz[1] + 4*x1*x15*x19*x20*x29*x51*x52*x53*xyz[1] + 3*x1*x25*x29*x51*x52*x53*xyz[0] + x10*x29*x87 - x11*x26*x93 - x125 - x138 - x29*x89 - x36*x95 - x58*x90);
	*Kxz = -x157*(2*x1*x13*x15*x18*x52*x53*x69*xyz[2] + 4*x1*x19*x20*x51*x52*x53*xyz[0]*xyz[2] - x10*x115 - x107*xyz[2] + 4*x11*x15*x20*x29*x77*xyz[2] - x110*xyz[2] - x121*x65*xyz[2] + x124*x13*x18*x20*x25*xyz[2] - x155 - x93*xyz[2]);
	*Kyy = -x126*(3*x1*x25*x29*x51*x52*x53*xyz[1] - x101*x36 - x11*x26*x91 - x138 - x56*(x38 + x5*x55));
	*Kyz = -x157*(3*x1*x10*x25*x29*x52*x53*x69*xyz[2] + 4*x1*x19*x20*x51*x52*x53*xyz[1]*xyz[2] - x10*x134 - x106*xyz[1]*xyz[2] - x116*x135*x80 + x13*x137*x19*x20*x25*x59*xyz[2] - x155 - 4*x158*x77*xyz[2] - 2*x26*x76*xyz[2] - x91*xyz[2]);
	*Kzz = x156*x38*(x10 + (1.0/2.0)*x105*(2*x11*x140*x15*x2*x20*x29*x42 - x145*x43 - x160*x44) + x139 + (1.0/2.0)*x152*(x140*x63 + x146*x159 - x160*x161) + (1.0/2.0)*x158*(x145*x44 + x153*x161 + x160*x45) - x40*x77);
}

static void KOKKOS_INLINE_FUNCTION
kerr_schild_four_metric(
	const double xyz[3],
	double a,
	double reps,
	double (*g4dd)[4][4],
	double (*g4uu)[4][4]
)
{
	double x0 = ((a)*(a));
	double x1 = ((xyz[2])*(xyz[2]));
	double x2 = x0*x1;
	double x3 = -x0 + x1 + ((xyz[0])*(xyz[0])) + ((xyz[1])*(xyz[1]));
	double x4 = fmax(reps, (1.0/2.0)*M_SQRT2*sqrt(x3 + sqrt(4*x2 + ((x3)*(x3)))));
	double x5 = ((x4)*(x4)*(x4));
	double x6 = 1/(x2 + ((x4)*(x4)*(x4)*(x4)));
	double x7 = a*xyz[1] + x4*xyz[0];
	double x8 = ((x4)*(x4));
	double x9 = x0 + x8;
	double x10 = 1/(x9);
	double x11 = 2.0*x6;
	double x12 = x11*x5;
	double x13 = x10*x12;
	double x14 = x13*x7;
	double x15 = a*xyz[0] - x4*xyz[1];
	double x16 = -x13*x15;
	double x17 = x8*xyz[2];
	double x18 = x11*x17;
	double x19 = 2*x6;
	double x20 = x19*x5/((x9)*(x9));
	double x21 = x20*((x7)*(x7));
	double x22 = x15*x20*x7;
	double x23 = -x22;
	double x24 = x10*x17*x19;
	double x25 = x24*x7;
	double x26 = ((x15)*(x15))*x20;
	double x27 = x15*x24;
	double x28 = -x27;
	double x29 = x1*x19*x4;
	double x30 = -x25;
	(*g4dd)[0][0] = 2.0*x5*x6 - 1;
	(*g4dd)[0][1] = (*g4dd)[1][0] = x14;
	(*g4dd)[0][2] = (*g4dd)[2][0] = x16;
	(*g4dd)[0][3] = (*g4dd)[3][0] = x18;
	(*g4dd)[1][1] = x21 + 1;
	(*g4dd)[1][2] = (*g4dd)[2][1] = x23;
	(*g4dd)[1][3] = (*g4dd)[3][1] = x25;
	(*g4dd)[2][2] = x26 + 1;
	(*g4dd)[2][3] = (*g4dd)[3][2] = x28;
	(*g4dd)[3][3] = x29 + 1;
	(*g4uu)[0][0] = -x12 - 1;
	(*g4uu)[0][1] = (*g4uu)[1][0] = x14;
	(*g4uu)[0][2] = (*g4uu)[2][0] = x16;
	(*g4uu)[0][3] = (*g4uu)[3][0] = x18;
	(*g4uu)[1][1] = 1 - x21;
	(*g4uu)[1][2] = (*g4uu)[2][1] = x22;
	(*g4uu)[1][3] = (*g4uu)[3][1] = x30;
	(*g4uu)[2][2] = 1 - x26;
	(*g4uu)[2][3] = (*g4uu)[3][2] = x27;
	(*g4uu)[3][3] = 1 - x29;
}

static void KOKKOS_INLINE_FUNCTION
kerr_schild_to_bl_jac(
	const double xyz[3],
	double a,
	double reps,
	double (*j)[3][3]
)
{
	double x0 = ((a)*(a));
	double x1 = ((xyz[2])*(xyz[2]));
	double x2 = -x0 + x1 + ((xyz[0])*(xyz[0])) + ((xyz[1])*(xyz[1]));
	double x3 = fmax(reps, (1.0/2.0)*M_SQRT2*sqrt(x2 + sqrt(4*x0*x1 + ((x2)*(x2)))));
	double x4 = a*xyz[1] + x3*xyz[0];
	double x5 = 2*x3;
	double x6 = ((x3)*(x3));
	double x7 = 1/(x0 - x5 + x6);
	double x8 = x5*x7*(x3 - 1) - 1;
	double x9 = -x8;
	double x10 = a*xyz[0] - x3*xyz[1];
	double x11 = a*x10;
	double x12 = sqrt(-x1/x6 + 1);
	double x13 = (1.0/sqrt(((x10)*(x10)) + ((x4)*(x4))));
	double x14 = x12*x13;
	double x15 = x3*x4;
	double x16 = x11 + x15;
	double x17 = xyz[2]/x3;
	double x18 = x13*x17;
	double x19 = a*x4 - x10*x3;
	(*j)[0][0] = -x14*(x0*x4*x7*x9 - x11*x3*x7*x9 - x4);
	(*j)[0][1] = x16*x18;
	(*j)[0][2] = -x14*x19;
	(*j)[1][0] = -x14*(a*x15*x7*x8 + x0*x10*x7*x8 + x10);
	(*j)[1][1] = x18*x19;
	(*j)[1][2] = x14*x16;
	(*j)[2][0] = x17;
	(*j)[2][1] = -x12*x3;
	(*j)[2][2] = 0;
}

static void KOKKOS_INLINE_FUNCTION
transform_vector_bl2ks(
	const double xyz[3],
	double a,
	double reps,
	const double vBL[4],
	double (*vKS)[4]
)
{
	double x0 = ((a)*(a));
	double x1 = ((xyz[2])*(xyz[2]));
	double x2 = -x0 + x1 + ((xyz[0])*(xyz[0])) + ((xyz[1])*(xyz[1]));
	double x3 = fmax(reps, (1.0/2.0)*M_SQRT2*sqrt(x2 + sqrt(4*x0*x1 + ((x2)*(x2)))));
	double x4 = 2*x3;
	double x5 = ((x3)*(x3));
	double x6 = 1/(x0 - x4 + x5);
	double x7 = a*xyz[1];
	double x8 = x3*xyz[0];
	double x9 = x7 + x8;
	double x10 = ((x9)*(x9));
	double x11 = a*xyz[0] - x3*xyz[1];
	double x12 = -x11;
	double x13 = x3*x9;
	double x14 = xyz[2]/x3;
	double x15 = vBL[2]*x14;
	double x16 = a*x9;
	double x17 = x12*x3;
	double x18 = sqrt(-x1/x5 + 1);
	double x19 = vBL[3]*x18;
	double x20 = x3 - 1;
	double x21 = x4*x6;
	double x22 = -x20*x21 + 1;
	double x23 = x0*x6;
	double x24 = vBL[1]*x18;
	double x25 = x20*x21 - 1;
	(*vKS)[0] = vBL[0] + 2*vBL[1]*x6;
	(*vKS)[1] = -(x15*(a*x12 - x13) + x19*(x16 + x17) + x24*(a*x17*x22*x6 + x22*x23*x9 - x7 - x8))/sqrt(x10 + ((x12)*(x12)));
	(*vKS)[2] = (x15*(-x11*x3 + x16) + x19*(a*x11 + x13) - x24*(x11*x23*x25 + x11 + x16*x25*x3*x6))/sqrt(x10 + ((x11)*(x11)));
	(*vKS)[3] = vBL[1]*x14 - vBL[2]*x18*x3;
}

#endif 
