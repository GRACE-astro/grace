
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
	double x30 = x11*x29;
	double x31 = x15*x30;
	double x32 = x21*xyz[2];
	double x33 = x31*x32;
	double x34 = x13*((x25)*(x25));
	double x35 = x22*x34;
	double x36 = x25*x30;
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
	double x49 = x29*x48;
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
	double x56 = x38 + x4*x55;
	double x57 = x55*xyz[0];
	double x58 = x15*x57;
	double x59 = 4*x20;
	double x60 = x19*x59;
	double x61 = -x58*x60 + 3*x58;
	double x62 = x38*x56;
	double x63 = 4*x31;
	double x64 = -x57*x63 + x62;
	double x65 = x61 + x64;
	double x66 = -x25;
	double x67 = ((x66)*(x66));
	double x68 = x13*x67;
	double x69 = 1/(x22*(x41 + x68) + 1);
	double x70 = x23*x69;
	double x71 = x65*x70;
	double x72 = x40*x69;
	double x73 = x72 - 1;
	double x74 = -x73;
	double x75 = x50*(x0 + x6) + 1;
	double x76 = x54*x75;
	double x77 = x18*x76;
	double x78 = x0 + x77;
	double x79 = x59*x78;
	double x80 = x16*x30;
	double x81 = 3*x76;
	double x82 = x10*x29;
	double x83 = x81*x82;
	double x84 = 4*x77;
	double x85 = x15*x20;
	double x86 = 8*x19;
	double x87 = x85*x86;
	double x88 = x57*x87;
	double x89 = 2*x76;
	double x90 = -x16*x83 + x17*x84 - x31*x89*xyz[0] + 4*x58 + x64 + x79*x80 - x88;
	double x91 = x55*xyz[1];
	double x92 = 3*x91;
	double x93 = 2*a;
	double x94 = -x1*x51*x52*x53*xyz[0]*xyz[1] + x93;
	double x95 = -x94;
	double x96 = x38*x95;
	double x97 = x57*xyz[1] + x93;
	double x98 = x38*x97;
	double x99 = x60*x91;
	double x100 = 8*x31;
	double x101 = -x100*x57*x66 + x15*x96 - x15*x98 - x16*x92 + x16*x99 + 6*x58*x66 + x62*x66 - x66*x88 + 4*x80*x91;
	double x102 = x22*x69;
	double x103 = x13*x66;
	double x104 = x102*x103;
	double x105 = x101*x104;
	double x106 = x10*x39;
	double x107 = 1 - x70;
	double x108 = x72*x90;
	double x109 = x18*x20;
	double x110 = x102*x68;
	double x111 = 1 - x110;
	double x112 = x20*x47;
	double x113 = x112*x30;
	double x114 = x15*x99;
	double x115 = -x114;
	double x116 = x115 + x15*x92;
	double x117 = x116 - x63*x91 + x98;
	double x118 = x117*x70;
	double x119 = 3*x57;
	double x120 = 4*x66;
	double x121 = x57*x60;
	double x122 = x121*x66;
	double x123 = -x122;
	double x124 = x119*x66 - x120*x30*x57 + x123 + x96;
	double x125 = x110*x124;
	double x126 = x10*x97;
	double x127 = -x31*x66*x79;
	double x128 = 2*x91;
	double x129 = x128*x15;
	double x130 = x30*x76;
	double x131 = x130*xyz[0];
	double x132 = x131*x66;
	double x133 = x31*x76*xyz[1];
	double x134 = x13*x15;
	double x135 = x134*x84;
	double x136 = -x135*x66;
	double x137 = x128*x31;
	double x138 = x15*x81;
	double x139 = x138*x82;
	double x140 = x139*x66;
	double x141 = x114 - x126 + x127 - x129 + x132 + x133 + x136 + x137 + x140;
	double x142 = x10*x95;
	double x143 = 2*x57;
	double x144 = x143*x66;
	double x145 = x144*x30;
	double x146 = x122 - x142 - x144 + x145;
	double x147 = -x141 - x146;
	double x148 = x147*x72;
	double x149 = x126 - x137;
	double x150 = 1 - x60;
	double x151 = x150*x74;
	double x152 = 4*x10;
	double x153 = x152*x78;
	double x154 = x138 + x14*x89 - x153*x85 - x63*x76;
	double x155 = x19*x21;
	double x156 = x155*x69/((x12)*(x12)*(x12));
	double x157 = x154*x156*x16;
	double x158 = x123 + x141 + x142 + x144 - x145;
	double x159 = x2*x57;
	double x160 = x21*x69;
	double x161 = x150*x160;
	double x162 = x11*x160;
	double x163 = x161*x2*x55;
	double x164 = (1.0/2.0)*x10*x112*xyz[2];
	double x165 = x38 + x5*x55;
	double x166 = x66*x91;
	double x167 = x165*x38;
	double x168 = 4*x166;
	double x169 = x167 - x168*x30;
	double x170 = -x166*x60 + x169 + x66*x92;
	double x171 = x30*x67;
	double x172 = x20*x66;
	double x173 = x30*xyz[1];
	double x174 = x168 + x169 + x171*x79 - x172*x86*x91 - x173*x66*x89 - x67*x83 + x68*x84;
	double x175 = -x100*x166 - x119*x67 + x121*x67 + 6*x15*x166 + x15*x167 - x166*x87 + 4*x171*x57 - x66*x96 + x66*x98;
	double x176 = x102*x134;
	double x177 = x175*x176;
	double x178 = x174*x72;
	double x179 = -x120*x130 - x153*x172 + x24*x89 + x66*x81;
	double x180 = x156*x179*x67;
	double x181 = x115 + x127 + x129 + x132 + x133 + x136 + x140 + x146 + x149;
	double x182 = x2*x91;
	double x183 = x2*x76;
	double x184 = x152 - x153*x39 + x183;
	double x185 = 2*x183;
	double x186 = x152*x183*x29;
	double x187 = 8*x39*x78;
	double x188 = -x135*x2 + x15*x186 + x159*x60 - x159 + x185*x30*xyz[0] - x187*x31 + x63;
	double x189 = x21*x42;
	double x190 = 4*x25;
	double x191 = x13*x190*x2*x77 + x173*x185 + x182*x60 - x182 - x186*x25 + x187*x36 - x190*x30;
	double x192 = x189*x36;
	double x193 = (1.0/2.0)*x20;
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
	*betaX = x27*x49;
	*betaY = -x22*x25*x49;
	*betaZ = x11*x32*x48;
	*Kxx = -x113*(2*x1*x11*x15*x29*x51*x52*x53*xyz[0] - x10*x56 - x106*(x105 + x71 - x74*x90) - x109*x17*(x105 - x107*x65 + x108) + x13*x18*x20*x25*(-x101*x111 + x108*x66 + x66*x71) - x61);
	*Kxy = -1.0/2.0*x113*(3*x1*x25*x51*x52*x53*xyz[0] + x10*x94 - x116 - x121*x25 + 2*x13*x18*x20*x25*x66*(-x111*x124 + x118 + x148) - x143*x36 - x149 - x23*(-x107*x117 + x125 + x148) - x40*(x118 + x125 - x147*x74));
	*Kxz = -x164*(2*x1*x13*x15*x18*x52*x53*x75 + 4*x1*x19*x20*x51*x52*x53*xyz[0] + 4*x11*x15*x20*x29*x78 + 2*x13*x18*x20*x25*(-x111*x158 + x14*x163*x66 + x157*x66) - x131 - x139 - x143 - x155*x17*(x103*x158*x162 - x107*x154*x29 + x159*x161) - x40*(x104*x158 - x151*x57 + x157));
	*Kyy = -x113*(3*x1*x25*x51*x52*x53*xyz[1] - x10*x165 - x106*(x110*x170 - x174*x74 + x177) - x109*x134*(-x107*x175 + x15*x178 + x170*x27*x68*x69) - x128*x36 + x13*x18*x20*x25*x66*(-x111*x170 + x177 + x178) - x25*x99);
	*Kyz = -x164*(3*x1*x10*x25*x29*x52*x53*x75 + 4*x1*x19*x20*x51*x52*x53*xyz[1] - x128 + 2*x13*x19*x20*x25*x66*(-x111*x179*x29 + x134*x162*x181 + x161*x182) - x130*xyz[1] - x134*x22*(-x107*x181 + x15*x163*x24 + x15*x180) - 2*x26*x77 - x36*x79 - x40*(-x151*x91 + x176*x181 + x180));
	*Kzz = x112*x38*(x10 + (1.0/2.0)*x106*(x184*x73 + x188*x189*x31 - x191*x192) + x183 + x193*x31*(2*x11*x15*x184*x2*x20*x29*x42 - x188*x43 - x191*x44) + x193*x36*(x184*x192*x2 + x188*x44 + x191*x45) - x40*x78);
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
