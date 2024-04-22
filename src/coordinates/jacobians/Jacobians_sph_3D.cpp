/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                       This file is part of 'Thunder'                       *
 ******************************************************************************/
#include <thunder/coordinates/jacobians/Jacobians_sph_3D.hh> 
#include <thunder/utils/math.hh>
#include <math.h>
namespace thunder { 
double Jac_sph_3D_00(double L, double R, double eta, double si, double so, double xi) {

   double Jac_sph_3D_00_result;
   Jac_sph_3D_00_result = -(L*si - R*so - (L*si - L - R*so + R)*sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3))/sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3);
   return Jac_sph_3D_00_result;

}

double Jac_sph_3D_01(double L, double R, double eta, double si, double so, double xi, double zeta) {

   double Jac_sph_3D_01_result;
   Jac_sph_3D_01_result = -2*(si*(2*L*eta - L) - zeta*(si*(2*L*eta - L) - so*(2*R*eta - R)))/pow(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3, 3.0/2.0);
   return Jac_sph_3D_01_result;

}

double Jac_sph_3D_02(double L, double R, double eta, double si, double so, double xi, double zeta) {

   double Jac_sph_3D_02_result;
   Jac_sph_3D_02_result = -2*(2*L*si*xi - L*si + zeta*(L*si - R*so - 2*xi*(L*si - R*so)))/pow(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3, 3.0/2.0);
   return Jac_sph_3D_02_result;

}

double Jac_sph_3D_10(double L, double R, double eta, double si, double so, double xi) {

   double Jac_sph_3D_10_result;
   Jac_sph_3D_10_result = -(si*(2*L*eta - L) - so*(2*R*eta - R) + (-L + R + 2*eta*(L - R) - si*(2*L*eta - L) + so*(2*R*eta - R))*sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3))/sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3);
   return Jac_sph_3D_10_result;

}

double Jac_sph_3D_11(double L, double R, double eta, double si, double so, double xi, double zeta) {

   double Jac_sph_3D_11_result;
   Jac_sph_3D_11_result = 2*(16*L*math::int_pow<4>(eta) - 32*L*math::int_pow<3>(eta) + 40*L*math::int_pow<2>(eta) - 24*L*eta + 9*L - si*(16*L*math::int_pow<4>(eta) - 32*L*math::int_pow<3>(eta) + 40*L*math::int_pow<2>(eta) - 24*L*eta + 9*L) - 16*math::int_pow<4>(xi)*(L*si - L) + 32*math::int_pow<3>(xi)*(L*si - L) + 8*math::int_pow<2>(xi)*(4*L*math::int_pow<2>(eta) - 4*L*eta + 5*L - si*(4*L*math::int_pow<2>(eta) - 4*L*eta + 5*L)) - 8*xi*(4*L*math::int_pow<2>(eta) - 4*L*eta + 3*L - si*(4*L*math::int_pow<2>(eta) - 4*L*eta + 3*L)) - zeta*(9*L - 9*R + 16*math::int_pow<4>(eta)*(L - R) - 32*math::int_pow<3>(eta)*(L - R) + 40*math::int_pow<2>(eta)*(L - R) - 24*eta*(L - R) - si*(16*L*math::int_pow<4>(eta) - 32*L*math::int_pow<3>(eta) + 40*L*math::int_pow<2>(eta) - 24*L*eta + 9*L) + so*(16*R*math::int_pow<4>(eta) - 32*R*math::int_pow<3>(eta) + 40*R*math::int_pow<2>(eta) - 24*R*eta + 9*R) - 16*math::int_pow<4>(xi)*(L*si - L - R*so + R) + 32*math::int_pow<3>(xi)*(L*si - L - R*so + R) + 8*math::int_pow<2>(xi)*(5*L - 5*R + 4*math::int_pow<2>(eta)*(L - R) - 4*eta*(L - R) - si*(4*L*math::int_pow<2>(eta) - 4*L*eta + 5*L) + so*(4*R*math::int_pow<2>(eta) - 4*R*eta + 5*R)) - 8*xi*(3*L - 3*R + 4*math::int_pow<2>(eta)*(L - R) - 4*eta*(L - R) - si*(4*L*math::int_pow<2>(eta) - 4*L*eta + 3*L) + so*(4*R*math::int_pow<2>(eta) - 4*R*eta + 3*R))) + 2*(2*L*si*math::int_pow<2>(xi) - 2*L*si*xi + L*si - zeta*(L*si - R*so + 2*math::int_pow<2>(xi)*(L*si - R*so) - 2*xi*(L*si - R*so)))*sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3))/(16*math::int_pow<4>(eta) - 32*math::int_pow<3>(eta) + 40*math::int_pow<2>(eta) - 24*eta + 16*math::int_pow<4>(xi) - 32*math::int_pow<3>(xi) + 8*math::int_pow<2>(xi)*(4*math::int_pow<2>(eta) - 4*eta + 5) - 8*xi*(4*math::int_pow<2>(eta) - 4*eta + 3) + 9);
   return Jac_sph_3D_11_result;

}

double Jac_sph_3D_12(double L, double R, double eta, double si, double so, double xi, double zeta) {

   double Jac_sph_3D_12_result;
   Jac_sph_3D_12_result = -2*(2*si*xi*(2*L*eta - L) - si*(2*L*eta - L) + zeta*(si*(2*L*eta - L) - so*(2*R*eta - R) - 2*xi*(si*(2*L*eta - L) - so*(2*R*eta - R))))/pow(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3, 3.0/2.0);
   return Jac_sph_3D_12_result;

}

double Jac_sph_3D_20(double L, double R, double eta, double si, double so, double xi) {

   double Jac_sph_3D_20_result;
   Jac_sph_3D_20_result = (L*si - R*so - 2*xi*(L*si - R*so) - sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3)*(L*si - L - R*so + R - 2*xi*(L*si - L - R*so + R)))/sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3);
   return Jac_sph_3D_20_result;

}

double Jac_sph_3D_21(double L, double R, double eta, double si, double so, double xi, double zeta) {

   double Jac_sph_3D_21_result;
   Jac_sph_3D_21_result = -2*(2*si*xi*(2*L*eta - L) - si*(2*L*eta - L) + zeta*(si*(2*L*eta - L) - so*(2*R*eta - R) - 2*xi*(si*(2*L*eta - L) - so*(2*R*eta - R))))/pow(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3, 3.0/2.0);
   return Jac_sph_3D_21_result;

}

double Jac_sph_3D_22(double L, double R, double eta, double si, double so, double xi, double zeta) {

   double Jac_sph_3D_22_result;
   Jac_sph_3D_22_result = 2*(16*L*math::int_pow<4>(eta) - 32*L*math::int_pow<3>(eta) + 40*L*math::int_pow<2>(eta) - 24*L*eta + 9*L - si*(16*L*math::int_pow<4>(eta) - 32*L*math::int_pow<3>(eta) + 40*L*math::int_pow<2>(eta) - 24*L*eta + 9*L) - 16*math::int_pow<4>(xi)*(L*si - L) + 32*math::int_pow<3>(xi)*(L*si - L) + 8*math::int_pow<2>(xi)*(4*L*math::int_pow<2>(eta) - 4*L*eta + 5*L - si*(4*L*math::int_pow<2>(eta) - 4*L*eta + 5*L)) - 8*xi*(4*L*math::int_pow<2>(eta) - 4*L*eta + 3*L - si*(4*L*math::int_pow<2>(eta) - 4*L*eta + 3*L)) - zeta*(9*L - 9*R + 16*math::int_pow<4>(eta)*(L - R) - 32*math::int_pow<3>(eta)*(L - R) + 40*math::int_pow<2>(eta)*(L - R) - 24*eta*(L - R) - si*(16*L*math::int_pow<4>(eta) - 32*L*math::int_pow<3>(eta) + 40*L*math::int_pow<2>(eta) - 24*L*eta + 9*L) + so*(16*R*math::int_pow<4>(eta) - 32*R*math::int_pow<3>(eta) + 40*R*math::int_pow<2>(eta) - 24*R*eta + 9*R) - 16*math::int_pow<4>(xi)*(L*si - L - R*so + R) + 32*math::int_pow<3>(xi)*(L*si - L - R*so + R) + 8*math::int_pow<2>(xi)*(5*L - 5*R + 4*math::int_pow<2>(eta)*(L - R) - 4*eta*(L - R) - si*(4*L*math::int_pow<2>(eta) - 4*L*eta + 5*L) + so*(4*R*math::int_pow<2>(eta) - 4*R*eta + 5*R)) - 8*xi*(3*L - 3*R + 4*math::int_pow<2>(eta)*(L - R) - 4*eta*(L - R) - si*(4*L*math::int_pow<2>(eta) - 4*L*eta + 3*L) + so*(4*R*math::int_pow<2>(eta) - 4*R*eta + 3*R))) + 2*(si*(2*L*math::int_pow<2>(eta) - 2*L*eta + L) - zeta*(si*(2*L*math::int_pow<2>(eta) - 2*L*eta + L) - so*(2*R*math::int_pow<2>(eta) - 2*R*eta + R)))*sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3))/(16*math::int_pow<4>(eta) - 32*math::int_pow<3>(eta) + 40*math::int_pow<2>(eta) - 24*eta + 16*math::int_pow<4>(xi) - 32*math::int_pow<3>(xi) + 8*math::int_pow<2>(xi)*(4*math::int_pow<2>(eta) - 4*eta + 5) - 8*xi*(4*math::int_pow<2>(eta) - 4*eta + 3) + 9);
   return Jac_sph_3D_22_result;

}

double J_sph_3D(double L, double R, double eta, double si, double so, double xi, double zeta) {

   double J_sph_3D_result;
   J_sph_3D_result = -4*(2*math::int_pow<3>(si)*(6*math::int_pow<3>(L)*math::int_pow<2>(eta) 
   - 6*math::int_pow<3>(L)*eta + 5*math::int_pow<3>(L)) - 2*math::int_pow<2>(si)*(9*math::int_pow<3>(L) 
   - 3*math::int_pow<2>(L)*R + 4*math::int_pow<2>(eta)*(3*math::int_pow<3>(L) - math::int_pow<2>(L)*R) 
   - 4*eta*(3*math::int_pow<3>(L) - math::int_pow<2>(L)*R)) + si*(9*math::int_pow<3>(L) 
   - 6*math::int_pow<2>(L)*R + 4*math::int_pow<2>(eta)*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) 
   - 4*eta*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R)) - so*(4*math::int_pow<2>(L)*R*math::int_pow<2>(eta) 
   - 4*math::int_pow<2>(L)*R*eta + 3*math::int_pow<2>(L)*R + 2*math::int_pow<2>(si)*(6*math::int_pow<2>(L)*R*math::int_pow<2>(eta) 
   - 6*math::int_pow<2>(L)*R*eta + 5*math::int_pow<2>(L)*R) - 4*si*(4*math::int_pow<2>(L)*R*math::int_pow<2>(eta) - 4*math::int_pow<2>(L)*R*eta 
   + 3*math::int_pow<2>(L)*R)) + 4*math::int_pow<2>(xi)*(3*math::int_pow<3>(L)*math::int_pow<3>(si) - 2*math::int_pow<2>(si)*(3*math::int_pow<3>(L) 
   - math::int_pow<2>(L)*R) + si*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) - so*(3*math::int_pow<2>(L)*R*math::int_pow<2>(si) - 4*math::int_pow<2>(L)*R*si 
   + math::int_pow<2>(L)*R)) - 4*xi*(3*math::int_pow<3>(L)*math::int_pow<3>(si) - 2*math::int_pow<2>(si)*(3*math::int_pow<3>(L) - math::int_pow<2>(L)*R) + si*(3*math::int_pow<3>(L) 
   - 2*math::int_pow<2>(L)*R) - so*(3*math::int_pow<2>(L)*R*math::int_pow<2>(si) - 4*math::int_pow<2>(L)*R*si + math::int_pow<2>(L)*R)) + math::int_pow<2>(zeta)*(2*math::int_pow<3>(si)*(6*math::int_pow<3>(L)*math::int_pow<2>(eta) 
   - 6*math::int_pow<3>(L)*eta + 5*math::int_pow<3>(L)) - 6*math::int_pow<2>(si)*(3*math::int_pow<3>(L) - 3*math::int_pow<2>(L)*R + 4*math::int_pow<2>(eta)*(math::int_pow<3>(L) 
   - math::int_pow<2>(L)*R) - 4*eta*(math::int_pow<3>(L) - math::int_pow<2>(L)*R)) + 3*si*(3*math::int_pow<3>(L) - 6*math::int_pow<2>(L)*R + 3*L*math::int_pow<2>(R) 
   + 4*math::int_pow<2>(eta)*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) - 4*eta*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R))) 
   - 2*math::int_pow<3>(so)*(6*math::int_pow<3>(R)*math::int_pow<2>(eta) - 6*math::int_pow<3>(R)*eta + 5*math::int_pow<3>(R)) - 6*math::int_pow<2>(so)*(3*L*math::int_pow<2>(R) 
   - 3*math::int_pow<3>(R) + 4*math::int_pow<2>(eta)*(L*math::int_pow<2>(R) - math::int_pow<3>(R)) - 4*eta*(L*math::int_pow<2>(R) - math::int_pow<3>(R)) 
   - si*(6*L*math::int_pow<2>(R)*math::int_pow<2>(eta) - 6*L*math::int_pow<2>(R)*eta + 5*L*math::int_pow<2>(R))) - 3*so*(3*math::int_pow<2>(L)*R - 6*L*math::int_pow<2>(R) + 3*math::int_pow<3>(R) 
   + 4*math::int_pow<2>(eta)*(math::int_pow<2>(L)*R - 2*L*math::int_pow<2>(R) + math::int_pow<3>(R)) - 4*eta*(math::int_pow<2>(L)*R - 2*L*math::int_pow<2>(R) + math::int_pow<3>(R)) 
   + 2*math::int_pow<2>(si)*(6*math::int_pow<2>(L)*R*math::int_pow<2>(eta) - 6*math::int_pow<2>(L)*R*eta + 5*math::int_pow<2>(L)*R) - 4*si*(3*math::int_pow<2>(L)*R - 3*L*math::int_pow<2>(R) 
   + 4*math::int_pow<2>(eta)*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)) - 4*eta*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)))) + 12*math::int_pow<2>(xi)*(math::int_pow<3>(L)*math::int_pow<3>(si) 
   - math::int_pow<3>(R)*math::int_pow<3>(so) - 2*math::int_pow<2>(si)*(math::int_pow<3>(L) - math::int_pow<2>(L)*R) + si*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) 
   + math::int_pow<2>(so)*(3*L*math::int_pow<2>(R)*si - 2*L*math::int_pow<2>(R) + 2*math::int_pow<3>(R)) - so*(3*math::int_pow<2>(L)*R*math::int_pow<2>(si) + math::int_pow<2>(L)*R - 2*L*math::int_pow<2>(R) 
   + math::int_pow<3>(R) - 4*si*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)))) - 12*xi*(math::int_pow<3>(L)*math::int_pow<3>(si) - math::int_pow<3>(R)*math::int_pow<3>(so) - 2*math::int_pow<2>(si)*(math::int_pow<3>(L) 
   - math::int_pow<2>(L)*R) + si*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) + math::int_pow<2>(so)*(3*L*math::int_pow<2>(R)*si - 2*L*math::int_pow<2>(R) + 2*math::int_pow<3>(R)) 
   - so*(3*math::int_pow<2>(L)*R*math::int_pow<2>(si) + math::int_pow<2>(L)*R - 2*L*math::int_pow<2>(R) + math::int_pow<3>(R) - 4*si*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R))))) 
   - 2*zeta*(2*math::int_pow<3>(si)*(6*math::int_pow<3>(L)*math::int_pow<2>(eta) - 6*math::int_pow<3>(L)*eta + 5*math::int_pow<3>(L)) - 2*math::int_pow<2>(si)*(9*math::int_pow<3>(L) 
   - 6*math::int_pow<2>(L)*R + 4*math::int_pow<2>(eta)*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) - 4*eta*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R)) + si*(9*math::int_pow<3>(L) 
   - 12*math::int_pow<2>(L)*R + 3*L*math::int_pow<2>(R) + 4*math::int_pow<2>(eta)*(3*math::int_pow<3>(L) - 4*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) - 4*eta*(3*math::int_pow<3>(L) 
   - 4*math::int_pow<2>(L)*R + L*math::int_pow<2>(R))) - 2*math::int_pow<2>(so)*(4*L*math::int_pow<2>(R)*math::int_pow<2>(eta) - 4*L*math::int_pow<2>(R)*eta + 3*L*math::int_pow<2>(R) 
   - si*(6*L*math::int_pow<2>(R)*math::int_pow<2>(eta) - 6*L*math::int_pow<2>(R)*eta + 5*L*math::int_pow<2>(R))) - 2*so*(3*math::int_pow<2>(L)*R - 3*L*math::int_pow<2>(R) 
   + 4*math::int_pow<2>(eta)*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)) - 4*eta*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)) + 2*math::int_pow<2>(si)*(6*math::int_pow<2>(L)*R*math::int_pow<2>(eta) 
   - 6*math::int_pow<2>(L)*R*eta + 5*math::int_pow<2>(L)*R) - 2*si*(6*math::int_pow<2>(L)*R - 3*L*math::int_pow<2>(R) + 4*math::int_pow<2>(eta)*(2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R)) 
   - 4*eta*(2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R)))) + 4*math::int_pow<2>(xi)*(3*math::int_pow<3>(L)*math::int_pow<3>(si) - 2*math::int_pow<2>(si)*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) 
   + si*(3*math::int_pow<3>(L) - 4*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) + math::int_pow<2>(so)*(3*L*math::int_pow<2>(R)*si - 2*L*math::int_pow<2>(R)) 
   - 2*so*(3*math::int_pow<2>(L)*R*math::int_pow<2>(si) + math::int_pow<2>(L)*R - L*math::int_pow<2>(R) - 2*si*(2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R)))) 
   - 4*xi*(3*math::int_pow<3>(L)*math::int_pow<3>(si) - 2*math::int_pow<2>(si)*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) + si*(3*math::int_pow<3>(L) - 4*math::int_pow<2>(L)*R 
   + L*math::int_pow<2>(R)) + math::int_pow<2>(so)*(3*L*math::int_pow<2>(R)*si - 2*L*math::int_pow<2>(R)) - 2*so*(3*math::int_pow<2>(L)*R*math::int_pow<2>(si) + math::int_pow<2>(L)*R - L*math::int_pow<2>(R)
   - 2*si*(2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R))))) - sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3)*(-3*math::int_pow<3>(L) + 3*math::int_pow<2>(L)*R 
   - 4*math::int_pow<2>(eta)*(math::int_pow<3>(L) - math::int_pow<2>(L)*R) + 4*eta*(math::int_pow<3>(L) - math::int_pow<2>(L)*R) + 2*math::int_pow<3>(si)*(2*math::int_pow<3>(L)*math::int_pow<2>(eta)
   - 2*math::int_pow<3>(L)*eta + 3*math::int_pow<3>(L)) - 4*math::int_pow<2>(si)*(3*math::int_pow<3>(L) - math::int_pow<2>(L)*R + math::int_pow<2>(eta)*(3*math::int_pow<3>(L) - math::int_pow<2>(L)*R)
   - eta*(3*math::int_pow<3>(L) - math::int_pow<2>(L)*R)) + si*(9*math::int_pow<3>(L) - 6*math::int_pow<2>(L)*R + 4*math::int_pow<2>(eta)*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R)
   - 4*eta*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R)) - so*(4*math::int_pow<2>(L)*R*math::int_pow<2>(eta) - 4*math::int_pow<2>(L)*R*eta + 3*math::int_pow<2>(L)*R 
   + 2*math::int_pow<2>(si)*(2*math::int_pow<2>(L)*R*math::int_pow<2>(eta) - 2*math::int_pow<2>(L)*R*eta + 3*math::int_pow<2>(L)*R) - 8*si*(math::int_pow<2>(L)*R*math::int_pow<2>(eta) 
   - math::int_pow<2>(L)*R*eta + math::int_pow<2>(L)*R)) + 4*math::int_pow<2>(xi)*(math::int_pow<3>(L)*math::int_pow<3>(si) - math::int_pow<3>(L) + math::int_pow<2>(L)*R 
   - math::int_pow<2>(si)*(3*math::int_pow<3>(L) - math::int_pow<2>(L)*R) + si*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) - so*(math::int_pow<2>(L)*R*math::int_pow<2>(si) 
   - 2*math::int_pow<2>(L)*R*si + math::int_pow<2>(L)*R)) - 4*xi*(math::int_pow<3>(L)*math::int_pow<3>(si) - math::int_pow<3>(L) + math::int_pow<2>(L)*R - math::int_pow<2>(si)*(3*math::int_pow<3>(L) 
   - math::int_pow<2>(L)*R) + si*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) - so*(math::int_pow<2>(L)*R*math::int_pow<2>(si) - 2*math::int_pow<2>(L)*R*si + math::int_pow<2>(L)*R)) 
   + math::int_pow<2>(zeta)*(-3*math::int_pow<3>(L) + 9*math::int_pow<2>(L)*R - 9*L*math::int_pow<2>(R) + 3*math::int_pow<3>(R) - 4*math::int_pow<2>(eta)*(math::int_pow<3>(L) - 3*math::int_pow<2>(L)*R 
   + 3*L*math::int_pow<2>(R) - math::int_pow<3>(R)) + 4*eta*(math::int_pow<3>(L) - 3*math::int_pow<2>(L)*R + 3*L*math::int_pow<2>(R) - math::int_pow<3>(R)) 
   + 2*math::int_pow<3>(si)*(2*math::int_pow<3>(L)*math::int_pow<2>(eta) - 2*math::int_pow<3>(L)*eta + 3*math::int_pow<3>(L)) - 12*math::int_pow<2>(si)*(math::int_pow<3>(L) 
   - math::int_pow<2>(L)*R + math::int_pow<2>(eta)*(math::int_pow<3>(L) - math::int_pow<2>(L)*R) - eta*(math::int_pow<3>(L) - math::int_pow<2>(L)*R)) + 3*si*(3*math::int_pow<3>(L) - 6*math::int_pow<2>(L)*R + 3*L*math::int_pow<2>(R) + 4*math::int_pow<2>(eta)*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) - 4*eta*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R))) - 2*math::int_pow<3>(so)*(2*math::int_pow<3>(R)*math::int_pow<2>(eta) - 2*math::int_pow<3>(R)*eta + 3*math::int_pow<3>(R)) - 6*math::int_pow<2>(so)*(2*L*math::int_pow<2>(R) - 2*math::int_pow<3>(R) + 2*math::int_pow<2>(eta)*(L*math::int_pow<2>(R) - math::int_pow<3>(R)) - 2*eta*(L*math::int_pow<2>(R) - math::int_pow<3>(R)) - si*(2*L*math::int_pow<2>(R)*math::int_pow<2>(eta) - 2*L*math::int_pow<2>(R)*eta + 3*L*math::int_pow<2>(R))) - 3*so*(3*math::int_pow<2>(L)*R - 6*L*math::int_pow<2>(R) + 3*math::int_pow<3>(R) + 4*math::int_pow<2>(eta)*(math::int_pow<2>(L)*R - 2*L*math::int_pow<2>(R) + math::int_pow<3>(R)) - 4*eta*(math::int_pow<2>(L)*R - 2*L*math::int_pow<2>(R) + math::int_pow<3>(R)) + 2*math::int_pow<2>(si)*(2*math::int_pow<2>(L)*R*math::int_pow<2>(eta) - 2*math::int_pow<2>(L)*R*eta + 3*math::int_pow<2>(L)*R) - 8*si*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R) + math::int_pow<2>(eta)*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)) - eta*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)))) + 4*math::int_pow<2>(xi)*(math::int_pow<3>(L)*math::int_pow<3>(si) - math::int_pow<3>(L) + 3*math::int_pow<2>(L)*R - 3*L*math::int_pow<2>(R) - math::int_pow<3>(R)*math::int_pow<3>(so) + math::int_pow<3>(R) - 3*math::int_pow<2>(si)*(math::int_pow<3>(L) - math::int_pow<2>(L)*R) + 3*si*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) + 3*math::int_pow<2>(so)*(L*math::int_pow<2>(R)*si - L*math::int_pow<2>(R) + math::int_pow<3>(R)) - 3*so*(math::int_pow<2>(L)*R*math::int_pow<2>(si) + math::int_pow<2>(L)*R - 2*L*math::int_pow<2>(R) + math::int_pow<3>(R) - 2*si*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)))) - 4*xi*(math::int_pow<3>(L)*math::int_pow<3>(si) - math::int_pow<3>(L) + 3*math::int_pow<2>(L)*R - 3*L*math::int_pow<2>(R) - math::int_pow<3>(R)*math::int_pow<3>(so) + math::int_pow<3>(R) - 3*math::int_pow<2>(si)*(math::int_pow<3>(L) - math::int_pow<2>(L)*R) + 3*si*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) + 3*math::int_pow<2>(so)*(L*math::int_pow<2>(R)*si - L*math::int_pow<2>(R) + math::int_pow<3>(R)) - 3*so*(math::int_pow<2>(L)*R*math::int_pow<2>(si) + math::int_pow<2>(L)*R - 2*L*math::int_pow<2>(R) + math::int_pow<3>(R) - 2*si*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R))))) - 2*zeta*(-3*math::int_pow<3>(L) + 6*math::int_pow<2>(L)*R - 3*L*math::int_pow<2>(R) - 4*math::int_pow<2>(eta)*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) + 4*eta*(math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) + 2*math::int_pow<3>(si)*(2*math::int_pow<3>(L)*math::int_pow<2>(eta) - 2*math::int_pow<3>(L)*eta + 3*math::int_pow<3>(L)) - 4*math::int_pow<2>(si)*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R + math::int_pow<2>(eta)*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) - eta*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R)) + si*(9*math::int_pow<3>(L) - 12*math::int_pow<2>(L)*R + 3*L*math::int_pow<2>(R) + 4*math::int_pow<2>(eta)*(3*math::int_pow<3>(L) - 4*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) - 4*eta*(3*math::int_pow<3>(L) - 4*math::int_pow<2>(L)*R + L*math::int_pow<2>(R))) - 2*math::int_pow<2>(so)*(2*L*math::int_pow<2>(R)*math::int_pow<2>(eta) - 2*L*math::int_pow<2>(R)*eta + 2*L*math::int_pow<2>(R) - si*(2*L*math::int_pow<2>(R)*math::int_pow<2>(eta) - 2*L*math::int_pow<2>(R)*eta + 3*L*math::int_pow<2>(R))) - 2*so*(3*math::int_pow<2>(L)*R - 3*L*math::int_pow<2>(R) + 4*math::int_pow<2>(eta)*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)) - 4*eta*(math::int_pow<2>(L)*R - L*math::int_pow<2>(R)) + 2*math::int_pow<2>(si)*(2*math::int_pow<2>(L)*R*math::int_pow<2>(eta) - 2*math::int_pow<2>(L)*R*eta + 3*math::int_pow<2>(L)*R) - 4*si*(2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R) + math::int_pow<2>(eta)*(2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R)) - eta*(2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R)))) + 4*math::int_pow<2>(xi)*(math::int_pow<3>(L)*math::int_pow<3>(si) - math::int_pow<3>(L) + 2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R) - math::int_pow<2>(si)*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) + si*(3*math::int_pow<3>(L) - 4*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) + math::int_pow<2>(so)*(L*math::int_pow<2>(R)*si - L*math::int_pow<2>(R)) - 2*so*(math::int_pow<2>(L)*R*math::int_pow<2>(si) + math::int_pow<2>(L)*R - L*math::int_pow<2>(R) - si*(2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R)))) - 4*xi*(math::int_pow<3>(L)*math::int_pow<3>(si) - math::int_pow<3>(L) + 2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R) - math::int_pow<2>(si)*(3*math::int_pow<3>(L) - 2*math::int_pow<2>(L)*R) + si*(3*math::int_pow<3>(L) - 4*math::int_pow<2>(L)*R + L*math::int_pow<2>(R)) + math::int_pow<2>(so)*(L*math::int_pow<2>(R)*si - L*math::int_pow<2>(R)) - 2*so*(math::int_pow<2>(L)*R*math::int_pow<2>(si) + math::int_pow<2>(L)*R - L*math::int_pow<2>(R) - si*(2*math::int_pow<2>(L)*R - L*math::int_pow<2>(R)))))))/pow(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3, 3.0/2.0);
   return J_sph_3D_result;

}

double Jac_sph_log_3D_00(double L, double R, double eta, double xi, double zeta) {

   double Jac_sph_log_3D_00_result;
   Jac_sph_log_3D_00_result = 1.0*pow(R/L, -0.5)*sqrt(L*R)*exp(1.0*zeta*log(R/L))*log(R/L)/sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3);
   return Jac_sph_log_3D_00_result;

}

double Jac_sph_log_3D_01(double L, double R, double eta, double xi, double zeta) {

   double Jac_sph_log_3D_01_result;
   Jac_sph_log_3D_01_result = -2*(2*eta*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) - pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R))/pow(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3, 3.0/2.0);
   return Jac_sph_log_3D_01_result;

}

double Jac_sph_log_3D_02(double L, double R, double eta, double xi, double zeta) {

   double Jac_sph_log_3D_02_result;
   Jac_sph_log_3D_02_result = -2*(2*xi*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) - pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R))/pow(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3, 3.0/2.0);
   return Jac_sph_log_3D_02_result;

}

double Jac_sph_log_3D_10(double L, double R, double eta, double xi, double zeta) {

   double Jac_sph_log_3D_10_result;
   Jac_sph_log_3D_10_result = (2.0*eta*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R)*log(R/L) - 1.0*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R)*log(R/L))/sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3);
   return Jac_sph_log_3D_10_result;

}

double Jac_sph_log_3D_11(double L, double R, double eta, double xi, double zeta) {

   double Jac_sph_log_3D_11_result;
   Jac_sph_log_3D_11_result = 4*(2*math::int_pow<2>(xi)*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) - 2*xi*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) + pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R))*sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3)/(16*math::int_pow<4>(eta) - 32*math::int_pow<3>(eta) + 40*math::int_pow<2>(eta) - 24*eta + 16*math::int_pow<4>(xi) - 32*math::int_pow<3>(xi) + 8*math::int_pow<2>(xi)*(4*math::int_pow<2>(eta) - 4*eta + 5) - 8*xi*(4*math::int_pow<2>(eta) - 4*eta + 3) + 9);
   return Jac_sph_log_3D_11_result;

}

double Jac_sph_log_3D_12(double L, double R, double eta, double xi, double zeta) {

   double Jac_sph_log_3D_12_result;
   Jac_sph_log_3D_12_result = 2*(2*eta*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) - 2*xi*(2*eta*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) - pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R)) - pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R))/pow(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3, 3.0/2.0);
   return Jac_sph_log_3D_12_result;

}

double Jac_sph_log_3D_20(double L, double R, double eta, double xi, double zeta) {

   double Jac_sph_log_3D_20_result;
   Jac_sph_log_3D_20_result = (2.0*xi*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R)*log(R/L) - 1.0*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R)*log(R/L))/sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3);
   return Jac_sph_log_3D_20_result;

}

double Jac_sph_log_3D_21(double L, double R, double eta, double xi, double zeta) {

   double Jac_sph_log_3D_21_result;
   Jac_sph_log_3D_21_result = 2*(2*eta*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) - 2*xi*(2*eta*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) - pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R)) - pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R))/pow(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3, 3.0/2.0);
   return Jac_sph_log_3D_21_result;

}

double Jac_sph_log_3D_22(double L, double R, double eta, double xi, double zeta) {

   double Jac_sph_log_3D_22_result;
   Jac_sph_log_3D_22_result = 4*(2*math::int_pow<2>(eta)*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) - 2*eta*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) + pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R))*sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3)/(16*math::int_pow<4>(eta) - 32*math::int_pow<3>(eta) + 40*math::int_pow<2>(eta) - 24*eta + 16*math::int_pow<4>(xi) - 32*math::int_pow<3>(xi) + 8*math::int_pow<2>(xi)*(4*math::int_pow<2>(eta) - 4*eta + 5) - 8*xi*(4*math::int_pow<2>(eta) - 4*eta + 3) + 9);
   return Jac_sph_log_3D_22_result;

}

double J_sph_log_3D(double L, double R, double eta, double xi, double zeta) {

   double J_sph_log_3D_result;
   J_sph_log_3D_result = (16384.0*math::int_pow<12>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 98304.0*pow(eta, 11)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 319488.0*pow(eta, 10)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 696320.0*pow(eta, 9)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1121280.0*pow(eta, 8)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 1388544.0*pow(eta, 7)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1352704.0*pow(eta, 6)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 1041408.0*math::int_pow<5>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 630720.0*math::int_pow<4>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 293760.0*math::int_pow<3>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 101088.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 23328.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 16384.0*pow(xi, 12)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 98304.0*pow(xi, 11)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + pow(xi, 10)*(98304.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 98304.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 319488.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + pow(xi, 9)*(-491520.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 491520.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 696320.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + pow(xi, 8)*(245760.0*math::int_pow<4>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 491520.0*math::int_pow<3>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1597440.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 1351680.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1121280.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + pow(xi, 7)*(-983040.0*math::int_pow<4>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1966080.0*math::int_pow<3>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 3440640.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 2457600.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 1388544.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + pow(xi, 6)*(327680.0*pow(eta, 6)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 983040.0*math::int_pow<5>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 3194880.0*math::int_pow<4>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 4751360.0*math::int_pow<3>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 5468160.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 3256320.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1352704.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + pow(xi, 5)*(-983040.0*pow(eta, 6)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 2949120.0*math::int_pow<5>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 6144000.0*math::int_pow<4>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 7372800.0*math::int_pow<3>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 6426624.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 3231744.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 1041408.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + math::int_pow<4>(xi)*(245760.0*pow(eta, 8)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 983040.0*pow(eta, 7)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 3194880.0*pow(eta, 6)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 6144000.0*math::int_pow<5>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 8693760.0*math::int_pow<4>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 8294400.0*math::int_pow<3>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 5729280.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 2442240.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 630720.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + math::int_pow<3>(xi)*(-491520.0*pow(eta, 8)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1966080.0*pow(eta, 7)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 4751360.0*pow(eta, 6)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 7372800.0*math::int_pow<5>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 8294400.0*math::int_pow<4>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 6594560.0*math::int_pow<3>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 3778560.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1382400.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 293760.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + math::int_pow<2>(xi)*(98304.0*pow(eta, 10)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 491520.0*pow(eta, 9)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1597440.0*pow(eta, 8)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 3440640.0*pow(eta, 7)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 5468160.0*pow(eta, 6)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 6426624.0*math::int_pow<5>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 5729280.0*math::int_pow<4>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 3778560.0*math::int_pow<3>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1814400.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 570240.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 101088.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + xi*(-98304.0*pow(eta, 10)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 491520.0*pow(eta, 9)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 1351680.0*pow(eta, 8)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 2457600.0*pow(eta, 7)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 3256320.0*pow(eta, 6)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 3231744.0*math::int_pow<5>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 2442240.0*math::int_pow<4>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 1382400.0*math::int_pow<3>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 570240.0*math::int_pow<2>(eta)*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) + 155520.0*eta*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L) - 23328.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L)) + 2916.0*pow(R/L, -1.5)*pow(R/L, 3.0*zeta)*pow(L*R, 1.5)*log(R/L))/(sqrt(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi) - 4*xi + 3)*(16384*pow(eta, 14) - 114688*pow(eta, 13) + 430080*math::int_pow<12>(eta) - 1089536*pow(eta, 11) + 2057216*pow(eta, 10) - 3032064*pow(eta, 9) + 3582208*pow(eta, 8) - 3435520*pow(eta, 7) + 2686656*pow(eta, 6) - 1705536*math::int_pow<5>(eta) + 867888*math::int_pow<4>(eta) - 344736*math::int_pow<3>(eta) + 102060*math::int_pow<2>(eta) - 20412*eta + 16384*pow(xi, 14) - 114688*pow(xi, 13) + 28672*pow(xi, 12)*(4*math::int_pow<2>(eta) - 4*eta + 15) - 57344*pow(xi, 11)*(12*math::int_pow<2>(eta) - 12*eta + 19) + 7168*pow(xi, 10)*(48*math::int_pow<4>(eta) - 96*math::int_pow<3>(eta) + 360*math::int_pow<2>(eta) - 312*eta + 287) - 7168*pow(xi, 9)*(240*math::int_pow<4>(eta) - 480*math::int_pow<3>(eta) + 920*math::int_pow<2>(eta) - 680*eta + 423) + 1792*pow(xi, 8)*(320*pow(eta, 6) - 960*math::int_pow<5>(eta) + 3600*math::int_pow<4>(eta) - 5600*math::int_pow<3>(eta) + 7020*math::int_pow<2>(eta) - 4380*eta + 1999) - 1024*pow(xi, 7)*(2240*pow(eta, 6) - 6720*math::int_pow<5>(eta) + 15120*math::int_pow<4>(eta) - 19040*math::int_pow<3>(eta) + 17892*math::int_pow<2>(eta) - 9492*eta + 3355) + 448*pow(xi, 6)*(1280*pow(eta, 8) - 5120*pow(eta, 7) + 19200*pow(eta, 6) - 39680*math::int_pow<5>(eta) + 61280*math::int_pow<4>(eta) - 62400*math::int_pow<3>(eta) + 46576*math::int_pow<2>(eta) - 21136*eta + 5997) - 448*pow(xi, 5)*(3840*pow(eta, 8) - 15360*pow(eta, 7) + 39680*pow(eta, 6) - 65280*math::int_pow<5>(eta) + 79008*math::int_pow<4>(eta) - 67136*math::int_pow<3>(eta) + 41520*math::int_pow<2>(eta) - 16272*eta + 3807) + 112*math::int_pow<4>(xi)*(3072*pow(eta, 10) - 15360*pow(eta, 9) + 57600*pow(eta, 8) - 138240*pow(eta, 7) + 245120*pow(eta, 6) - 316032*math::int_pow<5>(eta) + 308640*math::int_pow<4>(eta) - 221120*math::int_pow<3>(eta) + 115740*math::int_pow<2>(eta) - 39420*eta + 7749) - 224*math::int_pow<3>(xi)*(3072*pow(eta, 10) - 15360*pow(eta, 9) + 44800*pow(eta, 8) - 87040*pow(eta, 7) + 124800*pow(eta, 6) - 134272*math::int_pow<5>(eta) + 110560*math::int_pow<4>(eta) - 68160*math::int_pow<3>(eta) + 30780*math::int_pow<2>(eta) - 9180*eta + 1539) + 28*math::int_pow<2>(xi)*(4096*math::int_pow<12>(eta) - 24576*pow(eta, 11) + 92160*pow(eta, 10) - 235520*pow(eta, 9) + 449280*pow(eta, 8) - 654336*pow(eta, 7) + 745216*pow(eta, 6) - 664320*math::int_pow<5>(eta) + 462960*math::int_pow<4>(eta) - 246240*math::int_pow<3>(eta) + 96552*math::int_pow<2>(eta) - 25272*eta + 3645) - 28*xi*(4096*math::int_pow<12>(eta) - 24576*pow(eta, 11) + 79872*pow(eta, 10) - 174080*pow(eta, 9) + 280320*pow(eta, 8) - 347136*pow(eta, 7) + 338176*pow(eta, 6) - 260352*math::int_pow<5>(eta) + 157680*math::int_pow<4>(eta) - 73440*math::int_pow<3>(eta) + 25272*math::int_pow<2>(eta) - 5832*eta + 729) + 2187));
   return J_sph_log_3D_result;

}
}