/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                       This file is part of 'Thunder'                       *
 ******************************************************************************/
#include <thunder/coordinates/jacobians/Jacobians_sph_2D.hh>
#include <math.h>
namespace thunder { 
double Jac_sph_2D_00(double L, double R, double eta, double si, double so) {

   double Jac_sph_2D_00_result;
   Jac_sph_2D_00_result = -(L*si - R*so - sqrt(4*pow(eta, 2) - 4*eta + 2)*(L*si - L - R*so + R))/sqrt(4*pow(eta, 2) - 4*eta + 2);
   return Jac_sph_2D_00_result;

}

double Jac_sph_2D_01(double L, double R, double eta, double si, double so, double zeta) {

   double Jac_sph_2D_01_result;
   Jac_sph_2D_01_result = -2*(2*eta - 1)*(L*si - zeta*(L*si - R*so) - sqrt(4*pow(eta, 2) - 4*eta + 2)*(L*si - L - zeta*(L*si - L - R*so + R)))/pow(4*pow(eta, 2) - 4*eta + 2, 3.0/2.0) - (2*eta - 1)*(L*si - L - zeta*(L*si - L - R*so + R))/(2*pow(eta, 2) - 2*eta + 1);
   return Jac_sph_2D_01_result;

}

double Jac_sph_2D_10(double L, double R, double eta, double si, double so) {

   double Jac_sph_2D_10_result;
   Jac_sph_2D_10_result = -(si*(2*L*eta - L) - so*(2*R*eta - R) + sqrt(4*pow(eta, 2) - 4*eta + 2)*(-L + R + 2*eta*(L - R) - si*(2*L*eta - L) + so*(2*R*eta - R)))/sqrt(4*pow(eta, 2) - 4*eta + 2);
   return Jac_sph_2D_10_result;

}

double Jac_sph_2D_11(double L, double R, double eta, double si, double so, double zeta) {

   double Jac_sph_2D_11_result;
   Jac_sph_2D_11_result = -2*(2*eta - 1)*(si*(2*L*eta - L) - zeta*(si*(2*L*eta - L) - so*(2*R*eta - R)) + sqrt(4*pow(eta, 2) - 4*eta + 2)*(2*L*eta - L - si*(2*L*eta - L) - zeta*(-L + R + 2*eta*(L - R) - si*(2*L*eta - L) + so*(2*R*eta - R))))/pow(4*pow(eta, 2) - 4*eta + 2, 3.0/2.0) + 2*(L*si - zeta*(L*si - R*so) + (2*eta - 1)*(2*L*eta - L - si*(2*L*eta - L) - zeta*(-L + R + 2*eta*(L - R) - si*(2*L*eta - L) + so*(2*R*eta - R)))/sqrt(4*pow(eta, 2) - 4*eta + 2) - sqrt(4*pow(eta, 2) - 4*eta + 2)*(L*si - L - zeta*(L*si - L - R*so + R)))/sqrt(4*pow(eta, 2) - 4*eta + 2);
   return Jac_sph_2D_11_result;

}

double J_sph_2D(double L, double R, double eta, double si, double so, double zeta) {

   double J_sph_inv_2D_result;
   J_sph_inv_2D_result = -(2*pow(L, 2) - 2*L*R + 4*pow(eta, 2)*(pow(L, 2) - L*R) - 4*eta*(pow(L, 2) - L*R) + pow(si, 2)*(4*pow(L, 2)*pow(eta, 2) - 4*pow(L, 2)*eta + 3*pow(L, 2)) - 2*si*(2*pow(L, 2) - L*R + 2*pow(eta, 2)*(2*pow(L, 2) - L*R) - 2*eta*(2*pow(L, 2) - L*R)) + so*(4*L*R*pow(eta, 2) - 4*L*R*eta + 2*L*R - si*(4*L*R*pow(eta, 2) - 4*L*R*eta + 3*L*R)) - zeta*(2*pow(L, 2) - 4*L*R + 2*pow(R, 2) + 4*pow(eta, 2)*(pow(L, 2) - 2*L*R + pow(R, 2)) - 4*eta*(pow(L, 2) - 2*L*R + pow(R, 2)) + pow(si, 2)*(4*pow(L, 2)*pow(eta, 2) - 4*pow(L, 2)*eta + 3*pow(L, 2)) - 4*si*(pow(L, 2) - L*R + 2*pow(eta, 2)*(pow(L, 2) - L*R) - 2*eta*(pow(L, 2) - L*R)) + pow(so, 2)*(4*pow(R, 2)*pow(eta, 2) - 4*pow(R, 2)*eta + 3*pow(R, 2)) + 2*so*(2*L*R - 2*pow(R, 2) + 4*pow(eta, 2)*(L*R - pow(R, 2)) - 4*eta*(L*R - pow(R, 2)) - si*(4*L*R*pow(eta, 2) - 4*L*R*eta + 3*L*R))) - sqrt(4*pow(eta, 2) - 4*eta + 2)*(2*pow(L, 2)*pow(si, 2) - si*(2*pow(L, 2) - L*R) - so*(2*L*R*si - L*R) - 2*zeta*(pow(L, 2)*pow(si, 2) + pow(R, 2)*pow(so, 2) - si*(pow(L, 2) - L*R) - so*(2*L*R*si - L*R + pow(R, 2)))))/(2*pow(eta, 2) - 2*eta + 1);
   return J_sph_inv_2D_result;

}

double Jac_sph_log_2D_00(double L, double R, double eta, double zeta) {

   double Jac_sph_log_2D_00_result;
   Jac_sph_log_2D_00_result = 1.0*pow(R/L, -0.5)*sqrt(L*R)*exp(1.0*zeta*log(R/L))*log(R/L)/sqrt(4*pow(eta, 2) - 4*eta + 2);
   return Jac_sph_log_2D_00_result;

}

double Jac_sph_log_2D_01(double L, double R, double eta, double zeta) {

   double Jac_sph_log_2D_01_result;
   Jac_sph_log_2D_01_result = -(2*eta*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R) - pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R))/((2*pow(eta, 2) - 2*eta + 1)*sqrt(4*pow(eta, 2) - 4*eta + 2));
   return Jac_sph_log_2D_01_result;

}

double Jac_sph_log_2D_10(double L, double R, double eta, double zeta) {

   double Jac_sph_log_2D_10_result;
   Jac_sph_log_2D_10_result = (2.0*eta*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R)*log(R/L) - 1.0*pow(R/L, -0.5)*pow(R/L, 1.0*zeta)*sqrt(L*R)*log(R/L))/sqrt(4*pow(eta, 2) - 4*eta + 2);
   return Jac_sph_log_2D_10_result;

}

double Jac_sph_log_2D_11(double L, double R, double eta, double zeta) {

   double Jac_sph_log_2D_11_result;
   Jac_sph_log_2D_11_result = (1.0/2.0)*pow(R/L, -0.5)*sqrt(L*R)*sqrt(4*pow(eta, 2) - 4*eta + 2)*exp(1.0*zeta*log(R/L))/(4*pow(eta, 4) - 8*pow(eta, 3) + 8*pow(eta, 2) - 4*eta + 1);
   return Jac_sph_log_2D_11_result;

}

double J_sph_log_2D(double L, double R, double eta, double zeta) {

   double J_sph_log_inv_2D_result;
   J_sph_log_inv_2D_result = (2.0*pow(eta, 2)*1.0/(R/L)*pow(R/L, 2.0*zeta)*pow(L*R, 1.0)*log(R/L) - 2.0*eta*1.0/(R/L)*pow(R/L, 2.0*zeta)*pow(L*R, 1.0)*log(R/L) + 0.5*1.0/(R/L)*pow(R/L, 2.0*zeta)*pow(L*R, 1.0)*log(R/L) + 0.5*1.0/(R/L)*pow(L*R, 1.0)*exp(2.0*zeta*log(R/L))*log(R/L))/(4*pow(eta, 4) - 8*pow(eta, 3) + 8*pow(eta, 2) - 4*eta + 1);
   return J_sph_log_inv_2D_result;

}
}