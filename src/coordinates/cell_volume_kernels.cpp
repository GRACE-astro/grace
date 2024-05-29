/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                       This file is part of 'project'                       *
 ******************************************************************************/
#include <grace/coordinates/cell_volume_kernels.h>
#include <math.h>
namespace grace {

double dVol_sph(double L, double R, double deta, double dzeta, double eta0, double zeta0) {

   double dVol_sph_2D_result;
   dVol_sph_2D_result = pow(L, 2)*deta*pow(dzeta, 2) - 2*pow(L, 2)*deta*dzeta - 1.0/2.0*pow(R, 2)*pow(dzeta, 2)*atan(2*eta0 - 1) + (1.0/2.0)*pow(R, 2)*pow(dzeta, 2)*atan(2*deta + 2*eta0 - 1) + zeta0*(2*pow(L, 2)*deta*dzeta - 2*L*R*dzeta*log(fabs(-2*eta0 + M_SQRT2*sqrt(2*pow(eta0, 2) - 2*eta0 + 1) + 1)) + 2*L*R*dzeta*log(fabs(2*deta + 2*eta0 - M_SQRT2*sqrt(2*pow(deta, 2) - 2*deta + 2*pow(eta0, 2) + 2*eta0*(2*deta - 1) + 1) - 1)) - pow(R, 2)*dzeta*atan(2*eta0 - 1) + pow(R, 2)*dzeta*atan(2*deta + 2*eta0 - 1)) - (L*R*pow(dzeta, 2) - L*R*dzeta)*log(fabs(-2*eta0 + M_SQRT2*sqrt(2*pow(eta0, 2) - 2*eta0 + 1) + 1)) + (L*R*pow(dzeta, 2) - L*R*dzeta)*log(fabs(2*deta + 2*eta0 - M_SQRT2*sqrt(2*pow(deta, 2) - 2*deta + 2*pow(eta0, 2) + 2*eta0*(2*deta - 1) + 1) - 1));
   return dVol_sph_2D_result;

}


double dVol_sph_ext(double L, double R, double deta, double dzeta, double eta0, double zeta0) {

   double dVol_sph_2D_ext_result;
   dVol_sph_2D_ext_result = zeta0*(-dzeta*(pow(L, 2) - 2*L*R + pow(R, 2))*atan(2*eta0 - 1) + dzeta*(pow(L, 2) - 2*L*R + pow(R, 2))*atan(2*deta + 2*eta0 - 1)) - 1.0/2.0*(pow(dzeta, 2)*(pow(L, 2) - 2*L*R + pow(R, 2)) - 2*dzeta*(pow(L, 2) - L*R))*atan(2*eta0 - 1) + (1.0/2.0)*(pow(dzeta, 2)*(pow(L, 2) - 2*L*R + pow(R, 2)) - 2*dzeta*(pow(L, 2) - L*R))*atan(2*deta + 2*eta0 - 1);
   return dVol_sph_2D_ext_result;

}

double dVol_sph_log(double L, double R, double deta, double dzeta, double eta0, double zeta0) {
   double dVol_sph_2D_log_result;
   dVol_sph_2D_log_result = zeta0*(-dzeta*(pow(L, 2) - 2*L*R + pow(R, 2))*atan(2*eta0 - 1) + dzeta*(pow(L, 2) - 2*L*R + pow(R, 2))*atan(2*deta + 2*eta0 - 1)) - 1.0/2.0*(pow(dzeta, 2)*(pow(L, 2) - 2*L*R + pow(R, 2)) - 2*dzeta*(pow(L, 2) - L*R))*atan(2*eta0 - 1) + (1.0/2.0)*(pow(dzeta, 2)*(pow(L, 2) - 2*L*R + pow(R, 2)) - 2*dzeta*(pow(L, 2) - L*R))*atan(2*deta + 2*eta0 - 1);
   return dVol_sph_2D_log_result;
}

} /* namespace grace */