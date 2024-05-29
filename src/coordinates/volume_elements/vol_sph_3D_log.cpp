/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                       This file is part of 'GRACE'                       *
 ******************************************************************************/
#include <grace/coordinates/volume_elements/vol_sph_3D_log.hh>
#include <grace/utils/math.hh>
#include <math.h>

double dVol_sph_3D_log(double L, double R, double dxi, double dzeta, double eta, double xi0, double zeta0) {

   double dVol_sph_3D_log_result;
   dVol_sph_3D_log_result = -1.0/3.0*((2*pow(R, 3*zeta0)*xi0*(math::int_pow<3>(L)*pow(R, 3*dzeta)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(eta) 
      - 4*eta + 4*math::int_pow<2>(xi0) - 4*xi0 + 3)) - pow(L, 3*dzeta + 3)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) - 4*xi0 + 3))) 
      - pow(R, 3*zeta0)*(math::int_pow<3>(L)*pow(R, 3*dzeta)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) - 4*xi0 + 3)) 
      - pow(L, 3*dzeta + 3)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) - 4*xi0 + 3))))*sqrt(fabs(4*math::int_pow<2>(eta) 
      - 4*eta + 4*math::int_pow<2>(xi0) - 4*xi0 + 3))*fabs(4*math::int_pow<2>(dxi) - 4*dxi + 4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) + 4*xi0*(2*dxi - 1) + 3) 
      - (2*pow(R, 3*zeta0)*dxi*(math::int_pow<3>(L)*pow(R, 3*dzeta)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(dxi) - 4*dxi + 4*math::int_pow<2>(eta) 
      - 4*eta + 4*math::int_pow<2>(xi0) + 4*xi0*(2*dxi - 1) + 3)) - pow(L, 3*dzeta + 3)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(dxi) - 4*dxi + 4*math::int_pow<2>(eta) 
      - 4*eta + 4*math::int_pow<2>(xi0) + 4*xi0*(2*dxi - 1) + 3))) + 2*pow(R, 3*zeta0)*xi0*(math::int_pow<3>(L)*pow(R, 3*dzeta)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(dxi) 
      - 4*dxi + 4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) + 4*xi0*(2*dxi - 1) + 3)) - pow(L, 3*dzeta + 3)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(dxi) 
      - 4*dxi + 4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) + 4*xi0*(2*dxi - 1) + 3))) 
      - pow(R, 3*zeta0)*(math::int_pow<3>(L)*pow(R, 3*dzeta)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(dxi) - 4*dxi + 4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) 
      + 4*xi0*(2*dxi - 1) + 3)) - pow(L, 3*dzeta + 3)*cos((1.0/2.0)*atan2(0, 4*math::int_pow<2>(dxi) - 4*dxi + 4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) 
      + 4*xi0*(2*dxi - 1) + 3))))*fabs(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) - 4*xi0 + 3)*sqrt(fabs(4*math::int_pow<2>(dxi) - 4*dxi + 4*math::int_pow<2>(eta) 
      - 4*eta + 4*math::int_pow<2>(xi0) + 4*xi0*(2*dxi - 1) + 3)))/((2*pow(L, 3*dzeta + 3*zeta0)*math::int_pow<2>(eta) - 2*pow(L, 3*dzeta + 3*zeta0)*eta 
      + pow(L, 3*dzeta + 3*zeta0))*fabs(4*math::int_pow<2>(eta) - 4*eta + 4*math::int_pow<2>(xi0) - 4*xi0 + 3)*fabs(4*math::int_pow<2>(dxi) - 4*dxi + 4*math::int_pow<2>(eta) 
      - 4*eta + 4*math::int_pow<2>(xi0) + 4*xi0*(2*dxi - 1) + 3));
   return dVol_sph_3D_log_result;

}
