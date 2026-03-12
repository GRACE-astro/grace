#ifndef GRACE_UTILS_FMTORUS_HH
#define GRACE_UTILS_FMTORUS_HH

#include <Kokkos_Core.hpp>

struct torus_params_t {
  double spin;                                  // black hole spin
  double lapse_excision;                        // excision parameters
  double gamma_adi;                             // EOS parameters
  bool prograde;                              // flag indicating disk is prograde (FM)
  double r_edge, r_peak, rho_max;            // fixed torus parameters
  double l_peak;                                // fixed torus parameters
  double c_param;                               // calculated chakrabarti parameter
  double n_param;                               // fixed or calculated chakrabarti parameter
  double log_h_edge, log_h_peak;                // calculated torus parameters
  double ptot_over_rho_peak, rho_peak;          // more calculated torus parameters
  double r_outer_edge;                          // even more calculated torus parameters
  double psi, sin_psi, cos_psi;                 // tilt parameters
  double rho_min, rho_pow, pgas_min, pgas_pow;  // background parameters
  double rho_excise, pgas_excise;               // excision values 
  bool is_vertical_field;                     // use vertical field configuration
  bool fm_torus, chakrabarti_torus;           // FM versus Chakrabarti torus IC
};

KOKKOS_INLINE_FUNCTION
static void CalculateCN(struct torus_params_t pgen, double *cparam, double *nparam);

KOKKOS_INLINE_FUNCTION
static double CalculateL(struct torus_params_t pgen, double r, double sin_theta);

KOKKOS_INLINE_FUNCTION
static double CalculateCovariantUT(struct torus_params_t pgen, double r, double sin_theta, double l);

KOKKOS_INLINE_FUNCTION
static double CalculateLFromRPeak(struct torus_params_t pgen, double r);

KOKKOS_INLINE_FUNCTION
static double LogHAux(struct torus_params_t pgen, double r, double sin_theta);

KOKKOS_INLINE_FUNCTION
static double CalculateT(struct torus_params_t pgen, double rho, double ptot_over_rho);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct torus_params_t pgen,
                                         double x1, double x2, double x3,
                                         double *pr, double *ptheta, double *pphi);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(struct torus_params_t pgen,
                                           double r, double theta, double phi, double *pu0,
                                           double *pu1, double *pu2, double *pu3);

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(struct torus_params_t pgen,
                                     double r, double sin_theta, double *pu0, double *pu3);

KOKKOS_INLINE_FUNCTION
static void CalculateVectorPotentialInTiltedTorus(struct torus_params_t pgen,
                                                  double r, double theta, double phi,
                                                  double *patheta, double *paphi);

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct torus_params_t pgen,
                            double a0_bl, double a1_bl, double a2_bl, double a3_bl,
                            double x1, double x2, double x3,
                            double *pa0, double *pa1, double *pa2, double *pa3);

//----------------------------------------------------------------------------------------
// Function for calculating angular momentum variable l in Fishbone-Moncrief torus
// Inputs:
//   r: desired radius of pressure maximum
// Outputs:
//   returned value: l = u^t u_\phi such that pressure maximum occurs at r_peak
// Notes:
//   beware many different definitions of l abound; this is *not* -u_phi/u_t
//   Harm has a similar function: lfish_calc() in init.c
//     Harm's function assumes M = 1 and that corotation is desired
//     it is equivalent to this, though seeing this requires much manipulation
//   implements (3.8) from Fishbone & Moncrief 1976, ApJ 207 962
//   assumes corotation

KOKKOS_INLINE_FUNCTION
static double CalculateLFromRPeak(struct torus_params_t pgen, double r) {
  double sgn = (pgen.prograde) ? 1.0 : -1.0;
  double num = sgn*(SQR(r*r) + SQR(pgen.spin*r) - 2.0*SQR(pgen.spin)*r)
           - pgen.spin*(r*r - pgen.spin*pgen.spin)*sqrt(r);
  double denom = SQR(r) - 3.0*r + sgn*2.0*pgen.spin*sqrt(r);
  return 1.0/r * sqrt(1.0/r) * num/denom;
}


//----------------------------------------------------------------------------------------
// Function to calculate enthalpy in Fishbone-Moncrief torus or Chakrabarti torus
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: log(h)
// Notes:
//   enthalpy defined here as h = p_gas/rho
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//   implements first half of (FM 3.6)
//   references Chakrabarti, S. 1985, ApJ 288, 1

KOKKOS_INLINE_FUNCTION
static double LogHAux(struct torus_params_t pgen, double r, double sin_theta) {
  double logh;
  if (pgen.fm_torus) {
    double sin_sq_theta = SQR(sin_theta);
    double cos_sq_theta = 1.0 - sin_sq_theta;
    double delta = SQR(r) - 2.0*r + SQR(pgen.spin);            // \Delta
    double sigma = SQR(r) + SQR(pgen.spin)*cos_sq_theta;       // \Sigma
    double aa = SQR(SQR(r)+SQR(pgen.spin)) - delta*SQR(pgen.spin)*sin_sq_theta;  // A
    double exp_2nu = sigma * delta / aa;                       // \exp(2\nu) (FM 3.5)
    double exp_2psi = aa / sigma * sin_sq_theta;               // \exp(2\psi) (FM 3.5)
    double exp_neg2chi = exp_2nu / exp_2psi;                   // \exp(-2\chi) (cf. FM 2.15)
    double omega = 2.0*pgen.spin*r/aa;                         // \omega (FM 3.5)
    double var_a = sqrt(1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi);
    double var_b = 0.5 * log((1.0+var_a) / (sigma*delta/aa));
    double var_c = -0.5 * var_a;
    double var_d = -pgen.l_peak * omega;
    logh = var_b + var_c + var_d;                            // (FM 3.4)
  } else { // Chakrabarti
    double l = CalculateL(pgen, r, sin_theta);
    double u_t = CalculateCovariantUT(pgen, r, sin_theta, l);
    double l_edge = CalculateL(pgen, pgen.r_edge, 1.0);
    double u_t_edge = CalculateCovariantUT(pgen, pgen.r_edge, 1.0, l_edge);
    double h = u_t_edge/u_t;
    if (pgen.n_param==1.0) {
      h *= pow(l_edge/l, SQR(pgen.c_param)/(SQR(pgen.c_param)-1.0));
    } else {
      double pow_c = 2.0/pgen.n_param;
      double pow_l = 2.0-2.0/pgen.n_param;
      double pow_abs = pgen.n_param/(2.0-2.0*pgen.n_param);
      h *= (pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l   , pow_l)), pow_abs) *
            pow(fabs(1.0 - pow(pgen.c_param, pow_c)*pow(l_edge, pow_l)), -1.0*pow_abs));
    }
    if (isfinite(h) && h >= 1.0) {
      logh = log(h);
    } else {
      logh = -1.0;
    }
  }
  return logh;
}

//----------------------------------------------------------------------------------------
// Function for calculating c, n parameters controlling angular momentum profile
// in Chakrabarti torus, where l = c * lambda^n. edited so that n can be pre-specified
// such that the assumption of keplerian angular momentum at the inner edge is dropped

KOKKOS_INLINE_FUNCTION
static void CalculateCN(struct torus_params_t pgen, double *cparam, double *nparam) {
  double n_input = pgen.n_param;
  double nn; // slope of angular momentum profile
  double cc; // constant of angular momentum profile
  double l_edge = ((SQR(pgen.r_edge) + SQR(pgen.spin) - 2.0*pgen.spin*sqrt(pgen.r_edge))/
                 (sqrt(pgen.r_edge)*(pgen.r_edge - 2.0) + pgen.spin));
  double l_peak = ((SQR(pgen.r_peak) + SQR(pgen.spin) - 2.0*pgen.spin*sqrt(pgen.r_peak))/
                 (sqrt(pgen.r_peak)*(pgen.r_peak - 2.0) + pgen.spin));
  double lambda_edge = sqrt((l_edge*(-2.0*pgen.spin*l_edge + SQR(pgen.r_edge)*pgen.r_edge
                                   + SQR(pgen.spin)*(2.0+pgen.r_edge)))/
                          (2.0*pgen.spin + l_edge*(pgen.r_edge - 2.0)));
  double lambda_peak = sqrt((l_peak*(-2.0*pgen.spin*l_peak + SQR(pgen.r_peak)*pgen.r_peak
                                   + SQR(pgen.spin)*(2.0+pgen.r_peak)))/
                          (2.0*pgen.spin + l_peak*(pgen.r_peak - 2.0)));
  if (n_input == 0.0) {
    nn = log(l_peak/l_edge)/log(lambda_peak/lambda_edge);
    cc = l_edge*pow(lambda_edge, -nn);
  } else {
    nn = n_input;
    cc = l_peak*pow(lambda_peak, -nn);
  }
  *cparam = cc;
  *nparam = nn;
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating l in Chakrabarti torus

KOKKOS_INLINE_FUNCTION
static double CalculateL(struct torus_params_t pgen, double r, double sin_theta) {
  // Compute BL metric components
  double sigma = SQR(r) + SQR(pgen.spin)*(1.0-SQR(sin_theta));
  double g_00 = -1.0 + 2.0*r/sigma;
  double g_03 = -2.0*pgen.spin*r/sigma*SQR(sin_theta);
  double g_33 = (SQR(r) + SQR(pgen.spin) +
               2.0*SQR(pgen.spin)*r/sigma*SQR(sin_theta))*SQR(sin_theta);

  // Perform bisection
  double l_min = 1.0;
  double l_max = 100.0;
  double l_val = 0.5*(l_min + l_max);
  int max_iterations = 25;
  double tol_rel = 1.0e-8;
  for (int n=0; n<max_iterations; ++n) {
    double error_rel = 0.5*(l_max - l_min)/l_val;
    if (error_rel < tol_rel) {
      break;
    }
    double residual = pow(l_val/pgen.c_param, 2.0/pgen.n_param) +
                    (l_val*g_33 + SQR(l_val)*g_03)/(g_03 + l_val*g_00);
    if (residual < 0.0) {
      l_min = l_val;
      l_val = 0.5 * (l_min + l_max);
    } else if (residual > 0.0) {
      l_max = l_val;
      l_val = 0.5 * (l_min + l_max);
    } else if (residual == 0.0) {
      break;
    }
  }
  return l_val;
}

//----------------------------------------------------------------------------------------
// Function to calculate time component of contravariant four velocity in BL
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_theta: sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: u_t

KOKKOS_INLINE_FUNCTION
static double CalculateCovariantUT(struct torus_params_t pgen, double r, double sin_theta, double l) {
  // Compute BL metric components
  double sigma = SQR(r) + SQR(pgen.spin)*(1.0-SQR(sin_theta));
  double g_00 = -1.0 + 2.0*r/sigma;
  double g_03 = -2.0*pgen.spin*r/sigma*SQR(sin_theta);
  double g_33 = (SQR(r) + SQR(pgen.spin) +
               2.0*SQR(pgen.spin)*r/sigma*SQR(sin_theta))*SQR(sin_theta);

  // Compute time component of covariant BL 4-velocity
  double u_t = -sqrt(fmax((SQR(g_03) - g_00*g_33)/(g_33 + 2.0*l*g_03 + SQR(l)*g_00), 0.0));
  return u_t;
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct torus_params_t pgen,
                                         double x1, double x2, double x3,
                                         double *pr, double *ptheta, double *pphi) {
  double rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  double r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
          pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside tilted torus
// Inputs:
//   r: Boyer-Lindquist r
//   theta,phi: Boyer-Lindquist theta and phi in BH-aligned coordinates
// Outputs:
//   pu0,pu1,pu2,pu3: u^\mu set (Boyer-Lindquist coordinates)
// Notes:
//   first finds corresponding location in untilted torus
//   next calculates velocity at that point in untilted case
//   finally transforms that velocity into coordinates in which torus is tilted

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTiltedTorus(struct torus_params_t pgen,
                                           double r, double theta, double phi, double *pu0,
                                           double *pu1, double *pu2, double *pu3) {
  // Calculate corresponding location
  double sin_theta = sin(theta);
  double cos_theta = cos(theta);
  double sin_phi = sin(phi);
  double cos_phi = cos(phi);
  double sin_vartheta, cos_vartheta, varphi;
  if (pgen.psi != 0.0) {
    double x = sin_theta * cos_phi;
    double y = sin_theta * sin_phi;
    double z = cos_theta;
    double varx = pgen.cos_psi * x - pgen.sin_psi * z;
    double vary = y;
    double varz = pgen.sin_psi * x + pgen.cos_psi * z;
    sin_vartheta = sqrt(SQR(varx) + SQR(vary));
    cos_vartheta = varz;
    varphi = atan2(vary, varx);
  } else {
    sin_vartheta = fabs(sin_theta);
    cos_vartheta = cos_theta;
    varphi = (sin_theta < 0.0) ? (phi - M_PI) : phi;
  }
  double sin_varphi = sin(varphi);
  double cos_varphi = cos(varphi);

  // Calculate untilted velocity
  double u0_tilt, u3_tilt;
  CalculateVelocityInTorus(pgen, r, sin_vartheta, &u0_tilt, &u3_tilt);
  double u1_tilt = 0.0;
  double u2_tilt = 0.0;

  // Account for tilt
  *pu0 = u0_tilt;
  *pu1 = u1_tilt;
  if (pgen.psi != 0.0) {
    double dtheta_dvartheta =
        (pgen.cos_psi * sin_vartheta
         + pgen.sin_psi * cos_vartheta * cos_varphi) / sin_theta;
    double dtheta_dvarphi = -pgen.sin_psi * sin_vartheta * sin_varphi / sin_theta;
    double dphi_dvartheta = pgen.sin_psi * sin_varphi / SQR(sin_theta);
    double dphi_dvarphi = sin_vartheta / SQR(sin_theta)
        * (pgen.cos_psi * sin_vartheta + pgen.sin_psi * cos_vartheta * cos_varphi);
    *pu2 = dtheta_dvartheta * u2_tilt + dtheta_dvarphi * u3_tilt;
    *pu3 = dphi_dvartheta * u2_tilt + dphi_dvarphi * u3_tilt;
  } else {
    *pu2 = u2_tilt;
    *pu3 = u3_tilt;
  }
  if (sin_theta < 0.0) {
    *pu2 *= -1.0;
    *pu3 *= -1.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing 4-velocity components at a given position inside untilted disk
// Inputs:
//   r: Boyer-Lindquist r
//   sin_theta: sine of Boyer-Lindquist theta
// Outputs:
//   pu0: u^t set (Boyer-Lindquist coordinates)
//   pu3: u^\phi set (Boyer-Lindquist coordinates)
// Notes:
//   The formula for u^3 as a function of u_{(\phi)} is tedious to derive, but this
//       matches the formula used in Harm (init.c).

KOKKOS_INLINE_FUNCTION
static void CalculateVelocityInTorus(struct torus_params_t pgen,
                                    double r, double sin_theta, double *pu0, double *pu3) {
  // Compute BL metric components
  double sin_sq_theta = SQR(sin_theta);
  double cos_sq_theta = 1.0 - sin_sq_theta;
  double delta = SQR(r) - 2.0*r + SQR(pgen.spin);              // \Delta
  double sigma = SQR(r) + SQR(pgen.spin)*cos_sq_theta;         // \Sigma
  double aa = SQR(SQR(r)+SQR(pgen.spin)) - delta*SQR(pgen.spin)*sin_sq_theta;  // A
  double g_00 = -(1.0 - 2.0*r/sigma); // g_tt
  double g_03 = -2.0*pgen.spin*r/sigma * sin_sq_theta; // g_tp
  double g_33 = (sigma + (1.0 + 2.0*r/sigma) *
              SQR(pgen.spin) * sin_sq_theta) * sin_sq_theta; // g_pp
  double g00 = -aa/(delta*sigma); // g^tt
  double g03 = -2.0*pgen.spin*r/(delta*sigma); // g^tp

  double u0 = 0.0, u3 = 0.0;
  // Compute non-zero components of 4-velocity
  if (pgen.fm_torus) {
    double exp_2nu = sigma * delta / aa;                 // \exp(2\nu) (FM 3.5)
    double exp_2psi = aa / sigma * sin_sq_theta;         // \exp(2\psi) (FM 3.5)
    double exp_neg2chi = exp_2nu / exp_2psi;             // \exp(-2\chi) (cf. FM 2.15)
    double u_phi_proj_a = 1.0 + 4.0*SQR(pgen.l_peak)*exp_neg2chi;
    double u_phi_proj_b = -1.0 + sqrt(u_phi_proj_a);
    double u_phi_proj = sqrt(0.5 * u_phi_proj_b);        // (FM 3.3)
    u_phi_proj *= (pgen.prograde) ? 1.0 : -1.0;
    double u3_a = (1.0+SQR(u_phi_proj)) / (aa*sigma*delta);
    double u3_b = 2.0*pgen.spin*r * sqrt(u3_a);
    double u3_c = sqrt(sigma/aa) / sin_theta;
    u3 = u3_b + u3_c * u_phi_proj;
    double u0_a = (SQR(g_03) - g_00*g_33) * SQR(u3);
    double u0_b = sqrt(u0_a - g_00);
    u0 = -1.0/g_00 * (g_03*u3 + u0_b);
  } else { // Chakrabarti torus
    double l = CalculateL(pgen, r, sin_theta);
    double u_0 = CalculateCovariantUT(pgen, r, sin_theta, l); // u_t
    double omega = -(g_03 + l*g_00)/(g_33 + l*g_03);
    u0 = (g00 - l*g03) * u_0; // u^t
    u3 = omega * u0; // u^p
  }
  *pu0 = u0;
  *pu3 = u3;
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct torus_params_t pgen,
                            double a0_bl, double a1_bl, double a2_bl, double a3_bl,
                            double x1, double x2, double x3,
                            double *pa0, double *pa1, double *pa2, double *pa3) {
  double rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  double r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  double delta = SQR(r) - 2.0*r + SQR(pgen.spin);
  *pa0 = a0_bl + 2.0*r/delta * a1_bl;
  *pa1 = a1_bl * ( (r*x1+pgen.spin*x2)/(SQR(r) + SQR(pgen.spin)) - x2*pgen.spin/delta) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-pgen.spin*x1)/(SQR(r) + SQR(pgen.spin)) + x1*pgen.spin/delta) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(pgen.spin)));
  return;
}


#endif 