#ifndef GRACE_PHYSICS_ID_FMTORUS_KERRSCHILD_HH
#define GRACE_PHYSICS_ID_FMTORUS_KERRSCHILD_HH

#include <array>
//**************************************************************************************************/
/* Auxiliaries */
//**************************************************************************************************/
/**
* @brief Helper indices for getting the metric and extrinsic curvature components
*/

namespace detail {

inline static double GRACE_HOST_DEVICE det(
   double gxx, double gxy, double gxz, double gyy, double gyz, double gzz
) 
{ 
return -(SQR(gxz)*gyy) 
            + 2*gxy*gxz*gyz 
            - gxx*SQR(gyz) 
            - SQR(gxy)*gzz 
            + gxx*gyy*gzz ;  
}

inline static void GRACE_HOST_DEVICE inverse(
   double invdet,
   double gxx, double gxy, double gxz, double gyy, double gyz, double gzz,
   double * uxx, double * uxy, double * uxz, double * uyy, double * uyz, double * uzz
)
{
   *uxx = (gyy*gzz - SQR(gyz))* invdet;
   *uxy = (-gxy*gzz + gxz*gyz)* invdet;
   *uxz = (-(gxz*gyy) + gxy*gyz)* invdet;
   *uyy = (gzz*gxx - SQR(gxz))* invdet;
   *uyz = (gxy*gxz - gxx*gyz)* invdet ; 
   *uzz = (-SQR(gxy) + gxx*gyy) *invdet;
}

}

KOKKOS_INLINE_FUNCTION
static void ComputeMetricAndInverse(double x, double y, double z, bool minkowski, double a,
                             double glower[][4], double gupper[][4]) {
  // NOTE(@pdmullen): The following commented out floor on z dealt with the metric
  // singularity encountered for small z near the horizon (e.g., see g_00). However, this
  // floor was operating on z even for r_ks > 1.0, where (I believe) the metric should be
  // well-behaved for all z.  We floor r_ks to 1.0, therefore, the floor on z is
  // seemingly not necessary, however, if something goes awry for z ~ 0 in future
  // applications (even after flooring r_ks = 1.0), consider revisiting this z floor...
  // if (fabs(z) < (SMALL_NUMBER)) z = (SMALL_NUMBER);
  double rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  double r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  double eps = 1e-6;
  if (r < eps) {
    r = 0.5*(eps + r*r/eps);
  }
  //r = fmax(r, 1.0);  // floor r_ks to 0.5*(r_inner + r_outer)

  // Set covariant components
  // null vector l
  double l_lower[4];
  l_lower[0] = 1.0;
  l_lower[1] = (r*x + (a)*y)/( SQR(r) + SQR(a) );
  l_lower[2] = (r*y - (a)*x)/( SQR(r) + SQR(a) );
  l_lower[3] = z/r;

  // g_nm = f*l_n*l_m + eta_nm, where eta_nm is Minkowski metric
  double f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  if (minkowski) {f=0.0;}
  glower[0][0] = f * l_lower[0]*l_lower[0] - 1.0;
  glower[0][1] = f * l_lower[0]*l_lower[1];
  glower[0][2] = f * l_lower[0]*l_lower[2];
  glower[0][3] = f * l_lower[0]*l_lower[3];
  glower[1][0] = glower[0][1];
  glower[1][1] = f * l_lower[1]*l_lower[1] + 1.0;
  glower[1][2] = f * l_lower[1]*l_lower[2];
  glower[1][3] = f * l_lower[1]*l_lower[3];
  glower[2][0] = glower[0][2];
  glower[2][1] = glower[1][2];
  glower[2][2] = f * l_lower[2]*l_lower[2] + 1.0;
  glower[2][3] = f * l_lower[2]*l_lower[3];
  glower[3][0] = glower[0][3];
  glower[3][1] = glower[1][3];
  glower[3][2] = glower[2][3];
  glower[3][3] = f * l_lower[3]*l_lower[3] + 1.0;

  // Set contravariant components
  // null vector l
  double l_upper[4];
  l_upper[0] = -1.0;
  l_upper[1] = l_lower[1];
  l_upper[2] = l_lower[2];
  l_upper[3] = l_lower[3];

  // g^nm = -f*l^n*l^m + eta^nm, where eta^nm is Minkowski metric
  gupper[0][0] = -f * l_upper[0]*l_upper[0] - 1.0;
  gupper[0][1] = -f * l_upper[0]*l_upper[1];
  gupper[0][2] = -f * l_upper[0]*l_upper[2];
  gupper[0][3] = -f * l_upper[0]*l_upper[3];
  gupper[1][0] = gupper[0][1];
  gupper[1][1] = -f * l_upper[1]*l_upper[1] + 1.0;
  gupper[1][2] = -f * l_upper[1]*l_upper[2];
  gupper[1][3] = -f * l_upper[1]*l_upper[3];
  gupper[2][0] = gupper[0][2];
  gupper[2][1] = gupper[1][2];
  gupper[2][2] = -f * l_upper[2]*l_upper[2] + 1.0;
  gupper[2][3] = -f * l_upper[2]*l_upper[3];
  gupper[3][0] = gupper[0][3];
  gupper[3][1] = gupper[1][3];
  gupper[3][2] = gupper[2][3];
  gupper[3][3] = -f * l_upper[3]*l_upper[3] + 1.0;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ComputeADMDecomposition
//! \brief computes ADM quantitiese in Cartesian Kerr-Schild coordinates

// QUESTION: doesn't this assume bh_mass to be 1?
KOKKOS_INLINE_FUNCTION
static void ComputeADMDecomposition(double x, double y, double z, bool minkowski, double a,
               double * alp,
               double * betax, double * betay, double * betaz,
               double * psi4,
               double * gxx, double * gxy, double * gxz, double * gyy, double * gyz, double * gzz,
               double * Kxx, double * Kxy, double * Kxz, double * Kyy, double * Kyz, double * Kzz) {
  // See comments above in ComputeMetricAndInverse
  double rad = sqrt(SQR(x) + SQR(y) + SQR(z));
  double r = sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(z)))/2.0);
  double eps = 1e-6;
  if (r < eps) {
    r = 0.5*(eps + r*r/eps);
  }
  //r = fmax(r, 1.0);

  // l covector (spatial components only)
  double l_d[3];
  l_d[0] = (r*x + (a)*y)/( SQR(r) + SQR(a) );
  l_d[1] = (r*y - (a)*x)/( SQR(r) + SQR(a) );
  l_d[2] = z/r;

  // l vector (spatial components only)
  double l_u[3] = {l_d[0], l_d[1], l_d[2]};

  //
  // g_nm = 2*H*l_n*l_m + eta_nm, where eta is the Minkowski metric
  double H = SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  if (minkowski) {H=0.0;}

  *alp = 1.0/sqrt(1. + 2.*H);
  *betax = 2.*H/(1. + 2.*H)*l_u[0];
  *betay = 2.*H/(1. + 2.*H)*l_u[1];
  *betaz = 2.*H/(1. + 2.*H)*l_u[2];
  double const beta_d[3] = {2.*H*l_u[0], 2.*H*l_u[1], 2.*H*l_u[2]};

  *gxx = 2.*H*l_d[0]*l_d[0] + 1.;
  *gxy = 2.*H*l_d[0]*l_d[1];
  *gxz = 2.*H*l_d[0]*l_d[2];
  *gyy = 2.*H*l_d[1]*l_d[1] + 1.;
  *gyz = 2.*H*l_d[1]*l_d[2];
  *gzz = 2.*H*l_d[2]*l_d[2] + 1.;

  //
  // conformal factor
  double const det = detail::det(*gxx, *gxy, *gxz, *gyy, *gyz, *gzz);
  *psi4 = pow(det, 1./3.);

  //
  // inverse metric
  double uxx, uxy, uxz, uyy, uyz, uzz;
  detail::inverse(1./det,
             *gxx, *gxy, *gxz, *gyy, *gyz, *gzz,
             &uxx, &uxy, &uxz, &uyy, &uyz, &uzz);
  double const g_uu[3][3] = {
    uxx, uxy, uxz,
    uxy, uyy, uyz,
    uxz, uyz, uzz
  };

  //
  // derivatives of the three metric (expressions taken from below)
  double const qa = 2.0*SQR(r) - SQR(rad) + SQR(a);
  double const qb = SQR(r) + SQR(a);
  double const qc = 3.0*SQR(a * z) - SQR(r)*SQR(r);
  double const dH_d[3] = {
    SQR(H)*x/(pow(r,3)) * ( ( qc ) )/ qa,
    SQR(H)*y/(pow(r,3)) * ( ( qc ) )/ qa,
    SQR(H)*z/(pow(r,5)) * ( ( qc * qb ) / qa - 2.0*SQR(a*r))
  };

  // \partial_i l_k
  double const dl_dd[3][3] = {
    // \partial_x l_k
    {x*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + r/( qb ),
    x*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) - a/( qb ),
    - x*z/(r*qa)},
    // \partial_y l_k
    {y*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(qb) * qa ) + a/( qb ),
    y*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(qb) * qa ) + r/( qb ),
    - y*z/(r*qa)},
    // \partial_z l_k
    {z/r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( (qb) * qa ),
    z/r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( (qb) * qa ),
    - SQR(z)/(SQR(r)*r) * ( qb )/( qa ) + 1.0/r},
  };

  double dg_ddd[3][3][3] = {0.0};
  for (int i = 0; i < 3; i++)
  for (int a = 0; a < 3; a++)
  for (int b = 0; b < 3; b++) {
    dg_ddd[i][a][b] = 2.*dH_d[i]*l_d[a]*l_d[b] +
                      2.*H*dl_dd[i][a]*l_d[b] +
                      2.*H*l_d[a]*dl_dd[i][b];
  }

  //
  // Compute Christoffel symbols
  double Gamma_udd[3][3][3];
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b)
  for (int c = 0; c < 3; ++c) {
    Gamma_udd[a][b][c] = 0.0;
    for (int d = 0; d < 3; ++d) {
      Gamma_udd[a][b][c] += 0.5*g_uu[a][d]*
                            (dg_ddd[c][b][d] + dg_ddd[b][d][c] - dg_ddd[d][b][c]);
    }
  }

  //
  // Derivatives of the shift vector
  double const dbeta_dd[3][3] = {
    // \partial_x \beta_i
    {2.*dH_d[0]*l_d[0] + 2.*H*dl_dd[0][0],
    2.*dH_d[0]*l_d[1] + 2.*H*dl_dd[0][1],
    2.*dH_d[0]*l_d[2] + 2.*H*dl_dd[0][2]},
    // \partial_y \beta_i
    {2.*dH_d[1]*l_d[0] + 2.*H*dl_dd[1][0],
    2.*dH_d[1]*l_d[1] + 2.*H*dl_dd[1][1],
    2.*dH_d[1]*l_d[2] + 2.*H*dl_dd[1][2]},
    // \partial_z \beta_i
    {2.*dH_d[2]*l_d[0] + 2.*H*dl_dd[2][0],
    2.*dH_d[2]*l_d[1] + 2.*H*dl_dd[2][1],
    2.*dH_d[2]*l_d[2] + 2.*H*dl_dd[2][2]},
  };
  /*double dbeta_dd[3][3];
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      dbeta_dd[a][b] = 2.*dH_d[a]*l_d[b] + 2.*H*dl_dd[a][b];
    }
  }*/

  //
  // Covariant derivative of the shift vector
  double Dbeta_dd[3][3];
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    Dbeta_dd[a][b] = dbeta_dd[a][b];
    for (int d = 0; d < 3; ++d) {
      Dbeta_dd[a][b] -= Gamma_udd[d][a][b]*beta_d[d];
    }
  }

  //
  // Extrinsic curvature: K_ab = 1/(2 alp) * (D_a beta_b + D_b beta_a)
  /*double delta[3][3] = {0.};
  delta[0][0] = 1.;
  delta[1][1] = 1.;
  delta[2][2] = 1.;*/
  double K_dd[3][3];
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    K_dd[a][b] = (Dbeta_dd[a][b] + Dbeta_dd[b][a])/(2.*(*alp));
    //K_dd[a][b] = 2*(*alp)/SQR(r)*(delta[a][b] - (2. + 1./r)*l_d[a]*l_d[b]);
  }

  *Kxx = K_dd[0][0];
  *Kxy = K_dd[0][1];
  *Kxz = K_dd[0][2];
  *Kyy = K_dd[1][1];
  *Kyz = K_dd[1][2];
  *Kzz = K_dd[2][2];
}


#endif /* GRACE_PHYSICS_ID_FMTORUS_KERRSCHILD_HH */