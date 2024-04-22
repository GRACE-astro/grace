/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                       This file is part of 'Thunder'                       *
 ******************************************************************************/


#ifndef THUNDER__JACOBIANS_JACOBIANS_SPH_2D__H
#define THUNDER__JACOBIANS_JACOBIANS_SPH_2D__H
namespace thunder {
double Jac_sph_2D_00(double L, double R, double eta, double si, double so);
double Jac_sph_2D_01(double L, double R, double eta, double si, double so, double zeta);
double Jac_sph_2D_10(double L, double R, double eta, double si, double so);
double Jac_sph_2D_11(double L, double R, double eta, double si, double so, double zeta);
double J_sph_2D(double L, double R, double eta, double si, double so, double zeta);
double Jac_sph_log_2D_00(double L, double R, double eta, double zeta);
double Jac_sph_log_2D_01(double L, double R, double eta, double zeta);
double Jac_sph_log_2D_10(double L, double R, double eta, double zeta);
double Jac_sph_log_2D_11(double L, double R, double eta, double zeta);
double J_sph_log_2D(double L, double R, double eta, double zeta);
}
#endif

