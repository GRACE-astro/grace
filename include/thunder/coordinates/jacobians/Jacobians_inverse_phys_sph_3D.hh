/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                       This file is part of 'Thunder'                       *
 ******************************************************************************/


#ifndef THUNDER__JACOBIANS_JACOBIANS_INVERSE_PHYS_SPH_3D__H
#define THUNDER__JACOBIANS_JACOBIANS_INVERSE_PHYS_SPH_3D__H
namespace thunder{
double Jac_sph_inv_phys_3D_00(double L, double R, double si, double so, double x, double y, double z);
double Jac_sph_inv_phys_3D_01(double L, double R, double si, double so, double x, double y, double z);
double Jac_sph_inv_phys_3D_02(double L, double R, double si, double so, double x, double y, double z);
double Jac_sph_inv_phys_3D_10(double x, double y);
double Jac_sph_inv_phys_3D_11(double x);
int Jac_sph_inv_phys_3D_12();
double Jac_sph_inv_phys_3D_20(double x, double z);
int Jac_sph_inv_phys_3D_21();
double Jac_sph_inv_phys_3D_22(double x);
double J_sph_inv_phys_3D(double L, double R, double si, double so, double x, double y, double z);
double Jac_sph_log_inv_phys_3D_00(double L, double R, double x, double y, double z);
double Jac_sph_log_inv_phys_3D_01(double L, double R, double x, double y, double z);
double Jac_sph_log_inv_phys_3D_02(double L, double R, double x, double y, double z);
double Jac_sph_log_inv_phys_3D_10(double x, double y);
double Jac_sph_log_inv_phys_3D_11(double x);
int Jac_sph_log_inv_phys_3D_12();
double Jac_sph_log_inv_phys_3D_20(double x, double z);
int Jac_sph_log_inv_phys_3D_21();
double Jac_sph_log_inv_phys_3D_22(double x);
double J_sph_log_inv_phys_3D(double L, double R, double x, double y, double z);
}
#endif

