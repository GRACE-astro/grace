/**
 * @file cell_surface_3d_helpers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-04-17
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *                                    
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *   
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *   
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */

#ifndef THUNDER_COORDINATES_SURFACE_ELEMENTS_3D_HELPERS_HH
#define THUNDER_COORDINATES_SURFACE_ELEMENTS_3D_HELPERS_HH

#include <thunder_config.h>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/utils/integration.hh>

namespace thunder { namespace detail {
/*******************************************************************************************************************************************************/
/*******************************************************************************************************************************************************/
/*                                               LOGARITHMIC RADIUS DISTRIBUTION                                                                       */
/*******************************************************************************************************************************************************/
/*******************************************************************************************************************************************************/
double THUNDER_ALWAYS_INLINE 
cell_surface_zeta_3D_log_aux(double ri, double ro, double zeta, double eta, double xi) {
    return -(math::int_pow<2>(ri)*pow(ro/ri,2*zeta)*atan((2 + 4*math::int_pow<2>(xi) + 
         sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi) - 
         2*xi*(2 + sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi)))/(-1 + 2*eta))); 
}

double THUNDER_ALWAYS_INLINE 
cell_surface_eta_3D_log_aux(double ri, double ro, double zeta, double eta, double xi) {
    return sqrt((1 + 2*(-1 + eta)*eta)/(4 + 8*(-1 + eta)*eta))*math::int_pow<2>(ri)*pow(ro/ri,2*zeta)*
   atan((-1 + 2*xi)/sqrt(2 + 4*(-1 + eta)*eta)); 
}

double THUNDER_ALWAYS_INLINE 
cell_surface_xi_3D_log_aux(double ri, double ro, double zeta, double eta, double xi) {
    return math::int_pow<2>(ri)*pow(ro/ri,2*zeta)*sqrt((1 + 2*(-1 + xi)*xi)/(4 + 8*(-1 + xi)*xi))*
   atan((-1 + 2*eta)/sqrt(2 + 4*(-1 + xi)*xi)) ; 
}

double THUNDER_ALWAYS_INLINE 
get_cell_surface_zeta_3D_log(double ri, double ro, double zeta0, double eta0, double xi0, double deta, double dxi) {
    return (cell_surface_zeta_3D_log_aux(ri,ro,zeta0,eta0,xi0) + cell_surface_zeta_3D_log_aux(ri,ro,zeta0,eta0+deta,xi0+dxi))
    - (cell_surface_zeta_3D_log_aux(ri,ro,zeta0,eta0+deta,xi0) + cell_surface_zeta_3D_log_aux(ri,ro,zeta0,eta0,xi0+dxi)) ; 
}

double THUNDER_ALWAYS_INLINE 
get_cell_surface_eta_3D_log(double ri, double ro, double zeta0, double eta0, double xi0, double dzeta, double dxi) {
    return (cell_surface_eta_3D_log_aux(ri,ro,zeta0,eta0,xi0) + cell_surface_eta_3D_log_aux(ri,ro,zeta0+dzeta,eta0,xi0+dxi))
    - (cell_surface_eta_3D_log_aux(ri,ro,zeta0+dzeta,eta0,xi0) + cell_surface_eta_3D_log_aux(ri,ro,zeta0,eta0,xi0+dxi)) ; 
}

double THUNDER_ALWAYS_INLINE 
get_cell_surface_zxi_3D_log(double ri, double ro, double zeta0, double eta0, double xi0, double dzeta, double deta) {
    return (cell_surface_xi_3D_log_aux(ri,ro,zeta0,eta0,xi0) + cell_surface_xi_3D_log_aux(ri,ro,zeta0+dzeta,eta0+deta,xi0))
    - (cell_surface_xi_3D_log_aux(ri,ro,zeta0,eta0+deta,xi0) + cell_surface_xi_3D_log_aux(ri,ro,zeta0+dzeta,eta0,xi0)) ; 
}
/*******************************************************************************************************************************************************/
/*******************************************************************************************************************************************************/
/*                                              NON LOGARITHMIC RADIUS DISTRIBUTION                                                                    */
/*******************************************************************************************************************************************************/
/*******************************************************************************************************************************************************/
/*                                                      ZETA=CONST FACES                                                                               */
/*******************************************************************************************************************************************************/
double THUNDER_ALWAYS_INLINE 
cell_surface_zeta_3D_ext_aux(double ri, double ro, double zeta, double eta, double xi) {
    return -(math::int_pow<2>(ri - ri*zeta + ro*zeta)*atan((2 + sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi) - 2*xi*(2 - 2*xi + sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi)))/(-1 + 2*eta))); 
}
// This needs to be integrated twice! 
double THUNDER_ALWAYS_INLINE 
cell_surface_zeta_3D_int_aux(double ri, double ro, double zeta, double eta, double xi) {
    return (4*sqrt(math::int_pow<4>(ri)*math::int_pow<3>(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi)*math::int_pow<4>(-1 + zeta) - 
       8*math::int_pow<3>(ri)*ro*(1 + (-1 + eta)*eta + (-1 + xi)*xi)*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,1.5)*
        math::int_pow<3>(-1 + zeta)*zeta + 4*math::int_pow<2>(ri)*math::int_pow<2>(ro)*(2 + (-1 + eta)*eta + (-1 + xi)*xi)*
        (3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi)*math::int_pow<2>(-1 + zeta)*math::int_pow<2>(zeta) - 
       4*ri*math::int_pow<3>(ro)*sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi)*(-1 + zeta)*math::int_pow<3>(zeta) + math::int_pow<4>(ro)*math::int_pow<4>(zeta)))
    /pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,1.5); 
}
/*******************************************************************************************************************************************************/
/*                                                      ETA=CONST FACES                                                                                */
/*******************************************************************************************************************************************************/
double THUNDER_ALWAYS_INLINE 
cell_surface_eta_3D_ext_aux(double ri, double ro, double zeta, double eta, double xi) {
    return (math::int_pow<2>(ri - ri*zeta + ro*zeta)*atan((-1 + 2*xi)/sqrt(2 + 4*(-1 + eta)*eta)))/2.; 
}
// This is already integrated in zeta
double THUNDER_ALWAYS_INLINE 
cell_surface_eta_3D_int_aux(double ri, double ro, double zeta, double eta, double xi) {
    return ((ro - ri*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5))*(ro*zeta + 
       ri*(pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5) - zeta*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5)))*
     (-(ri*ro*(-1 + zeta)*zeta*log(abs((1 + pow(1 + 4*(-1 + xi)*xi,-0.5)*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5))*
             pow(1 - pow(1 + 4*(-1 + xi)*xi,-0.5)*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5),-1)))*pow(2 + 4*(-1 + eta)*eta,0.5)) + 
       pow(2 + 4*(-1 + eta)*eta,0.5)*pow(ri,2)*pow(1 + 4*(-1 + xi)*xi,0.5)*pow(-1 + zeta,2) + 
       atan(pow(2 + 4*(-1 + eta)*eta,-0.5)*pow(1 + 4*(-1 + xi)*xi,0.5))*pow(ro,2)*pow(zeta,2))*
     pow(pow(ro - ri*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5),2)*
       pow(ro*zeta + ri*(pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5) - zeta*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5)),2),-0.5)*math::sgn(-0.5 + xi))/2. ; 
}
/*******************************************************************************************************************************************************/
/*                                                      XI=CONST FACES                                                                                 */
/*******************************************************************************************************************************************************/
double THUNDER_ALWAYS_INLINE 
cell_surface_xi_3D_ext_aux(double ri, double ro, double zeta, double eta, double xi) {
    return (math::int_pow<2>(ri - ri*zeta + ro*zeta)*atan((-1 + 2*eta)/sqrt(2 + 4*(-1 + xi)*xi)))/2.; 
}
// This is already integrated in zeta
double THUNDER_ALWAYS_INLINE 
cell_surface_xi_3D_int_aux(double ri, double ro, double zeta, double eta, double xi) {
    return (pow(1 + 2*(-1 + xi)*xi,0.5)*(ro - ri*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5))*
     (ro*zeta + ri*(pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5) - zeta*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5)))*
     (-(ri*ro*(-1 + zeta)*zeta*log(abs((1 + pow(1 + 4*(-1 + eta)*eta,-0.5)*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5))*
             pow(1 - pow(1 + 4*(-1 + eta)*eta,-0.5)*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5),-1)))*pow(2 + 4*(-1 + xi)*xi,0.5)) + 
       pow(1 + 4*(-1 + eta)*eta,0.5)*pow(ri,2)*pow(2 + 4*(-1 + xi)*xi,0.5)*pow(-1 + zeta,2) + 
       atan(pow(1 + 4*(-1 + eta)*eta,0.5)*pow(2 + 4*(-1 + xi)*xi,-0.5))*pow(ro,2)*pow(zeta,2))*
     pow((1 + 2*(-1 + xi)*xi)*pow(ro - ri*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5),2)*
       pow(ro*zeta + ri*(pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5) - zeta*pow(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi,0.5)),2),-0.5)*math::sgn(-0.5 + eta))/2.; 
}

double THUNDER_ALWAYS_INLINE 
cell_surface_zeta_3D_ext(double ri, double ro, double zeta, double eta0, double xi0, double deta, double dxi)
{
    auto const func = [&] ( double eta, double xi ) {
        return cell_surface_zeta_3D_ext_aux(ri,ro,zeta,eta,xi) ; 
    } ; 
    return utils::nd_quadrature_integrate<2,5,double>({eta0,xi0},{eta0+deta,xi0+dxi},func) ; 
}

double THUNDER_ALWAYS_INLINE 
cell_surface_eta_3D_ext(double ri, double ro, double zeta0, double eta, double xi0, double dzeta, double dxi)
{
    auto const func = [&] ( double zeta, double xi ) {
        return cell_surface_eta_3D_ext_aux(ri,ro,zeta,eta,xi) ; 
    } ; 
    return utils::eval_2d_primitive(zeta0,xi0,zeta0+dzeta,xi0+dxi,func) ; 
}

double THUNDER_ALWAYS_INLINE 
cell_surface_xi_3D_ext(double ri, double ro, double zeta0, double eta0, double xi, double dzeta, double deta)
{
    auto const func = [&] ( double zeta, double eta ) {
        return cell_surface_xi_3D_ext_aux(ri,ro,zeta,eta,xi) ; 
    } ; 
    return utils::eval_2d_primitive(zeta0,eta0,zeta0+dzeta,eta0+deta,func) ; 
}

double THUNDER_ALWAYS_INLINE 
cell_surface_zeta_3D_int(double ri, double ro, double zeta, double eta0, double xi0, double deta, double dxi)
{
    auto const func = [&] ( double eta, double xi ) {
        return cell_surface_zeta_3D_int_aux(ri,ro,zeta,eta,xi) ; 
    } ; 
    return utils::eval_2d_primitive(eta0,xi0,eta0+deta,xi0+dxi,func) ; 
}

double THUNDER_ALWAYS_INLINE 
cell_surface_eta_3D_int(double ri, double ro, double zeta0, double eta, double xi0, double dzeta, double dxi)
{
    auto const func = [&] ( double zeta, double xi ) {
        return cell_surface_eta_3D_int_aux(ri,ro,zeta,eta,xi) ; 
    } ; 
    return utils::eval_2d_primitive(zeta0,xi0,zeta0+dzeta,xi0+dxi,func) ; 
}

double THUNDER_ALWAYS_INLINE 
cell_surface_xi_3D_int(double ri, double ro, double zeta0, double eta0, double xi, double dzeta, double deta)
{
    auto const func = [&] ( double zeta, double eta ) {
        return cell_surface_xi_3D_int_aux(ri,ro,zeta,eta,xi) ; 
    } ; 
    return utils::eval_2d_primitive(zeta0,eta0,zeta0+dzeta,eta0+deta,func) ; 
}


}}

#endif 