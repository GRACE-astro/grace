/**
 * @file line_element_helpers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-04-17
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
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

#ifndef GRACE_COORDINATES_LINE_ELEMENTS_HELPERS_HH
#define GRACE_COORDINATES_LINE_ELEMENTS_HELPERS_HH

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/integration.hh>

namespace grace { namespace detail {

/* Logarithmic */
double GRACE_ALWAYS_INLINE 
line_element_zeta_log_aux(double ri, double ro, double zeta, double eta, double xi)
{
    return sqrt(math::int_pow<2>(ri)*pow(log(ro/ri),2)*pow(ro/ri,2*zeta)) ; 
}

double GRACE_ALWAYS_INLINE 
line_element_eta_log_aux(double ri, double ro, double zeta, double eta, double xi)
{
    return ri*atan((-1 + 2*eta)/sqrt(2 + 4*(-1 + xi)*xi))*pow(ro/ri,zeta);
}

double GRACE_ALWAYS_INLINE 
line_element_xi_log_aux(double ri, double ro, double zeta, double eta, double xi)
{
    return ri*atan((-1 + 2*xi)/sqrt(2 + 4*(-1 + eta)*eta))*pow(ro/ri,zeta);
}
/* External */
double GRACE_ALWAYS_INLINE 
line_element_eta_ext_aux(double ri, double ro, double zeta, double eta, double xi)
{
    return (ri - ri*zeta + ro*zeta)*atan((-1 + 2*eta)/sqrt(2 + 4*(-1 + xi)*xi));
}

double GRACE_ALWAYS_INLINE 
line_element_xi_ext_aux(double ri, double ro, double zeta, double eta, double xi)
{
    return (ri - ri*zeta + ro*zeta)*atan((-1 + 2*xi)/sqrt(2 + 4*(-1 + eta)*eta));
}
/* Internal */

double GRACE_ALWAYS_INLINE 
line_element_zeta_aux(double si, double so, double ri, double ro, double zeta, double eta, double xi)
{
    return zeta*sqrt(math::int_pow<2>(ro + ri*si*(-1 + sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi)) - ri*sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi)));
}
/* All of these need numerical integration ! */
double GRACE_ALWAYS_INLINE 
line_element_eta_int_aux(double ri, double ro, double zeta, double eta, double xi)
{
    return 2*sqrt((math::int_pow<2>(1 - 2*eta)*math::int_pow<2>(ro)*math::int_pow<2>(zeta) + math::int_pow<2>(1 - 2*eta)*math::int_pow<2>(ro)*math::int_pow<2>(1 - 2*xi)*math::int_pow<2>(zeta) + 
       math::int_pow<2>(-2*ro*(1 + 2*(-1 + xi)*xi)*zeta + ri*(-1 + zeta)*math::int_pow<3>(sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi))))/math::int_pow<3>(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi));
}

double GRACE_ALWAYS_INLINE 
line_element_xi_int_aux(double ri, double ro, double zeta, double eta, double xi)
{
    return 2*sqrt(math::int_pow<2>(ro)*math::int_pow<2>(1 - 2*xi)*math::int_pow<2>(zeta) + math::int_pow<2>(1 - 2*eta)*math::int_pow<2>(ro)*math::int_pow<2>(1 - 2*xi)*math::int_pow<2>(zeta) + 
     math::int_pow<2>(-2*(1 + 2*(-1 + eta)*eta)*ro*zeta + ri*(-1 + zeta)*math::int_pow<3>(sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi))))/math::int_pow<3>(sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi)); 
}

/* Zeta line elements */
double GRACE_ALWAYS_INLINE 
get_line_element_zeta_int(double ri, double ro, double zeta0, double eta, double xi, double dzeta) {
    return line_element_zeta_aux(0.,1.,ri,ro,zeta0+dzeta,eta,xi) - line_element_zeta_aux(0.,1.,ri,ro,zeta0,eta,xi) ; 
}

double GRACE_ALWAYS_INLINE 
get_line_element_zeta_ext(double ri, double ro, double zeta0, double eta, double xi, double dzeta) {
    return line_element_zeta_aux(1.,1.,ri,ro,zeta0+dzeta,eta,xi) - line_element_zeta_aux(1.,1.,ri,ro,zeta0,eta,xi) ; 
}

double GRACE_ALWAYS_INLINE 
get_line_element_zeta_log(double ri, double ro, double zeta0, double eta, double xi, double dzeta) {
    return line_element_zeta_log_aux(ri,ro,zeta0+dzeta,eta,xi) - line_element_zeta_log_aux(ri,ro,zeta0,eta,xi) ; 
}
/* Eta line elements */
template< size_t N >
double GRACE_ALWAYS_INLINE 
get_line_element_eta_int(double ri, double ro, double zeta, double eta0, double xi, double deta) {
    auto const _integrand = [&] (double eta) {
        return line_element_eta_int_aux(ri,ro,zeta,eta,xi) ; 
    } ; 
    return utils::gauss_legendre_quadrature<N>(eta0,eta0+deta,1e-15,_integrand) ; 
}

double GRACE_ALWAYS_INLINE 
get_line_element_eta_ext(double ri, double ro, double zeta, double eta0, double xi, double deta) {
    return line_element_eta_ext_aux(ri,ro,zeta,eta0+deta,xi) - line_element_eta_ext_aux(ri,ro,zeta,eta0,xi) ; 
}

double GRACE_ALWAYS_INLINE 
get_line_element_eta_log(double ri, double ro, double zeta, double eta0, double xi, double deta) {
    return line_element_eta_log_aux(ri,ro,zeta,eta0+deta,xi) - line_element_eta_log_aux(ri,ro,zeta,eta0,xi) ; 
}
/* Xi line elements */
template< size_t N >
double GRACE_ALWAYS_INLINE 
get_line_element_xi_int(double ri, double ro, double zeta, double eta, double xi0, double dxi) {
    auto const _integrand = [&] (double xi) {
        return line_element_xi_int_aux(ri,ro,zeta,eta,xi) ; 
    } ; 
    return utils::gauss_legendre_quadrature<N>(xi0,xi0+dxi,1e-15,_integrand) ; 
}

double GRACE_ALWAYS_INLINE 
get_line_element_xi_ext(double ri, double ro, double zeta, double eta, double xi0, double dxi) {
    return line_element_xi_ext_aux(ri,ro,zeta,eta,xi0+dxi) - line_element_xi_ext_aux(ri,ro,zeta,eta,xi0) ; 
}

double GRACE_ALWAYS_INLINE 
get_line_element_xi_log(double ri, double ro, double zeta, double eta, double xi0, double dxi) {
    return line_element_xi_log_aux(ri,ro,zeta,eta,xi0+dxi) - line_element_xi_log_aux(ri,ro,zeta,eta,xi0) ;  
}

}}
#endif 