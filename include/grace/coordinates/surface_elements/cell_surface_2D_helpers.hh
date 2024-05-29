/**
 * @file cell_surface_2D_helpers.hh
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

#ifndef GRACE_COORDINATES_SURFACE_ELEMENTS_2D_HELPERS_HH
#define GRACE_COORDINATES_SURFACE_ELEMENTS_2D_HELPERS_HH

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/integration.hh>

namespace grace { namespace detail {

double GRACE_ALWAYS_INLINE 
cell_surface_zeta_2D_int_aux_integrand(double ri, double ro, double zeta, double eta) {
    return (2*sqrt(math::int_pow<2>(1 - 2*eta)*math::int_pow<2>(ro)*math::int_pow<2>(zeta) 
    + math::int_pow<2>(math::int_pow<3>(sqrt(2 + 4*(-1 + eta)*eta))*ri*(1 - zeta) + ro*zeta)))/math::int_pow<3>(sqrt(2 + 4*(-1 + eta)*eta)) ; 
}

double GRACE_ALWAYS_INLINE 
cell_surface_zeta_2D_ext_aux(double ri, double ro, double zeta, double eta) {
    return -((ri*(1 - zeta) + ro*zeta)*atan(1 - 2*eta)) ; 
}

double GRACE_ALWAYS_INLINE 
cell_surface_eta_2D_int_aux(double ri, double ro, double zeta, double eta) {
    return zeta*fabs(-(sqrt(2 - 4*(1 - eta)*eta)*ri) + ro) ; 
}

double GRACE_ALWAYS_INLINE 
cell_surface_eta_2D_ext_aux(double ri, double ro, double zeta, double eta) {
    return zeta * ( ro - ri ) ; 
}


double GRACE_ALWAYS_INLINE 
cell_surface_zeta_2D_log_aux(double ri, double ro, double zeta, double eta) {
    return -((math::int_pow<2>(ri)*pow(ro/ri,2*zeta)*(1 - 2*eta + (2 + 4*(-1 + eta)*eta)*atan(1 - 2*eta)))/(2 + 4*(-1 + eta)*eta)) ; 
}

double GRACE_ALWAYS_INLINE 
cell_surface_eta_2D_log_aux(double ri, double ro, double zeta, double eta) {
    return (math::int_pow<2>(ri)*pow(ro/ri,2*zeta)*log(ro/ri))/2. ; 
}

/* internal */
template<size_t N>
double GRACE_ALWAYS_INLINE 
get_cell_surface_zeta_2D_int(double ri, double ro, double zeta, double eta0, double deta)
{
    auto const _integrand = [&] ( double eta ) {
        return cell_surface_zeta_2D_int_aux_integrand(ri,ro,zeta,eta) ; 
    } ; 
    return utils::gauss_legendre_quadrature<N>(eta0,eta0+deta,1e-15, _integrand) ; 
}

double GRACE_ALWAYS_INLINE 
get_cell_surface_eta_2D_int(double ri, double ro, double zeta0, double eta, double dzeta)
{
    return cell_surface_eta_2D_int_aux(ri,ro,zeta0+dzeta,eta) - cell_surface_eta_2D_int_aux(ri,ro,zeta0,eta) ; 
}

/* external */
double GRACE_ALWAYS_INLINE 
get_cell_surface_zeta_2D_ext(double ri, double ro, double zeta, double eta0, double deta)
{
    return cell_surface_zeta_2D_ext_aux(ri,ro,zeta,eta0+deta) - cell_surface_zeta_2D_ext_aux(ri,ro,zeta,eta0) ; 
}


double GRACE_ALWAYS_INLINE 
get_cell_surface_eta_2D_ext(double ri, double ro, double zeta0, double eta, double dzeta)
{
    return cell_surface_eta_2D_ext_aux(ri,ro,zeta0+dzeta,eta) - cell_surface_eta_2D_ext_aux(ri,ro,zeta0,eta) ; 
}

/* Logarithmic */
double GRACE_ALWAYS_INLINE 
get_cell_surface_zeta_2D_log(double ri, double ro, double zeta, double eta0, double deta)
{
    return cell_surface_zeta_2D_log_aux(ri,ro,zeta,eta0+deta) - cell_surface_zeta_2D_log_aux(ri,ro,zeta,eta0) ; 
}

double GRACE_ALWAYS_INLINE 
get_cell_surface_eta_2D_log(double ri, double ro, double zeta0, double eta, double dzeta)
{
    return cell_surface_eta_2D_log_aux(ri,ro,zeta0+dzeta,eta) - cell_surface_eta_2D_log_aux(ri,ro,zeta0,eta) ; 
}


}}

#endif 