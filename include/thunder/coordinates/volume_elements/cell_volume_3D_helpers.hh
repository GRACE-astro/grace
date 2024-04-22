/**
 * @file cell_volume_3d_helpers.hh
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

#ifndef THUNDER_COORDINATES_VOLUME_ELEMENTS_3D_HELPERS_HH 
#define THUNDER_COORDINATES_VOLUME_ELEMENTS_3D_HELPERS_HH

#include <thunder_config.h>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/utils/integration.hh>
namespace thunder {
namespace detail {
double THUNDER_ALWAYS_INLINE
cell_vol_3D_numerical_term_prefactor(double si, double so, double ri, double ro, double zeta, double eta)
{
  return -2*(ri*(-1 + si)*(-1 + zeta) - ro*(-1 + so)*zeta)*math::int_pow<2>(ri*si*(-1 + zeta) - ro*so*zeta); 
}

double THUNDER_ALWAYS_INLINE
cell_vol_3D_numerical_term_aux(double si, double so, double ri, double ro, double zeta, double eta, double xi)
{
    return atan((1 - 2*eta + sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi))/sqrt(2 + 4*(-1 + xi)*xi))/sqrt(0.5 + (-1 + xi)*xi) ; 
}

template<size_t N>
double THUNDER_ALWAYS_INLINE
cell_vol_3D_numerical(double si, double so, double ri, double ro, double zeta, double eta, double xi0, double dxi)
{
  auto const term = [&] (double const _xi) {
    return cell_vol_3D_numerical_term_aux(si,so,ri,ro,zeta,eta,_xi) ; 
  } ;
  
  return cell_vol_3D_numerical_term_prefactor(si,so,ri,ro,zeta,eta) * utils::gauss_legendre_quadrature<N>(xi0,xi0+dxi,1e-16,term) ; 
}

double THUNDER_ALWAYS_INLINE 
cell_vol_3D_analytic(double si, double so, double ri, double ro, double zeta, double eta, double xi)
{ 
    double const sqrt_term = sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi) ;
    double const int1 = (4*eta*xi*math::int_pow<3>(ri*(-1 + si)*(-1 + zeta) - ro*(-1 + so)*zeta) + math::int_pow<3>(ri*si*(-1 + zeta) - ro*so*zeta)*
	  atan((2 + 4*math::int_pow<2>(xi) + sqrt_term - 2*xi*(2 + sqrt_term))/(-1 + 2*eta+1e-20)))/3. ;  
    double const int2 =  -0.25*(math::int_pow<2>(ri*(-1 + si)*(-1 + zeta) - ro*(-1 + so)*zeta)*(ri*si*(-1 + zeta) - ro*so*zeta)*
		  ( 8*xi
		    + 4*atan(1 - 2*xi)
		    - 4*atan((-1 + xi*(1 - 2*xi + sqrt_term - 2*eta*(-2 + 2*eta + sqrt_term)))/(-2 + 4*math::int_pow<2>(eta)*(-1 + xi) + sqrt_term + 2*eta*(-1 + xi)*(-2 + sqrt_term) - xi*(-3 + 2*xi + sqrt_term)))
		    + 2*log(1 + 2*(-1 + xi)*xi)
		    + (-4 + 8*eta)*log(-1 + 2*xi + sqrt_term)
		    - 8*xi*log(1 - 2*eta + sqrt_term)
		    -2*log((128*(1 + 2*(-1 + eta)*eta)*(2 + 2*(-1 + xi)*xi - sqrt_term + 2*eta*(-2 + 2*eta + sqrt_term)))/(math::int_pow<4>(1 - 2*eta+1e-5)*(1 + 2*(-1 + xi)*xi))))) ;
    return int1 + int2 ;    
}

// TODO: check if this simplifies when si,so are fixed
template< size_t N >
double THUNDER_ALWAYS_INLINE
get_cell_volume_3D(double si, double so, double ri, double ro, double z0, double e0, double x0, double dz, double de, double dx) {

  double const dV2 = cell_vol_3D_numerical<N>(si,so,ri,ro,z0,e0,x0,dx) + cell_vol_3D_numerical<N>(si,so,ri,ro,z0+dz,e0+de,x0,dx)
    - (cell_vol_3D_numerical<N>(si,so,ri,ro,z0+dz,e0,x0,dx) + cell_vol_3D_numerical<N>(si,so,ri,ro,z0,e0+de,x0,dx)) ;

  double const dV1 = ( cell_vol_3D_analytic(si,so,ri,ro,z0,e0,x0+dx) + cell_vol_3D_analytic(si,so,ri,ro,z0+dz,e0+de,x0+dx)
		       - (cell_vol_3D_analytic(si,so,ri,ro,z0+dz,e0,x0+dx) + cell_vol_3D_analytic(si,so,ri,ro,z0,e0+de,x0+dx)) )
    -( cell_vol_3D_analytic(si,so,ri,ro,z0,e0,x0) + cell_vol_3D_analytic(si,so,ri,ro,z0+dz,e0+de,x0)
       - (cell_vol_3D_analytic(si,so,ri,ro,z0+dz,e0,x0) + cell_vol_3D_analytic(si,so,ri,ro,z0,e0+de,x0))) ;
  return dV1 + dV2 ; 
}


double THUNDER_ALWAYS_INLINE 
get_cell_volume_3D_log_aux(double ri, double ro, double zeta, double eta, double xi) {
    double const sqrt_term = sqrt(3 + 4*(-1 + eta)*eta + 4*(-1 + xi)*xi) ;
    return (pow(ro/ri,-1.5 + 3*zeta)*pow(ri*ro,1.5)*
     atan((2 + sqrt_term - 
         2*xi*(2 - 2*xi + sqrt_term))/(-1 + 2*eta))*log(ro/ri))/(3.*(log(ri) - log(ro))) ; 
}

double THUNDER_ALWAYS_INLINE 
get_cell_volume_3D_log(double ri, double ro, double zeta0, double eta0, double xi0, double dzeta, double deta, double dxi) {
    return (get_cell_volume_3D_log_aux(ri,ro,zeta0,eta0,xi0+dxi)+get_cell_volume_3D_log_aux(ri,ro,zeta0+dzeta,eta0+deta,xi0+dxi)-(get_cell_volume_3D_log_aux(ri,ro,zeta0+dzeta,eta0,xi0+dxi)+get_cell_volume_3D_log_aux(ri,ro,zeta0,eta0+deta,xi0+dxi)))
    - (get_cell_volume_3D_log_aux(ri,ro,zeta0,eta0,xi0)+get_cell_volume_3D_log_aux(ri,ro,zeta0+dzeta,eta0+deta,xi0)-(get_cell_volume_3D_log_aux(ri,ro,zeta0+dzeta,eta0,xi0)+get_cell_volume_3D_log_aux(ri,ro,zeta0,eta0+deta,xi0))) ; 
}

} } 

#endif 