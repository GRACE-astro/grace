/**
 * @file cell_surface_helpers.hh
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

#ifndef GRACE_COORDINATES_SURFACE_ELEMENTS_HELPERS_HH
#define GRACE_COORDINATES_SURFACE_ELEMENTS_HELPERS_HH

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/integration.hh>

#include <grace/coordinates/surface_elements/cell_surface_2D_helpers.hh>
#include <grace/coordinates/surface_elements/cell_surface_3D_helpers.hh>

namespace grace { namespace detail {

template< size_t N >
double GRACE_ALWAYS_INLINE 
get_cell_surface_zeta_int(double ri, double ro, VEC(double zeta, double eta0, double xi0), VECD(double deta, double dxi))
{
    #ifdef GRACE_3D 
    return get_cell_surface_zeta_3D_int<N>(ri,ro,zeta,eta0,xi0,deta,dxi) ; 
    #else 
    return get_cell_surface_zeta_2D_int<N>(ri,ro,zeta,eta0,deta) ; 
    #endif 
}

double GRACE_ALWAYS_INLINE 
get_cell_surface_eta_int(double ri, double ro, VEC(double zeta0, double eta, double xi0), VECD(double dzeta, double dxi))
{
    #ifdef GRACE_3D 
    return get_cell_surface_eta_3D_int(ri,ro,zeta0,eta,xi0,dzeta,dxi) ; 
    #else 
    return get_cell_surface_eta_2D_int(ri,ro,zeta0,eta,dzeta) ; 
    #endif 
}

double GRACE_ALWAYS_INLINE 
get_cell_surface_xi_int(double ri, double ro, VEC(double zeta0, double eta0, double xi), VECD(double dzeta, double deta))
{
    #ifdef GRACE_3D 
    return get_cell_surface_xi_3D_int(ri,ro,zeta0,eta0,xi,dzeta,deta) ; 
    #else 
    return -1. ;
    #endif 
}

/* External */
double GRACE_ALWAYS_INLINE 
get_cell_surface_zeta_ext(double ri, double ro, VEC(double zeta, double eta0, double xi0), VECD(double deta, double dxi))
{
    #ifdef GRACE_3D 
    return get_cell_surface_zeta_3D_ext(ri,ro,zeta,eta0,xi0,deta,dxi) ; 
    #else 
    return get_cell_surface_zeta_2D_ext(ri,ro,zeta,eta0,deta) ; 
    #endif 
}

double GRACE_ALWAYS_INLINE 
get_cell_surface_eta_ext(double ri, double ro, VEC(double zeta0, double eta, double xi0), VECD(double dzeta, double dxi))
{
    #ifdef GRACE_3D 
    return get_cell_surface_eta_3D_ext(ri,ro,zeta0,eta,xi0,dzeta,dxi) ; 
    #else 
    return get_cell_surface_eta_2D_ext(ri,ro,zeta0,eta,dzeta) ; 
    #endif 
}

double GRACE_ALWAYS_INLINE 
get_cell_surface_xi_ext(double ri, double ro, VEC(double zeta0, double eta0, double xi), VECD(double dzeta, double deta))
{
    #ifdef GRACE_3D 
    return get_cell_surface_xi_3D_ext(ri,ro,zeta0,eta0,xi,dzeta,deta) ; 
    #else 
    return -1. ;
    #endif 
}
/* Logarithmic */
double GRACE_ALWAYS_INLINE 
get_cell_surface_zeta_log(double ri, double ro, VEC(double zeta, double eta0, double xi0), VECD(double deta, double dxi))
{
    #ifdef GRACE_3D 
    return get_cell_surface_zeta_3D_log(ri,ro,zeta,eta0,xi0,deta,dxi) ; 
    #else 
    return get_cell_surface_zeta_2D_log(ri,ro,zeta,eta0,deta) ; 
    #endif 
}

double GRACE_ALWAYS_INLINE 
get_cell_surface_eta_log(double ri, double ro, VEC(double zeta0, double eta, double xi0), VECD(double dzeta, double dxi))
{
    #ifdef GRACE_3D 
    return get_cell_surface_eta_3D_log(ri,ro,zeta0,eta,xi0,dzeta,dxi) ; 
    #else 
    return get_cell_surface_eta_2D_log(ri,ro,zeta0,eta,dzeta) ; 
    #endif 
}

double GRACE_ALWAYS_INLINE 
get_cell_surface_xi_log(double ri, double ro, VEC(double zeta0, double eta0, double xi), VECD(double dzeta, double deta))
{
    #ifdef GRACE_3D 
    return get_cell_surface_xi_3D_log(ri,ro,zeta0,eta0,xi,dzeta,deta) ; 
    #else 
    return -1. ;
    #endif 
}
}}
#endif 