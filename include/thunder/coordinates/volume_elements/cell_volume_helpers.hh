/**
 * @file cell_volume_2D_helpers.hh
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


#include <thunder_config.h>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/utils/integration.hh>

#include <thunder/coordinates/volume_elements/cell_volume_3D_helpers.hh>
#include <thunder/coordinates/volume_elements/cell_volume_2D_helpers.hh>

#ifndef THUNDER_COORDINATES_VOLUME_ELEMENTS_HELPERS_HH 
#define THUNDER_COORDINATES_VOLUME_ELEMENTS_HELPERS_HH

namespace thunder { namespace detail {

template < size_t N >
double THUNDER_ALWAYS_INLINE 
get_cell_volume_int(double ri, double ro, VEC(double zeta0, double eta0, double xi0), VEC(double dzeta, double deta, double dxi)) {
    #ifdef THUNDER_3D 
    return get_cell_volume_3D_int<N>(ri,ro,zeta0,eta0,xi0,dzeta,deta,dxi) ; 
    #else 
    return get_cell_volume_2D_int(ri,ro,zeta0,eta0,dzeta,deta) ; 
    #endif 
}

double THUNDER_ALWAYS_INLINE 
get_cell_volume_ext(double ri, double ro, VEC(double zeta0, double eta0, double xi0), VEC(double dzeta, double deta, double dxi)) {
    #ifdef THUNDER_3D 
    return get_cell_volume_3D_ext(ri,ro,zeta0,eta0,xi0,dzeta,deta,dxi) ; 
    #else 
    return get_cell_volume_2D_ext(ri,ro,zeta0,eta0,dzeta,deta) ; 
    #endif 
}

double THUNDER_ALWAYS_INLINE 
get_cell_volume_log(double ri, double ro, VEC(double zeta0, double eta0, double xi0), VEC(double dzeta, double deta, double dxi)) {
    #ifdef THUNDER_3D 
    return get_cell_volume_3D_log(ri,ro,zeta0,eta0,xi0,dzeta,deta,dxi) ; 
    #else 
    return get_cell_volume_2D_log(ri,ro,zeta0,eta0,dzeta,deta) ; 
    #endif 
}

} } 

#endif 