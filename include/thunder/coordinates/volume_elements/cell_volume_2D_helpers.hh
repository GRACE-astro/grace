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

#ifndef THUNDER_COORDINATES_VOLUME_ELEMENTS_2D_HELPERS_HH 
#define THUNDER_COORDINATES_VOLUME_ELEMENTS_2D_HELPERS_HH

#include <thunder_config.h>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/utils/integration.hh>


namespace thunder {

namespace detail {

double THUNDER_ALWAYS_INLINE 
cell_volume_2D_aux(double si, double so, double ri, double ro, double zeta, double eta) {
    return eta*math::int_pow<2>(ri)*math::int_pow<2>(-1 + si)*(-2 + zeta)*zeta + ri*(-1 + si)*zeta*(ro + ri*si*(-2 + zeta) - ro*zeta)*asinh(1 - 2*eta) - 
   ((ro - ri*si)*zeta*(-(ri*si*(-2 + zeta)) + ro*zeta)*atan(1 - 2*eta))/2. ;
}

double THUNDER_ALWAYS_INLINE 
cell_volume_2D_log_aux(double ri, double ro, double zeta, double eta) {
    return -0.5*(math::int_pow<2>(ri)*pow(ro/ri,2*zeta)*atan(1 - 2*eta)) ;
}



double THUNDER_ALWAYS_INLINE 
get_cell_volume_2D_ext(double ri, double ro, double zeta0, double eta0, double dzeta, double deta) {
    auto const _integrand = [&] (double zeta, double eta) {
        return cell_volume_2D_aux(1.,1.,ri,ro,zeta,eta) ; 
    } ; 
    return utils::eval_2d_primitive(zeta0,eta0,zeta0+dzeta,eta0+deta, _integrand) ;  
}

double THUNDER_ALWAYS_INLINE 
get_cell_volume_2D_int(double ri, double ro, double zeta0, double eta0, double dzeta, double deta) {
    auto const _integrand = [&] (double zeta, double eta) {
        return cell_volume_2D_aux(0.,1.,ri,ro,zeta,eta) ; 
    } ; 
    return utils::eval_2d_primitive(zeta0,eta0,zeta0+dzeta,eta0+deta, _integrand) ;  
}

double THUNDER_ALWAYS_INLINE 
get_cell_volume_2D_log(double ri, double ro, double zeta0, double eta0, double dzeta, double deta) {
    auto const _integrand = [&] (double zeta, double eta) {
        return cell_volume_2D_log_aux(ri,ro,zeta,eta) ; 
    } ; 
    return utils::eval_2d_primitive(zeta0,eta0,zeta0+dzeta,eta0+deta, _integrand) ;  
}

} }

#endif /* THUNDER_COORDINATES_VOLUME_ELEMENTS_2D_HELPERS_HH */