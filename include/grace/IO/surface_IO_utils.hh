/**
 * @file surface_IO_utils.hh
 * @author Carlo Musolino (musolino@aei.mpg.de)
 * @brief 
 * @date 2025-10-03
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#ifndef GRACE_IO_UTILS_LOWLEVEL_HH
#define GRACE_IO_UTILS_LOWLEVEL_HH

#include <grace_config.h>


#include <grace/amr/grace_amr.hh>
#include <grace/utils/grace_utils.hh>




namespace grace { 
    
/**
 * @brief Cube descriptor in the form of its vertex coordinates
 */
struct cube_desc_t {
    std::array<std::array<double,3>,8> v ; 
} ; 

using intersected_cell_descriptor_t = std::pair<size_t,std::tuple<size_t,size_t,size_t,size_t>>;

namespace amr { namespace detail {

inline static std::array<double, 3>
get_quad_coord_lbounds(quadrant_t quad, size_t itree) {
    auto const dx_quad  = 1./(1<<quad.level()) ; 
    auto const qcoords = quad.qcoords();
    auto quad_coords = std::array<double, 3>{
        qcoords[0] * dx_quad,
        qcoords[1] * dx_quad,
        qcoords[2] * dx_quad
    };

    auto tree_coords = amr::get_tree_vertex(itree, 0) ; 
    auto dx_tree = amr::get_tree_spacing(itree) ; 

    return std::array<double, 3>{
                quad_coords[0] * dx_tree[0] + tree_coords[0],
                quad_coords[1] * dx_tree[1] + tree_coords[1],
                quad_coords[2] * dx_tree[2] + tree_coords[2]
            };
}

inline static std::array<double, 3>
get_quad_coord_lbounds(size_t iquad) {
    auto itree = amr::get_quadrant_owner(iquad) ; 
    auto quad = amr::get_quadrant(itree, iquad) ; 
    return get_quad_coord_lbounds(quad, itree) ; 
}

inline static cube_desc_t make_cube(quadrant_t quad, p4est_topidx_t itree) 
{
    
    auto cl = get_quad_coord_lbounds(quad, itree) ; 

    auto dx = amr::get_tree_spacing(itree)[0] * 1./(1<<quad.level()) ;

    cube_desc_t cube{} ; 
    cube.v[0] = std::array<double,3>(
        {cl[0], cl[1], cl[2]}
    ) ; 
    cube.v[1] = std::array<double,3>(
        {cl[0] + dx, cl[1], cl[2]}
    ) ; 
    cube.v[2] = std::array<double,3>(
        {cl[0], cl[1] + dx, cl[2]}
    ) ;
    cube.v[3] = std::array<double,3>(
        {cl[0] + dx, cl[1] + dx, cl[2]}
    ) ;

    cube.v[4] = std::array<double,3>(
        {cl[0], cl[1], cl[2]+dx}
    ) ; 
    cube.v[5] = std::array<double,3>(
        {cl[0] + dx, cl[1], cl[2]+dx}
    ) ; 
    cube.v[6] = std::array<double,3>(
        {cl[0], cl[1] + dx, cl[2]+dx}
    ) ;
    cube.v[7] = std::array<double,3>(
        {cl[0] + dx, cl[1] + dx, cl[2]+dx}
    ) ;

    return cube ; 
}


inline static double get_inv_cell_spacing(size_t iquad, int8_t idir) {
    auto itree = amr::get_quadrant_owner(iquad) ; 
    auto quad = amr::get_quadrant(itree, iquad) ; 
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto dx_tree = amr::get_tree_spacing(itree) ; 
    auto idx_quad = (1<<quad.level())/dx_tree[0] ; 
    return idx_quad * nx ; 
}



}}} /* namespace grace::amr::detail */ 

#endif /* GRACE_IO_UTILS_LOWLEVEL_HH */