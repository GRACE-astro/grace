/**
 * @file interpolate_on_spheres.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-10-03
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
#ifndef GRACE_IO_INTERP_ON_SPHERES_HH
#define GRACE_IO_INTERP_ON_SPHERES_HH

#include <grace_config.h>
#include <grace/utils/device.hh>
#include <grace/utils/inline.hh>

#include <grace/IO/spherical_surfaces.hh>

namespace grace { namespace amr {

namespace detail {

/**
 * @brief Sphere descriptor in the form 
 * \f[
 * (\mathbf{x}-\mathbf{c})^2 - r^2 = 0 \, .
 * \f]
 */
struct sphere_desc_t {
    std::array<double,3> c ; //!< 0,1,2 for x y z 
    double r ; //!< x,y,z offsets 
} ;

struct sphere_point_t {
    std::array<double,3> coords; 
    size_t qid, i,j,k;
    bool found
} ; 

}

struct sphere_surface_interpolator_t {


    sphere_surface_interpolator_t(
        spherical_surface_iface& _sphere
    ) : sphere(_sphere) {} 

    void find_intersection() ; 

    void interpolate() ; 

    private:
    
    using point_arr_t = sc_array_t<sphere_point_t> ; 

    spherical_surface_iface& sphere ; 

    std::vector<size_t> quads ; 
    std::vector<std::array<size_t,3>> cells ;  

    size_t ncells, glob_nq, glob_ncells, local_quad_offset ; //!< Filled during output for convenience  

} ; 

}}

#endif /* GRACE_IO_INTERP_ON_SPHERES_HH */