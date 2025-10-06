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

#include <grace_config.h>
#include <grace/utils/device.hh>
#include <grace/utils/inline.hh>

#include <grace/IO/spherical_surfaces.hh>
#include <grace/IO/sphere_interpolator.hh>

namespace grace { namespace amr{

bool is_inside(
    std::array<double,3> const& point, 
    cube_desc_t const& cube
)
{
    return ( point[0] >= cube.v[0][0] and point[0] < cube.v[1][0] )
       and ( point[1] >= cube.v[0][1] and point[1] < cube.v[2][1] )
       and ( point[2] >= cube.v[0][2] and point[2] < cube.v[4][2] ) ; 

}

int grace_search_plane(
    p4est_t* forest,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant, 
    p4est_locidx_t local_num,
    void* point
)
{
    // Fetch data 
    auto p = reinterpret_cast<sphere_point_t*>(point) ;
    // now construct a cube from the quadrant 
    auto cube = make_cube(quadrant_t{quadrant}, which_tree) ; 
    bool inside = is_inside(p->coords, cube) ;  
    if ( inside and local_num >= 0 ) {
        p->found = true ;
        p->qid = // maybe trees(itree).data() - quadrant? (?)

    }
}

void sphere_surface_interpolator_t::find_intersection() {
    //step 1 construct an array of point_t 
    std::vector<sphere_point_t> _buf(sphere.npoints) ;
    //...
    //p4est search, stores qid in points
    // ... 
    // construct device view with points that 
    // actually intersect the grid
    // ... 
    //loop over cells store back in ponints
}

}}
