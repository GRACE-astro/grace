/**
 * @file octree_search_class.hh
 * @author  Keneth Miler (miler@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-07-08
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
#ifndef GRACE_AMR_OCTREE_SEARCH_CLASS_HH
#define GRACE_AMR_OCTREE_SEARCH_CLASS_HH

#include <Kokkos_Core.hpp>
#include <grace_config.h>
#include <grace/IO/hdf5_output.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/quadrant.hh>
#include <grace/amr/forest.hh>

#include "surface_IO_utils.hh"

namespace grace { namespace amr {

/**
 * @brief Plane descriptor in the form 
 * \f[
 * \mathbf{n} \cdot (\mathbf{x} - \mathbf{d}) = \mathbf{0} \, .
 * \f]
 * Where n[dir] = 1 and other components are 0 
 */
struct plane_desc_t {
    std::string name ; 
    double dir ; //!< 0,1,2 for x y z 
    std::array<double,3> d ;  //!< x,y,z offsets 
} ;




static inline bool intersects(plane_desc_t const& plane, cube_desc_t const& cube) {
    bool pos = false, neg = false;
    double const dx = cube.v[1][0] - cube.v[0][0] ; 
    #pragma unroll
    for( int i=0; i<8; ++i) {
        double f = (cube.v[i][plane.dir] - plane.d[plane.dir]) ; 
        if(f >= 0) pos = true;
        if(f < 0) neg = true;
    }
    return pos && neg ; 
}

int grace_search_plane(
    p4est_t* forest,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant, 
    p4est_locidx_t local_num,
    void* point
) ; 


struct oct_tree_plane_slicer_t {

    oct_tree_plane_slicer_t(
        plane_desc_t const& plane,
        size_t nq) 
    : _nq(nq), _plane(plane) {}

    void slice() ;

    size_t n_sliced_quads() const { return sliced_quads.size() ; }

    size_t _nq ; 
    plane_desc_t _plane ; //!< The plane that is used to slice 
    std::vector<size_t> sliced_quads ; //!< Local quad-ids of sliced quads 
    std::vector<size_t > sliced_cell_offsets ; //!< Map quad_id -> offset from ngz 
    
    size_t ncells, glob_nq, glob_ncells, local_quad_offset ; //!< Filled during output for convenience 

    private:
    void search() ; 
    void find_cells() ; 
    void reset_quads() {
        // reset quadrant status 
        for( size_t iq=0UL; iq<_nq; iq+=1UL) {
            auto quad = amr::get_quadrant(iq) ; 
            quad.set_user_int(0) ; 
        }
    }

} ; 
} // namespace amr
} // namespace grace
#endif
