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

enum class plane_axis { XY, XZ, YZ } ; 

/**
 * @brief Plane descriptor in the form 
 * \f[
 * \mathbf{n} \cdot (\mathbf{x} - \mathbf{d}) = \mathbf{0} \, .
 * \f]
 * Where n[dir] = 1 and other components are 0 
 */
struct plane_desc_t {
    std::string name ; 
    plane_axis axis ;
    double coord ; 
} ;




inline static bool
quadrant_intersects_plane(quadrant_t quad, p4est_topidx_t itree,
                           const plane_desc_t& plane)
{
    auto cl = detail::get_quad_coord_lbounds(quad, itree);
    double dx = amr::get_tree_spacing(itree)[0] * (1.0 / (1 << quad.level()));

    double lo, hi;
    switch (plane.axis) {
        case plane_axis::XY: lo = cl[2]; hi = cl[2] + dx; break;
        case plane_axis::XZ: lo = cl[1]; hi = cl[1] + dx; break;
        case plane_axis::YZ: lo = cl[0]; hi = cl[0] + dx; break;
    }

    // Half-open interval: [lo, hi)
    // Guarantees exactly one quadrant owns any given plane position.
    return (plane.coord >= lo) && (plane.coord < hi);
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
        size_t ncells,
        size_t nq) 
    : _nq(nq), _ncells(ncells), _plane(plane) {
        _ngz = amr::get_n_ghosts() ; 
    }

    void slice() ;

    size_t n_sliced_quads() const { return sliced_quads.size() ; }

    size_t _nq, _ncells, _ngz ;                         //!< 
    plane_desc_t _plane ;                      //!< The plane that is used to slice 
    std::vector<size_t> sliced_quads ;         //!< Local quad-ids of sliced quads 
    std::vector<size_t > sliced_cell_offsets ; //!< Map quad_id -> offset from ngz 
    
    size_t ncells, glob_nq, glob_ncells, local_quad_offset ; //!< Filled during output for convenience 

    private:
    void search() ; 
    void find_cells() ; 

} ; 
} // namespace amr
} // namespace grace
#endif
