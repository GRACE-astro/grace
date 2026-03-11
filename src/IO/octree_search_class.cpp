// octree_search_class.cc
#include <grace/IO/octree_search_class.hh>
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
namespace grace { namespace amr {
#ifdef GRACE_3D


int grace_search_plane(
    p4est_t* forest,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant, 
    p4est_locidx_t local_num,
    void* point
) 
{
    ASSERT(point, "Nullptr") ; 
    // get the plane we are checking 
    auto plane = static_cast<plane_desc_t*>(point) ; 
    // finally check for intersection 
    quadrant_t quad{quadrant} ; 
    bool intersect = quadrant_intersects_plane(quad, which_tree, *plane) ;
    // if the quadrant is a leaf we write back 
    // to its user_int to flag it 
    if ( local_num >= 0 and intersect ) {
        auto quadlist = static_cast<std::vector<size_t>*>(forest->user_pointer) ; 
        quadlist->push_back( local_num /*+ amr::get_local_quadrants_offset(which_tree)*/ ) ; 
    }
    return intersect ; 
}


void oct_tree_plane_slicer_t::search() {
    std::vector<plane_desc_t> _buf{_plane} ; 
    auto plane_arr = sc_array_new_data(
        _buf.data(),sizeof(plane_desc_t), 1
    ) ; 
    // get forest ptr 
    p4est_t * p4est = grace::amr::forest::get().get(); 
    p4est->user_pointer = reinterpret_cast<void*>(&(sliced_quads)) ;
    // search 
    p4est_search_local(
        p4est, 
        false, 
        nullptr, 
        &grace_search_plane,
        plane_arr
    ) ; 
}

void oct_tree_plane_slicer_t::find_cells() {
    sliced_cell_offsets.clear() ; 
    sliced_cell_offsets.reserve(sliced_quads.size()) ; 
    auto plane_axis = _plane.axis ; 
    int plane_dir ; 
    if ( plane_axis == amr::plane_axis::XY ) {
        plane_dir = 2; 
    } else if ( plane_axis == amr::plane_axis::YZ ) {
        plane_dir = 0; 
    } else {
        plane_dir = 1; 
    }
    for ( auto const& iq: sliced_quads ) {
        auto const idx = detail::get_inv_cell_spacing(iq, plane_dir);
        auto const qc = detail::get_quad_coord_lbounds(iq) ; 

        double const local = (_plane.coord - qc[plane_dir]) * idx;

        size_t const offset = static_cast<int>(
            std::clamp(std::floor(local), 0.0, static_cast<double>(_ncells - 1))
        );
        sliced_cell_offsets.push_back(offset) ; 
        
    }
}

void oct_tree_plane_slicer_t::slice() {
    search() ; 
    find_cells() ;
}

#endif // GRACE_3D
    

} // namespace amr
} // namespace grace
