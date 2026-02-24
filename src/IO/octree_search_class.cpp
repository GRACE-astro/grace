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
    // now construct a cube from the quadrant 
    auto cube  = detail::make_cube(quadrant_t{quadrant}, which_tree) ; 
    // finally check for intersection 
    bool intersect = intersects(*plane,cube) ;
    GRACE_TRACE("Cube ({},{},{}) intersect ? {} local_num {} which_tree {}",
		cube.v[0][0],cube.v[0][1],cube.v[0][2], intersect, local_num, which_tree) ; 
    // if the quadrant is a leaf we write back 
    // to its user_int to flag it 
    if ( local_num >= 0 and intersect ) {
        quadrant->p.user_int = 1 ;
    }
    return intersect ; 
}


void oct_tree_plane_slicer_t::search() {
    std::vector<plane_desc_t> _buf{_plane} ; 
    auto plane_arr = sc_array_new_data(
        _buf.data(),sizeof(plane_desc_t), 1
    ) ; 
    // search 
    p4est_search_local(
        grace::amr::forest::get().get(), 
        false, 
        nullptr, 
        &grace_search_plane,
        plane_arr
    ) ; 
    // collect quadrants that returned 1 
    for( size_t iq=0UL; iq<_nq; iq+=1UL) {
        auto quad = amr::get_quadrant(iq) ; 
        if (quad.get_user_int()) sliced_quads.push_back(iq) ; 
    }
}

void oct_tree_plane_slicer_t::find_cells() {
    sliced_cell_offsets.clear() ; 
    sliced_cell_offsets.reserve(sliced_quads.size()) ; 
    for ( auto const& iq: sliced_quads ) {
        auto const idx = detail::get_inv_cell_spacing(iq, _plane.dir);
        auto const qc = detail::get_quad_coord_lbounds(iq) ; 
        size_t const offset = math::floor_int(
            Kokkos::fabs(qc[_plane.dir] - _plane.d[_plane.dir]) * idx + 1e-15
        ) ; 
        sliced_cell_offsets.push_back(offset) ; 
        
    }
}

void oct_tree_plane_slicer_t::slice() {
    reset_quads() ; 
    search() ; 
    find_cells() ;
}

#endif // GRACE_3D
    

} // namespace amr
} // namespace grace
