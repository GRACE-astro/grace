// octree_search_class.hh
#pragma once

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

class OctreeSlicer {
public:
    struct SlicedQuadrantInfo {
        size_t globalIndex;
        size_t treeIndex;
        size_t localQuadrantIdx;
    };

    struct SlicedCellInfo {
        SlicedQuadrantInfo q;
        size_t i, j, k;
    };
        
    struct PointSearchData {
        OctreeSlicer* slicer;
        std::array<double, 3> point;
        size_t iterator;
    };

    // Constructor
    OctreeSlicer();

    // Main interface
    void find_sliced_cells();
    size_t num_sliced_cells() const { return slicedCells_.size(); }
    const auto& sliced_quadrants() const { return slicedQuadrants_; }
    const auto& sliced_cells() const { return slicedCells_; }

    // For Konrad maybe useful
    void find_sliced_cells_for_sphere(int num_points, double radius);

private:
    // Data members
    forest_impl_t& forest_ = forest::get() ;
    grace::cartesian_coordinate_system_impl_t& coord_system_= coordinate_system::get();
    std::vector<SlicedQuadrantInfo> slicedQuadrants_;
    std::vector<SlicedCellInfo> slicedCells_;

    // Private methods
    std::pair<std::array<double, 3>, std::array<double, 3>> 
    get_physical_coordinates_private(p4est_t* p4est, p4est_topidx_t which_tree,
                            p4est_quadrant_t* quadrant, p4est_locidx_t local_num);

    // Search operations
    void octree_search();
    void search_quadrants();
    void search_cells();

    // p4est callback handlers
    static int handle_search(p4est_t* p4est, p4est_topidx_t which_tree,
                            p4est_quadrant_t* quadrant, p4est_locidx_t local_num,
                            void* user_data);
    static int handle_reset(p4est_t* p4est, p4est_topidx_t which_tree,
                           p4est_quadrant_t* quadrant, p4est_locidx_t local_num,
                           void* user_data);
    static int handle_point_search(p4est_t* p4est, p4est_topidx_t which_tree,
                            p4est_quadrant_t* quadrant, p4est_locidx_t local_num,
                            void* user_data);

    // Helper to generate sphere points
    static sc_array_t* generate_sphere_points(int num_points, double radius);
};

#endif // GRACE_3D
} // namespace amr
} // namespace grace