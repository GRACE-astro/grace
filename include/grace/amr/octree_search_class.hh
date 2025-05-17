// octree_search_class.hh
#ifndef GRACE_AMR_OCTREE_SEARCH_CLASS_HH
#define GRACE_AMR_OCTREE_SEARCH_CLASS_HH

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
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
    struct ReducedCellInfo {
        SlicedQuadrantInfo q;
        size_t i; // Information of the index staying const. in the slice
    };
    // Constructor
    OctreeSlicer();

    OctreeSlicer(std::string plane);

    // Main interface
    void find_sliced_cells();
    void set_localToSlicedIdx() {generate_continuous_indices();}
    size_t num_sliced_quadrants() const {return slicedQuadrants_.size();}
    size_t num_sliced_cells() const { return num_sliced_quadrants() * nx_ * ny_ * nz_; }
    const auto& sliced_quadrants() const { return slicedQuadrants_; }
    const auto& reduced_slice() const {return ReducedslicedCells_ ;}
    const auto& sliced_cells() const { return slicedCells_; }
    const auto& get_localToSlicedIdx() const { return localToSlicedIdx_; }
    const auto& get_dir() const {return dir_;}

    std::tuple<size_t,size_t,size_t>
    get_quadrant_extents()
    {
        return std::make_tuple(nx_,ny_,nz_) ;  
    }
    std::tuple<size_t,size_t>
    get_2D_quadrant_extents()
    {
        if (dir_ == 0)
            return std::make_tuple(ny_,nz_) ;  
        else if (dir_ == 1 )
            return std::make_tuple(nx_,nz_) ;
        else if (dir_ == 2 )
            return std::make_tuple(nx_,ny_) ;
        else 
            return std::make_tuple(nx_,ny_) ;
    }


    // For Konrad maybe useful
    void find_sliced_cells_for_sphere(int num_points, double radius);

private:
    // Data members
    forest_impl_t& forest_ = forest::get() ;
    grace::cartesian_coordinate_system_impl_t& coord_system_= coordinate_system::get();
    std::vector<SlicedQuadrantInfo> slicedQuadrants_;
    std::vector<SlicedCellInfo> slicedCells_;
    std::vector<ReducedCellInfo> ReducedslicedCells_;

    size_t nx_,ny_,nz_;
    std::unordered_map<size_t, size_t> localToSlicedIdx_;
    
    size_t dir_;

    // Private methods
    std::pair<std::array<double, 3>, std::array<double, 3>> 
    get_physical_coordinates_private(p4est_t* p4est, p4est_topidx_t which_tree,
                            p4est_quadrant_t* quadrant, p4est_locidx_t local_num);

    // Search operations
    void octree_search();
    void search_quadrants();
    void search_cells();
    void Reduced_search_cells();

    // p4est callback handlers
    static int handle_search_x(p4est_t* p4est, p4est_topidx_t which_tree,
                            p4est_quadrant_t* quadrant, p4est_locidx_t local_num,
                            void* user_data);
    static int handle_search_y(p4est_t* p4est, p4est_topidx_t which_tree,
                            p4est_quadrant_t* quadrant, p4est_locidx_t local_num,
                            void* user_data);
    static int handle_search_z(p4est_t* p4est, p4est_topidx_t which_tree,
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
    
    void generate_continuous_indices();

};


#endif // GRACE_3D
} // namespace amr
} // namespace grace
#endif