// octree_search_class.cc
#include <grace/amr/octree_search_class.hh>
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

OctreeSlicer::OctreeSlicer() {}

void OctreeSlicer::find_sliced_cells() {
    octree_search();
    search_quadrants();
    search_cells();
}

std::pair<std::array<double, 3>, std::array<double, 3>> 
OctreeSlicer::get_physical_coordinates_private(p4est_t* p4est, p4est_topidx_t which_tree,
                                      p4est_quadrant_t* quadrant, p4est_locidx_t local_num) {
    quadrant_t quad(quadrant);
    auto tree = forest_.tree(which_tree);

    auto const dx_quad  = 1./(1<<quad.level()) ; 
    auto const qcoords = quad.qcoords();
    auto quad_coords = std::array<double, 3>{
        qcoords[0] * dx_quad,
        qcoords[1] * dx_quad,
        qcoords[2] * dx_quad
    };

    // Das so zu machen wäre overkill
    std::pair<double,double> xbnd { 
        get_param<double>("amr", "xmin"),
        get_param<double>("amr", "xmax")
    } ;
    std::pair<double,double> ybnd { 
        get_param<double>("amr", "ymin"),
        get_param<double>("amr", "ymax")
    } ;
    std::pair<double,double> zbnd { 
        get_param<double>("amr", "zmin"),
        get_param<double>("amr", "zmax")
    } ;
    
    auto grid_bnd_ = std::array<std::pair<double,double>, GRACE_NSPACEDIM>{
        VEC(
        xbnd,ybnd,zbnd
        )
    } ;

    auto is_periodic_ = std::array<bool, GRACE_NSPACEDIM> {
        VEC(get_param<bool>("amr","periodic_x"), 
            get_param<bool>("amr","periodic_y"), 
            get_param<bool>("amr","periodic_z"))
    } ;

    auto const tree_coords = get_tree_vertex(which_tree,0UL) ; 
    auto const dx_tree     = get_tree_spacing(which_tree)    ;


    auto qcoords_lower = std::array<double, 3>{
        quad_coords[0] * dx_tree[0] + tree_coords[0],
        quad_coords[1] * dx_tree[1] + tree_coords[1],
        quad_coords[2] * dx_tree[2] + tree_coords[2]
    };
    
    
    for( int id=0; id<GRACE_NSPACEDIM; ++id){
        if( is_periodic_[id] ) {
            if ( qcoords_lower[id] < grid_bnd_[id].first ) {
                qcoords_lower[id] = grid_bnd_[id].second - (grid_bnd_[id].first-qcoords_lower[id]) ;
            } else if ( qcoords_lower[id] > grid_bnd_[id].second) {
                qcoords_lower[id] = grid_bnd_[id].first + (qcoords_lower[id]-grid_bnd_[id].second) ; 
            }
        }
    }

    auto qcoords_upper = std::array<double, 3>{
        qcoords_lower[0] + dx_tree[0]*dx_quad,
        qcoords_lower[1] + dx_tree[1]*dx_quad,
        qcoords_lower[2] + dx_tree[2]*dx_quad
    };

    return std::make_pair(qcoords_lower, qcoords_upper);
}

int OctreeSlicer::handle_search(p4est_t* p4est, p4est_topidx_t which_tree,
                               p4est_quadrant_t* quadrant, p4est_locidx_t local_num,
                               void* points) {
    
    OctreeSlicer slicer;
    
    if (local_num < 0) {
        // Skip trees that are not valid (e.g. ghost trees)
        std::array<double, 3> qcoords_upper = {0.0, 0.0, 0.0};
        std::array<double, 3> qcoords_lower = {0.0, 0.0, 0.0};

        std::tie(qcoords_lower, qcoords_upper) = slicer.get_physical_coordinates_private(p4est, which_tree, quadrant, local_num);

        auto xmin = qcoords_lower[0];
        auto xmax = qcoords_upper[0];

        if (xmin <= 0.0 && xmax > 0.0) {
            return 1; // Found a quadrant that is intersected
        }
        return 0; // Skip trees that are not the first one
    }

    std::array<size_t, 3> ijk = {0, 0, 0};

    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 

    std::array<double, 3> pcoords;
    std::array<double, 3> pcoords_max;
 
    pcoords = slicer.coord_system_.get_physical_coordinates(ijk, local_num, {VEC(0.0,0.0,0.0)}, false);
    pcoords_max = slicer.coord_system_.get_physical_coordinates(ijk, local_num, {VEC(static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz))}, false);

    auto x_min = pcoords[0];
    auto x_max = pcoords_max[0];

    if (x_min <= 0.0 && x_max > 0.0) {
        if (local_num < 0) {
            return 1; // Skip trees that are not valid (e.g. ghost trees)
        }
    
        quadrant->p.user_int = 0; // Reset the user data to 0
        return 1; // Found a quadrant that is intersected
    }
    return 0;
}

int OctreeSlicer::handle_reset(p4est_t* p4est, p4est_topidx_t which_tree,
                              p4est_quadrant_t* quadrant, p4est_locidx_t local_num,
                              void* points) {
    quadrant->p.user_int = 0;
    return 0;
}

void OctreeSlicer::octree_search() {
    // Reset quadrant markers
    p4est_search_local(forest_.get(), false, &handle_reset, nullptr, nullptr);
    
    // Perform search operation
    p4est_search_local(forest_.get(), false, &handle_search, nullptr, nullptr);
}

void OctreeSlicer::search_quadrants() {
    int num_sliced_quadrants = slicedQuadrants_.size();
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = get_quadrant_extents() ;
    auto num_sliced_cells = ny*nz*num_sliced_quadrants;

    size_t first = forest_.first_local_tree();
    size_t last = forest_.last_local_tree();
    
    for (size_t i = first; i <= last; ++i) // Loop from first to last local tree
    {
        auto tree = forest_.tree(i);  // Assuming there is a function to access a tree by index
        size_t quadrant_offset = tree.quadrants_offset();  // Number of quadrants in the tree
        size_t num_quadrants = tree.num_quadrants();  // Number of quadrants in the tree
        for (size_t j = 0; j < num_quadrants; ++j)  // Loop through all quadrants in the tree
        {
            auto quadrant = tree.quadrant(j);  // Get the j-th quadrant in the tree (adjust if needed)

            if (quadrant.get_user_data<int>() != 0)
            {
                slicedQuadrants_.push_back({quadrant_offset+j, i, j});
            }
        }
    } 
}

void OctreeSlicer::search_cells() {
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ;

    for (auto& q : slicedQuadrants_)
    {
        auto tree = forest_.tree(q.treeIndex);
        size_t quadrant_offset = tree.quadrants_offset();  // Number of quadrants in the tree
        size_t num_quadrants = tree.num_quadrants();  // Number of quadrants in the tree
        std::array<size_t, 3> ijk = {0, 0, 0};

        for (size_t i = 0; i < nx; ++i)
        {
            for (size_t j = 0; j < ny; ++j)
            {
                for (size_t k = 0; k < nz; ++k)
                {
                    double nx1, ny1, nz1;
                    nx1 = i;
                    ny1 = j;
                    nz1 = k;
                    double nx2, ny2, nz2;
                    nx2 = i+1;
                    ny2 = j+1;
                    nz2 = k+1;
                    // Get physical coordinates of the cell corners
                    std::array<double, 3> pcoords = coord_system_.get_physical_coordinates(ijk, q.globalIndex  , {VEC(nx1, ny1, nz1)}, false);
                    std::array<double, 3> pcoords_max = coord_system_.get_physical_coordinates(ijk, q.globalIndex, {VEC(nx2, ny2, nz2)}, false);
                    double x_min = pcoords[0];
                    double x_max = pcoords_max[0];
                    //printf("x_min: %f, x_max: %f\n", x_min, x_max);

                    if (x_min <= 0.0 && x_max > 0.0) {
                        slicedCells_.push_back({q, i, j, k});
                    }
                }
            }
        }
    }
}

void OctreeSlicer::find_sliced_cells_for_sphere(int num_points, double radius) {
    slicedQuadrants_.clear();
    slicedCells_.clear();
    // Beispiel: Wenn die Größe bei der Point Search bekannt ist:
    slicedCells_.resize(num_points);
    slicedQuadrants_.resize(num_points);

    sc_array_t* points = generate_sphere_points(num_points, radius);
    if (!points) return;

    for (size_t i = 0; i < points->elem_count; ++i) {
        std::array<double, 3>& point = *static_cast<std::array<double, 3>*>(sc_array_index(points, i));

        sc_array_t* sc_data = new sc_array_t;
        sc_data->elem_count = 1;
        sc_data->elem_size = sizeof(PointSearchData);
        sc_data->array = static_cast<char*>(malloc(sc_data->elem_count * sc_data->elem_size));
        // Add data to the array
        PointSearchData data = {this, point, i};
        PointSearchData* element = static_cast<PointSearchData*>(sc_array_push(sc_data));
        *element = data;

        printf("We are now entering p4est_search_local\n");
        p4est_search_local(forest_.get(),true, nullptr, handle_point_search, sc_data);
        // Cleanup
        sc_array_destroy(sc_data);
        delete sc_data;
    }


    // Cleanup
    free(points->array);
    delete points;

}

int OctreeSlicer::handle_point_search(p4est_t* p4est, p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant, p4est_locidx_t local_num,
    void* user_data) {
    printf("So we are failing at the p4est level\n");
    sc_array_t* sc_data = reinterpret_cast<sc_array_t*>(user_data);
    if (sc_data->elem_count == 0) return 0;
    
    PointSearchData* data = reinterpret_cast<PointSearchData*>(sc_data);
    OctreeSlicer* slicer = data->slicer;
    const auto& point = data->point;
    const auto& iterator = data->iterator;
    printf("poimt[0] %f\n", point[0]);
    printf("poimt[1] %f\n", point[1]);
    printf("poimt[2] %f\n", point[2]);
    
    std::array<double, 3> upper = {0.0, 0.0, 0.0};
    std::array<double, 3> lower = {0.0, 0.0, 0.0};
    std::tie(lower, upper) = slicer->get_physical_coordinates_private(p4est, which_tree, quadrant, local_num);
    
    printf("Are we coming this this far\n");
    printf("local_num %d\n", local_num);
    if (lower[0] <= point[0] && upper[0] > point[0] &&
        lower[1] <= point[1] && upper[1] > point[1] &&
        lower[2] <= point[2] && upper[2] > point[2]) {
        
        if (local_num < 0) {
            return 1;
        }

        SlicedQuadrantInfo qInfo;
        qInfo.globalIndex = local_num;
        qInfo.treeIndex = which_tree;
        qInfo.localQuadrantIdx = local_num;
            
        size_t nx, ny, nz;
        std::tie(nx, ny, nz) = grace::amr::get_quadrant_extents();
            
        double dx = (upper[0] - lower[0]) / nx;
        double dy = (upper[1] - lower[1]) / ny;
        double dz = (upper[2] - lower[2]) / nz;
            
        size_t i = static_cast<size_t>((point[0] - lower[0]) / dx);
        size_t j = static_cast<size_t>((point[1] - lower[1]) / dy);
        size_t k = static_cast<size_t>((point[2] - lower[2]) / dz);
            
        i = std::min(i, nx - 1);
        j = std::min(j, ny - 1);
        k = std::min(k, nz - 1);
        
        printf("passiert hier der Fehler i,j,k = %zu %zu %zu\n", i, j, k);
        SlicedCellInfo cellInfo{qInfo, i, j, k};
        slicer->slicedCells_[iterator] = (cellInfo);
        slicer->slicedQuadrants_[iterator] = (qInfo);
        printf("SlicedQuadrants_ size %zu\n", slicer->slicedQuadrants_.size());
            
        return 1;
    }
    return 0;
}


sc_array_t* OctreeSlicer::generate_sphere_points(int num_points, double radius) {
    using value_type = std::array<double, 3>;
    
    sc_array_t* sc_array = new sc_array_t;
    sc_array->elem_count = num_points;
    sc_array->elem_size = sizeof(value_type);
    sc_array->array = static_cast<char*>(malloc(sc_array->elem_count * sc_array->elem_size));
    value_type* data = reinterpret_cast<value_type*>(sc_array->array);
    const double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
    const double pi = 3.141592653589793;

    for (int i = 0; i < num_points; ++i) {
        double z = 1.0 - (2.0 * i + 1.0) / num_points;
        double theta = 2.0 * pi * i / golden_ratio;
        double xy_radius = std::sqrt(1.0 - z*z);
        data[i] = {xy_radius * std::cos(theta) * radius,
                   xy_radius * std::sin(theta) * radius,
                   z * radius};
    }

    return sc_array;
}



#endif // GRACE_3D
} // namespace amr
} // namespace grace