/**
 * @file index_helper.hh
 * @author  (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-02-19
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

#include <Kokkos_Core.hpp>
#include <grace_config.h>
#include <grace/IO/hdf5_output.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/amr/p4est_headers.hh>


namespace grace {
#ifdef GRACE_3D
std::pair<std::array<double, 3>, std::array<double, 3>>
get_physical_coordinates_constraints(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant, p4est_locidx_t local_num, void *point) 
{
    amr::quadrant_t quad(quadrant);
    amr::forest_impl_t& forest = amr::forest::get();
    auto tree = forest.tree(which_tree);

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

    auto const tree_coords = amr::get_tree_vertex(which_tree,0UL) ; 
    auto const dx_tree     = amr::get_tree_spacing(which_tree)    ;


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

//*****************************************************************************************************
/**
 * @brief Function to search for quadrants that intersect with a slicing plane.
 * 
 * This function checks if a quadrant is intersected by a plane at x = 0. 
 * If the quadrant crosses this plane, it is marked and counted.
 *
 * @param p4est Pointer to the forest structure.
 * @param which_tree Index of the tree being processed.
 * @param quadrant Pointer to the quadrant being checked.
 * @param local_num Local index of the quadrant.
 * @param point Unused parameter (reserved for future use).
 * 
 * @return 1 if the quadrant is intersected, 0 otherwise.
 */
int 
my_search_function(p4est_t * p4est,
    p4est_topidx_t which_tree,
    p4est_quadrant_t * quadrant,
    p4est_locidx_t local_num,
    void *point) 
{
    if (local_num < 0) {
        // Skip trees that are not valid (e.g. ghost trees)
        std::array<double, 3> qcoords_upper = {0.0, 0.0, 0.0};
        std::array<double, 3> qcoords_lower = {0.0, 0.0, 0.0};

        std::tie(qcoords_lower, qcoords_upper) = get_physical_coordinates_constraints(p4est, which_tree, quadrant, local_num, point);

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

    auto& coord_system = coordinate_system::get() ; 
    pcoords = coord_system.get_physical_coordinates(ijk, local_num, {VEC(0.0,0.0,0.0)}, false);
    pcoords_max = coord_system.get_physical_coordinates(ijk, local_num, {VEC(static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz))}, false);

    auto x_min = pcoords[0];
    auto x_max = pcoords_max[0];

    if (x_min <= 0.0 && x_max > 0.0) {
        quadrant->p.user_int += 1; // Mark the quadrant as not intersected
        //printf("Quadrant %d in tree %d is sliced by the plane.\n", local_num, which_tree);
        return 1; // Found a quadrant that is intersected
    }
    else {
        quadrant->p.user_int = 0; // Mark the quadrant as not intersected
    }
    return 0;
}

//*****************************************************************************************************
/**
 * @brief Function to determine if a given point is inside a quadrant.
 * 
 * This function checks whether a point is inside the bounding box of a quadrant.
 * If the point is within the quadrant's extents, it prints a message and returns 1.
 *
 * @param p4est Pointer to the forest structure.
 * @param which_tree Index of the tree being processed.
 * @param quadrant Pointer to the quadrant being checked.
 * @param local_num Local index of the quadrant.
 * @param point Pointer to the coordinates of the point being checked.
 * 
 * @return 1 if the point is inside the quadrant, 0 otherwise.
 */
int 
my_points_function(p4est_t* p4est,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant,
    p4est_locidx_t local_num,
    void* point)
{
    if (local_num < 0)    {
        // Skip trees that are not valid (e.g. ghost trees)
        std::array<double, 3> qcoords_upper = {0.0, 0.0, 0.0};
        std::array<double, 3> qcoords_lower = {0.0, 0.0, 0.0};
        std::tie(qcoords_lower, qcoords_upper) = get_physical_coordinates_constraints(p4est, which_tree, quadrant, local_num, point);

        auto& point_coord = *static_cast<std::array<double,3>*>(point);

        if (qcoords_lower[0] <= point_coord[0] && qcoords_upper[0] > point_coord[0] &&
            qcoords_lower[1] <= point_coord[1] && qcoords_upper[1] > point_coord[1] &&
            qcoords_lower[2] <= point_coord[2] && qcoords_upper[2] > point_coord[2])
        {
            // All conditions are satisfied, so the quadrant intersects the plane.
            return 1;
        }
        else {
            // One or more conditions are not met.
            return 0;
        }
    }
    std::array<size_t, 3> ijk = {0, 0, 0};

    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 

    // Retrieve the coordinate system from the grace library.
    auto& coord_system = grace::coordinate_system::get();
    std::array<double, 3> lower = coord_system.get_physical_coordinates(ijk, local_num, {VEC(0.0,0.0,0.0)}, false);
    std::array<double, 3> upper = coord_system.get_physical_coordinates(ijk, local_num, {VEC(static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz))}, 0);

    auto& point_coord = *static_cast<std::array<double,3>*>(point);
    if (lower[0] <= point_coord[0] && upper[0] > point_coord[0] &&
        lower[1] <= point_coord[1] && upper[1] > point_coord[1] &&
        lower[2] <= point_coord[2] && upper[2] > point_coord[2]){
        // All conditions are satisfied, so the quadrant intersects the plane.
        quadrant->p.user_int += 1;
        return 1;
    }
    else {
        // One or more conditions are not met.
        return 0;
    }
}

//*****************************************************************************************************
/**
 * @brief Function to rewrite user data to one.
 * 
 * This functions takes in every quadrant and resets the user data to 0
 *
 * @param p4est Pointer to the forest structure.
 * @param which_tree Index of the tree being processed.
 * @param quadrant Pointer to the quadrant being checked.
 * @param local_num Local index of the quadrant.
 * @param point Pointer to the coordinates of the point being checked.
 * 
 * @return 1
 */
int reset_user_data(p4est_t* p4est,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant,
    p4est_locidx_t local_num,
    void* point)
{
    if (local_num < 0) {
        return 1; // Skip trees that are not valid (e.g. ghost trees)
    }

    quadrant->p.user_int = 0; // Reset the user data to 0
    return 0;
}

void octree_search() 
{
    // Need a p4est
    auto& forest = grace::amr::forest::get() ; 

    // Reset the user data to 0
    p4est_search_local_t reset_func = reset_user_data;
    p4est_search_local(forest.get(), false, reset_func, nullptr, nullptr);

    // Test the quadrant slicing
    p4est_search_local_t search_func = my_search_function;
    p4est_search_local(forest.get(), false, search_func, nullptr, nullptr);

    


    // Reset the user data to 0
    //p4est_search_local(forest.get(), false, reset_func, nullptr, nullptr);

    //sc_array_t* points = generate_sphere_points(500,0.5);
    //p4est_search_local_t point_search_func = my_points_function;
    //p4est_search_local(forest.get(), true , nullptr, point_search_func, points);
}
    #endif //GRACE_3D

int num_sliced_trees()
{
    auto& forest = grace::amr::forest::get() ;
    size_t first = forest.first_local_tree();
    size_t last = forest.last_local_tree();
    size_t num_sliced_trees = 0;

    for (size_t i = first; i <= last; ++i) // Loop from first to last local tree
    {
        auto tree = forest.tree(i);  // Assuming there is a function to access a tree by index
        size_t quadrant_offset = tree.quadrants_offset();  // Number of quadrants in the tree
        size_t num_quadrants = tree.num_quadrants();  // Number of quadrants in the tree
        for (size_t j = 0; j < num_quadrants; ++j)  // Loop through all quadrants in the tree
        {
            auto quadrant = tree.quadrant(j);  // Get the j-th quadrant in the tree (adjust if needed)

            if (quadrant.get_user_data<int>() != 0)
            {
                num_sliced_trees += quadrant.get()->p.user_int;
            }
        }
    } 
    return num_sliced_trees;
}

// Define a structure to hold quadrant information.
struct SlicedQuadrantInfo {
    size_t globalIndex;       // For example: computed as quadrant_offset + j.
    size_t treeIndex;         // Which tree the quadrant came from.
    size_t localQuadrantIdx;  // The quadrant's index within the tree.
};

// Create a vector to store the information.
std::vector<SlicedQuadrantInfo> slicedQuadrants;
struct SlicedCellInfo {
    SlicedQuadrantInfo q;
    size_t i, j, k;
};

void search_cells_in_octrees()
{
    octree_search(); // This function is setting all the necessary user data.

    int num_sliced_quadrants = num_sliced_trees();
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    auto num_sliced_cells = ny*nz*num_sliced_quadrants;

    auto& forest = grace::amr::forest::get() ;
    size_t first = forest.first_local_tree();
    size_t last = forest.last_local_tree();
    
    for (size_t i = first; i <= last; ++i) // Loop from first to last local tree
    {
        auto tree = forest.tree(i);  // Assuming there is a function to access a tree by index
        size_t quadrant_offset = tree.quadrants_offset();  // Number of quadrants in the tree
        size_t num_quadrants = tree.num_quadrants();  // Number of quadrants in the tree
        for (size_t j = 0; j < num_quadrants; ++j)  // Loop through all quadrants in the tree
        {
            auto quadrant = tree.quadrant(j);  // Get the j-th quadrant in the tree (adjust if needed)

            if (quadrant.get_user_data<int>() != 0)
            {
                //printf("quad offset: %zu\n", tree.quadrants_offset());
                //printf("!Quadrant %zu in tree %zu is sliced by the plane.\n", quadrant_offset+j, i);
                slicedQuadrants.push_back({quadrant_offset+j, i, j});
            }
        }
    } // End of octree_search, hopefully deleting every variable here 

    std::vector<SlicedCellInfo> slicedCells;
    auto &coord_system = coordinate_system::get();

    for (auto& q : slicedQuadrants)
    {
        auto tree = forest.tree(q.treeIndex);
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
                    std::array<double, 3> pcoords = coord_system.get_physical_coordinates(ijk, q.globalIndex  , {VEC(nx1, ny1, nz1)}, false);
                    std::array<double, 3> pcoords_max = coord_system.get_physical_coordinates(ijk, q.globalIndex, {VEC(nx2, ny2, nz2)}, false);
                    double x_min = pcoords[0];
                    double x_max = pcoords_max[0];
                    //printf("x_min: %f, x_max: %f\n", x_min, x_max);

                    if (x_min <= 0.0 && x_max > 0.0) {
                        slicedCells.push_back({q, i, j, k});
                        //printf("Cell (%zu, %zu, %zu)\n", i, j, k);
                    }
                }
            }
        }
    }
    printf("num_sliced_cells: %zu\n", num_sliced_cells);
    printf("slicedQuadrants: %zu\n", slicedQuadrants.size());
    printf("Sliced cells: ahahahahhahah %zu\n", slicedCells.size());
}
}
