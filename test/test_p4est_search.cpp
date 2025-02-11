#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <grace_config.h>
#include <grace/IO/hdf5_output.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/vtk_output.hh>
#include <grace/amr/p4est_headers.hh> //Added this
#include <iostream>

namespace {
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
int my_search_function(p4est_t * p4est,
                       p4est_topidx_t which_tree,
                       p4est_quadrant_t * quadrant,
                       p4est_locidx_t local_num,
                       void *point) {

        if (local_num < 0) {
            return 1; // Skip trees that are not the first one
        }

        std::array<size_t, 3> ijk = {0, 0, 0};

        size_t nx,ny,nz; 
        std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 

        std::array<double, 3> pcoords;
        std::array<double, 3> pcoords_max;

        auto& coord_system = grace::coordinate_system::get() ; 
        pcoords = coord_system.get_physical_coordinates(ijk, local_num, {VEC(0.0,0.0,0.0)}, false);
        pcoords_max = coord_system.get_physical_coordinates(ijk, local_num, {VEC(static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz))}, false);
        
        auto x_min = pcoords[0];
        auto x_max = pcoords_max[0];

        if (x_min <= 0.0 && x_max > 0.0) {
            quadrant->p.user_int += 1; // Mark the quadrant as not intersected
            printf("Quadrant %d in tree %d is sliced by the plane.\n", local_num, which_tree);
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
int my_points_function(p4est_t* p4est,
                       p4est_topidx_t which_tree,
                       p4est_quadrant_t* quadrant,
                       p4est_locidx_t local_num,
                       void* point)
{
    if (local_num < 0)
    {
        // Skip trees that are not valid (e.g. ghost trees)
        return 1;
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
        lower[2] <= point_coord[2] && upper[2] > point_coord[2])
    {
        // All conditions are satisfied, so the quadrant intersects the plane.
        printf("Quadrant %d in tree %d intersects the point\n",
               local_num, which_tree);
        return 1;
    }
    else
    {
        // One or more conditions are not met.
        //printf("Quadrant %d in tree %d does not intersect the plane\n", local_num, which_tree);
        return 1;
    }
    
}

//*****************************************************************************************************
/**
 * @brief Generate a set of test points.
 * 
 * This function creates a static array of 10 predefined test points.
 * The points are stored in an sc_array_t structure for easy access.
 * 
 * @return Pointer to the dynamically allocated sc_array_t containing the points.
 */
sc_array_t* generate_test_points() {
    using value_type = std::array<double, 3>;
    
    // Initialize an sc_array_t
    sc_array_t* sc_array = new sc_array_t;
    sc_array->elem_count = 10; // Example size
    sc_array->elem_size = sizeof(value_type);
    sc_array->array = static_cast<char*>(malloc(sc_array->elem_count * sc_array->elem_size));
    
    if (!sc_array->array) {
        printf("Memory allocation failed!\n");
        delete sc_array;
        return nullptr;
    }
    
    // Populate the array with some values
    value_type* data = reinterpret_cast<value_type*>(sc_array->array);
    data[0] = { -0.9, 0.1, 0.1 };
    data[1] = { -0.7, 0.5, 0.2 };
    data[2] = { -0.3, 0.2, 0.7 };
    data[3] = {  0.0, 0.0, 0.0 };
    data[4] = {  0.0, 0.5, 0.5 };
    data[5] = {  0.3, 0.8, 0.2 };
    data[6] = {  0.5, 0.3, 0.4 };
    data[7] = {  0.7, 0.9, 0.6 };
    data[8] = {  0.9, 0.6, 0.8 };
    data[9] = {  0.8, 0.4, 0.9 };

    // Wrap it using sc_array_view_t
    grace::sc_array_view_t<value_type> view(sc_array);

    return sc_array;
}

//*****************************************************************************************************
/**
 * @brief Generate points distributed on a sphere using Fibonacci lattice.
 * 
 * This function generates a set of points evenly distributed on the surface of a sphere
 * using the golden ratio method. It is useful for approximating a uniform spherical distribution.
 * 
 * @param num_points The number of points to generate.
 * @param radius The radius of the sphere.
 * 
 * @return Pointer to the dynamically allocated sc_array_t containing the generated points.
 */
sc_array_t* generate_sphere_points(int num_points, double radius) {
    using value_type = std::array<double, 3>;
    
    sc_array_t* sc_array = new sc_array_t;
    sc_array->elem_count = num_points;
    sc_array->elem_size = sizeof(value_type);
    sc_array->array = static_cast<char*>(malloc(sc_array->elem_count * sc_array->elem_size));
    
    if (!sc_array->array) {
        printf("Memory allocation failed!\n");
        delete sc_array;
        return nullptr;
    }
    
    value_type* data = reinterpret_cast<value_type*>(sc_array->array);
    const double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
    const double pi = 3.14159265358979323846;
    
    for (int i = 0; i < num_points; ++i) {
        // Calculate z using a linear distribution
        double z = 1.0 - (2.0 * i + 1.0) / num_points;
        // Calculate theta using the golden ratio
        double theta = 2.0 * pi * i / golden_ratio;
        
        // Compute x and y coordinates on the unit sphere
        double radius_xy = std::sqrt(1.0 - z * z);
        double x = radius_xy * std::cos(theta);
        double y = radius_xy * std::sin(theta);
        
        // Scale by the desired radius
        data[i] = {x * radius, y * radius, z * radius};
    }
    
    return sc_array;
}

// Function evaluates if a point is inside a spherecell
void punkt_in_kugelzelle(double x, double y, double z, double R, int Nr, int Ntheta, int Nphi) {
    // 1. Kugelkoordinaten berechnen
    double r_p = std::sqrt(x*x + y*y + z*z);
    double theta = std::atan2(y, x);   // Längengrad (-π bis π)
    double phi = std::acos(z / r_p);   // Breitengrad (0 bis π)

    // 2. Zellengröße bestimmen
    double delta_r = R / Nr;
    double delta_theta = 2 * M_PI / Ntheta;
    double delta_phi = M_PI / Nphi;

    // 3. Zellenindex bestimmen
    int r_index = std::min(Nr - 1, static_cast<int>(r_p / delta_r));
    int theta_index = std::min(Ntheta - 1, static_cast<int>((theta + M_PI) / delta_theta)); // Theta wird auf [0, 2π] verschoben
    int phi_index = std::min(Nphi - 1, static_cast<int>(phi / delta_phi));
}

} // namespace
TEST_CASE("Volume hdf5 surface output", "[vol_hdf5_surf_out]")
{
    printf("---- KENS TERMINAL OUTPUT ----\n");
    
    // Need a p4est
    auto& forest = grace::amr::forest::get() ; 
    p4est_search_local_t search_func = my_search_function;
    p4est_search_local(forest.get(), false, search_func, nullptr, nullptr);

    printf("----------------------------------------\n");

    sc_array_t* points = generate_sphere_points(15,0.5);
    p4est_search_local_t point_search_func = my_points_function;
    p4est_search_local(forest.get(), true , nullptr, point_search_func, points);


    printf("---- END OF KENS TERMINAL OUTPUT ----\n");
}