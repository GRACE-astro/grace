/**
 * @file test_lagrange_edge_interpolators.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de), Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-12-26
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

#include <catch2/catch_test_macros.hpp>
#include <grace/utils/numerics/lagrange_interpolators.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/utils/numerics/prolongation.hh>
#include <Kokkos_Core.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


#include <iostream>


TEST_CASE("lagrange_interp_edge", "[lagrange_interp_edge]")
{
    using namespace Kokkos ; 
    using namespace utils  ;
    using namespace grace  ;  
    using utils::delta;

    /****************************************************************/
    /*      SECOND ORDER ACCURATE EDGE-STAGGERED INTERPOLATORS      */
    /****************************************************************/
    /* We will be testing second order (linear)                     */
    /* Lagrange interpolators. We construct a linear                */
    /* function and fill a 3D Coarse View and fill out              */
    /* a 3D FIne View                                                */
    /****************************************************************/ 

    auto lin_func = KOKKOS_LAMBDA (std::array<double,3> const& xyz)
    {
        double x = xyz[0] ; double y = xyz[1] ; double z = xyz[2] ; 
        return 2.5 * x - 7.1 * y + 8.9 * z ; 
    } ; 

    int nx{8},ny{8},nz{8}, ngz{4} ; 
    View<double***, default_space> cell_centered(
        "cell_centered_data", nx+2*ngz,ny+2*ngz,nz+2*ngz 
    ) ; 
    View<double*** *, default_space> cell_centered_fine(
        "cell_centered_fine_data", nx+2*ngz,ny+2*ngz,nz+2*ngz, 8
    ) ;
    View<double***, default_space> corner_staggered(
        "corner_staggered_data", nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz 
    ) ;
    View<double*** *, default_space> corner_staggered_fine(
        "corner_staggered_data", nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz, 8 
    ) ;

    // hard-code the direction for now 
    constexpr size_t edgeDir= 2; // assume edge-staggered variable is in xy-dir (e.g. A_z)

    // one index fewer for the non-staggered directions
    View<double***, default_space> edge_staggered(
        "edge_staggered_data", nx+1-delta(0,edgeDir)+2*ngz,ny+1-delta(1,edgeDir)+2*ngz,nz+1-delta(2,edgeDir)+2*ngz) ;

    View<double*** *, default_space> edge_staggered_fine(
        "edge_staggered_data", nx+1-delta(0,edgeDir)+2*ngz,ny+1-delta(1,edgeDir)+2*ngz,nz+1-delta(2,edgeDir)+2*ngz, 8 
    ) ;


    double const x0{-1}, x1{1} ; 
    double const h { (x1-x0)/nx } ; 

    auto coords_center = KOKKOS_LAMBDA (int i, int j, int k)
    {
        return std::array<double,3> {
            x0 + (i - ngz + 0.5) * h , 
            x0 + (j - ngz + 0.5) * h ,
            x0 + (k - ngz + 0.5) * h 
        } ; 
    } ; 

    auto coords_corner = KOKKOS_LAMBDA (int i, int j, int k)
    {
        return std::array<double,3> {
            x0 + (i-ngz) * h , 
            x0 + (j-ngz) * h ,
            x0 + (k-ngz) * h 
        } ; 
    } ;

    // exactly like in the corner case, except one direction has a +0.5 (here it's the z coordinate)
    auto coords_edge = KOKKOS_LAMBDA (int i, int j, int k)
    {
        return std::array<double,3> {
            x0 + (i-ngz + delta(0,edgeDir)*0.5) * h , 
            x0 + (j-ngz + delta(1,edgeDir)*0.5) * h ,
            x0 + (k-ngz + delta(2,edgeDir)*0.5) * h 
        } ; 
    } ;

    MDRangePolicy<Rank<3>> policy_centers {
        {0,0,0},
        {nx+2*ngz, ny+2*ngz, nz+2*ngz}
    } ; 
    parallel_for("fill_data",policy_centers,
    KOKKOS_LAMBDA( int i, int j, int k) {
        auto const& xyz_center = coords_center(i,j,k) ; 
        cell_centered(i,j,k) = lin_func(xyz_center)   ; 
    }) ; 

    MDRangePolicy<Rank<3>> policy_corners {
        {0,0,0},
        {nx+1+2*ngz, ny+1+2*ngz, nz+1+2*ngz}
    } ; 
    parallel_for("fill_data",policy_corners,
    KOKKOS_LAMBDA( int i, int j, int k) {
        auto const& xyz_corner = coords_corner(i,j,k) ; 
        corner_staggered(i,j,k) = lin_func(xyz_corner)   ; 
    }) ; 

    // policy edges:
    MDRangePolicy<Rank<3>> policy_edges{
        {0,0,0},
        {nx+2*ngz +1-delta(0,edgeDir), ny+2*ngz +1-delta(1,edgeDir), nz+2*ngz +1-delta(2,edgeDir)}
    } ; 
    parallel_for("fill_data_edge",policy_edges,
    KOKKOS_LAMBDA( int i, int j, int k) {
        auto const& xyz_edge = coords_edge(i,j,k) ; 
        edge_staggered(i,j,k) = lin_func(xyz_edge)   ; 
    }) ; 

    /* Now we call the interpolations and check the results */
    /* Corners */
    MDRangePolicy<Rank<3>, IndexType<int>> policy_interp_corners {
        {0,0,0},
        {nx, ny, nz}
    } ; 
    parallel_for("interpolate_data",policy_interp_corners,
    KOKKOS_LAMBDA( int i, int j, int k) {
        int const ichild = math::floor_int((2*i)/nx) 
            + 2 * ( math::floor_int((2*j)/ny) + 2 * math::floor_int((2*k)/nz) ) ;
        int i_f = (2*i)%nx + ngz ; 
        int j_f = (2*j)%ny + ngz ; 
        int k_f = (2*k)%nz + ngz ; 
        auto fine_view = subview(corner_staggered_fine,ALL(),ALL(),ALL(),ichild) ;
        lagrange_prolongator_t<4>::interpolate(i_f,j_f,k_f,i+ngz,j+ngz,k+ngz,corner_staggered, fine_view) ; 
    }) ; 

  
    /* Edges  */
    // same policy - we iterate over cell centers as it's most logical and easiest to follow
    MDRangePolicy<Rank<3>, IndexType<int>> policy_interp_edges {
        {0,0,0},
        {nx, ny, nz}
    } ; 
    parallel_for("interpolate_data",policy_interp_edges,
    KOKKOS_LAMBDA( int i, int j, int k) {
        int const ichild = math::floor_int((2*i)/nx) 
            + 2 * ( math::floor_int((2*j)/ny) + 2 * math::floor_int((2*k)/nz) ) ;
        int i_f = (2*i)%nx + ngz ; 
        int j_f = (2*j)%ny + ngz ; 
        int k_f = (2*k)%nz + ngz ; 
        auto fine_view = subview(edge_staggered_fine,ALL(),ALL(),ALL(),ichild) ;
        lagrange_edge_prolongator_t<2,edgeDir>::interpolate(i_f,j_f,k_f,i+ngz,j+ngz,k+ngz,edge_staggered, fine_view) ; 
    }) ; 


    /* Check */
    auto h_corner_staggered = create_mirror_view(corner_staggered_fine) ;

    deep_copy(h_corner_staggered,corner_staggered_fine) ; 

    double const quad_side = (x1-x0)/2. ; 

    auto get_fine_coords_corner = [&] (int i, int j, int k, int q) {
        // q here picks out the child 
        // from one of the children quadrants 

        int xq = (q >> 0) & 1;  
        int yq = (q >> 1) & 1;  
        int zq = (q >> 2) & 1;

        std::array<double,3> xyz ; 
        xyz[0] = (i-ngz) * 0.5*h + x0 + xq * quad_side ;
        xyz[1] = (j-ngz) * 0.5*h + x0 + yq * quad_side ;
        xyz[2] = (k-ngz) * 0.5*h + x0 + zq * quad_side ;

        return xyz ; 
    } ; 

    for( int i=ngz; i<nx+1+ngz; ++i) {
        for( int j=ngz; j<ny+1+ngz; ++j) {
            for( int k=ngz; k<nz+1+ngz; ++k){
                for( int q=0; q<8; ++q) {
                    auto xyz = get_fine_coords_corner(i,j,k,q) ; 
                    auto yval = lin_func(xyz) ; 
   
                    // CHECK_THAT(
                    //     h_corner_staggered(i,j,k,q),
                    //     Catch::Matchers::WithinAbs( yval, 1e-10)
                    // ) ; 
                }
            }
        }
    }

    /* Check for edges: */
    // create mirror view
    auto h_edge_staggered = create_mirror_view(edge_staggered_fine) ;
    deep_copy(h_edge_staggered,edge_staggered_fine) ; 

    // create for checking 
    auto h_coarse_edge_staggered = create_mirror_view(edge_staggered) ;
    deep_copy(h_coarse_edge_staggered,edge_staggered) ; 


    // q here picks out the child of the coarse quadrant 
    auto get_fine_coords_edge = [&] (int i, int j, int k, int q) {

        int xq = (q >> 0) & 1;  
        int yq = (q >> 1) & 1;  
        int zq = (q >> 2) & 1;

        std::array<double,3> xyz ;  
        xyz[0] = (i-ngz) * 0.5*h + delta(0,edgeDir)* 0.25 * h + x0 + xq * quad_side ; // this is the same as in the case of corner
        xyz[1] = (j-ngz) * 0.5*h + delta(1,edgeDir)* 0.25 * h + x0 + yq * quad_side ; // same as corner 
        xyz[2] = (k-ngz) * 0.5*h + delta(2,edgeDir)* 0.25 * h + x0 + zq * quad_side ; // has to match centre location; we therefore shift by 0.25 of the h (half of 0.5h)

        // for i=j=k=ngz and q=0, we get: xyz[:] = {0,0, 0.25*h}
        // for i=j=k=ngz and q=0, we get for corner: xyz[:] = {0,0,0}
        // for i=j=ngz, k=ngz+1 and q=0, we get for corner: xyz[:] = {0,0,0.5h}
        // therefore, the z-staggered fine edge variable fits inbetween the two fine corners
        
        
        return xyz ; 
    } ; 

    
    // this loop is over the fine indices + all the children quadrants of the coarse quandrant 
    // inspecting different fine edges - to a given coarse edge - can be done by fixing q and changing ijk 
    for( int i=ngz; i<nx+1+ngz-delta(0,edgeDir); ++i) {
        for( int j=ngz; j<ny+1+ngz-delta(1,edgeDir); ++j) {
            for( int k=ngz; k<nz+1+ngz-delta(2,edgeDir); ++k){
                for( int q=0; q<8; ++q) {
                    auto xyz = get_fine_coords_edge(i,j,k,q) ; 
                    auto yval = lin_func(xyz) ; 

                    CHECK_THAT(
                        h_edge_staggered(i,j,k,q),
                        Catch::Matchers::WithinAbs( yval, 1e-10)
                    ) ; 
                }
            }
        }
    }









}