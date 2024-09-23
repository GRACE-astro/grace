/**
 * @file test_lagrange_interpolators.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-22
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
#include <grace/utils/lagrange_interpolators.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/utils/prolongation.hh>
#include <Kokkos_Core.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


TEST_CASE("lagrange_interp", "[lagrange_interp]")
{
    using namespace Kokkos ; 
    using namespace utils  ;
    using namespace grace  ;  

    /************************************************/
    /*      SECOND ORDER ACCURATE INTERPOLATORS     */
    /************************************************/
    /* We start by testing the second order (linear)*/
    /* Lagrange interpolators. We construct a linear*/
    /* function and fill a 3D View consisting of 4  */
    /* points (cell-centered grid) and 8 points     */
    /* (corner-staggered grid) and test our interp  */
    /* routines.                                    */
    /************************************************/ 

    auto lin_func = KOKKOS_LAMBDA (std::array<double,3> const& xyz)
    {
        double x = xyz[0] ; double y = xyz[1] ; double z = xyz[2] ; 
        return 2.5 * x - 7.1 * y + 8.9 * z ; 
    } ; 

    int nx{6},ny{6},nz{6}, ngz{2} ; 
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

    double const x0{-1}, x1{1} ; 
    size_t const N = 6 ; 
    double const h { (x1-x0)/N } ; 

    auto coords_center = KOKKOS_LAMBDA (int i, int j, int k)
    {
        return std::array<double,3> {
            x0 + (i + 0.5) * h , 
            x0 + (j + 0.5) * h ,
            x0 + (k + 0.5) * h 
        } ; 
    } ; 

    auto coords_corner = KOKKOS_LAMBDA (int i, int j, int k)
    {
        return std::array<double,3> {
            x0 + i * h , 
            x0 + j * h ,
            x0 + k * h 
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

    /* Now we call the interpolations and check the results */
    /* Corners */
    MDRangePolicy<Rank<3>, IndexType<int>> policy_interp_corners {
        {0,0,0},
        {nx, ny, nz}
    } ; 
    parallel_for("fill_data",policy_corners,
    KOKKOS_LAMBDA( int i, int j, int k) {
        int const ichild = math::floor_int((2*i)/nx) 
            + 2 * ( math::floor_int((2*j)/ny) + 2 * math::floor_int((2*k)/nz) ) ;
        int i_f = (2*i)%nx + ngz ; 
        int j_f = (2*j)%ny + ngz ; 
        int k_f = (2*k)%nz + ngz ; 
        auto fine_view = subview(corner_staggered_fine,ALL(),ALL(),ALL(),ichild) ;
        lagrange_prolongator_t<2>::interpolate(i_f,j_f,k_f,i+ngz,j+ngz,k+ngz,corner_staggered, fine_view) ; 
    }) ; 

    /* Check */
    auto h_cell_centered = create_mirror_view(cell_centered_fine) ; 
    auto h_corner_staggered = create_mirror_view(corner_staggered_fine) ;

    deep_copy(h_corner_staggered,corner_staggered_fine) ; 

    double const quad_side = (x1-x0)/2. ; 

    auto get_fine_coords_corner = [&] (int i, int j, int k, int q) {
        int xq = (q >> 0) & 1;  
        int yq = (q >> 1) & 1;  
        int zq = (q >> 2) & 1;

        std::array<double,3> xyz ; 

        xyz[0] = i * 0.5*h + x0 + xq * quad_side ;
        xyz[1] = j * 0.5*h + x0 + yq * quad_side ;
        xyz[2] = k * 0.5*h + x0 + zq * quad_side ;

        return xyz ; 
    } ; 

    for( int i=ngz; i<nx+2*ngz; ++i) {
        for( int j=ngz; j<ny+2*ngz; ++j) {
            for( int k=ngz; k<nz+2*ngz; ++k){
                for( int q=0; q<8; ++q) {
                    auto xyz = get_fine_coords_corner(i,j,k,q) ; 
                    auto yval = lin_func(xyz) ; 

                    REQUIRE_THAT(
                        h_corner_staggered(i,j,k,q),
                        Catch::Matchers::WithinAbs( yval, 1e-10)
                    ) ; 
                }
            }
        }
    }

}