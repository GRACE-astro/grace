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

    // hard-code the edge direction for now 
    constexpr size_t edgeDir= 2; // assume edge-staggered variable is in z dir

    // one index fewer for the staggered directions
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
        {nx+2*ngz+1-delta(0,edgeDir), ny+2*ngz+1-delta(1,edgeDir), nz+2*ngz+1-delta(2,edgeDir)}
    } ; 
    parallel_for("fill_data_edge",policy_edges,
    KOKKOS_LAMBDA( int i, int j, int k) {
        auto const& xyz_edge = coords_edge(i,j,k) ; 
        edge_staggered(i,j,k) = lin_func(xyz_edge)   ; 
    }) ; 


    // auto h_coarse_edge_staggered = create_mirror_view(edge_staggered) ;
    // deep_copy(h_coarse_edge_staggered,edge_staggered) ; 
    // for( int i=ngz; i<nx+1-delta(edgeDir,0)+ngz; ++i) {
    //     for( int j=ngz; j<ny+1-delta(edgeDir,1)+ngz; ++j) {
    //         for( int k=ngz; k<nz+1-delta(edgeDir,2)+ngz; ++k){

    //             if(std::abs(h_coarse_edge_staggered(i,j,k))<1.e-8) std::cout << "edge staggered so small why" << std::endl;
    //         }}}
    

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
    // MDRangePolicy<Rank<3>, IndexType<int>> policy_interp_edges {
    //     {0,0,0},
    //     {nx+1+ngz, ny+1+ngz, nz+ngz}
    // } ; 
    // parallel_for("interpolate_data",policy_interp_edges,
    // KOKKOS_LAMBDA( int i, int j, int k) {
    //     int const ichild = math::floor_int((2*i)/nx) 
    //         + 2 * ( math::floor_int((2*j)/ny) + 2 * math::floor_int((2*k)/nz) ) ;
    //     int i_f = (2*i)%nx + ngz ; 
    //     int j_f = (2*j)%ny + ngz ; 
    //     int k_f = (2*k)%nz + ngz ; 
    //     auto fine_view = subview(edge_staggered_fine,ALL(),ALL(),ALL(),ichild) ;
    //     lagrange_edge_prolongator_t<2,edgeDir>::interpolate(i_f,j_f,k_f,i+ngz,j+ngz,k+ngz,edge_staggered, fine_view) ; 
    // }) ; 

    // DO A DIFFERENT POLICY, I DON'T UNDERSTAND THE ONE ABOVE
        MDRangePolicy<Rank<3>, IndexType<int>> policy_interp_edges {
        {0,0,0},
        {nx+1, ny+1, nz}
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
        // q here picks out the corner
        int xq = (q >> 0) & 1;  
        int yq = (q >> 1) & 1;  
        int zq = (q >> 2) & 1;

        std::array<double,3> xyz ; 
        xyz[0] = (i-ngz) * 0.5*h + x0 + xq * quad_side ;
        xyz[1] = (j-ngz) * 0.5*h + x0 + yq * quad_side ;
        xyz[2] = (k-ngz) * 0.5*h + x0 + zq * quad_side ;

        return xyz ; 
    } ; 

    // for i=j=k=0, q will pick out corners of a coarse cube [-1.0, 0.0]^3, centred at {-0.5,-0.5,-0.5}
    // for i=j=k=8 (since for the corner staggering we have n+1 points in each direction),
    // the corners of the coarse cube will become [0,1.0]^3 and the cube itself is centred at {0.5,0.5,0.5}

    for( int i=ngz; i<nx+1+ngz; ++i) {
        for( int j=ngz; j<ny+1+ngz; ++j) {
            for( int k=ngz; k<nz+1+ngz; ++k){
                for( int q=0; q<8; ++q) {
                    auto xyz = get_fine_coords_corner(i,j,k,q) ; 
                    auto yval = lin_func(xyz) ; 
                    // if(i==ngz && j==ngz && k==ngz){
                    //     std::cout << i << j << k << std::endl;
                    //     std::cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << std::endl;
                    // }
                    // if(i==ngz+1 && j==ngz+1 && k==ngz+1){
                    //     std::cout << i << j << k << std::endl;
                    //     std::cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << std::endl;
                    // }
                    CHECK_THAT(
                        h_corner_staggered(i,j,k,q),
                        Catch::Matchers::WithinAbs( yval, 1e-10)
                    ) ; 
                }
            }
        }
    }


    std::cout << 'now edges' << std::endl;
    /* Check for edges: */
    // create mirror view
    auto h_edge_staggered = create_mirror_view(edge_staggered_fine) ;

    deep_copy(h_edge_staggered,edge_staggered_fine) ; 

    auto get_fine_coords_edge = [&] (int i, int j, int k, int q) {
        //first, get coarse edge location:
        std::array<double,3> xyz ; 
        xyz[0]=   x0 + (i-ngz + delta(0,edgeDir)*0.5) * h ; 
        xyz[1]=   x0 + (j-ngz + delta(1,edgeDir)*0.5) * h ;
        xyz[2]=   x0 + (k-ngz + delta(2,edgeDir)*0.5) * h ;
        // xyz is now identical to the coarse edge coordinates 
        // we now have to offset wrt. to the coarse edge, marked by (o)
        // convention is binary: 0  at the i-th location means minus 1/2 d, 1 means plus 1/2 in that direction
        // e.g. lower (+) is q=0 (000)
        //      upper (+) is q=4 (001)
        //      lower (x) is q=1 (100)
        //      upper (x) is q=5 (101)
        //  z  y
        //  | / 
        //  |/___ x
        //
        //       ____________________________
        //      /                           /|
        //     /                           / |
        //    /                           /  |
        //   *==========================*/   |
        //   |             |            |    |          
        //   |             |            |    |
        //   +      c      x     c      |   /|
        //   |             |            |  / |
        //   |             |            | /  |
        //   o=============|============o/   |
        //   |             |            |    |
        //   |             |            |    / 
        //   +      c      x     c      |   / 
        //   |             |            |  / 
        //   |             |            | /
        //   *===========================*
        //
        // therefore (with the help of bit-shifting and ChatGPT...)
        if constexpr(edgeDir==2){
            int xq = (q >> 0) & 1;            // gives 0 and 1  
            int yq = (q >> 1) & 1;            // gives 0 and 1 
            int zq = ((q >> 2) & 1) * 2 - 1;  // gives -1 and +1
            // note the different coefficient in the unstaggered direction!
            xyz[0] += xq * h/2.; 
            xyz[1] += yq * h/2.;
            xyz[2] += zq * h/4.;
        }
        else assert(false); // disregard other staggerings 
    
        return xyz ; 
    } ; 

    for( int i=ngz; i<nx+1-delta(edgeDir,0)+ngz; ++i) {
        for( int j=ngz; j<ny+1-delta(edgeDir,1)+ngz; ++j) {
            for( int k=ngz; k<nz+1-delta(edgeDir,2)+ngz; ++k){
                auto const& xyz_coarse_edge = coords_edge(i,j,k) ; 
                if(i==ngz && j==ngz && k==ngz){
                    std::cout << "coarse edge:" << std::endl;
                    std::cout << xyz_coarse_edge[0] << " " << xyz_coarse_edge[1] << " " << xyz_coarse_edge[2] << std::endl;
                }

                for( int q=0; q<8; ++q) {
                    auto xyz = get_fine_coords_edge(i,j,k,q) ; 

                    if(i==ngz && j==ngz && k==ngz){
                        //std::cout << i << j << k << std::endl;
                            //     int const ichild = math::floor_int((2*i)/nx) 
                            //     + 2 * ( math::floor_int((2*j)/ny) + 2 * math::floor_int((2*k)/nz) ) ;
                            // int i_f = (2*i)%nx + ngz ; 
                            // int j_f = (2*j)%ny + ngz ; 
                            // int k_f = (2*k)%nz + ngz ; 
                        // std::cout << "Child:" << ichild << "(if,jf,kf)=" << "(" << i_f << j_f << k_f << ")" <<std::endl;
                        std::cout << q << std::endl;
                        std::cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << std::endl;
                        std::cout << "fine_edge_val:" << h_edge_staggered(i,j,k,q) << std::endl;
                    }
                    // if(i==ngz+1 && j==ngz+1 && k==ngz+1){
                    //     std::cout << i << j << k << std::endl;
                    //     std::cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << std::endl;
                    // }
                    auto yval = lin_func(xyz) ; 

                    // let's just check the interpolation along the stagger direction:
                     if(q==0){
                        // CHECK_THAT(
                        //     h_edge_staggered(i,j,k,q),
                        //     Catch::Matchers::WithinAbs( yval, 1e-10)
                        // ) ; 
                        if(std::abs(h_edge_staggered(i,j,k,q)-yval)>1e-8){
                            std::cout << i << " " << j << " " << k << std::endl;
                            std::cout << xyz_coarse_edge << std::endl;
                            std::cout << xyz << std::endl;
                            
                        }
                      }
                    // many h_edge_staggered are 0, why?
                    
                   // std::cout << h_edge_staggered(i,j,k,q) << " " << yval << std::endl;
                }
            }
        }
    }









}