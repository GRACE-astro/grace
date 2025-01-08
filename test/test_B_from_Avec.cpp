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
#include <grace/data_structures/memory_defaults.hh>
#include <grace/utils/numerics/curl_operators.hh>
#include <grace/utils/numerics/matrix_helpers.tpp>
#include <Kokkos_Core.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


#include <iostream>


TEST_CASE("lagrange_interp_edge", "[lagrange_interp_edge]")
{
    using namespace Kokkos ; 
    using namespace utils  ;
    using namespace grace  ;  
    using utils::delta;

    /******************************************************************/
    /*      COMPUTATION OF THE MAGNETIC FIELD FROM VECTOR POTENTIAL   */
    /******************************************************************/
    /* We will be testing the Cartesian curl operator in flat         */
    /* spacetime. An unphysical (linear) arbitrary pressure profile   */
    /* and a poloidal vector potential form to fill the views are used*/
    /******************************************************************/ 

    auto lin_func_comp = KOKKOS_LAMBDA (std::array<double,3> const& xyz, int const& comp)
    {
        double x = xyz[0] ; double y = xyz[1] ; double z = xyz[2] ; 
        double A_i = comp+1; // arbitrary vector potential
        return A_i*(2.5 * x - 7.1 * y + 8.9 * z); 
    } ; 

    auto d_lin_func_comp = KOKKOS_LAMBDA (std::array<double,3> const& xyz, int const& comp, int const& der_dir)
    {
        double x = (der_dir!=0 ? 0. : 1.);
        double y = (der_dir!=1 ? 0. : 1.);
        double z = (der_dir!=2 ? 0. : 1.); 
        double A_i = comp+1.; // arbitrary vector potential
        return A_i*(2.5 * x - 7.1 * y + 8.9 * z); 
    } ; 

    // double const p_0   = 1.;
    // double const sigma = 1./0.25;
    // // gaussian pressure profile for easy testing
    // auto press_profile = KOKKOS_LAMBDA (std::array<double,3> const& xyz)
    // {
    //     double x = xyz[0] ; double y = xyz[1] ; double z = xyz[2] ; 
    //     double r = Kokkos::sqrt(x*x + y*y + z*z);
    //     return p_0 * Kokkos::exp(-r*r / (2*sigma*sigma)); 
    // } ; 

    // // gaussian derivative
    // auto der_press_profile = KOKKOS_LAMBDA (std::array<double,3> const& xyz, const int& dir)
    // {
    //     return -xyz[idir] / (sigma*sigma) * press_profile(xyz)  ; 
    // } ; 


    int nx{8},ny{8},nz{8}, ngz{4} ; 
    View<double***, default_space> cell_centered(
        "cell_centered_data", nx+2*ngz,ny+2*ngz,nz+2*ngz 
    ) ;
    View<double***, default_space> corner_staggered(
        "corner_staggered_data", nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz 
    ) ;

    // one index fewer for the staggered directions
    View<double***, default_space> edge_staggered_yz( // A_x
        "edge_staggered_yz_data", nx+2*ngz,ny+1+2*ngz,nz+1+2*ngz
    ) ;
    View<double***, default_space> edge_staggered_xz(  // A_y
        "edge_staggered_xz_data", nx+1+2*ngz,ny+2*ngz,nz+1+2*ngz
    ) ;
    View<double***, default_space> edge_staggered_xy(  // A_z
        "edge_staggered_xy_data", nx+1+2*ngz,ny+1+2*ngz,nz+2*ngz
    ) ;

    // one index fewer for the staggered directions
    View<double***, default_space> face_staggered_x( // B^x
        "face_staggered_x_data", nx+1+2*ngz,ny+2*ngz,nz+2*ngz
    ) ;
    View<double***, default_space> face_staggered_y( // B^y
        "face_staggered_y_data", nx+2*ngz,ny+1+2*ngz,nz+2*ngz
    ) ;
    View<double***, default_space> face_staggered_z( // B^z
        "face_staggered_z_data", nx+2*ngz,ny+2*ngz,nz+1+2*ngz
    ) ;


    // one index fewer for the staggered directions
    View<double***, default_space> face_staggered_x_curl( // B^x
        "face_staggered_x_curl_data", nx+1+2*ngz,ny+2*ngz,nz+2*ngz
    ) ;
    View<double***, default_space> face_staggered_y_curl( // B^y
        "face_staggered_y_curl_data", nx+2*ngz,ny+1+2*ngz,nz+2*ngz
    ) ;
    View<double***, default_space> face_staggered_z_curl( // B^z
        "face_staggered_z_curl_data", nx+2*ngz,ny+2*ngz,nz+1+2*ngz
    ) ;


    double const x0{-1}, x1{1} ; 
    double const h { (x1-x0)/nx } ; 
    double const dx{h}, dy{h},dz{h};

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
    auto coords_edge = KOKKOS_LAMBDA (int i, int j, int k, int edgeDir)
    {
        return std::array<double,3> {
            x0 + (i-ngz + delta(0,edgeDir)*0.5) * h , 
            x0 + (j-ngz + delta(1,edgeDir)*0.5) * h ,
            x0 + (k-ngz + delta(2,edgeDir)*0.5) * h 
        } ; 
    } ;

    // exactly like in the corner case, except two directions have a +0.5 (here it's the z coordinate)
    auto coords_face = KOKKOS_LAMBDA (int i, int j, int k, int faceDir)
    {
        // static constexpr int idir = std::get<0>(get_complementary_dirs<edgedir>());
        // static constexpr int jdir = std::get<1>(get_complementary_dirs<edgedir>());
        return std::array<double,3> {
            x0 + (i-ngz + 0.5*(1-delta(0,faceDir))) * h , 
            x0 + (j-ngz + 0.5*(1-delta(1,faceDir))) * h ,
            x0 + (k-ngz + 0.5*(1-delta(2,faceDir))) * h 
        } ; 
    } ;
    // note for each of these we go until nx+ngz () the ghost-zone in the un-staggered direction
    // and hence initialize also either the whole or most of the ghost-zone
    MDRangePolicy<Rank<4>> policy_all_edges {
        {0,0,0,0},
        {nx+2*ngz, ny+2*ngz, nz+2*ngz, 3}
    } ; 
    parallel_for("fill_edges_data",policy_all_edges,
    KOKKOS_LAMBDA( int i, int j, int k, int edge) {
        auto const& xyz_edge = coords_edge(i,j,k, edge) ; 
        if(edge==0) edge_staggered_yz(VEC(i,j,k)) = lin_func_comp(xyz_edge,edge);
        if(edge==1) edge_staggered_xz(VEC(i,j,k)) = lin_func_comp(xyz_edge,edge);
        if(edge==2) edge_staggered_xy(VEC(i,j,k)) = lin_func_comp(xyz_edge,edge);

    }) ; 
    
    // computing without the ghost-zones
    MDRangePolicy<Rank<4>> policy_all_faces {
        {0,0,0,0},
        {nx+2*ngz, ny+2*ngz, nz+2*ngz, 3}
    } ; 
    parallel_for("fill_faces_data",policy_all_faces,
    KOKKOS_LAMBDA( int i, int j, int k, int face) {
        auto const& xyz_face = coords_face(i,j,k, face) ; 
        //const& xyz, int const& comp, int const& der_dir)
        if(face==0) face_staggered_x(VEC(i,j,k)) = d_lin_func_comp(xyz_face,2,1)-d_lin_func_comp(xyz_face,1,2);
        if(face==1) face_staggered_y(VEC(i,j,k)) = d_lin_func_comp(xyz_face,0,2)-d_lin_func_comp(xyz_face,2,0);
        if(face==2) face_staggered_z(VEC(i,j,k)) = d_lin_func_comp(xyz_face,1,0)-d_lin_func_comp(xyz_face,0,1);

    }) ; 

    MDRangePolicy<Rank<3>> policy_compute_curl {
        {0,0,0},
        {nx+2*ngz, ny+2*ngz, nz+2*ngz}
    } ; 
    parallel_for("compute_curl",policy_compute_curl,
    KOKKOS_LAMBDA( int i, int j, int k) {
        using utils::curl_operator_t;

        curl_operator_t::apply(VEC(i,j,k), VEC(dx,dy,dz),
                               VEC(face_staggered_x_curl,face_staggered_y_curl,face_staggered_z_curl),
                               VEC(edge_staggered_yz,edge_staggered_xz,edge_staggered_xy)
                                );
    }) ; 

    auto h_face_staggered_x = create_mirror_view(face_staggered_x) ;
    deep_copy(h_face_staggered_x,face_staggered_x) ; 
    auto h_face_staggered_y = create_mirror_view(face_staggered_y) ;
    deep_copy(h_face_staggered_y,face_staggered_y) ; 
    auto h_face_staggered_z = create_mirror_view(face_staggered_z) ;
    deep_copy(h_face_staggered_z,face_staggered_z) ; 


    auto h_face_staggered_x_curl = create_mirror_view(face_staggered_x_curl) ;
    deep_copy(h_face_staggered_x_curl,face_staggered_x_curl) ; 
    auto h_face_staggered_y_curl = create_mirror_view(face_staggered_y_curl) ;
    deep_copy(h_face_staggered_y_curl,face_staggered_y_curl) ; 
    auto h_face_staggered_z_curl = create_mirror_view(face_staggered_z_curl) ;
    deep_copy(h_face_staggered_z_curl,face_staggered_z_curl) ; 

    // physical points are from ngz till nx+ngz+1 (i.e. i=ngz until i=nx+ngz)
    for( int i=ngz; i<nx+ngz+1; ++i) {
        for( int j=ngz; j<ny+ngz+1; ++j) {
            for( int k=ngz; k<nz+ngz+1; ++k){

                    CHECK_THAT(
                        h_face_staggered_x(VEC(i,j,k)),
                        Catch::Matchers::WithinAbs( h_face_staggered_x_curl(VEC(i,j,k)), 1e-10)
                    ) ; 
                     CHECK_THAT(
                        h_face_staggered_y(VEC(i,j,k)),
                        Catch::Matchers::WithinAbs( h_face_staggered_y_curl(VEC(i,j,k)), 1e-10)
                    ) ; 
                     CHECK_THAT(
                        h_face_staggered_z(VEC(i,j,k)),
                        Catch::Matchers::WithinAbs( h_face_staggered_z_curl(VEC(i,j,k)), 1e-10)
                    ) ; 
                }
            }
        }
    // }









}