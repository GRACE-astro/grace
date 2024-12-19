/**
 * @file test_regridding_edge.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-12-13
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/cell_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/numerics/gridloop.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#define DBG_EDGE_REGRID_TEST

TEST_CASE("Simple edge regrid", "[edge regrid]")
{
    using namespace grace::variables ; 
    using namespace grace ; 
    #if defined(GRACE_ENABLE_BURGERS) or defined(GRACE_ENABLE_SCALAR_ADV)
    int const DENS = U ; 
    int const DENS_ = U ; 
    auto params = grace::config_parser::get()["amr"] ; 
    params["refinement_criterion_var"] = "U" ; 
    #else
    auto params = grace::config_parser::get()["amr"] ; 
    params["refinement_criterion_var"] = "dens" ; 
    #endif

    auto const interp_order = grace::get_param<uint32_t>("amr","prolongation_order") ; 

    DECLARE_GRID_EXTENTS ; 

    /*************************************************/
    /*                Fetch arrays                   */
    /*************************************************/
    auto& state  = grace::variable_list::get().getstate()  ;
    auto& sstate  = grace::variable_list::get().getstaggeredstate()  ;
    auto& coord_system = grace::coordinate_system::get() ;
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    auto h_corner_mirror = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 

    // hard-coded for now 
    constexpr const size_t edgeDir=2; 
    static_assert(edgeDir==2); // do not change this until the example is adapted for other staggerings
    
    using MirrorViewType = decltype(Kokkos::create_mirror_view(sstate.edge_staggered_fields_xy));
    MirrorViewType h_edge_mirror;  // Declare h_edge_mirror with the appropriate type.
    h_edge_mirror = Kokkos::create_mirror_view(sstate.edge_staggered_fields_xy);  // change when other directions are added
   

    /*************************************************/
    /*            Define filling func                */
    /*************************************************/
    auto const h_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(8.5 * x, - 5.1 * y, -2*z) - 3.14 ; 
    } ; 
    auto const h_corner_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        if( interp_order == 2 ) {
            return EXPR(8.5 * x, - 5.1 * y, -2*z) - 3.14 ; 
        } else if ( interp_order == 4) {
            #ifdef GRACE_3D
            return 0.09645987612683005 + 0.9689256995609989*x + 0.9280564240107632*y - 0.27263220791463016*x*y + 1.6557688148274297*z - 1.8293477262261941*x*z + 
   1.8321409249345644*y*z - 0.6168312325224381*x*y*z + 1.7146635117999285*pow(x,2) - 0.8323622181656987*y*pow(x,2) - 1.1983364369285372*z*pow(x,2) + 
   0.11344784791220963*pow(x,3) + 1.46241660443817*pow(y,2) + 1.9071800878186975*x*pow(y,2) + 1.7912912453890968*z*pow(y,2) - 0.37430580597888685*pow(y,3) - 
   0.07020440743423961*pow(z,2) + 1.0902200536627111*x*pow(z,2) + 1.2434145608397085*y*pow(z,2) + 0.6321621456866486*pow(z,3) ; 
            #else 
            return 1.0354333039152808 + 1.6630034246569636*x + 1.3491577540970425*y - 1.6695252008930153*x*y - 1.205160193337056*pow(x,2) - 1.6913180599507545*y*pow(x,2) - 
   0.4452976970948681*pow(x,3) + 0.3512878541919209*pow(y,2) + 0.17773874176068194*x*pow(y,2) - 0.7151254966832106*pow(y,3) ; 
            #endif 
        } else {
            return - 1.; 
        }
    } ; 

    auto const h_edge_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        if( interp_order == 2 ) {
            return EXPR(8.5 * x, - 5.1 * y, -2*z) - 3.14 ; 
        } else {
            GRACE_INFO("interp_order != 2 unavailable for edge stagerred variables");
            return - 1.; 
        }
    } ; 
    /*************************************************/
    /*                   fill data                   */
    /*     here we fill the ghost zones as well.     */
    /*************************************************/
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                true
            ) ;
            h_state_mirror(VEC(i,j,k),DENS,q) = 
                h_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ;
        },
        {VEC(false,false,false)},
        true
    ) ; 
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                {VEC(0,0,0)}, 
                true
            ) ;
            h_corner_mirror(VEC(i,j,k),DENS,q) = 
                h_corner_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ;
        },
        {VEC(true,true,true)},
        true
    ) ; 
    using utils::delta;
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                {VEC(0.5*delta(0,edgeDir),0.5*delta(1,edgeDir),0.5*delta(2,edgeDir))}, 
                true
            ) ;
            h_edge_mirror(VEC(i,j,k),DENS,q) = 
                h_edge_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ;
        },
        {VEC(edgeDir!=0, edgeDir!=1, edgeDir!=2)}, // for edgeDir=2, this gives staggering=(true,true,false)
        true
    ) ; 
    // Since the get_physical_coordinates (the version with three arguments) used for cell-centred values
    // employs the {VEC(0.5,0.5,0.5)} as the third argument of the 4-argument version of the same routine,
    // it seems natural to assume that this routine here will have the 
    // VEC(0,0,0.5) for the z-edge (staggered_xy) variables 
    /*************************************************/
    /*                 Copy H2D                      */
    /*************************************************/
    Kokkos::deep_copy(state,h_state_mirror); 
    Kokkos::deep_copy(sstate.corner_staggered_fields,h_corner_mirror); 
    Kokkos::deep_copy(sstate.edge_staggered_fields_xy,h_edge_mirror); 
    /*************************************************/
    /*                   Regrid                      */
    /*************************************************/
    #ifdef DBG_REGRID_TEST
    /*write output and regrid*/
    grace::IO::write_cell_output(true,true,true) ; 
    grace::amr::regrid() ;  
    grace::runtime::get().increment_iteration() ; 
    grace::runtime::get().set_timestep(1) ; 
    grace::runtime::get().increment_time(); 
    grace::IO::write_cell_output(true,true,true) ; 
    #else
    grace::amr::regrid() ;
    #endif 
    /*************************************************/
    /*                 Copy D2H                      */
    /*************************************************/
    auto h_state_mirror_new = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror_new,state); 
    auto h_corner_mirror_new = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 
    Kokkos::deep_copy(h_corner_mirror_new,sstate.corner_staggered_fields); 
    auto h_edge_mirror_new = Kokkos::create_mirror_view(sstate.edge_staggered_fields_xy) ; 
    Kokkos::deep_copy(h_edge_mirror_new,sstate.edge_staggered_fields_xy); 
    /*************************************************/
    /*                   Check                       */
    /*************************************************/
    host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                true
            ) ;
            CHECK_THAT( h_state_mirror_new(VEC(i,j,k),DENS,q)
                      , Catch::Matchers::WithinAbs(
                                  h_func(VEC(pcoords[0],pcoords[1],pcoords[2]))
                                , 1e-12 )) ;
        },
        {VEC(false,false,false)},
        false
    ) ; 
    host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                {VEC(0,0,0)},
                true
            ) ;
            CHECK_THAT( h_corner_mirror_new(VEC(i,j,k),DENS,q)
                      , Catch::Matchers::WithinAbs(
                                  h_corner_func(VEC(pcoords[0],pcoords[1],pcoords[2]))
                                , 1e-12 )) ;
        },
        {VEC(true,true,true)},
        false
    ) ; 
    host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                {VEC(0.5*delta(0,edgeDir),0.5*delta(1,edgeDir),0.5*delta(2,edgeDir))}, 
                true
            ) ;
            CHECK_THAT( h_edge_mirror_new(VEC(i,j,k),DENS,q)
                      , Catch::Matchers::WithinAbs(
                                  h_edge_func(VEC(pcoords[0],pcoords[1],pcoords[2]))
                                , 1e-12 )) ;
        },
        {VEC(edgeDir!=0, edgeDir!=1, edgeDir!=2)},
        false
    ) ; 
}