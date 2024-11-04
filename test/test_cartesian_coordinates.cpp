/**
 * @file test_cartesian_coordinates.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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
#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/scalar_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/system/grace_system.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>


#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>

TEST_CASE("cart_coords")
{
    using namespace grace  ; 
    using namespace Kokkos ;

    DECLARE_GRID_EXTENTS ;  

    coord_array_t<GRACE_NSPACEDIM> pcoords_cc; // Centered 
    coord_array_t<GRACE_NSPACEDIM> pcoords_cv; // Staggered 

    fill_physical_coordinates(pcoords_cc, {false,false,false} ) ; 
    fill_physical_coordinates(pcoords_cv, {true,true,true} ) ; 


    REQUIRE( pcoords_cc.extent(0) == (nx+2*ngz) ) ; 

    REQUIRE( pcoords_cv.extent(0) == (nx+1+2*ngz) ) ; 

    auto& forest = grace::amr::forest::get() ; 
    int ntrees = (forest.last_local_tree() - forest.first_local_tree() + 1) ; 

    GRACE_INFO("ntrees: {}", ntrees) ; 

    for( int itree = forest.first_local_tree(); itree <= forest.last_local_tree(); ++itree ) 
    {
        auto d = grace::amr::get_tree_spacing(itree)   ; 
        auto tree = forest.tree(itree) ; 

        for( int ivert = 0; ivert<=7; ++ivert){
           auto v = grace::amr::get_tree_vertex(itree, ivert) ; 
           //auto v1 = grace::amr::get_tree_vertex(itree, 1) ; 
           //auto d = grace::amr::get_tree_spacing(itree)   ; 
           //auto tree = forest.tree(itree) ; 
           //GRACE_INFO("itree {} tree coordinates [ {}, {}, {} ]", itree, v[0],v[1],v[2]) ; 
           //GRACE_INFO("itree {} tree coordinates [ {}, {}, {} ]", itree, v1[0],v1[1],v1[2]) ; 
           GRACE_INFO("itree {} at ivert {} with tree coordinates [ {}, {}, {} ]", itree, ivert, v[0],v[1],v[2]) ; 
        }
        GRACE_INFO("        tree spacing     [ {}, {}, {} ]", d[0], d[1],d[2] ) ; 
        GRACE_INFO("        num quadrats      {}", tree.num_quadrants() ) ;
        GRACE_INFO("        tree offset       {}", tree.quadrants_offset() ) ;
    }

    auto h_coords = Kokkos::create_mirror_view(pcoords_cc) ;

    auto& coordsys = grace::coordinate_system::get() ; 
    
    for( int iq=0; iq<nq; ++iq) {
        if(iq == 8 || iq==16){
            GRACE_INFO("new tree-----") ;
        }
        GRACE_INFO("iq {} owner {}", iq, amr::get_quadrant_owner(iq) ) ; 
        for( int i=ngz; i<nx+ngz;i++){
        //auto xyz = coordsys.get_physical_coordinates({VEC(i,0,0)},iq,{VEC(0.5,0.5,0.5)},true) ; 
        //GRACE_INFO("i {}, pcoord 0: {} ",i, xx) ;
        }

    }



    
}
