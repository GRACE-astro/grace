/**
 * @file test_device_vector.cc
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-18
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

#include <grace_config.hh>
#include <grace/utils/device_vector.hh>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>


TEST_CASE("device_vector","[devicevector]")
{
    using namespace grace ; 
    using namespace Kokkos ; 

    // Test default ctor 
    {
        device_vector<int> dvec ; 
        REQUIRE( dvec.size() == 0 ) ;
        REQUIRE( dvec.device_size() == 0) ; 

    }
    // Test constructor with size 
    {
        device_vector<int> dvec(2) ; 
        REQUIRE( dvec.size() == 2 ) ;
        REQUIRE( dvec.device_size() == 2 ) ; 
    }
    // Test constructor with size and value
    {
        device_vector<int> dvec(2, 1) ; 
        REQUIRE( dvec.size() == 2 ) ;
        REQUIRE( dvec.device_size() == 2 ) ; 
        for( int ii=0; ii<2; ++ii ) {
            REQUIRE(dvec[ii] == 1) ; 
        }
        // and test also iterator based for loop
        for( auto const& x: dvec ) {
            REQUIRE(x == 1 ) ; 
        }
        // check device view, as well 
        auto hview = create_mirror_view(dvec.d_view) ; 
        deep_copy(hview,dvec.d_view) ; 
        for( int ii=0; ii<2; ++ii) {
            REQUIRE(hview(ii) == 1) ; 
        }
    }
    // Test ctor from std::vector 
    {
        std::vector<int> v {1,2,3} ; 
        device_vector<int> dvec{v} ; 
        REQUIRE( dvec.size() == 3 ) ;
        REQUIRE( dvec.device_size() == 3 ) ;
        for( int ii=0; ii<dvec.size(); ++ii ) {
            REQUIRE(dvec[ii] == v[ii]) ; 
        }
    }
    // Test move ctor 
    {
        std::vector<int> v {1,2,3} ;
        std::vector<int> w{v} 
        device_vector<int> dvec{std::move(w)} ; 
        REQUIRE( dvec.size() == 3 ) ;
        REQUIRE( dvec.device_size() == 3 ) ;
        for( int ii=0; ii<dvec.size(); ++ii ) {
            REQUIRE(dvec[ii] == v[ii]) ; 
        }
    }
    // Test push_back
    {
        device_vector<int> dvec ; 
        for( int ii=0; ii<10; ++ii) {
            dvec.push_back(ii) ; 
        }
        dvec.host_to_device() ; 
        REQUIRE(dvec.size() == 10);
        REQUIRE(dvec.device_size() == 10) ; 
        auto hview = create_mirror_view(dvec.d_view) ; 
        deep_copy(hview,dvec.d_view) ; 
        for( int ii=0; ii<10; ++ii) {
            REQUIRE(hview(ii) == ii) ;
            REQUIRE(dvec[ii] == ii)  ;  
        }
    }
    
    
}