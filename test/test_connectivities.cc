#include <catch2/catch_test_macros.hpp>
#include <thunder/amr/connectivity.hh>
#include <cassert>
#include <iostream>
TEST_CASE("connectivities", "[connectivities]")
{
    using namespace thunder ; 
    auto& conn = amr::connectivity::get() ; 
    REQUIRE( conn.is_valid() ) ; 
    
}