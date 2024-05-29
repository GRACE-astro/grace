#include <catch2/catch_test_macros.hpp>
#include <grace/amr/connectivity.hh>
#include <cassert>
#include <iostream>
TEST_CASE("connectivities", "[connectivities]")
{
    using namespace grace ; 
    auto& conn = amr::connectivity::get() ; 
    REQUIRE( conn.is_valid() ) ; 
    
}