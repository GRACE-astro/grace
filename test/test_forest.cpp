#include <catch2/catch_test_macros.hpp>
#include <grace/amr/forest.hh>
#include <iostream>

TEST_CASE("amr_forest", "[amr_forest]")
{
    using namespace grace ;
    auto& forest = amr::forest::get() ; 
    REQUIRE( forest.get() != nullptr ) ; 
}
