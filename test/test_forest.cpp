#include <catch2/catch_test_macros.hpp>
#include <thunder/amr/forest.hh>
#include <iostream>

TEST_CASE("amr_forest", "[amr_forest]")
{
    using namespace thunder ;
    auto& forest = amr::forest::get() ; 
    REQUIRE( forest.get() != nullptr ) ; 
}
