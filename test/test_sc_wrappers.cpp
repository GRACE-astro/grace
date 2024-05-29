#include <catch2/catch_test_macros.hpp>
#include <grace/utils/sc_wrappers.hh>
#include <iostream>

TEST_CASE("sc_array_view_t", "[sc_array_view]") 
{
    using namespace grace ; 

    sc_array_t * _arr = sc_array_new_count(sizeof(int), 100) ; 
    sc_array_view_t<int> view(_arr) ; 

    // test range based for 
    for( auto & x: view) x = 163; 
    // test size() 
    REQUIRE( view.size() == 100 ) ; 
    // check normal looping and accessing 
    for( int i=0; i<view.size(); ++i){
        REQUIRE( view[i] == 163 ) ; 
    }
    // check constant range based for loop 
    for( auto const& x: view){
        // x = 2 ; // Does not compile (rightfully)
        REQUIRE( x == 163 ) ;
    }
    sc_array_destroy(_arr) ; 
}