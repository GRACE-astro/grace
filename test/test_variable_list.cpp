#include <catch2/catch_test_macros.hpp>
#include <grace/utils/sc_wrappers.hh>
#include <grace_config.h>
#include <grace/data_structures/variables.hh>
#include <iostream>
#include <type_traits>
#include <grace/utils/type_name.hh> 

TEST_CASE("variable_list", "[variable_list]")
{
    using namespace grace ; 
    auto& vars = variable_list::get() ;
    std::cout << utils::type_name<var_array_t<GRACE_NSPACEDIM>>() << std::endl  ; 
    
    auto state = vars.getstate() ;  
     
    std::cout << state.label() << std::endl ; 
    int rank = GRACE_NSPACEDIM + 2 ;
    int nvars_evolved = variables::detail::num_evolved ; 
    REQUIRE( state.extent(rank-2) == nvars_evolved ) ; 
} 