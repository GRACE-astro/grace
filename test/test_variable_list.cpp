#include <catch2/catch_test_macros.hpp>
#include <thunder/utils/sc_wrappers.hh>
#include <thunder_config.h>
#include <thunder/data_structures/variables.hh>
#include <iostream>
#include <type_traits>
#include <thunder/utils/type_name.hh> 

TEST_CASE("variable_list", "[variable_list]")
{
    using namespace thunder ; 
    auto& vars = variable_list::get() ;
    std::cout << utils::type_name<var_array_t<THUNDER_NSPACEDIM>>() << std::endl  ; 
    
    auto state = vars.getstate() ;  
     
    std::cout << state.label() << std::endl ; 
    int rank = THUNDER_NSPACEDIM + 2 ;
    int nvars_evolved = variables::detail::num_evolved ; 
    REQUIRE( state.extent(rank-2) == nvars_evolved ) ; 
} 