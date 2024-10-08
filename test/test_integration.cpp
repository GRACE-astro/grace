#include <catch2/catch_test_macros.hpp>
#include <grace/utils/numerics/integration.hh>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>

TEST_CASE("integrate_nd", "[integrate_nd]")
{
    using namespace grace ; 
    auto const f_oned = [&] (double x) {
        return sin(x) ; 
    } ; 

    auto const int_f_oned = utils::nd_quadrature_integrate<1,5,double>(
        {-M_PI}, {M_PI}, f_oned
    ) ;
    REQUIRE_THAT( int_f_oned,  
                Catch::Matchers::WithinAbs(
                  0.0
                , 1e-3)) ;

    auto const f_twod = [&] (double x, double y) {
        return x*x*x + y; 
    } ; 

    auto const int_f_twod = utils::nd_quadrature_integrate<2,4,double>(
        {-2., -0.5}, {1., 2.}, f_twod
    ) ;
    REQUIRE_THAT( int_f_twod,  
                Catch::Matchers::WithinRel(
                  1.16770632690572
                , 1e-3)) ;

    auto const f_threed = [&] (double x, double y, double z) {
        return sin(x) * cos(y) * sin(z*x*x); 
    } ; 

    double const int_f_threed = utils::nd_quadrature_integrate<3,10,double>(
        {-2., -M_PI/2.,0.}, {M_PI, M_PI/2,2.},  f_threed
    ) ;

    REQUIRE_THAT( int_f_threed,  
                Catch::Matchers::WithinRel(
                  0.266409344233000
                , 1e-3)) ;
    
}