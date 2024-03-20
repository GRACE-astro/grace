#include <catch2/catch_test_macros.hpp>
#include <thunder/utils/interpolators.hh>
#include <thunder/data_structures/memory_defaults.hh>
#include <Kokkos_Core.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#define N 50 

TEST_CASE("linterp_nd", "[linterp_nd]")
{
    using namespace thunder ; 
    SECTION(
    "oned"
    ) {
    
    Kokkos::View<double *, default_space>
        x("coordinates", N);
    
    Kokkos::View<double *, default_space>
        y("values", N),  ym("interp_values",N-1);

    auto z = KOKKOS_LAMBDA (double& x)
    {
        return 2.5*x + 3.7 ; 
    } ; 

    double xl{0}, xu{1}; 
    double dx{(xu-xl)/N} ; 

    Kokkos::parallel_for("fill_arrays"
    , Kokkos::RangePolicy<default_execution_space>(0,N) 
    , KOKKOS_LAMBDA (const unsigned int i)
    {
        x(i)  = xl + dx*i;
        y(i)  = z(x(i)) ; 
    }) ; 

    Kokkos::parallel_for("fill_arrays"
    , Kokkos::RangePolicy<default_execution_space>(0,N-1) 
    , KOKKOS_LAMBDA (const unsigned int i)
    {
        double xx[2] ; 
        double yy[2] ; 
        int xp[2]    ; 
        utils::linear_interp_t<1>::get_parametric_coordinates(xp); 
        for( int is=0; is<2; ++is){
            xx[is] = x(i + xp[is])  ;
            yy[is] = y(i + xp[is])  ; 
        }
        auto interpolator = utils::linear_interp_t<1>(xx,yy) ; 
        ym(i) = interpolator.interpolate((x(i+1)+x(i))*0.5) ;  
    }) ;

    auto const h_result = Kokkos::create_mirror_view(ym) ;
    Kokkos::deep_copy(h_result, ym) ; 
    auto const z_host = [&] ( double x ) {
        return 2.5*x + 3.7 ; 
    } ; 
    for( int i=0; i<N-1; ++i){
        REQUIRE_THAT( h_result(i), 
            Catch::Matchers::WithinAbs(z_host(xl + (i + 0.5) * dx), 1e-03)) ; 
    }
    }
    SECTION("twod")
    {
    Kokkos::View<double ***, default_space>
        x("coordinates", N,N,2);
    
    Kokkos::View<double **, default_space>
        y("values", N,N),  ym("interp_values",(N-1),(N-1));
    
    auto z = KOKKOS_LAMBDA (double& x, double& y)
    {
        return 2.5*x + 4.2*y + 3.7 ; 
    } ; 

    double xl{0}, xu{1}; 
    double dx{(xu-xl)/N} ; 

    Kokkos::parallel_for("fill_arrays"
    , Kokkos::RangePolicy<default_execution_space>(0,N*N) 
    , KOKKOS_LAMBDA (const unsigned int idx)
    {
        int i = idx % N ; 
        int j = idx / N ; 
        x(i,j,0)  = xl + dx*i;
        x(i,j,1)  = xl + dx*j;
        y(i,j)  = z(x(i,j,0),x(i,j,1)) ; 
    }) ; 
    Kokkos::parallel_for("fill_arrays"
    , Kokkos::RangePolicy<default_execution_space>(0,(N-1)*(N-1)) 
    , KOKKOS_LAMBDA (const unsigned int idx)
    {
        int i = idx % (N-1) ; 
        int j = idx / (N-1) ;
        double xx[2*4] ; 
        double yy[4]   ;
        int xp[2*4]    ; 
        size_t stencil = utils::linear_interp_t<2>::stencil_size; 
        utils::linear_interp_t<2>::get_parametric_coordinates(xp) ; 
        for( int is=0; is< 4; ++is){
            xx[2*is + 0] = x(i + xp[2*is+0],j + xp[2*is+1],0);
            xx[2*is + 1] = x(i + xp[2*is+0],j + xp[2*is+1],1);
            yy[is] = y(i + xp[2*is+0],j + xp[2*is+1])        ; 
        }
        
        auto interpolator = utils::linear_interp_t<2>(xx,yy) ; 
        ym(i,j) = interpolator.interpolate( (x(i+1,j,0)+x(i,j,0))*0.5
                                          , (x(i,j+1,1)+x(i,j,1))*0.5 ) ;  
    }) ;

    auto h_result = Kokkos::create_mirror_view(ym) ;
    Kokkos::deep_copy(h_result, ym) ; 
    
    auto const z_host = [&] ( double x, double y ) {
        return 2.5*x + 4.2*y + 3.7 ; 
    } ; 
    for( int idx=0; idx<(N-1)*(N-1); ++idx){
        int i = idx % (N-1) ; 
        int j = idx / (N-1) ;
        REQUIRE_THAT( h_result(i,j),  
            Catch::Matchers::WithinAbs(
                  z_host(xl + (i + 0.5) * dx, xl + (j+0.5)*dx)
                , 1e-03)) ; 
    }
    
    }
    SECTION("threed")
    {
    Kokkos::View<double ****, default_space>
        x("coordinates", N,N,N,3);
    
    Kokkos::View<double ***, default_space>
        y("values", N,N,N),  ym("interp_values",(N-1),(N-1),(N-1));
    
    auto z = KOKKOS_LAMBDA (double& x, double& y, double& z)
    {
        return 2.5*x + 4.2*y - 5.1*z + 3.7 ; 
    } ; 

    double xl{0}, xu{1}; 
    double dx{(xu-xl)/N} ; 

    Kokkos::parallel_for("fill_arrays"
    , Kokkos::RangePolicy<default_execution_space>(0,N*N*N) 
    , KOKKOS_LAMBDA (const unsigned int idx)
    {
        int i = idx % N ; 
        int j = (idx / N) % N ; 
        int k = (idx / N) / N ;
        x(i,j,k,0)  = xl + dx*i;
        x(i,j,k,1)  = xl + dx*j;
        x(i,j,k,2)  = xl + dx*k;
        y(i,j,k)  = z(x(i,j,k,0),x(i,j,k,1),x(i,j,k,2)) ; 
    }) ; 
    Kokkos::parallel_for("fill_arrays"
    , Kokkos::RangePolicy<default_execution_space>(0,(N-1)*(N-1)*(N-1)) 
    , KOKKOS_LAMBDA (const unsigned int idx)
    {
        int i = idx % (N-1) ; 
        int j = (idx / (N-1)) % (N-1) ;
        int k = (idx / (N-1)) / (N-1) ;
        size_t constexpr stencil = utils::linear_interp_t<3>::stencil_size; 
        size_t constexpr npoints = stencil*stencil*stencil ;
        double xx[3*npoints] ; 
        double yy[npoints]   ;
        int xp[3*npoints]    ; 
        utils::linear_interp_t<3>::get_parametric_coordinates(xp) ; 
        for( int is=0; is< npoints; ++is){
            xx[3*is + 0] = x(i + xp[3*is+0],j + xp[3*is+1], k + xp[3*is+2], 0);
            xx[3*is + 1] = x(i + xp[3*is+0],j + xp[3*is+1], k + xp[3*is+2], 1);
            xx[3*is + 2] = x(i + xp[3*is+0],j + xp[3*is+1], k + xp[3*is+2], 2);
            yy[is] = y(i + xp[3*is+0],j + xp[3*is+1], k + xp[3*is+2])        ; 
        }
        
        auto interpolator = utils::linear_interp_t<3>(xx,yy) ; 
        ym(i,j,k) = interpolator.interpolate( (x(i+1,j,k,0)+x(i,j,k,0))*0.5
                                            , (x(i,j+1,k,1)+x(i,j,k,1))*0.5
                                            , (x(i,j,k+1,2)+x(i,j,k,2))*0.5 ) ;  
    }) ;
    auto h_result = Kokkos::create_mirror_view(ym) ;
    Kokkos::deep_copy(h_result, ym) ; 
    
    auto const z_host = [&] ( double x, double y, double z ) {
        return 2.5*x + 4.2*y - 5.1*z + 3.7  ; 
    } ; 
    for( int idx=0; idx<(N-1)*(N-1)*(N-1); ++idx){
        int i = idx % (N-1) ; 
        int j = (idx / (N-1)) % (N-1) ;
        int k = (idx / (N-1)) / (N-1) ;
        REQUIRE_THAT( h_result(i,j,k),  
            Catch::Matchers::WithinAbs(
                  z_host(xl + (i + 0.5) * dx, xl + (j+0.5)*dx, xl + (k+0.5)*dx)
                , 1e-03)) ; 
    }
    
    }
}
