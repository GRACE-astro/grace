#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/amr/amr_ghosts.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/gridloop.hh>
#include <grace/evolution/refluxing.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <grace/IO/cell_output.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>
#include <fstream>
#include <string>
#include <string>
#include <utility>
#include <stdexcept>

#include <grace/IO/spherical_surfaces.hh>

TEST_CASE("Test_EMF_REFLUX", "[refluxing]")
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    Kokkos::fence() ; 

    double r = 1.0;
    std::string name{"pippo"} ; 
    std::array<double,3> c{0,0,0} ; 
    size_t npt = 32 ; 
    // create a spherical surface 
    auto surf = std::make_unique<spherical_surface_t<uniform_sampler_t,no_tracking_policy_t,3>>(
                spherical_surface_t<uniform_sampler_t,no_tracking_policy_t,3>(name,r,c,npt)
            ); 
    auto& state = grace::variable_list::get().getstate() ; 
    auto state_h = create_mirror_view(state) ; 
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            state_h(i,j,k,0,q) = 1.0  ; 
        }, {false,false,false}, true 
    ) ; 
    Kokkos::deep_copy(state,state_h) ;

    Kokkos::View<double**> interp("test",2048,1);
    interpolate_on_sphere(*surf, std::vector<int>{0}, std::vector<int>{}, interp) ; 
    auto iv = Kokkos::create_mirror_view(interp) ; 
    Kokkos::deep_copy(iv,interp) ; 

    auto npoints = surf->intersecting_points_h.size() ; 
    double resL = 0 ; 
    for( int i=0; i<npoints; ++i) {
        auto ip = surf->intersecting_points_h[i] ; 
        double domega = surf->weights_h[ip] ; 
        GRACE_VERBOSE("Iv {} domega {}", iv(i,0), domega) ; 
        resL += domega * iv(i,0) ; 
    }

    double res ; 
    parallel::mpi_allreduce(
        &resL,
        &res,
        1,
        MPI_SUM
    ) ; 


    REQUIRE( fabs(res-4*M_PI*r*r) < 1e-15) ; 

}