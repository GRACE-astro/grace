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

TEST_CASE("TEST_INTERP", "[interpolation]")
{
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    Kokkos::fence() ; 
    GRACE_VERBOSE("Hello!") ; 
    auto& coord_system = grace::coordinate_system::get() ; 

    double r = 1.0;
    std::string name{"pippo"} ; 
    std::array<double,3> c{0,0,0} ; 
    size_t npt = 32 ; 
    // create a spherical surface 
    auto surf = std::make_unique<spherical_surface_t<uniform_sampler_t,no_tracking_policy_t>>(
                spherical_surface_t<uniform_sampler_t,no_tracking_policy_t>(name,r,c,npt)
            ); 
    auto& state = grace::variable_list::get().getstate() ; 
    auto state_h = create_mirror_view(state) ; 
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            std::array<double,3> lcoord {0.5,0.5,0.5} ; 
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)}, q, lcoord, true 
            ) ; 
            state_h(i,j,k,0,q) = SQR(pcoords[0])*pcoords[0] + SQR(pcoords[0])*pcoords[1] + 2*SQR(pcoords[2])*pcoords[2]  ; 
        }, {false,false,false}, true 
    ) ; 
    Kokkos::deep_copy(state,state_h) ;

    Kokkos::View<double**, grace::default_space> interp("test",0,0), interp_aux{};
    interpolate_on_sphere(*surf, std::vector<int>{0}, std::vector<int>{}, interp, interp_aux) ; 
    GRACE_VERBOSE("Size {} {}", interp.extent(0), interp.extent(1)) ; 
    auto iv = Kokkos::create_mirror_view(interp) ; 
    Kokkos::deep_copy(iv,interp) ; 
    
    auto npoints = surf->intersecting_points_h.size() ; 
    double resL = 0 ; 

    for( int i=0; i<npoints; ++i) {
        GRACE_VERBOSE("Iv {}", iv(i,0)) ;
    }

    for( int i=0; i<npoints; ++i) {
        auto ip = surf->intersecting_points_h[i] ; 
        double domega = surf->weights_h[ip] ; 
         
        auto& pcoords = surf->points_h[ip].second ; 
        double const val = SQR(pcoords[0])*pcoords[0] + SQR(pcoords[0])*pcoords[1] + 2*SQR(pcoords[2])*pcoords[2]  ; 
        
        REQUIRE(fabs(iv(i,0)-val)<1e-12) ; 
    }
    GRACE_TRACE("Done") ; 
}