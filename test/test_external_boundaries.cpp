#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/numerics/gridloop.hh>

#include <grace/IO/cell_output.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>

TEST_CASE("External boundaries", "[external_boundaries]") 
{
    using namespace grace::variables ; 
    using namespace grace ; 

    DECLARE_GRID_EXTENTS ; 

    /*************************************************/
    /*                Fetch arrays                   */
    /*************************************************/
    auto& state  = grace::variable_list::get().getstate()  ;
    auto& sstate  = grace::variable_list::get().getstaggeredstate()  ;
    auto& coord_system = grace::coordinate_system::get() ;
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    auto h_corner_mirror = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 
    /*************************************************/
    /*            Define filling func                */
    /*************************************************/
    auto const h_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        #if 1
        return EXPR(8.5 * x, - 5.1 * y, + 2*z) - 3.14 ; 
        #else 
        auto const r2 = x*x+y*y+z*z ; 
        return 1.-Kokkos::fabs(x) ; 
        #endif 
    } ;
    /*************************************************/
    /*                   fill data                   */
    /*     here we fill the ghost zones as well.     */
    /*************************************************/
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                true
            ) ;
            h_state_mirror(VEC(i,j,k),DENS,q) = 
                h_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ;
        },
        {VEC(false,false,false)},
        true
    ) ;
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoords = coord_system.get_physical_coordinates(
                {VEC(i,j,k)},
                q,
                {VEC(0,0,0)}, 
                true
            ) ;
            h_corner_mirror(VEC(i,j,k),DENS,q) = 
                h_corner_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ;
        },
        {VEC(true,true,true)},
        true
    ) ;
    /*************************************************/
    /*                 Copy H2D                      */
    /*************************************************/
    Kokkos::deep_copy(state,h_state_mirror); 
    Kokkos::deep_copy(sstate.corner_staggered_fields,h_corner_mirror); 
    auto& swap = grace::variable_list::get().getscratch() ; 
    auto& sswap = grace::variable_list::get().getstaggeredscratch() ; 
    Kokkos::deep_copy(swap, state) ; 
    Kokkos::deep_copy(sswap.corner_staggered_fields, sstate.corner_staggered_fields) ;
    /*************************************************/
    /* Set ghostzone values to NaN before filling    */
    /*************************************************/
    h_state_mirror = Kokkos::create_mirror_view(state) ; 
    h_corner_mirror = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ;
    Kokkos::deep_copy(h_state_mirror, state) ; 
    Kokkos::deep_copy(h_corner_mirror, sstate.corner_staggered_fields) ;
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            
            if(   is_ghostzone(VEC(i,j,k),VEC(nx,ny,nz),ngz) ) 
            {
            h_state_mirror(VEC(i,j,k),DENS,q) = 
                std::numeric_limits<double>::quiet_NaN();
            }
        },
        {VEC(false,false,false)},
        true
    ) ;
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {

            if(   is_ghostzone(VEC(i,j,k),VEC(nx+1,ny+1,nz+1),ngz) ) 
            {
                h_corner_mirror(VEC(i,j,k),DENS,q) = 
                    std::numeric_limits<double>::quiet_NaN();
            }
        },
        {VEC(true,true,true)},
        true
    ) ;
    /*************************************************/
    /*                 Copy H2D                      */
    /*************************************************/
    Kokkos::deep_copy(state,h_state_mirror); 
    Kokkos::deep_copy(sstate.corner_staggered_fields,h_corner_mirror); 

    /*************************************************/
    /*                 Apply BCs                     */
    /*************************************************/
    grace::IO::write_cell_output(true,true,true) ; 
    grace::amr::apply_boundary_conditions() ; 

    /*************************************************/
    /*                 Copy D2H                      */
    /*************************************************/
    auto h_state_mirror_new = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror_new,state); 
    auto h_corner_mirror_new = Kokkos::create_mirror_view(sstate.corner_staggered_fields) ; 
    Kokkos::deep_copy(h_corner_mirror_new,sstate.corner_staggered_fields);  

    
}