#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <thunder/amr/thunder_amr.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/IO/vtk_volume_output.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>



TEST_CASE("Simple regrid", "[regrid]")
{
    using namespace thunder::variables ; 

    DECLARE_VARIABLE_INDICES ; 
    
    auto& state  = thunder::variable_list::get().getstate() ;
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    int ngz = thunder::amr::get_n_ghosts() ; 

    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 

    auto ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ; 

    auto const func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(2.5 * x, - 5.1 * y, + 7.8*z) - 3.14 ; 
    } ; 

    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
        size_t const i = icell%(nx + 2*ngz) ; 
        size_t const j = (icell/(nx + 2*ngz)) % (ny + 2*ngz) ;
        #ifdef THUNDER_3D 
        size_t const k = 
            (icell/(nx + 2*ngz)/(ny + 2*ngz)) % (nz + 2*ngz) ; 
        size_t const q = 
            (icell/(nx + 2*ngz)/(ny + 2*ngz)/(nz + 2*ngz)) ;
        #else 
        size_t const q = (icell/(nx + 2*ngz)/(nx + 2*ngz)) ; 
        #endif   
        auto const coords = thunder::amr::get_physical_coordinates(icell, {VEC(0.5,0.5,0.5)}, true) ; 
        h_state_mirror(VEC(i,j,k),DENS,q) = func(VEC(coords[0],coords[1],coords[2])) ; 
    }
    Kokkos::deep_copy(state, h_state_mirror) ; 
    auto& swap = thunder::variable_list::get().getscratch() ; 
    Kokkos::deep_copy(swap, state) ; 
    
    thunder::IO::write_volume_cell_data() ;
    thunder::amr::regrid() ;  
    thunder::runtime::get().increment_iteration() ; 
    thunder::IO::write_volume_cell_data() ; 

    auto h_state_mirror_post_regrid = Kokkos::create_mirror_view(state) ;
    Kokkos::deep_copy(h_state_mirror_post_regrid, state) ; 
    nq = thunder::amr::get_local_num_quadrants() ;
    ncells = EXPR((nx),*(ny),*(nz))*nq ; 
    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
        auto const coords = thunder::amr::get_physical_coordinates(icell, {VEC(0.5,0.5,0.5)}, false) ; 
        size_t const i = icell%(nx) ; 
        size_t const j = (icell/(nx)) % (ny) ;
        #ifdef THUNDER_3D 
        size_t const k = 
            (icell/(nx)/(ny)) % (nz) ; 
        size_t const q = 
            (icell/(nx)/(ny)/(nz)) ;
        #else 
        size_t const q = (icell/(nx)/(ny)) ; 
        #endif 
        REQUIRE_THAT(h_state_mirror_post_regrid(VEC(i+ngz,j+ngz,k+ngz),DENS,q)
        , Catch::Matchers::WithinAbs(
                  func(VEC(coords[0],coords[1],coords[2]))
                , 1e-12)) ;
    } 
}