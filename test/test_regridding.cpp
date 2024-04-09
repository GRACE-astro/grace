#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <thunder/amr/thunder_amr.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/IO/vtk_volume_output.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>



TEST_CASE("Simple regrid", "[regrid]")
{
    using namespace thunder::variables ; 

    DECLARE_VARIABLE_INDICES ; 
    
    auto& state  = thunder::variable_list::get().getstate()  ;
    auto& coords = thunder::variable_list::get().getcoords() ; 
    auto& dx     = thunder::variable_list::get().getspacings(); 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    int ngz = thunder::amr::get_n_ghosts() ; 


    auto ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ; 

    auto const func = KOKKOS_LAMBDA (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(8.5 * x, - 5.1 * y, ) - 3.14 ; 
    } ;
    auto const h_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(8.5 * x, - 5.1 * y, ) - 3.14 ; 
    } ; 

    Kokkos::MDRangePolicy<Kokkos::Rank<THUNDER_NSPACEDIM+1>,thunder::default_execution_space>
        policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ; 

    Kokkos::parallel_for( "fill_data"
                        , policy 
                        , KOKKOS_LAMBDA( VEC(const int i, const int j, const int k), const int q)
        {
            EXPR(
            double x = coords(0,q) + (i-ngz+0.5) * dx(0,q);,
            double y = coords(1,q) + (j-ngz+0.5) * dx(1,q);,
            double z = coords(2,q) + (k-ngz+0.5) * dx(2,q);
            )
            state(VEC(i,j,k),DENS_,q) =  func(VEC(x,y,z)) ; 
        }
    
    );
    auto& swap = thunder::variable_list::get().getscratch() ; 
    Kokkos::deep_copy(swap, state) ; 
    
    thunder::IO::write_volume_cell_data() ;
    thunder::amr::regrid() ;  
    thunder::runtime::get().increment_iteration() ; 
    thunder::IO::write_volume_cell_data() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror, state) ; 
    auto h_coord_mirror = Kokkos::create_mirror_view(coords) ; 
    Kokkos::deep_copy(h_coord_mirror, coords) ; 
    auto& coord_system = thunder::coordinate_system::get() ; 
    nq = thunder::amr::get_local_num_quadrants() ;
    ncells = EXPR((nx),*(ny),*(nz))*nq ;
     
    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
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
        auto lcoords = coord_system.get_logical_coordinates(
              {VEC(i,j,k)}
            , q
            , {VEC(0.5,0.5,0.5)}
            , false 
        ) ; 
        REQUIRE_THAT(h_state_mirror(VEC(i+ngz,j+ngz,k+ngz),DENS,q)
        , Catch::Matchers::WithinAbs(
                  h_func(VEC(lcoords[0],lcoords[1],lcoords[2]))
                , 1e-12)) ;
    } 
}