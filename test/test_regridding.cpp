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

    auto ncells = EXPR((nx),*(ny),*(nz))*nq ; 

    auto const func = KOKKOS_LAMBDA (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(8.5 * x, - 5.1 * y, ) - 3.14 ; 
    } ;
    auto const h_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(8.5 * x, - 5.1 * y, ) - 3.14 ; 
    } ; 
    
    
    /* fill data */
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
    /* copy data to swap before regrid */
    auto& swap = thunder::variable_list::get().getscratch() ; 
    Kokkos::deep_copy(swap, state) ; 
    /* copy data to host */
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror,state); 
    auto& coord_system = thunder::coordinate_system::get() ; 
    /* compute total volume integrated value */
    double exact_total{0} ; 
    double total_volume{0} ; 
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

        auto const cell_volume = coord_system.get_cell_volume(
              {VEC(i,j,k)}
            , q
            , false
        ) ; 
        exact_total += h_state_mirror(VEC(i+ngz,j+ngz,k+ngz),DENS,q) * cell_volume ;
    }
    /*write output and regrid*/
    thunder::IO::write_volume_cell_data() ;
    thunder::amr::regrid() ;  
    thunder::runtime::get().increment_iteration() ; 
    thunder::IO::write_volume_cell_data() ; 
    /* compute the new volume integrated value */
    nq = thunder::amr::get_local_num_quadrants() ;
    ncells = EXPR((nx),*(ny),*(nz))*nq ;
    auto h_state_mirror_new = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror_new,state); 
    double total{0};
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
        auto const cell_volume = coord_system.get_cell_volume(
              {VEC(i,j,k)}
            , q
            , false
        ) ; 
        total += h_state_mirror_new(VEC(i+ngz,j+ngz,k+ngz),DENS,q) * cell_volume ; 
        
        REQUIRE_THAT(h_state_mirror_new(VEC(i+ngz,j+ngz,k+ngz),DENS,q)
        , Catch::Matchers::WithinAbs(
                  h_func(VEC(lcoords[0],lcoords[1],lcoords[2]))
                , 1e-12)) ;
    } 
    REQUIRE_THAT(total
                , Catch::Matchers::WithinAbs(
                  exact_total
                , 1e-12)) ;
}