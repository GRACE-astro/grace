#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <thunder/amr/thunder_amr.hh>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/IO/vtk_volume_output.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


static inline bool is_outside_grid(VEC(size_t i,size_t j, size_t k), int64_t q)
{
    auto params = thunder::config_parser::get()["amr"] ; 
    auto coord_type = thunder::config_parser::get()["amr"]["physical_coordinates"].as<std::string>() ; 
    auto pcoords = thunder::get_physical_coordinates({VEC(i,j,k)},q,{VEC(0.5,0.5,0.5)}, true) ;
    if( coord_type == "cartesian" )
    {
        double xmin = params["xmin"].as<double>() ;
        double ymin = params["ymin"].as<double>() ;
        double zmin = params["zmin"].as<double>() ;

        double xmax = params["xmax"].as<double>() ;
        double ymax = params["ymax"].as<double>() ;
        double zmax = params["zmax"].as<double>() ; 

        return (pcoords[0]<xmin) || (pcoords[0]>xmax) || pcoords[1]<ymin || pcoords[1]>ymax 
        #ifdef THUNDER_3D 
        || (pcoords[2]<zmin) || (pcoords[2]>zmax)
        #endif 
        ;

    } else if ( coord_type == "spherical" )
    {   
        auto const Ro = params["outer_region_radius"].as<double>() ;
        auto r2 = EXPR(
              math::int_pow<2>(pcoords[0]),
            + math::int_pow<2>(pcoords[1]),
            + math::int_pow<2>(pcoords[2])
        );

        return r2 > Ro*Ro ;
    } else {
        return -1 ; 
    }

}

TEST_CASE("Apply BC", "[boundaries]")
{
    using namespace thunder::variables ; 
    using namespace thunder ;
    using namespace Kokkos ; 

    DECLARE_VARIABLE_INDICES ; 
    
    auto& state  = thunder::variable_list::get().getstate()  ;
    auto& coords = thunder::variable_list::get().getcoords() ; 
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

    //Kokkos::MDRangePolicy<Kokkos::Rank<THUNDER_NSPACEDIM+1>,thunder::default_execution_space>
    //    policy({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq}) ; 
    Kokkos::MDRangePolicy<Kokkos::Rank<THUNDER_NSPACEDIM+1>,thunder::default_execution_space>
        policy({VEC(ngz,ngz,ngz),0},{VEC(nx+ngz,ny+ngz,nz+ngz),nq}) ; 
    Kokkos::parallel_for( "fill_data"
                        , policy 
                        , KOKKOS_LAMBDA( VEC(const int i, const int j, const int k), const int q)
        {
            state(VEC(i,j,k),DENS_,q) =  func(VEC(coords(VEC(i,j,k),0,q),coords(VEC(i,j,k),1,q),coords(VEC(i,j,k),2,q))) ; 
        }
    
    );

    Kokkos::MDRangePolicy<Kokkos::Rank<THUNDER_NSPACEDIM+1>,thunder::default_execution_space>
        policy_gz({VEC(0,ngz,ngz),0},{VEC(ngz,static_cast<long>(nx)+ngz,static_cast<long>(nx)+ngz),static_cast<long>(nq)}) ; 
    Kokkos::parallel_for( "fill_data"
                        , policy_gz
                        , KOKKOS_LAMBDA( VEC(const int i, const int j, const int k), const int q)
        {
            state(VEC(i,j,k),DENS_,q) =  -100 ;
            state(VEC(nx+ngz+i,j,k),DENS_,q) =  -100 ;

            state(VEC(j,i,k),DENS_,q) =  -100 ;
            state(VEC(j,nx+ngz+i,k),DENS_,q) =  -100 ;
        }
    
    );
    auto& swap = thunder::variable_list::get().getscratch() ; 
    Kokkos::deep_copy(swap, state) ; 
    
    //thunder::IO::write_volume_cell_data() ;
    //thunder::amr::regrid() ;  
    thunder::amr::apply_boundary_conditions() ; 
    thunder::runtime::get().increment_iteration() ; 
    thunder::IO::write_volume_cell_data() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror, state) ; 
    auto h_coord_mirror = Kokkos::create_mirror_view(coords) ; 
    Kokkos::deep_copy(h_coord_mirror, coords) ; 


    nq = thunder::amr::get_local_num_quadrants() ;
    ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;

    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
        size_t const i = icell%(nx+2*ngz) ; 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef THUNDER_3D 
        size_t const k = 
            (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
        size_t const q = 
            (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
        #else 
        size_t const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
        #endif 
        if( is_outside_grid(VEC(i,j,k),q) ){
                continue ; 
        }
        if(
            ((i<ngz) || (i>nx+ngz-1)) and 
            ((j<ngz) || (j>ny+ngz-1))
        ){
            continue ; 
        }
        auto pcoords = get_physical_coordinates(
            {VEC(i,j,k)},q, {VEC(0.5,0.5,0.5)}, true 
        ) ;
        /*
        std::cout << "Indices: " << icell << ", " << q << ", " << i << ", " << j << std::endl ;
        std::cout << "Coordinates: " << pcoords[0] << ", " << pcoords[1] << std::endl ; 
        std::cout << "In ghost-zones: " << ((i<ngz) || (j<ngz) || (i>nx+ngz-1) || (j>ny+ngz-1) ? "yes\n" : "no\n") ; 
        */
        REQUIRE_THAT(h_state_mirror(VEC(i,j,k),DENS,q)
        , Catch::Matchers::WithinAbs(
                  h_func(VEC(h_coord_mirror(VEC(i,j,k),0,q),h_coord_mirror(VEC(i,j,k),1,q),h_coord_mirror(VEC(i,j,k),2,q)))
                , 1e-12)) ;
    } 
}