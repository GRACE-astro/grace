#include <catch2/catch_test_macros.hpp>
#include <thunder_config.h>
#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <thunder/amr/forest.hh>
#include <thunder/amr/connectivity.hh>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/amr/amr_functions.hh>
#include <thunder/IO/vtk_output.hh>
#include <iostream>
#include <fstream>
#include <thunder/data_structures/variables.hh>
#include <thunder/data_structures/memory_defaults.hh>
#include <thunder/data_structures/macros.hh>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#define TEST_DBG_

template< typename T > 
void test_arrays_within_rel(std::array<T,THUNDER_NSPACEDIM> const& A, std::array<T,THUNDER_NSPACEDIM> const& B, T const& eps ) {
    for( int ii=0; ii<THUNDER_NSPACEDIM; ++ii) {
        REQUIRE_THAT(A[ii],Catch::Matchers::WithinRel(B[ii], eps));
    }
}

TEST_CASE("coordinates'\t'[coords_test]")
{
    using namespace thunder ;
    using namespace Kokkos ; 
    ASSERT( parallel::mpi_comm_size() == 1, "This test cannot be ran on more than 1 rank.") ; 
    auto& vars = thunder::variable_list::get() ; 
    auto& params = thunder::config_parser::get() ; 
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ;
    int64_t nq = amr::get_local_num_quadrants() ;
    #ifdef THUNDER_3D 
    View<double *****, default_space> lcoords("logical_coordinates", nx+2*ngz,ny+2*ngz,nz+2*ngz,3, nq) ; 
    View<double *****, default_space> pcoords("physical_coordinates", nx+2*ngz,ny+2*ngz,nz+2*ngz,3, nq) ; 
    #else 
    View<double ****, default_space> lcoords("logical_coordinates", nx+2*ngz,ny+2*ngz,2,nq) ; 
    View<double ****, default_space> pcoords("physical_coordinates", nx+2*ngz,ny+2*ngz,2,nq) ;
    #endif 

    auto& grid_coords = vars.getcoords() ;
    auto& dx = vars.getspacings() ;  
    auto const lcoords_mirror = create_mirror_view(lcoords);
    auto const pcoords_mirror = create_mirror_view(pcoords);
    auto const gcoords_mirror = create_mirror_view(grid_coords);
    
    int first_tree = amr::forest::get().first_local_tree() ; 
    int last_tree = amr::forest::get().last_local_tree()   ; 
    auto& coord_system = coordinate_system::get() ; 
    auto device_coords = coord_system.get_device_coord_system();
    /*
    for(int itree=first_tree; itree<=last_tree; ++itree){
        auto tree = amr::forest::get().tree(itree) ; 
        int q_offset = tree.quadrants_offset() ; 
        int num_quadrants = tree.num_quadrants()     ;
        parallel_for("fill_coords_arrays", 
                      MDRangePolicy<Rank<THUNDER_NSPACEDIM+1>>(
                        {VEC(0,0,0),q_offset}, {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),num_quadrants+q_offset}),
            KOKKOS_LAMBDA (VEC(size_t const& i, size_t const& j, size_t const& k), int const q){
                double _pcoords[THUNDER_NSPACEDIM] ; 
                double _lcoords[THUNDER_NSPACEDIM] = 
                {VEC(grid_coords(0,q) + (i - ngz + 0.5) * dx(0,q)
                    ,grid_coords(1,q) + (j - ngz + 0.5) * dx(1,q)
                    ,grid_coords(2,q) + (k - ngz + 0.5) * dx(2,q))};
                device_coords.get_physical_coordinates(itree,_lcoords,_pcoords) ;
                device_coords.get_logical_coordinates(itree,_pcoords,_lcoords) ;  
                for(int idim=0; idim<THUNDER_NSPACEDIM; ++idim){
                    lcoords(VEC(i,j,k),idim,q) = _lcoords[idim];
                    pcoords(VEC(i,j,k),idim,q) = _pcoords[idim];
                }
            }            
        ) ; 
    }

    deep_copy(lcoords_mirror,lcoords); deep_copy(pcoords_mirror,pcoords);
    */
    auto cell_volume_mirror = create_mirror_view(
        thunder::variable_list::get().getvolumes()
    ); 
    deep_copy(cell_volume_mirror, thunder::variable_list::get().getvolumes()) ; 
    #ifdef THUNDER_CARTESIAN_COORDINATES 
    EXPR(
    double x_extent = 
        params["amr"]["xmax"].as<double>() - params["amr"]["xmin"].as<double>() ;,
    double y_extent = 
        params["amr"]["ymax"].as<double>() - params["amr"]["ymin"].as<double>() ;,
    double z_extent = 
        params["amr"]["zmax"].as<double>() - params["amr"]["zmin"].as<double>() ;)
    double const total_grid_volume = EXPR(
        x_extent, * y_extent, * z_extent 
    ) ; 
    #elif defined(THUNDER_SPHERICAL_COORDINATES)
    double Ro = params["amr"]["outer_region_radius"].as<double>() ; 
    #ifdef THUNDER_3D 
    double const total_grid_volume = 
        4./3.*M_PI * math::int_pow<3>(Ro) ; 
    #else 
    double const total_grid_volume = M_PI * math::int_pow<2>(Ro) ; 
    #endif 
    #endif
    double vol_test = 0 ;
    for(int itree=first_tree; itree<=last_tree; ++itree){
        auto tree = amr::forest::get().tree(itree) ; 
        int q_offset = tree.quadrants_offset() ; 
        int num_quadrants = tree.num_quadrants()     ;
        size_t const ncells = EXPR((nx),*(ny),*(nz))*num_quadrants; 
        for( size_t icell=0UL; icell<ncells; icell+=1UL)
        {
            size_t const i = icell%(nx) ; 
            size_t const j = (icell/(nx)) % (ny) ;
            #ifdef THUNDER_3D 
            size_t const k = 
                (icell/(nx)/(ny)) % (nz) ; 
            size_t const q = 
                (icell/(nx)/(ny)/(nz)) + q_offset;
            #else 
            size_t const q = (icell/(nx)/(ny)) + q_offset; 
            #endif 
            vol_test += cell_volume_mirror(
                VEC(ngz+i,ngz+j,ngz+k),q 
            ) ; 
            auto quad = amr::get_quadrant(itree, q) ;  
            auto qcoords = quad.qcoords() ; 
            auto dx_quad = 1./(1<<quad.level()); 
            EXPR(
            auto dx_cell = dx_quad/nx ;,
            auto dy_cell = dx_quad/ny ;,
            auto dz_cell = dx_quad/nz ;) 
            std::array<double,THUNDER_NSPACEDIM> true_lcoords 
                = {VEC(
                    dx_quad * qcoords[0] + (i+0.5) * dx_cell,
                    dx_quad * qcoords[1] + (j+0.5) * dy_cell,
                    dx_quad * qcoords[2] + (k+0.5) * dz_cell
                )} ; 

            auto const phys_coords = coord_system.get_physical_coordinates({VEC(i,j,k)},q,false) ;
            auto const phys_coords2 = coord_system.get_physical_coordinates(itree,true_lcoords) ; 
            test_arrays_within_rel(phys_coords,phys_coords2, 1e-12 ) ; 
            auto const log_coords = coord_system.get_logical_coordinates(itree,phys_coords) ; 
            test_arrays_within_rel(log_coords,true_lcoords, 1e-12 ) ; 
            /*
            EXPR(
                REQUIRE_THAT(pcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),0,q),
                Catch::Matchers::WithinAbs(
                    phys_coords[0], 1e-12 
                ));,
                REQUIRE_THAT(pcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),1,q),
                Catch::Matchers::WithinAbs(
                    phys_coords[1], 1e-12 
                ));,
                REQUIRE_THAT(pcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),2,q),
                Catch::Matchers::WithinAbs(
                    phys_coords[2], 1e-12 
                ));
            )

            EXPR(
                REQUIRE_THAT(lcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),0,q),
                Catch::Matchers::WithinAbs(
                    true_lcoords[0], 1e-12 
                ));,
                REQUIRE_THAT(lcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),1,q),
                Catch::Matchers::WithinAbs(
                    true_lcoords[1], 1e-12 
                ));,
                REQUIRE_THAT(lcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),2,q),
                Catch::Matchers::WithinAbs(
                    true_lcoords[2], 1e-12 
                ));
            )
            */
        }

    }
    REQUIRE_THAT(vol_test,
                Catch::Matchers::WithinRel(
                   total_grid_volume, 1e-12 
    ));
    
}