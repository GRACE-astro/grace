#include <catch2/catch_test_macros.hpp>
#include <thunder_config.h>
#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <thunder/amr/forest.hh>
#include <thunder/amr/connectivity.hh>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/amr/amr_functions.hh>
#include <iostream>
#include <fstream>
#include <thunder/data_structures/variables.hh>
#include <thunder/data_structures/memory_defaults.hh>
#include <thunder/data_structures/macros.hh>
#include <catch2/matchers/catch_matchers_floating_point.hpp>


TEST_CASE("coordinates'\t'[coords_test]")
{
    using namespace thunder ;
    using namespace Kokkos ; 
    auto& vars = thunder::variable_list::get() ; 

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

    auto const lcoords_mirror = create_mirror_view(lcoords);
    auto const pcoords_mirror = create_mirror_view(pcoords);
    auto const gcoords_mirror = create_mirror_view(grid_coords);
    
    int first_tree = amr::forest::get().first_local_tree() ; 
    int last_tree = amr::forest::get().last_local_tree()   ; 
    auto& coord_system = coordinate_system::get() ; 
    auto device_coords = coord_system.get_device_coord_system();
    for(int itree=first_tree; itree<=last_tree; ++itree){
        std::cout << "Tree: " << itree << std::endl ; 
        auto tree = amr::forest::get().tree(itree) ; 
        int q_offset = tree.quadrants_offset() ; 
        int num_quadrants = tree.num_quadrants()     ;
        parallel_for("fill_coords_arrays", 
                      MDRangePolicy<Rank<THUNDER_NSPACEDIM+1>>(
                        {VEC(0,0,0),q_offset}, {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),num_quadrants+q_offset}),
            KOKKOS_LAMBDA (VEC(size_t const& i, size_t const& j, size_t const& k), int const q){
                double _pcoords[THUNDER_NSPACEDIM] ; 
                double _lcoords[THUNDER_NSPACEDIM] = 
                {VEC(grid_coords(VEC(i,j,k),0,q)
                    ,grid_coords(VEC(i,j,k),1,q)
                    ,grid_coords(VEC(i,j,k),2,q))};
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

    for(int itree=first_tree; itree<=last_tree; ++itree){
        std::cout << "Tree " << itree << std::endl ; 
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
            auto const log_coords = coord_system.get_logical_coordinates(itree,phys_coords) ; 
            auto const log_coords2 = coord_system.get_logical_coordinates(phys_coords) ;

            std::cout << q << ", " << i << ", " << j << '\n'
                      << phys_coords[0] << ", " << phys_coords[1] << '\n'
                      << true_lcoords[0] << ", " << true_lcoords[1] << '\n' 
                      << log_coords[0] << ", " << log_coords[1] << '\n' 
                      << pcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),0,q) << ", " << pcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),1,q) << '\n'
                      << lcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),0,q) << ", " << lcoords_mirror(VEC(i+ngz,j+ngz,k+ngz),1,q) << '\n'; 
            EXPR(
                REQUIRE_THAT(log_coords[0],
                Catch::Matchers::WithinAbs(
                    true_lcoords[0], 1e-12 
                ));,
                REQUIRE_THAT(log_coords[1],
                Catch::Matchers::WithinAbs(
                    true_lcoords[1], 1e-12 
                ));,
                REQUIRE_THAT(log_coords[2],
                Catch::Matchers::WithinAbs(
                    true_lcoords[2], 1e-12 
                ));)

                EXPR(
                REQUIRE_THAT(log_coords2[0],
                Catch::Matchers::WithinAbs(
                    true_lcoords[0], 1e-12 
                ));,
                REQUIRE_THAT(log_coords2[1],
                Catch::Matchers::WithinAbs(
                    true_lcoords[1], 1e-12 
                ));,
                REQUIRE_THAT(log_coords2[2],
                Catch::Matchers::WithinAbs(
                    true_lcoords[2], 1e-12 
                ));
            )

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
        }

    }
    
}