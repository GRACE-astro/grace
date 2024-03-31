#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <thunder/amr/thunder_amr.hh>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/IO/vtk_volume_output.hh>
#include <iostream>



TEST_CASE("Volume VTK output", "[vol_vtk_out]")
{
    using namespace thunder::variables ; 

    std::cout << "Starting..." << std::endl ;

    DECLARE_VARIABLE_INDICES ; 

    std::cout << DENS_ << std::endl ; 
    std::cout << thunder::get_variable_index("dens") << std::endl ;  
    auto state  = thunder::variable_list::get().getstate() ;
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    int ngz = thunder::amr::get_n_ghosts() ; 
    std::cout << "nx,ny(,nz),nq: " << EXPR(
        nx << ", " <<,
        ny << ", " <<,
        nz << ", " <<
    ) nq << std::endl ;
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 

    auto const ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ; 

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
        auto const coords = thunder::get_physical_coordinates({VEC(i,j,k)},q, {VEC(0.5,0.5,0.5)}, true) ; 
        double const r2 = EXPR( math::int_pow<2>(coords[0]),
                              + math::int_pow<2>(coords[1]),
                              + math::int_pow<2>(coords[2]) )  ; 
        h_state_mirror(VEC(i,j,k),DENS,q) = exp( - r2 / 0.5 ) ; 
    }
    Kokkos::deep_copy(state, h_state_mirror) ; 
    Kokkos::parallel_for("Fill_beta_vector"
                        , Kokkos::MDRangePolicy<Kokkos::Rank<THUNDER_NSPACEDIM+1>, thunder::default_execution_space>
                          ({VEC(0,0,0),0}, {VEC(nx,ny,nz),nq})
                        , KOKKOS_LAMBDA (VEC(int i, int j, int k), int q)
                        {
                            state(VEC(i,j,k),BETAX_,q) = 1.0 ; 
                            state(VEC(i,j,k),BETAY_,q) = 0.0 ; 
                            state(VEC(i,j,k),BETAZ_,q) = 0.0 ; 
                        }) ; 
    std::cout << "Calling output routine..." << std::endl ;
    thunder::IO::write_volume_cell_data() ; 

}