#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <thunder_config.h>
#include <thunder/IO/hdf5_output.hh>
#include <thunder/amr/thunder_amr.hh>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/IO/vtk_output.hh>
#include <iostream>
TEST_CASE("Volume hdf5 output", "[vol_hdf5_out]")
{
    #ifdef THUNDER_ENABLE_BURGERS 
    int const DENS = U ; 
    int const DENS_ = U ; 
    int const BETAX_ = U ; 
    int const BETAY_ = U ; 
    int const BETAZ_ = U ; 
    #endif
    auto state  = thunder::variable_list::get().getstate() ;
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    int ngz = thunder::amr::get_n_ghosts() ; 
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

    thunder::IO::write_cell_data_hdf5(true,true,true) ; 
    thunder::IO::write_cell_data_vtk(true,false,false) ; 
}