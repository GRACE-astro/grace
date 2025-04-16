#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <grace_config.h>
#include <grace/IO/hdf5_output.hh>
#include <grace/IO/hdf5_output_sliced.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/vtk_output.hh>
#include <iostream>
TEST_CASE("Volume hdf5 output", "[vol_hdf5_out]")
{
    #ifdef GRACE_ENABLE_BURGERS 
    int const DENS = U ; 
    int const DENS_ = U ; 
    int const BETAX_ = U ; 
    int const BETAY_ = U ; 
    int const BETAZ_ = U ; 
    #endif
    auto aux  = grace::variable_list::get().getaux() ;
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    int ngz = grace::amr::get_n_ghosts() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(aux) ; 

    auto const ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ; 

    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
        size_t const i = icell%(nx + 2*ngz) ; 
        size_t const j = (icell/(nx + 2*ngz)) % (ny + 2*ngz) ;
        #ifdef GRACE_3D 
        size_t const k = 
            (icell/(nx + 2*ngz)/(ny + 2*ngz)) % (nz + 2*ngz) ; 
        size_t const q = 
            (icell/(nx + 2*ngz)/(ny + 2*ngz)/(nz + 2*ngz)) ;
        #else 
        size_t const q = (icell/(nx + 2*ngz)/(nx + 2*ngz)) ; 
        #endif 
        auto const coords = grace::get_physical_coordinates({VEC(i,j,k)},q, {VEC(0.5,0.5,0.5)}, true) ; 
        double const r2 = EXPR( math::int_pow<2>(coords[0]),
                              + math::int_pow<2>(coords[1]),
                              + math::int_pow<2>(coords[2]) )  ; 
        h_state_mirror(VEC(i,j,k),RHO,q) = exp( - r2 / 0.5 ) ; 

        h_state_mirror(VEC(i,j,k),VELX,q) = Kokkos::sqrt(r2) * sin(M_PI*coords[0]) ; 
        h_state_mirror(VEC(i,j,k),VELY,q) = Kokkos::sqrt(r2) * cos(M_PI*coords[1]) ; 
        h_state_mirror(VEC(i,j,k),VELZ,q) = coords[0] ; 

    }
    Kokkos::deep_copy(aux, h_state_mirror) ;

    grace::IO::write_volume_cell_data_sliced_hdf5() ; 
    //grace::IO::write_cell_data_vtk(true,false,false) ; 
}