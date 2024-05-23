/**
 * @file test_regridding.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-04-12
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */
#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <thunder/amr/thunder_amr.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/IO/vtk_output.hh>
#include <thunder/parallel/mpi_wrappers.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

template< typename coords_t > 
std::array<double,THUNDER_NSPACEDIM>
get_coords_buffer_zone(
    std::array<size_t,THUNDER_NSPACEDIM>const& ijk,
    std::array<double,THUNDER_NSPACEDIM>const& lcoords,
    int64_t q,
    coords_t& coord_system
){
    using namespace thunder ; 
    #ifdef THUNDER_ENABLE_BURGERS 
    int const DENS = U ; 
    int const DENS_ = U ; 
    #endif 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int iface = EXPR(
          (ijk[0] < ngz) * 0 
        + (ijk[0] > nx + ngz-1) * 1,
        + (ijk[1] < ngz) * 2 
        + (ijk[1] > ny + ngz-1) * 3,
        + (ijk[2] < ngz) * 4 
        + (ijk[2] > nz + ngz-1) * 5) ;
    if( iface >= P4EST_FACES) iface = 0;   
    auto& conn = amr::connectivity::get();
    int itree = amr::get_quadrant_owner(q) ;
    int    itree_b  = conn.tree_to_tree(itree, iface) ; 
    int    iface_b  = conn.tree_to_face(itree, iface) ; 
    int    polarity = conn.tree_to_tree_polarity(itree,iface) ;
     
    EXPR(
    int ig = EXPR( 
          (iface==0) * (ngz-1-ijk[0])
        + (iface==1) * (ijk[0]-nx-ngz),
        + (iface==2) * (ngz-1-ijk[1])
        + (iface==3) * (ijk[1]-ny-ngz),
        + (iface==4) * (ngz-1-ijk[2])
        + (iface==5) * (ijk[2]-nz-ngz)  ) ;, 
    int j  = EXPR( 
          (iface/2==0) * ijk[1],
        + (iface/2==1) * ijk[0],
        + (iface/2==2) * ijk[0] ) ;, 
    int k  = EXPR( 
          (iface/2==0) * ijk[2],
        + (iface/2==1) * ijk[2],
        + (iface/2==2) * ijk[1] ) ; )

    EXPR(
    double i_b = EXPR(
          (iface_b==0) * (
            (!polarity) * (ngz-1-ig+0.5)
          + (polarity)  * (ig+0.5) )
        + (iface_b==1) * (
            (!polarity) * (-1-ig+0.5)
          + (polarity)  * (-ngz+ig+0.5)),
        + (iface_b/2==1) * (j-ngz+0.5),
        + (iface_b/2==2) * (j-ngz+0.5)
    );,
    double j_b = EXPR(
          (iface_b==2) * (
            (!polarity) * (ngz-1-ig)
          + (polarity)  * (ig+0.5) )
        + (iface_b==3) * (
            (!polarity) * (-1-ig+0.5)
          + (polarity)  * (-ngz+ig+0.5) ),
        + (iface_b/2==0) * (j-ngz+0.5),
        + (iface_b/2==2) * (k-ngz+0.5) 
    );,
    double k_b = EXPR(
          (iface_b==4) * (
            (!polarity) * (ngz-1-ig+0.5)
          + (polarity)  * (+ig+0.5) )
        + (iface_b==5) * (
            (!polarity) * (-1-ig+0.5)
          + (polarity)  * (-ngz+ig+0.5) ),
        + (iface_b/2==0) * (k-ngz+0.5),
        + (iface_b/2==1) * (k-ngz+0.5)
    ) ;
    )

    auto quad_coords = coord_system.get_logical_coordinates(
        {VEC(0,0,0)},q,{VEC(0.,0.,0.)},false
    ) ; 
    auto quad = amr::get_quadrant(q); 
    std::array<double,THUNDER_NSPACEDIM> dxl =
    {VEC(
        1./(1<<quad.level())/nx,
        1./(1<<quad.level())/ny,
        1./(1<<quad.level())/nz
    )};
    EXPR(
    double const x = quad_coords[0]
        + ( iface == 1 ) * dxl[0] * (nx);,
    double const y = quad_coords[1]
        + ( iface == 3 ) * dxl[1] * (ny);,
    double const z = quad_coords[2]
        + ( iface == 5 ) * dxl[2] * (nz);
    )
    auto pcoords = get_physical_coordinates(
          itree
        , {VEC(x,y,z)}
    ) ; 
    auto lcoords_b = get_logical_coordinates(
          itree_b 
        , pcoords
    ) ; 
    EXPR(
    lcoords_b[0] += dxl[0] * (i_b) ;,
    lcoords_b[1] += dxl[1] * (j_b) ;,
    lcoords_b[2] += dxl[2] * (k_b) ;
    )
    return lcoords_b ; 
}

TEST_CASE("Simple regrid", "[regrid]")
{
    using namespace thunder::variables ; 

    #ifdef THUNDER_ENABLE_BURGERS 
    int const DENS = U ; 
    int const DENS_ = U ; 
    auto params = thunder::config_parser::get()["amr"] ; 
    params["refinement_criterion_var"] = "U" ; 
    #endif 
    #ifdef THUNDER_ENABLE_SCALAR_ADV
    int const DENS = U ; 
    int const DENS_ = U ; 
    auto params = thunder::config_parser::get()["amr"] ; 
    params["refinement_criterion_var"] = "U" ; 
    #endif 
    auto& state  = thunder::variable_list::get().getstate()  ;
    auto& coords = thunder::variable_list::get().getcoords() ; 
    auto& dx     = thunder::variable_list::get().getspacings(); 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    int ngz = thunder::amr::get_n_ghosts() ; 
    auto ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ; 
    
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 
    auto& coord_system = thunder::coordinate_system::get() ; 

    auto const h_func = [&] (VEC(const double& x,const double& y,const double &z))
    {
        return EXPR(8.5 * x, - 5.1 * y, -2*z) - 3.14 ; 
    } ; 
    /*************************************************/
    /*                   fill data                   */
    /*     here we fill the ghost zones as well.     */
    /*************************************************/
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
        /* Physical coordinates of cell center */
        auto pcoords = coord_system.get_physical_coordinates(
            {VEC(i,j,k)},
            q,
            true
        ) ; 
        h_state_mirror(VEC(i,j,k),DENS,q) = h_func(VEC(pcoords[0],pcoords[1],pcoords[2])) ; 
    }
    
    /* copy data to device */
    Kokkos::deep_copy(state,h_state_mirror); 
    //auto& swap = thunder::variable_list::get().getscratch() ; 
    //Kokkos::deep_copy(swap, state) ; 
    
    /*****************************************/
    /* compute total volume integrated value */
    /* here the ghostzones are excluded.     */
    /*****************************************/
    ncells = EXPR((nx),*(ny),*(nz))*nq ;
    double exact_total{0}, exact_total_local{0} ;  
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
        exact_total_local += h_state_mirror(VEC(i+ngz,j+ngz,k+ngz),DENS,q) * cell_volume ;
    }
    parallel::mpi_allreduce(&exact_total_local,&exact_total,1,sc_MPI_SUM) ; 
    /*write output and regrid*/
    //thunder::IO::write_cell_output(true,true,true) ; 
    thunder::amr::regrid() ;  
    thunder::runtime::get().increment_iteration() ; 
    //thunder::IO::write_cell_output(true,true,true) ; 
    /* compute the new volume integrated value */
    nq = thunder::amr::get_local_num_quadrants() ; // new number of quadrants (after regrid)
    ncells = EXPR((nx),*(ny),*(nz))*nq ;
    /* Copy data from device after regrid      */
    auto h_state_mirror_new = Kokkos::create_mirror_view(state) ; 
    Kokkos::deep_copy(h_state_mirror_new,state); 
    double total_local{0},total{0}; 
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
        auto const pcoords = coord_system.get_physical_coordinates(
            {VEC(i,j,k)},
            q,
            false
        ) ; 
        total_local += h_state_mirror_new(VEC(i+ngz,j+ngz,k+ngz),DENS,q) * cell_volume ; 
        #ifdef THUNDER_CARTESIAN_COORDINATES
        /* In spherical coordinates this won't work (and it should not!) */
        REQUIRE_THAT(h_state_mirror_new(VEC(i+ngz,j+ngz,k+ngz),DENS,q)
        , Catch::Matchers::WithinAbs(
                  h_func(VEC(pcoords[0],pcoords[1],pcoords[2]))
                , 1e-12)) ;
        #endif 
    } 
    parallel::mpi_allreduce(&total_local,&total,1,sc_MPI_SUM) ; 
    REQUIRE_THAT(total
                , Catch::Matchers::WithinRel(
                  exact_total
                , 1e-12)) ;
}