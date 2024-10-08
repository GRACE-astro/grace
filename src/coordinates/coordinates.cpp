/**
* @file coordinates.cpp
* @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
* @brief 
* @version 0.1
* @date 2024-03-12
* 
* @copyright This file is part of GRACE.
* GRACE is an evolution framework that uses Finite Difference
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

#include <grace_config.h>

#include <Kokkos_Core.hpp>

#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/system/print.hh>
#include <grace/config/config_parser.hh>

#include <grace/data_structures/variables.hh>
#include <grace/data_structures/memory_defaults.hh>


#include <grace/utils/grace_utils.hh>

#include <omp.h>

#include <chrono>
#include <string> 

namespace grace { 

void fill_cell_coordinates( scalar_array_t<GRACE_NSPACEDIM>& coords
                          , scalar_array_t<GRACE_NSPACEDIM>& ispacing
                          , scalar_array_t<GRACE_NSPACEDIM>& spacing
                          , cell_vol_array_t<GRACE_NSPACEDIM>& volume
                          , staggered_coordinate_arrays_t& surfaces_and_edges) 
{
    using namespace grace ; 
    auto& forest = grace::amr::forest::get()        ; 
    auto& conn   = grace::amr::connectivity::get()  ; 
    auto& params = grace::config_parser::get()      ;

    size_t nx {params["amr"]["npoints_block_x"].as<size_t>()} ; 
    size_t ny {params["amr"]["npoints_block_y"].as<size_t>()} ; 
    size_t nz {params["amr"]["npoints_block_z"].as<size_t>()} ; 
    
    auto nq = amr::get_local_num_quadrants() ; 

    auto h_coords = Kokkos::create_mirror_view(coords) ; 
    auto h_idx = Kokkos::create_mirror_view(ispacing) ; 
    auto h_dx  = Kokkos::create_mirror_view(spacing) ; 
    auto h_vol = Kokkos::create_mirror_view(volume) ; 
    
    auto h_surfx = Kokkos::create_mirror_view(surfaces_and_edges.cell_face_surfaces_x) ; 
    auto h_surfy = Kokkos::create_mirror_view(surfaces_and_edges.cell_face_surfaces_y) ; 
    auto h_surfz = Kokkos::create_mirror_view(surfaces_and_edges.cell_face_surfaces_z) ; 
    #ifdef GRACE_3D 
    auto h_edgexy = Kokkos::create_mirror_view(surfaces_and_edges.cell_edge_lengths_xy) ; 
    auto h_edgeyz = Kokkos::create_mirror_view(surfaces_and_edges.cell_edge_lengths_yz) ; 
    auto h_edgexz = Kokkos::create_mirror_view(surfaces_and_edges.cell_edge_lengths_xz) ; 
    #endif 
    decltype(h_surfx) surf[GRACE_NSPACEDIM] = {
        VEC(h_surfx, h_surfy, h_surfz)
    } ; 
    #ifdef GRACE_3D 
    decltype(h_edgexy) edge[GRACE_NSPACEDIM] = {
        VEC(h_edgeyz,h_edgexz,h_edgexy) 
    } ; 
    #endif 
    auto clock_start = std::chrono::high_resolution_clock::now() ;
    double avg_time = 0. ;  
    /* 2) Number of ghostzones for evolved vars */
    long ngz { params["amr"]["n_ghostzones"].as<long>() } ;
    #pragma omp parallel for schedule(static), reduction(+:avg_time)
    for( int iquad=0; iquad<nq; ++iquad ) {
        auto const& coord_system = coordinate_system::get() ;
        auto itree = amr::get_quadrant_owner(iquad) ;  
        amr::quadrant_t quadrant = amr::get_quadrant(itree,iquad) ; 
        auto const dx_lev = 1.0 / ( 1UL<<quadrant.level() ) ; 
        auto const VEC(dx_quad{dx_lev/nx}, dy_quad{dx_lev/ny}, dz_quad{dx_lev/nz}) ; 
        //! HERE TO SWICH COORDS
        auto const tree_spacing = amr::get_tree_spacing(itree)[0] ; 
        auto const VEC(dx_phys{dx_quad*tree_spacing}, dy_phys{dy_quad*tree_spacing}, dz_phys{dz_quad*tree_spacing}) ; 
        /* coordinates of lower left corner of quadrant */
        auto const qcoords = quadrant.qcoords() ; 
        h_coords(0,iquad) = qcoords[0] * dx_lev; 
        h_coords(1,iquad) = qcoords[1] * dx_lev;
        h_coords(2,iquad) = qcoords[2] * dx_lev;
        EXPR(
        h_idx(0,iquad) = 1./dx_phys ;, 
        h_idx(1,iquad) = 1./dy_phys ;,
        h_idx(2,iquad) = 1./dz_phys ;)
        EXPR(
        h_dx(0,iquad) = dx_phys ;,
        h_dx(1,iquad) = dy_phys ;,
        h_dx(2,iquad) = dz_phys ;)
        auto thread_clock_start = std::chrono::high_resolution_clock::now() ;
        EXPR( for(size_t i=0; i<nx+2*ngz; ++i), for(size_t j=0; j<ny+2*ngz; ++j), for(size_t k=0; k<nz+2*ngz; ++k) ) {
            h_vol(VEC(i,j,k),iquad) = coord_system.get_cell_volume(
                  {VEC( qcoords[0] * dx_lev + (int(i)-ngz) * dx_quad
                      , qcoords[1] * dx_lev + (int(j)-ngz) * dy_quad
                      , qcoords[2] * dx_lev + (int(k)-ngz) * dz_quad) }
                , itree 
                , {VEC(dx_quad,dy_quad,dz_quad)}
                , true )  ; 
            ASSERT_DBG(
                !std::isnan(h_vol(VEC(i,j,k),iquad)),
                "Cell volume NaN at " 
                EXPR(<< i ,<< ", " << j,<< ", " << k)
                << ", " << iquad << ", " << itree << '\n'
                EXPR(<< h_coords(0,iquad) + dx_quad * (i-ngz) 
                    ,<< ", " << h_coords(1,iquad) + dy_quad * (j-ngz)
                    ,<< ", " << h_coords(2,iquad) + dz_quad * (k-ngz)) << ", " << dx_quad 
            ) ;
            
            ASSERT_DBG(
                h_vol(VEC(i,j,k),iquad)>0,
                "Non positive cell volume " << h_vol(VEC(i,j,k),iquad)
                << " at " 
                EXPR(<< i ,<< ", " << j,<< ", " << k)
                << ", " << iquad << ", " << itree << '\n'
                EXPR(<< h_coords(0,iquad) + dx_quad * (i-ngz) 
                    ,<< ", " << h_coords(1,iquad) + dy_quad * (j-ngz)
                    ,<< ", " << h_coords(2,iquad) + dz_quad * (k-ngz)) << ", " << dx_quad 
            ) ;
        }

        for(int idim=0; idim<GRACE_NSPACEDIM; ++idim) { 
            EXPR( for(size_t i=0; i<nx+2*ngz+utils::delta(0,idim); ++i), for(size_t j=0; j<ny+2*ngz+utils::delta(1,idim); ++j), for(size_t k=0; k<nz+2*ngz+utils::delta(2,idim); ++k) ) 
            {
                surf[idim](VEC(i,j,k),iquad) = coord_system.get_cell_face_surface(
                    {VEC( qcoords[0] * dx_lev + (int(i)-ngz) * dx_quad
                      , qcoords[1] * dx_lev + (int(j)-ngz) * dy_quad
                      , qcoords[2] * dx_lev + (int(k)-ngz) * dz_quad) }
                    , idim
                    , itree 
                    , {VEC(dx_quad,dy_quad,dz_quad)}
                    , true ) ; 
            }
            #ifdef GRACE_3D 
            EXPR( for(size_t i=0; i<nx+2*ngz+1-utils::delta(0,idim); ++i), for(size_t j=0; j<ny+2*ngz+1-utils::delta(1,idim); ++j), for(size_t k=0; k<nz+2*ngz+1-utils::delta(2,idim); ++k) ) 
            {
                edge[idim](VEC(i,j,k),iquad) = coord_system.get_cell_edge_length(
                    {VEC( qcoords[0] * dx_lev + (int(i)-ngz) * dx_quad
                      , qcoords[1] * dx_lev + (int(j)-ngz) * dy_quad
                      , qcoords[2] * dx_lev + (int(k)-ngz) * dz_quad) }
                    , idim
                    , itree 
                    , {VEC(dx_quad,dy_quad,dz_quad)}
                    , true ) ; 
            }
            #endif 

        }
        auto thread_clock_end = std::chrono::high_resolution_clock::now() ;
        avg_time += double(std::chrono::duration_cast <std::chrono::microseconds> (thread_clock_end - thread_clock_start).count());
    } /* quadrant loop */
    auto clock_end = std::chrono::high_resolution_clock::now() ;
    float currentTime = float(std::chrono::duration_cast <std::chrono::microseconds> (clock_end - clock_start).count());
    GRACE_VERBOSE("Coordinate filling loop took {:.3e} mus.",currentTime) ; 
    clock_start = std::chrono::high_resolution_clock::now() ; 
    Kokkos::deep_copy(coords,h_coords) ; 
    Kokkos::deep_copy(ispacing,h_idx) ; 
    Kokkos::deep_copy(spacing,h_dx) ; 
    Kokkos::deep_copy(volume,h_vol) ;
    Kokkos::deep_copy(surfaces_and_edges.cell_face_surfaces_x, h_surfx) ; 
    Kokkos::deep_copy(surfaces_and_edges.cell_face_surfaces_y, h_surfy) ; 
    Kokkos::deep_copy(surfaces_and_edges.cell_face_surfaces_z, h_surfz) ; 
    #ifdef GRACE_3D 
    Kokkos::deep_copy(surfaces_and_edges.cell_edge_lengths_xy, h_edgexy) ; 
    Kokkos::deep_copy(surfaces_and_edges.cell_edge_lengths_xz, h_edgexz) ; 
    Kokkos::deep_copy(surfaces_and_edges.cell_edge_lengths_yz, h_edgeyz) ;
    #endif  
    clock_end = std::chrono::high_resolution_clock::now() ; 
    currentTime = float(std::chrono::duration_cast <std::chrono::microseconds> (clock_end - clock_start).count());
    GRACE_VERBOSE("Coordinate view copy (h2d) took {:.3e} mus.",currentTime) ;
}

void fill_physical_coordinates( coord_array_t<GRACE_NSPACEDIM>& pcoords 
                              , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates) 
{
    DECLARE_GRID_EXTENTS ;
    using namespace grace ; 

    Kokkos::realloc(pcoords,VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),GRACE_NSPACEDIM,nq) ; 

    auto h_coords = Kokkos::create_mirror_view(pcoords) ; 
    auto& coord_system = grace::coordinate_system::get() ; 
    # if 0 
    grace::host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoordsl = 
                coord_system.get_physical_coordinates(
                          {VEC(i,j,k)}
                        , q
                        , cell_coordinates
                        , true 
            ) ; 
            EXPR(
            h_coords(VEC(i,j,k),0,q) = pcoordsl[0] ;,
            h_coords(VEC(i,j,k),1,q) = pcoordsl[1] ;,
            h_coords(VEC(i,j,k),2,q) = pcoordsl[2] ; 
            )
        }, true 
    ) ; 
    #endif 

    for( int q=0; q<nq; ++q ) {
        EXPR( for(size_t i=0; i<nx+2*ngz; ++i), for(size_t j=0; j<ny+2*ngz; ++j), for(size_t k=0; k<nz+2*ngz; ++k) ) {
            auto pcoordsl = 
                coord_system.get_physical_coordinates(
                          {VEC(i,j,k)}
                        , q
                        , cell_coordinates
                        , true 
            ) ;
            EXPR(
            h_coords(VEC(i,j,k),0,q) = pcoordsl[0] ;,
            h_coords(VEC(i,j,k),1,q) = pcoordsl[1] ;,
            h_coords(VEC(i,j,k),2,q) = pcoordsl[2] ; 
            )
        }
    }
 
    Kokkos::deep_copy(pcoords,h_coords) ; 
}

void fill_jacobian_matrices( jacobian_array_t& jac 
                           , jacobian_array_t& inv_jac 
                           , std::array<double, GRACE_NSPACEDIM> const& cell_coordinates ) 
{
    DECLARE_GRID_EXTENTS ;
    using namespace grace ; 

    Kokkos::realloc(jac,VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),GRACE_NSPACEDIM,GRACE_NSPACEDIM,nq) ; 
    Kokkos::realloc(inv_jac,VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),GRACE_NSPACEDIM,GRACE_NSPACEDIM,nq) ; 

    auto hj  = Kokkos::create_mirror_view(jac)     ; 
    auto hij = Kokkos::create_mirror_view(inv_jac) ; 

    auto& coord_system = grace::coordinate_system::get() ; 

    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto J = coord_system.get_jacobian_matrix(
                  {VEC(i,j,k)}
                , q
                , cell_coordinates
                , true 
            ) ; 

            auto Ji = coord_system.get_inverse_jacobian_matrix(
                  {VEC(i,j,k)}
                , q
                , cell_coordinates
                , true 
            ) ; 

            int idx = 0 ;
            #pragma unroll GRACE_NSPACEDIM*GRACE_NSPACEDIM
            for( int im=0; im<GRACE_NSPACEDIM; ++im){
                for( int jm=0; jm<GRACE_NSPACEDIM; ++jm){
                    hj(VEC(i,j,k),im,jm,q)  = J[idx]  ;
                    hij(VEC(i,j,k),im,jm,q) = Ji[idx] ;
                    idx ++ ; 
                }
            }
            
        }, true 
    ) ; 
 
    Kokkos::deep_copy(jac    , hj ) ;
    Kokkos::deep_copy(inv_jac, hij) ;  
}

} /* namespace grace */ 