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
* it under the  terms of the GNU General Public License as published by
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
#include <grace/utils/gridloop.hh>

#include <omp.h>

#include <chrono>
#include <string> 

namespace grace { 

void fill_cell_spacings(scalar_array_t<GRACE_NSPACEDIM>& ispacing,
                        scalar_array_t<GRACE_NSPACEDIM>& spacing)
{
    using namespace grace ; 

    auto& forest = grace::amr::forest::get()        ; 
    auto& conn   = grace::amr::connectivity::get()  ; 
    auto& params = grace::config_parser::get()      ;

    size_t nx {params["amr"]["npoints_block_x"].as<size_t>()} ; 
    size_t ny {params["amr"]["npoints_block_y"].as<size_t>()} ; 
    size_t nz {params["amr"]["npoints_block_z"].as<size_t>()} ; 
    
    auto nq = amr::get_local_num_quadrants() ; 

    auto h_idx = Kokkos::create_mirror_view(ispacing) ; 
    auto h_dx  = Kokkos::create_mirror_view(spacing) ; 
    /* 2) Number of ghostzones for evolved vars */
    long ngz { params["amr"]["n_ghostzones"].as<long>() } ;
    for( int iquad=0; iquad<nq; ++iquad ) {
        auto const& coord_system = coordinate_system::get() ;
        auto itree = amr::get_quadrant_owner(iquad) ;  
        amr::quadrant_t quadrant = amr::get_quadrant(itree,iquad) ; 
        auto const dx_lev = quadrant.spacing() ; 
        auto const VEC(dx_quad{dx_lev/nx}, dy_quad{dx_lev/ny}, dz_quad{dx_lev/nz}) ; 
        //! HERE TO SWICH COORDS
        auto const tree_spacing = amr::get_tree_spacing(itree)[0] ; 
        auto const VEC(dx_phys{dx_quad*tree_spacing}, dy_phys{dy_quad*tree_spacing}, dz_phys{dz_quad*tree_spacing}) ; 

        EXPR(
        h_idx(0,iquad) = 1./dx_phys ;, 
        h_idx(1,iquad) = 1./dy_phys ;,
        h_idx(2,iquad) = 1./dz_phys ;)
        EXPR(
        h_dx(0,iquad) = dx_phys ;,
        h_dx(1,iquad) = dy_phys ;,
        h_dx(2,iquad) = dz_phys ;)
    }

    Kokkos::deep_copy(ispacing,h_idx) ; 
    Kokkos::deep_copy(spacing,h_dx) ; 
}

void fill_physical_coordinates( coord_array_t<GRACE_NSPACEDIM>& pcoords 
                              , grace::var_staggering_t stag
                              , bool spherical )
{
    auto off = get_index_staggerings(stag) ; 
    std::array<double,GRACE_NSPACEDIM> cell_coordinates = {
        VEC( off[0] ? 0 : 0.5,
             off[1] ? 0 : 0.5,
             off[2] ? 0 : 0.5 )
    } ; 
    fill_physical_coordinates(pcoords,cell_coordinates,stag,spherical) ; 
}

void fill_physical_coordinates( coord_array_t<GRACE_NSPACEDIM>& pcoords 
                              , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates
                              , grace::var_staggering_t stag 
                              , bool spherical ) 
{
    DECLARE_GRID_EXTENTS ;
    using namespace grace ; 
    auto off = get_index_staggerings(stag) ; 
    Kokkos::realloc(pcoords,VEC(nx+2*ngz+off[0],ny+2*ngz+off[1],nz+2*ngz+off[2]),GRACE_NSPACEDIM,nq) ; 

    auto h_coords = Kokkos::create_mirror_view(pcoords) ; 
    auto& coord_system = grace::coordinate_system::get() ; 
    # if 0
    std::array<bool,GRACE_NSPACEDIM> stagger = {VEC(static_cast<bool>(off[0]), static_cast<bool>(off[1]), static_cast<bool>(off[2]))} ; 
    grace::host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            auto pcoordsl = 
                coord_system.get_physical_coordinates(
                          {VEC(i,j,k)}
                        , q
                        , cell_coordinates
                        , true 
            ) ; 
            if ( spherical ) pcoordsl = coord_system.cart_to_sph(pcoordsl) ; 
            EXPR(
            h_coords(VEC(i,j,k),0,q) = pcoordsl[0] ;,
            h_coords(VEC(i,j,k),1,q) = pcoordsl[1] ;,
            h_coords(VEC(i,j,k),2,q) = pcoordsl[2] ; 
            )
        }, stagger, true 
    ) ; 
    #endif 
    #if 1
    for( int q=0; q<nq; ++q ) {
        EXPR( for(size_t i=0; i<nx+2*ngz+off[0]; ++i), for(size_t j=0; j<ny+2*ngz+off[1]; ++j), for(size_t k=0; k<nz+2*ngz+off[2]; ++k) ) {
            auto pcoordsl = 
                coord_system.get_physical_coordinates(
                          {VEC(i,j,k)}
                        , q
                        , cell_coordinates
                        , true 
            ) ;
            if ( spherical ) pcoordsl = coord_system.cart_to_sph(pcoordsl) ; 
            EXPR(
            h_coords(VEC(i,j,k),0,q) = pcoordsl[0] ;,
            h_coords(VEC(i,j,k),1,q) = pcoordsl[1] ;,
            h_coords(VEC(i,j,k),2,q) = pcoordsl[2] ; 
            )
        }
    }
    #endif 
    Kokkos::deep_copy(pcoords,h_coords) ; 
}

} /* namespace grace */ 