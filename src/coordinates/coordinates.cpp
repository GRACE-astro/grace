/**
* @file coordinates.cpp
* @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
* @brief 
* @version 0.1
* @date 2024-03-12
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

#include <thunder_config.h>

#include <Kokkos_Core.hpp>

#include <thunder/amr/thunder_amr.hh>
#include <thunder/coordinates/coordinates.hh>
#include <thunder/coordinates/coordinate_systems.hh>

#include <thunder/config/config_parser.hh>

#include <thunder/data_structures/variables.hh>
#include <thunder/data_structures/macros.hh>
#include <thunder/data_structures/memory_defaults.hh>


#include <thunder/utils/thunder_utils.hh>

#include <string> 

namespace thunder { 

void fill_cell_coordinates( scalar_array_t<THUNDER_NSPACEDIM>& coords
                          , scalar_array_t<THUNDER_NSPACEDIM>& ispacing
                          , scalar_array_t<THUNDER_NSPACEDIM>& spacing
                          , cell_vol_array_t<THUNDER_NSPACEDIM>& volume) 
{
    using namespace thunder ; 
    auto& forest = thunder::amr::forest::get()        ; 
    auto& conn   = thunder::amr::connectivity::get()  ; 
    auto& params = thunder::config_parser::get()      ;

    size_t nx {params["amr"]["npoints_block_x"].as<size_t>()} ; 
    size_t ny {params["amr"]["npoints_block_y"].as<size_t>()} ; 
    size_t nz {params["amr"]["npoints_block_z"].as<size_t>()} ; 

    auto h_coords = Kokkos::create_mirror_view(coords) ; 
    auto h_idx = Kokkos::create_mirror_view(ispacing) ; 
    auto h_dx  = Kokkos::create_mirror_view(spacing) ; 
    auto h_vol = Kokkos::create_mirror_view(volume) ; 

    auto& coord_system = coordinate_system::get() ;
    /* 2) Number of ghostzones for evolved vars */
    long ngz { params["amr"]["n_ghostzones"].as<long>() } ;
    for( int itree=forest.first_local_tree(); itree<=forest.last_local_tree(); ++itree)
    {
        auto tree = forest.tree(itree) ; 
        auto quadrants = tree.quadrants() ;
        size_t quad_offset = tree.quadrants_offset() ;  
        for( int iquad=0; iquad<quadrants.size(); ++iquad ) {
            amr::quadrant_t quadrant = tree.quadrant(iquad) ; 
            auto const dx_lev = 1.0 / ( 1UL<<quadrant.level() ) ; 
            auto const VEC(dx_quad{dx_lev/nx}, dy_quad{dx_lev/ny}, dz_quad{dx_lev/nz}) ; 
            /* coordinates of lower left corner of quadrant */
            auto const qcoords = quadrant.qcoords() ; 
            size_t iquad_glob = iquad + quad_offset ; 
            h_coords(0,iquad_glob) = qcoords[0] * dx_lev; 
            h_coords(1,iquad_glob) = qcoords[1] * dx_lev;
            h_coords(2,iquad_glob) = qcoords[2] * dx_lev;
            EXPR(
            h_idx(0,iquad_glob) = 1./dx_quad ;, 
            h_idx(1,iquad_glob) = 1./dy_quad ;,
            h_idx(2,iquad_glob) = 1./dz_quad ;)
            EXPR(
            h_dx(0,iquad_glob) = dx_quad ;,
            h_dx(1,iquad_glob) = dy_quad ;,
            h_dx(2,iquad_glob) = dz_quad ;)
            EXPR(
            for(size_t i=0; i<nx+2*ngz; ++i){,
                for(size_t j=0; j<ny+2*ngz; ++j){,
                    for(size_t k=0; k<nz+2*ngz; ++k){
            )
                        h_vol(VEC(i,j,k),iquad_glob) = coord_system.get_cell_volume(
                              {VEC(i,j,k)}
                            , iquad_glob
                            , itree 
                            , {VEC(dx_quad,dy_quad,dz_quad)}
                            , true )  ; 
            EXPR(
                    },
                },
            })
        } /* quadrant loop */
    } /* tree loop */
    Kokkos::deep_copy(coords,h_coords) ; 
    Kokkos::deep_copy(ispacing,h_idx) ; 
    Kokkos::deep_copy(spacing,h_dx) ; 
    Kokkos::deep_copy(volume,h_vol) ;

    
}

} /* namespace thunder */ 