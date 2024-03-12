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

#include <Kokkos_Core.hh>
#include <thunder/amr/p4est_headers.hh>

#include <thunder/amr/coordinates.hh>
#include <thunder/amr/connectivity.hh>
#include <thunder/amr/forest.hh>

#include <thunder/config/config_parser.hh>

#include <thunder/data_structures/variables.hh>

#include <thunder/utils/device.h>
#include <thunder/utils/math.hh> 

#include <string> 

namespace thunder { 

void fill_coordinates() 
{
    auto& forest = thunder::amr::forest::get()        ; 
    auto& conn   = thunder::amr::connectivity::get()  ; 
    auto& param  = thunder::amr::config_parser::get() ;
    auto& vars   = thunder::variable_list::get()      ;  

    auto coords = vars.getcoords() ; 

    auto h_coords = Kokkos::get_mirror_view(coords)  ; 

    std::string coord_system = param["amr"]["physical_coordinates"].as<std::string>() ; 
    size_t nx {params["amr"]["npoints_block_x"].as<size_t>()} ; 
    size_t ny {params["amr"]["npoints_block_y"].as<size_t>()} ; 
    size_t nz {params["amr"]["npoints_block_z"].as<size_t>()} ; 
    /* 2) Number of ghostzones for evolved vars */
    size_t ngz { params["amr"]["n_ghostzones"].as<size_t>() } ;

    if( coord_system == "cartesian" )
    {
        /* 
        * Here we do the following: 
        * 1) We loop over all local trees and determine their vertex coordinates 
        * 2) We loop over the quadrants in each tree and determine the cell coordinates 
        * 3) We write this to device 
        */ 
        auto trees = forest.get_trees() ; 
        for( int itree=forest.first_local_tree(); itree<forest.last_local_tree(), ++itree)
        {
            auto quadrants = tree_t( trees[itree] ).get_quadrants() ; 
            /* coordinates of lower left corner vertex */  
            auto const ll_vertex_coords = conn.get_vertex_coordinates(itree,0UL) ; 
            /* coordinates of lower right corner vertex */ 
            auto const lr_vertex_coords = conn.get_vertex_coordinates(itree,2UL) ; 
            /* each tree is a square (resp. cube) in coordinate space */ 
            auto const dx = lr_vertex_coords[0] - ll_vertex_coords[0] ; 
            for( int iquad=0; iquad<quadrants.size(); ++iquad ) {
                auto quadrant = quadrant_t( qadrants[iquad] ) ; 
                auto const dx_lev = dx / ( 1UL<<quadrant.level() ) ; 
                /* coordinates of lower left corner of quadrant */
                auto const qcoords = quadrant.qcoords() ; 
                auto quad_coords = ll_vertex_coords ; 
                for( int idir=0; idir<quad_coords.size(); ++idir) quad_coords[idir] += dx_lev * qcoords[idir] ; 
                double VEC( qx{quad_coords[0]}, qy{quad_coords[1]}, qz{quad_coords[2]} ) ; 
                /* launch a tiny kernel to fill the coord array */ 
                Kokkkos::parallel_for( "fill_coords_cartesian"
                        , Kokkos::MDRange<Rank<THUNDER_NSPACEDIM>>( {VEC(0,0,0)}
                                                                  , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz )} )
                        , KOKKOS_LAMBDA ( VEC(int i, int j, int k) )
                        {
                            
                        } ) ;  
            }
        }
    } else if ( coord_system == "spherical" ) {

    } else {
        ASSERT(0, "Should never be here.") ; 
    }


}

} /* namespace thunder */ 