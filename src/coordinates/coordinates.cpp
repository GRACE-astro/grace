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

#include <thunder/config/config_parser.hh>

#include <thunder/data_structures/variables.hh>
#include <thunder/data_structures/macros.hh>
#include <thunder/data_structures/memory_defaults.hh>


#include <thunder/utils/thunder_utils.hh>

#include <string> 

namespace thunder { 

void fill_cell_coordinates(coord_array_t<THUNDER_NSPACEDIM>& coords, scalar_array_t<THUNDER_NSPACEDIM>& ispacing) 
{
    using namespace thunder ; 
    auto& forest = thunder::amr::forest::get()        ; 
    auto& conn   = thunder::amr::connectivity::get()  ; 
    auto& params = thunder::config_parser::get()      ;

    size_t nx {params["amr"]["npoints_block_x"].as<size_t>()} ; 
    size_t ny {params["amr"]["npoints_block_y"].as<size_t>()} ; 
    size_t nz {params["amr"]["npoints_block_z"].as<size_t>()} ; 
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
        /* launch a tiny kernel to fill the coord array */ 
        Kokkos::parallel_for( THUNDER_EXECUTION_TAG("AMR","fill_coords_spherical")
                , Kokkos::MDRangePolicy<Kokkos::Rank<THUNDER_NSPACEDIM>,default_execution_space>( {VEC(0,0,0)}
                                                            , {VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz )} )
                , KOKKOS_LAMBDA ( VEC(int i, int j, int k) )
                {
                    EXPR(
                    coords(VEC(i,j,k),0,iquad_glob) = dx_lev * qcoords[0] + ( i - ngz + 0.5 ) * dx_quad ;,
                    coords(VEC(i,j,k),1,iquad_glob) = dx_lev * qcoords[1] + ( j - ngz + 0.5 ) * dy_quad ;,
                    coords(VEC(i,j,k),2,iquad_glob) = dx_lev * qcoords[2] + ( k - ngz + 0.5 ) * dz_quad ; 
                    ) 
                    EXPR(
                    ispacing(0,iquad_glob) = 1./dx_quad ;,
                    ispacing(1,iquad_glob) = 1./dy_quad ;,
                    ispacing(2,iquad_glob) = 1./dz_quad ; 
                    ) 
                } ) ;  
    }
}
    
}

} /* namespace thunder */ 