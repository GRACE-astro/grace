/**
 * @file coordinate_systems.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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

#include <thunder/amr/thunder_amr.hh> 
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/coordinates/cartesian_coordinate_systems.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/data_structures/macros.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/errors/error.hh> 

#include <array> 
#include <cstring>

namespace thunder { 

cartesian_coordinate_system_impl_t::cartesian_coordinate_system_impl_t()
{
    using namespace thunder ;
    using namespace Kokkos ; 
    
    int ntrees = amr::connectivity::get().get()->num_trees;

    tree_vertices_ =
        View<double*, default_space>( "device_coords_tree_vertices"
                                    , THUNDER_NSPACEDIM*ntrees ) ;
    tree_spacings_ =
        View<double*, default_space>( "device_coords_tree_vertices"
                                    , THUNDER_NSPACEDIM*ntrees ) ;

    auto h_tree_spacings = create_mirror_view(tree_spacings_) ;
    auto h_tree_vertices = create_mirror_view(tree_vertices_) ; 
    for(int itree=0; itree<ntrees; ++itree)
    {
        auto const _vertex = amr::get_tree_vertex(itree,0UL) ;
        auto const dx      = amr::get_tree_spacing(itree)    ; 
        for(int idim=0; idim<THUNDER_NSPACEDIM; ++idim){
            h_tree_vertices(THUNDER_NSPACEDIM*itree+idim) = _vertex[idim] ; 
            h_tree_spacings(THUNDER_NSPACEDIM*itree+idim) = dx[idim]      ; 
        }
         
    }

    deep_copy(tree_vertices_,h_tree_vertices);
    deep_copy(tree_spacings_,h_tree_spacings);
}


std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
cartesian_coordinate_system_impl_t::get_physical_coordinates(
      int const itree
    , std::array<double, THUNDER_NSPACEDIM> const& logical_coordinates ) 
{
    auto const tree_coords = amr::get_tree_vertex(itree,0UL) ; 
    auto const dx_tree     = amr::get_tree_spacing(itree) ;
    return {VEC(
        logical_coordinates[0] * dx_tree[0] + tree_coords[0],
        logical_coordinates[1] * dx_tree[1] + tree_coords[1],
        logical_coordinates[2] * dx_tree[2] + tree_coords[2]
    )} ; 
}

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
cartesian_coordinate_system_impl_t::get_physical_coordinates(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , int64_t q 
    , std::array<double, THUNDER_NSPACEDIM> const& cell_coordinates
    , bool use_ghostzones )
{
    using namespace thunder ;

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int64_t nq = amr::get_local_num_quadrants()      ;
    int ngz = amr::get_n_ghosts()                    ; 

    int64_t itree = amr::get_quadrant_owner(q)   ; 
    amr::quadrant_t quad = amr::get_quadrant(itree,q) ; 

    auto const dx_quad  = 1./(1<<quad.level()) ; 
    auto const qcoords = quad.qcoords()     ; 

    EXPR(
    auto const dx_cell = dx_quad / nx ;, 
    auto const dy_cell = dx_quad / ny ;,
    auto const dz_cell = dx_quad / nz ;
    ) 

    std::array<double,THUNDER_NSPACEDIM> lcoords {
        VEC(
            qcoords[0] * dx_quad + (ijk[0] + cell_coordinates[0] - use_ghostzones * ngz) * dx_cell, 
            qcoords[1] * dx_quad + (ijk[1] + cell_coordinates[1] - use_ghostzones * ngz) * dy_cell, 
            qcoords[2] * dx_quad + (ijk[2] + cell_coordinates[2] - use_ghostzones * ngz) * dz_cell
        ) 
    } ; 

    return get_physical_coordinates(itree, lcoords) ; 
}


std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
cartesian_coordinate_system_impl_t::get_physical_coordinates(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , int64_t q 
    , bool use_ghostzones )
{
    return get_physical_coordinates(ijk,q,{VEC(0.5,0.5,0.5)},use_ghostzones);
} 

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
cartesian_coordinate_system_impl_t::get_logical_coordinates(
      int itree
    , std::array<double, THUNDER_NSPACEDIM> const& physical_coordinates )
{ 
    auto const tree_coords = amr::get_tree_vertex(itree,0UL) ; 
    auto const dx_tree     = amr::get_tree_spacing(itree) ;
    return {VEC(
        (physical_coordinates[0] - tree_coords[0])/dx_tree[0],
        (physical_coordinates[1] - tree_coords[1])/dx_tree[1],
        (physical_coordinates[2] - tree_coords[2])/dx_tree[2]
    )} ; 
}

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
cartesian_coordinate_system_impl_t::get_logical_coordinates(
    std::array<double, THUNDER_NSPACEDIM> const& physical_coordinates )
{
    int ntrees = amr::connectivity::get().get()->num_trees; 

    for( int itree=0; itree<ntrees; ++itree)
    {
        auto const tree_000 = amr::get_tree_vertex(itree,0UL) ; 
        auto const tree_100 = amr::get_tree_vertex(itree,1UL) ; 
        auto const tree_010 = amr::get_tree_vertex(itree,2UL) ; 
        #ifdef THUNDER_3D 
        auto const tree_001 = amr::get_tree_vertex(itree,4UL) ; 
        #endif 
        if(
                physical_coordinates[0] > tree_000[0]  
            and physical_coordinates[0] < tree_100[0]  
            and physical_coordinates[1] > tree_000[1]  
            and physical_coordinates[1] < tree_010[1]  
            #ifdef THUNDER_3D 
            and physical_coordinates[2] > tree_000[2]  
            and physical_coordinates[2] < tree_001[2]  
            #endif 
        ){
            return {VEC(
            (physical_coordinates[0] - tree_000[0])/(tree_100[0]-tree_000[0]),
            (physical_coordinates[1] - tree_000[1])/(tree_010[1]-tree_000[1]),
            (physical_coordinates[2] - tree_000[2])/(tree_001[2]-tree_000[2])
            )} ; 
        }
    }
    ERROR("Point (" << physical_coordinates[0] 
    << "," << physical_coordinates[1] 
    #ifdef THUNDER_3D 
    << physical_coordinates[2] 
    #endif 
    << ") is ouside the grid."
    << "If you're seeking the coordinates "
    << "of a point in the ghost-zones "
    << "please use the version of this function "
    << "that takes the tree index as input.") ; 
}

} /* namespace thunder */ 