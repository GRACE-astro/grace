/**
 * @file amr_functions.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief This file contains free functions that are used throughtout the 
 *        code to access amr related data or trigger amr related actions.
 * @version 0.1
 * @date 2024-02-29
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

 #ifndef AMR_FUNCTIONS_HH
 #define AMR_FUNCTIONS_HH 

#include <thunder_config.h>

#include <thunder/utils/inline.h> 
#include <thunder/utils/device.h> 

#include <thunder/data_structures/macros.hh>
#include <thunder/amr/quadrant.hh>
#include <thunder/amr/tree.hh>
#include <thunder/amr/forest.hh> 

#include <thunder/config/config_parser.hh>

#include <array>
#include <tuple>
#include <cstdlib>

namespace thunder { namespace amr {

/**
 * @brief Get the number of grid cells per quadrant 
 *        in each direction. 
 * \ingroup amr
 * @return a tuple containing the number of grid cells per quadrant 
 *         in each direction.
 */
decltype(auto) 
get_quadrant_extents()
{
    auto& config = thunder::config_parser::get() ; 
    auto const nx = config["amr"]["npoints_block_x"].as<size_t>() ; 
    auto const ny = config["amr"]["npoints_block_y"].as<size_t>() ; 
    auto const nz = config["amr"]["npoints_block_x"].as<size_t>() ; 
    return std::make_tuple(nx,ny,nz) ;  
}

/**
 * @brief Get the number of ghost cells. 
 * \ingroup amr 
 * @return number of ghost cells. 
 */
int 
get_n_ghosts()
{
    auto& config = thunder::config_parser::get() ; 
    return config["amr"]["n_ghostzones"].as<int>() ; 
}
/**
 * @brief Find the tree that owns a quadrant 
 *        given the quadrant's cumulative local index. 
 * 
 * @param iquad Index of the quadrant between 
 *        0 and <code>forest::local_num_quadrants()</code>
 * @return size_t Index of the tree that owns this quadrant.
 */
size_t 
get_quadrant_owner(size_t iquad)
{
    auto& forest = thunder::amr::forest::get() ;
    for(size_t itree=forest.first_local_tree();
        itree <= forest.last_local_tree(); 
        itree+=1UL)
    {
        if( forest.tree(itree).quadrants_offset > iquad ){
            return itree ; 
        }
    }
    ASSERT_DBG(0, 
    "In get_quadrant_owner: " << iquad << " is not owned by any local tree.") ;
    return -1 ; 
}
/**
 * @brief Get a quadrant given its cumulative local index
 *        and the index of the owning tree.
 * 
 * @param which_tree Tree owning the quadrant. 
 * @param iquad      Quadrant cumulative local index.
 * @return quadrant_t The quadrant.
 */
quadrant_t THUNDER_ALWAYS_INLINE 
get_quadrant(size_t which_tree, size_t iquad)
{
    tree_t tree = thunder::amr::forest::get().tree(which_tree) ;
    return tree.quadrant(iquad-tree.quadrants_offset()) ; 
}
/**
 * @brief Free function form of <code>amr::connectivity().tree_vertex</code>
 * 
 * @param which_tree Tree index 
 * @param which_vertex Vertex index in z-ordering
 * @return Array containing physical (xyz) coordinates of the vertex.
 */
std::array<double,THUNDER_NSPACEDIM> THUNDER_ALWAYS_INLINE
get_tree_vertex(size_t which_tree, size_t which_vertex)
{
    return thunder::amr::connectivity::get().vertex_coordinates(which_tree,which_vertex);
}

/**
 * @brief Free function form of 
 *        <code>amr::connectivity().tree_spacing</code>
 * 
 * @param which_tree Tree index 
 * @return Array containing physical coordinate extent of the tree in each direction.
 *         Only really makes sense for rectilinear coordinates.
 */
std::array<double,THUNDER_NSPACEDIM> THUNDER_ALWAYS_INLINE
get_tree_spacing(size_t which_tree, size_t which_vertex)
{
    return thunder::amr::connectivity::get().tree_coordinate_exents(which_tree);
}



} } /* thunder::amr */ 

 #endif 