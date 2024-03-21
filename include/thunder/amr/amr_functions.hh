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

#include <vector>
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
std::tuple<size_t,size_t,size_t> get_quadrant_extents() ; 

/**
 * @brief Get the number of ghost cells. 
 * \ingroup amr 
 * @return number of ghost cells. 
 */
int 
get_n_ghosts() ; 

/**
 * @brief Get the number of local quadrants. 
 * \ingroup amr 
 * @return number quadrants on this rank. 
 */
size_t 
get_local_num_quadrants() ; 

/**
 * @brief Find the tree that owns a quadrant 
 *        given the quadrant's cumulative local index. 
 * 
 * @param iquad Index of the quadrant between 
 *        0 and <code>forest::local_num_quadrants()</code>
 * @return size_t Index of the tree that owns this quadrant.
 */
size_t 
get_quadrant_owner(size_t iquad) ;
/**
 * @brief Get a quadrant given its cumulative local index
 *        and the index of the owning tree.
 * 
 * @param which_tree Tree owning the quadrant. 
 * @param iquad      Quadrant cumulative local index.
 * @return quadrant_t The quadrant.
 */
quadrant_t  
get_quadrant(size_t which_tree, size_t iquad) ; 
/**
 * @brief Get a quadrant given its cumulative local index.
 * 
 * @param iquad       Quadrant cumulative local index.
 * @return quadrant_t The quadrant.
 */
quadrant_t  
get_quadrant(size_t iquad) ; 
/**
 * @brief Get local index of
 *        a quadrant 
 */
int64_t 
get_quadrant_locidx(quadrant_t quad);
/**
 * @brief Get local index of
 *        a quadrant 
 */
int64_t 
get_quadrant_locidx(p4est_quadrant_t* quad);
/**
 * @brief For halo quadrants: get owner mpi 
 *        rank.
 * \cond thunder_detail
 */
int 
get_halo_quad_owner(quadrant_t& quad);
/**
 * @brief For halo quadrants: get owner mpi 
 *        rank.
 * \cond thunder_detail
 */
int 
get_halo_quad_owner(p4est_quadrant_t* quad);
/**
 * @brief Free function form of <code>amr::connectivity().tree_vertex</code>
 * 
 * @param which_tree Tree index 
 * @param which_vertex Vertex index in z-ordering
 * @return Array containing physical (xyz) coordinates of the vertex.
 */
std::array<double,THUNDER_NSPACEDIM> 
get_tree_vertex(size_t which_tree, size_t which_vertex) ; 

/**
 * @brief Free function form of 
 *        <code>amr::connectivity().tree_spacing</code>
 * 
 * @param which_tree Tree index 
 * @return Array containing physical coordinate extent of the tree in each direction.
 *         Only really makes sense for rectilinear coordinates.
 */
std::array<double,THUNDER_NSPACEDIM> 
get_tree_spacing(size_t which_tree) ;
/**
 * @brief Get a vector containing the first global quadrant of each rank
 *        + 1 (the last entry is the total number of quadrant across all ranks).
 */
std::vector<int64_t>
get_global_quadrant_offsets() ; 
/**
 * @brief Get physical coordinates of a point inside a local cell.
 * 
 * @param icell Index of the local cell 
 * @param local_coords Logical coordinates \f$[0,1]^{d}\f$ of the points inside the cell
 * @param include_gzs Include ghostzones in the calculation
 * 
 */
std::array<double, THUNDER_NSPACEDIM> 
get_physical_coordinates( size_t icell
                        , std::array<double,THUNDER_NSPACEDIM> const& local_coords = {VEC(0.,0.,0.)} 
                        , bool include_gzs=false) ; 
/**
 * @brief Get physical coordinates of a point inside a local cell, given 
 *        the quadrant and spatial indices.
 * 
 * @param ijk Spatial indices of cell inside the quadrant
 * @param nq  Local quadrant index  
 * @param local_coords Logical coordinates \f$[0,1]^{d}\f$ of the points inside the cell
 * @param include_gzs Include ghostzones in the calculation
 * 
 * NB: Including ghostzones means that \f$ijk={0,0(,0)}\f$ has negative logical coordinates,
 *     and the logical coordinates origin lies at \f$ijk={ngz,ngz(,ngz)}\f$.
 */
std::array<double, THUNDER_NSPACEDIM> 
get_physical_coordinates( std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
                        , size_t nq 
                        , std::array<double,THUNDER_NSPACEDIM> const& local_coords = {VEC(0.,0.,0.)} 
                        , bool include_gzs=false) ;

} } /* thunder::amr */ 

 #endif 