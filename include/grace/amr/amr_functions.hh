/**
 * @file amr_functions.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief This file contains free functions that are used throughtout the 
 *        code to access amr related data or trigger amr related actions.
 * @version 0.1
 * @date 2024-02-29
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

 #ifndef AMR_FUNCTIONS_HH
 #define AMR_FUNCTIONS_HH 

#include <grace_config.h>

#include <grace/utils/inline.h> 
#include <grace/utils/device.h> 

#include <grace/amr/quadrant.hh>

#include <vector>
#include <array>
#include <tuple>
#include <cstdlib>

namespace grace { namespace amr {
#if 0
struct grid_properties_view_t {
    grid_properties_view_t() 
     : _gp(forest::get().get_grid_properties())
    {}  

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    size_t nx() const { return _gp(0) ; }

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    size_t ny() const { return _gp(1) ; }

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    size_t nz() const { return _gp(2) ; }

    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    size_t ngz() const { return _gp(3) ; }

    static_readonly_view_t<size_t,4> _gp ; 
} ; 
#endif 
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
 * \ingroup amr
 * 
 * @param iquad Index of the quadrant between 
 *        0 and <code>forest::local_num_quadrants()</code>
 * @return size_t Index of the tree that owns this quadrant.
 */
int 
get_quadrant_owner(size_t iquad) ;
/**
 * @brief Get the local quadrants offset of a tree.
 * \ingroup amr
 * 
 * @param itree The tree
 */
size_t get_local_quadrants_offset(size_t itree) ;
/**
 * @brief Get a quadrant given its cumulative local index
 *        and the index of the owning tree.
 * \ingroup amr
 * 
 * @param which_tree Tree owning the quadrant. 
 * @param iquad      Quadrant cumulative local index.
 * @return quadrant_t The quadrant.
 */ 
quadrant_t  
get_quadrant(size_t which_tree, size_t iquad) ; 
/**
 * @brief Get a quadrant given its cumulative local index.
 * \ingroup amr
 * 
 * @param iquad       Quadrant cumulative local index.
 * @return quadrant_t The quadrant.
 */
quadrant_t  
get_quadrant(size_t iquad) ; 
/**
 * @brief Get local index of
 *        a quadrant.
 * \ingroup amr
 */
int64_t 
get_quadrant_locidx(quadrant_t quad);
/**
 * @brief Get local index of
 *        a quadrant.
 * \ingroup amr
 */
int64_t 
get_quadrant_locidx(p4est_quadrant_t* quad);
/**
 * @brief Determine whether coordinates flip across 
 *        tree boundary.
 * \ingroup amr
 * 
 * @param treeid Index of tree
 * @param face   Face index in z-order
 * @return int 1 if coordinates flip 0 otherwise 
 */
int trees_have_opposite_polarity( int64_t treeid, int face ); 
/**
 * @brief Free function form of <code>amr::connectivity().tree_vertex</code>.
 * \ingroup amr
 * 
 * @param which_tree Tree index 
 * @param which_vertex Vertex index in z-ordering
 * @return Array containing physical (xyz) coordinates of the vertex.
 */
std::array<double,GRACE_NSPACEDIM> 
get_tree_vertex(size_t which_tree, size_t which_vertex) ; 

/**
 * @brief Free function form of 
 *        <code>amr::connectivity().tree_spacing</code>.
 * \ingroup amr
 * 
 * @param which_tree Tree index 
 * @return Array containing physical coordinate extent of the tree in each direction.
 *         Only really makes sense for rectilinear coordinates.
 */
std::array<double,GRACE_NSPACEDIM> 
get_tree_spacing(size_t which_tree) ;
/**
 * @brief Get a vector containing the first global quadrant of each rank
 *        + 1 (the last entry is the total number of quadrant across all ranks).
 * \ingroup amr
 */
std::vector<int64_t>
get_global_quadrant_offsets() ; 

namespace detail {
extern int64_t _nx; 
extern int64_t _ny;
extern int64_t _nz;
extern int _ngz;

}

#define DECLARE_GRID_EXTENTS                              \
size_t nx, ny, nz;                                        \
std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; \
auto const ngz = grace::amr::get_n_ghosts()             ; \
auto const nq  = grace::amr::get_local_num_quadrants()   


} } /* grace::amr */ 

 #endif 