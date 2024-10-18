/**
 * @file boundary_conditions.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-21
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
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

#ifndef GRACE_AMR_BC_HH 
#define GRACE_AMR_BC_HH 

#include <grace/data_structures/variable_properties.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/utils/device/device_vector.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <Kokkos_Vector.hpp>

#include <vector> 
#include <set> 

namespace grace { namespace amr {
/**
 * @brief Struct containing information about
 *        a simple interior face.
 * \ingroup amr
 */
struct simple_face_info_t 
{
    int has_polarity_flip;
    int is_ghost     ; 
    int8_t which_face_a ; 
    int8_t which_face_b ; 
    int64_t qid_a    ;
    int64_t qid_b    ; 
    int which_tree_a ; 
    int which_tree_b ; 

} ; 
/**
 * @brief Struct containing information about a 
 *        hanging interior face.
 * \ingroup amr
 */
struct hanging_face_info_t 
{
    int has_polarity_flip;
    int level_coarse       ;
    int level_fine         ; 
    int8_t which_face_coarse ; 
    int8_t which_face_fine   ; 
    int which_tree_coarse ; 
    int which_tree_fine   ;
    int8_t is_ghost_coarse ;
    int8_t is_ghost_fine[P4EST_CHILDREN/2] ; 
    int64_t qid_coarse ; 
    int64_t qid_fine[P4EST_CHILDREN/2] ; 
} ; 
/**
 * @brief Struct containing information about
 *        a simple interior edge.
 * \ingroup amr
 */
struct simple_edge_info_t 
{
    int is_ghost     ; 
    int8_t which_edge_a ; 
    int8_t which_edge_b ; 
    int64_t qid_a    ;
    int64_t qid_b    ; 
    int which_tree_a ; 
    int which_tree_b ; 

} ; 
/**
 * @brief Struct containing information about a 
 *        hanging interior edge.
 * \ingroup amr
 */
struct hanging_edge_info_t 
{
    int level_coarse       ;
    int level_fine         ; 
    int8_t which_edge_coarse ; 
    int8_t which_edge_fine   ; 
    int which_tree_coarse ; 
    int which_tree_fine   ;
    int8_t is_ghost_coarse ;
    int8_t is_ghost_fine[2] ; 
    int64_t qid_coarse ; 
    int64_t qid_fine[2] ; 
} ; 
/**
 * @brief Struct containing information about
 *        a simple interior corner.
 * \ingroup amr
 */
struct simple_corner_info_t 
{
    int is_ghost     ; 
    int8_t which_corner_a ; 
    int8_t which_corner_b ; 
    int64_t qid_a    ;
    int64_t qid_b    ; 
    int which_tree_a ; 
    int which_tree_b ; 

} ; 
/**
 * @brief Struct containing information about a 
 *        hanging interior corner.
 * \ingroup amr
 */
struct hanging_corner_info_t 
{
    int level_coarse       ;
    int level_fine         ; 
    int8_t which_corner_coarse ; 
    int8_t which_corner_fine   ; 
    int which_tree_coarse ; 
    int which_tree_fine   ;
    int8_t is_ghost_coarse ;
    int8_t is_ghost_fine   ; 
    int64_t qid_coarse ; 
    int64_t qid_fine   ; 
} ; 
/**
 * @brief Struct containing the information about 
 *        the coarse quadrants touching hanging faces.
 * \ingroup amr
 */
struct hanging_coarse_quadrant_info_t 
{
    std::vector<int64_t> snd_quadid ; 
    std::vector<std::set<int>>     snd_procid ; 
    std::vector<int64_t> rcv_quadid ; 
    std::vector<int>     rcv_procid ; 
} ; 
/**
 * @brief Struct containing information about which 
 *        faces in the local forest are hanging.
 * \ingroup amr
 * The default constructor initializes the underlying 
 * container to the correct size and fills it with zeroes.
 */
struct hanging_fine_face_info_t 
{
    std::vector<std::uint8_t> _is_hanging ; 

    hanging_fine_face_info_t() 
     : _is_hanging( P4EST_FACES * amr::get_local_num_quadrants(), 0)
    {}

    std::uint8_t& operator[] (size_t const& i)
    {
        return _is_hanging[i] ; 
    }

    std::uint8_t operator[] (size_t const& i) const 
    {
        return _is_hanging[i] ; 
    }
} ; 
/**
 * @brief Struct containing information about which 
 *        edges in the local forest are hanging.
 * \ingroup amr
 * The default constructor initializes the underlying 
 * container to the correct size and fills it with zeroes.
 */
struct hanging_fine_edge_info_t 
{
    std::vector<std::uint8_t> _is_hanging ; 

    hanging_fine_edge_info_t() 
     : _is_hanging( P8EST_EDGES * amr::get_local_num_quadrants(), 0)
    {}

    std::uint8_t& operator[] (size_t const& i)
    {
        return _is_hanging[i] ; 
    }

    std::uint8_t operator[] (size_t const& i) const 
    {
        return _is_hanging[i] ; 
    }
} ;
/**
 * @brief 
 * 
 */
struct grace_phys_bc_info_t {
    int64_t qid ; //!< Index of quadrant facing the outside of the grid
    int8_t  dir_x, dir_y, dir_z ; //!< Direction of boundary 
    int8_t face, edge, corner   ; 
} ; 

/**
 * @brief Collection of informations about face neighbors.
 * \ingroup amr
 */
struct grace_face_info_t 
{
    int n_hanging_ghost_faces{0}; 
    int n_simple_ghost_faces{0} ;
    Kokkos::vector<simple_face_info_t>         simple_interior_info       ; 
    Kokkos::vector<hanging_face_info_t>        hanging_interior_info      ;
    grace::device_vector<grace_phys_bc_info_t> phys_boundary_info         ; 
} ; 
/**
 * @brief Collection of informations about edge neighbors.
 * \ingroup amr
 */
struct grace_edge_info_t 
{
    int n_hanging_ghost_edges{0}; 
    int n_simple_ghost_edges{0} ;
    int n_exterior_edges{0}     ; 
    int n_edges_total{0}     ; 
    Kokkos::vector<simple_edge_info_t>   simple_interior_info       ; 
    Kokkos::vector<hanging_edge_info_t>  hanging_interior_info      ;
    grace::device_vector<grace_phys_bc_info_t> phys_boundary_info  ; 
} ; 
/**
 * @brief Collection of informations about corner neighbors.
 * \ingroup amr
 */
struct grace_corner_info_t 
{
    int n_hanging_ghost_corners{0}; 
    int n_simple_ghost_corners{0} ;
    Kokkos::vector<simple_corner_info_t>  simple_interior_info      ; 
    Kokkos::vector<hanging_corner_info_t> hanging_interior_info     ;
    grace::device_vector<grace_phys_bc_info_t>  phys_boundary_info  ; 
} ; 
/**
 * @brief The user data passed to <code>p4est_iterate</code>
 * \ingroup amr
 * This struct holds all the necessary information to apply 
 * boundary conditions and fill ghostzones.
 */
struct grace_neighbor_info_t 
{
    grace_face_info_t   face_info   ;
    grace_corner_info_t corner_info ; 
    #ifdef GRACE_3D 
    grace_edge_info_t   edge_info   ; 
    #endif
    hanging_coarse_quadrant_info_t      coarse_hanging_quads_info ; 
    hanging_fine_face_info_t            fine_hanging_faces_info   ; 
    hanging_fine_edge_info_t            fine_hanging_edges_info   ;
} ; 

struct grace_transfer_context_t 
{ 
    std::vector<sc_MPI_Request> _requests ; 
    std::vector<grace::var_array_t<GRACE_NSPACEDIM>> _buffs ; 
    std::vector<grace::staggered_variable_arrays_t>  _staggered_buffs ; 
    void reset() { 
        for( auto& x: _buffs ) {
            Kokkos::realloc(x, VEC(0,0,0),0,0) ; 
        }
        for( auto& x: _staggered_buffs) {
            x.realloc(VEC(0,0,0),0,0,0,0,0) ; 
        }
        _buffs.clear() ; 
        _staggered_buffs.clear() ; 
        _requests.clear() ;
    } ; 
} ; 

/**
 * @brief Apply all boundary conditions and fill ghostzones on state array.
 * \ingroup amr
 * This function fills all the ghost-cells in the <code>state</code> array.
 * This includes applying physical boundary conditions at domain edges, as 
 * well as filling all internal ghost-zones across simple and hanging faces.
 * The state array needs to be in a valid state at all interior points when entering 
 * this function. No assumptions are made on the content of the ghostzones of each
 * quadrant and each variable for the state array when entering this function. Auxiliries
 * and scratch state are left untouched by this function, unless a non-trivial boundary 
 * condition is requested on an auxiliary variable. When this function returns, \b all 
 * ghostzones for all quadrants are in a valid state for the <code>state</code> array.
 * All interior ghost-zones operations are guaranteed to be second order accurate, total 
 * variation diminishing, and volume average preserving.
 */
void apply_boundary_conditions() ;
/**
 * @brief Apply all boundary conditions on the var array.
 * \ingroup amr
 * @param vars The state array where BCs are applied.
 * @param staggered_vars The staggered variable state array where BCs should be applied. 
 * Specialized version of \ref apply_boundary_conditions which allows 
 * the caller to specify which state array needs its ghostzones to be filled.
 */
void apply_boundary_conditions( grace::var_array_t<GRACE_NSPACEDIM>& vars
                              , grace::staggered_variable_arrays_t& staggered_vars) ;


}} /* namespace grace::amr */

#endif /* GRACE_AMR_BC_HH */