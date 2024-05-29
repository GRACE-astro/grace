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

#include <Kokkos_Vector.hpp>

#include <vector> 
#include <set> 

namespace grace { namespace amr {

struct simple_face_info_t 
{
    int has_polarity_flip;
    int is_ghost     ; 
    int which_face_a ; 
    int which_face_b ; 
    int64_t qid_a    ;
    int64_t qid_b    ; 
    int which_tree_a ; 
    int which_tree_b ; 

} ; 

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

struct hanging_coarse_quadrant_info_t 
{
    std::vector<int64_t> snd_quadid ; 
    std::vector<std::set<int>>     snd_procid ; 
    std::vector<int64_t> rcv_quadid ; 
    std::vector<int>     rcv_procid ; 
} ; 

struct grace_face_info_t 
{
    int n_hanging_ghost_faces{0}; 
    int n_simple_ghost_faces{0} ;
    Kokkos::vector<int64_t>             phys_boundary_info        ;
    Kokkos::vector<simple_face_info_t>  simple_interior_info      ; 
    Kokkos::vector<hanging_face_info_t> hanging_interior_info     ; 
    hanging_coarse_quadrant_info_t      coarse_hanging_quads_info ;  
} ; 

void apply_boundary_conditions() ;

void apply_boundary_conditions(grace::var_array_t<GRACE_NSPACEDIM>& vars) ;


}} /* namespace grace::amr */

#endif /* GRACE_AMR_BC_HH */