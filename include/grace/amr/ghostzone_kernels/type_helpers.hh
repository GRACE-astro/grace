/**
 * @file type_helpers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-09-05
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
#ifndef GRACE_AMR_TYPE_HELPERS_HH
#define GRACE_AMR_TYPE_HELPERS_HH

#include <grace_config.h>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variable_properties.hh>

#include <Kokkos_Core.hpp>

#include <tuple>

namespace grace {

struct view_alias_t {
    grace::var_array_t* _view_ptr, _view_ptr_p ; 
    grace::staggered_variable_arrays_t* _stag_view_ptr, _stag_view_ptr_p ;
    template< var_staggering_t staggering >
    grace::var_array_t get() {
        if constexpr ( staggering == STAG_CENTER ) {
            return *_view_ptr; 
        } else if constexpr ( staggering == STAG_FACEX ) {
            return (*_stag_view_ptr).face_staggered_fields_x ; 
        } else if constexpr ( staggering == STAG_FACEY ) {
            return (*_stag_view_ptr).face_staggered_fields_y ; 
        } else if constexpr ( staggering == STAG_FACEZ ) {
            return (*_stag_view_ptr).face_staggered_fields_z ; 
        } else if constexpr ( staggering == STAG_EDGEXY ) {
            return (*_stag_view_ptr).edge_staggered_fields_xy ; 
        } else if constexpr ( staggering == STAG_EDGEXZ ) {
            return (*_stag_view_ptr).edge_staggered_fields_xz ; 
        } else if constexpr ( staggering == STAG_EDGEYZ ) {
            return (*_stag_view_ptr).edge_staggered_fields_yz ; 
        } else {
            return (*_stag_view_ptr).corner_staggered_fields ; 
        }
        
    }

    grace::var_array_t get_p() {
        if constexpr ( staggering == STAG_CENTER ) {
            return *_view_ptr_p; 
        } else if constexpr ( staggering == STAG_FACEX ) {
            return (*_stag_view_ptr_p).face_staggered_fields_x ; 
        } else if constexpr ( staggering == STAG_FACEY ) {
            return (*_stag_view_ptr_p).face_staggered_fields_y ; 
        } else if constexpr ( staggering == STAG_FACEZ ) {
            return (*_stag_view_ptr_p).face_staggered_fields_z ; 
        } else if constexpr ( staggering == STAG_EDGEXY ) {
            return (*_stag_view_ptr_p).edge_staggered_fields_xy ; 
        } else if constexpr ( staggering == STAG_EDGEXZ ) {
            return (*_stag_view_ptr_p).edge_staggered_fields_xz ; 
        } else if constexpr ( staggering == STAG_EDGEYZ ) {
            return (*_stag_view_ptr_p).edge_staggered_fields_yz ; 
        } else {
            return (*_stag_view_ptr_p).corner_staggered_fields ; 
        }
        
    }

    view_alias_t(
        grace::var_array_t* ptr, 
        grace::staggered_variable_arrays_t* stag_ptr 
    ) : _view_ptr(ptr)
      , _view_ptr_p(nullptr)
      , _stag_view_ptr(stag_ptr)
      , _stag_view_ptr_p(nullptr)
    {} 

    view_alias_t(
        grace::var_array_t* ptr, 
        grace::var_array_t* ptr_p, 
        grace::staggered_variable_arrays_t* stag_ptr,
        grace::staggered_variable_arrays_t* stag_ptr_p
    ) : _view_ptr(ptr)
      , _view_ptr_p(ptr_p)
      , _stag_view_ptr(stag_ptr)
      , _stag_view_ptr_p(stag_ptr_p)
    {} 
} ; 

template< typename T >
using readonly_view_t = Kokkos::View<const T*, grace::default_space, Kokkos::MemoryTraits<Kokkos::RandomAccess>> ;

template< typename T, size_t N >
using static_readonly_view_t = Kokkos::View<const T[N], grace::default_space, Kokkos::MemoryTraits<Kokkos::RandomAccess>> ;

template< typename T, size_t N >
using readonly_twod_view_t = Kokkos::View<const T*[N], grace::default_space, Kokkos::MemoryTraits<Kokkos::RandomAccess>> ;


using gpu_task_desc_t = std::pair<size_t, uint8_t> ; 
using gpu_hanging_task_desc_t = std::tuple<size_t,uint8_t,uint8_t> ; 

using bucket_t = std::array<std::vector<gpu_task_desc_t>,3> ; 
using hang_bucket_t = std::array<std::vector<gpu_hanging_task_desc_t>,3> ; 

namespace detail {

inline constexpr std::array<const char*,3> elem_kind_names = {
    "face", "edge", "corner"
} ; 

}

}


#endif /*GRACE_AMR_TYPE_HELPERS_HH*/