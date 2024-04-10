/**
 * @file cartesian_coordinate_systems.hh
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

#ifndef THUNDER_AMR_CARTESIAN_COORDINATES_SYSTEMS_HH 
#define THUNDER_AMR_CARTESIAN_COORDINATES_SYSTEMS_HH

#include <thunder_config.h>

#include <Kokkos_Core.hpp>

#include <thunder/utils/thunder_utils.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/data_structures/thunder_data_structures.hh>

#include<array>

namespace thunder { 

struct cartesian_device_coordinate_system_impl_t
{
    cartesian_device_coordinate_system_impl_t( Kokkos::View<double*,thunder::default_space> vertices
                                             , Kokkos::View<double*,thunder::default_space> spacings )
        : tree_vertices_(vertices), tree_spacings_(spacings) 
    {} ; 
    
    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    get_physical_coordinates( int itree, double * l_coords, double * p_coords) const
    {
        EXPR(
        p_coords[0] = 
            tree_vertices_(THUNDER_NSPACEDIM * itree + 0UL) + l_coords[0] * tree_spacings_(THUNDER_NSPACEDIM * itree + 0UL);,
        p_coords[1] = 
            tree_vertices_(THUNDER_NSPACEDIM * itree + 1UL) + l_coords[1] * tree_spacings_(THUNDER_NSPACEDIM * itree + 1UL);,
        p_coords[2] = 
            tree_vertices_(THUNDER_NSPACEDIM * itree + 2UL) + l_coords[2] * tree_spacings_(THUNDER_NSPACEDIM * itree + 2UL);
        )
        return ;
    };

    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    get_logical_coordinates( int itree, double * p_coords, double * l_coords) const 
    {
        EXPR(
        l_coords[0] = 
            (p_coords[0] - tree_vertices_(THUNDER_NSPACEDIM * itree + 0UL)) / tree_spacings_(THUNDER_NSPACEDIM * itree + 0UL);,
        l_coords[1] = 
            (p_coords[1] - tree_vertices_(THUNDER_NSPACEDIM * itree + 1UL)) / tree_spacings_(THUNDER_NSPACEDIM * itree + 1UL);,
        l_coords[2] = 
            (p_coords[2] - tree_vertices_(THUNDER_NSPACEDIM * itree + 2UL)) / tree_spacings_(THUNDER_NSPACEDIM * itree + 2UL);

        )
        return ;
    };

    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    transfer_coordinates( int tree_a, int tree_b, 
                          int face_a, int face_b,
                          double * l_coords_a, double * l_coords_b  )
    {
        get_physical_coordinates(tree_a, l_coords_a, l_coords_a) ; 
        get_logical_coordinates(tree_b, l_coords_a, l_coords_b ) ; 
    }
 private:
    Kokkos::View<double*,thunder::default_space> tree_vertices_, tree_spacings_ ;

} ; 

class cartesian_coordinate_system_impl_t 
{
 public: 

    std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
    get_physical_coordinates( 
          int const itree
        , std::array<double, THUNDER_NSPACEDIM> const& logical_coordinates );  

    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates(
           std::array<size_t, THUNDER_NSPACEDIM> const& ijk
        , int64_t q 
        , std::array<double, THUNDER_NSPACEDIM> const& cell_coordinates
        , bool use_ghostzones 
    ) ;

    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_physical_coordinates(
           std::array<size_t, THUNDER_NSPACEDIM> const& ijk
        , int64_t q 
        , bool use_ghostzones 
    ) ;

    std::array<double, THUNDER_NSPACEDIM>
    THUNDER_HOST get_logical_coordinates(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , int64_t q 
    , std::array<double, THUNDER_NSPACEDIM> const& cell_coordinates
    , bool use_ghostzones) ;

    std::array<double,THUNDER_NSPACEDIM> 
    THUNDER_HOST get_logical_coordinates(
          int itree
        , std::array<double,THUNDER_NSPACEDIM> const& physical_coordinates
    ) ; 

    std::array<double,THUNDER_NSPACEDIM> 
    THUNDER_HOST get_logical_coordinates(
        std::array<double,THUNDER_NSPACEDIM> const& physical_coordinates
    ) ;

    std::array<double, THUNDER_NSPACEDIM*THUNDER_NSPACEDIM>
    THUNDER_HOST get_jacobian(
        std::array<double,THUNDER_NSPACEDIM> const& physical_coordinates 
    ) ; 

    double
    THUNDER_HOST get_cell_volume(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q
    , bool use_ghostzones); 

    double
    THUNDER_HOST get_cell_volume(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& dxl 
    , bool use_ghostzones) ;

    cartesian_device_coordinate_system_impl_t THUNDER_ALWAYS_INLINE 
    get_device_coord_system() {
        return cartesian_device_coordinate_system_impl_t{tree_vertices_,tree_spacings_} ;
    }

 private:        
    Kokkos::View<double*,thunder::default_space> tree_vertices_, tree_spacings_ ;
    
    cartesian_coordinate_system_impl_t() ; 

    static constexpr size_t longevity = THUNDER_COORDINATE_SYSTEM ; 

    friend class utils::singleton_holder<cartesian_coordinate_system_impl_t, memory::default_create> ; 
    friend class memory::new_delete_creator<cartesian_coordinate_system_impl_t,memory::new_delete_allocator> ;
} ; 



} /* namespace thunder */

#endif /* THUNDER_AMR_COORDINATES_SYSTEMS_HH */