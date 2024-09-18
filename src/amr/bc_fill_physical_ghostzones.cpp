/**
 * @file bc_fill_physical_ghostzones.cpp
 * @author  Carlo Musolino
 * @brief 
 * @date 2024-09-17
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#include <grace_config.h>

#include <grace/amr/amr_functions.hh>
#include <grace/amr/boundary_conditions.hh>
#include <grace/amr/bc_helpers.tpp>
#include <grace/amr/bc_fill_physical_ghostzones.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/device_vector.hh> 

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {
// TODO header 
void fill_physical_boundaries(
      grace::var_array_t<GRACE_NSPACEDIM>& vars 
    , grace::staggered_variable_arrays_t& staggered_vars 
    , grace::device_vector<grace::amr::grace_phys_bc_info_t> const& face_phys_bc 
    , grace::device_vector<grace::amr::grace_phys_bc_info_t> const& corner_phys_bc
    #ifdef GRACE_3D 
    , grace::device_vector<grace::amr::grace_phys_bc_info_t> const& edge_phys_bc 
    #endif
)
{
    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ;

    int nvars_cell_center = variables::get_n_evolved() ; 
    for(int ivar=0; ivar<nvars_cell_center; ++ivar){
        auto bc_type = variables::get_bc_type(ivar, grace::var_staggering_t::CELL_CENTER) ; 
        if( bc_type == "outgoing" )
        {
            auto var = Kokkos::subview( vars
                                      , VEC( Kokkos::ALL() 
                                           , Kokkos::ALL() 
                                           , Kokkos::ALL() )
                                      , ivar 
                                      , Kokkos::ALL() ) ; 
            apply_phys_bc<outgoing_bc_t>(
                  var
                , nx,ny,nz,ngz 
                , face_phys_bc
                , corner_phys_bc
                #ifdef GRACE_3D 
                , edge_phys_bc
                #endif 
            ) ; 
        } else if ( bc_type == "third_order_lagrange" ) { 
            auto var = Kokkos::subview( vars
                                      , VEC( Kokkos::ALL() 
                                           , Kokkos::ALL() 
                                           , Kokkos::ALL() )
                                      , ivar 
                                      , Kokkos::ALL() ) ; 
            apply_phys_bc<extrap_bc_t<3>>(
                  var
                , nx,ny,nz,ngz 
                , face_phys_bc
                , corner_phys_bc
                #ifdef GRACE_3D 
                , edge_phys_bc
                #endif 
            ) ; 
        } else if (bc_type == "none" ) {
            /* Nothing to do here */
        } else {
            ERROR("Unrecognized bc type for variable " << ivar << ".\n") ;
        }
    }
    
    /* Corner staggered */
    int nvars_cell_corner = variables::get_n_evolved_corner_staggered() ; 
    for(int ivar=0; ivar<nvars_cell_corner; ++ivar){
        auto bc_type = variables::get_bc_type(ivar, grace::var_staggering_t::CORNER) ; 
        if ( bc_type == "outgoing" ) { 
            auto var = Kokkos::subview( staggered_vars.corner_staggered_fields
                                      , VEC( Kokkos::ALL() 
                                           , Kokkos::ALL() 
                                           , Kokkos::ALL() )
                                      , ivar 
                                      , Kokkos::ALL() ) ; 
            apply_phys_bc<outgoing_bc_t>(
                  var
                , nx+1,ny+1,nz+1,ngz 
                , face_phys_bc
                , corner_phys_bc
                #ifdef GRACE_3D 
                , edge_phys_bc
                #endif 
            ) ;
        } else if( bc_type == "third_order_lagrange" ) {
            auto var = Kokkos::subview( staggered_vars.corner_staggered_fields
                                      , VEC( Kokkos::ALL() 
                                           , Kokkos::ALL() 
                                           , Kokkos::ALL() )
                                      , ivar 
                                      , Kokkos::ALL() ) ; 
            apply_phys_bc<extrap_bc_t<3>>(
                  var
                , nx+1,ny+1,nz+1,ngz 
                , face_phys_bc
                , corner_phys_bc
                #ifdef GRACE_3D 
                , edge_phys_bc
                #endif 
            ) ; 
        } else if (bc_type == "none" ) {
            /* Nothing to do here */
        } else {
            ERROR("Unrecognized bc type for variable " << ivar << ".\n") ;
        }
    }

    Kokkos::fence() ;
}

}} /* namespace grace::amr */