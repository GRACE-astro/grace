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
#include <grace/utils/device/device_vector.hh> 

#include <grace/data_structures/variables.hh>
#include <grace/coordinates/coordinates.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

void fill_physical_boundaries(
      grace::var_array_t<GRACE_NSPACEDIM>& vars 
    , grace::var_array_t<GRACE_NSPACEDIM>& old_vars 
    , grace::staggered_variable_arrays_t& staggered_vars 
    , grace::staggered_variable_arrays_t& old_staggered_vars 
    , grace::device_vector<grace::amr::grace_phys_bc_info_t>& face_phys_bc 
    , grace::device_vector<grace::amr::grace_phys_bc_info_t>& corner_phys_bc
    #ifdef GRACE_3D 
    , grace::device_vector<grace::amr::grace_phys_bc_info_t>& edge_phys_bc 
    #endif
    , double const dt, double const dtfact
)
{
    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ;

    int nvars_cell_center = variables::get_n_evolved() ; 


    auto& idx = grace::variable_list::get().getinvspacings() ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; // NB these are corner coords! 
    grace::fill_physical_coordinates(pcoords, {VEC(true,true,true)}) ;

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
                , var
                , nx,ny,nz,ngz 
                , face_phys_bc
                , corner_phys_bc
                #ifdef GRACE_3D 
                , edge_phys_bc
                #endif 
                , outgoing_bc_t{}
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
                , var
                , nx,ny,nz,ngz 
                , face_phys_bc
                , corner_phys_bc
                #ifdef GRACE_3D 
                , edge_phys_bc
                #endif 
                , extrap_bc_t<3>{}
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
                , var
                , nx+1,ny+1,nz+1,ngz 
                , face_phys_bc
                , corner_phys_bc
                #ifdef GRACE_3D 
                , edge_phys_bc
                #endif 
                , outgoing_bc_t{}
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
                , var
                , nx+1,ny+1,nz+1,ngz 
                , face_phys_bc
                , corner_phys_bc
                #ifdef GRACE_3D 
                , edge_phys_bc
                #endif 
                , extrap_bc_t<3>{}
            ) ; 
        } else if (bc_type == "none" ) {
            /* Nothing to do here */
        } else {
            ERROR("Unrecognized bc type for variable " << ivar << ".\n") ;
        }
    }

    auto const apply_sommerfeld_bc = [&] (
        int ivar, 
        double const f0, double const v0 
    ) {
        auto dst_var = Kokkos::subview( staggered_vars.corner_staggered_fields
            , VEC( Kokkos::ALL() 
                 , Kokkos::ALL() 
                 , Kokkos::ALL() )
            , ivar 
            , Kokkos::ALL() ) ; 
        auto src_var = Kokkos::subview( old_staggered_vars.corner_staggered_fields
                , VEC( Kokkos::ALL() 
                     , Kokkos::ALL() 
                     , Kokkos::ALL() )
                , ivar 
                , Kokkos::ALL() ) ; 
        apply_phys_bc<extrap_bc_t<3>>(
                dst_var 
                , src_var
                , nx+1,ny+1,nz+1,ngz 
                , face_phys_bc
                , corner_phys_bc
                #ifdef GRACE_3D 
                , edge_phys_bc
                #endif
                , extrap_bc_t<3>{}
            ) ; 
            apply_phys_bc<sommerfeld_bc_t>(
                dst_var 
            , src_var
            , nx+1,ny+1,nz+1,ngz 
            , face_phys_bc
            , corner_phys_bc
            #ifdef GRACE_3D 
            , edge_phys_bc
            #endif 
            , sommerfeld_bc_t{idx, pcoords, dt, dtfact, f0, v0, VEC(nx+1,ny+1,nz+1) ,ngz}
        ) ;
    } ;
    double const v0_h = sqrt(2.) ;  
    // Metric 
    apply_sommerfeld_bc(
        GTXX,   1.0, 1.0
    ) ; 
    apply_sommerfeld_bc(
        GTXY,   0.0, 1.0
    ) ;
    apply_sommerfeld_bc(
        GTXZ,   0.0, 1.0
    ) ;
    apply_sommerfeld_bc(
        GTYY,   1.0, 1.0
    ) ;
    apply_sommerfeld_bc(
        GTYZ,   0.0, 1.0
    ) ;
    apply_sommerfeld_bc(
        GTZZ,   1.0, 1.0
    ) ;

    // Conffact 
    apply_sommerfeld_bc(
        PHI, 1., v0_h
    ) ;

    // Gamma 
    apply_sommerfeld_bc(
        GAMMAX, 0., 1.
    ) ;
    apply_sommerfeld_bc(
        GAMMAY, 0., 1.
    ) ;   
    apply_sommerfeld_bc(
        GAMMAZ, 0., 1.
    ) ;  

    // Curvature trace 
    apply_sommerfeld_bc(
        K, 0., 1.
    ) ;  

    // Extrinsic curvature (traceless conformal)
    apply_sommerfeld_bc(
        ATXX,   0.0, 1.0
    ) ; 
    apply_sommerfeld_bc(
        ATXY,   0.0, 1.0
    ) ;
    apply_sommerfeld_bc(
        ATXZ,   0.0, 1.0
    ) ;
    apply_sommerfeld_bc(
        ATYY,   0.0, 1.0
    ) ;
    apply_sommerfeld_bc(
        ATYZ,   0.0, 1.0
    ) ;
    apply_sommerfeld_bc(
        ATZZ,   0.0, 1.0
    ) ;

    // lapse 
    apply_sommerfeld_bc(
        ALP,   1.0, v0_h
    ) ;

    // shift 
    apply_sommerfeld_bc(
        BETAX, 0., 1.
    ) ;
    apply_sommerfeld_bc(
        BETAY, 0., 1.
    ) ;   
    apply_sommerfeld_bc(
        BETAZ, 0., 1.
    ) ;

    // Gamma driver 
    apply_sommerfeld_bc(
        BX, 0., 1.
    ) ;
    apply_sommerfeld_bc(
        BY, 0., 1.
    ) ;   
    apply_sommerfeld_bc(
        BZ, 0., 1.
    ) ;

    Kokkos::fence() ;
}

}} /* namespace grace::amr */