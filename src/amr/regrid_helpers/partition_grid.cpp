/**
 * @file partition_grid.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-24
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

#include <grace/amr/regrid_helpers.hh>
#include <grace/amr/amr_functions.hh> 
#include <grace/amr/forest.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/system/grace_system.hh>

#include <Kokkos_Core.hpp>

#include <vector>

namespace grace { namespace amr {


std::vector<p4est_transfer_context_t *> 
grace_partition_begin(
    grace::var_array_t<GRACE_NSPACEDIM>& state, 
    grace::var_array_t<GRACE_NSPACEDIM>& state_swap,
    grace::staggered_variable_arrays_t& sstate, 
    grace::staggered_variable_arrays_t& sstate_swap
) {
    using namespace grace  ; 
    using namespace Kokkos ;
    GRACE_VERBOSE("Initiating transfer of data for parallel partition.") ; 
    /***************************************************/
    /*                Get grid properties              */
    /***************************************************/
    size_t nx,ny,nz                                        ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents()       ; 
    auto ngz = amr::get_n_ghosts()                         ; 
    size_t nq = amr::get_local_num_quadrants()             ;
    int nvars_cell_centered      = state.extent(GRACE_NSPACEDIM)               ; 
    int nvars_face_staggered     = variables::get_n_evolved_face_staggered()   ; 
    int nvars_edge_staggered     = variables::get_n_evolved_edge_staggered()   ; 
    int nvars_corner_staggered   = variables::get_n_evolved_corner_staggered() ; 
    /******************************************************************************************/
    /*                      Partition the new forest in parallel                              */
    /*                      we store global quadrant offsets, then                            */
    /*                      partition the forest, transfer state data                         */
    /*                      asynchronously, and realloc other fields                          */
    /*                      in the meanwhile. Coordinates are recomputed                      */
    /*                      but the auxiliary fields are left empty.                          */
    /******************************************************************************************/
    auto const glob_qoffsets = amr::get_global_quadrant_offsets() ;
    /******************************************************************************************/
    /*                                    Partition forest                                    */
    /******************************************************************************************/
    size_t transfer_count = p4est_partition_ext( forest::get().get()
                                               , 0
                                               , nullptr  ) ; 
    auto const new_glob_qoffsets = amr::get_global_quadrant_offsets() ;
    /******************************************************************************************/
    /*                          Compute send/receive sizes                                    */
    /******************************************************************************************/
    size_t const quadrant_data_size = EXPR(   (nx+2*ngz)
                                          , * (ny+2*ngz)
                                          , * (nz+2*ngz)  ) * nvars_cell_centered * sizeof(double); 
    size_t const quadrant_data_size_face_x_staggered = 
        EXPR(   (nx+1+2*ngz)
            , * (ny+2*ngz)
            , * (nz+2*ngz)  ) * nvars_face_staggered * sizeof(double);
    size_t const quadrant_data_size_face_y_staggered = 
        EXPR(   (nx+2*ngz)
            , * (ny+1+2*ngz)
            , * (nz+2*ngz)  ) * nvars_face_staggered * sizeof(double);
    size_t const quadrant_data_size_face_z_staggered = 
        EXPR(   (nx+2*ngz)
            , * (ny+2*ngz)
            , * (nz+1+2*ngz)  ) * nvars_face_staggered * sizeof(double);
    #ifdef GRACE_3D
    size_t const quadrant_data_size_edge_xy_staggered = 
        EXPR(   (nx+1+2*ngz)
            , * (ny+1+2*ngz)
            , * (nz+2*ngz)  ) * nvars_edge_staggered * sizeof(double);
    size_t const quadrant_data_size_edge_xz_staggered = 
        EXPR(   (nx+1+2*ngz)
            , * (ny+2*ngz)
            , * (nz+1+2*ngz)  ) * nvars_edge_staggered * sizeof(double);
    size_t const quadrant_data_size_edge_yz_staggered = 
        EXPR(   (nx+2*ngz)
            , * (ny+1+2*ngz)
            , * (nz+1+2*ngz)  ) * nvars_edge_staggered * sizeof(double);
    #endif 
    size_t const quadrant_data_size_corner_staggered = 
        EXPR(   (nx+1+2*ngz)
            , * (ny+1+2*ngz)
            , * (nz+1+2*ngz)  ) * nvars_corner_staggered * sizeof(double);
    size_t const nq_local = amr::get_local_num_quadrants() ; 
    /******************************************************************************************/
    /*                              Realloc data and partition forest                         */
    /******************************************************************************************/  
    Kokkos::realloc( state      ,   VEC(  nx + 2*ngz 
                                        , ny + 2*ngz 
                                        , nz + 2*ngz )
                                ,   nvars_cell_centered
                                ,   nq_local 
                                 ) ;
    sstate.realloc(VEC(nx,ny,nz), ngz, nq_local, nvars_face_staggered, nvars_edge_staggered, nvars_corner_staggered);
    /******************************************************************************************/
    /*                                Transfer data                                           */
    /******************************************************************************************/
    std::vector<p4est_transfer_context_t*> ctx ;
    int tag = parallel::GRACE_PARTITION_TAG  ; 
    auto context = 
        p4est_transfer_fixed_begin (
                  new_glob_qoffsets.data() 
                , glob_qoffsets.data()
                , parallel::get_comm_world() 
                , tag
                , reinterpret_cast<void*>(state.data())
                , reinterpret_cast<void*>(state_swap.data())
                , quadrant_data_size 
        ) ;
    ctx.push_back(context); 
    tag++ ; 
    if( nvars_face_staggered > 0 ) {
    context = 
        p4est_transfer_fixed_begin (
                  new_glob_qoffsets.data() 
                , glob_qoffsets.data()
                , parallel::get_comm_world() 
                , tag
                , reinterpret_cast<void*>(sstate.face_staggered_fields_x.data())
                , reinterpret_cast<void*>(sstate_swap.face_staggered_fields_x.data())
                , quadrant_data_size_face_x_staggered 
        ) ;
    ctx.push_back(context); 
    tag++ ;
    context = 
        p4est_transfer_fixed_begin (
                  new_glob_qoffsets.data() 
                , glob_qoffsets.data()
                , parallel::get_comm_world() 
                , tag
                , reinterpret_cast<void*>(sstate.face_staggered_fields_y.data())
                , reinterpret_cast<void*>(sstate_swap.face_staggered_fields_y.data())
                , quadrant_data_size_face_y_staggered 
        ) ;
    ctx.push_back(context); 
    tag++ ;
    context = 
        p4est_transfer_fixed_begin (
                  new_glob_qoffsets.data() 
                , glob_qoffsets.data()
                , parallel::get_comm_world() 
                , tag
                , reinterpret_cast<void*>(sstate.face_staggered_fields_z.data())
                , reinterpret_cast<void*>(sstate_swap.face_staggered_fields_z.data())
                , quadrant_data_size_face_z_staggered 
        ) ;
    ctx.push_back(context); 
    tag++ ;
    }
    #ifdef GRACE_3D
    if (nvars_edge_staggered > 0) {
    context = 
        p4est_transfer_fixed_begin (
                  new_glob_qoffsets.data() 
                , glob_qoffsets.data()
                , parallel::get_comm_world() 
                , tag
                , reinterpret_cast<void*>(sstate.edge_staggered_fields_xy.data())
                , reinterpret_cast<void*>(sstate_swap.edge_staggered_fields_xy.data())
                , quadrant_data_size_edge_xy_staggered 
        ) ;
    ctx.push_back(context); 
    tag++ ;
    context = 
        p4est_transfer_fixed_begin (
                  new_glob_qoffsets.data() 
                , glob_qoffsets.data()
                , parallel::get_comm_world() 
                , tag
                , reinterpret_cast<void*>(sstate.edge_staggered_fields_xz.data())
                , reinterpret_cast<void*>(sstate_swap.edge_staggered_fields_xz.data())
                , quadrant_data_size_edge_xz_staggered 
        ) ;
    ctx.push_back(context); 
    tag++ ;
    context = 
        p4est_transfer_fixed_begin (
                  new_glob_qoffsets.data() 
                , glob_qoffsets.data()
                , parallel::get_comm_world() 
                , tag
                , reinterpret_cast<void*>(sstate.edge_staggered_fields_yz.data())
                , reinterpret_cast<void*>(sstate_swap.edge_staggered_fields_yz.data())
                , quadrant_data_size_edge_yz_staggered 
        ) ;
    ctx.push_back(context); 
    tag++ ;
    }
    #endif 
    if( nvars_corner_staggered > 0 ) {
    auto context1 = 
        p4est_transfer_fixed_begin (
                  new_glob_qoffsets.data() 
                , glob_qoffsets.data()
                , parallel::get_comm_world() 
                , tag
                , reinterpret_cast<void*>(sstate.corner_staggered_fields.data())
                , reinterpret_cast<void*>(sstate_swap.corner_staggered_fields.data())
                , quadrant_data_size_corner_staggered 
        ) ;
    ctx.push_back(context1); 
    tag++ ;
    }
    return ctx ; 
}

void grace_partition_finalize(std::vector<p4est_transfer_context_t *> const & context) {
    for( auto const ctx: context) 
        p4est_transfer_fixed_end(ctx) ; 
}


}} /* namespace grace::amr */