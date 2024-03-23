/**
 * @file boundary_conditions.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-21
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

#include <Kokkos_Core.hpp>

#include <thunder/amr/thunder_amr.hh>
#include <thunder/amr/bc_helpers.hh> 
#include <thunder/amr/bc_helpers.tpp> 
#include <thunder/amr/bc_kernels.tpp>
#include <thunder/system/thunder_system.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/data_structures/macros.hh>
#include <thunder/data_structures/memory_defaults.hh>
#include <thunder/data_structures/variable_indices.hh>
#include <thunder/data_structures/variable_properties.hh>
#include <thunder/data_structures/variables.hh>
#include <thunder/data_structures/variable_utils.hh>
#include <thunder/config/config_parser.hh>

namespace thunder { namespace amr {

void apply_boundary_conditions() {
    using namespace thunder ;
    /******************************************************/
    /* First step:                                        */
    /* Asynchronous data exchange for quadrants in the    */
    /* halo.                                              */
    /******************************************************/
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = get_quadrant_extents() ;
    int64_t ngz = get_n_ghosts() ; 
    auto& vars = variable_list::get().getstate()    ;
    auto& halo = variable_list::get().gethalo()    ;
    size_t nvars = variables::get_n_evolved() ; 
    /* Create ghost layer */
    p4est_ghost_t * halos = p4est_ghost_new( 
          forest::get().get() 
        , P4EST_CONNECT_FACE  
    ) ; 
    sc_array_view_t<p4est_quadrant_t> 
          halo_quads{ &(halos->ghosts) }
        , mirror_quads{ &(halos->mirrors) }  ;

    Kokkos::realloc(halo, VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nvars,halo_quads.size());  
    size_t send_size = EXPR(nx, *ny, *nz) * nvars ; 
    parallel::thunder_transfer_context_t context ;
    size_t rank = parallel::mpi_comm_rank() ; 
    for(int iproc=0; iproc<parallel::mpi_comm_size(); ++iproc){
        size_t first_halo  = halos->proc_offsets[iproc]   ; 
        size_t last_halo   = halos->proc_offsets[iproc+1] ;
        for( int ihalo=first_halo; ihalo<last_halo; ++ihalo ) {
            size_t iq_loc = get_quadrant_locidx(&(mirror_quads[ihalo])) ;
            context._rcv_rq.push_back(sc_MPI_Request{}) ; 
            auto hsview = Kokkos::subview(
                  halo
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                  hsview.data()
                , send_size
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._rcv_rq.back())
            ) ; 

            context._snd_rq.push_back(sc_MPI_Request{}) ; 
            auto sview = Kokkos::subview(
                  vars
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                ,Kokkos::ALL()
                , iq_loc ) ;
             
            parallel::mpi_isend(
                  sview.data()
                , send_size
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._snd_rq.back())
            ) ; 
        }
    } 
    /******************************************************/
    /* Second step:                                       */
    /* Iterate over all quadrant faces and store face     */
    /* information.                                       */
    /******************************************************/
    thunder_face_info_t face_info{} ;
    p4est_iterate(
          forest::get().get()
        , halos 
        , reinterpret_cast<void*>( &face_info )
        , nullptr
        , thunder_iterate_faces 
        , nullptr) ;
    /******************************************************/
    /* Third step:                                        */
    /* Apply physical boundary conditions.                */
    /******************************************************/
    auto phys_boundary_info = face_info.phys_boundary_info ; 
    phys_boundary_info.host_to_device() ; 
    for(int ivar=0; ivar<nvars; ++ivar){
        auto bc_type = variables::get_bc_type(ivar) ; 
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
                , phys_boundary_info
            ) ; 
        } else {
            ERROR("Unrecognized bc type for variable " << ivar << ".\n") ;
        }
    }
    /******************************************************/
    /* Fourth step:                                       */
    /* Copy and interpolate face data from internal       */
    /* boundaries.                                        */
    /******************************************************/
    parallel::mpi_waitall(context) ;
    auto simple_interior_info = face_info.simple_interior_info ; 
    simple_interior_info.host_to_device() ;
    copy_interior_ghostzones(simple_interior_info) ; 
    auto hanging_interior_info = face_info.hanging_interior_info ; 
    hanging_interior_info.host_to_device() ;
    auto interp_type = 
        config_parser::get()["amr"]["prolongation_interpolator_type"].as<std::string>() ; 
    if( interp_type == "linear" ){
        interp_hanging_ghostzones<utils::linear_interp_t<THUNDER_NSPACEDIM>>(hanging_interior_info) ; 
    } else {
        ERROR("Unsupported interpolator in ghost-zone exchange.") ; 
    }
    Kokkos::fence() ;
    /******************************************************/
    /* Transform vector and tensor components             */
    /* across tree boundaries (where applicable)          */
    /******************************************************/
    
    /******************************************************/
    /* De-allocate halo quadrant data                     */
    /******************************************************/
    Kokkos::realloc(halo, VEC(0,0,0), 0,0);
}

}} /* namespace thunder::amr */