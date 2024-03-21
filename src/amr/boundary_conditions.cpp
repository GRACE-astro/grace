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
#include <thunder/system/thunder_system.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/data_structures/macros.hh>
#include <thunder/data_structures/memory_defaults.hh>
#include <thunder/data_structures/variable_indices.hh>
#include <thunder/data_structures/variable_properties.hh>
#include <thunder/data_structures/variables.hh>
#include <thunder/config/config_parser.hh>

namespace thunder { namespace amr {

void apply_boundary_conditions() {
    using namespace thunder ;
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = get_quadrant_extents() ;
    auto& vars = variables::get().getstate()    ;
    size_t nvars = vars.extent(THUNDER_NSPACEDIM+1) ; 
    /* Create ghost layer */
    p4est_ghost_t * ghosts = p4est_ghost_new( 
          forest::get().get() 
        , P4EST_CONNECT_FACE  
    ) ; 
    sc_array_view_t<p4est_quadrant_t> 
          halo_quads{ &(ghosts->ghosts) }
        , mirror_quads{ &(ghosts->mirrors) }  ;
    size_t nhalo_quads = halo_quads.size() ;
    ASSERT( nhalo_quads == mirror_quads.size(),
          "Sanity check failed in ghost exchange." ) ;

    var_array_t<THUNDER_NSPACEDIM> 
        halo_data("ghost_data", VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nvars,nghost_quads);    
    size_t send_size = VEC(nx, *ny, *nz) * nvars ; 
    parallel::thunder_transfer_context_t context ; 
    for( int ihalo=0; ihalo<halo_quads.size(); ++ihalo ) {
        size_t iq_loc = get_quadrant_locidx(mirror_quads[ihalo]) ;
        size_t proc = get_halo_quad_owner(halo_quads[ihalo])      ; 
        
        context._rcv_rq.push_back(sc_MPI_Request{}) ; 
        parallel::mpi_irecv(
              (halo_data.get() + ihalo*send_size)
            , send_size
            , proc
            , parallel::THUNDER_HALO_EXCHANGE_TAG*nhalo_quads*2 + 2*ihalo 
            , parallel::get_comm_world()
            , &(context._rcv_rq.back())
        ) ; 

        context._snd_rq.push_back(sc_MPI_Request{}) ; 
        parallel::mpi_isend( 
              (vars.get() + iq_loc*send_size)
            , send_size 
            , proc
            , parallel::THUNDER_HALO_EXCHANGE_TAG*nhalo_quads*2 + 2*ihalo + 1
            , parallel::get_comm_world()
            , &(context._snd_rq.back())
        ) ; 
    }

    parallel::mpi_waitall(context) ; 




}

}} /* namespace thunder::amr */