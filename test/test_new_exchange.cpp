/**
 * @file test_new_exchange.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2025-09-09
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
#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/amr/amr_ghosts.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/vtk_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/data_structures/variable_utils.hh>

#include <grace/utils/task_queue.hh>

#include <string>

TEST_CASE("Unigrid exchange", "[unigrid]")
{
    using namespace grace ; 
    auto& ghost = grace::amr_ghosts::get() ; 
    ghost.update() ; 
    
    auto const & layer = ghost.get_ghost_layer() ; 
    auto rank = parallel::mpi_comm_rank() ; 
    auto nproc = parallel::mpi_comm_size() ;
    auto nq = grace::amr::get_local_num_quadrants() ; 
    std::cout << "Neighbor list updated" << std::endl ;
    size_t qid = 0 ;
    if ( rank == 0 ) {
    for( auto const& q: layer ) {
        std::cout << "quad-id " << qid << std::endl ;
        std::cout << "face neighbors: " << std::endl ;
        for( int i=0; i<P4EST_FACES; ++i) {
            std::cout << std::endl ; 
            auto face = q.faces[i] ; 
            std::string face_kind =  face.kind == grace::interface_kind_t::PHYS ? "phys bound" : "internal" ; 
            std::cout << "     " << i << " kind " << face_kind << '\n' ; 
            if( ! (face.kind == grace::interface_kind_t::PHYS) ){ 
                    std::cout << "     level diff " << (int) face.level_diff << '\n'
                        << "     neighbor quadid " << face.data.full.quad_id << '\n' 
                        << "     neighbor is remote " << face.data.full.is_remote << std::endl ;
                if (face.data.full.is_remote  ){
                    std::cout << "     owner rank id " << face.data.full.owner_rank << std::endl ;  
                }  
            }
        }
        qid ++ ; 
    }
    std::vector<std::size_t> send_rank_offsets, recv_rank_offsets ; //!< In # of elements
    std::vector<std::size_t> send_rank_sizes, recv_rank_sizes ; //!< In # of elements
    ghost.get_rank_offsets(send_rank_offsets, recv_rank_offsets) ; 
    ghost.get_rank_sizes(send_rank_sizes,recv_rank_sizes) ; 

    auto nq = amr::get_local_num_quadrants() ; 
    std::size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts() ; 
    std::size_t nvars = variables::get_n_evolved() ; 

    std::size_t face_size = nx*nx * ngz * nvars ; 

    std::cout << "Send/Recv buffer sizes and offsets per rank:\n";
    for (int r = 0; r < nproc; ++r) {
        std::cout << "  Rank " << r << ":\n"
                << "    send size   = " << send_rank_sizes[r] / face_size 
                << ", offset = " << send_rank_offsets[r] << "\n"
                << "    recv size   = " << recv_rank_sizes[r] / face_size 
                << ", offset = " << recv_rank_offsets[r] << "\n";
    }

    auto& mpi_tasks = ghost.get_mpi_tasks() ; 
    std::unordered_map<status_id_t, std::string> statuses ; 
    statuses[status_id_t::WAITING] = "waiting" ; 
    statuses[status_id_t::READY] = "ready" ; 
    statuses[status_id_t::RUNNING] = "running" ; 
    statuses[status_id_t::COMPLETE] = "complete" ; 
    statuses[status_id_t::FAILED] = "failed" ; 
    for( int i=0; i<mpi_tasks.size(); ++i ) {
        std::cout << "Task[" << mpi_tasks[i].task_id << "] status " << statuses[mpi_tasks[i].status] << std::endl ;  
    }

}
}