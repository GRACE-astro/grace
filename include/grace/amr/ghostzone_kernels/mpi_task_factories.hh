/**
 * @file task_factories.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Index fiesta.
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

 #include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_utils.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/assert.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <grace/amr/amr_ghosts.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/ghostzone_kernels/copy_kernels.hh>
#include <grace/amr/ghostzone_kernels/phys_bc_kernels.hh>
#include <grace/amr/ghostzone_kernels/restrict_kernels.hh>
#include <grace/amr/ghostzone_kernels/pack_unpack_kernels.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <unordered_set>
#include <vector>
#include <numeric>

#ifndef GRACE_AMR_GHOSTZONE_MPI_TASK_FACTORY_HH
#define GRACE_AMR_GHOSTZONE_MPI_TASK_FACTORY_HH

namespace grace {

/**
 * @brief Create a task that sends data to one rank via MPI.
 * @ingroup amr
 * @param rr Rank where data is sent.
 * @param send_buf Send buffer.
 * @param send_rank_offsets Data offsets per rank.
 * @param send_rank_sizes Data sizes per rank.
 * @param task_counter Current task counter.
 * @return mpi_task_t A task encapsulating the asynchronous dispatch of data to rank r.
 */
mpi_task_t make_mpi_send_task(
      std::size_t rr 
    , amr::ghost_array_t send_buf 
    , std::vector<std::size_t> const& send_rank_offsets 
    , std::vector<std::size_t> const& send_rank_sizes
    , task_id_t& task_counter 
)
{
    GRACE_TRACE("Registering MPI Send task (tid {}).\n    Send to Rank {} {} elements, offset {}", task_counter, rr, send_rank_sizes[rr], send_rank_offsets[rr]) ; 

    ASSERT(send_rank_offsets[rr] + send_rank_sizes[rr] <= send_buf.size(), "Send out-of-bounds" ) ; 

    mpi_task_t task ; 
    task._run = [&, send_buf, rr] (MPI_Request* req) {
        parallel::mpi_isend(
              send_buf.data() + send_rank_offsets[rr]
            , send_rank_sizes[rr]
            , rr 
            , 0
            , MPI_COMM_WORLD
            , req
        ) ;
    } ; 
    task.task_id = task_counter++; 
    return task ; 
}
/**
 * @brief Create a task that receives data from one rank via MPI.
 * @ingroup amr 
 * @param rr Rank where data is sent.
 * @param recv_buf Receive buffer.
 * @param recv_rank_offsets Data offsets per rank.
 * @param recv_rank_sizes Data sizes per rank.
 * @param task_counter Current task counter.
 * @return mpi_task_t A task encapsulating the asynchronous receive of data to rank r. 
 */
mpi_task_t make_mpi_recv_task(
      std::size_t rr 
    , amr::ghost_array_t recv_buf 
    , std::vector<std::size_t> const& recv_rank_offsets 
    , std::vector<std::size_t> const& recv_rank_sizes
    , task_id_t& task_counter 
)
{
    GRACE_TRACE("Registering MPI Receive task (tid {}).\n    Receive from Rank {} {} elements, offset {}", task_counter, rr, recv_rank_sizes[rr], recv_rank_offsets[rr]) ; 

    ASSERT(recv_rank_offsets[rr] + recv_rank_sizes[rr] <= recv_buf.size(), "Receive out-of-bounds" ) ; 

    mpi_task_t task ; 
    task._run = [&, recv_buf, rr] (MPI_Request* req) {
        parallel::mpi_irecv(
              recv_buf.data() + recv_rank_offsets[rr]
            , recv_rank_sizes[rr]
            , rr 
            , 0
            , MPI_COMM_WORLD
            , req
        ) ;
    } ; 
    task.task_id = task_counter++; 
    return task ; 
}


}

#endif 