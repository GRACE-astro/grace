/**
 * @file copy_kernels.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief This file contains the copy kernel for regrid.
 * @version 0.1
 * @date 2025-10-29
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

#ifndef GRACE_AMR_REGRID_MPI_KERNEL_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/limiters.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace {

static mpi_task_t make_mpi_send_task_regrid(
      std::size_t rr 
    , amr::face_buffer_t send_buf 
    , std::vector<int> const& send_rank_offsets 
    , std::vector<int> const& send_rank_sizes
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
            , parallel::GRACE_PARTITION_TAG
            , MPI_COMM_WORLD
            , req
        ) ;
    } ; 
    task.task_id = task_counter++; 
    return task ; 
}

static mpi_task_t make_mpi_recv_task_regrid(
      std::size_t rr 
    , amr::face_buffer_t recv_buf 
    , std::vector<int> const& recv_rank_offsets 
    , std::vector<int> const& recv_rank_sizes
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
            , parallel::GRACE_PARTITION_TAG
            , MPI_COMM_WORLD
            , req
        ) ;
    } ; 
    task.task_id = task_counter++; 
    return task ; 
}

} /* namespace grace */

#endif 