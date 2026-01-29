/**
 * @file task_queue.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Index fiesta.
 * @date 2025-09-08
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
#include <grace/utils/device_event.hh>
#include <grace/utils/device_stream.hh>
#include <grace/utils/device_stream_pool.hh>

#include <grace/parallel/mpi_wrappers.hh>

#include <grace/utils/task_queue.hh>

#include <grace/system/print.hh>

#include <functional> 
#include <vector>
#include <deque>

namespace grace {

task_t::~task_t() = default ; 

void executor::run(view_alias_t alias) {
  while (!ready.empty() || !gpu_pending.empty() || !mpi_pending.empty()) {

    // 1) Dispatch ready tasks
    while (!ready.empty()) {
      auto id = ready.front();
      ready.pop_front();
      auto& R = rt[id];
      auto& T = *R.t;

      switch (T.kind) {
        case task_kind_t::GPU_KERNEL: {
          GRACE_TRACE("Launching GPU task {}", id);
          static_cast<gpu_task_t&>(T).run(alias);
          gpu_pending.push_back(id);
          break;
        }
        case task_kind_t::MPI_TRANSFER: {
          GRACE_TRACE("Launching MPI task {}", id);
          static_cast<mpi_task_t&>(T).run(alias);
          mpi_pending.push_back(id);
          break;
        }
        case task_kind_t::CPU_EXEC: {
          GRACE_TRACE("Running CPU task {}", id);
          static_cast<cpu_task_t&>(T).run(alias);
          complete_and_release(id);
          break;
        }
      }
    }

    // 2) Poll GPU
    for (auto it = gpu_pending.begin(); it != gpu_pending.end(); ) {
      auto id = *it; auto& R = rt[id]; auto& T = *R.t;
      if (T.query() == status_id_t::COMPLETE) {
        GRACE_TRACE("GPU task {} complete", id);
        complete_and_release(id);
        it = gpu_pending.erase(it);
      } else {
        ++it;
      }
    }

    // 3) Poll MPI
    for (auto it = mpi_pending.begin(); it != mpi_pending.end(); ) {
      auto id = *it; auto& R = rt[id]; auto& T = *R.t;
      if (T.query() == status_id_t::COMPLETE) {
        GRACE_TRACE("MPI task {} complete", id);
        complete_and_release(id);
        it = mpi_pending.erase(it);
      } else {
        ++it;
      }
    }
  }
}


void executor::reset() {
  ready.clear();
  gpu_pending.clear();
  mpi_pending.clear();

  for (std::size_t id = 0; id < rt.size(); ++id) {
    auto& R = rt[id];
    auto& T = *R.t;

    R.pending = static_cast<int>(T._dependencies.size());

    T.status = (R.pending == 0 ? status_id_t::READY : status_id_t::WAITING);

    if ( T.kind == task_kind_t::MPI_TRANSFER) {
      reinterpret_cast<mpi_task_t*>(R.t)->mpi_req = sc_MPI_REQUEST_NULL ; 
    } else if ( T.kind == task_kind_t::GPU_KERNEL ) {
      reinterpret_cast<gpu_task_t*>(R.t)->dev_event.reset() ; 
    }
    
    if (R.pending == 0)
      ready.push_back(id);
  }
}


void executor::complete_and_release(task_id_t const& id) {
  auto& R = rt[id]; auto& T = *R.t;
  T.status = status_id_t::COMPLETE;
  for (auto nxt : T._dependents) {
    if (--rt[nxt].pending == 0) {
      rt[nxt].t->status = status_id_t::READY;
      ready.push_back(nxt);
    }
  }
}

} /* namespace grace */