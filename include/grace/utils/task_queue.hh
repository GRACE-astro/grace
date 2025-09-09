/**
 * @file amr_ghosts.cpp
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

#ifndef GRACE_UTILS_TASK_QUEUE_HH 
#define GRACE_UTILS_TASK_QUEUE_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/device_event.hh>
#include <grace/utils/device_stream.hh>
#include <grace/utils/device_stream_pool.hh>

#include <grace/parallel/mpi_wrappers.hh>

#include <functional> 
#include <vector>
#include <deque>

namespace grace {

using task_id_t = std::size_t ; 

enum task_kind_t: uint8_t { GPU_KERNEL, MPI_TRANSFER, CPU_EXEC  } ; 

enum status_id_t: uint8_t {  WAITING, READY, RUNNING, COMPLETE, FAILED }; 

struct completion_handle_t {
    task_kind_t kind ; 
    device_event_t dev_event{} ; 
    MPI_Request mpi_req{MPI_REQUEST_NULL} ;
} ; 

struct task_t {
    task_kind_t kind ; 
    status_id_t status ; 
    completion_handle_t handle ; 

    /**
     * @brief query the task for its status 
     */
    status_id_t query() const {
        switch (kind) {
            case task_kind_t::GPU_KERNEL: {
                auto s = handle.dev_event.query() ;
                if (s == DEVICE_SUCCESS) return status_id_t::COMPLETE;
                if (s == DEVICE_NOT_READY) return status_id_t::RUNNING;
                return status_id_t::FAILED; 
            }
            case task_kind_t::MPI_TRANSFER: {
                int flag = 0 ; 
                auto err = MPI_Test(const_cast<MPI_Request*>(&handle.mpi_req), &flag, MPI_STATUS_IGNORE) ;
                ASSERT( err==MPI_SUCCESS, "Error in MPI_Test, possibly null request passed.") ; 
                return flag ? status_id_t::COMPLETE : status_id_t::RUNNING ; 
            }
            case task_kind_t::CPU_EXEC: {
                return status ; 
            }
        }
        return status_id_t::COMPLETE ; 
    }; 

    /** @brief Launch execution of the task 
     */
    void run() ; 

    //! Dependencies 
    std::vector<task_id_t> _dependencies ; 
    //! Tasks depending on this one 
    std::vector<task_id_t> _dependents   ; 
    //! Task identification number (unique)
    task_id_t task_id ; 
} ; 

struct gpu_task_t : public task_t {

    gpu_task_t()
    {
      kind = task_kind_t::GPU_KERNEL ; 
      handle.kind = task_kind_t::GPU_KERNEL ; 
    }

    void run(device_stream_t const & stream) {
        handle.dev_event.reset() ; 
        status = status_id_t::RUNNING ; 
        _run(stream) ; 
        handle.dev_event.record(stream) ; 
    }

    //! The task itself 
    std::function<void(device_stream_t const&)> _run ; 
} ; 

struct mpi_task_t : public task_t {

    mpi_task_t()
    {
      kind = task_kind_t::MPI_TRANSFER ; 
      handle.kind = task_kind_t::MPI_TRANSFER ; 
    }

    void run() {
        ASSERT( status = status_id_t::READY, "Attempting to run task that is not ready") ;
        status = status_id_t::RUNNING ; 
        _run(handle) ; 
    }

    //! The task itself 
    std::function<void(completion_handle_t &)> _run ; 

} ; 

struct cpu_task_t : public task_t {

    cpu_task_t()
    {
      kind = task_kind_t::CPU_EXEC ; 
      handle.kind = task_kind_t::CPU_EXEC ; 
    }

    void run() {
        ASSERT( status = status_id_t::READY, "Attempting to run task that is not ready") ;
        status = status_id_t::RUNNING ; 
        _run() ; 
        status = status_id_t::COMPLETE ; 
    }

    //! The task itself 
    std::function<void()> _run ; 

} ; 


struct runtime_task_view {
  task_t* t;              // pointer to the actual task definition
  std::atomic<int> pending;  // number of unsatisfied dependencies
  uint16_t stream_id;     // preassigned stream (optional)
};


struct executor {

  /** 
  * @brief Run task queue represented in runtime_task_view 
  */
  void run() ; 

  /** 
  * @brief Complete a task and notify dependents 
  * @param id The task id to be released 
  */
  void complete_and_release(task_id_t const& id) ;

  /**
   * @brief Reset the task queue to its state prior to execution
   */
  void reset() ; 

  std::deque<task_id_t> ready ;  //!< FIFO queue of tasks ready for execution 
  std::vector<task_id_t> gpu_pending, mpi_pending ; //!< List of pending tasks 
  std::vector<runtime_task_view> rt     ; //!< Runtime view 
} ; 

void executor::run() {
  while (!ready.empty() || !gpu_pending.empty() || !mpi_pending.empty()) {

    // 1) Dispatch ready tasks
    while (!ready.empty()) {
      auto id = ready.front();
      ready.pop_front() ; 
      auto& R = rt[id];
      auto& T = *R.t;
      switch (T.kind) {
        case task_kind_t::GPU_KERNEL: {
          auto& stream = device_stream_pool::get().at(R.stream_id);
          static_cast<gpu_task_t&>(T).run(stream);
          gpu_pending.push_back(id);
          break;
        }
        case task_kind_t::MPI_TRANSFER: {
          static_cast<mpi_task_t&>(T).run();
          mpi_pending.push_back(id);
          break;
        }
        case task_kind_t::CPU_EXEC: {
          static_cast<cpu_task_t&>(T).run();
          complete_and_release(id);
          break;
        }
      }
    }

    // 2) Poll GPU
    for (auto it = gpu_pending.begin(); it != gpu_pending.end(); ) {
      auto id = *it; auto& R = rt[id]; auto& T = *R.t;
      if (T.query() == status_id_t::COMPLETE) {
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


} 
#endif /* GRACE_UTILS_TASK_QUEUE_HH */