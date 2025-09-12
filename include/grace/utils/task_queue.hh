/**
 * @file task_queue.hh
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


struct task_t {
    task_kind_t kind ; 
    status_id_t status ; 

    task_t() = default ; 
    virtual ~task_t()  ; 

    /** @brief Launch execution of the task 
     */
    virtual void run()=0; 

    /**
     * @brief query the task for its status 
     */
    virtual status_id_t query() const =0;


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
      status = status_id_t::WAITING ; 
      stream = nullptr ; 
    }

    void run() override {
        // note need to remember to reset event and attach it to stream where kernel is running 
        // INSIDE _run 
        ASSERT( status = status_id_t::READY, "Attempting to run task that is not ready") ;
        ASSERT( stream != nullptr, "Attempting to run on a null stream") ; 
        status = status_id_t::RUNNING ; 
        dev_event.reset() ; 
        _run() ; 
        dev_event.record(*stream) ; 
    }

    /**
     * @brief query the task for its status 
     */
    status_id_t query() const override {
        auto s = dev_event.query() ;
        if (s == DEVICE_SUCCESS) return status_id_t::COMPLETE;
        if (s == DEVICE_NOT_READY) return status_id_t::RUNNING;
        return status_id_t::FAILED;
    }

    //! Device event 
    grace::device_event_t dev_event ; 

    //! Device stream
    grace::device_stream_t* stream ; 

    //! The task itself 
    std::function<void()> _run ; 
} ; 

struct mpi_task_t : public task_t {

    mpi_task_t()
    {
      kind = task_kind_t::MPI_TRANSFER ; 
      status = status_id_t::WAITING ; 
    }

    void run() override {
        ASSERT( status = status_id_t::READY, "Attempting to run task that is not ready") ;
        status = status_id_t::RUNNING ; 
        _run(&mpi_req) ; 
    }

    /**
     * @brief query the task for its status 
     */
    status_id_t query() const override {
        int flag = 0 ; 
        auto err = MPI_Test(const_cast<MPI_Request*>(&mpi_req), &flag, MPI_STATUS_IGNORE) ;
        ASSERT( err==MPI_SUCCESS, "Error in MPI_Test, possibly null request passed.") ; 
        return flag ? status_id_t::COMPLETE : status_id_t::RUNNING ;
    }

    //! MPI request
    MPI_Request mpi_req ;

    //! The task itself 
    std::function<void(MPI_Request*)> _run ; 

} ; 

struct cpu_task_t : public task_t {

    cpu_task_t()
    {
      kind = task_kind_t::CPU_EXEC ; 
      status = status_id_t::WAITING ; 
    }

    void run() override {
        ASSERT( status = status_id_t::READY, "Attempting to run task that is not ready") ;
        status = status_id_t::RUNNING ; 
        _run() ; 
        status = status_id_t::COMPLETE ; 
    }

    /**
     * @brief query the task for its status 
     */
    status_id_t query() const override {
        return status ;
    }

    //! The task itself 
    std::function<void()> _run ; 

} ; 


struct runtime_task_view {
  task_t* t;              // pointer to the actual task definition
  std::atomic<int> pending;  // number of unsatisfied dependencies
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




} 
#endif /* GRACE_UTILS_TASK_QUEUE_HH */