/**
 * @file checkpoint_handler.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-27
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

#ifndef GRACE_SYSTEM_CHECKPOINT_HANDLER_HH
#define GRACE_SYSTEM_CHECKPOINT_HANDLER_HH

#include <grace_config.h>

#include <grace/utils/singleton_holder.hh> 
#include <grace/utils/creation_policies.hh>
#include <grace/utils/lifetime_tracker.hh> 

#include <grace/amr/forest.hh>


#include <filesystem>
#include <deque>

namespace grace {

//*****************************************************************************************************
//*****************************************************************************************************
class checkpoint_handler_impl_t 
{
 public:
    //*****************************************************************************************************
    /**
     * @brief Save the current state to a checkpoint file.
     * 
     */
    void save_checkpoint()  ; 
    //*****************************************************************************************************

    //*****************************************************************************************************
    /**
     * @brief Load the checkpoint corresponding to the given iteration.
     * 
     * @param iter Iteration corresponding to the checkpoint being loaded (if left default latest available is loaded). 
     */
    void load_checkpoint(int64_t iter = -1)  ; 
    //*****************************************************************************************************

    //*****************************************************************************************************
    /**
     * @brief Detect the presence of checkpoints.
     * 
     * This function searches for checkpoints in the 
     * directory specified by the parameter <code>checkpoint_dir</code>.
     */
    void detect_checkpoints()  ; 
    //*****************************************************************************************************

    //*****************************************************************************************************
    /**
     * @brief Detect whether a checkpoint should be saved now
     * 
     * This function checks if either walltime, simulation time, or 
     * iteration interval has elapsed since last checkpoint
     * 
     * @return true If a checkpoint is needed
     * @return false If a checkpoint is not needed
     */
    bool need_checkpoint()  ; 
    //*****************************************************************************************************

    //*****************************************************************************************************
    /**
     * @brief Check if a checkpoint exists.
     * 
     * @return true If a checkpoint exists
     * @return false If no checkpoint exists
     */
    bool have_checkpoint() const { return have_checkpoint_ ; }
    //*****************************************************************************************************
 private:

    //*****************************************************************************************************
    /**
     * @brief Delete the oldest checkpoint.
     * 
     */
    void delete_checkpoint()  ;
    //*****************************************************************************************************

    //*****************************************************************************************************
    std::deque<int64_t> checkpoint_list  ; //!< List of available checkpoints (iterations)
    unsigned int max_n_checkpoints       ; //!< Max number of checkpoints that are allowed to exist at once
    std::filesystem::path checkpoint_dir ; //!< Directory containing checkpoints 
    std::string checkpoint_interval_type ; //!< Which kind of criterion is used for checkpointing?
    double checkpoint_wtime_interval     ; //!< Walltime interval between checkpoints
    int64_t checkpoint_iter_interval     ; //!< Iteration interval between checkpoints
    double checkpoint_time_interval      ; //!< Simulation time interval between checkpoints
    double next_checkpoint_time          ; //!< Simulation time at which the next checkpoint should be saved
    int64_t next_checkpoint_iter         ; //!< Iteration at which the next checkpoint should be saved
    double next_checkpoint_wtime         ; //!< Walltime at which the next checkpoint should be saved
    bool have_checkpoint_                ; //!< Whether we have a checkpoint to load at init() 
    //*****************************************************************************************************

    //*****************************************************************************************************
    /**
     * @brief Construct a new checkpoint_handler_impl_t object
     * 
     */
    checkpoint_handler_impl_t() ;
    //*****************************************************************************************************

    //*****************************************************************************************************
    /**
     * @brief Destroy the checkpoint_handler_impl_t object
     * NB: This saves a checkpoint! 
     */
    ~checkpoint_handler_impl_t() ;
    //*****************************************************************************************************

    //*****************************************************************************************************
    friend class utils::singleton_holder<checkpoint_handler_impl_t, memory::default_create> ;
    friend class memory::new_delete_creator<checkpoint_handler_impl_t, memory::new_delete_allocator> ;
    //*****************************************************************************************************
    static constexpr size_t longevity = GRACE_CHECKPOINT_HANDLER ; //!< Schedule destruction
    //*****************************************************************************************************
}   ;
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief The checkpoint handler of GRACE.
 * 
 */
using checkpoint_handler = utils::singleton_holder<checkpoint_handler_impl_t, memory::default_create> ;

} // namespace grace 

#endif /* GRACE_SYSTEM_CHECKPOINT_HANDLER_HH */