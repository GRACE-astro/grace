/**
 * @file runtime_functions.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
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
#ifndef GRACE_SYSTEM_RUNTIME_FUNCTIONS 
#define GRACE_SYSTEM_RUNTIME_FUNCTIONS

namespace grace {
/**
 * @brief Get the master rank.
 * 
 * @return int The rank which is allowed to print to console.
 */
int master_rank() ;
/**
 * @brief Get the total runtime of this simulation.
 * 
 * @return double The runtime since <code>grace_initialize</code> was called (in seconds).
 */
double get_total_runtime() ; 
/**
 * @brief Get the simulation time.
 * 
 * @return double The simulation time.
 */
double get_simulation_time() ; 
/**
 * @brief Get the time at which this run started.
 * 
 * @return double The initial simulation time.
 */
double get_initial_simulation_time() ; 
/**
 * @brief Increment the simulation time 
 *        by the current timestep.
 */
void increment_simulation_time() ; 
/**
 * @brief Set the simulation time 
 *        by the current timestep.
 */
void set_simulation_time(double const& _new_t) ; 
/**
 * @brief Set the initial simulation time (only done when reading checkpoints)
 */
void set_initial_simulation_time(double const& _new_t) ; 
/**
 * @brief Get the iteration count.
 * 
 * @return size_t Current iteration count.
 */
size_t get_iteration() ; 
/**
 * @brief Increment iteration count.
 */
void increment_iteration() ; 
/**
 * @brief Set iteration count.
 */
void set_iteration(size_t const& _new_it) ; 
/**
 * @brief Set the timestep size.
 * 
 * @param _new_dt New timestep.
 */
void set_timestep(double const& _new_dt ) ; 
/**
 * @brief Get the timestep.
 * 
 * @return double The timestep size.
 */
double get_timestep() ;

bool check_termination_condition() ; 
}

#endif 