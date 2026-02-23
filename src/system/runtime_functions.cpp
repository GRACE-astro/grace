/**
 * @file runtime_functions.cpp
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

 #include <grace_config.h>
 
 #include <grace/system/grace_runtime.hh>
 #include <grace/system/mpi_runtime.hh>
 #include <grace/system/kokkos_runtime.hh>
 #include <grace/system/p4est_runtime.hh>
 #include <grace/system/runtime_functions.hh>

 #include <grace/config/config_parser.hh>

namespace grace {

int master_rank()
{
    return grace::mpi_runtime::get().master_rank() ; 
}

double get_total_runtime() {
    return grace::runtime::get().total_elapsed() ; 
}

double get_evol_runtime() {
    return grace::runtime::get().elapsed() ; 
}

double get_simulation_time() { 
    return grace::runtime::get().time() ; 
}

void set_simulation_time(double const& _new_t) { 
    grace::runtime::get().set_simulation_time(_new_t) ; 
}

double get_initial_simulation_time() {
    return grace::runtime::get().initial_time() ; 
}

void set_initial_simulation_time(double const& _new_t) {
    grace::runtime::get().set_initial_simulation_time(_new_t) ; 
}

size_t get_iteration() {
    return grace::runtime::get().iteration() ; 
}

void increment_simulation_time() {
    grace::runtime::get().increment_time() ; 
}

void increment_iteration() {
    grace::runtime::get().increment_iteration() ; 
}

void set_iteration(size_t const& _new_it) {
    grace::runtime::get().set_iteration(_new_it) ; 
}

void set_timestep(double const& _new_dt ) {
    grace::runtime::get().set_timestep(_new_dt) ; 
}

double get_timestep() {
    return grace::runtime::get().timestep() ; 
}

bool check_termination_condition()
{
    auto& runtime = grace::runtime::get() ; 
    auto term_cnd = runtime.termination_condition() ; 
    switch (term_cnd) {
        case terminate::ITERATION: 
        return runtime.iteration() >= runtime.max_iteration() ; 
        case terminate::TIME:
        return runtime.time() > runtime.max_time() ; 
        case terminate::WALLTIME:
        return runtime.total_elapsed()/3600 > runtime.max_walltime() ; 
        default:
        ERROR("Unrecognized termination condition.") ; 
    }
}
} /* namespace */ 