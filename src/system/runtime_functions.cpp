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
    return grace::runtime::get().elapsed() ; 
}

double get_simulation_time() { 
    return grace::runtime::get().time() ; 
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

void set_timestep(double const& _new_dt ) {
    grace::runtime::get().set_timestep(_new_dt) ; 
}

double get_timestep() {
    return grace::runtime::get().timestep() ; 
}

} /* namespace */ 