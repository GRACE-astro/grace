/**
 * @file runtime_functions.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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

 #include <thunder_config.h>
 
 #include <thunder/system/thunder_runtime.hh>
 #include <thunder/system/mpi_runtime.hh>
 #include <thunder/system/kokkos_runtime.hh>
 #include <thunder/system/p4est_runtime.hh>
 #include <thunder/system/runtime_functions.hh>

 #include <thunder/config/config_parser.hh>

namespace thunder {

int master_rank()
{
    return thunder::mpi_runtime::get().master_rank() ; 
}

int print_threshold()
{
    return thunder::mpi_runtime::get().print_threshold() ; 
}

} /* namespace */ 