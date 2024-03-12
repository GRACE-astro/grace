/**
 * @file print.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-12
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

#include <thunder/system/print_impl.hh> 
#include <thunder/system/thunder_runtime.hh>
#include <thunder/parallel/mpi_wrappers.hh> 

#include <iostream>

void print_message(int level, std::string const& message)
{
    auto& runtime = thunder::runtime::get() ; 
    int rank = parallel::mpi_comm_rank() ; 
    if( rank == runtime.master_rank and level < runtime.print_threshold )
    {
        std::cout << message << std::endl ;
    }
}