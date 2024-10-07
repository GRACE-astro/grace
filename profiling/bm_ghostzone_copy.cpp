/**
 * @file bm_ghostzone_copy.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-10-07
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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
#include <grace/system/grace_initialize.hh>
#include <grace/errors/assert.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include<> 

#define GRACE_BENCHMARK_MAIN()                                                                          \
int main( int argc, char* argv [])                                                                      \
{                                                                                                       \
    grace::initialize(argc,argv) ;                                                                      \
    ASSERT( paralle::mpi_comm_size() == 1, "Micro-benchmarks should be ran on 1 rank only." ) ;         \
    benchmark::Initialize(&argc, argv);                                                                 \
    benchmark::RunSpecifiedBenchmarks();                                                                \
    benchmark::Shutdown();                                                                              \
}

