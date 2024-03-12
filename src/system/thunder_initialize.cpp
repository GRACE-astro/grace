/**
 * @file thunder_inititalize_finalize.cpp
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

#include <thunder_config.h>

#include <thunder/system/thunder_initialize.hh> 

#include <thunder/system/mpi_runtime.hh>
#include <thunder/system/p4est_runtime.hh>
#include <thunder/system/kokkos_runtime.hh>
#include <thunder/system/thunder_runtime.hh>

#include <thunder/config/config_parser.hh>

#include <thunder/amr/connectivity.hh>
#include <thunder/amr/forest.hh>
#include <thunder/amr/coordinates.hh>

#include <thunder/data_structures/variables.hh>

#include <thunder/system/print.hh>

namespace thunder {

void initialize(int& argc, char* argv[])
{
    /* Find param file in argv */ 
    std::vector<int> iarg ; 
    std::string parfile("./params.yaml") ; 
    for(int i=1; i<argc; ++i) {
        if( std::string(argv[i]) == "--thunder-parfile") {
            iarg.push_back(i) ; 
            if( i+1 < argc ){
                parfile = std::string(argv[i+1]) ;
                iarg.push_back(i+1) ;
            }
        }
    }
    int argc_new = argc-iarg.size() ; 
    char* argv_new[argc_new] ; 
    int inew=0 ; 
    for( int i=0; i < argc; ++i){
        bool exclude=false ; 
        for( auto const& ii: iarg) exclude = (ii==i) ; 
        if ( not exclude) {
            argv_new[inew] = argv[i] ; 
            inew++ ; 
        }
    }
    argc = argc_new;
    argv = argv_new;  
    /* Initialize global objects in correct order */ 
    thunder::config_parser::initialize(parfile) ; 
    thunder::kokkos_runtime::initialize(&argc, argv) ; 
    thunder::mpi_runtime::initialize(argc, argv)  ; 
    thunder::runtime::initialize() ; 
    thunder::amr::connectivity::initialize() ; 
    thunder::amr::forest::initialize()       ;
    thunder::variable_list::initialize() ; 
    thunder::fill_coordinates() ; 
}


} /* namespace thunder */