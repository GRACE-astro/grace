/**
 * @file grace_inititalize_finalize.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-12
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

#include <grace/system/grace_initialize.hh> 

#include <grace/system/mpi_runtime.hh>
#include <grace/system/p4est_runtime.hh>
#include <grace/system/kokkos_runtime.hh>
#include <grace/system/grace_runtime.hh>

#include <grace/config/config_parser.hh>

#include <grace/amr/connectivity.hh>
#include <grace/amr/forest.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/profiling/profiling_runtime.hh>
#include <grace/utils/device/device_stream_pool.hh>
#include <grace/errors/error.hh>

#include <grace/parallel/mpi_wrappers.hh>

#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>
#include <grace/IO/vtk_output_auxiliaries.hh>

#include <grace/physics/eos/eos_storage.hh>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/common.h>

#include <Kokkos_Core.hpp> 

#include <map> 
#include <string> 
#include <filesystem>
#include <chrono> 
#include <iostream> 



namespace grace {

void initialize_loggers() {
    auto& params = grace::config_parser::get() ; 

    std::map<std::string, spdlog::level::level_enum> logging_levels {
       { "critical", spdlog::level::critical} ,
       { "err"     , spdlog::level::err     } , 
       { "warn"    , spdlog::level::warn    } ,
       { "info"    , spdlog::level::info    } ,
       { "verbose" , spdlog::level::debug   } , 
       { "trace"   , spdlog::level::trace   }
    } ; 
    int const rank = parallel::mpi_comm_rank() ; 

    std::string const console_log_level = params["system"]["console_log_level"].as<std::string>()     ; 
    std::string const file_log_level    = params["system"]["file_log_level"].as<std::string>()        ; 
    bool const flush_on_severity        = params["system"]["flush_logs_based_on_severity"].as<bool>() ;
    std::string const flush_level       = params["system"]["file_log_level"].as<std::string>()        ; 
    bool const flush_on_time            = params["system"]["flush_logs_based_on_time"].as<bool>()     ;
    double const flush_time             = params["system"]["flush_time"].as<double>()                 ;
    
    auto const file_log_basedir  = 
        std::filesystem::path(params["IO"]["log_output_base_directory"].as<std::string>()) ; 
    
    if( not std::filesystem::exists( file_log_basedir ) ){
            std::filesystem::create_directory(file_log_basedir) ; 
    }
    std::string const file_log_filename =  params["IO"]["log_output_base_filename"].as<std::string>() ; 

    std::filesystem::path log_fname = 
        file_log_basedir / (file_log_filename + "_" + std::to_string(rank) + ".out" );
    std::filesystem::path error_log_fname = 
        file_log_basedir / (file_log_filename + "_" + std::to_string(rank) + ".err" ); 
     std::filesystem::path backtrace_log_fname = 
        file_log_basedir / (std::string("backtrace") + "_" + std::to_string(rank) + ".err" ); 
    
    if( rank == 0) {
        auto stdout_logger = spdlog::stdout_color_mt("output_console") ; 
        auto stderr_logger = spdlog::stderr_color_mt("error_console")  ; 
        
        stdout_logger->set_pattern("[%^%l%$] %v") ; 
        stderr_logger->set_pattern("[%^%l%$] %v") ; 

        stdout_logger->set_level(logging_levels[console_log_level]) ; 
        stderr_logger->set_level(logging_levels[console_log_level]) ; 
    }
    
    std::string logger_name = std::string("file_logger_") + std::to_string(rank) ; 

    auto file_logger = spdlog::basic_logger_mt(logger_name, log_fname.string()) ; 
    file_logger->set_pattern("[%d-%m-%Y %H.%M:%S.%e] [%^%l%$] %v") ; 
    file_logger->set_level(logging_levels[file_log_level]) ; 
    logger_name = std::string("error_file_logger_") + std::to_string(rank) ; 
    auto error_file_logger = spdlog::basic_logger_mt(logger_name, error_log_fname.string()) ; 
    error_file_logger->set_pattern("[%d-%m-%Y %H.%M:%S.%e] [%^%l%$] %v") ; 
    error_file_logger->set_level(logging_levels[file_log_level]) ; 
    logger_name = std::string("backtrace_logger_") + std::to_string(rank) ; 
    auto backtrace_logger = spdlog::basic_logger_mt(logger_name, backtrace_log_fname.string()) ; 
    backtrace_logger->set_pattern( "[%d-%m-%Y %H.%M:%S.%e] [%^%l%$] %v") ; 
    backtrace_logger->set_level(spdlog::level::err) ;

    file_logger->info("Startup of file logger completed.") ; 
    
    error_file_logger->flush_on(spdlog::level::err); 
    backtrace_logger->flush_on(spdlog::level::err); 

    if(flush_on_time)
        spdlog::flush_every(std::chrono::duration<double>(flush_time)) ; 
    if(flush_on_severity)
        file_logger->flush_on(logging_levels[flush_level]) ;  

    spdlog::enable_backtrace(32);

    return ;
}


void initialize(int& argc, char* argv[])
{
    /* Find param file in argv */ 
    std::vector<int> iarg ; 
    std::string parfile("./params.yaml") ; 
    for(int i=1; i<argc; ++i) {
        if( std::string(argv[i]) == "--grace-parfile") {
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
    install_signal_handlers(); 
    grace::config_parser::initialize(parfile) ; 
    grace::mpi_runtime::initialize(argc, argv)  ;
    grace::kokkos_runtime::initialize(&argc, argv) ; 
    grace::initialize_loggers() ; 
    grace::device_stream_pool::initialize() ; 
    GRACE_INFO(GRACE_BANNER) ; 
    #ifdef GRACE_ENABLE_PROFILING
    grace::profiling_runtime::initialize() ; 
    #endif
    grace::p4est_runtime::initialize() ; 
    
    #ifdef GRACE_CARTESIAN_COORDINATES
    GRACE_INFO("GRACE running with cartesian coordinates.");
    #endif
    #ifdef GRACE_SPHERICAL_COORDINATES
    GRACE_INFO("GRACE running with spherical coordinates.");
    #endif 
    GRACE_INFO("Inititalizing connectivity object...") ; 
    grace::amr::connectivity::initialize() ; 
    grace::amr::forest::initialize()       ;
    grace::amr::detail::_nx = grace::config_parser::get()["amr"]["npoints_block_x"].as<int64_t>() ;
    grace::amr::detail::_ny = grace::config_parser::get()["amr"]["npoints_block_y"].as<int64_t>() ;
    grace::amr::detail::_nz = grace::config_parser::get()["amr"]["npoints_block_z"].as<int64_t>() ;
    grace::amr::detail::_ngz = grace::config_parser::get()["amr"]["n_ghostzones"].as<int>() ;
    GRACE_INFO("Allocating memory...");
    grace::variable_list::initialize() ;
    grace::runtime::initialize() ; 
    grace::coordinate_system::initialize() ;
    GRACE_INFO("Filling coordinate arrays...") ;
    grace::fill_cell_coordinates(
            grace::variable_list::get().getcoords()
        ,   grace::variable_list::get().getinvspacings()
        ,   grace::variable_list::get().getspacings()
        ,   grace::variable_list::get().getvolumes()
        ,   grace::variable_list::get().getstaggeredcoords() ) ;
    grace::IO::detail::init_auxiliaries()  ;
    grace::eos::initialize() ;
    GRACE_INFO("Initialization done.");
    GRACE_INFO("GRACE running on {} backend", GRACE_BACKEND) ; 
    //GRACE_INFO("GRACE running on {} total devices.", Kokkos::num_devices() ) ; 
    GRACE_INFO("Rank {} mapped to device_id {}", parallel::mpi_comm_rank(), Kokkos::device_id() ) ;
}


} /* namespace grace */