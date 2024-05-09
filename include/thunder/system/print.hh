/**
 * @file print.hh
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
#ifndef INCLUDE_THUNDER_SYSTEM_PRINT
#define INCLUDE_THUNDER_SYSTEM_PRINT

#include <thunder_config.h>

#include <thunder/parallel/mpi_wrappers.hh> 
#include <thunder/system/runtime_functions.hh> 

#include <string> 
#include <spdlog/spdlog.h>


/**
 * @brief Log a message with "info" priority on files and
 *        console.
 * \ingroup system
 */
#define THUNDER_INFO(...)                          \
do {                                               \
 int rank = parallel::mpi_comm_rank() ;            \
 if( rank == 0 )                                   \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->info(__VA_ARGS__) ;                   \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(rank) ;                         \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->info(__VA_ARGS__) ;                      \
} while(false)                                     \

/**
 * @brief Log a message with "critical" priority on files and
 *        console.
 * \ingroup system
 */
#define THUNDER_CRITICAL(...)        \
do {                                               \
 int rank = parallel::mpi_comm_rank() ;            \
 if( rank == 0 )                                   \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->critical(__VA_ARGS__) ;               \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(rank) ;                         \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->critical(__VA_ARGS__) ;                  \
} while(false)                                     \

/**
 * @brief Log a message with "warn" priority on files and
 *        console.
 * \ingroup system
 */
#define THUNDER_WARN(...)          \
do {                                               \
 int rank = parallel::mpi_comm_rank() ;            \
 if( rank == 0 )                                   \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->warn(__VA_ARGS__) ;                   \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(rank) ;                         \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->warn(__VA_ARGS__) ;                      \
} while(false)                                     \

/**
 * @brief Log a message with "debug" priority on files and
 *        console.
 * \ingroup system
 */
#define THUNDER_VERBOSE(...)        \
do {                                               \
 int rank = parallel::mpi_comm_rank() ;            \
 if( rank == 0 )                                   \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->debug(__VA_ARGS__) ;                  \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(rank) ;                         \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->debug(__VA_ARGS__) ;                     \
} while(false)                                     \

/**
 * @brief Log a message with "trace" priority on files and
 *        console.
 * \ingroup system
 */
#define THUNDER_TRACE(...)          \
do {                                               \
 int rank = parallel::mpi_comm_rank() ;            \
 if( rank == 0 )                                   \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->trace(__VA_ARGS__) ;                  \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(rank) ;                         \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->trace(__VA_ARGS__) ;                     \
} while(false)                                     \

#endif /* INCLUDE_THUNDER_SYSTEM_PRINT */
