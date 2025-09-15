/**
 * @file print.hh
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
#ifndef INCLUDE_GRACE_SYSTEM_PRINT
#define INCLUDE_GRACE_SYSTEM_PRINT

#include <grace_config.h>

#include <grace/parallel/mpi_wrappers.hh> 
#include <grace/system/runtime_functions.hh> 

#include <string> 
#include <spdlog/spdlog.h>


/**
 * @brief Log a message with "info" priority on files and
 *        console.
 * \ingroup system
 */
#define GRACE_INFO(...)                          \
do {                                               \
 int grace__internal__rank__ = parallel::mpi_comm_rank() ;        \
 if( grace__internal__rank__ == 0 )                               \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->info(__VA_ARGS__) ;                   \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(grace__internal__rank__) ;                     \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->info(__VA_ARGS__) ;                      \
} while(false)                                     \

/**
 * @brief Log a message with "critical" priority on files and
 *        console.
 * \ingroup system
 */
#define GRACE_CRITICAL(...)        \
do {                                               \
 int grace__internal__rank__ = parallel::mpi_comm_rank() ;        \
 if( grace__internal__rank__ == 0 )                               \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->critical(__VA_ARGS__) ;               \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(grace__internal__rank__) ;                     \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->critical(__VA_ARGS__) ;                  \
} while(false)                                     \

/**
 * @brief Log a message with "warn" priority on files and
 *        console.
 * \ingroup system
 */
#define GRACE_WARN(...)          \
do {                                               \
 int grace__internal__rank__ = parallel::mpi_comm_rank() ;        \
 if( grace__internal__rank__ == 0 )                               \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->warn(__VA_ARGS__) ;                   \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(grace__internal__rank__) ;                     \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->warn(__VA_ARGS__) ;                      \
} while(false)                                     \

/**
 * @brief Log a message with "debug" priority on files and
 *        console.
 * \ingroup system
 */
#define GRACE_VERBOSE(...)        \
do {                                               \
 int grace__internal__rank__ = parallel::mpi_comm_rank() ;        \
 if( grace__internal__rank__ == 0 )                               \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->debug(__VA_ARGS__) ;                  \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(grace__internal__rank__) ;                     \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->debug(__VA_ARGS__) ;                     \
} while(false)                                     \

/**
 * @brief Log a message with "trace" priority on files and
 *        console.
 * \ingroup system
 */
#define GRACE_TRACE(...)          \
do {                                               \
 int grace__internal__rank__ = parallel::mpi_comm_rank() ;        \
 if( grace__internal__rank__ == 0 )                               \
 {                                                 \
    auto console = spdlog::get("output_console") ; \
    console->trace(__VA_ARGS__) ;                  \
 }                                                 \
 std::string logger_name =                         \
    std::string("file_logger_")                    \
  + std::to_string(grace__internal__rank__) ;                     \
 auto logfile = spdlog::get(logger_name) ;         \
 logfile->trace(__VA_ARGS__) ;                     \
} while(false)                                     \

#endif /* INCLUDE_GRACE_SYSTEM_PRINT */
