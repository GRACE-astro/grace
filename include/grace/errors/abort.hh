/**
 * @file abort.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-03-13
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
#ifndef GRACE_ABORT_HH
#define GRACE_ABORT_HH

#include <string>
#include <grace_config.h>

//**************************************************************************************************
//**************************************************************************************************
//! \cond grace_detail
//**************************************************************************************************
/**
 * @brief Abort execution by throwing <code>std::runtime_error</code> and printing a message.
 * @param expression failed assertion.
 * @param file file where assertion failed.
 * @param line line where assertion failed.
 * @param function function where assertion failed.
 * @param message error message to be displayed.
 * 
 * The message gets appended to a backtrace.
 */
//**************************************************************************************************
[[noreturn]] void abort_with_message( const char* expression, 
                                      const char* file,
                                      int line,
                                      const char* function,
                                      const std::string& message ) ;
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Abort execution by throwing <code>std::runtime_error</code> and printing a message.
 * \ingroup system
 * @param file file where assertion failed.
 * @param line line where assertion failed.
 * @param function function where assertion failed.
 * @param message error message to be displayed.
 * 
 * The message gets appended to a backtrace. 
 */
//**************************************************************************************************
[[noreturn]] void abort_with_message( const char* file,
                                      int line,
                                      const char* function,
                                      const std::string& message ) ;
//**************************************************************************************************
//! \endcond
//**************************************************************************************************
//**************************************************************************************************
#endif 