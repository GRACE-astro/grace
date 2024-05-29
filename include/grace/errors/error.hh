/**
 * @file error.hh
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
#ifndef GRACE_ERROR_HH
#define GRACE_ERROR_HH

#include <string>
#include <stdexcept>
#include <type_traits>

#include <grace/errors/abort.hh>
#include <grace/utils/make_string.hh>
//**************************************************************************************************
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Throw a <code>std::runtime_error()</code> and abort the run with a message
 *        and a backtrace.
 * \ingroup system
 * @param m message to be displayed.
 */
//**************************************************************************************************
#define ERROR(m)                                                                \
do {                                                                            \
        abort_with_message(__FILE__,__LINE__,                                   \
                           static_cast<const char*>(__PRETTY_FUNCTION__),       \
                           utils::make_string{} << m ) ;                        \
} while(false)
//**************************************************************************************************
//! \cond grace_detail
//**************************************************************************************************
/**
 * @brief Install signal handler that catches <code>FP / SISGEV / SIGINT</code> and throws 
 *        <code>ERROR</code> instead.
 * \ingroup system
 */
void install_signal_handlers() ;
//**************************************************************************************************
//! \endcond
//**************************************************************************************************
//**************************************************************************************************
#endif