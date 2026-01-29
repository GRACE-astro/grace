/**
 * @file grace/errors/warn.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief define the WARN macro
 * @version 0.1
 * @date 2023-03-10
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

#ifndef GRACE_ERRORS_WARN_HH
#define GRACE_ERRORS_WARN_HH
//**************************************************************************************************
//**************************************************************************************************
//! \cond grace_detail
//**************************************************************************************************
/**
 * @brief Emit a warning message.
 * \ingroup system
 * @param warning_level severity of the warning. Level 0 aborts.
 * @param file file where the warning came from
 * @param line line where the warning came from
 * @param function function where the warning came from
 * @param fmt C-style format string
 * @param ... format arguments
 */
//**************************************************************************************************
void emit_warning_with_message(int warning_level, 
                               const char* file,
                               int line,
                               const char* function,
                               const char* fmt, ... ) ;
//**************************************************************************************************
//! \endcond
//**************************************************************************************************
/**
 * @brief macro wrapper to emit warning with message
 * \ingroup system
 * @param l level of the warning. Level 0 aborts.
 * @param m C-style format string 
 * @param ... format arguments
 */
//**************************************************************************************************
#define WARN(l,m,...) \
do { \
    emit_warning_with_message(l,__FILE__,__LINE__, \
                              static_cast<const char*>(__PRETTY_FUNCTION__), \
                              m ## __VA_OPT__(,) __VA_ARGS__); \
} while(false)
//**************************************************************************************************

//**************************************************************************************************
//**************************************************************************************************
#endif 