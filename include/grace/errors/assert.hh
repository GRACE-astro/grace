/**
 * @file assert.hh
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
#ifndef GRACE_ASSERT_HH
#define GRACE_ASSERT_HH

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include <grace/utils/make_string.hh>
#include <grace/errors/abort.hh>
//**************************************************************************************************
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Assert for a condition only when in debug mode.
 * \ingroup system
 * @param a boolean condition to check.
 * @param m string stream to be printed in case of failure.
 * NB: This is a no-op in production mode.
 */
//**************************************************************************************************
#ifdef GRACE_DEBUG
#define ASSERT_DBG(a,m)                                                   \
do {                                                                      \
    if(!(a)) {                                                            \
        utils::make_string error_message_ASSERT;                          \
        error_message_ASSERT << m ;                                       \
        abort_with_message(#a,                                            \
                           __FILE__,__LINE__,                             \
                           static_cast<const char*>(__PRETTY_FUNCTION__), \
                           error_message_ASSERT);                         \
    }                                                                     \
} while (false)
#else
#define ASSERT_DBG(a,m)
#endif
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Assert for a condition.
 * \ingroup system
 * @param a boolean condition to check.
 * @param m string stream to be printed in case of failure.
 */
//**************************************************************************************************
#define ASSERT(a,m)                                                       \
do {                                                                      \
    if(!(a)) {                                                            \
        utils::make_string error_message_ASSERT;                          \
        error_message_ASSERT << m ;                                       \
        abort_with_message(#a,                                            \
                           __FILE__,__LINE__,                             \
                           static_cast<const char*>(__PRETTY_FUNCTION__),       \
                           error_message_ASSERT);                   \
    }                                                                     \
} while (false)
#endif
//**************************************************************************************************

//**************************************************************************************************
//**************************************************************************************************