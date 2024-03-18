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

#include <thunder/utils/make_string.hh>
#include <thunder/system/print_impl.hh>

enum message_print_thresholds {
    HIGH_PRIORITY=0,
    VERBOSE,
    VERYVERBOSE 
} ; 



#define THUNDER_INFO(l,t,m)                 \
do {                                        \
 utils::make_string msg ;                   \
 msg << "[" << t << "]: " ;                 \
 print_message(l,msg << m) ;                \
} while(false)                              \

#define THUNDER_PRINT(m)                                  \
do {                                                      \
 print_message(HIGH_PRIORITY,utils::make_string{} << m) ; \
} while(false)                                            \


#endif /* INCLUDE_THUNDER_SYSTEM_PRINT */
