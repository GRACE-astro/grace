/**
 * @file lifetime_tracker.cc
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-03-13
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
#include <thunder/utils/lifetime_tracker.hh>
namespace utils { namespace pvt {
//**************************************************************************************************
unsigned int elements = 0 ;
//**************************************************************************************************
tracker_array ptracker_array = nullptr ;
//**************************************************************************************************
/**
 * @brief Scheduled at termination
 * 
 * This function calls destructors of the singletons in increasing order of their longevity.
 */
//**************************************************************************************************
void at_exit_func() {
    ASSERT(elements > 0 && ptracker_array != nullptr, 
           "std::atexit scheduled to delete null pointer."
           " Something is wrong"  ); 
    lifetime_tracker * ptop = ptracker_array[ elements - 1]  ;
    ptracker_array = static_cast<tracker_array> ( 
        std::realloc(ptracker_array, sizeof(*ptracker_array) * --elements) 
    ) ;
    delete ptop ; 
}
//**************************************************************************************************
} }
