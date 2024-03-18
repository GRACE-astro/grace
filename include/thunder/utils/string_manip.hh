/**
 * @file string_manip.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
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
#ifndef THUNDER_UTILS_STRING_MANIP_HH
#define THUNDER_UTILS_STRING_MANIP_HH

#include <thunder/utils/inline.h> 

#include <string> 
#include <sstream> 
#include <iomanip>

namespace utils {

template< typename T >
THUNDER_ALWAYS_INLINE std::string 
zero_padded( T const& in
           , int padding_size )
{ 
    std::string s = std::to_string(in) ; 
    return s.insert(0,padding_size,'0'); 
}


} /* namespace thunder::utils */ 

#endif /* THUNDER_UTILS_STRING_MANIP_HH */ 
