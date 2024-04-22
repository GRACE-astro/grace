/**
 * @file affine_transformation.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-04-18
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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
#ifndef THUNDER_UTILS_AFFINE_TRANSFORMATION_HH 
#define THUNDER_UTILS_AFFINE_TRANSFORMATION_HH

#include <thunder/utils/inline.h>

#include <array>
namespace utils {

template< typename T >
static T THUNDER_ALWAYS_INLINE
affine_transformation(
    T const& x,
    T const& A, T const& B, 
    T const& a, T const& b ) 
{
    return b/(B-A)*(x-A)+a/(B-A)*(B-x) ; 
}


}

#endif /*THUNDER_UTILS_AFFINE_TRANSFORMATION_HH*/