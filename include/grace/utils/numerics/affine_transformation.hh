/**
 * @file affine_transformation.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-04-18
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
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
#ifndef GRACE_UTILS_AFFINE_TRANSFORMATION_HH 
#define GRACE_UTILS_AFFINE_TRANSFORMATION_HH

#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>

#include <array>
namespace utils {

/**
 * @brief Apply an affine transformation.
 * 
 * @tparam T Type of data.
 * @param x Point to be transformed.
 * @param A Starting point of old interval.
 * @param B End point of old interval.
 * @param a Starting point of new interval.
 * @param b End point of new interval.
 * @return T The mapping of x from \f$[A,B]\f$ to \f$[a,b]\f$.
 */
template< typename T >
static T GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
affine_transformation(
    T const& x,
    T const& A, T const& B, 
    T const& a, T const& b ) 
{
    return b/(B-A)*(x-A)+a/(B-A)*(B-x) ; 
}


}

#endif /*GRACE_UTILS_AFFINE_TRANSFORMATION_HH*/