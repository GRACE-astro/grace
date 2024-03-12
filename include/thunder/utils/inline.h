/**
 * @file inline.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2023-03-22
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
#ifndef THUNDER_UTILS_INLINE_H
#define THUNDER_UTILS_INLINE_H

//**********************************************************************************************
//**********************************************************************************************
//**********************************************************************************************
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define THUNDER_FORCE_INLINE __forceinline
#else
#define THUNDER_FORCE_INLINE inline
#endif 
//**********************************************************************************************
//**********************************************************************************************
//**********************************************************************************************


//**********************************************************************************************
//**********************************************************************************************
//**********************************************************************************************
#if THUNDER_USE_ALWAYS_INLINE && defined(__GNUC__)
#define THUNDER_ALWAYS_INLINE inline __attribute__((always_inline))
#else 
#define THUNDER_ALWAYS_INLINE THUNDER_FORCE_INLINE 
#endif 
//**********************************************************************************************
//**********************************************************************************************
//**********************************************************************************************

#endif /* THUNDER_UTILS_INLINE_H */
