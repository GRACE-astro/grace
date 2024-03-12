/**
 * @file macros.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-06-13
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
 * methods to simulate relativistic astrophysical systems and plasma
 * dynamics.
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
#ifndef C412D999_BF0F_4E79_9C1D_C17BE6E275CB
#define C412D999_BF0F_4E79_9C1D_C17BE6E275CB

#ifdef THUNDER_3D 
#define VEC(X,Y,Z) X,Y,Z
#else 
#define VEC(X,Y,Z) X,Y
#endif 

#endif /* C412D999_BF0F_4E79_9C1D_C17BE6E275CB */
