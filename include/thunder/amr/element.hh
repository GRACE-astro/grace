/**
 * @file element.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2023-06-12
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
#ifndef A31235EF_9C50_43E4_AE5C_56A821B5603C
#define A31235EF_9C50_43E4_AE5C_56A821B5603C

#include <thunder/thunder_config.h> 
#include <thunder/amr/p4est_headers.hh>
#include <thunder/data_structures/data_vector.hh>
#include <thunder/data_structures/macros.hh>

namespace thunder { namespace amr { 


template< size_t ndim                   = THUNDER_NSPACEDIM             // number of (spatial) dimensions
        , coordinate_system coordinates = THUNDER_COORDINATE_SYSTEM >   // coordinate system 
class element {
 public:
    /**
    * @brief Convert a set of logical coordinates to physical coordinates.
    * 
    * @param xi         First logical coordinate
    * @param eta        Second logical coordinate
    * @param zeta       Third logical coordinate
    * @param x          
    * @param y 
    * @param z 
    * @return THUNDER_ALWAYS_INLINE 
    */
    THUNDER_ALWAYS_INLINE void 
    logical_to_physical( Kokkos::View<double*> xi, Kokkos::View<double*> eta, Kokkos::View<double*> zeta,
                         Kokkos::View<double*> x, Kokkos::View<double*> y, Kokkos::View<double*> z  )
    {

    } ; 


 private:
    p4est_quadrant_t* pquad_                   ; 
    int32_t ord_                               ;
} ; 


}} // namespace thunder/amr
#endif /* A31235EF_9C50_43E4_AE5C_56A821B5603C */
