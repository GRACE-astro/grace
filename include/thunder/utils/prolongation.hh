/**
 * @file prolongation.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-04-08
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

#ifndef THUNDER_UTILS_PROLONGATION_HH
#define THUNDER_UTILS_PROLONGATION_HH

namespace utils {

template < typename LimT 
         , size_t ndim > 
struct linear_prolongator_t 
{ } ; 

template< typename LimT > 
struct linear_prolongator_t<LimT,2> 
{
    static constexpr size_t stencil_size = 5UL ; 

    template< typename VarViewT
            , typename VolViewT >
    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    interpolate(  int i_f, int j_f
                , int i_c, int j_c 
                , VarViewT& fine_view 
                , VarViewT& coarse_view 
                , VolViewT& fine_vol 
                , VolViewT& coarse_vol )
    {
        
    }

} ; 
}
#endif /* THUNDER_UTILS_PROLONGATION_HH */