/**
 * @file div_preserving_prolongator.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-08-29
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
#ifndef GRACE_UTILS_DIV_PRESERVING_PROLONGATOR_HH
#define GRACE_UTILS_DIV_PRESERVING_PROLONGATOR_HH

#include <grace_config.h>
#include <grace/utils/numerics/math.hh>
#include <grace/utils/device/device.h>
#include <grace/utils/inline.h>

#include <Kokkos_Core.hpp>

namespace grace {

template< typename LimT >
struct div_preserving_prolongator_t {

    template< typename VarViewT
            , typename CoordViewT
            , typename VolViewT >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    interpolate(  VEC(int i_f, int j_f, int k_f)
                , VEC(int i_c, int j_c, int k_c)
                , int64_t q_f, int64_t q_c, int ngz, int ivar
                , CoordViewT& fine_coords
                , CoordViewT& coarse_coords 
                , CoordViewT& fine_dx  
                , CoordViewT& coarse_dx  
                , VarViewT& coarse_view 
                , VolViewT& fine_vol 
                , VolViewT& coarse_vol )
    {
        
    }
} ; 

}
#endif 