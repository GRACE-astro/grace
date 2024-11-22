/**
 * @file grid_transfer.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief This file contains specialized interpolators to transfer between cell and corner centered grids.
 * @date 2024-11-19
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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
#ifndef GRACE_UTILS_NUMERICS_GRID_TRANSFER_HH
#define GRACE_UTILS_NUMERICS_GRID_TRANSFER_HH

#include <grace_config.h>

#include <grace/utils/device/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/numerics/lagrange_interpolators.hh>

#include <Kokkos_Core.hpp>

namespace grace {

template< size_t order >
struct center_to_corner
{
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    interpolate(
        view_t& view, VEC( int i, int j,int k )
    ) 
    {
        static_assert(false, "order not available.") ; 
    }
} ; 

template<>
struct center_to_corner<2>
{
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    interpolate(
        view_t& view, VEC( int ic, int jc,int kc )
    ) 
    {
        return (view(VEC(-1 + ic,-1 + jc,-1 + kc)) 
              + view(VEC(-1 + ic,-1 + jc,kc)) 
              + view(VEC(-1 + ic,jc,-1 + kc)) 
              + view(VEC(-1 + ic,jc,kc)) 
              + view(VEC(ic,-1 + jc,-1 + kc)) 
              + view(VEC(ic,-1 + jc,kc)) 
              + view(VEC(ic,jc,-1 + kc)) + view(VEC(ic,jc,kc)))/8. ;
    }
} ; 

template<>
struct center_to_corner<4>
{
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    interpolate(
        view_t& view, VEC( int ic, int jc,int kc )
    ) 
    {
        return (-view(VEC(-2 + ic,-2 + jc,-2 + kc)) 
             + 9*view(VEC(-2 + ic,-2 + jc,-1 + kc)) 
             + 9*view(VEC(-2 + ic,-2 + jc,kc)) 
             - view(VEC(-2 + ic,-2 + jc,1 + kc)) 
             + 9*view(VEC(-2 + ic,-1 + jc,-2 + kc)) 
             - 81*view(VEC(-2 + ic,-1 + jc,-1 + kc)) 
             - 81*view(VEC(-2 + ic,-1 + jc,kc)) 
             + 9*view(VEC(-2 + ic,-1 + jc,1 + kc)) 
             + 9*view(VEC(-2 + ic,jc,-2 + kc)) 
             - 81*view(VEC(-2 + ic,jc,-1 + kc)) 
             - 81*view(VEC(-2 + ic,jc,kc)) 
             + 9*view(VEC(-2 + ic,jc,1 + kc)) 
             - view(VEC(-2 + ic,1 + jc,-2 + kc)) 
             + 9*view(VEC(-2 + ic,1 + jc,-1 + kc)) 
             + 9*view(VEC(-2 + ic,1 + jc,kc)) 
             - view(VEC(-2 + ic,1 + jc,1 + kc)) 
             + 9*view(VEC(-1 + ic,-2 + jc,-2 + kc)) 
             - 81*view(VEC(-1 + ic,-2 + jc,-1 + kc)) 
             - 81*view(VEC(-1 + ic,-2 + jc,kc)) 
             + 9*view(VEC(-1 + ic,-2 + jc,1 + kc)) 
             - 81*view(VEC(-1 + ic,-1 + jc,-2 + kc)) 
             + 729*view(VEC(-1 + ic,-1 + jc,-1 + kc)) 
             + 729*view(VEC(-1 + ic,-1 + jc,kc)) 
             - 81*view(VEC(-1 + ic,-1 + jc,1 + kc)) 
             - 81*view(VEC(-1 + ic,jc,-2 + kc)) 
             + 729*view(VEC(-1 + ic,jc,-1 + kc)) 
             + 729*view(VEC(-1 + ic,jc,kc)) 
             - 81*view(VEC(-1 + ic,jc,1 + kc)) 
             + 9*view(VEC(-1 + ic,1 + jc,-2 + kc)) 
             - 81*view(VEC(-1 + ic,1 + jc,-1 + kc)) 
             - 81*view(VEC(-1 + ic,1 + jc,kc)) 
             + 9*view(VEC(-1 + ic,1 + jc,1 + kc)) 
             + 9*view(VEC(ic,-2 + jc,-2 + kc)) 
             - 81*view(VEC(ic,-2 + jc,-1 + kc)) 
             - 81*view(VEC(ic,-2 + jc,kc)) 
             + 9*view(VEC(ic,-2 + jc,1 + kc)) 
             - 81*view(VEC(ic,-1 + jc,-2 + kc)) 
             + 729*view(VEC(ic,-1 + jc,-1 + kc)) 
             + 729*view(VEC(ic,-1 + jc,kc)) 
             - 81*view(VEC(ic,-1 + jc,1 + kc)) 
             - 81*view(VEC(ic,jc,-2 + kc)) 
             + 729*view(VEC(ic,jc,-1 + kc)) 
             + 729*view(VEC(ic,jc,kc)) 
             - 81*view(VEC(ic,jc,1 + kc)) 
             + 9*view(VEC(ic,1 + jc,-2 + kc)) 
             - 81*view(VEC(ic,1 + jc,-1 + kc)) 
             - 81*view(VEC(ic,1 + jc,kc)) 
             + 9*view(VEC(ic,1 + jc,1 + kc)) 
             - view(VEC(1 + ic,-2 + jc,-2 + kc)) 
             + 9*view(VEC(1 + ic,-2 + jc,-1 + kc)) 
             + 9*view(VEC(1 + ic,-2 + jc,kc)) 
             - view(VEC(1 + ic,-2 + jc,1 + kc)) 
             + 9*view(VEC(1 + ic,-1 + jc,-2 + kc)) 
             - 81*view(VEC(1 + ic,-1 + jc,-1 + kc)) 
             - 81*view(VEC(1 + ic,-1 + jc,kc)) 
             + 9*view(VEC(1 + ic,-1 + jc,1 + kc)) 
             + 9*view(VEC(1 + ic,jc,-2 + kc)) 
             - 81*view(VEC(1 + ic,jc,-1 + kc)) 
             - 81*view(VEC(1 + ic,jc,kc)) 
             + 9*view(VEC(1 + ic,jc,1 + kc)) 
             - view(VEC(1 + ic,1 + jc,-2 + kc)) 
             + 9*view(VEC(1 + ic,1 + jc,-1 + kc)) 
             + 9*view(VEC(1 + ic,1 + jc,kc)) 
             - view(VEC(1 + ic,1 + jc,1 + kc)))/4096. ; 
    }
} ; 

template< size_t order >
struct corner_to_center
{
    template< typename view_t >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    interpolate(
        view_t& view, VEC( int i, int j,int k )
    ) 
    {
        return utils::corner_staggered_lagrange_interp_t<2>::threed_interp<view_t>(view,VEC(i,j,k)) ; 
    }
} ;


}

#endif /* GRACE_UTILS_NUMERICS_GRID_TRANSFER_HH */