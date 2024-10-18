/**
 * @file restriction.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-04
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

#include <grace_config.h>
#include <grace/utils/device/device.h>
#include <grace/utils/inline.h> 
#include <grace/utils/numerics/math.hh>

#include <Kokkos_Core.hpp> 

#ifndef GRACE_UTILS_RESTRICTION_HH
#define GRACE_UTILS_RESTRICTION_HH

namespace utils {

struct vol_average_restrictor_t {
template< typename StateViewT
        , typename VolViewT >
static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
apply(
      VEC(int const& i, int const& j, int const& k)
    , StateViewT& state 
    , VolViewT& vol 
    , int64_t iq 
    , int ivar  ) 
{
    return  (EXPR(
      state(VEC(i,j,k),ivar,iq)*vol(VEC(i,j,k),iq)
    + state(VEC(i+1,j,k),ivar,iq)*vol(VEC(i+1,j,k),iq),
    + state(VEC(i,j+1,k),ivar,iq)*vol(VEC(i,j+1,k),iq)
    + state(VEC(i+1,j+1,k),ivar,iq)*vol(VEC(i+1,j+1,k),iq),
    + state(VEC(i,j,k+1),ivar,iq)*vol(VEC(i,j,k+1),iq)
    + state(VEC(i,j+1,k+1),ivar,iq)*vol(VEC(i,j+1,k+1),iq)
    + state(VEC(i+1,j,k+1),ivar,iq)*vol(VEC(i+1,j,k+1),iq)
    + state(VEC(i+1,j+1,k+1),ivar,iq)*vol(VEC(i+1,j+1,k+1),iq)
    )) / (EXPR(
      vol(VEC(i,j,k),iq)   + vol(VEC(i+1,j,k),iq  ),
    + vol(VEC(i,j+1,k),iq) + vol(VEC(i+1,j+1,k),iq),
    + vol(VEC(i,j,k+1),iq) + vol(VEC(i,j+1,k+1),iq)
    + vol(VEC(i+1,j,k+1),iq) + vol(VEC(i+1,j+1,k+1),iq)
    )) ; 
}
#ifdef GRACE_CARTESIAN_COORDINATES
/**
 * @brief Overload of restriction operator for Cartesian coordinates.
 * 
 * @tparam StateViewT Type of state array.
 * @param state state array.
 * @param iq    quadrant index
 * @param ivar  variable index
 * @return double The restricted coarse value of var computed from the fine values.
 */
template< typename StateViewT >
static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
apply(
      VEC(int const& i, int const& j, int const& k)
    , StateViewT& state 
    , int64_t iq 
    , int ivar  ) 
{
  return  (EXPR(
      state(VEC(i,j,k),ivar,iq)
    + state(VEC(i+1,j,k),ivar,iq),
    + state(VEC(i,j+1,k),ivar,iq)
    + state(VEC(i+1,j+1,k),ivar,iq),
    + state(VEC(i,j,k+1),ivar,iq)
    + state(VEC(i,j+1,k+1),ivar,iq)
    + state(VEC(i+1,j,k+1),ivar,iq)
    + state(VEC(i+1,j+1,k+1),ivar,iq)
    )) / P4EST_CHILDREN ; 
}
#endif 

} ; 

}

#endif /* GRACE_UTILS_RESTRICTION_HH */