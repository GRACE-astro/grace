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



// template <int edgedir>
// consteval std::tuple<int, int> get_complementary_dirs() {
//     constexpr std::array<std::tuple<int, int>, 3> complementary_dirs = {{
//         {1, 2}, // if edgedir == 0
//         {0, 2}, // if edgedir == 1
//         {0, 1}  // if edgedir == 2
//     }};
//     return complementary_dirs[edgedir];
// }

/**
 * @brief Restriction operator for edge-staggered variables.
 * \ingroup utils
 * \tparam edgeDir - non-staggered direction 
 */
template <size_t edgeDir> 
struct line_average_restrictor_t {


template< typename StateViewT
        , typename LineViewT >
static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
apply(
      VEC(int const& i, int const& j, int const& k)
    , StateViewT& state 
    , LineViewT& line 
    , int64_t iq 
    , int ivar  ) 
{
    return  (
      state(VEC(i,j,k),ivar,iq) * line(VEC(i,j,k),iq)
    + state(VEC(i+delta(0,edgeDir),j+delta(1,edgeDir),k+delta(2,edgeDir)),ivar,iq) * line(VEC(i+delta(0,edgeDir),j+delta(1,edgeDir),k+delta(2,edgeDir)),iq)
    ) / (
      line(VEC(i,j,k),iq) + line(VEC(i+delta(0,edgeDir),j+delta(1,edgeDir),k+delta(2,edgeDir)),iq)
    ) ; 
}
#ifdef GRACE_CARTESIAN_COORDINATES
/**
 * @brief Overload of edge-staggered restriction operator for Cartesian coordinates.
 *        Essentially just a trivial average of the fine edge values
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

  return  (
      state(VEC(i,j,k),ivar,iq)
    + state(VEC(i+delta(0,edgeDir),j+delta(1,edgeDir),k+delta(2,edgeDir)),ivar,iq)
    ) / 2.0 ; 
}
#endif 


#ifdef GRACE_CARTESIAN_COORDINATES

/**
 * @brief Overload of edge-staggered restriction operator for Cartesian coordinates.
 *        Automatically takes care of the extra stagger points without changing the policies for grid transfer operations
 *          
 * @tparam StateViewT Type of coarse state array.
 * @tparam StateViewT Type of fine state array.
 * @param coarse_state state array.
 * @param iq    quadrant index
 * @param ivar  variable index
 * @return void
 */
template< typename CoarseViewT, typename FineViewT >
static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
apply(VEC(int const& i_c, int const& j_c, int const& k_c)
    , VEC(int const& i_f, int const& j_f, int const& k_f)
    , CoarseViewT& coarse_state
    , FineViewT& fine_state
    , int64_t iq_c
    , int64_t iq_f
    , int ivar  ) 
{

    static constexpr int idir = std::get<0>(get_complementary_dirs<edgeDir>());
    static constexpr int jdir = std::get<1>(get_complementary_dirs<edgeDir>());

  // this take care of the additional staggered points in the directions orthogonal to edgeDir
  coarse_state(VEC(i_c,j_c,k_c), ivar, iq_c)  = (fine_state(VEC(i_f,j_f,k_f),ivar, iq_f)
                                                 +fine_state(VEC(i_f+delta(0,edgeDir),j_f+delta(1,edgeDir),k_f+delta(2,edgeDir)),ivar,iq_f)
                                                  ) / 2.0 ; 

  // coarse_state(VEC(i_c+delta(0,idir),j_c+delta(1,idir),k_c+delta(2,idir)), ivar, iq_c)  = (fine_state(VEC(i_f,j_f,k_f),ivar, iq_f)
  //                                                +fine_state(VEC(i_f+2*delta(0,idir)+delta(0,edgeDir),j_f+2*delta(1,idir)+delta(1,edgeDir),k_f+2*delta(2,idir)+delta(2,edgeDir)),ivar,iq_f)
  //                                                 ) / 2.0 ; 

  // coarse_state(VEC(i_c+delta(0,jdir),j_c+delta(1,jdir),k_c+delta(2,jdir)), ivar, iq_c)  = (fine_state(VEC(i_f,j_f,k_f),ivar, iq_f)
  //                                                +fine_state(VEC(i_f+2*delta(0,jdir)+delta(0,edgeDir),j_f+2*delta(1,jdir)+delta(1,edgeDir),k_f+2*delta(2,jdir)+delta(2,edgeDir)),ivar,iq_f)
  //                                                 ) / 2.0 ; 

  // coarse_state(VEC(i_c+delta(0,idir)+delta(0,jdir),j_c+delta(1,idir)+delta(1,jdir),k_c+delta(2,idir)+delta(2,jdir)), ivar, iq_c)  = (fine_state(VEC(i_f,j_f,k_f),ivar, iq_f)
  //                                                +fine_state(VEC(i_f+2*delta(0,idir)+2*delta(0,jdir)+delta(0,edgeDir),j_f+2*delta(1,idir)+2*delta(1,jdir)+delta(1,edgeDir),k_f+2*delta(2,idir)+2*delta(2,jdir)+delta(2,edgeDir)),ivar,iq_f)
  //                                                 ) / 2.0 ; 


}
#endif 


} ; 


}

#endif /* GRACE_UTILS_RESTRICTION_HH */