/**
 * @file curl_operators.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-01-04
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
#include <grace/utils/inline.h> 
#include <grace/utils/numerics/math.hh>

#include <Kokkos_Core.hpp> 

#ifndef GRACE_UTILS_CURL_OPERATORS_HH
#define GRACE_UTILS_CURL_OPERATORS_HH

namespace utils {

struct curl_operator_t {
  /**
 * @brief The curl operator used for converting the vector potential into magnetic field.
 *        Acts point-wise to produce the discretization of B^i = \eta^ijk D_j A_k
 *        TODO: implement e.g. the STAGGERED version of the b_from_vectorpotentialA function
 *              from the BHAC code 
 * @tparam FaceViewsT Type of face-staggered variables' arrays.
 * @tparam EdgeViewsT Type of edge-staggered variables' arrays.
 * @tparam SurfViewT Type of surface area arrays.
 * @tparam LineViewT Type of line length arrays.
 * @param state state array.
 * @param state state array.
 * @param state state array.
 * @param state state array.
 * @param state state array.
 * @param state state array.
 * @details By using the geometric data (line lengths and surface areas of cells), this routine initializes
 *          \b the \b magnetic \b field \b B^i \b directly (i.e. a true spatial 3-vector)
 *          This routine is second-order and to be interpreted in the finite-volume sense
 * @return void
 */
template< typename FaceViewsT,
          typename EdgeViewsT,
          typename SurfViewT,
          typename LineViewT >
static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
apply(
      VEC(int const& i, int const& j, int const& k)
    , FaceViewsT& faces_state 
    , EdgeViewsT& edges_state 
    , SurfViewT& surf 
    , LineViewT& line) 
{
  
  static_assert(false, "Not yet implemented.");

}
#ifdef GRACE_CARTESIAN_COORDINATES
/**
 * @brief Overload of the curl operator for Cartesian coordinates.
 *        Acts point-wise to produce the discretization of \sqrt{\gamma} B^i = \epsilon^ijk D_j A_k
 * 
 * @tparam FaceViewT Type of face-staggered variables' arrays, (e.g. 5-dim Kokkos view specialized for the variable and quadrant) 
 * @tparam EdgeViewT Type of edge-staggered variables' arrays. (e.g. 5-dim Kokkos view specialized for the variable and quadrant) 
 * @param Fx x-face (staggered directions: x) state array.
 * @param Fy y-face (staggered directions: y) state array.
 * @param Fz z-face (staggered directions: z) state array.
 * @param Ex x-edge (staggered directions: yz) state array.
 * @param Ey y-edge (staggered directions: xz) state array.
 * @param Ez z-edge (staggered directions: xy) state array.
 * @return void 
 * @details Note that this operator initializes the \b densitized version of the magnetic field, i.e. \b \sqrt{\gamma} \b B^i
 *          To recover the magnetic field, external metric data needs to be provided at a separate level in the code, at the locations of face centres
 * @remark this routine is second order accurate and could potentially benefit from a 4-th order implementation
 *         nonetheless, the convergence order is dictated first and foremost by the implementation of the evolution of the vector potential 
 * @warning when using this routine in any loop, note that face-staggered field components will also 
 *          be initialized in the non-staggered direction on one ghost point, in each non-staggered direction respectively
 *          e.g. loop from i=ngz up to nx+1+ngz will access
 *          By(ngz+nx,j,k), which is one point into the ghost zone 
 *          Therefore, making sensible use of these points is conditional on a previous and meaningful ghost-zone exchange of A_i
 */
template< typename FaceViewT,
          typename EdgeViewT>
static void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
apply(
      VEC(int const& i, int const& j, int const& k),
      VEC(double const& dx, double const& dy, double const& dz),
      VEC(FaceViewT& Fx, FaceViewT& Fy, FaceViewT& Fz),
      VEC(EdgeViewT const& Ex, EdgeViewT const& Ey, EdgeViewT const& Ez)
     ) 
{
    // for \sqrt{\gamma} B^x = d_y A_z - d_z A_y
    Fx(VEC(i,j,k)) = (1./dy) * (Ez(VEC(i,j+1,k)) - Ez(VEC(i,j,k))) 
                   - (1./dz) * (Ey(VEC(i,j,k+1)) - Ey(VEC(i,j,k)));

    // for \sqrt{\gamma} B^y = d_z A_x - d_x A_z
    Fy(VEC(i,j,k)) = (1./dz) * (Ex(VEC(i,j,k+1)) - Ex(VEC(i,j,k))) 
                   - (1./dx) * (Ez(VEC(i+1,j,k)) - Ez(VEC(i,j,k)));

    // for \sqrt{\gamma} B^z = d_x A_y - d_y A_x
    Fz(VEC(i,j,k)) = (1./dx) * (Ey(VEC(i+1,j,k)) - Ey(VEC(i,j,k))) 
                   - (1./dy) * (Ex(VEC(i,j+1,k)) - Ex(VEC(i,j,k)));

}
#endif 

} ; 




}

#endif /* GRACE_UTILS_CURL_OPERATORS_HH */