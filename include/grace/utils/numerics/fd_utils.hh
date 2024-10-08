/**
 * @file fd_utils.hh
 * @author  Carlo Musolino
 * @brief 
 * @date 2024-09-03
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

#include <grace_config.h>
#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>

#include <grace/data_structures/variable_properties.hh>

namespace grace {

namespace detail {

template< size_t order >
struct stencil {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) ;
} ;

template<>
struct stencil<2> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return 0.5 * ( var(VEC(i+utils::delta(idir,0), j+utils::delta(idir,1), k+utils::delta(idir,2)))
                   - var(VEC(i-utils::delta(idir,0), j-utils::delta(idir,1), k-utils::delta(idir,2)))) ;
  }
} ;

template< size_t ndirs ,  size_t idir, size_t ... idirs>
struct fd_der_recursive {

    template< size_t order >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    doit (grace::var_array_t<GRACE_NSPACEDIM> const var, VEC(int i, int j, int k), int ivar, int q) {

        auto const f = [&] (VEC(int i, int j, int k)) {
            return fd_der_recursive<ndirs-1,idirs...>::template doit<order>(var, VEC(i,j,k), ivar, q) ;
        } ;
        return return stencil<order>::template apply<idir>(f, VEC(i,j,k)) ;
    }
} ;

template< size_t idir>
struct fd_der_recursive<1,idir> {
    template< size_t order >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    doit(grace::var_array_t<GRACE_NSPACEDIM> const var, VEC(int i, int j, int k), int ivar, int q) {
        return stencil<order>::template apply<idir>(var, VEC(i,j,k), ivar, q) ;
    }
} ; 

} // namespace detail 

/**
 * @brief Compute the finite difference approximation of 
 *        derivatives of a grid variable inside a View.
 * 
 * @tparam der_order Order of accuracy of derivative stencil.
 * @tparam dirs      Direction(s) of derivatives to be applied.
 * @param u          View containing the variable on the grid.
 * @param i          x-cell index of the point where the derivative 
 *                   is needed.
 * @param j          y-cell index of the point where the derivative 
 *                   is needed.
 * @param k          z-cell index of the point where the derivative 
 *                   is needed.
 * @param ivar       Variable index within the View. 
 * @param q          Quadrant index.
 * @return double Finite difference approximation to the derivatives.
 */
template< size_t der_order
        , size_t ... dirs >
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
fd_der( grace::var_array_t<GRACE_NSPACEDIM> const u 
      , VEC( int const i, int const j, int const k)
      , int const ivar 
      , int const q ) 
{
    return fd_der_recursive<sizeof...(dirs),dirs...>::template doit<der_order>(u,VEC(i,j,k),ivar,q) ; 
}

}