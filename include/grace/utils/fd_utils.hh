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
#ifndef GRACE_UTILS_NUMERICS_FD_DER_HH
#define GRACE_UTILS_NUMERICS_FD_DER_HH

#include <grace_config.h>
#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/data_structures/variable_properties.hh>

#include <Kokkos_Core.hpp>

namespace grace {

namespace detail {

enum stencil_side_t {
  CENTRAL_STENCIL,
  RIGHT_SIDED_STENCIL,
  LEFT_SIDED_STENCIL,
  N_STENCILS 
} ; 

#define VAR var(VEC(i,j,k))
#define VARP(n) var(VEC(i+n*utils::delta(idir,0), j+n*utils::delta(idir,1), k+n*utils::delta(idir,2)))
#define VARM(n) var(VEC(i-n*utils::delta(idir,0), j-n*utils::delta(idir,1), k-n*utils::delta(idir,2)))

template< size_t order, size_t der_order, stencil_side_t side, size_t bnd_distance=0>
struct stencil {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) 
  {
    static_assert(false, "Stencil is not defined.") ; 
    return 0;
  };
} ;

template<>
struct stencil<2,1,CENTRAL_STENCIL> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return 0.5 * ( VARP(1) - VARM(1) ) ;
  }
} ;

template<>
struct stencil<2,2,CENTRAL_STENCIL> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return - 2.0 * VAR 
           +       VARP(1)
           +       VARM(1) ;
  }
} ;

template<>
struct stencil<2,1,RIGHT_SIDED_STENCIL,1> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return - 1.5 * VAR 
           + 2.0 * VARP(1)
           - 0.5 * VARP(2) ;
  }
} ;

template<>
struct stencil<2,1,RIGHT_SIDED_STENCIL,2> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ; 

template<>
struct stencil<2,1,RIGHT_SIDED_STENCIL,3> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ; 


template<>
struct stencil<2,2,RIGHT_SIDED_STENCIL,1> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return  2.0 * VAR 
          - 5.0 * VARP(1)
          + 4.0 * VARP(2)
          -       VARP(3);
  }
} ;

template<>
struct stencil<2,2,RIGHT_SIDED_STENCIL,2> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ; 

template<>
struct stencil<2,2,RIGHT_SIDED_STENCIL,3> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ;

template<>
struct stencil<2,1,LEFT_SIDED_STENCIL,1> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return  1.5 * VAR
          - 2.0 * VARM(1)
          + 0.5 * VARM(2) ;
  }
} ;

template<>
struct stencil<2,1,LEFT_SIDED_STENCIL,2> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ; 

template<>
struct stencil<2,1,LEFT_SIDED_STENCIL,3> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ;

template<>
struct stencil<2,2,LEFT_SIDED_STENCIL,1> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return  2.0 * VAR
          - 5.0 * VARM(1)
          + 4.0 * VARM(2)
          -       VARM(3);
  }
} ;

template<>
struct stencil<2,2,LEFT_SIDED_STENCIL,2> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ; 

template<>
struct stencil<2,2,LEFT_SIDED_STENCIL,3> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ;


template<>
struct stencil<4,1,CENTRAL_STENCIL> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return 1./12. * VARM(2)
         - 2./3.  * VARM(1)
         + 2./3.  * VARP(1)
         - 1./12. * VARP(2) ;
  }
} ; 

template<>
struct stencil<4,1,RIGHT_SIDED_STENCIL,1> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return - 5./6.  * VAR 
           - 0.25   * VARM(1)
           + 3./2.  * VARP(1)
           - 0.5    * VARP(2)
           + 1./12. * VARP(3) ; 
  }
} ; 

template<>
struct stencil<4,1,RIGHT_SIDED_STENCIL,2> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return - 25./12.  * VAR 
           + 4.       * VARP(1)
           - 3.       * VARP(2)
           + 4./3.    * VARP(3)
           - 0.25     * VARP(4) ; 
  }
} ; 

template<>
struct stencil<4,1,RIGHT_SIDED_STENCIL,3> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ;

template<>
struct stencil<4,1,LEFT_SIDED_STENCIL,1> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return   5./6.  * VAR 
           + 0.25   * VARP(1)
           - 3./2.  * VARM(1)
           + 0.5    * VARM(2)
           - 1./12. * VARM(3) ; 
  }
} ; 

template<>
struct stencil<4,1,LEFT_SIDED_STENCIL,2> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return   25./12.  * VAR 
           - 4.       * VARM(1)
           + 3.       * VARM(2)
           - 4./3.    * VARM(3)
           + 0.25     * VARM(4) ; 
  }
} ; 

template<>
struct stencil<4,1,LEFT_SIDED_STENCIL,3> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ;

template<>
struct stencil<4,2,CENTRAL_STENCIL> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return - 1./12.  * VARM(2)
           + 4./3.   * VARM(1)
           - 5./2.   * VAR 
           + 4./3.   * VARP(1)
           - 1./12.  * VARP(2) ;
  }
} ; 

template<>
struct stencil<4,2,RIGHT_SIDED_STENCIL,1> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return   5./6.  * VARM(1)
           - 5./4.  * VAR 
           - 1./3.  * VARP(1)
           + 7./6.  * VARP(2)
           - 0.5    * VARP(3)
           + 1./12. * VARP(4);
  }
} ; 

template<>
struct stencil<4,2,RIGHT_SIDED_STENCIL,2> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return   15./4.  * VAR 
           - 77./6.  * VARP(1)
           + 107./6. * VARP(2)
           - 13.     * VARP(3)
           + 61./12. * VARP(4)
           - 5./6.   * VARP(5) ;
  }
} ; 

template<>
struct stencil<4,2,RIGHT_SIDED_STENCIL,3> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ;

template<>
struct stencil<4,2,LEFT_SIDED_STENCIL,1> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return   1./12. * VARM(4)
           - 0.5    * VARM(3)
           + 7./6.  * VARM(2)
           - 1./3.  * VARM(1)
           - 5./4.  * VAR 
           + 5./6.  * VARP(1) ;
  }
} ; 

template<>
struct stencil<4,2,LEFT_SIDED_STENCIL,2> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return - 5./6.   * VARM(5) 
           + 61./12. * VARM(4)
           - 13.     * VARM(3)
           + 107./6. * VARM(2)
           - 77./6.  * VARM(1)
           + 15./4.  * VAR ;
  }
} ; 

template<>
struct stencil<4,2,LEFT_SIDED_STENCIL,3> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return std::numeric_limits<double>::quiet_NaN() ; 
  }
} ;

/**********************************************/
/*    Kreiss-Olinger dissipation operators    */
/**********************************************/
template<>
struct stencil<2,4,CENTRAL_STENCIL> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return      VARM(2)
         - 4. * VARM(1)
         + 6. * VAR
         - 4. * VARP(1)
         +      VARP(2) ;
  }
} ;

template<>
struct stencil<2,6,CENTRAL_STENCIL> {
  template< size_t idir, typename F >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  apply(F && var, VEC(int i, int j, int k)) {
    return           VARM(3)
           - 6.    * VARM(2)
           + 15.   * VARM(1)
           - 20.   * VAR 
           + 15.   * VARP(1)
           - 6.    * VARP(2)
           +         VARP(3) ;
  }
} ;

template< size_t ndirs ,  size_t der_order, size_t idir, size_t ... idirs>
struct fd_der_recursive {
    template< size_t order, typename ViewT >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    doit (ViewT const var, VEC(int i, int j, int k)) {
        auto const f = [&] (VEC(int i, int j, int k)) {
            return fd_der_recursive<ndirs-1,der_order,idirs...>::template doit<order>(var, VEC(i,j,k)) ;
        } ;
        return stencil<order,der_order,CENTRAL_STENCIL,0>::template apply<idir>(f, VEC(i,j,k)) ;
    }
} ;

template< size_t der_order, size_t idir >
struct fd_der_recursive<1,der_order, idir> {
    template< size_t order, typename ViewT >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    doit(ViewT const var, VEC(int i, int j, int k)) {
        return stencil<order,der_order,CENTRAL_STENCIL,0>::template apply<idir>(var, VEC(i,j,k)) ;
    }
} ; 

template< size_t der_order, size_t idir >
struct fd_der_recursive_upwind {
  template< size_t order, typename ViewT>
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
  doit( ViewT const var, VEC(int i, int j, int k), std::array<double,GRACE_NSPACEDIM> const& vec) {
    return ( vec[idir] > 0 ) * stencil<order,der_order,LEFT_SIDED_STENCIL,1>::template apply<idir>(var,VEC(i,j,k))
         + ( vec[idir] < 0 ) * stencil<order,der_order,RIGHT_SIDED_STENCIL,1>::template apply<idir>(var,VEC(i,j,k))
         + ( vec[idir] == 0 ) * stencil<order,der_order,CENTRAL_STENCIL,0>::template apply<idir>(var, VEC(i,j,k))  ;
  }
} ; 

template< size_t ndirs , size_t der_order, size_t idir, size_t ... idirs>
struct fd_der_bnd_check_recursive {
  template< size_t order, typename ViewT >
  static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
  doit (ViewT const var, VEC(int i, int j, int k), VEC(int nx, int ny, int nz), int ngz) {
      
      std::array<int,GRACE_NSPACEDIM> const imax {
        VEC( nx + 2 * ngz - 1, ny + 2 * ngz - 1, nz + 2*ngz - 1) 
      } ; 
      std::array<int,GRACE_NSPACEDIM> const ijk { VEC(i,j,k) } ; 

      auto const f = [&] (VEC(int i, int j, int k)) {
          return fd_der_bnd_check_recursive<ndirs-1,der_order,idirs...>::template doit<order>(var, VEC(i,j,k), VEC(nx,ny,nz), ngz) ;
      } ;
      if ( ijk[idir] >= order / 2 and (imax[idir]-ijk[idir]) >= order / 2 ) { 
        return stencil<order,der_order,CENTRAL_STENCIL,0>::template apply<idir>(f, VEC(i,j,k)) ;
      } else if ( (ijk[idir] >= order / 2 - 1) and ((order / 2 - 1) >= 0) ) {
        return stencil<order,der_order,RIGHT_SIDED_STENCIL,1>::template apply<idir>(f, VEC(i,j,k)) ;
      } else if ( (ijk[idir] >= order / 2 - 2) and ((order / 2 - 2) >= 0) ) {
        return stencil<order,der_order,RIGHT_SIDED_STENCIL,2>::template apply<idir>(f, VEC(i,j,k)) ;
      } else if ( (ijk[idir] >= order / 2 - 3) and ((order / 2 - 3) >= 0) ) {
        return stencil<order,der_order,RIGHT_SIDED_STENCIL,3>::template apply<idir>(f, VEC(i,j,k)) ;
      } else if ( ((imax[idir]-ijk[idir]) >= order / 2 - 1) and ((order / 2 - 1) >= 0) ) {
        return stencil<order,der_order,LEFT_SIDED_STENCIL,1>::template apply<idir>(f, VEC(i,j,k)) ;
      } else if ( ((imax[idir]-ijk[idir]) >= order / 2 - 2) and ((order / 2 - 2) >= 0) ) { 
        return stencil<order,der_order,LEFT_SIDED_STENCIL,2>::template apply<idir>(f, VEC(i,j,k)) ;
      } else if ( ((imax[idir]-ijk[idir]) >= order / 2 - 3) and ((order / 2 - 3) >= 0) ) { 
        return stencil<order,der_order,LEFT_SIDED_STENCIL,3>::template apply<idir>(f, VEC(i,j,k)) ;
      } else {
        printf("fd_der_bnd_check_recursive: no valid stencil at (%d,%d,%d) in dir %zu\n", i, j, k, idir);
        return std::numeric_limits<double>::quiet_NaN() ; 
      }
      
  }
} ; 

template< size_t der_order, size_t idir >
struct fd_der_bnd_check_recursive<1,der_order, idir> {
    template< size_t order, typename ViewT >
    static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    doit (ViewT const var, VEC(int i, int j, int k), VEC(int nx, int ny, int nz), int ngz) {
        std::array<int,GRACE_NSPACEDIM> const imax {
          VEC( nx + 2 * ngz - 1, ny + 2 * ngz - 1, nz + 2*ngz - 1) 
        } ; 
        std::array<int,GRACE_NSPACEDIM> const ijk { VEC(i,j,k) } ;
        if ( ijk[idir] >= order / 2 and (imax[idir]-ijk[idir]) >= order / 2 ) { 
          return stencil<order,der_order,CENTRAL_STENCIL,0>::template apply<idir>(var, VEC(i,j,k)) ;
        } else if ( (ijk[idir] >= order / 2 - 1) and ((order / 2 - 1) >= 0) ) {
          return stencil<order,der_order,RIGHT_SIDED_STENCIL,1>::template apply<idir>(var, VEC(i,j,k)) ;
        } else if ( (ijk[idir] >= order / 2 - 2) and ((order / 2 - 2) >= 0) ) {
          return stencil<order,der_order,RIGHT_SIDED_STENCIL,2>::template apply<idir>(var, VEC(i,j,k)) ;
        } else if ( (ijk[idir] >= order / 2 - 3) and ((order / 2 - 3) >= 0) ) {
          return stencil<order,der_order,RIGHT_SIDED_STENCIL,3>::template apply<idir>(var, VEC(i,j,k)) ;
        } else if ( ((imax[idir]-ijk[idir]) >= order / 2 - 1) and ((order / 2 - 1) >= 0) ) {
          return stencil<order,der_order,LEFT_SIDED_STENCIL,1>::template apply<idir>(var, VEC(i,j,k)) ;
        } else if ( ((imax[idir]-ijk[idir]) >= order / 2 - 2) and ((order / 2 - 2) >= 0) ) { 
          return stencil<order,der_order,LEFT_SIDED_STENCIL,2>::template apply<idir>(var, VEC(i,j,k)) ;
        } else if ( ((imax[idir]-ijk[idir]) >= order / 2 - 3) and ((order / 2 - 3) >= 0) ) { 
          return stencil<order,der_order,LEFT_SIDED_STENCIL,3>::template apply<idir>(var, VEC(i,j,k)) ;
        } else {
          return std::numeric_limits<double>::quiet_NaN() ; 
        }
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
fd_der( grace::var_array_t const u 
      , int const ivar 
      , VEC( int const i, int const j, int const k)
      , int const q ) 
{
    auto var = Kokkos::subview(u,VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), ivar, q) ; 
    return detail::fd_der_recursive<sizeof...(dirs),1,dirs...>::template doit<der_order>(var,VEC(i,j,k)) ; 
}

/**
 * @brief Compute the finite difference approximation of 
 *        second order derivatives of a grid variable inside a View.
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
fd_second_der( grace::var_array_t const u 
      , int const ivar 
      , VEC( int const i, int const j, int const k)
      , int const q ) 
{
    auto var = Kokkos::subview(u,VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), ivar, q) ; 
    return detail::fd_der_recursive<sizeof...(dirs),2,dirs...>::template doit<der_order>(var,VEC(i,j,k)) ; 
}

/**
 * @brief Compute the finite difference approximation of 
 *        derivative of a grid variable inside a View, upwinded
 *        w.r.t. the respective component of the vector vec.
 * 
 * @tparam der_order Order of accuracy of derivative stencil.
 * @tparam idir      Direction of derivative to be applied.
 * @param u          View containing the variable on the grid.
 * @param i          x-cell index of the point where the derivative 
 *                   is needed.
 * @param j          y-cell index of the point where the derivative 
 *                   is needed.
 * @param k          z-cell index of the point where the derivative 
 *                   is needed.
 * @param ivar       Variable index within the View. 
 * @param q          Quadrant index.
 * @param vec        Vector w.r.t. upwinding is applied.
 * @return double Finite difference approximation to the derivative.
 */
template< size_t der_order
        , size_t ... dirs >
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
fd_der_bnd_check( grace::var_array_t const u 
                , int const ivar 
                , VEC( int const i, int const j, int const k)
                , int const q 
                , VEC(int nx, int ny, int nz), int ngz) 
{
    auto var = Kokkos::subview(u,VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), ivar, q) ; 
    return detail::fd_der_bnd_check_recursive<sizeof...(dirs),1,dirs...>::template doit<der_order>(var,VEC(i,j,k),VEC(nx,ny,nz),ngz) ; 
}

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
 , size_t idir >
double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
fd_der_upwind( grace::var_array_t const u 
, int const ivar 
, VEC( int const i, int const j, int const k)
, int const q 
, std::array<double, GRACE_NSPACEDIM> const& vec) 
{
auto var = Kokkos::subview(u,VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), ivar, q) ; 
return detail::fd_der_recursive_upwind<1,idir>::template doit<der_order>(var,VEC(i,j,k), vec) ; 
}

}

#undef VAR 
#undef VARP
#undef VARM 

#endif /* GRACE_UTILS_NUMERICS_FD_DER_HH */