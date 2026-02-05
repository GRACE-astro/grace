/**
 * @file regrid_kernels.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Implementation of policy kernels for regridding.
 * @version 0.1
 * @date 2024-03-19
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
#ifndef GRACE_AMR_REGRID_KERNELS_TPP
#define GRACE_AMR_REGRID_KERNELS_TPP

#include <grace_config.h> 

#include <Kokkos_Core.hpp>

#include <grace/utils/grace_utils.hh>

namespace grace { namespace amr {
/**
 * @brief Second derivative Löhner-like criterion
 *        for regridding.
 * \ingroup amr
 * 
 * @tparam ViewT Type of variable view.
 */
template< typename ViewT > 
struct flash_second_deriv_criterion {

    ViewT u ; //!< Variable on which the criterion is evaluated.
    static constexpr double tiny = 1e-99; 
    double eps ; 
    flash_second_deriv_criterion(
        ViewT _u, double _eps
    ) : u(_u), eps(_eps) {}

    /**
     * @brief Evaluates the regridding criterion at a cell.
     * 
     * @param q Quadrant index.
     * @param eps Amplitude of denominator correction.
     * @return double The error estimate according to the Löhner criterion.
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator()  ( VEC( int const & i 
                     , int const & j
                     , int const & k ) 
                , int const& q) const 
    {   
        using Kokkos::fabs ; 
        double const num = EXPR(
              math::int_pow<2>(u(VEC(i+1,j,k),q) - 2*u(VEC(i,j,k),q) + u(VEC(i-1,j,k),q)),
            + math::int_pow<2>(u(VEC(i,j+1,k),q) - 2*u(VEC(i,j,k),q) + u(VEC(i,j-1,k),q)),
            + math::int_pow<2>(u(VEC(i,j,k+1),q) - 2*u(VEC(i,j,k),q) + u(VEC(i,j,k-1),q))
        ) ; 
        double const denom = EXPR(
              fabs(u(VEC(i+1,j,k),q) - u(VEC(i,j,k),q)) 
            + fabs(u(VEC(i,j,k),q) - u(VEC(i-1,j,k),q))
            + eps * (fabs(u(VEC(i+1,j,k),q)) - 2.*fabs(u(VEC(i,j,k),q)) + fabs(u(VEC(i-1,j,k),q))),
            + fabs(u(VEC(i,j+1,k),q) - u(VEC(i,j,k),q)) 
            + fabs(u(VEC(i,j,k),q) - u(VEC(i,j-1,k),q))
            + eps * (fabs(u(VEC(i,j+1,k),q)) - 2.*fabs(u(VEC(i,j,k),q)) + fabs(u(VEC(i,j-1,k),q))),
            + fabs(u(VEC(i,j,k+1),q) - u(VEC(i,j,k),q)) 
            + fabs(u(VEC(i,j,k),q) - u(VEC(i,j,k-1),q))
            + eps * (fabs(u(VEC(i,j,k+1),q)) - 2.*fabs(u(VEC(i,j,k),q)) + fabs(u(VEC(i,j,k-1),q)))
            + tiny
        ) ; 
        return num / denom ; 
    }
} ; 
/**
 * @brief Gradient based kernel
 *        for regridding.
 * \ingroup amr
 * 
 * @tparam ViewT Type of variable view.
 */
template< typename ViewT > 
struct gradient_criterion {

    ViewT u ; //!< Variable on which the criterion is evaluated.
    static constexpr double tiny = 1e-99; 
    /**
     * @brief Evaluate the regridding criterion.
     * 
     * @param q Quadrant index.
     * @return double Error estimate.
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator()  ( VEC( int const & i 
                     , int const & j
                     , int const & k ) 
                , int const& q ) const 
    {
        double grad[] = 
            {
                VEC( u(VEC(i+1,j,k),q)-u(VEC(i-1,j,k),q) 
                   , u(VEC(i,j+1,k),q)-u(VEC(i,j-1,k),q) 
                   , u(VEC(i+1,j,k+1),q)-u(VEC(i,j,k-1),q))
            } ; 
        return 0.5*Kokkos::sqrt(EXPR(grad[0]*grad[0],+grad[1]*grad[1],+grad[2]*grad[2])) / (u(VEC(i,j,k),q)+tiny) ; 
    }
} ;
/**
 * @brief Shear based kernel
 *        for regridding.
 * \ingroup amr
 * 
 * @tparam ViewT Type of variable view.
 */
template< typename ViewT > 
struct shear_criterion {

    ViewT VEC(vx,vy,vz) ; //!< Velocity components.
    static constexpr double tiny = 1e-99; 
    /**
     * @brief Evaluate the regridding criterion.
     * 
     * @param q Quadrant index.
     * @return double The error estimate.
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator()  ( VEC( int const & i 
                     , int const & j
                     , int const & k ) 
                , int const& q ) const 
    {
        using Kokkos::fabs ; 
        return math::max( 
              0.5*fabs(vy(VEC(i+1,j,k),q) - vy(VEC(i-1,j,k),q))
            , 0.5*fabs(vx(VEC(i,j+1,k),q) - vx(VEC(i,j-1,k),q))
            #ifdef GRACE_3D 
            , 0.5*fabs(vy(VEC(i,j,k+1),q) - vy(VEC(i,j,k-1),q))
            , 0.5*fabs(vx(VEC(i,j,k+1),q) - vx(VEC(i,j,k-1),q))
            , 0.5*fabs(vz(VEC(i+1,j,k),q) - vz(VEC(i-1,j,k),q))
            , 0.5*fabs(vz(VEC(i,j+1,k),q) - vz(VEC(i,j-1,k),q))
            #endif 
        ) ; 
    }
} ;
/**
* @brief Regrid criterion based on simple threshold on grid variable.
* \ingroup amr 
* @tparam ViewT The view type for the grid variable.
*/
template< typename ViewT > 
struct simple_threshold_criterion {

    ViewT u ; //!< Variable where the criterion is evaluated

    /**
    * @brief Evaluate regrid criterion.
    * This function returns the variable at the chosen point.
    */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator()  ( VEC( int const & i 
                     , int const & j
                     , int const & k ) 
                , int const& q ) const 
    {
        return u(VEC(i,j,k),q) ; 
    }
} ; 


}} /* namespace grace::amr */

#endif 