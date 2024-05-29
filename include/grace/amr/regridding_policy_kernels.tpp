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
 * @brief Second derivative FLASH-like kernel
 *        for regridding.
 * \ingroup amr
 * 
 * @tparam ViewT Type of variable view.
 */
template< typename ViewT > 
struct flash_second_deriv_criterion {

    ViewT u ; //!< Variable on which the criterion is evaluated.
    static constexpr double tiny = 1e-99; 

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator()  ( VEC( int const & i 
                     , int const & j
                     , int const & k ) 
                , int const& q
                , double const& eps ) const 
    {
        double VEC(
            Ex{ Kokkos::fabs( u(VEC(i+1,j,k),q) - 2*u(VEC(i,j,k),q) + u(VEC(i-1,j,k),q) )
                / (   Kokkos::fabs(u(VEC(i+1,j,k),q) - u(VEC(i,j,k),q)) 
                    + Kokkos::fabs(u(VEC(i,j,k),q) - u(VEC(i-1,j,k),q))
                    + eps*Kokkos::fabs(u(VEC(i+1,j,k),q) - 2*u(VEC(i,j,k),q) + u(VEC(i-1,j,k),q)) + tiny
                )}
        ,   Ey{ Kokkos::fabs( u(VEC(i,j+1,k),q) - 2*u(VEC(i,j,k),q) + u(VEC(i,j-1,k),q) )
                / (   Kokkos::fabs(u(VEC(i,j+1,k),q) - u(VEC(i,j,k),q)) 
                    + Kokkos::fabs(u(VEC(i,j,k),q) - u(VEC(i,j-1,k),q))
                    + eps*Kokkos::fabs(u(VEC(i,j+1,k),q) - 2*u(VEC(i,j,k),q) + u(VEC(i,j-1,k),q)) + tiny
                )}
        ,   Ez{ Kokkos::fabs( u(VEC(i,j,k+1),q) - 2*u(VEC(i,j,k),q) + u(VEC(i,j,k-1),q) )
                / (   Kokkos::fabs(u(VEC(i,j,k+1),q) - u(VEC(i,j,k),q)) 
                    + Kokkos::fabs(u(VEC(i,j,k),q) - u(VEC(i,j,k-1),q))
                    + eps*Kokkos::fabs(u(VEC(i,j,k+1),q) - 2*u(VEC(i,j,k),q) + u(VEC(i,j,k-1),q)) + tiny
                )}
        ) ; 
        double maxerr = Kokkos::fmax( Ex, Ey ) ; 
        #ifdef GRACE_3D 
        maxerr = Kokkos::fmax(maxerr, Ez) ; 
        #endif 
        return maxerr ; 
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