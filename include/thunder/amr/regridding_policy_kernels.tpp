/**
 * @file regrid_kernels.tpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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
#ifndef THUNDER_AMR_REGRID_KERNELS_TPP
#define THUNDER_AMR_REGRID_KERNELS_TPP

#include <thunder_config.h> 

#include <Kokkos_Core.hpp>

#include <thunder/utils/thunder_utils.hh>

namespace thunder { namespace amr {

template< typename ViewT > 
struct flash_second_deriv_criterion {

    ViewT u ; 
    static constexpr double tiny = 1e-99; 

    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
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
        #ifdef THUNDER_3D 
        maxerr = Kokkos::fmax(maxerr, Ez) ; 
        #endif 
        return maxerr ; 
    }
} ; 

template< typename ViewT > 
struct simple_threshold_criterion {

    ViewT u ; 

    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    operator()  ( VEC( int const & i 
                     , int const & j
                     , int const & k ) 
                , int const& q ) const 
    {
        return u(VEC(i,j,k),q) ; 
    }
} ; 


}} /* namespace thunder::amr */

#endif 