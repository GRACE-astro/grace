/**
 * @file kreiss_olinger.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-12-02
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

#ifndef GRACE_UTILS_KREISS_OLINGER_HH
#define GRACE_UTILS_KREISS_OLINGER_HH

#include <grace_config.h>

#include <grace/utils/numerics/math.hh>
#include <grace/utils/numerics/fd_utils.hh>

#include <Kokkos_Core.hpp>

namespace grace {

/**
 * @brief Compute Kreiss-Olinger dissipation of a given order on a variable.
 * 
 * @tparam truncation_order Truncation order of derivative calculation
 * @tparam deriv_order Order of the derivative
 * @tparam view_t Type of view containing the data
 * @param ivar Variable index
 * @param q Quadrant index
 * @param u View containing the data 
 * @param dx Spacing of the grid
 * @param epsdiss Dissipation amplitude
 * 
 * @return The Kreiss-Olinger dissipation term
 */
template< size_t truncation_order
        , size_t deriv_order 
        , typename view_t >
double GRACE_HOST_DEVICE
apply_kreiss_olinger_dissipation(
    VEC(int i, int j, int k), int ivar, int64_t q,
    view_t u, std::array<double,3> idx, double const epsdiss
)
{
    using namespace grace ; 

    auto var  = 
        Kokkos::subview(u,VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL()), ivar, q) ; 
    /*
    auto dvardx = 
        detail::fd_der_recursive<1,deriv_order+1,0>::template doit<truncation_order>(var,VEC(i,j,k)) * idx[0]; 
    auto dvardy = 
        detail::fd_der_recursive<1,deriv_order+1,1>::template doit<truncation_order>(var,VEC(i,j,k)) * idx[1];  
    auto dvardz = 
        detail::fd_der_recursive<1,deriv_order+1,2>::template doit<truncation_order>(var,VEC(i,j,k)) * idx[2];  
    */

    return epsdiss / 64 * (
              ( var(VEC(i-3,j,k)) - 6*var(VEC(i-2,j,k)) + 15*var(VEC(i-1,j,k)) - 20*var(VEC(i,j,k)) + 15*var(VEC(i+1,j,k)) - 6*var(VEC(i+2,j,k)) + var(VEC(i+3,j,k)) ) * idx[0]
            //+ ( var(VEC(i,j-3,k)) - 6*var(VEC(i,j-2,k)) + 15*var(VEC(i,j-1,k)) - 20*var(VEC(i,j,k)) + 15*var(VEC(i,j+1,k)) - 6*var(VEC(i,j+2,k)) + var(VEC(i,j+3,k)) ) * idx[1]
            //+ ( var(VEC(i,j,k-3)) - 6*var(VEC(i,j,k-2)) + 15*var(VEC(i,j,k-1)) - 20*var(VEC(i,j,k)) + 15*var(VEC(i,j,k+1)) - 6*var(VEC(i,j,k+2)) + var(VEC(i,j,k+3)) ) * idx[2]
        ); 
}
#if 0
/**
 * @brief 
 * 
 * @tparam truncation_order 
 * @tparam deriv_order 
 * @param u 
 * @param dx 
 * @param epsdiss 
 */
template< size_t truncation_order
        , size_t deriv_order >
void 
apply_kreiss_olinger_dissipation_block(
    grace::var_array_t<GRACE_NSPACEDIM> state, std::pair<int,int> varidx, grace::scalar_array_t dx, double const epsdiss
)
{
    using namespace grace ; 

    auto vars  = 
        Kokkos::subview( state,
                         VEC( Kokkos::ALL(),
                              Kokkos::ALL(),
                              Kokkos::ALL() ), 
                         Kokkos::pair<int,int>{varidx.first,varidx.second},
                         Kokkos::ALL()) ; 
    

}
#endif 
}

#endif /* GRACE_UTILS_KREISS_OLINGER_HH */