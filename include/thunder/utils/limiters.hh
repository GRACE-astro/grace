/**
 * @file limiters.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-04-09
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

#ifndef THUNDER_UTILS_LIMITERS_HH
#define THUNDER_UTILS_LIMITERS_HH

#include <thunder_config.h>
#include <thunder/utils/device.h>
#include <thunder/utils/inline.h>
#include <thunder/utils/math.hh>
#include <Kokkos_Core.hpp> 

namespace thunder {

struct minmod {
    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    operator() (double const& slopeL, double const& slopeR){
        auto const signL = math::sgn(slopeL) ;
        auto const signR = math::sgn(slopeR) ; 
        return 0.5 * ( signL + signR ) * math::min(Kokkos::fabs(slopeL),Kokkos::fabs(slopeR)) ; 
    }
} ; 

struct MCbeta {
    double const beta = 2. ; 
    double THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    operator() (double const& slopeL, double const& slopeR){
        auto const slopeC = 0.5 * (slopeR + slopeL) ; 
        auto const signR  = math::sgn(slopeR) ;
        auto const signC  = math::sgn(slopeC) ;
        return signR * Kokkos::max(0.,Kokkos::min(Kokkos::min( beta*Kokkos::fabs(slopeR)
                                                 , beta*signR*slopeL),signR*slopeC)) ; 

    }
} ; 

} /* namespace thunder */

#endif /* THUNDER_UTILS_LIMITERS_HH */