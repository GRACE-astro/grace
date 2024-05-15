/**
 * @file reconstruction.h
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-13
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

#ifndef THUNDER_UTILS_RECONSTRUCTION_HH 
#define THUNDER_UTILS_RECONSTRUCTION_HH

#include <thunder_config.h>
#include <thunder/utils/device.h>
#include <thunder/utils/inline.h>
#include <thunder/utils/limiters.hh>
#include <thunder/utils/matrix_helpers.tpp>

#include <thunder/data_structures/variable_properties.hh>

namespace thunder {

template< typename limiter_t >
struct slope_limited_reconstructor_t  
{
    template< typename ViewT >
    void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
    operator() (
          ViewT& u 
        , VEC( int const i
             , int const j 
             , int const k)
        , double& uL
        , double& uR 
        , int8_t idir )
    {
        limiter_t limiter{} ; 

        int const ip  = i + utils::delta(0,idir)   ; 
        int const im  = i - utils::delta(0,idir)   ; 
        int const imm = i - 2*utils::delta(0,idir) ; 

        int const jp  = j + utils::delta(1,idir)   ; 
        int const jm  = j - utils::delta(1,idir)   ; 
        int const jmm = j - 2*utils::delta(1,idir) ;
        
        #ifdef THUNDER_3D 
        int const kp  = k + utils::delta(2,idir)   ; 
        int const km  = k - utils::delta(2,idir)   ; 
        int const kmm = k - 2*utils::delta(2,idir) ;
        #endif 

        double slopeL = u(VEC(i,j,k)) - u(VEC(im,jm,km)) ; 
        double slopeR = u(VEC(ip,jp,kp)) - u(VEC(i,j,k)) ; 

        uR = u(VEC(i,j,k)) - 0.5 * limiter(slopeL,slopeR) ; 

        slopeL = u(VEC(im,jm,km)) - u(VEC(imm,jmm,kmm)) ; 
        slopeR = u(VEC(i,j,k)) - u(VEC(im,jm,km))       ; 

        uL = u(VEC(im,jm,km)) + 0.5 * limiter(slopeL,slopeR) ; 
    }
} ;
 
}
#endif /* THUNDER_UTILS_RECONSTRUCTION_HH */