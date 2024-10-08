/**
 * @file constexpr.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-04-18
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

#ifndef GRACE_UTILS_CONSTEXPR_HH 
#define GRACE_UTILS_CONSTEXPR_HH

#include <cstdlib>

#include <grace/utils/inline.h>

namespace utils {

namespace detail {

template< size_t Imax
        , size_t I = 0 >
struct ForwardLoop_impl_t {
    template<typename F, typename ... ArgsT >
    constexpr GRACE_ALWAYS_INLINE void operator()(F&& _func, ArgsT&& ... args) {
      _func(I,args...) ;
      if constexpr ( I == Imax -1 ) {
        return ;
      } else {
        ForwardLoop_impl_t<Imax,I+1>{} (_func, args...) ;
      }
    }
} ; 
}

template< size_t Imax
        , size_t I
        , typename F 
        , typename ... ArgsT >
static constexpr GRACE_ALWAYS_INLINE void ForwardLoop(F&& _func, ArgsT&& ...args ) {
    detail::ForwardLoop_impl_t<Imax,I>{}(_func,args...) ; 
}


}

#endif /* GRACE_UTILS_CONSTEPXR_HH */