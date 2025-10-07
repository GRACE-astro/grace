/**
 * @file fd_evolution_system.hh
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

#ifndef GRACE_EVOLVE_FD_EVOL_SYSTEM_HH
#define GRACE_EVOLVE_FD_EVOL_SYSTEM_HH


#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_properties.hh>

namespace grace {

template < typename EvolSystem_t > 
struct fd_evolution_system_t {

    fd_evolution_system_t( grace::var_array_t state_ 
                         , grace::var_array_t aux_ )
        : _state(state_), _aux(aux_)
    {} 


    template< size_t der_order >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_update( int const q 
                  , VEC( int const i 
                       , int const j 
                       , int const k)
                  , grace::scalar_array_t<GRACE_NSPACEDIM> const idx 
                  , grace::var_array_t const state_new 
                  , double const dt 
                  , double const dtfact ) const 
    {
        return static_cast<EvolSystem_t const*>(this)->compute_update_impl(q,VEC(i,j,k),idx,state_new,dt,dtfact) ;
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( auxiliaries_computation_kernel_t _tag
               , VEC( const int i 
               ,      const int j 
               ,      const int k)
               , int64_t q ) const 
    {
        static_cast<EvolSystem_t const *>(this)->compute_auxiliaries(VEC(i,j,k),q) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( eigenspeed_kernel_t _tag
               , VEC( const int i 
               ,      const int j 
               ,      const int k)
               , int64_t q ) const 
    {
        return static_cast<EvolSystem_t const *>(this)->compute_max_eigenspeed(VEC(i,j,k),q) ;
    } ; 

 protected:
    grace::var_array_t _state, _aux ; 

} ; 
} // namespace grace 
#endif 