/**
 * @file hrsc_evolution_system.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-13
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

#ifndef GRACE_EVOLUTION_HRSC_EVOLUTION_SYSTEM_HH
#define GRACE_EVOLUTION_HRSC_EVOLUTION_SYSTEM_HH

#include <grace_config.h>

#include <grace/utils/grace_utils.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/evolution/evolution_kernel_tags.hh>

namespace grace {


template< typename EvolSystem_t > 
struct hrsc_evolution_system_t {

    hrsc_evolution_system_t( grace::var_array_t<GRACE_NSPACEDIM> state_
                           , grace::var_array_t<GRACE_NSPACEDIM> aux_  )
     : _state(state_), _aux(aux_)
    {} 

    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( x_flux_computation_kernel_t _tag
               , thread_team_t& team 
               , VEC( const int i 
               ,      const int j 
               ,      const int k)
               , int ngz
               , grace::flux_array_t const fluxes) const 
    {
        static_cast<EvolSystem_t const *>(this)->compute_x_flux(team,VEC(i,j,k),ngz,fluxes) ;
    }

    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( y_flux_computation_kernel_t _tag
               , thread_team_t& team 
               , VEC( const int i 
               ,      const int j 
               ,      const int k)
               , int ngz
               , grace::flux_array_t const fluxes) const 
    {
        static_cast<EvolSystem_t const *>(this)->compute_y_flux(team,VEC(i,j,k),ngz,fluxes) ; 
    }

    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( z_flux_computation_kernel_t _tag
               , thread_team_t& team 
               , VEC( const int i 
               ,      const int j 
               ,      const int k)
               , int ngz
               , grace::flux_array_t const fluxes) const 
    {
        static_cast<EvolSystem_t const *>(this)->compute_z_flux(team,VEC(i,j,k),ngz,fluxes) ;
    }

    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( sources_computation_kernel_t _tag
               , thread_team_t& team 
               , VEC( const int i 
               ,      const int j 
               ,      const int k)
               , grace::var_array_t<GRACE_NSPACEDIM> const state_new 
               , double const dt 
               , double const dtfact ) const 
    {
        return static_cast<EvolSystem_t const *>(this)->compute_source_terms(team,VEC(i,j,k),state_new,dt,dtfact) ; 
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
    }
    
 protected: 
    grace::var_array_t<GRACE_NSPACEDIM> _state, _aux ; 
} ; 

}

#endif /* GRACE_EVOLUTION_HRSC_EVOLUTION_SYSTEM_HH */