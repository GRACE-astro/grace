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

    hrsc_evolution_system_t( grace::var_array_t state_
                           , grace::staggered_variable_arrays_t stag_state_
                           , grace::var_array_t aux_  )
     : _state(state_), _stag_state(stag_state_), _aux(aux_)
    {} 

    template< typename riemann_t 
            , typename recon_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_x_flux( int const q  
                  , VEC( const int i 
                  ,      const int j 
                  ,      const int k)
                  , grace::flux_array_t const fluxes
                  , grace::flux_array_t const vbar
                  , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                  , double const dt 
                  , double const dtfact ) const 
    {
        static_cast<EvolSystem_t const *>(this)->template 
            compute_x_flux_impl<riemann_t,recon_t>(q,VEC(i,j,k),fluxes,vbar,dx,dt,dtfact) ;
    }

    template< typename riemann_t 
            , typename recon_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_y_flux( int const q
                  , VEC( const int i 
                  ,      const int j 
                  ,      const int k)
                  , grace::flux_array_t const fluxes
                  , grace::flux_array_t const vbar
                  , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                  , double const dt 
                  , double const dtfact ) const 
    {
        static_cast<EvolSystem_t const *>(this)->template 
            compute_y_flux_impl<riemann_t,recon_t>(q,VEC(i,j,k),fluxes,vbar,dx,dt,dtfact) ; 
    }

    template< typename riemann_t 
            , typename recon_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_z_flux( int const q  
                  , VEC( const int i 
                  ,      const int j 
                  ,      const int k)
                  , grace::flux_array_t const fluxes
                  , grace::flux_array_t const vbar
                  , grace::scalar_array_t<GRACE_NSPACEDIM> const dx
                  , double const dt 
                  , double const dtfact ) const  
    {
        static_cast<EvolSystem_t const *>(this)->template 
            compute_z_flux_impl<riemann_t,recon_t>(q,VEC(i,j,k),fluxes,vbar,dx,dt,dtfact) ; 
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( sources_computation_kernel_t _tag
               , const int q 
               , VEC( const int i 
               ,      const int j 
               ,      const int k)
               , grace::scalar_array_t<GRACE_NSPACEDIM> const idx
               , grace::var_array_t const state_new 
               , double const dt 
               , double const dtfact ) const 
    {
        return static_cast<EvolSystem_t const *>(this)->compute_source_terms(q,VEC(i,j,k),idx,state_new,dt,dtfact) ; 
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( auxiliaries_computation_kernel_t _tag
               , VEC( const int i 
               ,      const int j 
               ,      const int k)
               , int64_t q 
               , grace::coord_array_t<GRACE_NSPACEDIM> pcoords ) const 
    {
        static_cast<EvolSystem_t const *>(this)->compute_auxiliaries(VEC(i,j,k),q,pcoords) ; 
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
    grace::var_array_t _state, _aux ; 
    grace::staggered_variable_arrays_t _stag_state;
} ; 

}

#endif /* GRACE_EVOLUTION_HRSC_EVOLUTION_SYSTEM_HH */