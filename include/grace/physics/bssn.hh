/**
 * @file bssn.hh
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

#ifndef GRACE_PHYSICS_BSSN
#define GRACE_PHYSICS_BSSN

#include <grace_config.h> 

#include <grace/utils/grace_utils.hh>

#include <grace/data_structures/variable_properties.hh>

#include <grace/evolution/fd_evolution_system.hh>

#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/bssn_helpers.hh>

#include <Kokkos_Core.hpp>

#include <array>


namespace grace {

template< size_t der_order >
grace::bssn_state_t GRACE_HOST_DEVICE 
compute_bssn_rhs( VEC(int i, int j, int k), int q
                , grace::var_array_t<GRACE_NSPACEDIM> const state
                , std::array<std::array<double,4>,4> const& Tmunu
                , std::array<double,GRACE_NSPACEDIM> const& idx);


struct bssn_system_t 
    : public fd_evolution_system_t<bssn_system_t> 
{
 private:
    
    using base_t = fd_evolution_system_t<bssn_system_t>  ;

 public:

    bssn_system_t( grace::var_array_t<GRACE_NSPACEDIM> state_ 
                 , grace::var_array_t<GRACE_NSPACEDIM> aux_ )
        : base_t(state_,aux_) 
    {} 

    template< size_t der_order >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_update_impl( int const q 
                       , VEC( int const i 
                            , int const j 
                            , int const k)
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const _idx 
                       , grace::var_array_t<GRACE_NSPACEDIM> const state_new 
                       , double const dt 
                       , double const dtfact ) const
    {

        std::array<double, GRACE_NSPACEDIM> idx{ VEC(_idx(0,q), _idx(1,q), _idx(2,q))} ;  

        grmhd_prims_array_t hydro_state ;
        FILL_PRIMS_ARRAY(hydro_state, this->_aux, q, VEC(i,j,k)); 

        bssn_state_t update = compute_bssn_rhs<der_order>(this->_state, hydro_state, idx)  ; 
        
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_auxiliaries( VEC( const int i
                            , const int j
                            , const int k)
                        , const int64_t q ) const 
    {

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_max_eigenspeed( VEC( const int i
                               , const int j
                               , const int k)
                          , const int64_t q ) const
    {
        return 1. ; 
    } 

} ; 

} // namespace grace 

#endif 