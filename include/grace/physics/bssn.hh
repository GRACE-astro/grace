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
#include <grace/physics/grmhd_metric_utils.hh>
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
                 , grace::var_array_t<GRACE_NSPACEDIM> aux_ 
                 , grace::staggered_variable_arrays_t  sstate_ )
        : base_t(state_,aux_,sstate_) 
    {} 

    template< size_t der_order >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_update_impl( int const q 
                       , VEC( int const i 
                            , int const j 
                            , int const k)
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const _idx 
                       , grace::var_array_t<GRACE_NSPACEDIM> const state_new 
                       , grace::staggered_variable_arrays_t const sstate_new 
                       , double const dt 
                       , double const dtfact ) const
    {
        auto& state  = this->_state                             ;
        auto& cstate = this->_sstate.corner_staggered_fields    ;
        auto& cstate_new = sstate_new.corner_staggered_fields    ;
        auto& aux    = this->_aux                               ;

        std::array<double, GRACE_NSPACEDIM> idx{ VEC(_idx(0,q), _idx(1,q), _idx(2,q))} ;  

        auto const metric = get_metric_array(
            state,cstate,
            VEC(i,j,k),
            q,
            {VEC(true,true,true)}
        ) ; 

        double W;
        grmhd_prims_array_t prims = get_primitives_cell_corner(
            aux,state,metric,W,VEC(i,j,k),q
        );

        // Fill Tmunu 
        std::array<std::array<double,4>,4> Tmunu ;
        double const u0 =  W/metric.alp() ; 
        std::array<double,4> uU { u0, prims[VXL]/u0, prims[VYL]/u0, prims[VZL]/u0 } ; 
        auto uD = metric.lower_4vec(uU)  ; 
        auto gdd = metric.invgmunu()     ; 
        int idx4[4][4] = {
            {0,1,2,3},
            {1,4,5,6},
            {2,5,7,8},
            {3,6,7,9}
        } ; 
        for( int mu=0; mu<4; ++mu ) {
            for( int nu=0; nu<4; ++nu) {
                Tmunu[mu][nu] = (prims[RHOL] + prims[PRESSL]) * uD[mu] * uD[nu] + prims[PRESSL] * gdd[idx4[mu][nu]] ;
            }
        }

        bssn_state_t update = compute_bssn_rhs<der_order>(VEC(i,j,k),q,cstate,Tmunu,idx)  ;   
        // Apply Berger-
        // Apply update
        cstate_new(VEC(i,j,k),PHI_,q) += dt * dtfact * update[PHIL] ;
        cstate_new(VEC(i,j,k),K_,q)   += dt * dtfact * update[KL]   ;
        int ww = 0 ; 
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),GTXX_+ww,q) += dt * dtfact * update[GTXXL+ww] ; ++ww;
        ww = 0 ; 
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),ATXX_+ww,q) += dt * dtfact * update[ATXXL+ww] ; ++ww;
        ww = 0 ; 
        cstate_new(VEC(i,j,k),GAMMAX_+ww,q) += dt * dtfact * update[GAMMAXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),GAMMAX_+ww,q) += dt * dtfact * update[GAMMAXL+ww] ; ++ww;
        cstate_new(VEC(i,j,k),GAMMAX_+ww,q) += dt * dtfact * update[GAMMAXL+ww] ; ++ww;

    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_auxiliaries( VEC( const int i
                            , const int j
                            , const int k)
                        , const int64_t q ) const 
    {
        // here we'll need to calculate the constraints 
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