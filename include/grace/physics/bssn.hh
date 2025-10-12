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

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/math.hh>
#include <grace/utils/metric_utils.hh>

#include <grace/data_structures/variable_properties.hh>

#include <grace/evolution/fd_evolution_system.hh>

#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/bssn_helpers.hh>

#include <Kokkos_Core.hpp>

#include <array>


namespace grace {

template< size_t der_order >
bssn_state_t GRACE_HOST_DEVICE 
compute_bssn_rhs( VEC(int i, int j, int k), int q
                , grace::var_array_t const state
                , std::array<std::array<double,4>,4> const& Tmunu
                , std::array<double,GRACE_NSPACEDIM> const& idx
                , double const k1, double const eta );


struct bssn_system_t 
    : public fd_evolution_system_t<bssn_system_t> 
{
 private:
    
    using base_t = fd_evolution_system_t<bssn_system_t>  ;

 public:

    bssn_system_t( grace::var_array_t state_ 
                 , grace::var_array_t aux_
                 , grace::staggered_variable_arrays_t sstate_
                 , double _k1, double _eta, double _epsdiss )
        : base_t(state_,aux_,sstate_), _k1(_k1), _eta(_eta), _epsdiss(_epsdiss)
    {} 

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_update_impl( int const q 
                       , VEC( int const i 
                            , int const j 
                            , int const k)
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const _idx 
                       , grace::var_array_t const state_new 
                       , grace::staggered_variable_arrays_t sstate_new
                       , double const dt 
                       , double const dtfact ) const
    {

        std::array<double, GRACE_NSPACEDIM> idx{ VEC(_idx(0,q), _idx(1,q), _idx(2,q))} ;  

        grmhd_prims_array_t hydro_state ;
        FILL_PRIMS_ARRAY(hydro_state, this->_aux, q, VEC(i,j,k)); 
        metric_array_t metric ;
        FILL_METRIC_ARRAY(metric, this->_state, q, VEC(i,j,k)) ; 

        auto const Tdd = get_Tmunu(hydro_state, metric); 

        bssn_state_t update = compute_bssn_rhs<BSSN_DER_ORDER>(VEC(i,j,k),q,this->_state, Tdd, idx, _k1, _eta)  ; 
        for( int ivar=GTXXL; ivar<NUM_BSSN_VARS; ++ivar) {
            state_new(VEC(i,j,k),GTXX_+ivar,q) += dt * dtfact * update[ivar] ;
        }

        impose_algebraic_constraints(state_new,VEC(i,j,k),q) ; 
        
    }

    KOKKOS_FUNCTION
    std::array<std::array<double,4>,4> get_Tmunu(
        grmhd_prims_array_t const& prims,
        metric_array_t const& metric
    ) const {
        std::array<std::array<double,4>,4> Tmunu ;

        double const W =  compute_W(prims,metric) ; 
        double const u0 = W / metric.alp() ; 
        std::array<double,3> uD3 = metric.lower({prims[VXL] + metric.beta(0), prims[VYL] + metric.beta(1), prims[VZL] + metric.beta(2)}) ; 
        // u_t = W ( beta^i v_i - alp ) with v == eulerian velocity! 
        auto uD0 = u0 * metric.contract_vec_covec({metric.beta(0),metric.beta(1),metric.beta(2)}, uD3) - metric.alp() * W ; 
        std::array<double,4> uD { uD0, uD3[0]*u0, uD3[1]*u0, uD3[2]*u0 } ; 
        auto gdd = metric.gmunu()     ; 
        int idx4[4][4] = {
            {0,1,2,3},
            {1,4,5,6},
            {2,5,7,8},
            {3,6,8,9}
        } ; 
        for( int mu=0; mu<4; ++mu ) {
            for( int nu=0; nu<4; ++nu) {
                // TODO missing b field contribution 
                Tmunu[mu][nu] = (prims[RHOL]*(1+prims[EPSL]) + prims[PRESSL]) * uD[mu] * uD[nu] 
                              + prims[PRESSL] * gdd[idx4[mu][nu]] ;
            }
        }

        return Tmunu ;
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

    void GRACE_HOST_DEVICE 
    impose_algebraic_constraints(grace::var_array_t state, VEC(int i, int j, int k), int q) const 
    {
        /* First impose the det(gtilde) = 1 constraint */
        double const gtxx = state(VEC(i,j,k),GTXX_+0,q);
        double const gtxy = state(VEC(i,j,k),GTXX_+1,q);
        double const gtxz = state(VEC(i,j,k),GTXX_+2,q);
        double const gtyy = state(VEC(i,j,k),GTXX_+3,q);
        double const gtyz = state(VEC(i,j,k),GTXX_+4,q);
        double const gtzz = state(VEC(i,j,k),GTXX_+5,q);

        double const detgt     = -(gtxz*gtxz*gtyy) + 2*gtxy*gtxz*gtyz - gtxx*(gtyz*gtyz) - gtxy*gtxy*gtzz + gtxx*gtyy*gtzz;
        double const cbrtdetgt = Kokkos::cbrt(detgt);

        state(VEC(i,j,k),GTXX_+0,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+1,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+2,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+3,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+4,q) /= cbrtdetgt ; 
        state(VEC(i,j,k),GTXX_+5,q) /= cbrtdetgt ; 

        /* And the trace-free Aij constraint next */

        double const gtXX=(-(gtyz*gtyz) + gtyy*gtzz)/detgt ;
        double const gtXY=(gtxz*gtyz - gtxy*gtzz)/detgt    ;
        double const gtXZ=(-(gtxz*gtyy) + gtxy*gtyz)/detgt ;
        double const gtYY=(-(gtxz*gtxz) + gtxx*gtzz)/detgt ;
        double const gtYZ=(gtxy*gtxz - gtxx*gtyz)/detgt    ;
        double const gtZZ=(-(gtxy*gtxy) + gtxx*gtyy)/detgt ; 

        double const Atxx = state(VEC(i,j,k),ATXX_+0,q);
        double const Atxy = state(VEC(i,j,k),ATXX_+1,q);
        double const Atxz = state(VEC(i,j,k),ATXX_+2,q);
        double const Atyy = state(VEC(i,j,k),ATXX_+3,q);
        double const Atyz = state(VEC(i,j,k),ATXX_+4,q);
        double const Atzz = state(VEC(i,j,k),ATXX_+5,q);

        double const ATR = Atxx*gtXX + 2*Atxy*gtXY + 2*Atxz*gtXZ + Atyy*gtYY + 2*Atyz*gtYZ + Atzz*gtZZ ; 
        
        state(VEC(i,j,k),ATXX_+0,q) -= 1./3. * state(VEC(i,j,k),GTXX_+0,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+1,q) -= 1./3. * state(VEC(i,j,k),GTXX_+1,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+2,q) -= 1./3. * state(VEC(i,j,k),GTXX_+2,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+3,q) -= 1./3. * state(VEC(i,j,k),GTXX_+3,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+4,q) -= 1./3. * state(VEC(i,j,k),GTXX_+4,q) * ATR ; 
        state(VEC(i,j,k),ATXX_+5,q) -= 1./3. * state(VEC(i,j,k),GTXX_+5,q) * ATR ; 

    }

    double _k1, _eta, _epsdiss;

} ; 

} // namespace grace 

#endif 