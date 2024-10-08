/**
 * @file admbase.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-11
 * 
 * @copyright This file is part of MagMA.
 * MagMA is an evolution framework that uses Discontinuous Galerkin
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

#ifndef GRACE_PHYSICS_ADMBASE_HH 
#define GRACE_PHYSICS_ADMBASE_HH

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/errors/error.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/evolution/evolution_kernel_tags.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/config/config_parser.hh>

#include <string>

namespace grace 
{

/**
 * @brief This class is the blueprint for any metric evolution system.
 * \ingroup physics
 * @tparam EvolSystem_t CRTP derived class for explicit implementation of the methods.
 */
template< typename EvolSystem_t > 
class metric_evolution_system_t 
    : public fd_evolution_system_t<metric_evolution_system_t> {

 using base_t = typename fd_evolution_system_t<metric_evolution_system_t> ; //!< Base class type

 public:
    /**
     * @brief Default ctor.
     * 
     */
    metric_evolution_system_t()  = default ; 
    /**
     * @brief Construct a metric evolution system given state and aux arrays.
     * 
     * @param state State array
     * @param aux   Aux array
     */
    metric_evolution_system_t(grace::var_array_t<GRACE_NSPACEDIM> const state, grace::var_array_t<GRACE_NSPACEDIM> const aux)
        : base_t(state,aux)
    { } ; 
    /**
     * @brief Dtor.
     * 
     */
    ~metric_evolution_system_t() = default ;

    /**
     * @brief Computation of RHS for metric evolution equations.
     * 
     * @tparam thread_team_t Type of thread team
     * 
     * This method is expected to add the rhs in place to the new state.
     */
    template< typename thread_team_t >
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    _compute_rhs_impl(
                      thread_team_t& team 
                    , VEC( const int i 
                    ,      const int j 
                    ,      const int k)
                    , grace::scalar_array_t<GRACE_NSPACEDIM> const idx
                    , grace::var_array_t<GRACE_NSPACEDIM> const state_new
                    , double const dt 
                    , double const dtfact ) const 
    {
        static_cast<EvolSystem_t const* > (this) -> compute_rhs_impl(VEC(i,j,k),q,idx,state_new,dt,dtfact) ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    _compute_max_eigenspeed_impl ( eigenspeed_kernel_t _tag
               , VEC( const int i
                    , const int j 
                    , const int k)
               , int64_t q ) const 
    {
        return static_cast<EvolSystem_t const* > (this)->compute_max_eigenspeed_impl(VEC(i,j,k),q) ; 
    }

    

 private:
    grace::var_array_t<GRACE_NSPACEDIM> _state, _aux ; 

} ; 


} /* namespace grace */

 #endif /* GRACE_PHYSICS_ADMBASE_HH */