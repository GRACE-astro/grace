/**
 * @file c2p.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-10
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

#ifndef GRACE_PHYSICS_EOS_C2P_HH
#define GRACE_PHYSICS_EOS_C2P_HH

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/eos/tabulated_eos.hh>
#include <grace/physics/grmhd_helpers.hh>


namespace grace {
/**
 * @brief Convert conservative variables to primitive ones.
 * 
 * @tparam eos_t Type of EOS.
 * @param cons Conservative variables (at one cell).
 * @param prims Primitive variables (at one cell).
 * @param metric Metric utilities.
 * @param eos Equation of State.
 * @param lapse_excision minimum lapse function below which MHD is excised.
 * Atmosphere conditions are enforced by this routine.
 * Conserved variables are recomputed to be consistent with inverted
 * primitives.
 */
template< typename eos_t >
void GRACE_HOST_DEVICE
conservs_to_prims(  grace::grmhd_cons_array_t& cons 
                  , grace::grmhd_prims_array_t& prims
                  , grace::metric_array_t const& metric 
                  , eos_t const& eos
                  , double const& lapse_excision) ; 

void GRACE_HOST_DEVICE
prims_to_conservs( grace::grmhd_prims_array_t& prims
                 , grace::grmhd_cons_array_t& cons 
                 , grace::metric_array_t const& metric ) ; 
// Explicit template instantiation
#define INSTANTIATE_TEMPLATE(EOS) \
extern template \
void GRACE_HOST_DEVICE \
conservs_to_prims<EOS>( grace::grmhd_cons_array_t&  \
                      , grace::grmhd_prims_array_t&  \
                      , grace::metric_array_t const&  \
                      , EOS const& eos \
                      , double const& ) 
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
INSTANTIATE_TEMPLATE(grace::tabulated_eos_t) ;  
#undef INSTANTIATE_TEMPLATE
}

#endif /* GRACE_PHYSICS_EOS_C2P_HH */