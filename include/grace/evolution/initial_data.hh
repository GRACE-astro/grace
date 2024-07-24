/**
 * @file initial_data.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-15
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
#ifndef GRACE_EVOLUTION_INITIAL_DATA_HH
#define GRACE_EVOLUTION_INITIAL_DATA_HH

#include <grace_config.h>
#include <grace/physics/eos/eos_types.hh>

namespace grace {
/**
 * \defgroup id Initial Data
 * 
 */

/**
 * @brief Set the initial data.
 * \ingroup id
 */
void set_initial_data() ; 
/**
 * @brief Implementation of set_initial_data for a concrete EOS type.
 * \ingroup id
 * \cond grace_detail
 * @tparam eos_t Type of active EOS.
 */
template< typename eos_t >
void set_initial_data_impl() ;

/**
 * @brief Transform vector and tensor variables
 *        from physical coordinates grid coordinates.
 * \ingroup id
 * \cond detail 
 */
void transform_to_logical_frame() ; 

#define INSTANTIATE_TEMPLATE(EOS)   \
extern template                     \
void set_initial_data_impl<EOS>()
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE
}

#endif 