/**
 * @file eos_types.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Compile-time EOS type traits (has_ye, etc.) used to specialise GRMHD kernels for the active equation-of-state.
 * @date 2024-06-17
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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

#ifndef GRACE_PHYSICS_EOS_EOS_TYPES_HH
#define GRACE_PHYSICS_EOS_EOS_TYPES_HH

#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/eos/ideal_gas_eos.hh>
#include <grace/physics/eos/tabulated_eos.hh>
#include <grace/physics/eos/tabulated_cold_eos.hh>

namespace grace {

// Conservative default: any unlisted EOS evolves Y_e.
// Add a specialization (value = false) for composition-free EOS to
// disable Y_e advection at compile time.
template < typename eos_t >
struct has_ye {
    constexpr static bool value = true ;
} ;

template<> struct has_ye<ideal_gas_eos_t>                          { constexpr static bool value = false ; } ;
template<> struct has_ye<hybrid_eos_t<piecewise_polytropic_eos_t>> { constexpr static bool value = false ; } ;
template<> struct has_ye<hybrid_eos_t<tabulated_cold_eos_t>>       { constexpr static bool value = false ; } ;
template<> struct has_ye<tabulated_eos_t>                          { constexpr static bool value = true  ; } ;

template< typename eos_t >
inline constexpr bool has_ye_v = has_ye<eos_t>::value ;

// Compile-time predicate: true iff the EOS has no built-in cold-curve
// concept (i.e. the only sensible epsilon floor is eps >= 0, not
// eps >= eps_cold(rho)). Used by c2p.cpp to gate the FOFC trigger on
// bottom-of-EOS-table clamps:
//   - ideal-gas:  eps clamps and T-floor ARE robustness signals
//                 (no algebraic floor protects us, anything below the
//                  configured min is a real failure) -> trigger FOFC
//   - hybrid / tabulated: clamps to the cold curve / table bottom are
//                 benign self-consistent EOS clamps; FOFC there destroys
//                 magnetic-field structure in rarefaction regions -> skip
template < typename eos_t >
struct is_ideal_gas {
    constexpr static bool value = false ;
} ;

template<> struct is_ideal_gas<ideal_gas_eos_t> { constexpr static bool value = true ; } ;

template< typename eos_t >
inline constexpr bool is_ideal_gas_v = is_ideal_gas<eos_t>::value ;

} // namespace grace

#endif