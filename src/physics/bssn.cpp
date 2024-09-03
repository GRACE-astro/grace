/**
 * @file bssn.cpp
 * @author  Christian Ecker
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

#include <grace_config.h> 

#include <grace/physics/bssn.hh>

#include <grace/utils/fd_utils.hh>

namespace grace {

template< size_t der_order >
bssn_state_t GRACE_HOST_DEVICE 
compute_bssn_rhs( grace::var_array_t<GRACE_NSPACEDIM> const state
                , std::array<std::array<double,4>,4> const& Tmunu
                , std::array<double,GRACE_NSPACEDIM> const& idx)
{
    /*
     * state --> contains all evolved variables 
     * e.g. if you need gtxx at i,j,k:
     * 
     *  gtxx := state(VEC(i,j,k), GTXX_, q) ; 
     * 
     * For computing derivatives:
     * 
     *  \partial_x gxx(i,j,k) := grace::fd_der<idir, der_order>(state, GTXX_, VEC(i,j,k), q) ; 
     * Where idir is 0, 1, 2 for X, Y, Z. 
     * 
     * double const betaxdy = grace::fd_der<1, der_order>(state,BETAX_, VEC(i,j,k), q) ; 
     * 
     * The BSSN vars indices are as follows:
     * 
     * GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_,
     * PHI_, K_, BETAX_, BETAY_, BETAZ_, ALP_,
     * GAMMAX_,GAMMAY_,GAMMAZ_,
     * ATXX_, ATXY_, ATXZ_, ATYY_, ATYZ_, ATZZ_
     * 
     * Hydro quantities:
     * 
     * I will pass T_{\mu\nu} to this function:
     * 
     * Ttt = Tmunu[0][0]  ; // ALL INDICES LOW, INDICES 0,1,2,3
     * 
     */
}


}