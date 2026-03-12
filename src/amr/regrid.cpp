/**
 * @file regrid.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-19
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#include <Kokkos_Core.hpp>

#include <grace/amr/regrid.hh>
#include <grace/amr/regrid/regrid_transaction.hh>
#include <grace/coordinates/coordinate_systems.hh>

#include <spdlog/stopwatch.h>

namespace grace { namespace amr { 

bool regrid() {
    Kokkos::Profiling::pushRegion("regrid") ;
    GRACE_INFO("Initiating regrid.") ;  
    spdlog::stopwatch sw ; 
    /******************************************************************************************/
    /*                              Do the thing                                              */
    /******************************************************************************************/
    regrid_transaction_t trx{} ; 
    bool grid_has_changed = trx.grid_has_changed() ; 
    Kokkos::fence() ; 
    trx.execute() ; 
    /******************************************************************************************/
    if ( grid_has_changed ) {
        GRACE_INFO("Regrid done: nq_intial {} final {} time elapsed {} s", trx.get_nq_init(), trx.get_nq_final(), sw) ; 
    } else {
        GRACE_INFO("Regrid done: grid unchanged, time elapsed {} s", sw) ; 
    }
    
    /******************************************************************************************/
    /******************************************************************************************/
    Kokkos::Profiling::popRegion() ;
    /******************************************************************************************/
    /*                                      All done                                          */
    /******************************************************************************************/
    return grid_has_changed ; 
    
}; 

}} /* namespace grace::amr */ 