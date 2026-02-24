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


namespace grace { namespace amr { 

void regrid() {
    Kokkos::Profiling::pushRegion("regrid") ;
    GRACE_VERBOSE("Initiating regrid.") ;  
    /******************************************************************************************/
    /*                              Do the thing                                              */
    /******************************************************************************************/
    regrid_transaction_t trx{} ; 
    Kokkos::fence() ; 
    trx.execute() ; 
    /******************************************************************************************/
    /******************************************************************************************/
    Kokkos::Profiling::popRegion() ;
    /******************************************************************************************/
    /*                         Update ghost layer                                             */
    /******************************************************************************************/
    auto& ghost = grace::amr_ghosts::get() ; 
    ghost.update() ;
    /******************************************************************************************/
    /*                                      All done                                          */
    /******************************************************************************************/
    
}; 

}} /* namespace grace::amr */ 