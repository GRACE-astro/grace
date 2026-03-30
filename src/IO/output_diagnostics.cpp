/**
 * @file output_diagnostics.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-17
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

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/device_vector.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/IO/spherical_surfaces.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/amr/amr_functions.hh>

#include <grace/config/config_parser.hh>

#include <grace/IO/diagnostics/black_hole_diagnostics.hh>
#ifdef GRACE_ENABLE_Z4C_METRIC
#include <grace/IO/diagnostics/gw_integrals.hh>
#include <grace/physics/z4c.hh>
#endif
#include <grace/IO/diagnostics/outflow_diagnostics.hh>
#include <Kokkos_Core.hpp>

#include <array>
#include <memory>


namespace grace { namespace IO { 


void output_diagnostics() {
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    using namespace Kokkos  ; 
    
    bh_diagnostics bh_diag{};
    bh_diag.compute_and_write() ;  
    #ifdef GRACE_ENABLE_Z4C_METRIC
    auto state   = variable_list::get().getstate() ; 
    auto sstate  = variable_list::get().getstaggeredstate() ; 
    auto aux     = variable_list::get().getaux() ; 
    auto idx     = variable_list::get().getinvspacings() ;  
    auto coords  = coordinate_system::get().get_device_coord_system() ; 
    // compute psi4 
    z4c_system_t z4c_eq_system(state,aux,sstate) ; 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>
        policy({VEC(ngz-1,ngz-1,ngz-1),0},{VEC(nx+ngz+1,ny+ngz+1,nz+ngz+1),nq}) ;
    parallel_for(GRACE_EXECUTION_TAG("EVOL","compute_psi4"), policy 
    , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
    {
        z4c_eq_system.compute_psi4(i,j,k,q,idx,coords) ; 
    }) ; 
    gw_integrals gw_ints{} ; 
    gw_ints.compute_and_write() ;  
    #endif
    outflows outfl{} ; 
    outfl.compute_and_write() ; 
}

void initialize_diagnostic_files() {
    bh_diagnostics bh_diag{};
    bh_diag.initialize_files() ; 
    #ifdef GRACE_ENABLE_Z4C_METRIC
    gw_integrals gw_ints{} ; 
    gw_ints.initialize_files() ;  
    #endif
    outflows outfl{} ; 
    outfl.initialize_files() ; 
    parallel::mpi_barrier() ;
}

} } 