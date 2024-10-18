/**
 * @file test_bssn.cpp
 * @author Christian Ecker (ecker@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-29
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

#include <catch2/catch_test_macros.hpp>
#include <Kokkos_Core.hpp>

#include <grace_config.h>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/IO/scalar_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/system/grace_system.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/physics/bssn.hh>

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>


#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>


TEST_CASE("bssn")
{
    using namespace Kokkos; 
    using namespace grace ; 

    DECLARE_GRID_PROPERTIES ; // defines nx, ny, nz, ngz, nq 

    // Get variables 
    auto& varlist = variable_list::get() ; 

    // Get state array 
    auto& state = varlist.getstate() ;

    // Get coordinates of grid corners 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    grace::fill_physical_coordinates(pcoords, /* staggering */ ) ;

    // Get grid spacing 
    auto& idx = varlist.getinvspacings() ; 

    auto metric_func = KOKKOS_LAMBDA ( double x, double y, double z)
    {
        return Kokkos::sin( M_PI * (x*x+y*y+z*z) ) ;
    } ; 

    // Fill state array 
    // Parallel loop (GPU)

    // Kokkos Policy --> Loop range 
    // RangePolicy --> 0, N
    // MDRangePolicy --> (0,N1) ... 
    // In our case: 3 dimensions (corner indices) + 1 (quadrant index)
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>>
        policy({VEC(0,0,0),0}, {nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz,nq}) ; 
    // Parallel for loop 
    parallel_for(
        "fill_data", 
        policy, 
        KOKKOS_LAMBDA (VEC(int i, int j, int k), int q) {
            double const x = pcoords(VEC(i,j,k),0,q) ; 
            double const y = pcoords(VEC(i,j,k),1,q) ;
            double const z = pcoords(VEC(i,j,k),2,q) ;
            // Body of parallel GPU loop 
            state(VEC(i,j,k), GTXX_, q) = metric_func(x,y,z) ; 

            // ... 
        }
    ) ; 

    
    


}
