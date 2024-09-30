/**
 * @file test_grid_loops.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-30
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

#include <grace/utils/gridloop.hh>

#include <catch2/catch_test_macros.hpp>

#include <Kokkos_Core.hpp>

TEST_CASE("grid_loop", "[utils][hostutils]")
{
    using namespace grace; 
    using namespace Kokkos ;

    DECLARE_GRID_EXTENTS ; 

    /****************************/
    /* Test serial loops        */
    /****************************/
    /* Set up an arrays on Host */ 
    /****************************/
    View<int EXPR(*,*,*)*, HostSpace> 
        arr{"arr",VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz), nq} ;
    View<int EXPR(*,*,*)*, HostSpace> 
        arr_x{"arr_x",VEC(nx+1+2*ngz,ny+2*ngz,nz+2*ngz), nq} ;
    View<int EXPR(*,*,*)*, HostSpace> 
        arr_y{"arr_y",VEC(nx+2*ngz,ny+1+2*ngz,nz+2*ngz), nq} ;
    View<int EXPR(*,*,*)*, HostSpace> 
        arr_z{"arr_z",VEC(nx+2*ngz,ny+2*ngz,nz+1+2*ngz), nq} ;
    View<int EXPR(*,*,*)*, HostSpace> 
        arr_xy{"arr_xy",VEC(nx+1+2*ngz,ny+1+2*ngz,nz+2*ngz), nq} ;
    View<int EXPR(*,*,*)*, HostSpace> 
        arr_xz{"arr_xz",VEC(nx+1+2*ngz,ny+2*ngz,nz+1+2*ngz), nq} ;
    View<int EXPR(*,*,*)*, HostSpace> 
        arr_yz{"arr_yz",VEC(nx+2*ngz,ny+1+2*ngz,nz+1+2*ngz), nq} ;
    View<int EXPR(*,*,*)*, HostSpace> 
        arr_xyz{"arr_xyz",VEC(nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz), nq} ; 

    // No stagger with gz 
    host_grid_loop<false>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q)
        {
            arr(VEC(i,j,k),q) = 42 ; 
        },
        {VEC(false,false,false)},
        true 
    ) ; 
    // check 
    size_t const ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*((nz+2*ngz))) * nq ; 
    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
        size_t const i = icell%(nx+2*ngz) ; 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
        size_t const k = 
            (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
        size_t const q = 
            (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
        #else 
        size_t const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
        #endif
        CHECK( arr(VEC(i,j,k),q) == 42 ) ; 
    } 

    // No stagger with gz 
    host_grid_loop<true>(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q)
        {
            arr(VEC(i,j,k),q) *= 2 ; 
        },
        {VEC(false,false,false)},
        true 
    ) ; 
    // check 
    
    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
        size_t const i = icell%(nx+2*ngz) ; 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
        size_t const k = 
            (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
        size_t const q = 
            (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
        #else 
        size_t const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
        #endif
        CHECK( arr(VEC(i,j,k),q) == 84 ) ; 
    } 

}