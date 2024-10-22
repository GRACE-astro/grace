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
#include <grace/physics/id/tov.hh>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Kokkos_Core.hpp>


#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>

#define DER_ORDER 2


TEST_CASE("TOV")
{
    using namespace Kokkos; 
    using namespace grace ; 

    DECLARE_GRID_EXTENTS ; // defines nx, ny, nz, ngz, nq 

    // Get variables 
    auto& varlist = variable_list::get() ; 

    // Get state array 
    auto& sstate = varlist.getstaggeredstate() ;
    auto& cstate = sstate.corner_staggered_fields ; 

    // Get coordinates of grid corners 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    GRACE_INFO("Filling coordinates.") ; 
    grace::fill_physical_coordinates(pcoords, {VEC(true,true,true)} ) ;
    Kokkos::fence() ; 
    GRACE_INFO("Done.") ;

    GRACE_INFO("Extents {} {} {} {} {}", cstate.extent(0),cstate.extent(1),cstate.extent(2),cstate.extent(3),cstate.extent(4)) ; 

    // Get grid spacing 
    auto& idx = varlist.getinvspacings() ; 

    // gaussian test function for state defintion
    auto test_func = KOKKOS_LAMBDA ( double a, double b, double s, double x, double y, double z)
    {
        return a+b*Kokkos::exp(-(x*x+y*y+z*z)/(2*s));
    } ; 

    grace::var_array_t<GRACE_NSPACEDIM> Tmunu("Tmunu", VEC(nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz), 16, nq) ;
    grace::var_array_t<GRACE_NSPACEDIM> rhs("RHS", VEC(nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz), NUM_BSSN_VARS, nq) ;

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
            
            // evaluating coordinate values at cell vertices
            #if 0
            double const x = pcoords(VEC(i,j,k),0,q); 
            double const y = pcoords(VEC(i,j,k),1,q);
            double const z = pcoords(VEC(i,j,k),2,q); 
            #endif 
            double x{0},y{0},z{0} ; 

            // filling the conformal metric with the test function
            cstate(VEC(i,j,k), GTXX_, q) = test_func(1,1,1,x,y,z);
            cstate(VEC(i,j,k), GTXY_, q) = test_func(2,1,1,x,y,z);
            cstate(VEC(i,j,k), GTYY_, q) = test_func(3,1,1,x,y,z);
            cstate(VEC(i,j,k), GTXZ_, q) = test_func(4,1,1,x,y,z);
            cstate(VEC(i,j,k), GTYZ_, q) = test_func(5,1,1,x,y,z);
            cstate(VEC(i,j,k), GTZZ_, q) = test_func(6,1,1,x,y,z);

            // lapse function
            cstate(VEC(i,j,k), ALP_, q) = test_func(0,1,1,x,y,z);

            // shift vector components
            cstate(VEC(i,j,k), BETAX_, q) = test_func(0,1,1,x,y,z);
            cstate(VEC(i,j,k), BETAY_, q) = test_func(0,1,1,x,y,z);
            cstate(VEC(i,j,k), BETAZ_, q) = test_func(0,1,1,x,y,z);

            // conformal factor
            cstate(VEC(i,j,k), PHI_, q) = test_func(0,1,1,x,y,z);

            // trace of the extrinsic curvature
            cstate(VEC(i,j,k), K_, q) = test_func(0,1,1,x,y,z);

            // conformal trace-free extrinsic curvature
            cstate(VEC(i,j,k), ATXX_, q) = test_func(0,1,1,x,y,z);
            cstate(VEC(i,j,k), ATXY_, q) = test_func(0,1,1,x,y,z);
            cstate(VEC(i,j,k), ATYY_, q) = test_func(0,1,1,x,y,z);
            cstate(VEC(i,j,k), ATXZ_, q) = test_func(0,1,1,x,y,z);
            cstate(VEC(i,j,k), ATYZ_, q) = test_func(0,1,1,x,y,z);
            cstate(VEC(i,j,k), ATZZ_, q) = test_func(0,1,1,x,y,z);
            #if 1
            // energy-momentum tensor components 
            for( int ww=0; ww<16; ++ww){
               Tmunu(VEC(i,j,k),ww,q) = test_func(0,1,1,x,y,z);
            }
            #endif 
        }
    ) ; 
    Kokkos::fence() ; 
    GRACE_INFO("Here.") ; 
    MDRangePolicy<Rank<GRACE_NSPACEDIM+1>>
        policyEOM({VEC(ngz,ngz,ngz),0}, {nx+1+ngz,ny+1+ngz,nz+1+ngz,nq}) ; 
    // Parallel for loop 
    parallel_for(
        "evaluate_EOM", 
        policyEOM, 
        KOKKOS_LAMBDA (VEC(int i, int j, int k), int q) {
            std::array<std::array<double,4>,4> TmunuL ;
            int idx4 = 0 ; 
            for( int ii=0; ii<4; ii++) {
                for( int jj=0; jj<4; ++jj){
                    TmunuL[ii][jj] = Tmunu(VEC(i,j,k),idx4,q) ;
                    ++idx4 ; 
                }
            }     
            std::array<double,GRACE_NSPACEDIM> idxL {idx(0,q),idx(1,q),idx(2,q )} ;        
            auto rhsL=compute_bssn_rhs<DER_ORDER>(VEC(i,j,k),q,cstate,TmunuL,idxL);
            for( int ivar=0; ivar<NUM_BSSN_VARS; ++ivar) 
                rhs(VEC(i,j,k),ivar,q) = rhsL[ivar] ;
        }
    ) ; 
    Kokkos::fence() ;
    GRACE_INFO("Here") ; 
    // copying the rhs vector to CPU
    auto h_rhs = Kokkos::create_mirror_view(rhs) ;
    Kokkos::deep_copy(h_rhs,rhs) ; 
    
    // fixing y and z coordinates to some values
    size_t jNow=ngz+ny/2,kNow=ngz+nz/2;
    auto& coordsys = grace::coordinate_system::get() ; 

    // looping over x-cordinate values at cell vertices 
    for(size_t i=ngz;i<nx+1+ngz;i++)
    {
        auto xyz = coordsys.get_physical_coordinates({VEC(i,jNow,kNow)},0,{VEC(0,0,0)},false) ; 
        std::cout<< xyz[0] << '\t' << xyz[1] << '\t' << xyz[2] << '\t' ; 
        for( int ivar =0 ; ivar< NUM_BSSN_VARS; ivar++){
            std::cout << h_rhs(VEC(i,jNow,kNow),ivar,0)<<'\t';
        }
        std::cout<<std::endl;
    }

}
   