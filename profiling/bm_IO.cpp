/**
 * @file bm_IO.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-24
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
#include <grace/system/grace_system.hh>
#include <grace/IO/vtk_output.hh>
#include <grace/IO/hdf5_output.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/grace_data_structures.hh>

#define N 1

int main(int argc, char* argv[]) {
    grace::initialize(argc, argv) ; 

    auto state  = grace::variable_list::get().getstate() ;
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    int ngz = grace::amr::get_n_ghosts() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 

    auto const ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ; 

    for( size_t icell=0UL; icell<ncells; icell+=1UL)
    {
        size_t const i = icell%(nx + 2*ngz) ; 
        size_t const j = (icell/(nx + 2*ngz)) % (ny + 2*ngz) ;
        #ifdef GRACE_3D 
        size_t const k = 
            (icell/(nx + 2*ngz)/(ny + 2*ngz)) % (nz + 2*ngz) ; 
        size_t const q = 
            (icell/(nx + 2*ngz)/(ny + 2*ngz)/(nz + 2*ngz)) ;
        #else 
        size_t const q = (icell/(nx + 2*ngz)/(nx + 2*ngz)) ; 
        #endif 
        auto const coords = grace::get_physical_coordinates({VEC(i,j,k)},q, {VEC(0.5,0.5,0.5)}, true) ; 
        double const r2 = EXPR( math::int_pow<2>(coords[0]),
                              + math::int_pow<2>(coords[1]),
                              + math::int_pow<2>(coords[2]) )  ; 
        h_state_mirror(VEC(i,j,k),U,q) = exp( - r2 / 0.5 ) ; 
    }
    Kokkos::deep_copy(state, h_state_mirror) ;

    for( int i=0; i<N; ++i) {
        grace::IO::write_cell_data_vtk(true,false,false); 
        grace::IO::write_cell_data_hdf5(true,false,false) ; 
    }
    grace::grace_finalize() ; 
}