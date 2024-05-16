/**
 * @file scalar_advection.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-15
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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

#include <thunder_config.h>
#include <thunder/physics/thunder_physical_systems.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/amr/amr_functions.hh>
#include <thunder/utils/thunder_utils.hh>

#include <thunder/data_structures/variable_indices.hh>
#include <thunder/data_structures/variables.hh>
#include <thunder/system/thunder_system.hh>
#include <thunder/coordinates/coordinate_systems.hh>

#include <Kokkos_Core.hpp>
#include <cmath>

namespace thunder {
void set_scalar_advection_initial_data() { 
    using namespace thunder ; 
    using namespace Kokkos  ; 

    #ifndef THUNDER_ENABLE_SCALAR_ADV 
    int const U = -1 ; 
    ERROR("Should not run scalar advection equation ID" 
          "if scalar advection equation evolution is not enabled.") ; 
    #endif 
    auto& state = thunder::variable_list::get().getstate() ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    
    int64_t nq = amr::get_local_num_quadrants() ;
    auto& params = thunder::config_parser::get() ; 
    double VEC(ax,ay,az) ; 
    EXPR(
    ax = params["scalar_advection"]["ax"].as<double>() ;,
    ay = params["scalar_advection"]["ay"].as<double>() ;,
    az = params["scalar_advection"]["az"].as<double>() ; )

    double sigma = params["scalar_advection"]["gaussian_sigma"].as<double>() ;
    EXPR(
    double xc = params["scalar_advection"]["gaussian_x_c"].as<double>() ;,
    double yc = params["scalar_advection"]["gaussian_y_c"].as<double>() ;,
    double zc = params["scalar_advection"]["gaussian_z_c"].as<double>() ; 
    )
    auto& coord_system = thunder::coordinate_system::get() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 

    int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
    for( int64_t icell=0; icell<ncells; ++icell) {
        size_t const i = icell%(nx+2*ngz); 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef THUNDER_3D 
        size_t const k = 
            (icell/(nx+2*ngz)/(ny+2*ngz)) % (nz+2*ngz) ; 
        size_t const q = 
            (icell/(nx+2*ngz)/(ny+2*ngz)/(nz+2*ngz)) ;
        #else 
        size_t const q = (icell/(nx+2*ngz)/(ny+2*ngz)) ; 
        #endif 
        /* Physical coordinates of cell center */
        auto pcoords = coord_system.get_physical_coordinates(
            {VEC(i,j,k)},
            q,
            true
        ) ; 
        EXPR(
        double const x0 = 
            xc - pcoords[0] ;, 
        double const y0 = 
            yc - pcoords[1] ;, 
        double const z0 = 
            zc - pcoords[2] ; 
        ) 
        double const r = sqrt( EXPR(
            math::int_pow<2>(x0), + math::int_pow<2>(y0), + math::int_pow<2>(z0)
        )) ; 
        h_state_mirror(VEC(i,j,k),U,q) = exp(- 0.5 * math::int_pow<2>(r)/math::int_pow<2>(sigma)) / sigma / sqrt(2*M_PI) ;
        
    }
    Kokkos::deep_copy(state,h_state_mirror) ;
}
}