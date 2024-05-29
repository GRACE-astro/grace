/**
 * @file burgers_equation.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-16
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

#include <grace_config.h>
#include <grace/physics/grace_physical_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/utils/grace_utils.hh>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/system/grace_system.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/physics/burgers.hh>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <string> 

namespace grace {

static void set_burgers_gaussian_id() {
    using namespace grace ; 
    using namespace Kokkos  ; 
    #ifndef GRACE_ENABLE_BURGERS 
    int const U = -1 ; 
    #endif 
    auto& state = grace::variable_list::get().getstate() ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nq = amr::get_local_num_quadrants() ;
    auto& params = grace::config_parser::get() ; 

    double sigma = params["burgers_equation"]["gaussian_sigma"].as<double>() ;
    EXPR(
    double xc = params["burgers_equation"]["gaussian_x_c"].as<double>() ;,
    double yc = params["burgers_equation"]["gaussian_y_c"].as<double>() ;,
    double zc = params["burgers_equation"]["gaussian_z_c"].as<double>() ; 
    )

    auto& coord_system = grace::coordinate_system::get() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 

    int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
    for( int64_t icell=0; icell<ncells; ++icell) {
        size_t const i = icell%(nx+2*ngz); 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
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

static void set_burgers_shocktube_id() {
    using namespace grace ; 
    using namespace Kokkos  ; 
    #ifndef GRACE_ENABLE_BURGERS 
    int const U = -1 ; 
    #endif 
    auto& state = grace::variable_list::get().getstate() ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nq = amr::get_local_num_quadrants() ;
    auto& params = grace::config_parser::get() ; 

    double uL = params["burgers_equation"]["shocktube_left_state"].as<double>() ;
    double uR = params["burgers_equation"]["shocktube_right_state"].as<double>() ;

    double xc = params["burgers_equation"]["shocktube_x_location"].as<double>() ;

    auto& coord_system = grace::coordinate_system::get() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 

    int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
    for( int64_t icell=0; icell<ncells; ++icell) {
        size_t const i = icell%(nx+2*ngz); 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
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

        h_state_mirror(VEC(i,j,k),U,q) = pcoords[0] <= xc ? uL : uR ;  
    }
    Kokkos::deep_copy(state,h_state_mirror) ;
}

static void set_burgers_three_state_shocktube_id() {
    using namespace grace ; 
    using namespace Kokkos  ; 
    #ifndef GRACE_ENABLE_BURGERS 
    int const U = -1 ; 
    #endif 
    auto& state = grace::variable_list::get().getstate() ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nq = amr::get_local_num_quadrants() ;
    auto& params = grace::config_parser::get() ; 

    double uL = params["burgers_equation"]["shocktube_left_state"].as<double>() ;
    double uC = params["burgers_equation"]["shocktube_central_state"].as<double>() ;
    double uR = params["burgers_equation"]["shocktube_right_state"].as<double>() ;

    double xc  = params["burgers_equation"]["shocktube_x_location"].as<double>() ;
    double xc2 = params["burgers_equation"]["shocktube_x_location_2"].as<double>() ;

    auto& coord_system = grace::coordinate_system::get() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 

    int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
    for( int64_t icell=0; icell<ncells; ++icell) {
        size_t const i = icell%(nx+2*ngz); 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
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

        h_state_mirror(VEC(i,j,k),U,q) = pcoords[0] <= xc ? uL : 
        ( pcoords[0] <= xc2 ? uC : uR ) ;  
    }
    Kokkos::deep_copy(state,h_state_mirror) ;
}

static void set_burgers_N_wave_id() {
    using namespace grace ; 
    using namespace Kokkos  ; 
    #ifndef GRACE_ENABLE_BURGERS 
    int const U = -1 ; 
    #endif 
    auto& state = grace::variable_list::get().getstate() ; 

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nq = amr::get_local_num_quadrants() ;
    auto& params = grace::config_parser::get() ; 

    auto& coord_system = grace::coordinate_system::get() ; 
    auto h_state_mirror = Kokkos::create_mirror_view(state) ; 

    int64_t ncells = EXPR((nx+2*ngz),*(ny+2*ngz),*(nz+2*ngz))*nq ;
    for( int64_t icell=0; icell<ncells; ++icell) {
        size_t const i = icell%(nx+2*ngz); 
        size_t const j = (icell/(nx+2*ngz)) % (ny+2*ngz) ;
        #ifdef GRACE_3D 
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
        double const x = pcoords[0] ; 
        h_state_mirror(VEC(i,j,k),U,q) = exp(-0.5*math::int_pow<2>(x-1)) - exp(-0.5*math::int_pow<2>(x+1)) ;  
    }
    Kokkos::deep_copy(state,h_state_mirror) ;
}

void set_burgers_initial_data() {
    using namespace grace ; 
    using namespace Kokkos ; 
    #ifndef GRACE_ENABLE_BURGERS
    ERROR("Should not run Burgers equation ID" 
          "if Burgers equation evolution is not enabled.") ; 
    #endif 
    auto& params = grace::config_parser::get() ;

    std::string which_id = params["burgers_equation"]["id_type"].as<std::string>() ;

    if( which_id == "gaussian") {
        set_burgers_gaussian_id() ; 
    } else if ( which_id == "shocktube" ) {
        set_burgers_shocktube_id() ; 
    } else if ( which_id == "three_states_shocktube") {
        set_burgers_three_state_shocktube_id() ; 
    } else if ( which_id == "oned_N_wave") {
        set_burgers_N_wave_id() ; 
    } else {
        ERROR("Unsupported id_type for Burgers equation.") ; 
    }
}

}