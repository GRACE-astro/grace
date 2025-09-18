/**
 * @file forest.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-02-29
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

#include <grace/amr/forest.hh>

#include <grace/amr/amr_flags.hh>
#include <grace/parallel/mpi_wrappers.hh> 
#include <grace/amr/connectivity.hh>
#include <grace/config/config_parser.hh>
#include <grace/system/print.hh>

namespace grace { namespace amr {

forest_impl_t::forest_impl_t() 
{ 
    GRACE_INFO("Initializing forest of oct-trees...")  ;
    auto & params       = grace::config_parser::get()   ; 
    auto & connectivity = grace::amr::connectivity::get() ; 
    int min_level( params["amr"]["initial_refinement_level"].as<int>() ) ; 
    _p4est =  p4est_new_ext(  parallel::get_comm_world()
                            , connectivity.get()   
                            , 0 
                            , min_level
                            , 1 
                            , sizeof(amr_flags_t)
                            , initialize_quadrant
                            , nullptr ) ; 
    GRACE_INFO("Forest initialized with {} ({}) total (local) quadrants."
                 , _p4est->global_num_quadrants, _p4est->local_num_quadrants ) ; 

    Kokkos::View<size_t[4]> _gp_d("grid_params") ; 
    auto _gp_h = Kokkos::create_mirror_view(_gp_d) ; 
    _gp_h(0) = params["amr"]["npoints_block_x"].as<size_t>() ; 
    _gp_h(1) = params["amr"]["npoints_block_y"].as<size_t>() ; 
    _gp_h(2) = params["amr"]["npoints_block_z"].as<size_t>() ; 
    _gp_h(3) = params["amr"]["n_ghostzones"].as<size_t>() ; 
    Kokkos::deep_copy(_gp_d, _gp_h) ;
    _grid_properties = _gp_d ; 
}

forest_impl_t::~forest_impl_t() 
{ 
    p4est_destroy(_p4est) ; 
}

} } /* namespace grace::amr */