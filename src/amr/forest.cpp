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

#include <grace/amr/regrid_helpers.hh>
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
}

forest_impl_t::~forest_impl_t() 
{ 
    p4est_destroy(_p4est) ; 
}

} } /* namespace grace::amr */