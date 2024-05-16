/**
 * @file thunder.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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
/**********************************************************************************/
/**********************************************************************************/
#include <thunder_config.h>
#include <code_modules.h>
/**********************************************************************************/
/**********************************************************************************/
#include <thunder/system/thunder_system.hh>
#include <thunder/amr/thunder_amr.hh>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/parallel/mpi_wrappers.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/evolution/evolve.hh>
#include <thunder/evolution/initial_data.hh>
#include <thunder/evolution/auxiliaries.hh>
#include <thunder/evolution/find_stable_timestep.hh>
#include <thunder/IO/vtk_volume_output.hh>
/**********************************************************************************/
/**********************************************************************************/
int main(int argc, char* argv[])
{
    /**********************************************************************************/
    /*                               Initialize runtime                               */
    /**********************************************************************************/
    thunder::initialize(argc, argv) ; 
    /**********************************************************************************/
    /**********************************************************************************/
    using namespace thunder::variables ;
    using namespace Kokkos ;
    using namespace thunder ; 
    /**********************************************************************************/
    /**********************************************************************************/
    auto& params = thunder::config_parser::get() ; 
    /**********************************************************************************/
    /*                                 Initial data                                   */
    /**********************************************************************************/
    thunder::set_initial_data() ; 
    bool regrid_at_postinitial = params["amr"]["regrid_at_postinitial"].as<bool>() ; 
    int postinitial_regrid_depth = params["amr"]["postinitial_regrid_depth"].as<int>() ; 
    /**********************************************************************************/
    /*                                 Post-Initial data                              */
    /**********************************************************************************/
    if( regrid_at_postinitial ) {
        for( int ilev=0; ilev<postinitial_regrid_depth; ++ilev){
            thunder::amr::regrid() ;  
            thunder::amr::apply_boundary_conditions() ; 
        }
    }
    thunder::IO::write_volume_cell_data() ;
    /**********************************************************************************/
    /**********************************************************************************/
    double final_time = params["evolution"]["final_time"].as<double>() ; 
    int64_t regrid_every = params["amr"]["regrid_every"].as<int64_t>() ; 
    int64_t volume_output_every = params["IO"]["volume_output_every"].as<int64_t>() ; 
    /**********************************************************************************/
    /*                           Evolution loop                                       */
    /**********************************************************************************/
    for( ; thunder::get_simulation_time() < final_time 
         ; thunder::increment_iteration(), thunder::increment_simulation_time() ) 
    {
        thunder::find_stable_timestep() ;
        THUNDER_INFO("Iter {} time {:.3f} dt {:.3e} ave M/h {:.3e}", thunder::get_iteration(), thunder::get_simulation_time(), thunder::get_timestep(), thunder::get_simulation_time()/thunder::get_total_runtime()*3.6e03 ) ; 
        thunder::evolve() ; 
        if (    (thunder::get_iteration() % regrid_every == 0) 
            and (regrid_every>0)) 
        {
            thunder::amr::regrid() ;  
            thunder::amr::apply_boundary_conditions() ;
        }
        if(    (thunder::get_iteration() % volume_output_every == 0) 
           and (volume_output_every>0) ) 
        {
            thunder::compute_auxiliary_quantities() ; 
            thunder::IO::write_volume_cell_data() ;
        } 
    }
    
    thunder::thunder_finalize() ; 
    return EXIT_SUCCESS ; 
}
/**********************************************************************************/
/**********************************************************************************/