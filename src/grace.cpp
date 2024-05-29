/**
 * @file grace.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
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
/**********************************************************************************/
/**********************************************************************************/
#include <grace_config.h>
#include <code_modules.h>
/**********************************************************************************/
/**********************************************************************************/
#include <grace/system/grace_system.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/config/config_parser.hh>
#include <grace/evolution/evolve.hh>
#include <grace/evolution/initial_data.hh>
#include <grace/evolution/auxiliaries.hh>
#include <grace/evolution/find_stable_timestep.hh>
#include <grace/IO/cell_output.hh>
#include <grace/IO/scalar_output.hh>
/**********************************************************************************/
/**********************************************************************************/
int main(int argc, char* argv[])
{
    /**********************************************************************************/
    /*                               Initialize runtime                               */
    /**********************************************************************************/
    grace::initialize(argc, argv) ; 
    /**********************************************************************************/
    /**********************************************************************************/
    using namespace grace::variables ;
    using namespace Kokkos ;
    using namespace grace ; 
    /**********************************************************************************/
    /**********************************************************************************/
    auto& params = grace::config_parser::get() ; 
    /**********************************************************************************/
    /*                                 Initial data                                   */
    /**********************************************************************************/
    GRACE_INFO("Setting initial data.") ; 
    grace::set_initial_data() ; 
    bool regrid_at_postinitial = params["amr"]["regrid_at_postinitial"].as<bool>() ; 
    int postinitial_regrid_depth = params["amr"]["postinitial_regrid_depth"].as<int>() ; 
    /**********************************************************************************/
    /*                                 Post-Initial data                              */
    /**********************************************************************************/
    if( regrid_at_postinitial ) {
        GRACE_INFO("Performing initial regrid.") ;
        for( int ilev=0; ilev<postinitial_regrid_depth; ++ilev){
            grace::amr::regrid() ;  
            grace::amr::apply_boundary_conditions() ; 
        }
    }
    bool reset_id_after_regrid = params["evolution"]["reset_id_after_regrid"].as<bool>() ; 
    if (reset_id_after_regrid) {
        GRACE_INFO("Resetting initial data after regrid.") ;
        grace::set_initial_data() ; 
    }
    grace::IO::write_cell_output(true,true,true) ;
    grace::IO::compute_reductions() ; 
    grace::IO::initialize_output_files() ; 
    grace::IO::write_scalar_output() ; 
    grace::IO::info_output() ;
    /**********************************************************************************/
    /**********************************************************************************/
    double final_time = params["evolution"]["final_time"].as<double>() ; 
    int64_t regrid_every = params["amr"]["regrid_every"].as<int64_t>() ; 
    int64_t volume_output_every = params["IO"]["volume_output_every"].as<int64_t>() ; 
    int64_t plane_surface_output_every = 
        params["IO"]["plane_surface_output_every"].as<int64_t>() ; 
    int64_t sphere_surface_output_every = 
        params["IO"]["sphere_surface_output_every"].as<int64_t>() ;
    int64_t scalar_output_every =
        params["IO"]["scalar_output_every"].as<int64_t>() ; 
    int64_t info_output_every =
        params["IO"]["info_output_every"].as<int64_t>() ; 
    std::string tstep_mode = params["evolution"]["timestep_selection_mode"].as<std::string>() ;
    if ( tstep_mode == "manual" ) {
        grace::set_timestep(params["evolution"]["timestep"].as<double>()) ; 
    }
    /**********************************************************************************/
    /*                           Evolution loop                                       */
    /**********************************************************************************/
    GRACE_INFO("Starting evolution.") ; 
    while( grace::get_simulation_time() < final_time ) 
    {   
        /**********************************************************************************/
        if(tstep_mode == "automatic"){
            grace::find_stable_timestep() ;
        }
        grace::evolve() ; 
        /**********************************************************************************/
        grace::increment_iteration(); grace::increment_simulation_time() ;
        int64_t iter = grace::get_iteration() ; 
        if (    (iter % regrid_every == 0) 
            and (regrid_every>0)) 
        {
            grace::amr::regrid() ;  
            grace::amr::apply_boundary_conditions() ;
        }
        grace::compute_auxiliary_quantities() ;
        if(    (volume_output_every>0) 
           or  (plane_surface_output_every>0) 
           or  (sphere_surface_output_every>0) ) 
        {
            bool do_out_vol = 
                (volume_output_every>0) and (iter % volume_output_every == 0) ; 
            bool do_out_planes =
                (plane_surface_output_every>0) and (iter % plane_surface_output_every == 0) ; 
            bool do_out_spheres = 
                (sphere_surface_output_every>0) and (iter % sphere_surface_output_every == 0) ; 
            grace::IO::write_cell_output(do_out_vol,do_out_planes,do_out_spheres) ;
        } 
        if(  ((scalar_output_every>0)
          and (iter % scalar_output_every == 0))
          or ((info_output_every>0)
          and (iter % info_output_every == 0)))
        {
            grace::IO::compute_reductions() ; 
        }
        if(   (scalar_output_every>0)
          and (iter % scalar_output_every == 0))
        {
            grace::IO::write_scalar_output() ; 
        }
        if(   (info_output_every>0)
          and (iter % info_output_every == 0))
        {
            grace::IO::info_output() ; 
        }
    }
    
    grace::grace_finalize() ; 
    return EXIT_SUCCESS ; 
}
/**********************************************************************************/
/**********************************************************************************/