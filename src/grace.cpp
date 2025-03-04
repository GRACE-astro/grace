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
    /**********************************************************************************/
    /**********************************************************************************/
    double final_time = grace::get_param<double>("evolution","final_time") ; 
    int64_t final_iteration = grace::get_param<int64_t>("evolution","final_iteration") ;
    double max_walltime = grace::get_param<double>("evolution","max_walltime") * 3600 ;
    /**********************************************************************************/
    int64_t regrid_every = grace::get_param<int64_t>("amr","regrid_every") ;  
    int64_t volume_output_every = grace::get_param<int64_t>("IO","volume_output_every") ;
    int64_t plane_surface_output_every = 
        grace::get_param<int64_t>("IO","plane_surface_output_every") ;
    int64_t sphere_surface_output_every = 
        grace::get_param<int64_t>("IO","sphere_surface_output_every") ;
    int64_t scalar_output_every =
        grace::get_param<int64_t>("IO","scalar_output_every") ;
    int64_t info_output_every =
        grace::get_param<int64_t>("IO","info_output_every") ;
    std::string tstep_mode = grace::get_param<std::string>("evolution","timestep_selection_mode") ;
    if ( tstep_mode == "manual" ) {
        grace::set_timestep(grace::get_param<double>("evolution","timestep")) ; 
    }
    if ( volume_output_every > 0 )
        grace::IO::write_cell_output(true,true,true) ;
    grace::IO::compute_reductions() ; 
    grace::IO::initialize_output_files() ; 
    if ( scalar_output_every > 0 )
        grace::IO::write_scalar_output() ;
    GRACE_INFO("Starting evolution.") ; 
    grace::IO::info_output() ;
    /**********************************************************************************/
    /*                           Evolution loop                                       */
    /**********************************************************************************/
    bool terminate = false ; 
    while( ! terminate ) 
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
	    grace::compute_auxiliary_quantities() ;
        }
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
        /**********************************************************************************/
        /* Save checkpoint if needed                                                      */
        /**********************************************************************************/
        if ( checkpoint_handler::get().need_checkpoint() ) {
            checkpoint_handler::get().save_checkpoint() ; 
        }
        /**********************************************************************************/
        /* Termination condition                                                          */
        /**********************************************************************************/
        if ( (grace::get_simulation_time() >= final_time) and (final_time > 0) ) {
            GRACE_INFO("Initiating termination sequence due to simulation time limit.") ;
            terminate = true ; 
        } else if ( (grace::get_iteration() >= final_iteration) and (final_iteration > 0)  ) {
            GRACE_INFO("Initiating termination sequence due to iteration limit.") ; 
            terminate = true ; 
        } else if ( (grace::get_total_runtime() >= max_walltime) and (max_walltime > 0) ) {
            GRACE_INFO("Initiating termination sequence due to walltime limit.") ;
            terminate = true ; 
        }
    }
    
    grace::grace_finalize() ; 
    return EXIT_SUCCESS ; 
}
/**********************************************************************************/
/**********************************************************************************/
