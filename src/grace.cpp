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
#include <H5public.h>
#include <hdf5.h>
void output_fluxes() {
    DECLARE_GRID_EXTENTS ; 
    auto fluxes = grace::variable_list::get().getfluxesarray() ; 
    auto fluxes_h = Kokkos::create_mirror_view(fluxes) ; 
    auto nvars_hrsc = grace::variables::get_n_hrsc(); 
    // dimensions: nx+1, ny+1, nz+1, nvars_hrsc, 3, nq 
    Kokkos::deep_copy(fluxes_h, fluxes);

    std::string const filename = "fluxes.h5" ; 

    // dimensions of the array
    const hsize_t dims[6] = {
        static_cast<hsize_t>(nx+1+2*ngz),
        static_cast<hsize_t>(ny+1+2*ngz),
        static_cast<hsize_t>(nz+1+2*ngz),
        static_cast<hsize_t>(nvars_hrsc),
        static_cast<hsize_t>(3),
        static_cast<hsize_t>(nq)
    };

    // total number of elements
    const size_t total_elems =
        static_cast<size_t>( (nx+1+2*ngz) * (nx+1+2*ngz) * (nx+1+2*ngz) * nvars_hrsc * 3 * nq );

    // allocate a contiguous buffer for HDF5 write
    std::vector<double> buffer(total_elems);

    // flatten the Kokkos view into the buffer
    size_t idx = 0;
    for (int i = 0; i < nx+2*ngz+1; ++i) {
        for (int j = 0; j < ny+2*ngz+1; ++j) {
            for (int k = 0; k < nz+2*ngz+1; ++k) {
                for (int v = 0; v < nvars_hrsc; ++v) {
                    for (int dir = 0; dir < 3; ++dir) {
                        for (int q = 0; q < nq; ++q) {
                            buffer[idx++] = fluxes_h(i,j,k,v,dir,q);
                        }
                    }
                }
            }
        }
    }

    // --- HDF5 writing ---
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error: could not create HDF5 file " << filename << std::endl;
        return;
    }

    hid_t dataspace_id = H5Screate_simple(6, dims, NULL);
    hid_t dataset_id   = H5Dcreate(file_id,
                                   "fluxes",
                                   H5T_NATIVE_DOUBLE,
                                   dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // write the data
    herr_t status = H5Dwrite(dataset_id,
                             H5T_NATIVE_DOUBLE,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             buffer.data());
    if (status < 0) {
        std::cerr << "Error writing dataset to " << filename << std::endl;
    }

    // cleanup
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);

    std::cout << "Fluxes written to " << filename << " (" << total_elems << " elements)" << std::endl;
}

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
    /**********************************************************************************/
    if ( grace::get_param<bool>("IO","do_initial_output") ) {
        GRACE_INFO("Performing initial output") ; 
        grace::IO::write_cell_output(volume_output_every>0,plane_surface_output_every>0,sphere_surface_output_every>0) ;
    }
        
    grace::IO::compute_reductions() ; 
    grace::IO::initialize_output_files() ; 
    grace::IO::write_scalar_output() ;
    GRACE_INFO("Starting evolution.") ; 
    grace::IO::info_output() ;
    /**********************************************************************************/
    /**********************************************************************************/
    
    std::string tstep_mode = grace::get_param<std::string>("evolution","timestep_selection_mode") ;
    if ( tstep_mode == "manual" ) {
        grace::set_timestep(grace::get_param<double>("evolution","timestep")) ; 
    }
    /**********************************************************************************/
    /*                           Evolution loop                                       */
    /**********************************************************************************/
    while( ! grace::check_termination_condition() ) 
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
    }
    
    grace::grace_finalize() ; 
    return EXIT_SUCCESS ; 
}
/**********************************************************************************/
/**********************************************************************************/
