/**
 * @file outflow_diagnostics.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief A collection of semi-random diagnostics 
 * @date 2026-01-15
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

#ifndef GRACE_IO_GRMHD_DIAGNOSTICS_HH
#define GRACE_IO_GRMHD_DIAGNOSTICS_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_indices.hh>

#include <grace/utils/metric_utils.hh>

#include <grace/utils/device_vector.hh>

#include <grace/system/grace_runtime.hh>
#include <grace/coordinates/coordinate_systems.hh>

#include <grace/config/config_parser.hh>
#include <grace/utils/reductions.hh>

#include <grace/physics/grmhd_helpers.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>

namespace grace {

struct grmhd_diagnostics {
    grmhd_diagnostics() {
        out_every = get_param<int>("mhd_diagnostics", "compute_other_diagnostics_every") ;
        disk_rho_thresh = get_param<double>("mhd_diagnostics", "disk_mass_rho_threshold") ; 
        auto& grace_runtime = grace::runtime::get() ; 
        std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;  
        std::string pfname = "grmhd_diagnostics.dat"; 
        fpath = bdir / pfname ; 
    }
    //**************************************************************************************************
    void compute_and_write() {
        auto& grace_runtime = grace::runtime::get() ; 
        size_t const iter = grace_runtime.iteration() ; 
        if ( (out_every > 0) && (iter%out_every == 0) ) {
            compute() ; 
            output() ; 
        }
    }
    //**************************************************************************************************
    void initialize_files() {
        static constexpr const size_t width = 20 ; 
        int proc = parallel::mpi_comm_rank() ; 
        if ( !std::filesystem::exists(fpath) and (proc == 0) ) {
            std::ofstream outfile(fpath.string());
            outfile << std::fixed << std::setprecision(15) ; 
            outfile << std::left << std::setw(width) << "Iteration" 
                    << std::left << std::setw(width) << "Time" 
                    << std::left << std::setw(width) << "disk_mass" << '\n' ;  
        }
        
        parallel::mpi_barrier() ; 
    }
    //**************************************************************************************************
    private: 
    //**************************************************************************************************
    void output() {
        int proc = parallel::mpi_comm_rank() ; 
        if ( proc == 0 ) { 
            auto& grace_runtime = grace::runtime::get() ; 
            size_t const iter = grace_runtime.iteration() ; 
            double const time = grace_runtime.time()      ;
            std::ofstream outfile(fpath.string(), std::ios::app) ;
            outfile << std::fixed << std::setprecision(15) ; 
            outfile << std::left << iter << '\t'
                << std::left << time << '\t' 
                << std::left << disk_mass << '\n' ;
            
        }
        parallel::mpi_barrier() ; 
    }
    //**************************************************************************************************
    void compute()
    {
        DECLARE_GRID_EXTENTS ; 

        using namespace grace ;
        using namespace Kokkos ; 

        auto& state = grace::variable_list::get().getstate() ; 
        auto& aux = grace::variable_list::get().getaux() ; 
        auto& dx = grace::variable_list::get().getspacings() ; 
        auto dc = coordinate_system::get().get_device_coord_system() ; 

        double thresh = disk_rho_thresh;

        double disk_mass_loc ; 

        MDRangePolicy<Rank<4>> policy(
            {ngz,ngz,ngz,0},
            {nx+ngz,ny+ngz,nz+ngz,nq}
        ) ; 
        parallel_reduce(
            GRACE_EXECUTION_TAG("DIAG", "compute_grmhd_diagnostics"),
            policy,
            KOKKOS_LAMBDA(int const i, int const j, int const k, int q, double& intloc) {
                metric_array_t metric ; 
                FILL_METRIC_ARRAY(metric,state,q,VEC(i,j,k)) ;

                grmhd_prims_array_t prims ;     
                FILL_PRIMS_ARRAY_ZVEC(
                    prims, aux, q, i,j,k
                ) ; 

                double xyz[3] ; 
                dc.get_physical_coordinates(i,j,k,q,xyz) ; 


                double densL = state(i,j,k,DENS_,q) ; 

                double cell_vol = dx(0,q) * dx(1,q) * dx(2,q) ; 
                if ( prims[RHOL] > thresh ) {
                    intloc += densL * cell_vol ; 
                }
            }, disk_mass_loc
        ) ;

        MPI_Allreduce(
            &disk_mass_loc,             // sendbuf
            &disk_mass,                 // recvbuf
            1,                          // count
            MPI_DOUBLE,                 // datatype
            MPI_SUM,                    // op
            MPI_COMM_WORLD              // comm
        );
    }
    //**************************************************************************************************
    double disk_rho_thresh      ; //!< Only integrate for rho > thresh 
    double disk_mass            ; //!< Disk mass diagnostic 
    int out_every               ; //!< Output frequency 
    std::filesystem::path fpath ; //!< Output file path
    //**************************************************************************************************
} ; 


}
#endif 