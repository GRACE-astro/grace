/**
 * @file black_hole_diagnostics.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-17
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
#ifndef GRACE_IO_DIAGNOSTICS_BASE_HH
#define GRACE_IO_DIAGNOSTICS_BASE_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/parallel/mpi_wrappers.hh>

#include <grace/config/config_parser.hh>

#include <grace/system/grace_runtime.hh>

#include <grace/utils/metric_utils.hh>

#include <vector> 

#include <Kokkos_Core.hpp>

namespace grace {

template < typename derived_t >
struct diagnostic_base_t {

    diagnostic_base_t(std::string const& diag_name) {
        auto names = get_param<std::vector<std::string>>(diag_name,"detector_names");
        std::vector<size_t> idxs ;

        auto& spheres = grace::spherical_surface_manager::get();

        for (auto const& n : names) {
            auto idx = spheres.get_index(n);
            if (idx < 0) {
                GRACE_WARN("Spherical detector {} not found", n);
            } else {
                idxs.push_back(idx);
            }
        }

        std::sort(idxs.begin(), idxs.end());
        idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());

        sphere_indices = std::move(idxs);
    }  



    void initialize_files() {
        if (parallel::mpi_comm_rank() != 0) return;
        auto& sphere_list = grace::spherical_surface_manager::get() ; 
        auto& grace_runtime = grace::runtime::get() ;
        std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;  
        static constexpr const size_t width = 20 ; 
        for( int i=0; i < sphere_indices.size(); ++i ) {
            auto const& detector = sphere_list.get(sphere_indices[i]) ;
            auto const& dname = detector.name ;
            for( int j=0; j<derived_t::n_fluxes; ++j) {
                auto const& flname = derived_t::flux_names[j] ;
                std::string pfname = grace_runtime.scalar_io_basename() + flname + "_" + dname + ".dat" ;
                std::filesystem::path fname = bdir / pfname ; 
                std::ofstream outfile(fname.string(),std::ios::app) ;
                outfile << std::fixed << std::setprecision(15) ; 
                outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;  
            }
        }
    }

    void write_fluxes() {
        if ( parallel::mpi_comm_rank() != 0 ) return ; 
        auto& sphere_list = grace::spherical_surface_manager::get() ; 
        auto& grace_runtime = grace::runtime::get() ; 
        std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ; 
        size_t const iter = grace_runtime.iteration() ; 
        double const time = grace_runtime.time()      ;

        for( int i=0; i < sphere_indices.size(); ++i ) {
            auto const& detector = sphere_list.get(sphere_indices[i]) ;
            auto const& dname = detector.name ;
            for( int j=0; j<derived_t::n_fluxes; ++j) {
                auto const& flname = derived_t::flux_names[j] ;
                std::string pfname = grace_runtime.scalar_io_basename() + flname + "_" + dname + ".dat" ;
                std::filesystem::path fname = bdir / pfname ; 
                std::ofstream outfile(fname.string(),std::ios::app) ;
                outfile << std::fixed << std::setprecision(15) ; 
                outfile << std::left << iter << '\t'
                        << std::left << time << '\t' 
                        << std::left << fluxes[i][j] << '\n' ; 
            }
        }
        
    }

    void compute_and_write() {
        // reset the fluxes 
        fluxes.clear();
        fluxes.resize(sphere_indices.size(), std::vector<double>(derived_t::n_fluxes));
        // compute 
        compute(); 
        // write to file 
        write_fluxes() ; 
    }

    protected:

    void compute() {

        auto& sphere_list = grace::spherical_surface_manager::get() ; 

        Kokkos::View<double**,grace::default_space> ivals("interp_vars",0,0) ;
        Kokkos::View<double**,grace::default_space> ivals_aux("interp_vars",0,0) ;

        for( int id=0; id < sphere_indices.size(); ++id ) {
            auto sphere_idx = sphere_indices[id];
            auto const& detector = sphere_list.get(sphere_idx) ;
            // first we interpolate 
            interpolate_on_sphere(
                detector, var_interp_idx, aux_interp_idx, ivals, ivals_aux
            ); 
            
            // compute local fluxes
            auto local_fluxes = static_cast<derived_t*>(this)->compute_local_fluxes(
                ivals, ivals_aux, detector
            ) ; 
            // aggregate results into global buffer
            parallel::mpi_allreduce(
                local_fluxes.data(),
                fluxes[id].data(),
                derived_t::n_fluxes,
                MPI_SUM
            );
        }

    }

    //! Indices of variables that need to be interpolated 
    std::vector<int> var_interp_idx, aux_interp_idx ; 
    //! Indices of spheres where output will happen 
    std::vector<size_t> sphere_indices ;
    //! Fluxes 
    std::vector<std::vector<double>> fluxes; 
} ; 


}

#endif /*GRACE_IO_DIAGNOSTICS_BASE_HH*/