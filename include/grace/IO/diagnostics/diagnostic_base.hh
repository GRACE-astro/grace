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

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/utils/metric_utils.hh>
#include <vector> 

#include <Kokkos_Core.hpp>

namespace grace {

template < typename derived_t >
struct diagnostic_base_t {

    diagnostic_base_t(std::string const& diag_name)
    {
        auto sphere_names = get_param<std::vector<std::string>>(diag_name,"detector_names") ; 
        sphere_indices = get_param<std::vector<size_t>>(diag_name,"detector_indices") ;

        auto& sphere_list = grace::spherical_surface_manager::get() ; 
        for( auto const& n: sphere_names ) {
            auto idx = sphere_list.get_index(n);
            if ( idx < 0 ) {
                GRACE_WARN("Spherical detector {} not found", n) ; 
            } else {
                sphere_indices.push_back(idx); 
            }
        }
        std::sort(sphere_indices.begin(), sphere_indices.end());
        sphere_indices.erase(
            std::unique(sphere_indices.begin(), sphere_indices.end()),
            sphere_indices.end()
        );
    }

    void initialize_files() {
        auto& sphere_list = grace::spherical_surface_manager::get() ; 

        if( parallel::mpi_comm_rank() == 0 ) {
            auto& grace_runtime = grace::runtime::get() ; 
            static constexpr const size_t width = 20 ; 
            std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ; 
            for( int i=0; i < sphere_indices.size(); ++i ) {
                auto const& detector = sphere_list.get(sphere_indices[i]) ;
                auto name = detector.name ;
                for ( auto const& fname: flux_names) {
                    std::string const pfname = grace_runtime.scalar_io_basename() + fname + "_" + name + ".dat" ;
                    std::filesystem::path fname = bdir /  pfname ; 
                    std::ofstream outfile(fname.string(),std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;  
                }
            }

        }
    }


    //! Indices of variables that need to be interpolated 
    std::vector<int> var_interp_idx, aux_interp_idx ; 
    //! Indices of spheres where output will happen 
    std::vector<size_t> sphere_indices ;
    //! Fluxes 
    std::array<std::vector<double>, derived_t::n_fluxes> fluxes; 
    //! Flux names 
    std::array<std::string, derived_t:n_fluxes> flux_names 
} ; 


}

#endif /*GRACE_IO_DIAGNOSTICS_BASE_HH*/