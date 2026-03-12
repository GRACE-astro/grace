/**
 * @file adm_quantities.hh
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
#ifndef GRACE_IO_ADM_DIAGNOSTICS_HH
#define GRACE_IO_ADM_DIAGNOSTICS_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/metric_utils.hh>

#include <grace/utils/device_vector.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/config/config_parser.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>

namespace grace {


struct adm_diagnostics {

    adm_diagnostics() {
        auto sphere_names = get_param<std::vector<std::string>>("bh_diagnostics","detector_names") ; 
        sphere_indices = get_param<std::vector<size_t>>("bh_diagnostics","detector_indices") ;
        
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


    std::vector<int> var_interp_idx_h, aux_interp_idx_h ; 
    std::vector<size_t> sphere_indices ;
    std::vector<double> Madm;
} ; 


} /* namespace grace */

#endif /*GRACE_IO_ADM_DIAGNOSTICS_HH*/