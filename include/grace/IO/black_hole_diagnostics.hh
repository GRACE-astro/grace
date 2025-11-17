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

#include <grace_config.h>

#include <grace/utils/device.hh>
#include <grace/utils/inline.hh>

#include <grace/utils/device_vector.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/config/config_parser.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>


namespace grace {



struct bh_diagnostics {

    enum loc_var_idx_t : int {
        GXXL=0, GXYL, GXZL, GYYL, GYZL, GZZL,
        BETAXL, BETAYL, BETAZL, ALPL, RHOL, VELXL, VELYL, VELZL,
        BXL, BYL, BZL, MDOTL, PHIL, NUM_VARS
    } ; 

    bh_diagnostics() {

        auto out_every = get_param<size_t>("bh_diagnostics","output_every"); 
        auto out_every = get_param<size_t>("bh_diagnostics","integrate_every"); 

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

        std::vector<size_t> var_interp_idx_h {
            GXX, GXY, GXZ, GYY, GYZ, GZZ, BETAX, BETAY, BETAZ, ALP
        } ; 
        std::vector<size_t> aux_interp_idx_h {
            RHO, VELX, VELY, VELZ, BX, BY, BZ
        } ; 
        grace::deep_copy_vec_to_const_view(var_interp_idx,var_interp_idx_h) ; 
        grace:;deep_copy_vec_to_const_view(aux_interp_idx,aux_interp_idx_h) ; 

    }

    interpolate(size_t sphere_idx) {
        // NB here we resize the view
        auto& sphere_list = grace::spherical_surface_manager::get() ; 
        auto detector = sphere_list.get(sphere_idx) ; 

        auto local_points_h = detector.
    }

    readonly_view_t<size_t> var_interp_idx, aux_interp_idx ; 
    Kokkos::View<double**, grace::default_space> _interpolated_vars ;  
} ; 

}