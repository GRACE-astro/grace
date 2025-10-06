/**
 * @file spherical_surfaces.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-10-03
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

std::unique_ptr<spherical_surface_iface> 
make_surface(
    std::string const& name,
    double r, std::array<double,3> const& c, size_t res,
    std::string const& tracking, 
    std::string const& sampling
) 
{
    if ( sampling == "healpix" ) {
        if ( tracking == "none" ) {
            return std::make_unique<spherical_surface_t<healpix_sampler_t,no_tracking_policy_t>>(
                spherical_surface_t<no_tracking_policy_t,healpix_sampler_t>(name,r,c,res)
            ); 
        } else {
            ERROR("Invalid tracking requested for spherical surface") ; 
        }
    } else {
        ERROR("Invalid sampling requested for spherical surface") ; 
    }
}

spherical_surface_manager_impl_t::spherical_surface_manager_impl_t() {

    // TODO check YAML copy policy 
    auto param_block = grace::get_param<YAML::Node>(
        "IO", "spherical_detectors"
    ) ; 

    for (auto const& det : param_block) {
        // what we need:
        // radius 
        // center 
        // name 
        // tracking type 
        // sampling policy 
        auto const r = det["radius"].as<double>() ; 
        auto const c = det["center"].as<std::array<double,3>() ; 
        auto const n = det["name"].as<std::string>() ;
        auto const res = det["res"].as<size_t>() ; 

        auto const tracking = det["tracking"].as<std::string>() ; 
        auto const sampling = det["sampling"].as<std::string>() ; 

        detectors.push_back(make_surface(n,r,c,res,tracking,sampling));
        name_map[n] = detectors.size() - 1; // store a mapping name -> idx 
    }

}

}