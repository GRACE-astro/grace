/**
 * @file spherical_surfaces.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
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

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/device_vector.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/config/config_parser.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>
#include <string>

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
                spherical_surface_t<healpix_sampler_t,no_tracking_policy_t>(name,r,c,res)
            ); 
        } else {
            ERROR("Invalid tracking requested for spherical surface") ; 
        }
    } else if ( sampling == "uniform" ) {
        if ( tracking == "none") {
            return std::make_unique<spherical_surface_t<uniform_sampler_t,no_tracking_policy_t>>(
                spherical_surface_t<uniform_sampler_t,no_tracking_policy_t>(name,r,c,res)
            ); 
        } else {
            ERROR("Invalid tracking requested for spherical surface") ;
        }
    } else {
        ERROR("Invalid sampling requested for spherical surface") ; 
    }
}

spherical_surface_manager_impl_t::spherical_surface_manager_impl_t() {

    GRACE_VERBOSE("Into spheres constructor.") ; 
    auto n_spheres = get_param<size_t>("spherical_surfaces","n_surfaces") ; 
    if (n_spheres>0){
        auto params = grace::config_parser::get().get() ; 
        auto sphere_pars = params["spherical_surfaces"]["spherical_detectors"];
        for (int i =0; i<n_spheres; ++i) {
            double r,xc,yc,zc ; 
            int res ; 
            std::string name, tracking, sampling ;
            try{
                r = sphere_pars[i]["radius"].as<double>() ; 
                xc = sphere_pars[i]["x_c"].as<double>() ; 
                yc = sphere_pars[i]["y_c"].as<double>() ; 
                zc = sphere_pars[i]["z_c"].as<double>() ; 
                name = sphere_pars[i]["name"].as<std::string>() ; 
                tracking = sphere_pars[i]["tracking"].as<std::string>() ; 
                sampling = sphere_pars[i]["sampling_policy"].as<std::string>() ; 
                res = sphere_pars[i]["resolution"].as<int>() ; 
            } catch (const YAML::BadConversion&) {
                ERROR("Could not construct sphere[" << i << "] due to parameter reading error.") ; 
            }
            detectors.push_back(make_surface(name,r,{{xc,yc,zc}},res,tracking,sampling));
            name_map[name] = detectors.size() - 1; // store a mapping name -> idx 
        }
    }
    GRACE_VERBOSE("Constructed spheres.") ; 
}


void interpolate_on_sphere( spherical_surface_iface const& surf
                       , std::vector<int> const& var_idx_h 
                       , std::vector<int> const& aux_idx_h 
                       , Kokkos::View<double**,grace::default_space>& out_vars 
                       , Kokkos::View<double**,grace::default_space>& out_aux  )
{
    GRACE_VERBOSE("Into sphere interpolation onto {}", surf.name) ; 
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    using namespace Kokkos ; 

    auto& aux = variable_list::get().getaux() ; 
    auto& state = variable_list::get().getstate() ; 

    surf.interpolator.interpolate(
        state, var_idx_h, out_vars
    ) ; 

    surf.interpolator.interpolate(
        aux, aux_idx_h, out_aux
    ) ;
}

}


