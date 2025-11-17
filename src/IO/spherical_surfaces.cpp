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

#include <grace/utils/device.hh>
#include <grace/utils/inline.hh>

#include <grace/utils/device_vector.hh>

#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/config/config_parser.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <memory>

#define LAGRANGE_INTERP_ORDER 3

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
            return std::make_unique<spherical_surface_t<healpix_sampler_t,no_tracking_policy_t,LAGRANGE_INTERP_ORDER>>(
                spherical_surface_t<no_tracking_policy_t,healpix_sampler_t,LAGRANGE_INTERP_ORDER>(name,r,c,res)
            ); 
        } else {
            ERROR("Invalid tracking requested for spherical surface") ; 
        }
    } else if ( sampling == "uniform" ) {
        if ( tracking == "none") {
            return std::make_unique<spherical_surface_t<healpix_sampler_t,no_tracking_policy_t,LAGRANGE_INTERP_ORDER>>(
                spherical_surface_t<no_tracking_policy_t,uniform_sampler_t,LAGRANGE_INTERP_ORDER>(name,r,c,res)
            ); 
        } else {
            ERROR("Invalid tracking requested for spherical surface") ;
        }
    } else {
        ERROR("Invalid sampling requested for spherical surface") ; 
    }
}

spherical_surface_manager_impl_t::spherical_surface_manager_impl_t() {


    auto n_spheres = get_param<size_t>("spherical_surfaces","n_surfaces") ; 


    #define GET_SPHERE_PARAMETERS(n) \
    std::ostringstream oss ; \
    oss << "spherical_surface_" << n ; \
    auto const r = get_param<double>("spherical_surfaces",oss.str(),"radius") ; \
    auto const xc = get_param<double>("spherical_surfaces",oss.str(),"x_c") ; \
    auto const yc = get_param<double>("spherical_surfaces",oss.str(),"y_c") ; \
    auto const zc = get_param<double>("spherical_surfaces",oss.str(),"z_c") ; \
    auto const name =  get_param<std::string>("spherical_surfaces",oss.str(),"name") ; \
    auto const res = get_param<size_t>("spherical_surfaces",oss.str(),"resolution") ; \
    auto const tracking = get_param<std::string>("spherical_surfaces",oss.str(),"tracking") ;\
    auto const sampling = get_param<std::string>("spherical_surfaces",oss.str(),"sampling") 
    for (int i =0; i<n_spheres; ++i) {
        GET_SPHERE_PARAMETERS(i);
        detectors.push_back(make_surface(name,r,{{x_c,y_c,z_c}},res,tracking,sampling));
        name_map[name] = detectors.size() - 1; // store a mapping name -> idx 
    }

}

}