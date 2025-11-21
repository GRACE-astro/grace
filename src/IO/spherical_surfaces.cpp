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

#define LAGRANGE_INTERP_ORDER 3

namespace grace {

std::unique_ptr<spherical_surface_iface<LAGRANGE_INTERP_ORDER>> 
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
                spherical_surface_t<healpix_sampler_t,no_tracking_policy_t,LAGRANGE_INTERP_ORDER>(name,r,c,res)
            ); 
        } else {
            ERROR("Invalid tracking requested for spherical surface") ; 
        }
    } else if ( sampling == "uniform" ) {
        if ( tracking == "none") {
            return std::make_unique<spherical_surface_t<uniform_sampler_t,no_tracking_policy_t,LAGRANGE_INTERP_ORDER>>(
                spherical_surface_t<uniform_sampler_t,no_tracking_policy_t,LAGRANGE_INTERP_ORDER>(name,r,c,res)
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
    auto const sampling = get_param<std::string>("spherical_surfaces",oss.str(),"sampling_policy") 
    for (int i =0; i<n_spheres; ++i) {
        
        GET_SPHERE_PARAMETERS(i);
        detectors.push_back(make_surface(name,r,{{xc,yc,zc}},res,tracking,sampling));
        name_map[name] = detectors.size() - 1; // store a mapping name -> idx 
    }
    GRACE_VERBOSE("Constructed spheres.") ; 
}


void interpolate_on_sphere( spherical_surface_iface<3> const& surf
                       , std::vector<int> const& var_idx_h 
                       , std::vector<int> const& aux_idx_h 
                       , Kokkos::View<double**,grace::default_space>& out )
{
    GRACE_VERBOSE("Into sphere interpolation onto {}", surf.name) ; 
    DECLARE_GRID_EXTENTS ; 
    using namespace grace ; 
    using namespace Kokkos ; 

    auto& aux = variable_list::get().getaux() ; 
    auto& state = variable_list::get().getstate() ; 

    auto nvars = var_idx_h.size() ; 
    auto naux  = aux_idx_h.size() ; 

    readonly_view_t<int> var_idx, aux_idx ; 
    deep_copy_vec_to_const_view(var_idx,var_idx_h) ; 
    deep_copy_vec_to_const_view(aux_idx,aux_idx_h) ; 

    auto n_loc_points = surf.intersecting_points_h.size() ; 
    Kokkos::realloc(out, n_loc_points, nvars + naux) ; 

    auto& cell_descs = surf.intersected_cells; 
    auto& int_weights = surf.interp_weights ; 
    auto& int_bias = surf.interp_stencils ; 
    
    MDRangePolicy<Rank<2>> policy_vars({0,0},{n_loc_points, nvars}) ;
    parallel_for(
        GRACE_EXECUTION_TAG("IO", "interp_to_sphere"),
        policy_vars,
        KOKKOS_LAMBDA (int const& ip, int const& iv) {
            auto const& cell = cell_descs(ip);
            auto q = cell.q ; 
            auto w = int_weights(ip) ; 
            auto u = subview(state, ALL(), ALL(), ALL(), var_idx(iv), q) ; 
            int bx = int_bias(ip,0); 
            int by = int_bias(ip,1);
            int bz = int_bias(ip,2);

            double val{0};
            for( int i=0; i<4; ++i) {
                for( int j=0; j<4; ++j) {
                    for( int k=0; k<4; ++k) {
                        int io = i - 2 + bx ; 
                        int jo = j - 2 + by ; 
                        int ko = k - 2 + bz ; 
                        int ic = static_cast<int>(ngz+cell.i) + io ; 
                        int jc = static_cast<int>(ngz+cell.j) + jo ; 
                        int kc = static_cast<int>(ngz+cell.k) + ko ; 
                        val += w.w[i][0] * w.w[j][1] * w.w[k][2] * u(ic,jc,kc) ;
                    }
                }
            }
            out(ip,iv) = val ;
        }   
    ) ; 

    MDRangePolicy<Rank<2>> policy_aux({0,0},{n_loc_points, naux}) ;
    parallel_for(
        GRACE_EXECUTION_TAG("IO", "interp_to_sphere"),
        policy_aux,
        KOKKOS_LAMBDA (int const& ip, int const& iv) {
            auto const& cell = cell_descs(ip);
            auto q = cell.q ; 
            auto w = int_weights(ip) ; 
            auto u = subview(aux, ALL(), ALL(), ALL(), aux_idx(iv), q) ; 
            auto bx = int_bias(ip,0); 
            auto by = int_bias(ip,1);
            auto bz = int_bias(ip,2);

            double val{0};
            for( int i=0; i<4; ++i) {
                for( int j=0; j<4; ++j) {
                    for( int k=0; k<4; ++k) {
                        int io = i - 2 + bx ; 
                        int jo = j - 2 + by ; 
                        int ko = k - 2 + bz ; 
                        int ic = static_cast<int>(ngz+cell.i) + io ; 
                        int jc = static_cast<int>(ngz+cell.j) + jo ; 
                        int kc = static_cast<int>(ngz+cell.k) + ko ; 
                        val += w.w[i][0] * w.w[j][1] * w.w[k][2] * u(ic,jc,kc) ;
                    }
                }
            }

            out(ip,nvars+iv) = val ;
        }   
    ) ; 
}

}


