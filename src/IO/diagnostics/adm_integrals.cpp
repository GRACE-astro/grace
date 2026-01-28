/**
 * @file adm_integrals.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
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

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/metric_utils.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/IO/diagnostics/adm_integrals.hh>

#include <array>
#include <vector>
#include <string> 

namespace grace {


std::vector<std::string> gw_integrals::flux_names = {"E_ADM", "Px_ADM", "Py_ADM", "Pz_ADM", "Jx_ADM", "Jy_ADM", "Jz_ADM"} ; 

std::array<double,gw_integrals::n_fluxes> 

gw_integrals::compute_local_fluxes(
    Kokkos::View<double**> ivals_d, 
    Kokkos::View<double**> ivals_aux_d, 
    spherical_surface_iface const& detector 
)
{
    GRACE_VERBOSE("Computing GW integrals on sphere {}", detector.name ) ; 

    auto npoints = detector.intersecting_points_h.size() ;
    GRACE_VERBOSE("We have {} points", npoints) ; 

    // initialize local flux array
    std::array<double,n_fluxes> flux_loc = {0.,0.,0.,0.} ; 

    // if no local points return 
    if (npoints == 0 ) return flux_loc ; 

    // copy to host 
    auto ivals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_aux_d);

    bool z_sym = get_param<bool>("amr","reflection_symmetries","z") ; 

    // fetch coord system 
    auto const& coord_system = grace::coordinate_system::get() ;

    // local reduction 
    for(int i=0; i<npoints; ++i) {
        auto ip = detector.intersecting_points_h[i] ; 

        cmplx_t psi{
            ivals(i,0), ivals(i,1)
        } ; 

        auto theta = detector.angles_h[ip][0] ; 
        auto phi   = detector.angles_h[ip][1] ; 

        auto Y2m2 = detail::Y2m2(theta,phi) ; 
        auto Y2m1 = detail::Y2m1(theta,phi) ; 
        auto Y20 = detail::Y20(theta,phi) ; 
        auto Y21 = detail::Y21(theta,phi) ; 
        auto Y22 = detail::Y22(theta,phi) ; 

        
        double const domega = detector.weights_h[ip] ; 

        cmplx_t psi2m2, psi2m1, psi20, psi21, psi22 ; 
        if ( ! z_sym ) {
            psi2m2 = psi * bar(Y2m2) ; 
            psi2m1 = psi * bar(Y2m1) ; 
            psi20 = psi * bar(Y20) ; 
            psi21 = psi * bar(Y21) ; 
            psi22 = psi * bar(Y22) ; 
        } else {
            psi2m2 = psi * bar(Y2m2) + bar(psi) * Y22; 
            psi2m1 = psi * bar(Y2m1) + bar(psi) * Y21; 
            psi20 = psi * bar(Y20) + bar(psi) * Y20 ; 
            psi21 = psi * bar(Y21) + bar(psi) * Y2m1; 
            psi22 = psi * bar(Y22) + bar(psi) * Y2m2; 
        }
        flux_loc[0] += domega * psi2m2.re ; 
        flux_loc[1] += domega * psi2m2.im ; 

        flux_loc[2] += domega * psi2m1.re ; 
        flux_loc[3] += domega * psi2m1.im ; 

        flux_loc[4] += domega * psi20.re ; 
        flux_loc[5] += domega * psi20.im ;

        flux_loc[6] += domega * psi21.re ; 
        flux_loc[7] += domega * psi21.im ; 

        flux_loc[8] += domega * psi22.re ; 
        flux_loc[9] += domega * psi22.im ; 
        
    }
    return flux_loc ;
}   


}