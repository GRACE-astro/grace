/**
 * @file black_hole_diagnostics.cpp
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

#include <grace/IO/diagnostics/gw_integrals.hh>

#include <array>
#include <vector>
#include <string> 

namespace grace {

namespace detail {

std::array<double,2> GRACE_ALWAYS_INLINE
Y2m2(double theta, double phi) 
{
    double const K = sqrt(5./(64.*M_PI)) * SQR((1.-cos(theta))) ; 
    return {K*cos(2*phi), -K*sin(2*phi)} ; 
}

std::array<double,2> GRACE_ALWAYS_INLINE
Y2m1(double theta, double phi) 
{
    double const K = - sqrt(5./(16.*M_PI)) * sin(theta)*(1.-cos(theta)) ; 
    return {K*cos(phi), -K*sin(phi)} ; 
}

std::array<double,2> GRACE_ALWAYS_INLINE
Y20(double theta, double phi) 
{
    double const K = sqrt(15./(32.*M_PI)) * SQR((sin(theta))) ; 
    return {K, 0} ; 
}

std::array<double,2> GRACE_ALWAYS_INLINE
Y21(double theta, double phi) 
{
    double const K = - sqrt(5./(16.*M_PI)) * sin(theta) * (1.+cos(theta)) ; 
    return {K*cos(phi), K*sin(phi)} ; 
}

std::array<double,2> GRACE_ALWAYS_INLINE
Y22(double theta, double phi) 
{
    double const K = sqrt(5./(64.*M_PI)) * SQR((1.+cos(theta))) ; 
    return {K*cos(2*phi), K*sin(2*phi)} ; 
}

}

std::vector<std::string> gw_integrals::flux_names = {"Psi2m2_re", "Psi2m2_im", "Psi2m1_re", "Psi2m1_im", "Psi20_re", "Psi20_im", "Psi21_re", "Psi21_im", "Psi22_im", "Psi22_re"} ; 

std::array<double,gw_integrals::n_fluxes> 

gw_integrals::compute_local_fluxes(
    Kokkos::View<double**> ivals_d, 
    spherical_surface_iface<3> const& detector 
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
    auto ivals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_d);


    // fetch coord system 
    auto const& coord_system = grace::coordinate_system::get() ;

    // local reduction 
    for(int i=0; i<npoints; ++i) {
        auto ip = detector.intersecting_points_h[i] ; 

        double psi4Re{ivals(i,0)}, psi4Im{ivals(i,1)} ; 

        auto theta = detector.angles_h[ip][0] ; 
        auto phi   = detector.angles_h[ip][1] ; 

        auto Y2m2 = detail::Y2m2(theta,phi) ; 
        auto Y2m1 = detail::Y2m1(theta,phi) ; 
        auto Y20 = detail::Y20(theta,phi) ; 
        auto Y21 = detail::Y21(theta,phi) ; 
        auto Y22 = detail::Y22(theta,phi) ; 


        double const domega = detector.weights_h[ip] ; 

        flux_loc[0] += domega * ( Y2m2[0] * psi4Re + Y2m2[1] * psi4Im ) ; 
        flux_loc[1] += domega * ( Y2m2[0] * psi4Im - Y2m2[1] * psi4Re ) ; 

        flux_loc[2] += domega * ( Y2m1[0] * psi4Re + Y2m1[1] * psi4Im ) ; 
        flux_loc[3] += domega * ( Y2m1[0] * psi4Im - Y2m1[1] * psi4Re ) ; 

        flux_loc[4] += domega * ( Y20[0] * psi4Re + Y20[1] * psi4Im ) ; 
        flux_loc[5] += domega * ( Y20[0] * psi4Im - Y20[1] * psi4Re ) ; 

        flux_loc[6] += domega * ( Y21[0] * psi4Re + Y21[1] * psi4Im ) ; 
        flux_loc[7] += domega * ( Y21[0] * psi4Im - Y21[1] * psi4Re ) ; 

        flux_loc[8] += domega * ( Y22[0] * psi4Re + Y22[1] * psi4Im ) ; 
        flux_loc[9] += domega * ( Y22[0] * psi4Im - Y22[1] * psi4Re ) ; 
    }
    return flux_loc ;
}   


}