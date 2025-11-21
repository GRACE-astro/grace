/**
 * @file black_hole_diagnostics.cpp
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

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/metric_utils.hh>

#include <grace/IO/spherical_surfaces.hh>

#include <grace/IO/diagnostics/black_hole_diagnostics.hh>

#include <array>
#include <vector>
#include <string> 

namespace grace {

std::vector<std::string> bh_diagnostics::flux_names = {"Mdot", "Edot", "Ldot", "Phi"} ; 

std::array<double,bh_diagnostics::n_fluxes> 
bh_diagnostics::compute_local_fluxes(
    Kokkos::View<double**> ivals_d, 
    spherical_surface_iface<3> const& detector 
) 
{
    GRACE_VERBOSE("Computing BH diagnostics on sphere {}", detector.name ) ; 

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

        double gxx{ivals(i,loc_var_idx_t::GXXL)}
                , gxy{ivals(i,loc_var_idx_t::GXYL)}
                , gxz{ivals(i,loc_var_idx_t::GXZL)}
                , gyy{ivals(i,loc_var_idx_t::GYYL)}
                , gyz{ivals(i,loc_var_idx_t::GYZL)}
                , gzz{ivals(i,loc_var_idx_t::GZZL)}
                , betax{ivals(i,loc_var_idx_t::BETAXL)}
                , betay{ivals(i,loc_var_idx_t::BETAYL)}
                , betaz{ivals(i,loc_var_idx_t::BETAZL)}
                , alp{ivals(i,loc_var_idx_t::ALPL)}
                , rho{ivals(i,loc_var_idx_t::RHOL)}
                , eps{ivals(i,loc_var_idx_t::EPSL)}
                , press{ivals(i,loc_var_idx_t::PRESSL)}
                , vx{ivals(i,loc_var_idx_t::VELXL)}
                , vy{ivals(i,loc_var_idx_t::VELYL)}
                , vz{ivals(i,loc_var_idx_t::VELZL)}
                , bx{ivals(i,loc_var_idx_t::BXL)}
                , by{ivals(i,loc_var_idx_t::BYL)}
                , bz{ivals(i,loc_var_idx_t::BZL)} ; 
        auto report_nan = [&](char const* name, double v) {
                if (std::isnan(v)) {
                    GRACE_VERBOSE("NaN detected in {} at i={}", name, i);
                }
            };
        {
            

            report_nan("gxx", gxx);
            report_nan("gxy", gxy);
            report_nan("gxz", gxz);
            report_nan("gyy", gyy);
            report_nan("gyz", gyz);
            report_nan("gzz", gzz);
            report_nan("betax", betax);
            report_nan("betay", betay);
            report_nan("betaz", betaz);
            report_nan("alp", alp);
            report_nan("rho", rho);
            report_nan("eps", eps);
            report_nan("press", press);
            report_nan("vx", vx);
            report_nan("vy", vy);
            report_nan("vz", vz);
            report_nan("bx", bx);
            report_nan("by", by);
            report_nan("bz", bz);
        }

        metric_array_t metric{
            {gxx,gxy,gxz,gyy,gyz,gzz}, {betax,betay,betaz}, alp
        } ;

        auto r = detector.radius ; 
        auto theta = detector.angles_h[ip][0] ; 
        auto phi   = detector.angles_h[ip][1] ; 
        auto x = detector.points_h[ip].second[0] ; 
        auto y = detector.points_h[ip].second[1] ; 
        auto z = detector.points_h[ip].second[2] ; 

        double const one_over_alp = 1./metric.alp() ;
        report_nan("one_over_alp",one_over_alp) ; 
        std::array<double,3> const vN {
            one_over_alp * ( vx + metric.beta(0) )
            , one_over_alp * ( vy + metric.beta(1) )
            , one_over_alp * ( vz + metric.beta(2) )
        } ; 
        double const W = 1./Kokkos::sqrt(1-metric.square_vec(vN)) ; 
        if (std::isnan(W)) {
            GRACE_VERBOSE(
                "NaN in W at point {} on detector {}:\n"
                "  Primitives:\n"
                "    g = [{}, {}, {}, {}, {}, {}]\n"
                "    beta = ({}, {}, {})\n"
                "    alp  = {}\n"
                "    rho  = {}, eps = {}, press = {}\n"
                "    v = ({}, {}, {})\n"
                "    B = ({}, {}, {})\n"
                "  Computed quantities:\n"
                "    vN = ({}, {}, {})   (Eulerian 3-velocity)\n"
                "    vN^2 = {}\n"
                "    1 - vN^2 = {}\n"
                "    one_over_alp = {}\n"
                "  Coordinates:\n"
                "    x = {}, y = {}, z = {}\n",
                i, detector.name,
                gxx, gxy, gxz, gyy, gyz, gzz,
                betax, betay, betaz,
                alp,
                rho, eps, press,
                vx, vy, vz,
                bx, by, bz,
                vN[0], vN[1], vN[2],
                metric.square_vec(vN),
                1.0 - metric.square_vec(vN),
                one_over_alp,
                x, y, z
            );
        }
        report_nan("W",W) ; 
        double const u0 = one_over_alp * W ; 
        std::array<double,4> uU{{ u0, vx * u0, vy * u0, vz * u0 }};
        auto uD = metric.lower_4vec(uU) ; 

        double b0 = uD[1] * bx + uD[2] * by + uD[3] * bz ; 
        double b1 = (bx+b0*uU[1])/u0;
        double b2 = (by+b0*uU[2])/u0;
        double b3 = (bz+b0*uU[3])/u0;
        report_nan("bt",b0) ; 
        report_nan("b1",b1) ; 
        report_nan("b2",b2) ; 
        report_nan("b3",b3) ; 

        auto bD = metric.lower_4vec({b0,b1,b2,b3}) ; 

        double bsq = b0 * bD[0] + b1 * bD[1] + b2 * bD[2] + b3*bD[3] ; 
        report_nan("b2",bsq) ; 
        double sth = sin(theta) ; 
        double sph = sin(phi)   ;
        double cph = cos(phi)   ;

        double ur, br, u_phi, b_phi, sqrtmdet ;
        // transform sph to cart 
        if ( coord_system.get_is_cks() ) {
            double a = coord_system.get_bh_spin() ; 
            double a2 = SQR(a) ; 
            double rad2 = SQR(x) + SQR(y) + SQR(z) ; 
            double r2 = SQR(r) ; 

            double drdx = r*x/(2.0*r2 - rad2 + a2) ; 
            double drdy = r*y/(2.0*r2 - rad2 + a2) ; 
            double drdz = (r*z + a2 * z / r) / (2.0*r2 - rad2 + a2) ;      
            
            ur = drdx * uU[1] + drdy * uU[2] + drdz * uU[3] ; 
            br = drdx * b1 + drdy * b2 + drdz * b3 ;

            u_phi = (-r*sph-a*cph)*sth*uD[1] + (r*cph-a*sph)*sth*uD[2] ; 
            b_phi = (-r*sph-a*cph)*sth*bD[1] + (r*cph-a*sph)*sth*bD[2] ; 

            sqrtmdet = (r2 * SQR(a*cos(theta))) ; 

        } else {
            double drdx = x/r ; 
            double drdy = y/r ; 
            double drdz = z/r ;      
        
            ur = drdx * uU[1] + drdy * uU[2] + drdz * uU[3] ; 
            br = drdx * b1 + drdy * b2 + drdz * b3 ;

            u_phi = (-r*sph)*sth*uD[1] + (r*cph)*sth*uD[2] ; 
            b_phi = (-r*sph)*sth*bD[1] + (r*cph)*sth*bD[2] ;
            sqrtmdet = metric.sqrtg() ; 
        }
        double rhoh = rho + rho * eps + press ; 

        double tr_0 = (rhoh + bsq) * ur * uD[0] - br * bD[0] ; 
        
        double tr_3 = (rhoh + bsq) * ur * u_phi - br * b_phi ; 

        double phi_l = 0.5 * fabs(br*u0 - b0*ur) ; 

        double const domega = detector.weights_h[ip] ; 
        report_nan("domega",domega) ; 
        report_nan("sqrtmdet",sqrtmdet) ; 
        // accretion rate
        flux_loc[0] += - domega * sqrtmdet * rho * ur ; 
        // energy flux 
        flux_loc[1] += - domega * sqrtmdet * tr_0 ; 
        // angular momentum flux 
        flux_loc[2] += domega * sqrtmdet * tr_3 ; 
        // magnetic flux 
        flux_loc[3] += domega * sqrtmdet * phi_l ; 

    }

    return flux_loc; 
}


}