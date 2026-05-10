/**
 * @file outflow_diagnostics.cpp
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

#include <grace/IO/diagnostics/outflow_diagnostics.hh>

#include <grace/utils/reductions.hh>


#include <array>
#include <vector>
#include <string> 

namespace grace {

std::vector<std::string> outflows::flux_names = {"Mdot_unbound_geo", "Mdot_unbound_bern", "Mdot_tot"} ; 

#ifdef GRACE_ENABLE_M1
#if defined(M1_NU_THREESPECIES)
std::vector<std::string> m1_outflows::flux_names = {"Lrad_nue", "Lrad_nuebar", "Lrad_nux"};
#elif defined(M1_NU_FIVESPECIES)
std::vector<std::string> m1_outflows::flux_names = {"Lrad_nue", "Lrad_nuebar", "Lrad_numu", "Lrad_numubar", "Lrad_nux"};
#else
std::vector<std::string> m1_outflows::flux_names = {"Lrad_nu1"};
#endif
#endif 

std::array<double,outflows::n_fluxes> 
outflows::compute_local_fluxes(
    Kokkos::View<double**> ivals_d, 
    Kokkos::View<double**> ivals_aux_d, 
    spherical_surface_iface const& detector 
) 
{
    GRACE_VERBOSE("Computing outflows on sphere {}", detector.name ) ; 

    auto npoints = detector.intersecting_points_h.size() ;
    GRACE_VERBOSE("We have {} points", npoints) ; 

    // initialize local flux array
    std::array<double,n_fluxes> flux_loc = {0.,0.,0.} ; 

    // if no local points return 
    if (npoints == 0 ) return flux_loc ; 


    // copy to host 
    auto ivals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_d);
    auto ivals_aux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_aux_d);


    // fetch coord system 
    auto const& coord_system = grace::coordinate_system::get() ;

    // local reduction 
    for(int i=0; i<npoints; ++i) {

        auto ip = detector.intersecting_points_h[i] ; 
        #ifndef GRACE_ENABLE_Z4C_METRIC
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
                , rho{ivals_aux(i,loc_aux_idx_t::RHOL)}
                , eps{ivals_aux(i,loc_aux_idx_t::EPSL)}
                , press{ivals_aux(i,loc_aux_idx_t::PRESSL)}
                , zx{ivals_aux(i,loc_aux_idx_t::ZXL)}
                , zy{ivals_aux(i,loc_aux_idx_t::ZYL)}
                , zz{ivals_aux(i,loc_aux_idx_t::ZZL)} ;
        #else 
        double gxx{ivals(i,loc_var_idx_t::GTXXL)}
                , gxy{ivals(i,loc_var_idx_t::GTXYL)}
                , gxz{ivals(i,loc_var_idx_t::GTXZL)}
                , gyy{ivals(i,loc_var_idx_t::GTYYL)}
                , gyz{ivals(i,loc_var_idx_t::GTYZL)}
                , gzz{ivals(i,loc_var_idx_t::GTZZL)}
                , chi{ivals(i,loc_var_idx_t::CHIL)}
                , betax{ivals(i,loc_var_idx_t::BETAXL)}
                , betay{ivals(i,loc_var_idx_t::BETAYL)}
                , betaz{ivals(i,loc_var_idx_t::BETAZL)}
                , alp{ivals(i,loc_var_idx_t::ALPL)}
                , rho{ivals_aux(i,loc_aux_idx_t::RHOL)}
                , eps{ivals_aux(i,loc_aux_idx_t::EPSL)}
                , press{ivals_aux(i,loc_aux_idx_t::PRESSL)}
                , zx{ivals_aux(i,loc_aux_idx_t::ZXL)}
                , zy{ivals_aux(i,loc_aux_idx_t::ZYL)}
                , zz{ivals_aux(i,loc_aux_idx_t::ZZL)} ;
        #endif 
        #ifndef GRACE_ENABLE_Z4C_METRIC
        metric_array_t metric{
            {gxx,gxy,gxz,gyy,gyz,gzz}, {betax,betay,betaz}, alp
        } ;
        #else 
        metric_array_t metric{
            {gxx,gxy,gxz,gyy,gyz,gzz}, chi, {betax,betay,betaz}, alp
        } ;
        #endif 
        auto r = detector.radius ; 
        auto theta = detector.angles_h[ip][0] ; 
        auto phi   = detector.angles_h[ip][1] ; 
        auto x = detector.points_h[ip].second[0] ; 
        auto y = detector.points_h[ip].second[1] ; 
        auto z = detector.points_h[ip].second[2] ; 

        double const one_over_alp = 1./metric.alp() ;

        // W^2 = 1 + z^2
        double const W = Kokkos::sqrt(1+metric.square_vec({zx,zy,zz})) ; 
        // \tilde{v}^i =  \alpha v^i - beta^i 
        // but later we need W vtilde so we omit the W at 
        // the denominator 
        double const vtx = (alp) * zx - W * metric.beta(0); 
        double const vty = (alp) * zy - W * metric.beta(1); 
        double const vtz = (alp) * zz - W * metric.beta(2); 

        // vt^r 
        double vtr = vtx * x/r + vty * y/r + vtz * z/r ; 

        // mass flux = sqrtg D \tilde{v}^r
        double const mass_flux = metric.sqrtg() * rho * vtr ; 

        // u_t = W ( - alpha + beta_i v^i )
        auto const W_beta_v = metric.contract_vec_vec(metric._beta, {zx,zy,zz}) ; 
        auto const u_t = W_beta_v - W * alp ; 

        // h = 1 + eps + P/rho 
        double const h = 1 + eps + press/rho ; 

        // angular integral element
        double const domega = detector.weights_h[ip] ; 

        if ( u_t < -1.0 ) {
            flux_loc[diag_var_idx_t::GEO_UNBOUND] += r * r * domega * mass_flux ; 
        }

        if ( h*u_t < -1.0 ) {
            flux_loc[diag_var_idx_t::BERN_UNBOUND] += r * r * domega * mass_flux ; 
        }

        flux_loc[diag_var_idx_t::TOT] += r * r * domega * mass_flux ;

    }

    // Scalar surface flux; sphere quadrature only spans the active subdomain
    // when reflection symmetries are on, so scale up to the full sphere.
    int const sym_mult = scalar_symmetry_multiplier();
    for (auto& f : flux_loc) f *= sym_mult;

    return flux_loc;
}

#ifdef GRACE_ENABLE_M1
// ---------------------------------------------------------------------------
// m1_outflows::compute_local_fluxes
// ---------------------------------------------------------------------------
std::array<double, m1_outflows::n_fluxes>
m1_outflows::compute_local_fluxes(
    Kokkos::View<double**> ivals_d,
    Kokkos::View<double**> /*ivals_aux_d*/,
    spherical_surface_iface const& detector)
{
    GRACE_VERBOSE("Computing M1 radiation luminosity on sphere {}", detector.name);

    auto npoints = detector.intersecting_points_h.size();
    std::array<double, n_fluxes> flux_loc{};
    flux_loc.fill(0.0);

    if (npoints == 0) return flux_loc;

    // Mirror the interpolated state to host
    auto ivals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_d);

    const double r = detector.radius;

    for (int i = 0; i < static_cast<int>(npoints); ++i) {
        auto ip = detector.intersecting_points_h[i];

        const double x = detector.points_h[ip].second[0];
        const double y = detector.points_h[ip].second[1];
        const double z = detector.points_h[ip].second[2];
        const double r_inv = (r > 0.0) ? (1.0 / r) : 0.0;

        // Unit outward radial normal in Cartesian coordinates
        const double nx = x * r_inv;
        const double ny = y * r_inv;
        const double nz = z * r_inv;

        const double domega = detector.weights_h[ip];

        // Read metric
        double const alp   = ivals(i, loc_var_idx_t::ALPL);
        double const betax = ivals(i, loc_var_idx_t::BETAXL);
        double const betay = ivals(i, loc_var_idx_t::BETAYL);
        double const betaz = ivals(i, loc_var_idx_t::BETAZL);

        // Species 0 (nu_e / single-species)
        {
            double const E  = ivals(i, loc_var_idx_t::E1L);
            double const Fx = ivals(i, loc_var_idx_t::FX1L);
            double const Fy = ivals(i, loc_var_idx_t::FY1L);
            double const Fz = ivals(i, loc_var_idx_t::FZ1L);
            
            // Physical coordinate flux: alpha * F^i - beta^i * E
            double const phys_Fx = alp * Fx - betax * E;
            double const phys_Fy = alp * Fy - betay * E;
            double const phys_Fz = alp * Fz - betaz * E;
            
            // Project onto outward radial normal
            flux_loc[0] += r * r * domega * (phys_Fx * nx + phys_Fy * ny + phys_Fz * nz);
        }

#ifdef M1_NU_THREESPECIES
        // Species 1 (anti-nu_e)
        {
            double const E  = ivals(i, loc_var_idx_t::E2L);
            double const Fx = ivals(i, loc_var_idx_t::FX2L);
            double const Fy = ivals(i, loc_var_idx_t::FY2L);
            double const Fz = ivals(i, loc_var_idx_t::FZ2L);
            
            // Physical coordinate flux: alpha * F^i - beta^i * E
            double const phys_Fx = alp * Fx - betax * E;
            double const phys_Fy = alp * Fy - betay * E;
            double const phys_Fz = alp * Fz - betaz * E;
            
            // Project onto outward radial normal
            flux_loc[1] += r * r * domega * (phys_Fx * nx + phys_Fy * ny + phys_Fz * nz);
        }
        // Species 2 (nu_x / heavy leptons)
        {
            double const E  = ivals(i, loc_var_idx_t::E3L);
            double const Fx = ivals(i, loc_var_idx_t::FX3L);
            double const Fy = ivals(i, loc_var_idx_t::FY3L);
            double const Fz = ivals(i, loc_var_idx_t::FZ3L);
            
            // Physical coordinate flux: alpha * F^i - beta^i * E
            double const phys_Fx = alp * Fx - betax * E;
            double const phys_Fy = alp * Fy - betay * E;
            double const phys_Fz = alp * Fz - betaz * E;
            
            // Project onto outward radial normal
            flux_loc[2] += r * r * domega * (phys_Fx * nx + phys_Fy * ny + phys_Fz * nz);
        }
#endif

#ifdef M1_NU_FIVESPECIES
        // Species 3 (nu_mu)
        {
            double const E  = ivals(i, loc_var_idx_t::E4L);
            double const Fx = ivals(i, loc_var_idx_t::FX4L);
            double const Fy = ivals(i, loc_var_idx_t::FY4L);
            double const Fz = ivals(i, loc_var_idx_t::FZ4L);
            
            // Physical coordinate flux: alpha * F^i - beta^i * E
            double const phys_Fx = alp * Fx - betax * E;
            double const phys_Fy = alp * Fy - betay * E;
            double const phys_Fz = alp * Fz - betaz * E;
            
            // Project onto outward radial normal
            flux_loc[3] += r * r * domega * (phys_Fx * nx + phys_Fy * ny + phys_Fz * nz);
        }
        // Species 4 (anti-nu_mu)
        {
            double const E  = ivals(i, loc_var_idx_t::E5L);
            double const Fx = ivals(i, loc_var_idx_t::FX5L);
            double const Fy = ivals(i, loc_var_idx_t::FY5L);
            double const Fz = ivals(i, loc_var_idx_t::FZ5L);
            
            // Physical coordinate flux: alpha * F^i - beta^i * E
            double const phys_Fx = alp * Fx - betax * E;
            double const phys_Fy = alp * Fy - betay * E;
            double const phys_Fz = alp * Fz - betaz * E;
            
            // Project onto outward radial normal
            flux_loc[4] += r * r * domega * (phys_Fx * nx + phys_Fy * ny + phys_Fz * nz);
        }
#endif
    }

    const int sym_mult = scalar_symmetry_multiplier();
    for (auto& f : flux_loc) f *= sym_mult;

    return flux_loc;
}
#endif /* GRACE_ENABLE_M1 */

}
