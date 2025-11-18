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
#ifndef GRACE_IO_BH_DIAGNOSTICS_HH
#define GRACE_IO_BH_DIAGNOSTICS_HH
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


#define SQR(a) (a)*(a)

namespace grace {



struct bh_diagnostics {

    enum loc_var_idx_t : int {
        GXXL=0, GXYL, GXZL, GYYL, GYZL, GZZL,
        BETAXL, BETAYL, BETAZL, ALPL, RHOL, EPSL, PRESSL, VELXL, VELYL, VELZL,
        BXL, BYL, BZL, NUM_VARS
    } ; 

    enum diag_var_idx_t : int {
        MDOT=0, EDOT, LDOT, PHI, N_DIAG_VARS
    } ; 

    bh_diagnostics() {

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

        var_interp_idx_h = std::vector<int> {
            GXX, GXY, GXZ, GYY, GYZ, GZZ, BETAX, BETAY, BETAZ, ALP
        } ; 
        aux_interp_idx_h = std::vector<int> {
            RHO, EPS, PRESS, VELX, VELY, VELZ, BX, BY, BZ
        } ; 
    }

    void initialize_files() {
        auto& sphere_list = grace::spherical_surface_manager::get() ; 

        if( parallel::mpi_comm_rank() == 0 ) {
            auto& grace_runtime = grace::runtime::get() ; 
            static constexpr const size_t width = 20 ; 
            std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ; 
            for( int i=0; i < sphere_indices.size(); ++i ) {
                auto const& detector = sphere_list.get(sphere_indices[i]) ;
                auto name = detector.name ;
                {
                    std::string const pfname = grace_runtime.scalar_io_basename() + "Mdot_" + name + ".dat" ;
                    std::filesystem::path fname = bdir /  pfname ; 
                    std::ofstream outfile(fname.string(),std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;  
                }
                {
                    std::string const pfname = grace_runtime.scalar_io_basename() + "Edot_" + name + ".dat" ;
                    std::filesystem::path fname = bdir /  pfname ; 
                    std::ofstream outfile(fname.string(),std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;  
                }
                {
                    std::string const pfname = grace_runtime.scalar_io_basename() + "Ldot_" + name + ".dat" ;
                    std::filesystem::path fname = bdir /  pfname ; 
                    std::ofstream outfile(fname.string(),std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ;  
                }
                {
                    std::string const pfname = grace_runtime.scalar_io_basename() + "Phi_" + name + ".dat" ;
                    std::filesystem::path fname = bdir /  pfname ; 
                    std::ofstream outfile(fname.string(),std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << std::setw(width) << "Iteration" << std::left << std::setw(width) << "Time" << std::left << std::setw(width) << "Value" << '\n' ; 
                }
            }

        }
    }

    void compute() {

        static constexpr size_t width = 20 ;


        auto& grace_runtime = grace::runtime::get() ; 
        size_t const iter = grace_runtime.iteration() ; 
        double const time = grace_runtime.time()      ;

        // compute fluxes 
        Mdot.resize(sphere_indices.size()) ; 
        Edot.resize(sphere_indices.size()) ; 
        Ldot.resize(sphere_indices.size()) ; 
        Phi.resize(sphere_indices.size()) ; 
        for( int i=0; i < sphere_indices.size(); ++i ) {
            auto fluxes = compute_diagnostics(sphere_indices[i]) ;
            Mdot[i] = fluxes[0] ; 
            Edot[i] = fluxes[1] ; 
            Ldot[i] = fluxes[2] ; 
            Phi[i] = fluxes[3] ; 
        }
        auto& sphere_list = grace::spherical_surface_manager::get() ; 
        
        // output 
        auto rank = parallel::mpi_comm_rank() ; 
        if ( rank == 0 ) {
            std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ; 
            for( int i=0; i < sphere_indices.size(); ++i ) {
                auto const& detector = sphere_list.get(sphere_indices[i]) ;
                auto name = detector.name ;    
                {
                    std::string const pfname = grace_runtime.scalar_io_basename() + "Mdot_" + name + ".dat" ;
                    std::filesystem::path fname = bdir /  pfname ; 
                    std::ofstream outfile(fname.string(),std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << iter << '\t'
                            << std::left << time << '\t' 
                            << std::left << Mdot[i] << '\n' ; 
                }
                {
                    std::string const pfname = grace_runtime.scalar_io_basename() + "Edot_" + name + ".dat" ;
                    std::filesystem::path fname = bdir /  pfname ; 
                    std::ofstream outfile(fname.string(),std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << iter << '\t'
                            << std::left << time << '\t' 
                            << std::left << Edot[i] << '\n' ; 
                }
                {
                    std::string const pfname = grace_runtime.scalar_io_basename() + "Ldot_" + name + ".dat" ;
                    std::filesystem::path fname = bdir /  pfname ; 
                    std::ofstream outfile(fname.string(),std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << iter << '\t'
                            << std::left << time << '\t' 
                            << std::left << Ldot[i] << '\n' ; 
                }
                {
                    std::string const pfname = grace_runtime.scalar_io_basename() + "Phi_" + name + ".dat" ;
                    std::filesystem::path fname = bdir /  pfname ; 
                    std::ofstream outfile(fname.string(),std::ios::app) ;
                    outfile << std::fixed << std::setprecision(15) ; 
                    outfile << std::left << iter << '\t'
                            << std::left << time << '\t' 
                            << std::left << Phi[i] << '\n' ; 
                }
            }
        }
            
        
    }

    std::array<double,N_DIAG_VARS> compute_diagnostics(size_t sphere_idx) {
        
        // NB here we resize the view
        auto& sphere_list = grace::spherical_surface_manager::get() ; 
        auto const& detector = sphere_list.get(sphere_idx) ; 
        auto const& coord_system = grace::coordinate_system::get() ;
        GRACE_VERBOSE("Computing BH diagnostics on sphere {}", detector.name ) ; 

        std::array<double,N_DIAG_VARS> flux_loc = {0.,0.,0.,0.} ; 

        auto npoints = detector.intersecting_points_h.size() ; 
        GRACE_VERBOSE("We have {} points", npoints) ; 
        if ( npoints > 0 ) {
            Kokkos::View<double**,grace::default_space> ivals_d("interp_vars",0,0) ;

            interpolate_on_sphere(
                detector, var_interp_idx_h, aux_interp_idx_h, ivals_d
            ); 

            auto ivals = Kokkos::create_mirror_view(ivals_d);
            Kokkos::deep_copy(ivals,ivals_d) ; 
            
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
                std::array<double,3> const vN {
                    one_over_alp * ( vx + metric.beta(0) )
                    , one_over_alp * ( vy + metric.beta(1) )
                    , one_over_alp * ( vz + metric.beta(2) )
                } ; 
                double const W = 1./Kokkos::sqrt(1-metric.square_vec(vN)) ; 

                double const u0 = one_over_alp * W ; 
                std::array<double,4> uU{{ u0, vx * u0, vy * u0, vz * u0 }};
                auto uD = metric.lower_4vec(uU) ; 

                double b0 = uD[1] * bx + uD[2] * by + uD[3] * bz ; 
                double b1 = (bx+b0*uU[1])/u0;
                double b2 = (by+b0*uU[2])/u0;
                double b3 = (bz+b0*uU[3])/u0;

                auto bD = metric.lower_4vec({b0,b1,b2,b3}) ; 

                double bsq = b0 * bD[0] + b1 * bD[1] + b2 * bD[2] + b3*bD[3] ; 

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

                // accretion rate
                flux_loc[0] += - domega * sqrtmdet * rho * ur ; 
                // energy flux 
                flux_loc[1] += - domega * sqrtmdet * tr_0 ; 
                // angular momentum flux 
                flux_loc[2] += domega * sqrtmdet * tr_3 ; 
                // magnetic flux 
                flux_loc[3] += domega * sqrtmdet * phi_l ; 

            }
        }
        parallel::mpi_barrier() ; 
        std::array<double,N_DIAG_VARS> flux_glob = {0.,0.,0.,0.}; 
        parallel::mpi_allreduce(
            flux_loc.data(),
            flux_glob.data(),
            N_DIAG_VARS,
            MPI_SUM
        );

        GRACE_VERBOSE("Mdot local {} global {}", flux_loc[0], flux_glob[0]) ; 
        return flux_glob ; 
    }

    std::vector<int> var_interp_idx_h, aux_interp_idx_h ; 
    std::vector<size_t> sphere_indices ;
    std::vector<double> Mdot, Edot, Ldot, Phi;
} ; 

}
#undef SQR

#endif /*GRACE_IO_BH_DIAGNOSTICS_HH*/