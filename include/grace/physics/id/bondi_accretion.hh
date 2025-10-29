/**
 * @file Bondi_accretion.hh
 * @author Marie Cassing
 * @brief Complete Bondi accretion initial data setup with logarithmic pre-tabulation
 * @date 2025-05-05
 *  
 * @copyright This file is part of the General Relativistic Astrophysics
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

#ifndef GRACE_PHYSICS_ID_BONDI_ACCRETION_MHD_HH
#define GRACE_PHYSICS_ID_BONDI_ACCRETION_MHD_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace grace {

template <typename eos_t>
struct bondi_accretion_id_t {
    using state_t = grace::var_array_t<GRACE_NSPACEDIM>;

    bondi_accretion_id_t(
        eos_t eos_,
        grace::coord_array_t<GRACE_NSPACEDIM> pcoords_,
        double M_, double mdot_, double r_sonic_,
        double gamma_, double rmin_, double rmax_,
        double bmag_, double beta_sonic_)
        : _pcoords(pcoords_), M(M_), mdot(mdot_), r_sonic(r_sonic_),
          gamma(gamma_), rmin(rmin_), rmax(rmax_),
          bmag(bmag_), beta_sonic(beta_sonic_) {

        N_points = 512;
        double PI = 3.1415926535897932;

        // Calculate sonic point parameters
        cs_sq = M / (2.0 * r_sonic - 3.0 * M);
        if (cs_sq > (gamma - 1.0)) {
            cs_sq = gamma - 1.0;
            r_sonic = 0.5 * M * (3.0 + 1.0 / cs_sq);
        }
        cs = sqrt(cs_sq);
        vs_sq = M / (2.0 * r_sonic);
        vs = sqrt(vs_sq);
        rhos = mdot / (4.0 * PI * vs * r_sonic * r_sonic);
        gmo = gamma - 1.0;
        hs = 1.0 / (1.0 - cs_sq / gmo);
        Kval = hs * cs_sq * pow(rhos, -gmo) / gamma;
        Qdot = hs * hs * (1.0 - 3.0 * vs_sq);

        // Set magnetic field strength
        if (beta_sonic > 0.0) {
            double psonic = Kval * pow(rhos, gamma);
            bmag = sqrt(2.0 * psonic / beta_sonic) * r_sonic * r_sonic / M / sqrt(1.0 - 2.0 * M / r_sonic);
        }

        // Set up logarithmic grid
        logrmin = log10(rmin);
        logrmax = log10(rmax);
        dlogr = (logrmax - logrmin) / (N_points - 1);

        // Find index closest to sonic point
        imin = 0;
        double rhotmp = 1e30;

        for (int i = 0; i < N_points; ++i) {
            logr_bondi[i] = logrmin + dlogr * i;
            r_bondi[i] = pow(10.0, logr_bondi[i]);
            //rschwarz[i] = r_bondi[i] * pow(1.0 + M/2.0/r_bondi[i],2.0);
            rschwarz[i] = 0.25 * (2.0 * r_bondi[i] + M) * (2.0 * r_bondi[i] + M) / r_bondi[i];
            double utmp = fabs(r_bondi[i] - r_sonic);
            if (utmp < rhotmp) {
                rhotmp = utmp;
                imin = i;
            }
        }

        // Initialize solution at sonic point
        rho_bondi[imin] = rhos;
        u_bondi[imin] = Kval * pow(rhos, gamma) / (gamma - 1.0);
        v_bondi[imin] = vs;

        // Solve outward from sonic point
        for (int i = imin + 1; i < N_points; ++i) {
            double rho_i = rho_bondi[i-1]; // Use previous solution as guess
            double rspher = r_bondi[i];
            double u, v;
            find_bondi_solution(rspher, rho_i, u, v, r_sonic, rhos, M, mdot, Kval, gamma, Qdot);
            rho_bondi[i] = rho_i;
            u_bondi[i] = u;
            v_bondi[i] = v;
            //velo[i] = mdot / (4.0 * PI * r_bondi[i] * r_bondi[i] * rho_i);
        }

        // Solve inward from sonic point
        for (int i = imin - 1; i >= 0; --i) {
            double rho_i = rho_bondi[i+1]; // Use previous solution as guess
            double rspher = r_bondi[i];
            double u, v;
            find_bondi_solution(rspher, rho_i, u, v, r_sonic, rhos, M, mdot, Kval, gamma, Qdot);
            rho_bondi[i] = rho_i;
            u_bondi[i] = u;
            v_bondi[i] = v;
            //velo[i] = mdot / (4.0 * PI * r_bondi[i] * r_bondi[i] * rho_i);
        }

        double rnew;
        int j;
        // find derivative near r=M in isotropic coords r=9/4M
        rnew = 2.25 * M;
        j = floor( (log10(rnew) - logrmin) / dlogr + 1.0);
        rho_check = rho_bondi[j];
        double u_check, v_check;
        find_bondi_solution(rnew, rho_check, u_check, v_check, r_sonic, rhos, M, mdot, Kval, gamma, Qdot);
        uisocheck = 4.0 * v_check/ 3.0;

        rnew = 0.25 * pow(3.02,2.0) * M/1.01;
        j = floor((log10(rnew) - logrmin) / dlogr + 1.0);
        double rho_check2 = rho_bondi[j];
        double u_check2, v_check2;
        find_bondi_solution( rnew, rho_check2, u_check2, v_check2, r_sonic, rhos, M, mdot, Kval, gamma, Qdot);
        double uisocheck2 = v_check2 / (1.0 - 1.0/2.02) / (1.0+ 1.0/2.02);

        drhodr = 100.0 * (rho_check2 - rho_check)/M;
        dudr = 100.0 * (uisocheck2 -uisocheck);
             
    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator()(VEC(int i, int j, int k), int q) const {
        double x = _pcoords(VEC(i,j,k), 0, q);
        double y = _pcoords(VEC(i,j,k), 1, q);
        #ifdef GRACE_3D
        double z = _pcoords(VEC(i,j,k), 2, q);
        #else
        double z = 0.0;
        #endif
        
        double M_tiny = 0.99999;
        double r = sqrt(x*x + y*y + z*z + 1.0e-16);
        double xhat = x / r, yhat = y / r, zhat = z / r;

        double rmin = 0.1*M;
        r = std::max(r, rmin);

        grmhd_id_t id;
        double psi = 1.0 + M / (2.0 * r);
        double psi4 = psi * psi * psi * psi;

        // Set spacetime metric (isotropic Schwarzschild)
        id.alp = (1.0 - M / (2.0 * r)) / (1.0 + M / (2.0 * r));
        id.gxx = id.gyy = id.gzz = psi4;
        id.gxy = id.gxz = id.gyz = 0.0;
        id.kxx = id.kxy = id.kxz = 0.0;
        id.kyy = id.kyz = id.kzz = 0.0;
        id.betax = id.betay = id.betaz = 0.0;

        // Convert to Schwarzschild radius for Bondi solution
        double r_sch = 0.25 * (2.0 * r + M) * (2.0 * r + M) / r;
        double roverm = r / M;
        
        // Get Bondi solution at this radius
        double rho, u, v, u_iso;
        if(roverm > M_tiny){
            int jb = floor((log10(r_sch) - logrmin)/dlogr + 1.0);
            if(jb >= N_points) jb = N_points - 1;

            double rhotmp = rho_bondi[jb] + (rho_bondi[jb+1] - rho_bondi[jb]) * (r_sch - r_bondi[jb])/(r_bondi[jb+1] - r_bondi[jb]);

            find_bondi_solution(r_sch, rhotmp, u, v, r_sonic, rhos, M, mdot, Kval, gamma, Qdot);        
            // Convert velocity to isotropic coordinates
            rho = rhotmp;
            u_iso = v / (1.0 - M / (2.0 * r)) / (1.0 + M / (2.0 * r));
        }
        else{
            if(roverm > 0.5 * M_tiny){
                rho = rho_check + drhodr * r *(r-M)/M;
            }
            else{
                rho = (rho_check + drhodr *M/4.0) * (1.0 - Kokkos::cos(2.0 * M_PI * r /M))/2.0;
            }
            double auiso = 1.5*uisocheck-0.5*dudr;
            double buiso = 0.5*dudr-0.5*uisocheck;
            u_iso =  roverm*(auiso + buiso * pow(roverm,2.0));
        }
        id.rho = rho;
        id.press = Kval * pow(rho, gamma);

        // Convert velocity to isotropic coordinates
        //double u_iso = v / (1.0 - M / (2.0 * r)) / (1.0 + M / (2.0 * r));
        double w_lorentz = sqrt(1.0 + id.gxx * u_iso * u_iso);

        id.vx = -u_iso * xhat / w_lorentz;
        id.vy = -u_iso * yhat / w_lorentz;
        id.vz = -u_iso * zhat / w_lorentz;

        // Set magnetic field
        if (bmag > 0.0) {
            double sdet = psi4;
            id.bx = bmag * M * M * xhat / (sdet * r * r);
            id.by = bmag * M * M * yhat / (sdet * r * r);
            id.bz = bmag * M * M * zhat / (sdet * r * r);
        } else {
            id.bx = id.by = id.bz = 0.0;
        }

        return id;
    }

private:
    GRACE_HOST_DEVICE void find_bondi_solution(double r, double& rho, double& u, double& v,
                                          double rs, double rhos, double M, double mdot,
                                          double Kval, double gamma, double Qdot) const {
        constexpr int max_iter = 50;
        constexpr double tol = 1e-12;
        constexpr double rho_floor = 1e-14;
        constexpr double delta_max = 1e6;
        double PI = 3.1415926535897932;

        /*
        //--------------------------------------
        // Initial guess like in ETK
        double ur = (r < r_sonic) ? pow(r, -0.5) : 0.5 * pow(r, -1.5);
        rho = mdot / (4.0 * PI * r * r * ur);
        if (rho <= 0.0 || !std::isfinite(rho)) rho = rhos;
        
        Kokkos::printf("Hi, rho is %f\n", rho);
        
        //--------------------------------------
        //if (rho <= 0.0) {
        //    rho = (r > 0.9 * r_sonic && r < 1.1 * r_sonic) ? rhos : mdot / (4.0 * PI * r * r * 0.5 * pow(r, -1.5));
        //}
        //--------------------------------------
        // Try to find closest tabulated index
        double log_r = log10(r);
        int i_low = int((log_r - logr_bondi[0]) / (logr_bondi[1] - logr_bondi[0]));
        i_low = std::max(0, std::min(i_low, 510)); // ensure in bounds
        
        // Interpolate guess from table
        double w = (log_r - logr_bondi[i_low]) / (logr_bondi[i_low+1] - logr_bondi[i_low]);
        rho = (1.0 - w) * rho_bondi[i_low] + w * rho_bondi[i_low+1];
        Kokkos::printf("Hello, rho is %f\n", rho);
        
        if (!std::isfinite(rho) || rho <= rho_floor) rho = rhos; // fallback
        //--------------------------------------
        */
        double ur;
        if( rho < 0. ) {  
          if( r > 0.9*rs && r < 1.1*rs ) { 
            rho = rhos;
          }
          else { 
            //  rhotmp = (sqrt(Qdot) - 1.) * (gamma_eos - 1.) / ( gamma_eos * K );
            if(r < rs) {  ur = pow(r,-0.5)     ;  }
            else       {  ur = 0.5*pow(r,-1.5) ;  }
            rho = mdot / (4.*PI * r * r * ur); 
          }
        } 
    
        int retval = 0;
        double r_sol = r;

        double errx = 1.;
        //double df = 1.;
        //double f = 1.;

        // Newton Raphson loop----------------
        for (int iter = 0; iter < max_iter; ++iter) {
            rho = std::max(rho, rho_floor);
            double v_local = mdot / (4.0 * PI * r_sol * r_sol * rho);
            double dv_drho = -v_local / rho;
            double dh_drho = Kval * gamma * pow(rho, gamma - 2.0);
            //double h = 1.0 + dh_drho * rho /(gamma - 1.0);
            double h = 1.0 + Kval * gamma * pow(rho, gamma - 1.0) / (gamma - 1.0);

            double term = 1.0 - 2.0 * M / r_sol + v_local * v_local;
            double resid = h * h * term - Qdot;
            double dresid = 2.0 * h * (dh_drho * term + h * v_local * dv_drho);

            double delta = resid / dresid;
            double func = 0.5 * resid * resid;
            double dfunc = -2.0 * func;

            if (!std::isfinite(delta) || fabs(delta) > delta_max) {
                rho = std::max(rhos, rho_floor * 10.0);
                //Kokkos::printf("delta infinte\n");
                retval = 1;
                break;
            }

            rho -= delta;
            if (rho < rho_floor || !std::isfinite(rho)) {
                rho = rho_floor;
                //rho = std::max(rhos, rho_floor * 10.0);
                //if (!std::isfinite(rho)) rho = rho_floor; //1.0;
                //u = Kval * pow(rho, gamma) / (gamma - 1.0);
                //v = mdot / (4.0 * PI * r_sol * r_sol * rho);
                //Kokkos::printf("rho infinite, rho <rhofloor \n");
                //Kokkos::printf("resid = %f, dresid = %f, r = %f, rho = %f \n", resid, dresid, r_sol, rho);
                break;
            }

            if (fabs(delta) < tol) {
                //Kokkos::printf("sol found \n");
                break;
            }
        }
        //-----------------------------

       int ntries = 10;
        if(retval){
            double dr = (r - rs)/(1.*(ntries-1));
            rho = rhos;
            r_sol = rs;
            r_sol  = rs;
            for (int itry=1; itry < ntries; itry++){
                r_sol += dr;
                // Newton Raphson loop----------------
                for (int iter = 0; iter < max_iter; ++iter) {
                    rho = std::max(rho, rho_floor);
                    double v_local = mdot / (4.0 * PI * r_sol * r_sol * rho);
                    double dv_drho = -v_local / rho;
                    double dh_drho = Kval * gamma * pow(rho, gamma - 2.0);
                    //double h = 1.0 + dh_drho * rho /(gamma - 1.0);
                    double h = 1.0 + Kval * gamma * pow(rho, gamma - 1.0) / (gamma - 1.0);
        
                    double term = 1.0 - 2.0 * M / r_sol + v_local * v_local;
                    double resid = h * h * term - Qdot;
                    double dresid = 2.0 * h * (dh_drho * term + h * v_local * dv_drho);
        
                    double delta = resid / dresid;
                    double func = 0.5 * resid * resid;
                    double dfunc = -2.0 * func;
        
                    if (!std::isfinite(delta) || fabs(delta) > delta_max) {
                        rho = std::max(rhos, rho_floor * 10.0);
                        //Kokkos::printf("delta infinte\n");
                        retval = 1;
                        break;
                    }
        
                    rho -= delta;
                    if (rho < rho_floor || !std::isfinite(rho)) {
                        rho = rho_floor;
                        //rho = std::max(rhos, rho_floor * 10.0);
                        //if (!std::isfinite(rho)) rho = rho_floor; //1.0;
                        //u = Kval * pow(rho, gamma) / (gamma - 1.0);
                        //v = mdot / (4.0 * PI * r_sol * r_sol * rho);
                        //Kokkos::printf("rho infinite, rho <rhofloor \n");
                        //Kokkos::printf("resid = %f, dresid = %f, r = %f, rho = %f \n", resid, dresid, r_sol, rho);
                        break;
                    }
        
                    if (fabs(delta) < tol) {
                        //Kokkos::printf("sol found \n");
                         break;
                    }
                }
                //-----------------------------                
            }

        }

        u = Kval * pow(rho, gamma) / (gamma - 1.0);
        v = mdot / (4.0 * PI * r * r * rho);
        if (!std::isfinite(u)) u = 0.0;
        if (!std::isfinite(v)) v = 0.0;
    }

    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords;
    double M, mdot, r_sonic, gamma;
    double rmin, rmax, bmag, beta_sonic;
    double cs_sq, cs, vs_sq, vs, rhos, gmo, hs, Kval, Qdot;
    double logr_bondi[512], r_bondi[512], rschwarz[512], rho_bondi[512], u_bondi[512], v_bondi[512];
    double logrmin, logrmax, dlogr;
    double rho_check, drhodr, uisocheck, dudr;
    int imin,N_points;
};

} // namespace grace

#endif /* GRACE_PHYSICS_ID_BONDI_ACCRETION_MHD_HH */