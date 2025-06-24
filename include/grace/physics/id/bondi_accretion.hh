/**
 * @file Bondi_accretion.hh
 * @author Marie Cassing (mcassing@itp.uni-frankfurt.de)
 * @brief The classical 2D GRMHD test, where the setup follows https://arxiv.org/pdf/1611.09720
 * @warning in the paper, the velocities v^i are WU^i and hence we need to rescale by W when setting our velocities 
 * @date 2025-05-0-5
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

//**************************************************************************************************
namespace grace {
//**************************************************************************************************
/**
 * \defgroup initial_data Initial Data
 * 
 */
//**************************************************************************************************
/**
 * @brief Bondi accretion MHD test - spherical accretion to test MHD
*
 * \ingroup initial_data
 * @tparam eos_t type of equation of state
 * @note this kernel has to be checked for and adjusted if needed 
 * should the magnetic field initialization method/location changes in the future (e.g. vec pot)
 * @note only in 2D/3D problems can divergence cleaning performance of the GLM method's be judged 
 *       in 1D (and flat spacetime), the evolution of phi_glm is trivial
 */
template < typename eos_t >
struct bondi_accretion_id_t {
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //!< Type of state vector
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Construct a new bondi_accretion_mhd id kernel object
     * 
     * @param eos Equation of state
     * @param pcoords Physical coordinate array
     * @param M Mass of Black Hole
     * @param Mdot Mass accretion rate at sonic point
     * @param r_sonic Radius of the sonic point
     * @param gamma Adiabatic index
     * @param rmin Radial minimal bound for solution
     * @param rmax Radial maximal bound for solution
     * @param bmag magnetic field
     * @param beta_sonic magnetic field parameter
     */
    // Constructor with parameters
    bondi_accretion_id_t(
        eos_t eos_,
        grace::coord_array_t<GRACE_NSPACEDIM> pcoords_,
        double M_, double mdot_, double r_sonic_,
        double gamma_, double rmin_, double rmax_,
        double bmag_ , double beta_sonic_)
        : _pcoords(pcoords_), M(M_), mdot(mdot_), r_sonic(r_sonic_),
          gamma(gamma_), rmin(rmin_), rmax(rmax_),
          bmag(bmag_), beta_sonic(beta_sonic_) //, _eos(eos_)
    {
        GRACE_INFO("In Bondi accretion setup.") ; 
        if(M <= 0.0) ERROR("M zero or negative");
        GRACE_INFO("Using input parameters of\n  \
                        M = {},\n \
                        M_dot = {},\n \
                        r_sonic ={},\n \
                        r_min = {},\n \
                        r_max ={},\n \
                        bmag = {}\n \
                        beta_sonic = {}\n \
                        gamma = {}", \
                        M,mdot,r_sonic,rmin,rmax,bmag,beta_sonic,gamma);

       // double cs, cs_sq, vs_sq, vs;
       // double rhos, gmo, hs, Kval, Qdot, psonic;

        // Calculate sonic point properties
        cs_sq = M / (2.0*r_sonic - 3.0*M);
        if (cs_sq > (gamma - 1.0)) {
            cs_sq = gamma - 1.0;
            r_sonic = 0.5 * M * (3.0 + 1.0/cs_sq);
        }
        
        cs = Kokkos::sqrt(cs_sq);
        vs_sq = M / (2.0 * r_sonic);
        vs = Kokkos::sqrt(vs_sq);
        rhos = mdot / (4.0 * M_PI * vs * r_sonic * r_sonic);
        gmo = gamma - 1.0;
        hs = 1.0 / (1.0 - cs_sq / gmo);
        Kval = hs * cs_sq * pow(rhos, -gmo) / gamma;
        Qdot = hs * hs * (1.0 - 3.0 * vs_sq);
        
        // Set magnetic field
        if (beta_sonic > 0.0) {
            double psonic = Kval * pow(rhos, gamma);
            bmag = Kokkos::sqrt(2.0*psonic/beta_sonic) * r_sonic*r_sonic/M 
                   / Kokkos::sqrt(1.0 - 2.0*M/r_sonic);
        }
    }

    // Main evaluation kernel
    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int i, int j, int k), int q) const 
    {
        double const x = _pcoords(VEC(i,j,k), 0, q);
        double const y = _pcoords(VEC(i,j,k), 1, q);
        #ifdef GRACE_3D 
        double const z = _pcoords(VEC(i,j,k), 2, q);
        #else 
        double const z = 0.0; 
        #endif 
        
        double const r = Kokkos::sqrt(x*x + y*y + z*z + 1.0e-16);
        double const xhat = x/r;
        double const yhat = y/r;
        double const zhat = z/r;
        
        grmhd_id_t id;
        
        // Schwarzschild metric in isotropic coordinates
        double psi = 1.0 + M/(2.0*r);
        double psi4 = psi*psi*psi*psi;
        
        id.alp = (1.0 - M/(2.0*r)) / (1.0 + M/(2.0*r));
        id.betax = 0.0; id.betay = 0.0; id.betaz = 0.0;
        
        id.gxx = psi4;
        id.gxy = 0.0;
        id.gxz = 0.0;
        id.gyy = psi4;
        id.gyz = 0.0;
        id.gzz = psi4;
        
        id.kxx = 0.0; id.kxy = 0.0; id.kxz = 0.0;
        id.kyy = 0.0; id.kyz = 0.0; id.kzz = 0.0;

        // Get Bondi solution at this radius
        double r_sch = 0.25 * (2.0*r + M)*(2.0*r + M)/r; // Isotropic to Schwarzschild
        double rho, u, v;
        find_bondi_solution(r_sch, rho, u, v, r_sonic, rhos, M, mdot, Kval, gamma, Qdot);
        
        id.rho = rho;
        id.press = Kval * pow(rho, gamma);
        
        // Convert velocity from Schwarzschild to isotropic coordinates
        double u_iso = v / (1.0 - M/(2.0*r)) / (1.0 + M/(2.0*r));
        double w_lorentz = Kokkos::sqrt(1.0 + id.gxx * u_iso*u_iso);
        
        id.vx = -u_iso * xhat / w_lorentz;
        id.vy = -u_iso * yhat / w_lorentz;
        id.vz = -u_iso * zhat / w_lorentz;
        
        // Magnetic field (if enabled)
        if (bmag > 0.0) {
            double det = psi4*psi4; // determinant of 3-metric
            double sdet = psi4;
            
            id.bx = bmag*M*M*xhat/(sdet*r*r);
            id.by = bmag*M*M*yhat/(sdet*r*r);
            id.bz = bmag*M*M*zhat/(sdet*r*r);
        } else {
            id.bx = 0.0; id.by = 0.0; id.bz = 0.0;
        }
        
        return std::move(id) ; 
    }
   // }

private:
    // Helper function to find Bondi solution at a given radius
    //GRACE_DEVICE void find_bondi_solution(double r, double& rho, double& u, double& v,
    //                                    double rs, double rhos, double M, double mdot,
    //                                    double Kval, double gamma, double Qdot) const   
    //GRACE_DEVICE void find_bondi_solution(double r, double& rho, double& u, double& v) const {
    GRACE_DEVICE void find_bondi_solution(double r, double& rho, double& u, double& v,
                                        double rs, double rhos, double M, double mdot,
                                        double Kval, double gamma, double Qdot) const {
        // Simple Newton-Raphson root finder for u, then backsolve for rho
        constexpr int max_iter = 30;
        constexpr double tol = 1e-10;
        double u_guess = -Kokkos::sqrt(M / r);
        double u_old = u_guess;
        double f, df;
        for (int n = 0; n < max_iter; ++n) {
            double w = Kokkos::sqrt(1.0 + u_old * u_old);
            double h = Kval * pow(mdot / (4.0 * M_PI * r * r * u_old), gmo) / gamma + 1.0;
            f = h * h * (1.0 - 2.0 * M / r + u_old * u_old) - Qdot;
            df = 2.0 * h * (1.0 - 2.0 * M / r + u_old * u_old)
                 - h * h * 2.0 * u_old / Kokkos::sqrt(1.0 + u_old * u_old);
            double delta = f / df;
            u_old -= delta;
            if (Kokkos::abs(delta) < tol) break;
        }
        u = u_old;
        v = -u / Kokkos::sqrt(1.0 + u * u);
        rho = mdot / (4.0 * M_PI * r * r * u);
    }

    // Member variables
    //eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double M, mdot, r_sonic, gamma;
    double rmin, rmax, bmag, beta_sonic;
    double cs_sq, cs, vs_sq, vs, rhos, gmo, hs, Kval, Qdot;
};

} // namespace grace

#endif /* GRACE_PHYSICS_ID_BONDI_ACCRETION_HH */
//**************************************************************************************************