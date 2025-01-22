/**
 * @file fmtorus.hh
 * @author Konrad Topolski (topolski @itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-10-17
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

/* ========================================================

 The code used here heavily relies on the SymPy-generated
 expressions for the metric and hydro fields used in the 
 WVUThorns/FishboneMoncriefID thorn 

========================================================== */

#ifndef GRACE_PHYSICS_ID_FMTORUS_HH
#define GRACE_PHYSICS_ID_FMTORUS_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/errors/error.hh>

// FMtorus-related includes:
#include <grace/physics/id/FMtorus/KerrSchild.hh>
#include <grace/physics/id/FMtorus/FMdisk_GRHD_velocities.hh>
#include <grace/physics/id/FMtorus/FMdisk_GRHD_rho_initial.hh>
#include <grace/physics/id/FMtorus/FMdisk_GRHD_hm1.hh>

#include <Kokkos_Core.hpp>


namespace grace {

        // template <typename eos_t>
        // concept PiecewisePolytropicEOS = std::is_same_v<eos_t, grace::piecewise_polytropic_eos_t>;

        /**
         * @brief FMtorus initial data kernel.
         * 
         * @tparam eos_t Eos type
         *         Due to the setup of the ID, the template
         *         specializes exclusively for hybrid eos and a specific setup 
         *         of the polytropic eos 
         */
        template < typename eos_t >
        //requires PiecewisePolytropicEOS<eos_t>
        struct fmtorus_id_t {
            using state_t = grace::var_array_t<GRACE_NSPACEDIM> ;

            // constructor: initializing the parameters, performing consistency checks on the ID
            fmtorus_id_t(
                eos_t eos_,
                grace::coord_array_t<GRACE_NSPACEDIM> pcoords,
                double a_, double M_, 
                double rho_min_, double press_min_, 
                double lapse_min_, double r_in_, double r_at_max_density_,
                double kappa_, double gamma_ )
                : \
                a(a_), M(M_),
                eos(eos_), _pcoords(pcoords), 
                rho_min(rho_min_), press_min(press_min_),
                lapse_min(lapse_min_), r_in(r_in_), r_at_max_density(r_at_max_density_),
                kappa(kappa_),gamma(gamma_)
            { 
                GRACE_INFO("In FMTorus setup.") ; 
                // consistency checks:
                if(r_in >= r_at_max_density) ERROR("r_in needs to be smaller than r_at_max_density!");
                if(a < 0.0) ERROR("Negative spins not supported! The setup of fluid profiles assumes the positive sign.");
                if(M <= 0.0) ERROR("M zero or negative");

                GRACE_INFO("Using input parameters of\n  \
                        a = {}\n,  \
                        M = {},\n \
                        r_in = {},\n \
                        r_at_max_density ={},\n \
                        kappa = {}\n \
                        gamma = {}", \
                        a,M,r_in,r_at_max_density,kappa,gamma);
   
                // First compute maximum pressure and density
                GRACE_VERBOSE("Computing max pressure and density");

                double P_max, rho_max;
                {
                    double hm1;
                    double xcoord = r_at_max_density;
                    double ycoord = 0.0;
                    double zcoord = 0.0;
                        {
                    #include <grace/physics/id/FMtorus/FMdisk_GRHD_hm1.hh>
                        }
                    rho_max = pow( hm1 * (gamma-1.0) / (kappa*gamma), 1.0/(gamma-1.0) );
                    P_max   = kappa * pow(rho_max, gamma);
                }

                // We enforce units such that rho_max = 1.0; if these units are not obeyed, then
                //    we error out. If we did not error out, then the value of kappa used in all
                //    EOS routines would need to be changed, and generally these appear as
                //    read-only parameters.
                if(fabs(P_max/rho_max - kappa) > 1e-8) {
                    GRACE_INFO("Error: To ensure that P = kappa*rho^Gamma, where rho_max = 1.0,\n, \
                        you must set (in your parfile) the polytropic constant kappa = P_max/rho_max = {} \n\n, \
                        Needed values for kappa, for common values of Gamma:\n \
                        For Gamma=4/3, use kappa=K_initial=K_poly = 4.249572342020724e-03 to ensure rho_max = 1.0,\n \
                        For Gamma=5/3, use kappa=K_initial=K_poly = 6.799315747233158e-03 to ensure rho_max = 1.0,\n \
                        For Gamma=2, use kappa=K_initial=K_poly = 8.499144684041449e-03 to ensure rho_max = 1.0\n", (P_max/rho_max) );
                    ERROR("Aborting the run");
                }

            } 

            // the main evaluation kernel as operator()
            grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
            operator() (VEC(int i, int j, int k), int q) const 
            {
                double const x = _pcoords(VEC(i,j,k),0,q);
                double const y = _pcoords(VEC(i,j,k),1,q);
                #ifdef GRACE_3D 
                double const z = _pcoords(VEC(i,j,k),2,q);
                #else 
                double const z = 0. ; 
                #endif 
                double const r = Kokkos::sqrt(EXPR(
                    math::int_pow<2>(x),
                    + math::int_pow<2>(y),
                    + math::int_pow<2>(z)
                )) ; 

                // at this radius.
                auto sol_metric = get_KS_metric(x,y,z,M,a) ;
                auto sol_vel = get_fluid_velocities(x,y,z,M,a,r_at_max_density) ;
                double rho_init = get_rho_initial(x,y,z,M,a,r_at_max_density,r_in,gamma,kappa);
                grmhd_id_t id ; 

                /* Set the metric */
                id.alp = std::max(lapse_min,sol_metric[KS_ALPHA]) ; 
                //id.alp   = sol_metric[KS_ALPHA] ;
                id.betax = sol_metric[KS_BETAX] ; 
                id.betay = sol_metric[KS_BETAY] ; 
                id.betaz = sol_metric[KS_BETAZ] ; 

                id.gxx = sol_metric[KS_GXX] ;
                id.gxy = sol_metric[KS_GXY] ;
                id.gxz = sol_metric[KS_GXZ] ;
                id.gyy = sol_metric[KS_GYY] ;
                id.gyz = sol_metric[KS_GYZ] ;
                id.gzz = sol_metric[KS_GZZ] ;

                id.kxx = sol_metric[KS_KXX];
                id.kxy = sol_metric[KS_KXY];
                id.kxz = sol_metric[KS_KXZ];
                id.kyy = sol_metric[KS_KYY];
                id.kyz = sol_metric[KS_KYZ];
                id.kzz = sol_metric[KS_KZZ];

                //double const hm1 = get_hm1(x,y,z,M,a,r_at_max_density,r_in,gamma,kappa);

                double hm1;
                bool set_to_atmosphere=false;
                unsigned int err ; 

                if(r > r_in) {
                    {// compute hm1
                        hm1=get_hm1(x,y,z,M,a,r_at_max_density,r_in,gamma,kappa);
                        //#include <grace/physics/id/FMtorus/FMdisk_GRHD_hm1.hh>
                    }
                    if(hm1 > 0) {
                        id.rho = pow( hm1 * (gamma-1.0) / (kappa*gamma), 1.0/(gamma-1.0) ) / rho_max;
                        id.press = kappa*pow(id.rho, gamma);
                        // P = (\Gamma - 1) rho epsilon
                        double eps = id.press / (id.rho * (gamma - 1.0));
                        id.ye    = eos.ye_beta_eq__press_cold(id.press,err) ;
                        // Get rho and eps from press 
                        id.rho   = eos.rho__press_cold_ye(id.press, id.ye, err) ; 
                        id.vx=sol_vel[0];
                        id.vy=sol_vel[1];
                        id.vz=sol_vel[2];

                        // convert Eulerian velocities to coordinate velocities:
                        id.vx = id.alp * id.vx - id.betax;
                        id.vy = id.alp * id.vy - id.betay;
                        id.vz = id.alp * id.vz - id.betaz;
                        // what about zvec? 

                    } else {
                        set_to_atmosphere=true;
                    }
                    } else {
                    set_to_atmosphere=true;
                }
                // Outside the disk? Set to atmosphere all hydrodynamic variables!
                if(set_to_atmosphere) {
                    // Choose an atmosphere such that
                    //   rho =       1e-5 * r^(-3/2), and
                    //   P   = k rho^gamma
                    // Add 1e-100 or 1e-300 to rr or rho to avoid divisions by zero.
                    id.rho   = 1e-5 * pow(r + 1e-100,-3.0/2.0);
                    id.press = kappa*pow(id.rho, gamma);
                    double eps = id.press / ((id.rho + 1e-300) * (gamma - 1.0));
                    //w_lorentz[idx] = 1.0;
                    id.vx = 0.0;
                    id.vy = 0.0;
                    id.vz = 0.0;
                }
                    // extra checks?
                    // /* Check if we are inside the star */
                    // double ye_atm  = _eos.ye_atmosphere()  ; 
                    // double rho_atm = _eos.rho_atmosphere() ;
                    
                    // if ( sol[1] > 1.001 * _press_atm ) {
                    //     id.press = sol[1] ; 
                    //     id.ye    = _eos.ye_beta_eq__press_cold(sol[1],err) ;
                    //     // Get rho and eps from press 
                    //     double eps ; 
                    //     id.rho   = _eos.rho__press_cold_ye(sol[1], id.ye, err) ; 
                    // } else {
                    //     id.rho   = rho_atm   ;
                    //     id.ye    = ye_atm    ;
                    //     id.press = _press_atm ; 
                    // }
                    // id.vx = 0 ; id.vy = 0; id.vz = 0;

                return std::move(id) ; 
            }

        // arguments to the constructor: 
        eos_t   eos         ;                            //!< Equation of state object 
        grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
        double a, M;                            //!< BH spin and mass
        double rho_min, press_min ;   //! < floor!
        double lapse_min ;      //!< floor on the lapse value - should also work without it!
        double r_in;            //!< Fixes the inner edge of the disk
        double r_at_max_density; //!<Radius at maximum disk density. Needs to be > r_in
        double kappa, gamma;   //!< EOS parameters 
        // will be filled out in the initialization but before fillin the state arrays: 
        double P_max, rho_max;   //! < these will be filled in the constructor 
        };

}

/* namespace grace */

#endif /* GRACE_PHYSICS_ID_FMTORUS_HH */