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
#include <grace/utils/device.h>

#include <grace/utils/runge_kutta.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/errors/error.hh>

#include <Kokkos_Core.hpp>


namespace grace {

        //**************************************************************************************************/
        /* Auxiliaries */
        //**************************************************************************************************/
        /**
         * @brief Helper indices for getting the metric and extrinsic curvature components
         */
        enum KS_spacetime {
            KS_ALPHA = 0,
            KS_BETAX,
            KS_BETAY,
            KS_BETAZ,
            KS_GXX,
            KS_GXY,
            KS_GXZ,
            KS_GYY,
            KS_GYZ,
            KS_GZZ,
            KS_KXX,
            KS_KXY,
            KS_KXZ,
            KS_KYY,
            KS_KYZ,
            KS_KZZ,
            KS_NUM_COMPS
        } ; 

        /**
     * @brief Return the Kerr-Schild metric in Kerr-Schild cartesian coordinates at location x,y,z
     * 
     * @param xcoord 1st coordinate 
     * @param ycoord 2nd coordinate 
     * @param zcoord 3rd coordinate 
     * @returns std::array<double,16> for alpha, beta^i, g_ij, K_ij components 
     */

    // needs M, a 
    std::array<double,KS_NUM_COMPS> GRACE_HOST_DEVICE
    get_KS_metric(double const xcoord, double const ycoord, double const zcoord) const
    {
        #include <grace/include/physics/id/FMtorus/KerrSchild.hh>

        std::arrray<double,KS_NUM_COMPS> loc_metric_state;
        loc_state[KS_ALPHA]=sqrt(FDPart3_15);
        loc_state[KS_BETAX]=2*FDPart3_15*FDPart3_16*FDPart3_9;
        loc_state[KS_BETAY]= FDPart3_19*FDPart3_20;
        loc_state[KS_BETAZ]= FDPart3_20*M*zcoord;
        loc_state[KS_GXX]=FDPart3_1*FDPart3_2*FDPart3_50 + FDPart3_2*FDPart3_34*FDPart3_6 + FDPart3_26*FDPart3_3*FDPart3_33*FDPart3_43 - FDPart3_41;
        loc_state[KS_GXY]=FDPart3_1*FDPart3_35*FDPart3_50 + FDPart3_2*FDPart3_33*FDPart3_36*FDPart3_39 - FDPart3_3*FDPart3_33*FDPart3_36*FDPart3_39 + FDPart3_35*FDPart3_51 - FDPart3_35*FDPart3_53;
        loc_state[KS_GXZ]=FDPart3_51*xcoord*zcoord - FDPart3_58*ycoord - FDPart3_62*xcoord*zcoord;
        loc_state[KS_GYY]=FDPart3_1*FDPart3_3*FDPart3_50 + FDPart3_2*FDPart3_53 + FDPart3_3*FDPart3_51 + FDPart3_41;
        loc_state[KS_GYZ]=FDPart3_51*ycoord*zcoord + FDPart3_58*xcoord - FDPart3_62*ycoord*zcoord;
        loc_state[KS_GZZ]=FDPart3_34*FDPart3_7 + FDPart3_48*FDPart3_64;
        loc_state[KS_KXX]=FDPart3_2*FDPart3_6*FDPart3_94 + FDPart3_66*FDPart3_79 + FDPart3_66*FDPart3_83 + FDPart3_70*FDPart3_99 + FDPart3_88 + FDPart3_92;
        loc_state[KS_KXY]=4*FDPart3_102*FDPart3_46 + FDPart3_102*FDPart3_82 - FDPart3_105*FDPart3_2*FDPart3_36 + FDPart3_105*FDPart3_3*FDPart3_36 + FDPart3_106*FDPart3_91 - FDPart3_12*FDPart3_35*FDPart3_76*FDPart3_98 - FDPart3_69*FDPart3_87*M + FDPart3_70*FDPart3_87*M;
        loc_state[KS_KXZ]=FDPart3_106*FDPart3_90*zcoord + FDPart3_107*FDPart3_109*FDPart3_90 - FDPart3_107*FDPart3_110*FDPart3_90 - FDPart3_111*FDPart3_90 + 2*FDPart3_112*FDPart3_19 - FDPart3_113*FDPart3_114;
        loc_state[KS_KYY]=FDPart3_116*FDPart3_79 + FDPart3_116*FDPart3_83 + FDPart3_3*FDPart3_6*FDPart3_94 + FDPart3_69*FDPart3_99 - FDPart3_88 - FDPart3_92;
        loc_state[KS_KYZ]=FDPart3_106*FDPart3_113*zcoord + FDPart3_107*FDPart3_109*FDPart3_113 - FDPart3_107*FDPart3_110*FDPart3_113 - FDPart3_111*FDPart3_113 - 2*FDPart3_112*FDPart3_16 + FDPart3_114*FDPart3_90;
        loc_state[KS_KZZ]=-FDPart3_1*FDPart3_37*FDPart3_61*FDPart3_83 + 4*FDPart3_5*FDPart3_64*FDPart3_77 + FDPart3_7*FDPart3_94;
            
        return loc_metric_state;
    }

    // needs a, M, r_at_max_density
    std::array<double, 3> GRACE_HOST_DEVICE
    get_fluid_velocities(double const xcoord, double const ycoord, double const zcoord) const 
    {
        std::array<double,3> vel;
        #include <grace/include/physics/id/FMtorus/KerrSchild.hh>
        vel[0]= FDPart3_10*FDPart3_12*xcoord - FDPart3_10*FDPart3_31*ycoord;
        vel[1] = FDPart3_10*FDPart3_12*ycoord + FDPart3_10*FDPart3_31*xcoord;
        vel[2] = FDPart3_10*FDPart3_12*zcoord;
        return vel;
    }


    double GRACE_HOST_DEVICE
    get_rho_initial(double const xcoord, double const ycoord, double const zcoord) const 
    {
        double rho_init;
        #include <grace/include/physics/id/FMtorus/FMdisk_GRHD_rho_initial.hh>
        rho_init = exp(log((gamma - 1)*(sqrt(FDPart3_24*(FDPart3_25 + 1)/(FDPart3_20*FDPart3_22))*exp((1.0/2.0)*FDPart3_15 - FDPart3_18*FDPart3_26/FDPart3_24 - 1.0/2.0*FDPart3_25 + FDPart3_26*FDPart3_3/FDPart3_7)/sqrt(FDPart3_7*(FDPart3_15 + 1)/(FDPart3_6*((r_in)*(r_in)))) - 1)/(gamma*kappa))/(gamma - 1));
        return rho_init;
    }


    double GRACE_HOST_DEVICE
    get_hm1(double const xcoord, double const ycoord, double const zcoord) const 
    {
        double hm1;
        #include <grace/include/physics/id/FMtorus/FM_disk_GRHD_hm1.hh>
        hm1 = exp(log((gamma - 1)*(sqrt(FDPart3_24*(FDPart3_25 + 1)/(FDPart3_20*FDPart3_22))*exp((1.0/2.0)*FDPart3_15 - FDPart3_18*FDPart3_26/FDPart3_24 - 1.0/2.0*FDPart3_25 + FDPart3_26*FDPart3_3/FDPart3_7)/sqrt(FDPart3_7*(FDPart3_15 + 1)/(FDPart3_6*((r_in)*(r_in)))) - 1)/(gamma*kappa))/(gamma - 1));
        return hm1;
    }


    /**
     * @brief FMtorus initial data kernel.
     * 
     * @tparam eos_t Eos type
     *         Due to the setup of the ID, the template
     *         specializes exclusively for hybrid eos and a specific setup 
     *         of the polytropic eos 
     */
    template < typename eos_t >
    //requires eos_t=grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>
    requires eos_t=grace::piecewise_polytropic_eos_t
    struct fmtorus_id_t {
        using state_t = grace::var_array_t<GRACE_NSPACEDIM> ;

        // constructor: initializing the parameters, performing consistency checks on the ID
        fmtorus_id_t(
            eos_t eos_,
            grace::coord_array_t<GRACE_NSPACEDIM> pcoords_,
            double a_, double M_, 
            double rho_min_, double press_min_, 
            double lapse_min_, double r_in_, double r_at_max_density_
            double kappa_, double gamma_ )
            : 
            eos(eos_), pcoords(pcoords_), 
            rho_min(rho_min_), press_min(press_min_),
            lapse_min(lapse_min_), r_in(r_in_), r_at_max_density(r_at_max_density_),
            kappa(kappa_),gamma(gamma_)
        { 
            GRACE_INFO("In FMTorus setup.") ; 
            // consistency checks:
            if(r_in >= r_at_max_density) ERROR("r_in needs to be smaller than r_at_max_density!");
            if(a < 0.0) ERROR("Negative spins not supported! The setup of fluid profiles assumes the positive sign.");
            if(M <= 0.0) ERROR("M zero or negative");

            GRACE_INFO("Using input parameters of\n 
                    a = {}\n, 
                    M = {},\n
                    r_in = {},\n
                    r_at_max_density ={},\n
                    kappa = {}\n
                    gamma = {}",
                    a,M,r_in,r_at_max_density,kappa,gamma);

            // First compute maximum pressure and density
            double P_max, rho_max;
            {
                double hm1;
                double xcoord = r_at_max_density;
                double ycoord = 0.0;
                double zcoord = 0.0;
                    {
                #include <grace/include/physics/id/FMtorus/FM_disk_GRHD_hm1.hh>
                    }
                rho_max = pow( hm1 * (gamma-1.0) / (kappa*gamma), 1.0/(gamma-1.0) );
                P_max   = kappa * pow(rho_max, gamma);
            }

            // We enforce units such that rho_max = 1.0; if these units are not obeyed, then
            //    we error out. If we did not error out, then the value of kappa used in all
            //    EOS routines would need to be changed, and generally these appear as
            //    read-only parameters.
            if(fabs(P_max/rho_max - kappa) > 1e-8) {
                ERROR("Error: To ensure that P = kappa*rho^Gamma, where rho_max = 1.0,\n,
                    you must set (in your parfile) the polytropic constant kappa = P_max/rho_max = {} \n\n,
                    Needed values for kappa, for common values of Gamma:\n
                    For Gamma=4/3, use kappa=K_initial=K_poly = 4.249572342020724e-03 to ensure rho_max = 1.0,\n
                    For Gamma=5/3, use kappa=K_initial=K_poly = 6.799315747233158e-03 to ensure rho_max = 1.0,\n
                    For Gamma=2, use kappa=K_initial=K_poly = 8.499144684041449e-03 to ensure rho_max = 1.0\n", P_max/rho_max);
            }

        } 

        // the main evaluation kernel as operator()
        grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        operator() (VEC(int i, int j, int k), int q) const 
        {
            double const x = pcoords(VEC(i,j,k),0,q);
            double const y = pcoords(VEC(i,j,k),1,q);
            #ifdef GRACE_3D 
            double const z = pcoords(VEC(i,j,k),2,q);
            #else 
            double const z = 0. ; 
            #endif 
            double const r = Kokkos::sqrt(EXPR(
                math::int_pow<2>(x),
                + math::int_pow<2>(y),
                + math::int_pow<2>(z)
            )) ; 

            // at this radius.
            auto sol_metric = get_KS_metric(x,y,z) ;
            auto sol_vel = get_fluid_velocities(x,y,z) ;
            double const hm1 = get_hm1(x,y,z);
            double rho_init = get_rhoinit(x,y,z);
            grmhd_id_t id ; 

              /* Set the metric */
            id.alp = std::max(alp_min,sol_metric[KS_ALPHA]) ; 

            id.betax = sol_metric[KS_BETAX] ; 
            id.betay = sol_metric[KS_BETAX] ; 
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

            double hm1;
            bool set_to_atmosphere=false;

            if(r > r_in) {
                {// compute hm1
                    #include <grace/include/physics/id/FMtorus/FM_disk_GRHD_hm1.hh>
                }
                if(hm1 > 0) {
                    id.rho = pow( hm1 * (gamma-1.0) / (kappa*gamma), 1.0/(gamma-1.0) ) / rho_max;
                    id.press = kappa*pow(id.rho, gamma);
                    // P = (\Gamma - 1) rho epsilon
                    id.eps = id.press / (id.rho * (gamma - 1.0));
                    id.vx=sol_vel[0];
                    id.vy=sol_vel[1];
                    id.vz=sol_vel[2];

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
                id.eps   = id.press / ((id.rho + 1e-300) * (gamma - 1.0));
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


        eos_t   _eos         ;                            //!< Equation of state object 
        grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
        double a, M;                            //!< BH spin and mass
        double rho_min, press_atm ;   //! < floor!
        double lapse_min ;      //!< floor on the lapse value - should also work without it!
        double r_in;            //!< Fixes the inner edge of the disk
        double r_at_max_density; //!<Radius at maximum disk density. Needs to be > r_in
        double kappa, gamma;   //!< EOS parameters 
        //Kokkos::View<double *, grace::default_space> mass, press, nu, r, dr ;  
    };

}
/* namespace grace */

#endif /* GRACE_PHYSICS_ID_FMTORUS_HH */