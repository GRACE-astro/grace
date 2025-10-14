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

#include <grace/utils/grace_utils.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/utils/runge_kutta.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/errors/error.hh>

// FMtorus-related includes:
#include <grace/physics/id/FMTorus/KerrSchild.hh>
#include <grace/physics/id/FMTorus/FMdisk_GRHD_velocities.hh>
#include <grace/physics/id/FMTorus/FMdisk_GRHD_rho_initial.hh>
#include <grace/physics/id/FMTorus/FMdisk_GRHD_hm1.hh>
// Kokkos utils and random
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>


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
        template < typename eos_t> //, typename B_field_config_t >
        //requires PiecewisePolytropicEOS<eos_t>
        struct fmtorus_id_t {
            using state_t = grace::var_array_t ;

            // store B-field config
            // B_field_config_t Bfield_config;

            // constructor: initializing the parameters, performing consistency checks on the ID
            fmtorus_id_t(
                eos_t eos_,
                grace::coord_array_t<GRACE_NSPACEDIM> pcoords_,
                double a_, double M_, 
                double rho_min_, double press_min_, 
                double lapse_min_, double r_in_, double r_at_max_density_,
                const Kokkos::Random_XorShift64_Pool<>& rand_pool_,
                double random_min_, double random_max_, double gamma_
            )
                : \
                eos(eos_), _pcoords(pcoords_), 
                M(M_), a(a_),
                rho_min(rho_min_), press_min(press_min_),
                lapse_min(lapse_min_), r_in(r_in_), r_at_max_density(r_at_max_density_),
                gamma(gamma_),
                rand_pool(rand_pool_),
                random_min(random_min_), random_max(random_max_)
            { 
                GRACE_INFO("In FMTorus setup. Make sure kappa of the ID-FMtorus matches the EOS kappa!") ; 
                // consistency checks:
                if(r_in >= r_at_max_density) ERROR("r_in needs to be smaller than r_at_max_density!");
                if(a < 0.0) ERROR("Negative spins not supported! The setup of fluid profiles assumes the positive sign.");
                if(M <= 0.0) ERROR("M zero or negative");
   
                // First compute maximum pressure and density (rho_max and P_max will be filled as members)
                {
                    double hm1;
                    double xcoord = r_at_max_density;
                    double ycoord = 0.0;
                    double zcoord = 0.0;
                    
                    hm1=get_hm1(xcoord,ycoord,zcoord,M,a,r_at_max_density,r_in);

                    rho_max = 1,0; //pow( hm1 * (gamma-1.0) / (kappa*gamma), 1.0/(gamma-1.0) );
                    kappa = hm1 * (gamma-1)/gamma;
                    P_max   = kappa * pow(rho_max, gamma);
                    
                }
                GRACE_INFO("Using input parameters of\n  \
                        a = {}\n,  \
                        M = {},\n \
                        lapse_min = {},\n \
                        r_in = {},\n \
                        r_at_max_density ={},\n \
                        kappa = {}\n \
                        gamma = {}", \
                        a,M,lapse_min,r_in,r_at_max_density,kappa,gamma);
                GRACE_INFO("Pmax: {}, rhomax: {}",P_max, rho_max);

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
                double const r = Kokkos::sqrt(EXPR(math::int_pow<2>(x), + math::int_pow<2>(y), + math::int_pow<2>(z) )) ; 

                const double r_eps = 1e-6;
                // AthenaK trick
                const double bigR = Kokkos::sqrt((math::int_pow<2>(r)-math::int_pow<2>(a)+Kokkos::sqrt(math::int_pow<2>(math::int_pow<2>(r)-math::int_pow<2>(a))+4.0*math::int_pow<2>(a)*math::int_pow<2>(z)))/2.0);

                // at this radius.
                auto sol_metric = get_KS_metric(x,y,z,M,a) ;
                grmhd_id_t id ; 

                /* Set the metric */
                id.alp = sol_metric[KS_ALPHA] ;  
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

                double hm1;
                bool set_to_atmosphere=id.alp<lapse_min;
                unsigned int err ; 

                if(r > r_in) {
                        // compute hm1
                        hm1=get_hm1(x,y,z,M,a,r_at_max_density,r_in);

                        if(hm1 > 0) {
                            id.rho = pow( hm1 * (gamma-1.0) / (kappa*gamma), 1.0/(gamma-1.0) ) / rho_max;
                            id.press = kappa*pow(id.rho, gamma);
                            auto sol_vel = get_fluid_velocities(x,y,z,M,a,r_at_max_density) ;
                            id.vx=sol_vel[0] ; 
                            id.vy=sol_vel[1] ; 
                            id.vz=sol_vel[2] ; 
                            // conversion to coordinate velocities is done outside of the kernel
                            // the above are Valencia velocities 
                            }
                        else{
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
                    id.vx = 0.0;
                    id.vy = 0.0;
                    id.vz = 0.0;
                    double ye_atm  = eos.ye_atmosphere()  ; 
                    id.ye = ye_atm;
                   // double rho_atm = _eos.rho_atmosphere() ;
                }
                
                
                // extra checks - if the atmosphere anywhere is below what the EOS / our limits support
                double rho_atm = eos.rho_atmosphere() ;

                // if(id.rho < rho_atm || r < r_in){ // if lower than atmosphere for some reason - or inside the inner disc radius
                if(id.rho < rho_atm){ // if lower than atmosphere for some reason - or inside the inner disc radius
                    id.rho = rho_atm;
                    id.press = kappa*pow(id.rho, gamma);
                    double ye_atm  = eos.ye_atmosphere()  ; 
                    id.ye = ye_atm;
                    id.vx=0;
                    id.vy=0;
                    id.vz=0;
                }

                // Add white noise  
                // note that this version, similar to ETK-Fishbone-MoncriefID, sets up the magnetic field
                // based on the unperturbed pressure values

                if(abs(random_min)>1e-6 || abs(random_max)>1e-6){
                    // Seed the random number generator
                    // random number 
                    auto generator = rand_pool.get_state();
                    const double eps = generator.drand(random_min,random_max);; 
                    //const double random_number_between_min_and_max = random_min + (random_max - random_min)*random_number_between_0_and_1;
                    id.press = id.press*(1.0 + eps);
                    // Add 1e-300 to rho to avoid division by zero when density is zero.
                    // eps[idx] = press[idx] /     ((rho[idx] + 1e-300) * (gamma - 1.0));
                    //recover modified density to be consistent
                    double ye_atm  = eos.ye_atmosphere()  ; 
                    id.ye = ye_atm;
                    id.rho   = Kokkos::pow(id.press/kappa, 1./gamma);  //eos.rho__press_cold_ye(id.press, id.ye, err) ; 
                    rand_pool.free_state(generator);
                }
                    
                return std::move(id) ; 
            }

        // arguments to the constructor: 
        eos_t   eos         ;                            //!< Equation of state object 
        grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
        /*============================================================*/
        double a, M;             //!< BH spin and mass
        double rho_min, press_min ;   //! < floor!
        double lapse_min ;      //!< floor on the lapse value - should also work without it!
        double r_in;            //!< Fixes the inner edge of the disk
        double r_at_max_density; //!<Radius at maximum disk density. Needs to be > r_in
        Kokkos::Random_XorShift64_Pool<> rand_pool;
        double random_min;
        double random_max;
        double kappa, gamma;   //!< EOS parameters 
        // will be filled out in the initialization but before fillin the state arrays: 
        double P_max, rho_max;   //! < these will be filled in the constructor 
        /*============================================================*/


        };

}

/* namespace grace */

#endif /* GRACE_PHYSICS_ID_FMTORUS_HH */