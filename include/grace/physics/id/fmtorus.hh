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
#include <grace/physics/id/FMtorus/KerrSchild.hh>
#include <grace/physics/id/FMtorus/FMdisk_GRHD_velocities.hh>
#include <grace/physics/id/FMtorus/FMdisk_GRHD_rho_initial.hh>
#include <grace/physics/id/FMtorus/FMdisk_GRHD_hm1.hh>
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
            using state_t = grace::var_array_t<GRACE_NSPACEDIM> ;

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
                double random_min_, double random_max_,
                double kappa_, double gamma_,
                bool set_Bfield_from_Avec,
                int Avec_type,
                int Avec_prescription,
                double Avec_Pcut,
                int Avec_n,
                double Avec_Ab
            )
                : \
                eos(eos_), pcoords(pcoords_), 
                M(M_), a(a_),
                rho_min(rho_min_), press_min(press_min_),
                lapse_min(lapse_min_), r_in(r_in_), r_at_max_density(r_at_max_density_),
                kappa(kappa_),gamma(gamma_),
                rand_pool(rand_pool_),
                random_min(random_min_), random_max(random_max_),
                _set_Bfield_from_Avec(set_Bfield_from_Avec),
                _Avec_type(Avec_type),
                _Avec_prescription(Avec_prescription),
                _Avec_Pcut(Avec_Pcut),
                _Avec_n(Avec_n),
                _Avec_Ab(Avec_Ab)
            { 
                GRACE_INFO("In FMTorus setup. Make sure kappa of the ID-FMtorus matches the EOS kappa!") ; 
                // consistency checks:
                if(r_in >= r_at_max_density) ERROR("r_in needs to be smaller than r_at_max_density!");
                if(a < 0.0) ERROR("Negative spins not supported! The setup of fluid profiles assumes the positive sign.");
                if(M <= 0.0) ERROR("M zero or negative");

                GRACE_INFO("Using input parameters of\n  \
                        a = {}\n,  \
                        M = {},\n \
                        lapse_min = {},\n \
                        r_in = {},\n \
                        r_at_max_density ={},\n \
                        kappa = {}\n \
                        gamma = {}", \
                        a,M,lapse_min,r_in,r_at_max_density,kappa,gamma);
   
                // First compute maximum pressure and density (rho_max and P_max will be filled as members)
                {
                    double hm1;
                    double xcoord = r_at_max_density;
                    double ycoord = 0.0;
                    double zcoord = 0.0;
                    //     {
                    // #include <grace/physics/id/FMtorus/FMdisk_GRHD_hm1.hh>
                    //     }
                    hm1=get_hm1(xcoord,ycoord,zcoord,M,a,r_at_max_density,r_in,gamma,kappa);

                    rho_max = pow( hm1 * (gamma-1.0) / (kappa*gamma), 1.0/(gamma-1.0) );
                    P_max   = kappa * pow(rho_max, gamma);
                }
                
                GRACE_INFO("Pmax:{}, rhomax:{}",P_max, rho_max);

                // We enforce units such that rho_max = 1.0; if these units are not obeyed, then
                //    we error out. If we did not error out, then the value of kappa used in all
                //    EOS routines would need to be changed, and generally these appear as
                //    read-only parameters.
                
                if(fabs(P_max/rho_max - kappa) > 1e-8) {
                    GRACE_INFO("Error: To ensure that P = kappa*rho^Gamma, where rho_max = 1.0,\n, \
                        you must set your kappa correctly to ensure that for rho_max=1.0, P_max = kappa; right now, Pmax, rhomax, ratio: {}, {}, {} \n\n, \
                        Needed values for kappa, for common values of Gamma (setup as in Etienne's FishboneMoncrief_IllinoisGRMHD-production_run-HIGHRES.par):\n \
                        For Gamma=4/3, use kappa=K_initial=K_poly = 4.249572342020724e-03 to ensure rho_max = 1.0,\n \
                        For Gamma=5/3, use kappa=K_initial=K_poly = 6.799315747233158e-03 to ensure rho_max = 1.0,\n \
                        For Gamma=2, use kappa=K_initial=K_poly = 8.499144684041449e-03 to ensure rho_max = 1.0\n", P_max, rho_max, P_max/rho_max);
                    ERROR("Aborting the run");
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
                double const r = Kokkos::sqrt(EXPR(math::int_pow<2>(x), + math::int_pow<2>(y), + math::int_pow<2>(z) )) ; 

                const double r_eps = 1e-6;
                // AthenaK trick
                const double bigR = Kokkos::sqrt((math::int_pow<2>(r)-math::int_pow<2>(a)+Kokkos::sqrt(math::int_pow<2>(math::int_pow<2>(r)-math::int_pow<2>(a))+4.0*math::int_pow<2>(a)*math::int_pow<2>(z)))/2.0);

                // if (r < eps) {
                //     r = 0.5*(eps + r*r/eps);
                // }

                // at this radius.
                auto sol_metric = get_KS_metric(x,y,z,M,a) ;
                // double rho_init = get_rho_initial(x,y,z,M,a,r_at_max_density,r_in,gamma,kappa);
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

                // grace::metric_array_t metric{{id.gxx,id.gxy,id.gxz,id.gyy,id.gyz,id.gzz},
                //                              {id.betax,id.betay,id.betaz},
                //                               id.alp };
                // const double sqrtdetgamma = metric.sqrtg();
                // if(sqrtdetgamma <=0 ) printf("Metric not positive-definite at (x,y,z)=(%1.8f,%1.8f,%1.8f) \n", x,y,z);
                // if(id.alp <=0 ) printf("Alpha non-positive at (x,y,z)=(%1.8f,%1.8f,%1.8f) \n", x,y,z);
                //double const hm1 = get_hm1(x,y,z,M,a,r_at_max_density,r_in,gamma,kappa);

                double hm1;
                bool set_to_atmosphere=false;
                unsigned int err ; 

                if(r > r_in) {
                        // compute hm1
                        hm1=get_hm1(x,y,z,M,a,r_at_max_density,r_in,gamma,kappa);

                        if(hm1 > 0) {
                            id.rho = pow( hm1 * (gamma-1.0) / (kappa*gamma), 1.0/(gamma-1.0) ) / rho_max;
                            id.press = kappa*pow(id.rho, gamma);
                            // P = (\Gamma - 1) rho epsilon
                            // double eps = id.press / (id.rho * (gamma - 1.0));
                            // double ye_atm  = eos.ye_atmosphere()  ; 
                            // id.ye = ye_atm;
                            // id.ye    = eos.ye_beta_eq__press_cold(id.press,err) ;
                            // Get rho and eps from press  ? is this needed?
                            // id.rho   = eos.rho__press_cold_ye(id.press, id.ye, err) ; 
                            auto sol_vel = get_fluid_velocities(x,y,z,M,a,r_at_max_density) ;

                            id.vx=sol_vel[0];
                            id.vy=sol_vel[1];
                            id.vz=sol_vel[2];
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


                // finally, set the vector potential (for now at cell centres)
                // the conversion to the B field cell-centred values happens outside of the loop
                if(_set_Bfield_from_Avec){
                    double AvecX{0.}, AvecY{0.}, AvecZ{0.};
                    /*============================================================*/
                    // The following sets up a vector potential of the form
                    // A_\phi = Rcyl^2 A_b max[(EE-EE_cut),0],
                    // where Rcyl is the cylindrical radius: sqrt(x^2+y^2),
                    // and EE \in {\rho, P} is the variable P or rho, specifying
                    //   whether the vector potential is proportional to P or rho
                    //   in the region greater than the cutoff.
                    // (see e.g https://arxiv.org/pdf/astro-ph/0510653.pdf)
                    /*============================================================*/
                    // This formulation assumes that A_r and A_\theta = 0;
                    // only A_\phi can be nonzero. The coordinate
                    // transformations are given by:
                    /*============================================================*/
                    // A_x = dphi/dx A_phi
                    //     = d[atan(y/x)]/dx A_phi
                    //     = -y/(x^2+y^2) A_phi
                    //     = -y/Rcyl^2 A_phi
                    // A_y = dphi/dy A_phi
                    //     = d[atan(y/x)]/dy A_phi
                    //     =  x/(x^2+y^2) A_phi
                    //     =  x/Rcyl^2 A_phi
                    /*============================================================*/

                    if(_Avec_type==0){ // 
                        // create an azimuthal vector potential A_phi (encoded via Ax, Ay); toroidal symmetry around the object
                        // magnetic field lines generated from a poloidal vector potential using B = ∇ × A will lie in meridional planes (r–theta planes in spherical coordinates), 
                        // forming closed loops around the axis of symmetry
                        if(_Avec_prescription==0){
                            const double press = id.press ;
                            // Avec_Ab needs to be picked so that the plasma beta is satisfatory
                            // for the user 
                            AvecX = -y * _Avec_Ab * pow(std::max(press - _Avec_Pcut, 0.0),  _Avec_n); 
                            AvecY =  x * _Avec_Ab * pow(std::max(press - _Avec_Pcut, 0.0),  _Avec_n); 
                            const double Rcyl2 = x*x + y*y;
                            // add radial fall-off for the vector potential or is pressure decrease enough to model the fall of?
                            // AvecX /= Rcyl2;
                            // AvecY /= Rcyl2;
                        }
                    }

                    id.ax = AvecX;
                    id.ay = AvecY;
                    id.az = AvecZ;
                    id.phi_em = 0.0;
                }

                // Add white noise  
                // note that this version, similar to ETK-Fishbone-MoncriefID, sets up the magnetic field
                // based on the unperturbed pressure values

                if(abs(random_min)>1e-6 || abs(random_max)>1e-6){
                    // Seed the random number generator
                    // random number 
                    auto generator = rand_pool.get_state();
                    const double random_number_between_0_and_1 = generator.drand();; 
                    const double random_number_between_min_and_max = random_min + (random_max - random_min)*random_number_between_0_and_1;
                    id.press = id.press*(1.0 + random_number_between_min_and_max);
                    // Add 1e-300 to rho to avoid division by zero when density is zero.
                    // eps[idx] = press[idx] /     ((rho[idx] + 1e-300) * (gamma - 1.0));
                    //recover modified density to be consistent
                    double ye_atm  = eos.ye_atmosphere()  ; 
                    id.ye = ye_atm;
                    id.rho   = eos.rho__press_cold_ye(id.press, id.ye, err) ; 
                    rand_pool.free_state(generator);
                }
                    
                return std::move(id) ; 
            }

        // arguments to the constructor: 
        eos_t   eos         ;                            //!< Equation of state object 
        grace::coord_array_t<GRACE_NSPACEDIM> pcoords ;  //!< Physical coordinates of cell centers
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
        bool _set_Bfield_from_Avec ;
        int _Avec_type ; // 0 - poloidal, 1 - dipole, monopole, linear (e.g. for shocktubes)
        int _Avec_prescription ;  // 0-pressure, 1 - density based
        double _Avec_Pcut ;
        int _Avec_n ;
        double _Avec_Ab ;
        /*============================================================*/

        };

}

/* namespace grace */

#endif /* GRACE_PHYSICS_ID_FMTORUS_HH */