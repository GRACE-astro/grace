/**
 * @file boosted_loop_advection.hh
 * @author Marie Cassing (mcassing@itp.uni-frankfurt.de)
 * @brief The classical 2D SRMHD test, where the setup follows https://arxiv.org/pdf/1611.09720
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

#ifndef GRACE_PHYSICS_ID_BOOSTED_LOOP_ADVECTION_MHD_HH
#define GRACE_PHYSICS_ID_BOOSTED_LOOP_ADVECTION_MHD_HH

#include <grace_config.h>

#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

//**************************************************************************************************
namespace grace {
//**************************************************************************************************
/**
 * \defgroup initial_data Initial Data
 * 
 */
//**************************************************************************************************
/**
 * @brief Boosted Loop Advection MHD test - a classical 2D testbed for inspecting the divB violations - MHD initial data kernel
*
 * \ingroup initial_data
 * @tparam eos_t type of equation of state
 * @note this kernel has to be checked for and adjusted if needed 
 * should the magnetic field initialization method/location changes in the future (e.g. vec pot)
 * @note only in 2D/3D problems can divergence cleaning performance of the GLM method's be judged 
 *       in 1D (and flat spacetime), the evolution of phi_glm is trivial
 */
template < typename eos_t >
struct boosted_loop_advection_mhd_id_t {
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //!< Type of state vector
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Construct a new boosted_loop_advection_mhd id kernel object
     * 
     * @param eos Equation of state
     * @param pcoords Physical coordinate array
     * @param rho density
     * @param press pressure
     * @param beta0 plasma beta
     * @param vc  velocity in terms of c
     */
     boosted_loop_advection_mhd_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rho
        , double press
        , double beta0
        , double vc
        , double B0
        , bool compensate
        )
        : _eos(eos)
        , _pcoords(pcoords)
        , _rho(rho), _press(press)
        , _beta0(beta0)
        , _vc(vc)
        , _B0(B0)
        , _compensate(compensate)
    {} 
    //**************************************************************************************************
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double bessel_J0(double x) const {
         double sum = 1.0;
         double term = 1.0;
         double x2 = x * x / 4.0;
     
         for (int k = 1; k <= 10; ++k) {
             term *= -x2 / (k * k);
             sum += term;
         }
     
         return sum;
     }
     
     GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
     double bessel_J1(double x) const {
         double sum = x / 2.0;
         double term = x / 2.0;
         double x2 = x * x / 4.0;
     
         for (int k = 1; k <= 10; ++k) {
             term *= -x2 / (k * (k + 1));
             sum += term;
         }
     
         return sum;
     }
    //**************************************************************************************************
    /**
     * @brief Obtain initial data at a point
     * 
     * @param i x cell index 
     * @param j y cell index 
     * @param k z cell index
     * @param q Quadrant index
     * @return grmhd_id_t Initial data at this point
     */
    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 

        const double x = _pcoords(VEC(i,j,k),0,q);
        const double y = _pcoords(VEC(i,j,k),1,q);
        const double z = _pcoords(VEC(i,j,k),2,q);

        /*** 
        double const r = Kokkos::sqrt(EXPR(
              math::int_pow<2>(x),
            + math::int_pow<2>(y),
            + math::int_pow<2>(z)
        )) ; 
        */
        double const r = Kokkos::sqrt(math::int_pow<2>(x)+ math::int_pow<2>(y)) ; 
        double const phi = Kokkos::atan2(y, x);


        id.rho = _rho;
        id.press = _press;
        // in paper it is p/2
        //id.rho = _press/2.0; 
        // beta0 = B^2(0)/2p

        const double alphat = 3.8317;
        const double Cconst = 0.01;

        // cylindrical coordinates
        double Bphi = 0.0;
        double Bz = 0.0;
        double Br = 0.0;


        //BFF = c0 alpha(J1(alphaR)^phi_kill + J0(alphaR)^z_kill)
       //----- cylindrical coordinates
       Br = 0;
       if(r<1){
         Bphi= bessel_J1(alphat*r);
         Bz = Kokkos::sqrt(math::int_pow<2>(bessel_J0(alphat*r)) + Cconst);
       }
       else{
         Bphi = 0.0;
         Bz = Kokkos::sqrt(math::int_pow<2>(bessel_J0(alphat*r)) + Cconst);
       }

       /*
       // cylinder to cartesian

        std::array<std::array<double,3>,3> gij {
            std::array<double,3>{1., 0, 0},
            std::array<double,3>{0,r*r,0},
            std::array<double,3>{0,0,0}
        } ; 

        std::array<std::array<double,3>,3> gpij {
            std::array<double,3>{0,0,0},
            std::array<double,3>{0,0,0},
            std::array<double,3>{0,0,0}
        } ;
         
        std::array<std::array<double,3>,3> J {
            std::array<double,3>{Kokkos::cos(phi), -r * Kokkos::sin(phi), 0.},
            std::array<double,3>{Kokkos::sin(phi),  r * Kokkos::cos(phi), 0.},
            std::array<double,3>{0., 0., 1.}
        };

        // ok actually dont need this:
        for( int ii=0; ii<3; ++ii){
            for( int jj=0; jj<3; ++jj){
                for( int ll=0; ll<3; ++ll ) {
                    for( int kk=0; kk<3; ++kk) {
                        gpij[ii][jj] += 
                            J[ll][ii] * J[kk][jj] * gij[ll][kk] ; 
                    }
                }
            }
        }

       */

       // ------------------------------
       //transform Bphi and Bz to Bx,y,z
       // cylinder to cartesian
 
       double Bx_cart = 0.0;
       double By_cart = 0.0;
       double Bz_cart = 0.0;

       Bx_cart = Br * Kokkos::cos(phi) - Bphi * Kokkos::sin(phi);
       By_cart = Br * Kokkos::sin(phi) - Bphi * Kokkos::sin(phi);
       Bz_cart = Bz; 
       
        const double zvecx = -0.99 * Kokkos::sin(y);
        const double zvecy =  0.99 * Kokkos::sin(x);
        const double A = zvecx*zvecx + zvecy*zvecy;
        const double W2 = A + 1.0 ; 
        const double W = Kokkos::sqrt(W2); 


        // convergence tests
        //id.vx = 0.5*Kokkos::sqrt(0.5)*1.0; // / W; 
        //id.vy = 0.5*Kokkos::sqrt(0.5)*1.0;  // / W;
        //id.vz = 0.;
               
        // need to rescale by W?
        id.vx = 1.0/Kokkos::sqrt(2.0)*(-1.0)*_vc; // / W; 
        id.vy = 1.0/Kokkos::sqrt(2.0)*(-1.0)*_vc;  // / W;
        id.vz = 0.;        
    
        // beta=-v ? 
        // set the Minkowski metric 
        //id.betax = sqrt(0.5); id.betay=0; id.betaz = 0; // convergence test

        // Lorentz boost of B field
        double shift_x;
        double shift_y;
        double shift_z;
 
         if(_compensate == true){
        shift_x = - id.vx; shift_y=-id.vy; shift_z = -id.vz; 
        }else{
        shift_x = 0; shift_y = 0; shift_z = 0; 
        }

        const double vx = id.vx;
        const double vy = id.vy;
        const double vz = id.vz;
        
        const double v2 = vx*vx + vy*vy + vz*vz;
        const double gamma = 1.0 / Kokkos::sqrt(1.0 - v2);

        const double betaDotB = shift_x*Bx_cart + shift_y*By_cart + shift_z*Bz_cart;

        /*
        printf("debug id \n");
        printf("vx %g vy %g vz %g \n",vx,vy,vz);
        printf("v2 %g \n",v2);
        printf("gamma %g \n",gamma);
        printf("gamma-term %g \n",(gamma / (gamma + 1.0)));
        printf("betaDotB %g \n",betaDotB);
        printf("betaDotB*shiftx %g \n",betaDotB*shift_x);
        printf("betaDotB*shiftx*gammaterm %g \n",(gamma / (gamma + 1.0)) * betaDotB * shift_x);
        printf("Bx_cart %g \n",Bx_cart);
        printf("Bx_trans %g \n",gamma * (Bx_cart - (gamma / (gamma + 1.0)) * betaDotB * shift_x));
        */

        id.bx = gamma * (Bx_cart - (gamma / (gamma + 1.0)) * betaDotB * shift_x);
        id.by = gamma * (By_cart - (gamma / (gamma + 1.0)) * betaDotB * shift_y);
        id.bz = gamma * (Bz_cart - (gamma / (gamma + 1.0)) * betaDotB * shift_z);

        id.alp = 1 ; 
        id.betax = 0; id.betay=0; id.betaz = 0; 
        id.gxx = 1; id.gyy = 1; id.gzz = 1;
        id.gxy = 0; id.gxz = 0; id.gyz = 0 ;
        id.kxx = 0; id.kyy = 0; id.kzz = 0 ;
        id.kxy = 0; id.kxz =0 ; id.kyz = 0 ; 

        // set the Lagrange multiplier to zero
        #ifdef GRACE_ENABLE_B_FIELD_GLM
        id.phi_glm = 0.;
        #endif 

        unsigned int err ; 
        id.ye  = _eos.ye_beta_eq__press_cold(id.press, err);
        return std::move(id) ; 
    }
    //**************************************************************************************************

    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rho, _press, _beta0, _vc, _B0;                    //!< Left and right states  
    bool _compensate;
    
    //**************************************************************************************************
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
//**************************************************************************************************
#endif /* GRACE_PHYSICS_ID_BOOSTED_LOOP_ADVECTION_MHD_HH */