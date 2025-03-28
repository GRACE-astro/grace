/**
 * @file gowdy_spacetime.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-11-25
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
#ifndef GRACE_PHYSICS_ID_GOWDY_SPACETIME_HH
#define GRACE_PHYSICS_ID_GOWDY_SPACETIME_HH

#include <grace_config.h>
#include <cmath>

#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
//**************************************************************************************************
namespace grace {
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Initial data for vacuum gowdy test
 * \ingroup initial_data
 * @tparam eos_t Type of equation of state
 */
//**************************************************************************************************
template < typename eos_t >
struct gowdy_spacetime_id_t {
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //! Type of state array
    //**************************************************************************************************
    /**
     * @brief Construct a gowdy_test_id kernel
     * 
     * @param eos Equation of state
     * @param pcoords physical coordinates
     * @param rho perturbation scale
     */
    gowdy_spacetime_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rho)
        : _eos(eos)
        , _pcoords(pcoords)
        , _rho(rho)
    {} 
    
    // Bessel J0
    //**************************************************************************************************
    KOKKOS_INLINE_FUNCTION
    static double bessel_j0(double x) {
        double sum = 1.0;
        double term = 1.0;
        double x2 = (x * x) / 4.0; 

        for (int k = 1; k < 100; k++) { // 100 terms of the expansion
            term *= -x2 / (k * k);
            sum += term;
            if (fabs(term) < 1e-15) break; 
        }
        return sum;
    }
    //**************************************************************************************************
    
    // Bessel J1
    //**************************************************************************************************
    KOKKOS_INLINE_FUNCTION
    static double bessel_j1(double x) {
        double sum = x / 2.0;
        double term = sum;
        double x2 = (x * x) / 4.0;

        for (int k = 1; k < 100; k++) { // 100 terms of the expansion
            term *= -x2 / (k * (k + 1));
            sum += term;
            if (fabs(term) < 1e-15) break;
        }
        return sum;
    }
    //**************************************************************************************************
    
    
    //**************************************************************************************************
    //**************************************************************************************************
    /**
     * @brief Return gowdy test initial data at a point
     * 
     * @param i x cell index
     * @param j y cell index
     * @param k z cell index
     * @param q quadrant index
     * 
     * @return grmhd_id_t Initial data objects for gowdy test
     */
    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        double const z = _pcoords(VEC(i,j,k),2,q) ;
        grmhd_id_t id ; 
        id.rho   = 0 ; 
        id.press = 0 ;
        id.ye    = 0 ;
        id.vx = id.vy = id.vz = 0 ;

        id.betax = id.betay = id.betaz = 0 ;
        
        double pi2  = 2 * M_PI ;
        double pi4  = 4 * M_PI ;
        double cosz = Kokkos::cos( pi2 * z ) ; 
        
        double t0 = -9.87532058290980 ; // 20th zero of the Bessel function (J0(2*pi*t0)=J0(2*pi*t)=0) 
        double k0 = 9.67076981276380 ; 
        double c0 = 0.00211951192146170 ; 

        double J0 = 0.220276908539934 ; //bessel_j0( pi2 ) ;
        double J1 = -0.212382530076369 ; //bessel_j1( pi2 ) ;
                
        // backward case
        //double const t  = k0 * std::exp( -c0 * t0 ) ; //t=9.87532058290966
        //double J0t = 0.0 ; // bessel_j0( pi2 * t ) ;
        //double J1t = -0.101293498933945 ; //bessel_j1( pi2 * t ) ;
                
        // forward case
        double t = 1.0 ;
        double J0t = J0 ; // bessel_j0( pi2 * t ) ;
        double J1t = J1 ; //bessel_j1( pi2 * t ) ;    

        // printf("%f, %f, %f, %f\n", t, J0t, J1, J1t);
        
        double P = J0t * cosz ;
        
        double dtP = - pi2 * J1t * cosz ;
        
        double lambda = - pi2 * t * J0t * J1t * cosz*cosz + 0.5 * pi2*pi2 *t*t * ( J0t*J0t + J1t*J1t ) 
                        - 0.5 * ( pi2*pi2 * ( J0*J0 + J1*J1 ) - pi2 * J0 + J1 ) ;  
                                                 
        double dtlambda = - pi2 * J0t * J1t * cosz*cosz + pi2*pi2 * t * J1t*J1t * cosz*cosz
                          - pi2 * t * J0t * cosz*cosz* ( pi2 * J0t - (1.0/ t) * J1t )
                          + pi2*pi2 * t * ( J0t*J0t + J1t*J1t ) + 0.5 * pi2*pi2 * t*t* ( -2 * pi2 * J0t * J1t
                          + 2 * J1t * ( pi2 * J0t - (1.0/ t) * J1t ) ) ;

        id.gxx = 1.0;//t * std::exp(P) ;
        id.gxy = 0.0 ;
        id.gxz = 0.0 ;
        id.gyy = 1.0;//t * std::exp(-P) ;
        id.gyz = 0.0 ;
        id.gzz = 1.0;//std::exp( lambda/2.0 ) / std::sqrt( t ) ; 

        // backward case
        //id.alp = -c0 * std::pow( k0, 3.0/4.0 ) * std::exp( -( 3.0/4.0 ) * c0 * t + lambda /4.0 ) ; 
        
        // forward case !physical gamma is needed
        id.alp = 1.0;//std::pow( id.gzz, 1.0/2.0 ) ;

        id.kxx = 0.0;//- 0.5 * std::pow( t, 1.0/4.0 ) * std::exp( -lambda/4.0 ) * std::exp( P ) * ( 1.0 + t * dtP ) ;
        id.kxy = 0.0 ;
        id.kxz = 0.0 ;
        id.kyy = 0.0;//- 0.5 * std::pow( t, 1.0/4.0 ) * std::exp( -lambda/4.0 ) * std::exp( -P ) * ( 1.0 - t * dtP ) ;
        id.kyz = 0.0 ;
        id.kzz = 0.0;//1.0/4.0 * std::pow( t, -1.0/4.0 ) * std::exp( lambda/4.0 ) * ( ( 1.0/ t ) - dtlambda ) ;
        
        return std::move(id) ; 
    }
    //**************************************************************************************************

    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rho ;                                     //!< Amplitude rho
    //**************************************************************************************************  
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
//**************************************************************************************************
#endif 
