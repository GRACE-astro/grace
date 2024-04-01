/**
 * @file spherical_device_inlines.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-28
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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

#ifndef THUNDER_COORDINATES_DEVICE_SPHERICAL_INLINES_HH 
#define THUNDER_COORDINATES_DEVICE_SPHERICAL_INLINES_HH

#include <thunder_config.h>
#include <thunder/utils/device.h>
#include <thunder/utils/inline.h>
#include <thunder/data_structures/macros.hh>


static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
logical_to_physical_cart(double L, double * F, double * S, double* R, double* lcoords, double* pcoords)
{   
    EXPR(
    pcoords[0] = (lcoords[0]*2 - 1.)*L ;, 
    pcoords[1] = (lcoords[1]*2 - 1.)*L ;,
    pcoords[2] = (lcoords[2]*2 - 1.)*L ;
    )
} ; 

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
physical_to_logical_cart(double L, double * F, double * S, double *R, double* pcoords, double* lcoords)
{   
    EXPR(
    lcoords[0] = (pcoords[0]/L + 1.)/2. ;, 
    lcoords[1] = (pcoords[1]/L + 1.)/2. ;,
    lcoords[2] = (pcoords[2]/L + 1.)/2. ;
    )
} ;

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
logical_to_physical_sph(double L, double * F, double * S, double *R, double* lcoords, double* pcoords)
{   
    double const xi = (2.*lcoords[1]-1.) ; 
    #ifdef THUNDER_3D 
    double const eta = (2.*lcoords[2]-1.) ;
    #endif  
    double const one_over_rho = 1./Kokkos::sqrt(EXPR( 1.
                                                    , + xi*xi
                                                    , + eta*eta )) ;
    double const z_coeff = F[1] + S[1] * one_over_rho ;
    double const z0      = F[0] + S[0] * one_over_rho ;
    double const z = z0 + z_coeff * lcoords[0] ;
    double tmp[THUNDER_NSPACEDIM] = {VEC(z, z*xi, z*eta)};
    EXPR(
    pcoords[0] = 0. ;, 
    pcoords[1] = 0. ;,
    pcoords[2] = 0. ;
    )
    for(int ii=0; ii<THUNDER_NSPACEDIM; ++ii){ 
        for(int jj=0; jj<THUNDER_NSPACEDIM;++jj){
            pcoords[ii] += R[jj+THUNDER_NSPACEDIM*ii] * tmp[jj] ;
        } 
    }
} ; 

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
logical_to_physical_sph_log(double L, double * F, double * S, double *R, double* lcoords, double* pcoords)
{   
    double const xi = (2.*lcoords[1]-1.) ; 
    #ifdef THUNDER_3D 
    double const eta = (2.*lcoords[2]-1.) ;
    #endif
    double const one_over_rho = 1./Kokkos::sqrt(EXPR( 1.
                                                    , + xi*xi
                                                    , + eta*eta )) ;
    double const z = Kokkos::exp(S[0] + S[1]*(2.*lcoords[0]-1.)) * one_over_rho ;
    double tmp[THUNDER_NSPACEDIM] = {VEC(z,z*xi,z*eta)};  
    EXPR(
    pcoords[0] = 0. ;, 
    pcoords[1] = 0. ;,
    pcoords[2] = 0. ;
    )
    for(int ii=0; ii<THUNDER_NSPACEDIM; ++ii) {
        for(int jj=0; jj<THUNDER_NSPACEDIM;++jj){
            pcoords[ii] += R[jj+THUNDER_NSPACEDIM*ii] * tmp[jj] ; 
        }
    }
} ; 

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
physical_to_logical_sph(double L, double * F, double * S, double* R, double* pcoords, double* lcoords)
{   
    double prot[THUNDER_NSPACEDIM] = {VEC(0.,0.,0.)} ; 
    for(int ii=0; ii<THUNDER_NSPACEDIM; ++ii) {
        for(int jj=0; jj<THUNDER_NSPACEDIM;++jj){
            prot[ii] += R[jj+THUNDER_NSPACEDIM*ii] * pcoords[jj] ;
        } 
    }
    double const z = prot[0];
    double const r = Kokkos::sqrt(
        EXPR(
              prot[0]*prot[0],
            + prot[1]*prot[1],
            + prot[2]*prot[2]
        )) ; 
    double const z_coeff = F[1] + S[1]*z/r ; 
    double const z_0 = F[0] + S[0]*z/r ;
    EXPR( 
    lcoords[0] = (z-z_0)/z_coeff     ;, 
    lcoords[1] =  0.5*(prot[1]/z+1.) ;,
    lcoords[2] =  0.5*(prot[2]/z+1.) ;
    )
} ;

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
physical_to_logical_sph_log(double L, double * F, double * S, double* R, double* pcoords, double* lcoords)
{   
    double prot[THUNDER_NSPACEDIM] = {VEC(0.,0.,0.)} ; 
    for(int ii=0; ii<THUNDER_NSPACEDIM; ++ii) for(int jj=0; jj<THUNDER_NSPACEDIM;++jj){
        prot[ii] += R[jj+THUNDER_NSPACEDIM*ii] * pcoords[jj] ; 
    }
    double const z = prot[0];
    double const r = Kokkos::sqrt(
        EXPR(
              prot[0]*prot[0],
            + prot[1]*prot[1],
            + prot[2]*prot[2]
        )) ; 
    EXPR( 
    lcoords[0] =  0.5*((Kokkos::log(r) - S[0]) / S[1] + 1.);,
    lcoords[1] =  0.5*(prot[1]/z+1.)  ;,
    lcoords[2] =  0.5*(prot[2]/z+1.)  ;
    )
} ;

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
sph_negative_r_ghost_coordinates(double L, double * F, double * S, double* R, double* lcoords, double* pcoords)
{
    double const xi = (2.*lcoords[1]-1.) ; 
    #ifdef THUNDER_3D 
    double const eta = (2.*lcoords[2]-1.) ;
    #endif  
    double const one_over_rho = 1./Kokkos::sqrt(EXPR( 1.
                                                    , + xi*xi
                                                    , + eta*eta )) ;
    double const z_coeff = 1. ;
    double const z0      = F[0] + S[0] * one_over_rho ;
    double const z = z0 + z_coeff * lcoords[0] ;
    double tmp[THUNDER_NSPACEDIM] = {VEC(z, z*xi, z*eta)};
    EXPR(
    pcoords[0] = 0. ;, 
    pcoords[1] = 0. ;,
    pcoords[2] = 0. ;
    )
    for(int ii=0; ii<THUNDER_NSPACEDIM; ++ii){ 
        for(int jj=0; jj<THUNDER_NSPACEDIM;++jj){
            pcoords[ii] += R[jj+THUNDER_NSPACEDIM*ii] * tmp[jj] ;
        } 
    }
}

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
cart_to_sph_transfer(double L, double * F1, double * S1, double * F2, double * S2, double* R1, double* R2, double* lcoord_cart, double* lcoords_sph)
{
    double pcoords[3] ; 
    logical_to_physical_cart(L,F1,S1,R1,lcoord_cart,pcoords) ;
    physical_to_logical_sph(L,F2,S2,R2,pcoords,lcoords_sph) ; 
}

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
sph_to_cart_transfer(double L, double * F1, double * S1, double * F2, double * S2, double* R1, double* R2, double* lcoords_sph, double* lcoords_cart)
{
    double pcoords[3] ; 
    sph_negative_r_ghost_coordinates(L,F1,S1,R1,lcoords_sph,pcoords) ;
    physical_to_logical_cart(L,F2,S2,R1,pcoords,lcoords_cart) ; 
}

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
sph_to_sph_negative_r_transfer(double L, double * F1, double * S1, double * F2, double * S2, double* R1, double* R2, double* lcoords_sph1, double* lcoords_sph2)
{
    double pcoords[3] ; 
    sph_negative_r_ghost_coordinates(L,F1,S1,R1,lcoords_sph1,pcoords) ;
    physical_to_logical_sph(L,F2,S2,R2,pcoords,lcoords_sph2) ; 
}

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
sph_to_sph_positive_r_transfer(double L, double * F1, double * S1, double * F2, double * S2, double* R1, double* R2, double* lcoords_sph1, double* lcoords_sph2)
{
    double pcoords[3] ; 
    logical_to_physical_sph(L,F1,S1,R1,lcoords_sph1,pcoords) ;
    physical_to_logical_sph(L,F2,S2,R2,pcoords,lcoords_sph2) ; 
}

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
sph_to_sph_angular_transfer(double L, double * F1, double * S1, double * F2, double * S2, double* R1, double* R2, double* lcoords_sph1, double* lcoords_sph2)
{
    double pcoords[3] ; 
    logical_to_physical_sph(L,F1,S1,R1,lcoords_sph1,pcoords) ;
    physical_to_logical_sph(L,F2,S2,R2,pcoords,lcoords_sph2) ; 
}

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
sph_to_sph_negative_r_transfer_log(double L, double * F1, double * S1, double * F2, double * S2, double* R1, double* R2, double* lcoords_sph1, double* lcoords_sph2)
{
    double pcoords[3] ; 
    logical_to_physical_sph_log(L,F1,S1,R1,lcoords_sph1,pcoords) ;
    physical_to_logical_sph_log(L,F2,S2,R2,pcoords,lcoords_sph2) ; 
}

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
sph_to_sph_positive_r_transfer_log(double L, double * F1, double * S1, double * F2, double * S2, double* R1, double* R2, double* lcoords_sph1, double* lcoords_sph2)
{
    double pcoords[3] ; 
    logical_to_physical_sph_log(L,F1,S1,R1,lcoords_sph1,pcoords) ;
    physical_to_logical_sph_log(L,F2,S2,R2,pcoords,lcoords_sph2) ; 
}

static void THUNDER_ALWAYS_INLINE THUNDER_HOST_DEVICE 
sph_to_sph_angular_transfer_log(double L, double * F1, double * S1, double * F2, double * S2, double* R1, double* R2, double* lcoords_sph1, double* lcoords_sph2)
{
    double pcoords[3] ; 
    logical_to_physical_sph_log(L,F1,S1,R1,lcoords_sph1,pcoords) ;
    physical_to_logical_sph_log(L,F2,S2,R2,pcoords,lcoords_sph2) ; 
}


#endif /* THUNDER_COORDINATES_DEVICE_SPHERICAL_INLINES_HH */