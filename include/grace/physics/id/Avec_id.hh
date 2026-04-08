/**
 * @file fmtorus.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-10-13
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
#ifndef GRACE_PHYS_ID_AVEC_HH
#define GRACE_PHYS_ID_AVEC_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

namespace grace {

struct Avec_poloidal_id_t {

    Avec_poloidal_id_t(
        double cut
    ) : _cut(cut), c1({0,0,0}), c2({0,0,0})
    {
        _A_phi_1 = get_param<double>("grmhd", "Avec_ID","A_phi") ;
        _A_phi_2 = get_param<double>("grmhd", "Avec_ID","A_phi_2") ;
        if (_A_phi_2<0) _A_phi_2 = _A_phi_1 ; 

        _A_n = get_param<double>("grmhd", "Avec_ID","A_n") ;

        _is_binary = get_param<bool>("grmhd", "Avec_ID", "is_binary") ; 
        
        c1[0] = get_param<double>("grmhd", "Avec_ID","x_c_1") ;
        c1[1] = get_param<double>("grmhd", "Avec_ID","y_c_1") ;
        c1[2] = get_param<double>("grmhd", "Avec_ID","z_c_1") ;

        c2[0] = get_param<double>("grmhd", "Avec_ID","x_c_2") ;
        c2[1] = get_param<double>("grmhd", "Avec_ID","y_c_2") ;
        c2[2] = get_param<double>("grmhd", "Avec_ID","z_c_2") ;
    }

    template<size_t idir>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double get(
        std::array<double,3> const& coords,
        double const& var
    ) const  
    {
        if ( _is_binary ) {
            return get_binary<idir>(coords,var); 
        } else {
            return get_single<idir>(coords,var); 
        }
    }

    template<size_t idir>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double get_single(
        std::array<double,3> const& coords,
        double const& var
    ) const  
    {
        double const _A = _A_phi_1 * Kokkos::pow(Kokkos::max(var-_cut,0.0),_A_n) ; 
        std::array<double,3> A {
            -coords[1] * _A,
            coords[0] * _A,
            0
        } ; 
        return A[idir] ;
    }

    template<size_t idir>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double get_binary(
        std::array<double,3> const& coords,
        double const& var
    ) const  
    {
        double x1[3] = {coords[0]-c1[0], coords[1]-c1[1], coords[2]-c1[2]};
        double x2[3] = {coords[0]-c2[0], coords[1]-c2[1], coords[2]-c2[2]};

        double d1 = SQR(x1[0]) + SQR(x1[1]) + SQR(x1[2]) ;
        double d2 = SQR(x2[0]) + SQR(x2[1]) + SQR(x2[2]) ;
        

        std::array<double,3> A{0,0,0} ;
        if (d1<d2) {   
            double const _A = _A_phi_1 * Kokkos::pow(Kokkos::max(var-_cut,0.0),_A_n) ; 
            A[0] = -x1[1] * _A ; 
            A[1] = x1[0] * _A ; 
        } else {
            double const _A = _A_phi_2 * Kokkos::pow(Kokkos::max(var-_cut,0.0),_A_n) ; 
            A[0] = -x2[1] * _A ; 
            A[1] = x2[0] * _A ; 
        } 
        return A[idir] ;
    }

    double _cut, _A_phi_1,_A_phi_2, _A_n ;
    bool _is_binary ; 
    std::array<double,3> c1,c2 ;  
} ; 

}

#endif /* GRACE_PHYS_ID_AVEC_HH */