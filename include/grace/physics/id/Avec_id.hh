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

struct Avec_toroidal_id_t {

    template<size_t idir>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double get(
        std::array<double,3> const& coords,
        double const& var
    ) const  
    {
        double const _A = _A_phi * Kokkos::pow(Kokkos::max(var-_cut,0.0),_A_n) ; 
        std::array<double,3> A {
            -coords[1] * _A,
            coords[0] * _A,
            0
        } ; 
        return A[idir] ;
    }

    double _cut, _A_phi, _A_n ; 
} ; 

struct Avec_toroidal_rho_rel_id_t {

    template<size_t idir>
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    double get(
        std::array<double,3> const& coords,
        double const& press, double const& rho
    ) const  
    {
        double const _A = _A_phi * Kokkos::max(rho/_rhomax-_rhocut,0.0) ; 
        std::array<double,3> A {
            -coords[1] * _A,
            coords[0] * _A,
            0
        } ; 
        return A[idir] ;
    }

    double _rhomax, _rhocut, _A_phi, _A_n ; 
    
} ; 

}

#endif /* GRACE_PHYS_ID_AVEC_HH */