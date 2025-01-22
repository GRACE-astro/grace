/**
 * @file minkowski.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-01-21
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

#ifndef GRACE_PHYSICS_ID_SCHWARZSCHILD_KS_HH
#define GRACE_PHYSICS_ID_SCHWARZSCHILD_KS_HH

#include <grace_config.h>

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
 * @brief Initial data for trivial BSSN checks
 * \ingroup initial_data
 * @tparam eos_t Type of equation of state
 */
//**************************************************************************************************
template < typename eos_t >
struct schwarzschild_ks_id_t {
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //! Type of state array
    //**************************************************************************************************
    /**
     * @brief Construct a new Schwarzschild kernel
     * 
     * @param eos Equation of state
     * @param pcoords physical coordinates
     */
    schwarzschild_ks_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords
        , double M)
        : _eos(eos)
        , _pcoords(pcoords)
        , _M(M)
    {} 
    //**************************************************************************************************
    //**************************************************************************************************
    /**
     * @brief Return Schwarzschild initial data at a point
     * 
     * @param i x cell index
     * @param j y cell index
     * @param k z cell index
     * @param q quadrant index
     * 
     * @return grmhd_id_t Initial data objects for Schwarzschild ID test
     */
    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        double const x = _pcoords(VEC(i,j,k),0,q);
        double const y = _pcoords(VEC(i,j,k),1,q);
        double const z = _pcoords(VEC(i,j,k),2,q);
        double const r = Kokkos::sqrt(x*x + y*y + z*z);

        if(r<0.1) printf("R smaller than 0.05!");

        grmhd_id_t id ; 
        id.rho   = 0 ; 
        id.press = 0 ;
        id.ye    = 0 ;
        id.vx = id.vy = id.vz = 0;

        // Table 2.1 in 'Numerical Relativity
        // Solving Einstein’s Equations on the Computer'
        id.alp = 1./Kokkos::sqrt(1. + 2.*_M/r) ; 

        id.betax = 2.*_M * x * (id.alp*id.alp) / (r*r);
        id.betay = 2.*_M * y * (id.alp*id.alp) / (r*r);
        id.betaz = 2.*_M * z * (id.alp*id.alp) / (r*r);
      
        
        id.gxx = 1. ;
        id.gyy = 1. ;
        id.gzz = 1. ; 
       
        id.gxy = 0. ;
        id.gxz = 0. ;
        id.gyz = 0. ; 

        id.gxx+= (2.*_M ) * x * x / (r*r*r);
        id.gxy+= (2.*_M ) * x * y / (r*r*r);
        id.gxz+= (2.*_M ) * x * z / (r*r*r);
        id.gyy+= (2.*_M ) * y * y / (r*r*r);
        id.gyz+= (2.*_M ) * y * z / (r*r*r);
        id.gzz+= (2.*_M ) * z * z / (r*r*r);

        
      
        id.kxx = 1. ;
        id.kyy = 1. ;
        id.kzz = 1. ; 
       
        id.kxy = 0. ;
        id.kxz = 0. ;
        id.kyz = 0. ; 

        id.kxx += -(2. + _M/r)*x*x/(r*r) ;
        id.kyy += -(2. + _M/r)*y*y/(r*r) ;
        id.kzz += -(2. + _M/r)*z*z/(r*r) ; 
       
        id.kxy += -(2. + _M/r)*x*y/(r*r) ;
        id.kxz += -(2. + _M/r)*x*z/(r*r) ;
        id.kyz += -(2. + _M/r)*y*z/(r*r) ; 

        id.kxx *= 2.*_M*id.alp/(r*r) ;
        id.kyy *= 2.*_M*id.alp/(r*r) ;
        id.kzz *= 2.*_M*id.alp/(r*r) ; 
       
        id.kxy *= 2.*_M*id.alp/(r*r) ;
        id.kxz *= 2.*_M*id.alp/(r*r) ;
        id.kyz *= 2.*_M*id.alp/(r*r) ; 

        return std::move(id) ; 
    }
    //**************************************************************************************************

    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _M; // mass
    //**************************************************************************************************  
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
//**************************************************************************************************
#endif /* GRACE_PHYSICS_ID_SCHWARZSCHILD_KS_HH */