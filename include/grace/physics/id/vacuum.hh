/**
 * @file vacuum.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-24
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

#ifndef GRACE_PHYSICS_ID_VACUUM_HH
#define GRACE_PHYSICS_ID_VACUUM_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

#include "FMTorus/KerrSchild.hh"

namespace grace {


template < typename eos_t >
struct vacuum_id_t {
    using state_t = grace::var_array_t ; 
    
    vacuum_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rho, double press, double vx, double vy, double vz 
        , bool is_cks )
        : _eos(eos)
        , _pcoords(pcoords)
        , _rho(rho), _press(press), _vx(vx), _vy(vy), _vz(vz)
        , _is_cks(is_cks)
    {} 

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 

        id.rho = _rho ; 
        id.press = _press ;
        id.vx = _vx ; 
        id.vy = _vy ; 
        id.vz = _vz ; 
        id.bx = id.by = id.bz = 0 ; 

        id.vx = _vx; id.vy = _vy; id.vz = _vz;
        if ( ! _is_cks ) {
            id.betax = 0; id.betay=0; id.betaz = 0; 
            id.alp = 1 ; 
            id.gxx = 1; id.gyy = 1; id.gzz = 1;
            id.gxy = 0; id.gxz = 0; id.gyz = 0 ;
            id.kxx = 0; id.kyy = 0; id.kzz = 0 ;
            id.kxy = 0; id.kxz =0 ; id.kyz = 0 ; 
        } else {
            double const x = this->_pcoords(i,j,k,0,q) ; 
            double const y = this->_pcoords(i,j,k,1,q) ; 
            double const z = this->_pcoords(i,j,k,2,q) ; 
            double psi4; 
            ComputeADMDecomposition(
                x,y,z, false, 0,
                &id.alp, &id.betax, &id.betay, &id.betaz,
                &psi4, &id.gxx, &id.gxy, &id.gxz, &id.gyy, &id.gyz, &id.gzz,
                &id.kxx, &id.kxy, &id.kxz, &id.kyy, &id.kyz, &id.kzz
            ) ; 
        }
        
        eos_err_t err ; 
        id.ye  = _eos.ye_cold__press(id.press, err);
        double h, csnd2;
        id.eps = _eos.eps_h_csnd2_temp_entropy__press_rho_ye(
            h, csnd2, id.temp, id.entropy, id.press, id.rho, id.ye, err
        ) ; 
        return std::move(id) ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rho, _press, _vx, _vy, _vz;               //!< Background state
    bool _is_cks;
} ; 


}

#endif 