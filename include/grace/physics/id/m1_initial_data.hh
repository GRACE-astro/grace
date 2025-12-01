/**
 * @file m1_initial_data.hh
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

#ifndef GRACE_PHYSICS_ID_M1_HH
#define GRACE_PHYSICS_ID_M1_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/coordinates/coordinate_systems.hh>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

namespace grace {

struct m1_id_t {
    double erad, fradx, frady, fradz ; //! lower indices
} ; 

struct zero_m1_id_t {
    zero_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        /* we assume coords are spherical here! */
        double rtp[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 

        auto E_atmo = atmo.E_fl * Kokkos::pow(rtp[0], atmo.E_fl_scaling) ; 

        bool excise = excision.excise_by_radius ? rtp[0] <= excision.r_ex : false ; /*we don't have alp here*/
        
        if ( excise ) {
            id.erad = excision.E_ex ; 
        } else {
            id.erad = E_atmo ;
        }
        id.fradx = id.frady = id.fradz = 0. ; 
        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
} ; 

struct straight_beam_m1_id_t {
    straight_beam_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 
        
        id.erad = atmo.E_fl ;
        id.fradx = id.frady = id.fradz = 0. ; 

        if ( xyz[0] <= -0.25 and 
            xyz[1] < 0.0625 and xyz[1] > - 0.0625 and 
            xyz[2] < 0.0625 and xyz[2] > - 0.0625) {
            id.erad = id.fradx = 1.0 ; 
        }

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
} ;

struct scattering_diffusion_m1_id_t {
    scattering_diffusion_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords,
        double _ks, double _t0
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords), ks(_ks), t0(_t0)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 
        double r2 = SQR(xyz[0])+SQR(xyz[1])+SQR(xyz[2]);
        double r = sqrt(r2) ; 
        id.erad = Kokkos::pow(ks/t0,3./2.) * Kokkos::exp(-3*ks*r2/(4.*t0)) ; 

        double Hr = r/(2.*t0) * id.erad ; 

        id.fradx = xyz[0]/r * Hr ; 
        id.frady = xyz[1]/r * Hr ; 
        id.fradz = xyz[2]/r * Hr ; 

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    double ks, t0;
} ;

struct moving_scattering_diffusion_m1_id_t {
    moving_scattering_diffusion_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords,
        double _v0
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords), v0(_v0)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 

        id.erad = Kokkos::exp(-9.0*SQR(xyz[0])) ; 

        double const W2 = 1./(1-SQR(v0)) ; 
        double J = 3.*id.erad  / (4.*W2-1.); 

        id.fradx = 4./3. * J * W2 * v0 ; 
        id.frady = id.fradz = 0. ;  

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    double v0;
} ;


struct emitting_sphere_m1_id_t {
    emitting_sphere_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        coord_array_t<GRACE_NSPACEDIM> _pcoords
    ) : atmo(_atmo), excision(_excision), pcoords(_pcoords)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] = {
            pcoords(VEC(i,j,k),0,q),
            pcoords(VEC(i,j,k),1,q),
            pcoords(VEC(i,j,k),2,q)
        }; 

        double r2 = SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2]) ; 
        double r = sqrt(r2) ; 

        if ( r < 1. ) {
            id.erad = 1. ; 
            id.fradx=id.frady=id.fradz = 0 ; 
        } else {
            id.erad = 1/r2 ;
            id.fradx = 0.5/r2 * xyz[0]/r ; 
            id.frady = 0.5/r2 * xyz[1]/r ; 
            id.fradz = 0.5/r2 * xyz[2]/r ; 
        }

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
} ;

} /* namespace grace */
#endif /*GRACE_PHYSICS_ID_M1_HH*/