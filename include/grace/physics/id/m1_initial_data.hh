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
}

struct zero_m1_id_t {
    zero_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        device_coordinate_system _coord_system
    ) : atmo(_atmo), excision(_excision), coord_system(_coord_system)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double rtp[3] ; 
        coord_system.get_spherical_coordinates(VEC(i,j,k),q,rtp) ;

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
} ; 

struct straight_beam_m1_id_t {
    straight_beam_m1_id_t(
        m1_atmo_params_t _atmo, 
        m1_excision_params_t _excision,
        device_coordinate_system _coord_system
    ) : atmo(_atmo), excision(_excision), coord_system(_coord_system)
    {}

    m1_id_t KOKKOS_INLINE_FUNCTION 
    operator() (
        VEC(int const i, int const j, int const k), 
        int const q) const 
    {
        m1_id_t id ; 
        double xyz[3] ; 
        coord_system.get_cartesian_coordinates(VEC(i,j,k),q,xyz) ;
        
        id.erad = id.fradx = id.frady = id.fradz = 0. ; 

        if ( xyz[0] <= -0.5 ) {
            id.erad = id.fradx = 1.0 ; 
        }

        return id ; 
    }

    m1_atmo_params_t atmo ; 
    m1_excision_params_t excision ; 
} ;


} /* namespace grace */
#endif /*GRACE_PHYSICS_ID_M1_HH*/