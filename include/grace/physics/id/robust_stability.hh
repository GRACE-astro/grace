/**
 * @file robust_stability.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-02-04
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

#ifndef GRACE_PHYSICS_ID_ROBUST_STABILITY_HH
#define GRACE_PHYSICS_ID_ROBUST_STABILITY_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <Kokkos_Random.hpp>

namespace grace {

template < typename eos_t, typename rand_gen_t >
struct robust_stability_test_id_t {
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //! Type of state array
    //**************************************************************************************************
    /**
     * @brief Construct a robust_stability_test id kernel
     * 
     * @param eos Equation of state
     * @param pcoords physical coordinates
     * @param gen random number generator
     * @param rho perturbation scale
     */
    robust_stability_test_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , rand_gen_t gen, double rho)
        : _eos(eos)
        , _pcoords(pcoords)
        , _gen(gen)
        , _rho(rho)
    {} 
    //**************************************************************************************************
    //**************************************************************************************************
    /**
     * @brief Return robust stability test initial data at a point
     * 
     * @param i x cell index
     * @param j y cell index
     * @param k z cell index
     * @param q quadrant index
     * 
     * @return grmhd_id_t Initial data objects for robust stability test
     */
    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        auto generator = _gen.get_state() ; 

        auto eps = [&] () { return generator.drand(-1e-10/math::int_pow<2>(_rho),1e-10/math::int_pow<2>(_rho)) ; } ; 
        double const x = _pcoords(VEC(i,j,k),0,q);
        grmhd_id_t id ; 
        id.rho   = 0 ; 
        id.press = 0 ;
        id.ye    = 0 ;
        id.vx = id.vy = id.vz = 0;

        id.betax = eps() ;
        id.betay = eps() ; 
        id.betaz = eps() ; 
        

        id.gxx = 1. + eps() ;
        id.gyy = 1. + eps() ;
        id.gzz = 1. + eps() ; 
        id.gxy = eps() ; 
        id.gxz = eps() ; 
        id.gyz = eps() ; 

        id.alp = 1. + eps() ; 

        id.kxx = eps() ; 
        id.kxy = eps() ; 
        id.kxz = eps() ; 
        id.kyz = eps() ;

        id.kyy = eps() ;  
        id.kzz = eps() ;

        _gen.free_state(generator) ; 

        return std::move(id) ; 
    }
    //**************************************************************************************************

    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    rand_gen_t _gen ;                                 //!< Random number generator
    double _rho ;                                     //!< Perturbation scale
    //**************************************************************************************************  
} ; 

}
#endif /* GRACE_PHYSICS_ID_ROBUST_STABILITY_HH */