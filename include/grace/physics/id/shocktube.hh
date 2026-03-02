/**
 * @file shocktube.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-07-22
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

#ifndef GRACE_PHYSICS_ID_SHOCKTUBE_HH
#define GRACE_PHYSICS_ID_SHOCKTUBE_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

namespace grace {


template < typename eos_t >
struct shocktube_id_t {
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; 
    
    shocktube_id_t(
        eos_t eos
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords
        , double rhoL, double rhoR)
        : _eos(eos), _pcoords(pcoords), _rhoL(rhoL), _rhoR(rhoR)
    {
        // EOS views live on device so calls must happen inside a Kokkos kernel
        Kokkos::View<double[4], grace::default_space> result("shocktube_init");

        Kokkos::parallel_for("shocktube_init", 1, KOKKOS_LAMBDA(int) {
            unsigned int err;

            double rhoL_tmp = rhoL;
            double ye_guess = 0.5;
            double pL_guess = eos.press_cold__rho_ye(rhoL_tmp, ye_guess, err);
            double yeL      = eos.ye_beta_eq__press_cold(pL_guess, err);
            double pL       = eos.press_cold__rho_ye(rhoL_tmp, yeL, err);

            double rhoR_tmp = rhoR;
            ye_guess        = 0.5;
            double pR_guess = eos.press_cold__rho_ye(rhoR_tmp, ye_guess, err);
            double yeR      = eos.ye_beta_eq__press_cold(pR_guess, err);
            double pR       = eos.press_cold__rho_ye(rhoR_tmp, yeR, err);

            result(0) = pL;
            result(1) = yeL;
            result(2) = pR;
            result(3) = yeR;
        });
        Kokkos::fence();

        // Copy results back to host
        auto h_result = Kokkos::create_mirror_view(result);
        Kokkos::deep_copy(h_result, result);

        _pL  = h_result(0);
        _yeL = h_result(1);
        _pR  = h_result(2);
        _yeR = h_result(3);
    }

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id;
        if (_pcoords(VEC(i,j,k),0,q) <= 0) {
            id.rho   = _rhoL;
            id.press = _pL;
            id.ye    = _yeL;
        } else {
            id.rho   = _rhoR;
            id.press = _pR;
            id.ye    = _yeR;
        }

        id.vx = 0; id.vy = 0.; id.vz = 0.;
        id.betax = 0; id.betay = 0; id.betaz = 0;
        id.alp = 1;
        id.gxx = 1; id.gyy = 1; id.gzz = 1;
        id.gxy = 0; id.gxz = 0; id.gyz = 0;
        id.kxx = 0; id.kyy = 0; id.kzz = 0;
        id.kxy = 0; id.kxz = 0; id.kyz = 0;

        return std::move(id);
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rhoL, _rhoR, _pL, _pR, _yeL, _yeR;                    //!< Left and right states  
} ; 


}

#endif 