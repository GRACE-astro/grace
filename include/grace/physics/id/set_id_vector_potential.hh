/**
 * @file set_id_vector_potential.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-15
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


#ifndef GRACE_PHYSICS_SET_ID_VECPOT
#define GRACE_PHYSICS_SET_ID_VECPOT

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/utils/grace_utils.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/utils/runge_kutta.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/errors/error.hh>

#include <algorithm> // std::max
#include <cmath>     // std::pow

namespace grace {

    enum class BVEC_TYPE {
        NONE = 0,
        POLOIDAL,
        TOROIDAL
    }; // add more if needed

    enum class AVEC_PRESCRIPTION {
        NONE = 0,
        PRESSURE,
        DENSITY
    }; // add more

   enum AVEC_PARAMS{
        AVEC_PCUT = 0,
        AVEC_RHOCUT,
        AVEC_N,
        AVEC_AB,
        AVEC_AB_N, // negative vals 
        AVEC_AB_P, // positive vals
        AVEC_NUMPARAMS
   };

    // Primary template: default zero-vector behaviour.
    template<BVEC_TYPE BV = BVEC_TYPE::NONE, AVEC_PRESCRIPTION AP =  AVEC_PRESCRIPTION::NONE>
    struct GRACE_HOST_DEVICE avec_t {
            
        GRACE_HOST_DEVICE avec_t(const std::array<double,AVEC_PARAMS::AVEC_NUMPARAMS>& avec_params, const grace::coord_array_t<GRACE_NSPACEDIM>& pcoords) 
            :  _avec_params(avec_params), _pcoords(pcoords)
                { }

        GRACE_HOST_DEVICE
        void operator()(double &AUx, double &AUy, double &AUz,
                    const grace::var_array_t<GRACE_NSPACEDIM>& state,
                    const grace::var_array_t<GRACE_NSPACEDIM>& aux, 
                    const grace::scalar_array_t<GRACE_NSPACEDIM>& idx,
                    VEC(int i, int j, int k), int q) const noexcept
        { // empty body 
             AUx = AUy = AUz = 0.0;
        }

        std::array<double,AVEC_PARAMS::AVEC_NUMPARAMS> _avec_params;
        grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    };

    // Primary template: default zero-vector behaviour.
    // template<BVEC_TYPE BV = BVEC_TYPE::NONE, AVEC_PRESCRIPTION AP = AVEC_PRESCRIPTION::NONE>
    // struct GRACE_HOST_DEVICE avec_t {
    // GRACE_HOST_DEVICE
    // void operator()(double &AUx, double &AUy, double &AUz,
    //                 const grace::var_array_t<GRACE_NSPACEDIM>& state,
    //                 const grace::var_array_t<GRACE_NSPACEDIM>& aux, 
    //                 const grace::scalar_array_t<GRACE_NSPACEDIM>& idx,
    //                 VEC(int i, int j, int k), int q) const noexcept
    // {
    //     AUx = AUy = AUz = 0.0;
    // }};

    // Specialization for (POLOIDAL, PRESSURE)
    template<>
    struct GRACE_HOST_DEVICE avec_t<BVEC_TYPE::POLOIDAL, AVEC_PRESCRIPTION::PRESSURE>
    : public avec_t<>  {
        using base_t = avec_t<>;
        using base_t::base_t; // inherit constructors from base_t
        // device/host-callable functor that fills vector potential given coordinates + pressure
        GRACE_HOST_DEVICE
        void operator()(double &AUx, double &AUy, double &AUz,
                    const grace::var_array_t<GRACE_NSPACEDIM>& state,
                    const grace::var_array_t<GRACE_NSPACEDIM>& aux, 
                    const grace::scalar_array_t<GRACE_NSPACEDIM>& idx,
                    VEC(int i, int j, int k), int q) const noexcept
        {
            const double _Avec_Pcut = _avec_params[AVEC_PARAMS::AVEC_PCUT];
            const double _Avec_n = _avec_params[AVEC_PARAMS::AVEC_N];
            const double _Avec_Ab = _avec_params[AVEC_PARAMS::AVEC_AB];

            const double pterm = std::max(aux(VEC(i,j,k), PRESS_, q) - _Avec_Pcut, 0.0);
            const double power = std::pow(pterm, static_cast<double>(_Avec_n));
            double const x = _pcoords(VEC(i,j,k),0,q);
            double const y = _pcoords(VEC(i,j,k),1,q);
            double const z = _pcoords(VEC(i,j,k),2,q);
            AUx = -y * _Avec_Ab * power;
            AUy =  x * _Avec_Ab * power;
            AUz = 0.0;
            // optionally add radial fall-off: divide by (x*x + y*y) if desired, but beware division by zero
        }

    }; 

    using B_field_zero_t    = avec_t<>;
    using B_field_pol_pres_t = avec_t<BVEC_TYPE::POLOIDAL, AVEC_PRESCRIPTION::PRESSURE>;
    // using B_field_tor_den_t = avec_t<BVEC_TYPE::TOROIDAL, AVEC_PRESCRIPTION::DENSITY>;
    // add more as needed
} // namespace grace

#endif /* GRACE_PHYSICS_SET_ID_VECPOT */