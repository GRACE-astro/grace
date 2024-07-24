/**
 * @file tov.hh
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

#ifndef GRACE_PHYSICS_ID_TOV_HH
#define GRACE_PHYSICS_ID_TOV_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

#include <Kokkos_Core.hpp>

namespace grace {

template < typename eos_t >
struct tov_id_t {
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ;

    tov_id_t(
          state_t state, state_t aux
        , eos_t eos
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rhoC )
        : _state(state), _aux(aux), _eos(eos)
        , _pcoords(pcoords), _rhoC(rhoC)
    {
        Kokkos::parallel_for("tov_solve", 1,
        KOKKOS_LAMBDA (int const& dummy ) {
            solve() ; 
        }) ; 
    } 

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int i, int j, int k), int q, eos_t const& eos) const 
    {

    }

    std::array<double,3> GRACE_HOST_DEVICE
    solve(double const R) 
    {
        auto const rhs = [=] (double const& r, std::array<double,3> const& state) {
            
            double const m     = state[0] ; 
            double const press = state[1] ; 
            double const phi   = state[2] ; 

            unsigned int err ;
            double ye = 0 ;
            auto const e = _eos.energy_cold__press_cold_ye(press, ye, err) ; 
            double const dPdr = -(e + press) * ( m + 4*M_PI * math::int_pow<3>(r) * P) / (r*(r-2.*m)+1e-50); 
            return std::array<double,3> {
                  4. * M_PI * math::int_pow<2>(r) * e 
                , dPdr 
                , -dPdr/(e + press)  
            } ; 
        }
        unsigned int err ; 
        double ye = 0 ; 
        auto const _pressC = _eos.press_cold__rho_ye(_rhoC, ye, err ) ; 
        rk45_t<3> solver{{0.,R}, {0., _pressC, 0.},  1e-12} ; 
        auto solution = solver.solve(rhs) ; 
    }

    state_t _state, _aux ;                            //!< State and aux arrays    
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rhoC;                                     //!< Central density 


      
} ;

} /* namespace grace */

#endif /* GRACE_PHYSICS_ID_TOV_HH */