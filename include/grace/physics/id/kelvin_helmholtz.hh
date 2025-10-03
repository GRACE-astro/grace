/**
 * @file kelvin_helmholtz.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-10-01
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
#ifndef GRACE_PHYSICS_ID_KHI_HH
#define GRACE_PHYSICS_ID_KHI_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>
#include <grace/utils/math.hh>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

namespace grace {

template < typename eos_t >
struct kelvin_helmholtz_id_t {
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; 
    
    kelvin_helmholtz_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords )
        : _eos(eos)
        , _pcoords(pcoords)
    {} 

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {
        grmhd_id_t id ; 

        double const x = _pcoords(VEC(i,j,k),0,q) ;
        double const y = _pcoords(VEC(i,j,k),1,q) ;
        double const z = _pcoords(VEC(i,j,k),2,q) ;

        double const rho0{1.5}, rho1{0.5}, al{0.01}, yl{0.25}, vx0{0.5} ; 
        double const sigma{0.2} ; 
        double const dvx0{0.01}, dvy0{0.01}, dvz0{0.01} ; 

        auto _tanh = Kokkos::tanh(
            (Kokkos::fabs(y - yl) ) / al 
        ) ; 
        auto _exp = [&] () {
            return Kokkos::exp(
                -math::int_pow<2>(y - yl)/sigma/sigma
            ) ;
        } ;  

        //double const dvx = dvx0 * Kokkos::sin(2 * M_PI * x * nx) ; 
        double const dvy = dvy0 * Kokkos::sin(4 * M_PI * x)*Kokkos::sin(4 * M_PI * z) ; 
        double const dvz = dvz0 * Kokkos::sin(4 * M_PI * x)*Kokkos::sin(4 * M_PI * z) ; 

        id.rho = rho0 - rho1 * _tanh ; 

        id.vx = vx0 * _tanh  ; 
        id.vy = dvy * _exp() ; 
        id.vz = dvz * _exp() ; 

        id.press = 2.5 ; 

        id.betax = 0; id.betay=0; id.betaz = 0; 
        id.alp = 1 ; 
        id.gxx = 1; id.gyy = 1; id.gzz = 1;
        id.gxy = 0; id.gxz = 0; id.gyz = 0 ;
        id.kxx = 0; id.kyy = 0; id.kzz = 0 ;
        id.kxy = 0; id.kxz =0 ; id.kyz = 0 ; 
        unsigned int err ; 
        id.ye  = _eos.ye_beta_eq__press_cold(id.press, err);
        return std::move(id) ; 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
} ; 

}
#endif /* GRACE_PHYSICS_ID_KHI_HH */