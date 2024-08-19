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

#include <grace/utils/runge_kutta.hh>
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

        Kokkos::View<double [3], grace::default_space> tov_params ; 
        

        Kokkos::parallel_for("solve_tov", 1,  [=,this] GRACE_HOST_DEVICE (int dummy){
            unsigned int err ; 
            double ye, eps;     
            double temp = 0.0; 
            /* Find central pressure, eps, ye */
            double const _pressC_loc = eos.press_eps_ye__beta_eq__rho_temp_impl(eps,ye,_rhoC,temp,err) ; 
            rk45_t<3> solver{{0.,30.}, {0., _pressC_loc, 0.},  1e-5, 1e-03} ; 
            while(true) {
                solver.advance_step((*this).rhs) ; 
                if ( solver.state[1] < 1e-16 ) {
                    break ;  
                }
            }
            tov_params(0) = solver.t ; 
            tov_params(1) = solver.state[0] ; 
            tov_params(2) = _pressC_loc ; 
        }) ; 
        auto h_tov_params = Kokkos::create_mirror_view(tov_params) ; 
        Kokkos::deep_copy(h_tov_params, tov_params) ; 
        GRACE_INFO("TOV solver (all in code units):\n"
                   "   Central density: {}\n"
                   "   Central pressure: {}\n"
                   "   Mass:   {}\n"   
                   "   Radius: {}\n", _rhoC, h_tov_params(2), h_tov_params(1), h_tov_params(1)) ; 
        _M = h_tov_params(1) ; 
        _R = h_tov_params(0) ; 
        _pressC = h_tov_params(2) ; 
    } 

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int i, int j, int k), int q, eos_t const& eos) const 
    {
        double const x = _pcoords(VEC(i,j,k),0,q);
        double const y = _pcoords(VEC(i,j,k),1,q);
        #ifdef GRACE_3D 
        double const z = _pcoords(VEC(i,j,k),2,q);
        #else 
        double const z = 0. ; 
        #endif 
        double const r = Kokkos::sqrt(EXPR(
              math::int_pow<2>(x),
            + math::int_pow<2>(y),
            + math::int_pow<2>(z)
        )) ; 

        // This returns: ADM mass, pressure and metric potential
        // at this radius.
        auto const sol = solve(r) ;

        auto sview = Kokkos::subview(_aux, VEC(i,j,k), Kokkos::ALL(), q) ;
        auto state_sview = Kokkos::subview(_state, VEC(i,j,k), Kokkos::ALL(), q) ;

        unsigned int err ; 
        
        /* Check if we are inside the star */
        double ye_atm  = _eos.ye_atmosphere()  ; 
        double rho_atm = _eos.rho_atmosphere() ;
        double press_atm = _eos.press_cold__rho_ye(rho_atm,ye_atm, err) ; 

        if ( sol[1] > 1.001 * press_atm ) {
            sview(PRESS_) = sol[1] ; 
            sview(YE_)  = _eos.ye_beta_eq__press_cold(sol[1],err) ;
            // Get rho and eps from press 
            sview(RHO_) = _eos.rho_eps_cold__press_cold_ye(sview(EPS_), sol[1], sview(YE_), err) ; 
        } else {
            sview(RHO_)   = rho_atm   ;
            sview(YE_)    = ye_atm    ;
            sview(PRESS_) = press_atm ; 
        }
        sview(VELX_) = 0. ; sview(VELY_) = 0.; sview(VELZ_) = 0. ; 
        sview(ZVECX_) = 0. ; sview(ZVECY_) = 0.; sview(ZVECZ_) = 0. ; 

        /* Set the metric */
        state_sview(ALP_)   = 
            -Kokkos::sqrt(Kokkos::exp(sol[2])) ; 
        state_sview(BETAX_) = 0. ; 
        state_sview(BETAY_) = 0. ; 
        state_sview(BETAZ_) = 0. ; 

        double const f1 = Kokkos::sqrt(math::int_pow<2>(x) + math::int_pow<2>(y)) ;

        std::array<std::array<double,3>,3> gij {
            std::array<double,3>{1./(1. - 2*sol[0]/r), 0, 0},
            std::array<double,3>{0,-r*r,0},
            std::array<double,3>{0,0,-(f1*f1)}
        } ; 

        std::array<std::array<double,3>,3> gpij {
            std::array<double,3>{0,0,0},
            std::array<double,3>{0,0,0},
            std::array<double,3>{0,0,0}
        } ;
         
        std::array<std::array<double,3>,3> J { 
            std::array<double,3>{x/r, y/r, z/r},
            std::array<double,3>{x*z/(f1*r*r), y*z/(f1*r*r), -f1/(r*r)},
            std::array<double,3>{y/f1, -x/f1, 0.}
        } ; 

        for( int ii=0; ii<3; ++ii){
            for( int jj=0; jj<3; ++jj){
                for( int ll=0; ll<3; ++ll ) {
                    for( int kk=0; kk<3; ++kk) {
                        gpij[ii][jj] += 
                            J[ii][ll] * J[jj][kk] * gij[ll][kk] ; 
                    }
                }
            }
        }

        state_sview(GXX_) = gpij[0][0] ;
        state_sview(GXY_) = gpij[0][1] ;
        state_sview(GXZ_) = gpij[0][2] ;
        state_sview(GYY_) = gpij[1][1] ;
        state_sview(GYZ_) = gpij[1][2] ;
        state_sview(GZZ_) = gpij[2][2] ;

        state_sview(KXX_) = 0. ;
        state_sview(KXY_) = 0. ;
        state_sview(KXZ_) = 0. ;
        state_sview(KYY_) = 0. ;
        state_sview(KYZ_) = 0. ;
        state_sview(KZZ_) = 0. ;

    }

    std::array<double,3> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
    rhs(double const& r, std::array<double,3> const& state) {
        double const m     = state[0] ; 
        double const press = state[1] ; 
        double const phi   = state[2] ; 

        unsigned int err ;
        double ye = 0 ;
        auto const e = _eos.energy_cold__press_cold_ye(press, ye, err) ; 
        double const dPdr = -(e + press) * ( m + 4*M_PI * math::int_pow<3>(r) * press) / (r*(r-2.*m)+1e-50); 
        return std::array<double,3> {
            4. * M_PI * math::int_pow<2>(r) * e 
            , dPdr 
            , -dPdr/(e + press + 1e-50)  
        } ; 
    }

    std::array<double,3> GRACE_HOST_DEVICE
    solve(double const R) 
    {
        if( R > _R ) {
            return std::array<double,3>{
                _M, 
                0,
                Kokkos::log(1.-2*_M/R)
            } ; 
        } else {
            rk45_t<3> solver{{0.,R}, {0., _pressC, 0.},  1e-3, 1e-03} ; 
            return solver.solve((*this).rhs) ;
        }
    }

    state_t _state, _aux ;                            //!< State and aux arrays    
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rhoC, _pressC;                            //!< Central density 
    double _M, _R;                                    //!< Mass and Radius
} ;

} /* namespace grace */

#endif /* GRACE_PHYSICS_ID_TOV_HH */