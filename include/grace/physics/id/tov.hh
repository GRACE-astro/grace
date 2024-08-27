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

template< typename eos_t >
static std::array<double,3> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
tov_rhs(double const& r, std::array<double,3> const& state, eos_t const& _eos) {
    double m     = state[0] ; 
    double press = state[1] ; 
    double phi   = state[2] ; 

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

template < typename eos_t >
struct tov_id_t {
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ;

    tov_id_t(
          eos_t eos
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rhoC )
        : _eos(eos), _pcoords(pcoords), _rhoC(rhoC)
    { 

        Kokkos::View<double *, grace::default_space> tov_params("TOV_parameters", 6) ; 
        
        GRACE_INFO("In TOV setup.") ; 
        Kokkos::parallel_for("solve_tov", 1,  KOKKOS_LAMBDA (int dummy){
            unsigned int err ; 
            double ye, eps;     
            double temp = 0.0; 
            double rho = rhoC ;
            double rho_atm = eos.rho_atmosphere() ;
            double ye_atm  = eos.ye_atmosphere()   ; 
            double press_atm = eos.press_cold__rho_ye(rho_atm,ye_atm, err) ;
            /* Find central pressure, eps, ye */
            double const _pressC_loc = eos.press_eps_ye__beta_eq__rho_temp(eps,ye,rho,temp,err) ; 
            rk45_t<3> solver{{0.,30.}, {0., _pressC_loc, 0.},  1e-5, 1e-03} ; 
            auto cback = [&] (double const& r, std::array<double,3> const& state) -> std::array<double,3>
            { 
                return tov_rhs<eos_t>(r,state,eos) ; 
            } ; 
            bool stored = false ;
            while(true) {
                solver.advance_step( cback ) ; 
                if ( solver.state[1] < 1e-14 ) {
                    break ; 
                } else if ( solver.state[1] < 1e-12  and not stored ) {
                    tov_params(5) = solver.t ; //!< Store the transition radius
                    stored = true ; 
                }
                
            }
            tov_params(0) = solver.t ; 
            tov_params(1) = solver.state[0] ; 
            tov_params(2) = _pressC_loc ; 
            tov_params(3) = solver.state[2] ; 
            tov_params(4) = press_atm ; 
        }) ; 
        auto h_tov_params = Kokkos::create_mirror_view(tov_params) ; 
        Kokkos::deep_copy(h_tov_params, tov_params) ; 
        GRACE_INFO("TOV solver (all in code units):\n"
                   "   Central density  : {}\n"
                   "   Central pressure : {}\n"
                   "   Mass             : {}\n"   
                   "   Radius           : {}", _rhoC, h_tov_params(2), h_tov_params(1), h_tov_params(0)) ; 
        _M = h_tov_params(1) ; 
        _R = h_tov_params(0) ; 
        _pressC = h_tov_params(2) ;
        _compactness = _M/_R ; 
        _nu_corr = 0.5 * log(1-2*_compactness) - h_tov_params(3) ;  
        _press_atm = h_tov_params(4) ; 
        _transition_radius = h_tov_params(5) ; 
    } 

    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int i, int j, int k), int q) const 
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
        auto sol = solve(r) ;

        grmhd_id_t id ; 

        unsigned int err ; 
        
        /* Check if we are inside the star */
        double ye_atm  = _eos.ye_atmosphere()  ; 
        double rho_atm = _eos.rho_atmosphere() ;
         
        if ( sol[1] > 1.001 * _press_atm ) {
            id.press = sol[1] ; 
            id.ye    = _eos.ye_beta_eq__press_cold(sol[1],err) ;
            // Get rho and eps from press 
            double eps ; 
            id.rho   = _eos.rho__press_cold_ye(sol[1], id.ye, err) ; 
        } else {
            id.rho   = rho_atm   ;
            id.ye    = ye_atm    ;
            id.press = _press_atm ; 
        }
        id.vx = 0 ; id.vy = 0; id.vz = 0;

        /* Set the metric */
        id.alp   = 
            Kokkos::exp(sol[2]) ; 
        id.betax = 0. ; 
        id.betay = 0. ; 
        id.betaz = 0. ; 

        double const f1 = Kokkos::sqrt(math::int_pow<2>(x) + math::int_pow<2>(y)) ;

        std::array<std::array<double,3>,3> gij {
            std::array<double,3>{1./(1. - 2*sol[0]/r), 0, 0},
            std::array<double,3>{0,r*r,0},
            std::array<double,3>{0,0,(f1*f1)}
        } ; 

        std::array<std::array<double,3>,3> gpij {
            std::array<double,3>{0,0,0},
            std::array<double,3>{0,0,0},
            std::array<double,3>{0,0,0}
        } ;
         
        std::array<std::array<double,3>,3> J { 
            std::array<double,3>{x/r, y/r, z/r},
            std::array<double,3>{x*z/(f1*r*r), y*z/(f1*r*r), -f1/(r*r)},
            std::array<double,3>{-y/(f1*f1), x/(f1*f1), 0.}
        } ; 

        for( int ii=0; ii<3; ++ii){
            for( int jj=0; jj<3; ++jj){
                for( int ll=0; ll<3; ++ll ) {
                    for( int kk=0; kk<3; ++kk) {
                        gpij[ii][jj] += 
                            J[ll][ii] * J[kk][jj] * gij[ll][kk] ; 
                    }
                }
            }
        }

        id.gxx = gpij[0][0] ;
        id.gxy = gpij[0][1] ;
        id.gxz = gpij[0][2] ;
        id.gyy = gpij[1][1] ;
        id.gyz = gpij[1][2] ;
        id.gzz = gpij[2][2] ;

        id.kxx = 0. ;
        id.kxy = 0. ;
        id.kxz = 0. ;
        id.kyy = 0. ;
        id.kyz = 0. ;
        id.kzz = 0. ;
        return std::move(id) ; 
    }

    

    std::array<double,3> GRACE_HOST_DEVICE
    solve(double const R) const
    {
        if( R > _R ) {
            return std::array<double,3>{
                _M, 
                0,
                Kokkos::log(Kokkos::sqrt(1.-2*_M/R)) 
            } ; 
        } else { 
            rk45_t<3> solver{{0.,R}, {0., _pressC, 0.},  1e-5, 1e-03} ;
            auto cback = [&] (double const& r, std::array<double,3> const& state) -> std::array<double,3>
            { 
                return tov_rhs<eos_t>(r,state,_eos) ; 
            } ; 
            solver.solve( cback ) ; 
            solver.state[2] += _nu_corr ; 
            auto out = solver.state ; 
            if ( R > _transition_radius ) {
                double const w = _R - _transition_radius ; 
                auto const phi = 0.5 * ( 1 + Kokkos::tanh((R-_transition_radius)/(w))) ; 
                out[0] = phi * out[0] + ( 1. - phi ) * _M ; 
                out[1] = phi * out[1] ;
                out[2] = phi * out[2] + ( 1. - phi ) * Kokkos::log(Kokkos::sqrt(1.-2*_M/R));
            } 
            return out ; 
        } 
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rhoC, _pressC;                            //!< Central density 
    double _M, _R;                                    //!< Mass and Radius
    double _compactness, _nu_corr ;                   //!< Compactness and matching of metric potential
    double _transition_radius, _press_atm ; 
} ;

} /* namespace grace */

#endif /* GRACE_PHYSICS_ID_TOV_HH */