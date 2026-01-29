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

#define R_MAX 30 
#define N_POINTS 100000

namespace grace {
namespace detail {
/**
 * @brief Right hand side of TOV equations
 * 
 * @tparam eos_t Eos type.
 * @param r Radius.
 * @param state Array containing (m,press,nu).
 * @param _eos Eos.
 * @return std::array<double,3> Array containing rhs: (dm/dr,dP/dr,dnu/dr).
 */
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
/**
 * @brief Solve the TOV equations.
 * 
 * \ingroup
 * \cond grace_detail 
 * @tparam eos_t Equation of state type
 * @param eos Equation of state object
 * @param rho_c Central density 
 * @param pcoords Coordinates of gridpoints
 * @param mass m(r) solution
 * @param press P(r) solution
 * @param nu nu(r) solution
 * @param r Solution axis
 * @param dr Solution axis resolution
 * @param press_c Central pressure
 * @param M Mass of the TOV star
 * @param R Radius of the TOV star
 * @param press_atm Atmosphere pressure
 * @param nu_corr Correction on nu (to make it Schwarzschild outside the star)
 * @param transition_radius Radius where we start transitioning to Schwarzschild solution
 * @return size_t Number of points in solution arrays
 * 
 * This function calls a trivial kernel that solves the TOV equations using a Runge-Kutta 45 ODE solver,
 * returning bulk properties of the solution and the solution on the grid.
 */
template< typename eos_t >
static size_t solve_tov_equations(
    eos_t eos, 
    double const & rho_c,
    Kokkos::View<double *, grace::default_space> mass,
    Kokkos::View<double *, grace::default_space> press,
    Kokkos::View<double *, grace::default_space> nu,
    Kokkos::View<double *, grace::default_space> r,
    Kokkos::View<double *, grace::default_space> dr,
    double & press_c, 
    double & M,
    double & R,
    double & press_atm,
    double & nu_corr,
    double & transition_radius
)
{
    Kokkos::View<double *, grace::default_space> tov_params("TOV_parameters", 7) ; 
    Kokkos::parallel_for("solve_tov", 1,  KOKKOS_LAMBDA (int dummy){
        unsigned int err ; 
        double ye, eps;     
        double temp = 0.0; 
        double rho = rho_c ;
        double rho_atm = eos.rho_atmosphere() ;
        double ye_atm  = eos.ye_atmosphere()   ; 
        double press_atm = eos.press_cold__rho_ye(rho_atm,ye_atm, err) ;
        /* Find central pressure, eps, ye */
        double const _pressC_loc = eos.press_eps_ye__beta_eq__rho_temp(eps,ye,rho,temp,err) ; 
        rk45_t<3> solver{{0.,R_MAX}, {0., _pressC_loc, 0.}, 1e-04, 1e-02 } ; 
        auto cback = [&] (double const& r, std::array<double,3> const& state) -> std::array<double,3>
        { 
            return tov_rhs<eos_t>(r,state,eos) ; 
        } ; 
        bool stored = false ;
        mass(0) = 0. ;
        press(0) = _pressC_loc ; 
        nu(0)  = 0. ; 
        size_t ii = 0 ;
        while(true) {
            solver.advance_step( cback ) ; 
            ii++ ; 
            mass(ii) = solver.state[0]  ; 
            press(ii) = solver.state[1] ;
            nu(ii) = solver.state[2]    ;
            r(ii) = solver.t ; 
            dr(ii) = solver.dt ; 
            if ( solver.state[1] < 1e-12 ) {
                break ; 
            } else if ( solver.state[1] < 1e-10  and not stored ) {
                tov_params(5) = solver.t ; //!< Store the transition radius
                stored = true ; 
            }
            
        }

        tov_params(0) = solver.t ; 
        tov_params(1) = solver.state[0] ; 
        tov_params(2) = _pressC_loc ; 
        tov_params(3) = solver.state[2] ; 
        tov_params(4) = press_atm ; 
        tov_params(6) = ii+1  ; 
    }) ; 
    auto h_tov_params = Kokkos::create_mirror_view(tov_params) ; 
    Kokkos::deep_copy(h_tov_params, tov_params) ;

    M = h_tov_params(1) ; 
    R = h_tov_params(0) ;
    press_c = h_tov_params(2) ;
    nu_corr = 0.5 * log(1-2*M/R) - h_tov_params(3) ;  
    press_atm = h_tov_params(4) ; 
    transition_radius = h_tov_params(5) ; 
    return static_cast<size_t>(h_tov_params(6)) ; 
}
}
/**
 * @brief TOV initial data kernel.
 * 
 * @tparam eos_t Eos type.
 */
template < typename eos_t >
struct tov_id_t {
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ;

    tov_id_t(
          eos_t eos
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rhoC )
        : _eos(eos), _pcoords(pcoords), _rhoC(rhoC)
    { 

        Kokkos::View<double *, grace::default_space> massl("TOV_mass", N_POINTS) ; 
        Kokkos::View<double *, grace::default_space> pressl ("TOV_press", N_POINTS) ; 
        Kokkos::View<double *, grace::default_space> nul ("TOV_nu", N_POINTS) ; 
        Kokkos::View<double *, grace::default_space> rl ("TOV_nu", N_POINTS) ; 
        Kokkos::View<double *, grace::default_space> drl ("TOV_nu", N_POINTS) ; 

        /* Solve ODEs */
        GRACE_INFO("In TOV setup.") ; 
        _npoints = detail::solve_tov_equations(
              eos
            , rhoC
            , massl
            , pressl
            , nul
            , rl
            , drl 
            , _pressC
            , _M
            , _R
            , _press_atm
            , _nu_corr
            , _transition_radius
        ) ; 
        _compactness = _M/_R ; 
        GRACE_INFO("TOV solver (all in code units):\n"
                   "   Central density  : {}\n"
                   "   Central pressure : {}\n"
                   "   Mass             : {}\n"   
                   "   Radius           : {}", _rhoC, _pressC, _M, _R) ; 
        Kokkos::resize(massl, static_cast<size_t>(_npoints)) ; 
        Kokkos::resize(pressl, static_cast<size_t>(_npoints)) ; 
        Kokkos::resize(nul, static_cast<size_t>(_npoints)) ; 
        mass = massl; 
        press = pressl; 
        nu = nul ; 
        r = rl   ;
        dr = drl   ;
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
        auto sol = get_solution(r) ;

        grmhd_id_t id ; 

        unsigned int err ; 
        
        /* Check if we are inside the star */
        double ye_atm  = _eos.ye_atmosphere()  ; 
        double rho_atm = _eos.rho_atmosphere() ;
         
        if ( sol[1] > 1.001 * _press_atm ) {
            id.press = sol[1] ; 
            id.ye    = _eos.ye_beta_eq__press_cold(sol[1],err) ;
            // Get rho from press 
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
    get_solution(double const R) const
    {
        if( R > _R ) {
            return std::array<double,3>{
                _M, 
                0,
                Kokkos::log(Kokkos::sqrt(1.-2*_M/R)) 
            } ; 
        } else { 
            auto out = interp_solution(R) ; 
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

    std::array<double,3> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    interp_solution(double const R) const {
        size_t idx = find_index(R); 
        double lambda = (R - r(idx)) / dr(idx);
        return {
            mass(idx) * ( 1- lambda ) + mass(idx+1) *  (lambda),
            press(idx) * ( 1-lambda ) + press(idx+1) *   (lambda),
            nu(idx) * ( 1-lambda ) + nu(idx+1) *   (lambda) + _nu_corr 
        } ; 
    }

    size_t GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
    find_index(double const R) const { 
        int lower = 0;
        int upper = _npoints - 1;
        // simple bisection should do it
        while (upper - lower > 1) {
            int tmp = lower + (upper - lower) / 2;
            if (R < r(tmp))
                upper = tmp;
            else
                lower = tmp;
        }
        return lower;
    }

    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rhoC, _pressC;                            //!< Central density 
    double _M, _R;                                    //!< Mass and Radius
    double _compactness, _nu_corr ;                   //!< Compactness and matching of metric potential
    double _transition_radius, _press_atm ; 
    size_t _npoints ; 
    static constexpr double _dr = R_MAX / N_POINTS ; 
    Kokkos::View<double *, grace::default_space> mass, press, nu, r, dr ;  
} ;

} /* namespace grace */

#endif /* GRACE_PHYSICS_ID_TOV_HH */