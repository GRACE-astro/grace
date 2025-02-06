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
#include <grace/utils/device/device.h>

#include <grace/utils/numerics/runge_kutta.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

#include <Kokkos_Core.hpp>

//**************************************************************************************************
#define R_MAX 30 
#define N_POINTS 100000
//**************************************************************************************************
namespace grace {
//**************************************************************************************************
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
//**************************************************************************************************
/**
 * @brief TOV initial data kernel.
 * \ingroup initial_data
 * @tparam eos_t Eos type.
 */
template < typename eos_t >
struct tov_id_t {
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //!< State array type
    //**************************************************************************************************
    /**
     * @brief Construct a new tov id kernel
     * 
     * @param eos Equation of state
     * @param pcoords Physical coordinates array 
     * @param rhoC Central density [code units]
     */
    tov_id_t(
          eos_t eos
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , double rhoC )
        : _eos(eos), _pcoords(pcoords), _rhoC(rhoC)
    { 

        Kokkos::View<double *, grace::default_space> tov_params("TOV_parameters", 7) ; 

        Kokkos::View<double *, grace::default_space> massl("TOV_mass", N_POINTS) ; 
        Kokkos::View<double *, grace::default_space> pressl ("TOV_press", N_POINTS) ; 
        Kokkos::View<double *, grace::default_space> nul ("TOV_nu", N_POINTS) ; 
        Kokkos::View<double *, grace::default_space> rl ("TOV_radii", N_POINTS) ; 
        Kokkos::View<double *, grace::default_space> drl ("TOV_radii_spacing", N_POINTS) ; 

        
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
            rk45_t<3> solver{{0.,R_MAX}, {0., _pressC_loc, 0.}, 1e-04, 1e-02 } ; 
            auto cback = [&] (double const& r, std::array<double,3> const& state) -> std::array<double,3>
            { 
                return tov_rhs<eos_t>(r,state,eos) ; 
            } ; 
            bool stored = false ;
            massl(0) = 0. ;
            pressl(0) = _pressC_loc ; 
            nul(0)  = 0. ; 
        
            size_t ii = 0 ;
            while(true) {
                solver.advance_step( cback ) ; 
                drl(ii) = solver.dt          ; 
                ii++                         ; 
                massl(ii) = solver.state[0]  ; 
                pressl(ii) = solver.state[1] ;
                nul(ii) = solver.state[2]    ;
                rl(ii) = solver.t            ; 
                
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
        _npoints = static_cast<size_t>(h_tov_params(6)) ; 
        #if 0
        Kokkos::resize(massl, static_cast<size_t>(h_tov_params(6))) ; 
        Kokkos::resize(pressl, static_cast<size_t>(h_tov_params(6))) ; 
        Kokkos::resize(nul, static_cast<size_t>(h_tov_params(6))) ; 

        Kokkos::realloc(mass, static_cast<size_t>(h_tov_params(6))) ; 
        Kokkos::realloc(press, static_cast<size_t>(h_tov_params(6))) ; 
        Kokkos::realloc(nu, static_cast<size_t>(h_tov_params(6))) ; 
        
        Kokkos::deep_copy(mass,massl) ; 
        Kokkos::deep_copy(press,pressl) ; 
        Kokkos::deep_copy(nu,nul) ; 
        #endif 
        Kokkos::resize(massl, static_cast<size_t>(h_tov_params(6))) ; 
        Kokkos::resize(pressl, static_cast<size_t>(h_tov_params(6))) ; 
        Kokkos::resize(nul, static_cast<size_t>(h_tov_params(6))) ; 
        mass = massl; 
        press = pressl; 
        nu = nul ; 
        r = rl   ;
        dr = drl   ;
    } 
    //**************************************************************************************************
    //**************************************************************************************************
    /**
     * @brief Return initial data at a point
     * 
     * @param i x cell index
     * @param j y cell index
     * @param k z cell index
     * @param q quadrant index
     * @return grmhd_id_t Initial data at requested point
     */
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
        double const rL = Kokkos::max(Kokkos::sqrt(EXPR(
              math::int_pow<2>(x),
            + math::int_pow<2>(y),
            + math::int_pow<2>(z)
        )),  1e-45) ; 

        // This returns: ADM mass, pressure and metric potential
        // at this radius.
        auto sol = get_solution(rL) ;

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

        double const mL = sol[0] ; 
        double const nuL = sol[2] ; 
        
        id.vx = 0 ; id.vy = 0; id.vz = 0;
        
        /* Set the metric */
        id.alp   = 
            Kokkos::exp(nuL) ; 
        id.betax = 0. ; 
        id.betay = 0. ; 
        id.betaz = 0. ; 

        id.gxx = (rL*rL*rL - 2*mL*(y*y + z*z))/(rL*rL*(-2*mL + rL));
        id.gxy = (2*mL*x*y)/(rL*rL*(-2*mL + rL));
        id.gxz = (2*mL*x*z)/(rL*rL*(-2*mL + rL));
        id.gyy = (rL*rL*rL - 2*mL*(x*x + z*z))/(rL*rL*(-2*mL + rL));
        id.gyz = (2*mL*y*z)/(rL*rL*(-2*mL + rL));
        id.gzz = (rL*rL*rL - 2*mL*(x*x + y*y))/(rL*rL*(-2*mL + rL));

        id.kxx = 0. ;
        id.kxy = 0. ;
        id.kxz = 0. ;
        id.kyy = 0. ;
        id.kyz = 0. ;
        id.kzz = 0. ;
        
        return std::move(id); 
    }
    //**************************************************************************************************

    //**************************************************************************************************
    std::array<double,3> GRACE_HOST_DEVICE
    get_solution(double const R) const
    {
        if( R > _R ) {
            return std::array<double,3>{
                _M, 
                0,
                0.5*Kokkos::log(1.-2*_M/R)
            } ; 
        } else { 
            auto out = interp_solution(R) ; 
            if ( R > _transition_radius ) {
                double const w = _R - _transition_radius ; 
                auto const phi = 0.5 * ( 1 + Kokkos::tanh((R-_transition_radius)/(w))) ; 
                out[0] = phi * out[0] + ( 1. - phi ) * _M ; 
                out[1] = phi * out[1] ;
                out[2] = phi * out[2] + ( 1. - phi ) * 0.5*Kokkos::log(1.-2*_M/R);
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
    //**************************************************************************************************
    //**************************************************************************************************
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
    //**************************************************************************************************
    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rhoC, _pressC;                            //!< Central density 
    double _M, _R;                                    //!< Mass and Radius
    double _compactness, _nu_corr ;                   //!< Compactness and matching of metric potential
    double _transition_radius, _press_atm ;           //!< Atmosphere pressure and transition radius
    size_t _npoints ;                                 //!< Number of points in solution
    static constexpr double _dr = R_MAX / N_POINTS ;  //!< Spacing
    Kokkos::View<double *, grace::default_space> mass, press, nu, r, dr ; //!< Arrays containing TOV solution 
    //**************************************************************************************************
} ;
//**************************************************************************************************
} /* namespace grace */
//**************************************************************************************************
#endif /* GRACE_PHYSICS_ID_TOV_HH */