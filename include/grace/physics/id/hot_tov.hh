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

 #ifndef GRACE_PHYSICS_ID_HOT_TOV_HH
 #define GRACE_PHYSICS_ID_HOT_TOV_HH
 
 #include <grace_config.h>
 
 #include <grace/utils/inline.h>
 #include <grace/utils/device.h>
 
 #include <grace/utils/runge_kutta.hh>
 #include <grace/data_structures/variable_indices.hh>
 #include <grace/data_structures/variables.hh>
 #include <grace/data_structures/variable_properties.hh>
 #include <grace/physics/grmhd_helpers.hh>
 #include <grace/amr/amr_functions.hh>
 #include <grace/utils/integration.hh>
 #include <grace/physics/id/tov.hh>
 #include <grace/physics/m1_helpers.hh>
 
 #include <Kokkos_Core.hpp>
 
 #include <fstream>
 
 //**************************************************************************************************
 #define R_MAX 50 
 #define N_POINTS 500000
 //**************************************************************************************************
 namespace grace {
 //**************************************************************************************************

 //**************************************************************************************************
 /**
  * @brief TOV initial data kernel.
  * \ingroup initial_data
  * @tparam eos_t Eos type.
  */
 template < typename eos_t >
 struct hot_tov_id_t {
     //**************************************************************************************************
     using state_t = grace::var_array_t ; //!< State array type
     //**************************************************************************************************
     /**
      * @brief Construct a new tov id kernel
      * 
      * @param eos Equation of state
      * @param pcoords Physical coordinates array 
      * @param rhoC Central density [code units]
      */
     hot_tov_id_t(
           eos_t eos
         , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
         , atmo_params_t atmo_params
         , double rhoC, double press_floor, double dr, double pert_amp 
         , m1_atmo_params_t atmo 
         , m1_excision_params_t excision)
         : _eos(eos), _pcoords(pcoords), _atmo_params(atmo_params), _rhoC(rhoC), _press_floor(press_floor), _dr(dr), _pert_amp(pert_amp),
         _atmo(atmo), _excision(excision)
     { 
 
        Kokkos::View<double *, grace::default_space> tov_params("TOV_parameters", 7) ; 
        mass = Kokkos::View<double *, grace::default_space>("mass", N_POINTS) ;
        press = Kokkos::View<double *, grace::default_space>("press", N_POINTS) ;
        nu = Kokkos::View<double *, grace::default_space>("nu", N_POINTS) ;
        r = Kokkos::View<double *, grace::default_space>("r", N_POINTS) ;
        r_iso = Kokkos::View<double *, grace::default_space>("r_iso", N_POINTS) ;

        GRACE_INFO("In TOV setup.") ; 
        solve_tov(eos,rhoC,press_floor,dr,mass,r,press,nu,r_iso,tov_params);
        Kokkos::fence() ; 
        GRACE_INFO("TOV solver done.") ; 
        auto h_tov_params = Kokkos::create_mirror_view(tov_params) ; 
        Kokkos::deep_copy(h_tov_params, tov_params) ; 
        GRACE_INFO("TOV solver (all in code units):\n"
                "   Central density  : {}\n"
                "   Central pressure : {}\n"
                "   Mass             : {}\n"   
                "   Radius           : {}\n"
                "   Isotropic Radius : {}", _rhoC, h_tov_params(2), h_tov_params(1), h_tov_params(0),h_tov_params(6)) ; 
        // check that the solver actually solved
        ASSERT(static_cast<size_t>(h_tov_params(5))>1, "Fatal error in TOV solver, npt==1.");

        
        _M = h_tov_params(1) ; 
        _R = h_tov_params(0) ; 
        _R_iso = h_tov_params(6) ; 
        _pressC = h_tov_params(2) ;
        _compactness = _M/_R ; 
        _press_atm = h_tov_params(4) ; 
        _npoints = static_cast<size_t>(h_tov_params(5)) ;

        Kokkos::resize(mass, static_cast<size_t>(_npoints)) ; 
        Kokkos::resize(press, static_cast<size_t>(_npoints)) ; 
        Kokkos::resize(nu, static_cast<size_t>(_npoints)) ; 
        Kokkos::resize(r, static_cast<size_t>(_npoints)) ; 
        Kokkos::resize(r_iso, static_cast<size_t>(_npoints)) ;
   
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
        auto rs  = get_r_schwarzschild(rL) ; 

        grmhd_id_t id ; 

        eos_err_t err ; 
        
        /* Check if we are inside the star */
        double ye_atm    = _atmo_params.ye_fl  ; 
        double rho_atm   = _atmo_params.rho_fl ; 
        // note: this has to be the "cold EOS temperature"
        // otherwise atmo is inconsistent! 
        double temp_atm  = _atmo_params.temp_fl ; 
        double press_atm = _eos.press_cold__rho(rho_atm, err) ; 

        if ( sol[0] > 1.001 * press_atm ) {
            id.press = sol[0] ; 
            id.ye    = _eos.ye_cold__press(sol[0],err) ;
            // Get rho and eps from press 
            id.rho   = _eos.rho__press_cold(sol[0], err) ; 
            id.eps   = _eos.eps_cold__rho(id.rho, err) ; 
            // and the rest 
            // in case this is a const entropy slice 
            //id.temp = _eos.temp_cold__rho(id.rho, err) ; 
            id.temp = 20.0 ; 
            // in case this is a const temp slice 
            id.entropy = _eos.entropy_cold__rho(id.rho, err) ; 
            // perturb 
            double s[3] = {
                x/rL, y/rL, z/rL
            } ; 
            id.vx = _pert_amp * s[0] * rL ; 
            id.vy = _pert_amp * s[1] * rL ; 
            id.vz = _pert_amp * s[2] * rL ; 
        } else {
            id.rho   = rho_atm   ;
            id.ye    = ye_atm    ;
            id.temp  = temp_atm ; 
            // get the rest 
            double dummy ;
            id.press = _eos.press_eps_csnd2_entropy__temp_rho_ye(
                id.eps, dummy, id.entropy, id.temp, id.rho, id.ye, err
            ) ; 
            id.vx = id.vy = id.vz = 0.0 ; 
        }

        double const nuL = sol[1] ; 
        
        id.bx = id.by = id.bz = 0;
        /* Set the metric */
        id.alp   = 
            Kokkos::exp(nuL) ; 
        id.betax = 0. ; 
        id.betay = 0. ; 
        id.betaz = 0. ; 
        
        double const psi4 = rL>0 ? SQR((rs/rL)) : 1.0;
        id.gxx = id.gyy = id.gzz = psi4 ; 
        id.gxy = id.gxz = id.gyz =  0;

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
    std::array<double,2> GRACE_HOST_DEVICE
    get_solution(double const R) const
    {
        double const Rs = R * math::int_pow<2>( 1 + 0.5 * _M / R ) ; 
        if( R > _R_iso ) {
            return std::array<double,2>{
                0,
                0.5*Kokkos::log(1.-2*_M/Rs)
            } ; 
        } else { 
            return {
                interp_solution(R, r_iso, press   ),
                interp_solution(R, r_iso, nu      )  
            };
        } 
    }

    double GRACE_HOST_DEVICE 
    get_r_schwarzschild(double R) const {
    if( R > _R_iso ) {
            return SQR((1+0.5*_M/R)) * R ; 
        } else { 
            return interp_solution(R, r_iso, r) ; 
        } 
    }

    double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    interp_solution(
        double const R, 
        Kokkos::View<double*, grace::default_space> x,  
        Kokkos::View<double*, grace::default_space> y
    ) const {
        size_t idx = find_index(R ,x); 
        double lambda = (R - x(idx)) / ( x(idx+1) - x(idx) );
        return y(idx) * ( 1- lambda ) + y(idx+1) *  (lambda) ; 
    }
    //**************************************************************************************************
    //**************************************************************************************************
    size_t GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
    find_index(double const R, Kokkos::View<double*, grace::default_space> x) const { 
        int lower = 0;
        int upper = _npoints - 1;
        // simple bisection should do it
        while (upper - lower > 1) {
            int tmp = lower + (upper - lower) / 2;
            if (R < x(tmp))
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
    atmo_params_t _atmo_params ; 
    double _rhoC, _pressC;                            //!< Central density 
    double _press_floor, _dr ;                        //!< Pressure at star's edge
    double _M, _R, _R_iso;                            //!< Mass and Radius
    double _compactness ;                             //!< Compactness
    double _press_atm ;                               //!< Atmosphere pressure
    double _pert_amp  ;                               //!< Perturbation delta v^r = A r 
    size_t _npoints ;                                 //!< Number of points in solution
    double _C ;                                       //!< Conversion between isotropic and Schwartzschild coordinates 
    m1_atmo_params_t _atmo ; 
    m1_excision_params_t _excision ; 
    Kokkos::View<double *, grace::default_space> mass, press, nu, r, r_iso ; //!< Arrays containing TOV solution 
    //**************************************************************************************************
 } ;
 //**************************************************************************************************
 } /* namespace grace */
 //**************************************************************************************************
 #endif /* GRACE_PHYSICS_ID_HOT_TOV_HH */
