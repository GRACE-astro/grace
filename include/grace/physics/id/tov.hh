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
 #include <grace/utils/integration.hh>
 
 #include <Kokkos_Core.hpp>
 
 #include <fstream>
 
 //**************************************************************************************************
 #define R_MAX 30 
 #define N_POINTS 1000000
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
  * @return std::array<double,4> Array containing rhs: (dm/dr,dP/dr,dnu/dr,dR_iso/dr).
  */
 template< typename eos_t >
 static std::array<double,4> GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
 tov_rhs(double const& r, std::array<double,4> const& state, eos_t const& _eos) {
    if ( r < 1e-12 ) {
        return {0,0,0,1.0} ; 
    }

    double m     = state[0] ; 
    double press = state[1] ; 
    double phi   = state[2] ; 
    double riso  = state[3] ; 

    unsigned int err ;
    double ye = 0 ;
    auto const e = _eos.energy_cold__press_cold_ye(press, ye, err) ; 
    double const A = 1.0/(1.0-2.0*m/r) ; 
    double const B = (m + 4*M_PI*SQR(r)*r * press) / (SQR(r)) ; 

    return std::array<double,4> {
        4. * M_PI * math::int_pow<2>(r) * e 
        , -(e + press) * A * B   
        , A * B 
        , riso / (r)*sqrt(A)
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
     using state_t = grace::var_array_t ; //!< State array type
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
         , atmo_params_t atmo_params
         , double rhoC, double press_floor )
         : _eos(eos), _pcoords(pcoords), _atmo_params(atmo_params), _rhoC(rhoC), _press_floor(press_floor)
     { 
 
         Kokkos::View<double *, grace::default_space> tov_params("TOV_parameters", 7) ; 
         mass = Kokkos::View<double *, grace::default_space>("mass", N_POINTS) ;
         press = Kokkos::View<double *, grace::default_space>("press", N_POINTS) ;
         nu = Kokkos::View<double *, grace::default_space>("nu", N_POINTS) ;
         r = Kokkos::View<double *, grace::default_space>("r", N_POINTS) ;
         dr = Kokkos::View<double *, grace::default_space>("dr", N_POINTS) ;
         r_iso = Kokkos::View<double *, grace::default_space>("r_iso", N_POINTS) ;
 
         auto rl = r ; auto massl = mass ; auto pressl = press ; auto drl = dr ; auto nul = nu ; auto risol = r_iso ; 

         GRACE_INFO("In TOV setup.") ; 
         Kokkos::parallel_for("solve_tov", 1, KOKKOS_LAMBDA (int dummy){
             unsigned int err ; 
             double ye, eps;     
             double temp = 0.0; 
             double rho = rhoC ;

             /* Find central pressure, eps, ye */
             double const _pressC_loc = eos.press_eps_ye__beta_eq__rho_temp(eps,ye,rho,temp,err) ; 
             rk45_t<4> solver{{0.,R_MAX}, {0., _pressC_loc, 0., 0.}, 1e-04, 1e-02 } ;
 
             auto cback = [&] (double const& r, std::array<double,4> const& state) -> std::array<double,4>
             { 
                 return tov_rhs<eos_t>(r,state,eos) ; 
             } ; 
 
             massl(0) = 0. ;
             pressl(0) = _pressC_loc ; 
             nul(0)   = 0.0 ; 
             risol(0) = 0.0 ; 
         
             size_t ii = 0 ;
             while(true) {
                 solver.advance_step( cback ) ; 
                 drl(ii) = solver.dt          ; 
                 ii++                         ; 
                 massl(ii) = solver.state[0]  ; 
                 pressl(ii) = solver.state[1] ;
                 nul(ii) = solver.state[2]    ;
                 rl(ii) = solver.t            ; 
                 risol(ii) = solver.state[3]  ;
                 
                 if ( ( solver.state[1] <= press_floor) 
                   or ( solver.state[1] <= 0.0 ) 
                   or (ii>=N_POINTS-1) ) 
                {
                    break ; 
                } 
                 
             }
            // find M and R at the edge:
            auto const _linterp = [] (double x, double x0, double x1, double y0, double y1) {
                double lambda = (x-x0)/(x1-x0) ;
                return y0 * (1.0-lambda) + y1 * lambda ;  
            } ; 
            double r_edge = _linterp(press_floor, pressl(ii-1), pressl(ii), rl(ii-1), rl(ii)) ; 
            double m_tov = _linterp(press_floor, pressl(ii-1), pressl(ii), massl(ii-1), massl(ii)) ; 

            // replace last solved point with edge 
            pressl(ii) = press_floor ; 
            massl(ii)  = m_tov ; 
            nul(ii)    = _linterp(r_edge,rl(ii-1),rl(ii),nul(ii-1),nul(ii)) ; 
            risol(ii)  = _linterp(r_edge,rl(ii-1),rl(ii),risol(ii-1),risol(ii)) ; 

            rl(ii)  = r_edge ; 
            drl(ii) = r_edge - rl(ii-1) ; 

            // rescale nu and r_iso 
            // to match schwarzschild exterior 
            double const corr = 0.5 * log(1-2*m_tov/r_edge) - nul(ii) ; 
            double const riso_corr = 0.5*(r_edge-m_tov+sqrt(r_edge*(r_edge-2*m_tov)))/risol(ii) ; 
            for ( int j=0; j<ii+1; ++j) {
                nul(j) += corr ; 
                risol(j) *= riso_corr ; 
            }

            // store these to extract them 
            // from the kernel 
            tov_params(0) = r_edge ; 
            tov_params(1) = m_tov ; 
            tov_params(2) = _pressC_loc ; 
            tov_params(3) = solver.state[2] ; 
            tov_params(4) = press_floor ; 
            tov_params(5) = ii+1  ; 
            tov_params(6) = risol(ii) ; 
         }) ; 
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
         _M = h_tov_params(1) ; 
         _R = h_tov_params(0) ; 
         _R_iso = h_tov_params(6) ; 
         _pressC = h_tov_params(2) ;
         _compactness = _M/_R ; 
         _nu_corr = 0.5 * log(1-2*_compactness) - h_tov_params(3) ;  
         _press_atm = h_tov_params(4) ; 
         _npoints = static_cast<size_t>(h_tov_params(5)) ;
 
         Kokkos::resize(mass, static_cast<size_t>(_npoints)) ; 
         Kokkos::resize(press, static_cast<size_t>(_npoints)) ; 
         Kokkos::resize(nu, static_cast<size_t>(_npoints)) ; 
         Kokkos::resize(r, static_cast<size_t>(_npoints)) ; 
         Kokkos::resize(dr, static_cast<size_t>(_npoints)) ; 
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
 
         unsigned int err ; 
         
         /* Check if we are inside the star */
         double ye_atm  = _atmo_params.ye_fl  ; 
         double rho_atm = _atmo_params.rho_fl ; 
         double press_atm = _eos.press_cold__rho_ye(rho_atm,ye_atm,err) ; 

         if ( sol[0] > 1.001 * press_atm ) {
             id.press = sol[0] ; 
             id.ye    = _eos.ye_beta_eq__press_cold(sol[0],err) ;
             // Get rho and eps from press 
             double eps ; 
             id.rho   = _eos.rho__press_cold_ye(sol[0], id.ye, err) ; 
         } else {
             id.rho   = rho_atm   ;
             id.ye    = ye_atm    ;
             id.press = press_atm ; 
         }
 
         double const nuL = sol[1] ; 
         
         id.vx = 0 ; id.vy = 0; id.vz = 0;
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
     double _press_floor ;                             //!< Pressure at star's edge
     double _M, _R, _R_iso;                            //!< Mass and Radius
     double _compactness, _nu_corr ;                   //!< Compactness and matching of metric potential
     double _press_atm ;                               //!< Atmosphere pressure
     size_t _npoints ;                                 //!< Number of points in solution
     double _C ;                                       //!< Conversion between isotropic and Schwartzschild coordinates 
     Kokkos::View<double *, grace::default_space> mass, press, nu, r, dr, r_iso ; //!< Arrays containing TOV solution 
     //**************************************************************************************************
 } ;
 //**************************************************************************************************
 } /* namespace grace */
 //**************************************************************************************************
 #endif /* GRACE_PHYSICS_ID_TOV_HH */
