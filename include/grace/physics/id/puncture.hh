/**
 * @file linear_gw.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-01-15
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

 #ifndef GRACE_PHYSICS_ID_BRILL_WAVE_HH
 #define GRACE_PHYSICS_ID_BRILL_WAVE_HH
 
 #include <grace_config.h>
 
 #include <grace/utils/inline.h>
 #include <grace/utils/device/device.h>
 
 #include <grace/data_structures/variable_indices.hh>
 #include <grace/data_structures/variables.hh>
 #include <grace/data_structures/variable_properties.hh>
 #include <grace/physics/grmhd_helpers.hh>
 
 //**************************************************************************************************
 namespace grace {
 //**************************************************************************************************
 //**************************************************************************************************
 /**
  * @brief Initial data for vacuum gauge wave test
  * \ingroup initial_data
  * @tparam eos_t Type of equation of state
  */
 //**************************************************************************************************
 template < typename eos_t >
 struct puncture_id_t {
     //**************************************************************************************************
     using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //! Type of state array
     //**************************************************************************************************
     /**
      * @brief Construct a new gauge_wave_id kernel
      * 
      * @param eos Equation of state
      * @param pcoords physical coordinates
      * @param A Amplitude of gauge wave
      * @param d Wavelength of gauge wave
      */
     puncture_id_t(
           eos_t eos 
         , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
         , double m)
         : _eos(eos)
         , _pcoords(pcoords)
         , _m(m)
     {} 
     //**************************************************************************************************
     //**************************************************************************************************
     /**
      * @brief Return gauge wave initial data at a point
      * 
      * @param i x cell index
      * @param j y cell index
      * @param k z cell index
      * @param q quadrant index
      * 
      * @return grmhd_id_t Initial data objects for gauge wave test
      */
     grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
     operator() (VEC(int const i, int const j, int const k), int const q) const 
     {
        double const x = _pcoords(VEC(i,j,k),0,q);
        double const y = _pcoords(VEC(i,j,k),1,q);
        double const z = _pcoords(VEC(i,j,k),2,q);

        double const r_iso = Kokkos::sqrt(x*x+y*y+z*z) ; 

       // double const r_ = 0.5 * ( 
        //    Kokkos::sqrt(r*r-2*_m*r) + r - _m 
        //) + 1e-45 ; 

        grmhd_id_t id ; 
        id.rho   = 0 ; 
        id.press = 0 ;
        id.ye    = 0 ;
        id.vx = id.vy = id.vz = 0;

        
        double const psi = 1 + _m / (2*r_iso+1e-45); 

        id.betax = id.betay = id.betaz = 0 ; 

        id.gxx = id.gyy = id.gzz = math::int_pow<4>(psi) ; 
        id.gxy = id.gxz = id.gyz = 0 ; 

        id.alp = 1.; // math::int_pow<2>((1-_m/(2*r_iso))/(1+_m/(2*r_iso))) ; 

        id.kxx = id.kxy = id.kxz = id.kyz = id.kyy = id.kzz = 0 ;

        return std::move(id) ; 
     }
     //**************************************************************************************************
 
     //**************************************************************************************************
     eos_t   _eos         ;                            //!< Equation of state object 
     grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
     double _m ;                                   //!< Left and right states
     //**************************************************************************************************  
 } ; 
 //**************************************************************************************************
 //**************************************************************************************************
 }
 //**************************************************************************************************
 #endif /* GRACE_PHYSICS_ID_PUNCTURE_HH */