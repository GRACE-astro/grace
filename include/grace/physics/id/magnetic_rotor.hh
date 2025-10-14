/**
 * @file magnetic_rotor.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-05-12
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

#ifndef GRACE_PHYSICS_ID_MAGNETIC_ROTOR_HH
#define GRACE_PHYSICS_ID_MAGNETIC_ROTOR_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
// #include <grace/utils/device/device.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>

//**************************************************************************************************
namespace grace {



//**************************************************************************************************
/**
 * \defgroup initial_data Initial Data
 * 
 */
//**************************************************************************************************
/**
 * @brief Magnetic rotor ID - initial data kernel
 * \ingroup initial_data
 * @tparam eos_t type of equation of state
 * @note this kernel has to be adjused if magnetic field initialization method and storage location changes in the future (e.g. vec pot);
 * We base the setup on the rather recent: 
 * https://arxiv.org/pdf/2407.20946 
 * @warning This MHD ID kernel will have to be adapted in the future if vector potential is used instead 
 * 
 */
template < typename eos_t >
struct magnetic_rotor_id_t {

    //**************************************************************************************************
    using state_t = grace::var_array_t ; //!< Type of state vector
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Construct a new id kernel object
     * 
     * @param eos Equation of state
     * @param pcoords Physical coordinate array
     * @param rho_in density of the rotor
     * @param rho_out density of the outside medium
     * @param press uniform pressure in the problem
     * @param B0 B0 magnitude of the initially uniform and isotropic magnetic field

     */
     magnetic_rotor_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , const double rho_in
        , const double rho_out
        , const double press
        , const double B0
    )
        : _eos(eos)
        , _pcoords(pcoords)
        , _rho_in(rho_in)
        , _rho_out(rho_out)
        , _press(press)
        , _B0(B0)
    {} 
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Obtain initial data at a point
     * 
     * @param i x cell index 
     * @param j y cell index 
     * @param k z cell index
     * @param q Quadrant index
     * @return grmhd_id_t Initial data at this point
     */
    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int const i, int const j, int const k), int const q) const 
    {

        grmhd_id_t id ; 

        const double x = _pcoords(VEC(i,j,k),0,q);
        const double y = _pcoords(VEC(i,j,k),1,q);
        const double z = _pcoords(VEC(i,j,k),2,q);
   
        // set Minkowski metric 
        id.betax = 0; id.betay=0; id.betaz = 0; 
        id.alp = 1 ; 
        id.gxx = 1; id.gyy = 1; id.gzz = 1;
        id.gxy = 0; id.gxz = 0; id.gyz = 0 ;
        id.kxx = 0; id.kyy = 0; id.kzz = 0 ;
        id.kxy = 0; id.kxz =0 ; id.kyz = 0 ; 

        // set the Lagrange multiplier to zero
        #ifdef GRACE_ENABLE_B_FIELD_GLM
        id.phi_glm = 0.;
        #endif 
 
        double const r = Kokkos::sqrt(math::int_pow<2>(x) + math::int_pow<2>(y)); 
        double const phi = Kokkos::atan2(y, x);

        const double Omega = 9.95;
        const double vel_norm = r * Omega;
        // initialize the hydro state
        if(r <= 0.1){
            id.rho   = _rho_in;
            id.vx =  -vel_norm * Kokkos::sin(phi);
            id.vy =   vel_norm * Kokkos::cos(phi);
        }
        else{
            id.rho   = _rho_out;
            id.vx = 0.0;
            id.vy = 0.0;
        }
        // pressure
        id.press = _press;
    
        // magnetic guiding field:
        id.bx = _B0;
        id.by = 0.0;
        id.bz = 0.0;
    
        id.vz    = 0.0 ;

        unsigned int err ; 
        id.ye  = _eos.ye_beta_eq__press_cold(id.press, err);
       
        return std::move(id) ; 
    }


    //**************************************************************************************************

    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    double _rho_in ;
    double _rho_out ;
    double _press ; 
    double _B0 ; 
    //**************************************************************************************************
} ; 

//**************************************************************************************************
//**************************************************************************************************
}
//**************************************************************************************************
#endif  /* GRACE_PHYSICS_ID_MAGNETIC_ROTOR_HH */