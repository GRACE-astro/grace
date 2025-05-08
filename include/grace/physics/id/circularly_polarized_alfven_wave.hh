/**
 * @file circularly_polarized_alfven_wave.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-05-08
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

#ifndef GRACE_PHYSICS_ID_CP_LARGE_AMPL_ALFVEN_WAVE_HH
#define GRACE_PHYSICS_ID_CP_LARGE_AMPL_ALFVEN_WAVE_HH

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

// namespace linear_mhd_utils{

//     enum WAVE_TYPE{
//         CONTACT, 
//         SLOW_MAGNETOSONIC,
//         ALFVEN,
//         FAST_MAGNETOSONIC,
//         N_WAVE_TYPES
//     };

//     enum WAVE_DIRECTION{
//         RIGHT, 
//         LEFT,
//         STANDING
//     };
// }



//**************************************************************************************************
/**
 * \defgroup initial_data Initial Data
 * 
 */
//**************************************************************************************************
/**
 * @brief Large amplitude circularly polarized Alfven wave - initial data kernel
 * \ingroup initial_data
 * @tparam eos_t type of equation of state
 * @note this kernel has to be adjused if magnetic field initialization method and storage location changes in the future (e.g. vec pot);
 * We base the setup on: 
 * https://arxiv.org/pdf/0704.3206
 * and 
 * https://arxiv.org/pdf/0912.4692 
 * the latter just for the sanity-check computation of the Alfven speed
 * @warning This MHD ID kernel will have to be adapted in the future if vector potential is used instead 
 * 
 */
template < typename eos_t >
struct cp_alfven_wave_id_t {

    // using WAVE_TYPE = grace::linear_mhd_utils::WAVE_TYPE;
    // using WAVE_DIRECTION = grace::linear_mhd_utils::WAVE_DIRECTION;
    //**************************************************************************************************
    using state_t = grace::var_array_t<GRACE_NSPACEDIM> ; //!< Type of state vector
    //**************************************************************************************************

    //**************************************************************************************************
    /**
     * @brief Construct a new id kernel object
     * 
     * @param eos Equation of state
     * @param pcoords Physical coordinate array
     * @param rho uniform density in the problem
     * @param press uniform pressure in the problem
     * @param ampl amplitude of the wave
     * @param B0 B0 magnitude of the guiding magnetic field
     * @param k wavevector magnitude
     * @param v propagation speed of the medium (v)
     // for v=0 we have a Standing CP large-amplitude Alfven wave
     // note that v_A below will be the phase speed
     
     */
     cp_alfven_wave_id_t(
          eos_t eos 
        , grace::coord_array_t<GRACE_NSPACEDIM> pcoords 
        , const double rho
        , const double press
        , const double B0
        , const double ampl
        , const double k
        , const double v
    )
        : _eos(eos)
        , _pcoords(pcoords)
        , _rho(rho)
        , _press(press)
        , _B0(B0)
        , _ampl(ampl)
        , _k(k)
        , _v(v)

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
 
        // initialize the hydro state
        id.rho   = _rho;
        id.press = _press;
        
        // guiding field:
        id.bx = _B0;

        unsigned int err ; 
        id.ye  = _eos.ye_beta_eq__press_cold(id.press, err);

        double h, cs2_loc; 
        auto  eps_tmp = eos.eps_h_csnd2__press_rho_ye(
            h,cs2_loc,id.press,id.rho,id.ye,err
        ) ; 
    
        // for the subsequent part, we need to set the speed of the wave 
        // eq (85) in https://arxiv.org/pdf/0704.3206

        const double tmp = (_B0*_B0)/(_rho*h + _B0*_B0 * (1+_ampl*_ampl) );
        const double v_A = tmp / (0.5 *( 1. + Kokkos::sqrt(1. - Kokkos::pow(2*_ampl*tmp, 2))     ));

        // with this at hand, we proceed to compute the transverse magnetic field and velocity field components:
        // eq (82) at t=0
        id.by = _ampl * _B0 * Kokkos::cos(_k*x);
        id.bz = _ampl * _B0 * Kokkos::sin(_k*x);
    
        id.vx    = _v ; //fluid bulk velocity 
        id.vy    = -v_A*id.by/id.bx ; // transverse fluid velocity components associated with the propagation of the Alfven wave
        id.vz    = -v_A*id.bz/id.bx ;


        // Performing sanity check for the Alfven wave speed:
        // eq. (38) in the generic expression of https://arxiv.org/pdf/0912.4692
        //
        // compute the lorentz factor
        // (id.vi are the Eulerian velocities here)
        const double W = 1.0/Kokkos::sqrt(1 - id.vx*id.vx - id.vy*id.vy - id.vz*id.vz);

        metric_array_t minkowski_metric ({1.,0.,0.,1.,0.,1.},{0.,0.,0.},1.) ; 

        // compute the comoving magnetic field and its square
        std::array<double, 4> smallb ; 
        get_smallb_from_eulerianB(minkowski_metric,
            {id.bx, id.by, id.bz},            
            {id.vx, id.vy, id.vz},            
            smallb)

        
        double const b2 = id.bx*id.bx + id.by*id.by + id.bz*id.bz;

        // compute the total enthalpy:
        const double varEps = _rho* h + b2; 

        double const v_A_plus  = (id.bx + Kokkos::sqrt(varEps)*id.vx ) / (smallb[0] + Kokkos::sqrt(varEps) * W) ;
        double const v_A_minus = (id.bx - Kokkos::sqrt(varEps)*id.vx ) / (smallb[0] - Kokkos::sqrt(varEps) * W)

        // if neither of these velocities (derived from a full-wave decomposition of the whole SRMHD system) 
        // match the Alfven velocities obtained further above, we have a problem:

        if(math::abs(v_A - v_A_plus)<1e-4 && math::abs(v_A - v_A_minus)<1e-4){
            printf("Alfven velocity inconsistent!\n");
        }
        
        return std::move(id) ; 

    }
    //**************************************************************************************************

    //**************************************************************************************************
    eos_t   _eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    // WAVE_DIRECTION _wave_movement ;                    
    double _rho ;
    double _press ; 
    double _ampl ; 
    double _B0 ; 
    double _k ; 
    double _v ; 
    //**************************************************************************************************
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
//**************************************************************************************************
#endif  /* GRACE_PHYSICS_ID_CP_LARGE_AMPL_ALFVEN_WAVE_HH */