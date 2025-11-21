/**
 * @file fmparams.hh
 * @author Konrad Topolski (topolski @itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-10-17
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

#ifndef GRACE_PHYSICS_ID_FMTORUS_HH
#define GRACE_PHYSICS_ID_FMTORUS_HH

#include <grace_config.h>

#include <grace/utils/inline.h>
#include <grace/utils/device.h>

#include <grace/utils/grace_utils.hh>
#include <grace/utils/metric_utils.hh>
#include <grace/utils/runge_kutta.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/errors/error.hh>



// Kokkos utils and random
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <grace/physics/id/FMTorus/KerrSchild.hh>
#include <grace/physics/id/FMTorus/fmtorus_utils.hh>

namespace grace {


// Useful container for physical parameters of torus


/**
* @brief FMtorus initial data kernel.
* 
* @tparam eos_t Eos type
*         Due to the setup of the ID, the template
*         specializes exclusively for hybrid eos and a specific setup 
*         of the polytropic eos 
*/
template < typename eos_t> //, typename B_field_config_t >
//requires PiecewisePolytropicEOS<eos_t>
struct fmtorus_id_t {
    using state_t = grace::var_array_t ;

    // constructor: initializing the parameters, performing consistency checks on the ID
    fmtorus_id_t(
        eos_t eos_,
        grace::coord_array_t<GRACE_NSPACEDIM> pcoords_,
        torus_params_t _params,
        double pert_amp
    )
        : \
        eos(eos_), _pcoords(pcoords_), 
        params(_params),
        rand_pool64(Kokkos::Random_XorShift64_Pool<>(1234)),
        pert_amp(pert_amp)
    { 
        using Kokkos::exp ; 

        double const gm1 = params.gamma_adi - 1. ; 
        if ( params.fm_torus ) {
            params.l_peak = CalculateLFromRPeak(params, params.r_peak) ; 
        } else if ( params.chakrabarti_torus ) {
            CalculateCN(params,&params.c_param,&params.n_param) ; 
            params.l_peak = CalculateL(params,params.r_peak,1.0) ; 
        } else {
            ERROR("Unrecognized torus_id_type") ; 
        }
        params.log_h_edge = LogHAux(params,params.r_edge,1.0) ; 
        params.log_h_peak = LogHAux(params,params.r_peak,1.0) - params.log_h_edge ; 
        params.ptot_over_rho_peak = gm1/params.gamma_adi * (exp(params.log_h_peak)-1.0) ; 
        params.rho_peak = Kokkos::pow(params.ptot_over_rho_peak, 1.0/gm1) / params.rho_max ;
        
        GRACE_INFO("Some parameters, l_peak {} log(h)_edge {} log(h)_peak {} ptot_over_rho_peak {} rho_peak {} "
                 , params.l_peak , params.log_h_edge, params.log_h_peak, params.ptot_over_rho_peak, params.rho_peak);

        // find outer edge 
        double ra = params.r_peak ; 
        double rb = 2*ra ; 
        double log_h_trial = LogHAux(params,rb,1.) - params.log_h_edge ; 

        for (int iter=0; iter<10000; ++iter) {
            if (log_h_trial <= 0) {
            break;
            }
            rb *= 2.;
            log_h_trial = LogHAux(params, rb, 1.) - params.log_h_edge;
        }
        for (int iter=0; iter<10000; ++iter) {
            if (fabs(ra - rb) < 1.e-3) {
            break;
            }
            double  r_trial = (ra + rb) / 2.;
            if (LogHAux(params, r_trial, 1.) > params.log_h_edge) {
            ra = r_trial;
            } else {
            rb = r_trial;
            }
        }
        params.r_outer_edge = ra;
        GRACE_INFO("Found outer edge of torus {}", params.r_outer_edge) ; 
    } 

    // the main evaluation kernel as operator()
    grmhd_id_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    operator() (VEC(int i, int j, int k), int q) const 
    {
        // nb
        double const x = _pcoords(VEC(i,j,k),0,q);
        double const y = _pcoords(VEC(i,j,k),1,q);
        #ifdef GRACE_3D 
        double const z = _pcoords(VEC(i,j,k),2,q);
        #else 
        double const z = 0. ; 
        #endif 

        double const gm1 = params.gamma_adi - 1.0 ; 

        grmhd_id_t id ; 

        // at this radius.
        double dummy ;
        ComputeADMDecomposition(
            x,y,z, false, params.spin,
            &id.alp, &id.betax, &id.betay, &id.betaz,
            &dummy, &id.gxx, &id.gxy, &id.gxz, &id.gyy, &id.gyz, &id.gzz,
            &id.kxx, &id.kxy, &id.kxz, &id.kyy, &id.kyz, &id.kzz
        ) ; 

        double r,theta,phi ; 
        GetBoyerLindquistCoordinates(params, x, y, z, &r, &theta, &phi);

        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        double sin_phi = sin(phi);
        double cos_phi = cos(phi);

        // Account for tilt
        double  sin_vartheta;
        if (params.psi != 0.0) {
            double  x = sin_theta * cos_phi;
            double  y = sin_theta * sin_phi;
            double  z = cos_theta;
            double  varx = params.cos_psi * x - params.sin_psi * z;
            double  vary = y;
            sin_vartheta = sqrt(SQR(varx) + SQR(vary));
        } else {
            sin_vartheta = fabs(sin_theta);
        }

        // Determine if we are in the torus
        double log_h;
        bool in_torus = false;
        if (r >= params.r_edge) {
            log_h = LogHAux(params, r, sin_vartheta) - params.log_h_edge;  // (FM 3.6)
            if (log_h >= 0.0) {
                in_torus = true;
            }
        }

        double rho_bg, pgas_bg ; 
        if (r>1) {
            rho_bg = params.rho_min * pow(r, params.rho_pow);
            pgas_bg = params.pgas_min * pow(r, params.pgas_pow);
        } else {
            rho_bg = params.rho_excise ; 
            pgas_bg = params.pgas_excise ; 
        }
        
        
        double  rho = 0.0;
        double  pgas = 0.0;
        double  uu1 = 0.0;
        double  uu2 = 0.0;
        double  uu3 = 0.0;
        double  urad = 0.0;

        double  perturbation = 0.0;
        // Overwrite primitives inside torus
        if (in_torus) {
            auto rand_gen = rand_pool64.get_state(); // get random number state this thread
            perturbation = 2.0*pert_amp*(rand_gen.frand() - 0.5);
            rand_pool64.free_state(rand_gen);        // free state for use by other threads

            // Calculate thermodynamic variables
            double  ptot_over_rho = gm1/params.gamma_adi * (exp(log_h) - 1.0);
            rho = pow(ptot_over_rho, 1.0/gm1) / params.rho_peak;
            double  temp = ptot_over_rho;
            pgas = temp * rho;
            // Calculate velocities in Boyer-Lindquist coordinates
            double  u0_bl, u1_bl, u2_bl, u3_bl;
            CalculateVelocityInTiltedTorus(params, r, theta, phi,
                                            &u0_bl, &u1_bl, &u2_bl, &u3_bl);

            // Transform to preferred coordinates
            double  u0, u1, u2, u3;
            TransformVector(params, u0_bl, 0.0, u2_bl, u3_bl,
                            x, y, z, &u0, &u1, &u2, &u3);

            double  glower[4][4], gupper[4][4];
            ComputeMetricAndInverse(x, y, z, false, params.spin,
                                    glower, gupper);
            uu1 = (u1/u0 - gupper[0][1]/gupper[0][0])/id.alp;
            uu2 = (u2/u0 - gupper[0][2]/gupper[0][0])/id.alp;
            uu3 = (u3/u0 - gupper[0][3]/gupper[0][0])/id.alp;
        }
        

        id.vx = uu1 ; 
        id.vy = uu2 ; 
        id.vz = uu3 ; 

        id.rho = rho ; 
        id.press = pgas * ( 1. + perturbation ) ;

	if ( ( rho < rho_bg * (1 + 1e-3) ) or ( pgas < pgas_bg * ( 1 + 1e-3 ) ) ) {
	  id.rho = rho_bg; id.press = pgas_bg;
	  id.vx  = id.vy = id.vz = 0 ; 
	}

        id.ye = 0 ; 
        return id ;
    }

    // arguments to the constructor: 
    eos_t   eos         ;                            //!< Equation of state object 
    grace::coord_array_t<GRACE_NSPACEDIM> _pcoords ;  //!< Physical coordinates of cell centers
    /*============================================================*/
    torus_params_t params ; 
    Kokkos::Random_XorShift64_Pool<> rand_pool64;
    double pert_amp;
    /*============================================================*/
};

}

/* namespace grace */
#endif /* GRACE_PHYSICS_ID_FMTORUS_HH */
