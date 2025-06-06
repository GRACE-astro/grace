/**
 * @file advanced_riemann_solvers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-28
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
#ifndef GRACE_UTILS_ADVANCED_RIEMANN_SOLVERS_HH
#define GRACE_UTILS_ADVANCED_RIEMANN_SOLVERS_HH

#include <grace_config.h>

#include <grace/utils/math.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device.h> 
#include <grace/utils/metric_utils.hh>
#include <grace/data_structures/macros.hh>
#include <grace/utils/riemann_solvers.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <Kokkos_Core.hpp> 

namespace grace {

/**
 * @brief HLLC riemann solver in direction idir.
 * 
 * @tparam idir The direction of the riemann problem being solved. 
 */
template< int idir > 
struct hllc_riemann_solver_t {

    using tetrad_t = std::array<std::array<double,4>,4> ;
    
 public:
     

    GRACE_HOST_DEVICE
    hllc_riemann_solver_t( grace::metric_array_t const& _metric )
     : metric(_metric)
    {
        get_tetrad_basis(metric,inertial_tetrad,inertial_cotetrad) ; 
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    transform_velocities_to_tetrad_frame( double& u0
                                        , grace::grmhd_prims_array_t& prims
                                        , std::array<double,3>& uD ) const 
    {
        std::array<double,4> umu { 
              u0
            , prims[VXL]*u0 
            , prims[VYL]*u0
            , prims[VZL]*u0
        } ; 

        //u0 = metric.contract_4dvec_4dcovec(inertial_cotetrad[0],umu) ; !TODO
        for(int ii=0; ii<3;++ii) {
            uD[ii] = umu[ii+1] ; // metric.contract_4dvec_4dcovec(inertial_cotetrad[1+ii],umu) ; !TODO
            // In the tetrad frame lower and upper spatial indices are the same
            prims[VXL+ii]  = uD[ii] / u0 ; 
        }
    }
    #ifdef GRACE_DO_MHD
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    transform_magnetic_fields_to_tetrad_frame(grace::grmhd_prims_array_t& prims) const 
    {
        std::array<double,3> B { 
              prims[BXL] 
            , prims[BYL]
            , prims[BZL]
        } ; 

        //u0 = metric.contract_4dvec_4dcovec(inertial_cotetrad[0],umu) ; !TODO
        for(int ii=0; ii<3;++ii) {
            for (int jj=0; jj<3; ++jj) {
                prims[BXL+ii]  =  inertial_cotetrad[ii][jj]*B[jj]; 
            }
        }
    }
    #endif // GRACE_DO_MHD
    /**
     * @brief Compute HLLC fluxes for relativistic Hydro.
     * 
     * @param fL Left fluxes.
     * @param fR Right fluxes.
     * @param uL Left conserved state.
     * @param uR Right conserved state.
     * @param pL Left primitive state (in local frame).
     * @param pR Right primitive state (in local frame).
     * @param cmin lambdaL (this has \b positive sign).
     * @param cmax lambdaR (this has \b positive sign).
     * @return grace::grmhd_cons_array_t The HLLC flux.
     * 
     * Note that in GRACE notation the velocity is not the eulerian one but rather
     * \f$v^i=u^i/u^t\f$. Also, our conserved energy density \f$\tau\f$ follows the 
     * Valencia notation and is defined as \f$n^\mu n^\nu T_{\mu\nu} - D\f$. For this
     * reason some of the terms in this function are different from what appears in 
     * https://arxiv.org/abs/2205.04487 which is the main source followed in this 
     * implementation.
     */
    grace::grmhd_cons_array_t 
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( 
          grace::grmhd_cons_array_t const& fL 
        , grace::grmhd_cons_array_t const& fR 
        , grace::grmhd_cons_array_t const& uL 
        , grace::grmhd_cons_array_t const& uR 
        , grace::grmhd_prims_array_t const& pL 
        , grace::grmhd_prims_array_t const& pR
        , double const cmin 
        , double const cmax )
    {
        grace::grmhd_cons_array_t uHLLE, fHLLE, uHLLC, fHLLC; 
        hll_riemann_solver_t hlle_solver ; 
        int var_indices[] = {DENSL, TAUL, STXL, STYL, STZL, BXL, BYL, BZL, YESL, ENTSL} ; 
        for( int ii=0; ii<10; ++ii) {
            int const ivar = var_indices[ii] ; 
            fHLLE[ivar] = 
                hlle_solver(fL[ivar],fR[ivar],uL[ivar],uR[ivar],cmin,cmax) ; 
            uHLLE[ivar] = 
                hlle_solver.get_state(fL[ivar],fR[ivar],uL[ivar],uR[ivar],cmin,cmax) ;
        }

        // Magnetic Field is constant on the face, so we have to copy it.
        fHLLE[BGXL+idir] = fL[BGXL+idir] ;

        double const vi = get_interface_velocity() ; // Carlo wrote TODO but seems to be fine.
        double const lambdaC = get_contact_wave_speed(uHLLE,fHLLE) ; 

        grace::grmhd_cons_array_t ucL, ucR ; 

        // Check for magnetic field strength
        bool const has_magnetic_field = (fHLLE[BGXL+idir] > 1e-15);

        // Common calculations for both cases
        constexpr int t1 = (idir + 1) % 3;
        constexpr int t2 = (idir + 2) % 3;

        // Set magnetic field components (same for both cases)
        ucL[BGXL+idir] = uHLLE[BGXL+idir];
        ucR[BGXL+idir] = uHLLE[BGXL+idir];
        ucL[BGXL + t1] = uHLLE[BGXL + t1];
        ucR[BGXL + t1] = uHLLE[BGXL + t1];
        ucL[BGXL + t2] = uHLLE[BGXL + t2];
        ucR[BGXL + t2] = uHLLE[BGXL + t2];

        ucL[YESL] = uHLLE[YESL];
        ucR[YESL] = uHLLE[YESL];
        ucL[ENTSL] = uHLLE[ENTSL];
        ucR[ENTSL] = uHLLE[ENTSL];

        // Density (same formula for both cases)
        ucL[DENSL] = uL[DENSL] * (-cmin - pL[VXL+idir]) / (-cmin - lambdaC);
        ucR[DENSL] = uR[DENSL] * ( cmax - pR[VXL+idir]) / ( cmax - lambdaC);

        // Calculate p_star and other variables based on magnetic field presence
        double p_star, v_star_B_star = 0.0;
        double vt1 = 0.0, vt2 = 0.0, gamma_star = 1.0;

        if (has_magnetic_field) {
            vt1 = (uHLLE[BGXL + t1]*lambdaC - fHLLE[BGXL + t1]) / uHLLE[BGXL+idir];
            vt2 = (uHLLE[BGXL + t2]*lambdaC - fHLLE[BGXL + t2]) / uHLLE[BGXL+idir];

            double const v2 = lambdaC*lambdaC + vt1*vt1 + vt2*vt2;
            gamma_star = 1.0 / Kokkos::sqrt(1 - v2);
            v_star_B_star = lambdaC * uHLLE[BGXL+idir] + vt1 * uHLLE[BGXL + t1] + vt2 * uHLLE[BGXL + t2];

            p_star = fHLLE[STXL + idir] + Kokkos::pow(uHLLE[BGXL+idir]/gamma_star, 2)
                     -lambdaC * (fHLLE[TAUL]+fHLLE[DENSL] - uHLLE[BGXL+idir]*v_star_B_star);
        } else {
            p_star = -lambdaC * (fHLLE[TAUL]+fHLLE[DENSL]) + fHLLE[STXL+idir];
        }

        // Energy components
        ucL[TAUL] = (-cmin * uL[TAUL] - fL[TAUL] + p_star * lambdaC - v_star_B_star*uHLLE[BGXL+idir]) / (-cmin - lambdaC);
        ucR[TAUL] = ( cmax * uR[TAUL] + fR[TAUL] + p_star * lambdaC - v_star_B_star*uHLLE[BGXL+idir]) / ( cmax - lambdaC);

        // Momentum in normal direction
        ucL[STXL+idir] = (ucL[TAUL] + ucL[DENSL] + p_star) * lambdaC - v_star_B_star*uHLLE[BGXL+idir];
        ucR[STXL+idir] = (ucR[TAUL] + ucR[DENSL] + p_star) * lambdaC - v_star_B_star*uHLLE[BGXL+idir];

        // Transverse momentum components
        if (has_magnetic_field) {
            ucL[STXL+t1] = (-uHLLE[BGXL + idir] * (uHLLE[BGXL + t1]/gamma_star/gamma_star + v_star_B_star * vt1)
                           - fL[STXL + t1] + (-cmin) * uL[STXL+t1]) / (-cmin - lambdaC);
            ucR[STXL+t1] = (-uHLLE[BGXL + idir] * (uHLLE[BGXL + t1]/gamma_star/gamma_star + v_star_B_star * vt1)
                           - fR[STXL + t1] + ( cmax) * uR[STXL+t1]) / ( cmax - lambdaC);

            ucL[STXL+t2] = (-uHLLE[BGXL + idir] * (uHLLE[BGXL + t2]/gamma_star/gamma_star + v_star_B_star * vt2)
                           - fL[STXL + t2] + (-cmin) * uL[STXL+t2]) / (-cmin - lambdaC);
            ucR[STXL+t2] = (-uHLLE[BGXL + idir] * (uHLLE[BGXL + t2]/gamma_star/gamma_star + v_star_B_star * vt2)
                           - fR[STXL + t2] + ( cmax) * uR[STXL+t2]) / ( cmax - lambdaC);
        } else {
            ucL[STXL+t1] = uL[STXL+t1]*(-cmin - pL[VXL+idir])/(-cmin - lambdaC);
            ucR[STXL+t1] = uR[STXL+t1]*( cmax - pR[VXL+idir])/( cmax - lambdaC);

            ucL[STXL+t2] = uL[STXL+t2]*(-cmin - pL[VXL+idir])/(-cmin - lambdaC);
            ucR[STXL+t2] = uR[STXL+t2]*( cmax - pR[VXL+idir])/( cmax - lambdaC);
        }

        // Handle supersonic case
        if (lambdaC > 1.0) {
            ucL = uHLLE;
            ucR = uHLLE;
        }
        /***********************************************************************/

        if ( -cmin >= vi ) {
            fHLLC = fL ;
            uHLLC = uL ;  
        } else if ( -cmin < vi and vi < lambdaC ) {
            for( int iv=0; iv<uL.size(); ++iv) {
                fHLLC[iv] = fL[iv] - cmin * ( ucL[iv] - uL[iv] ) ; 
            }
            uHLLC = ucL ;
        } else if ( lambdaC <= vi and vi < cmax  ) { 
            for( int iv=0; iv<uL.size(); ++iv) {
                fHLLC[iv] = fR[iv] + cmax * ( ucR[iv] - uR[iv] ) ; 
            }
            uHLLC = ucR ;  
        } else {
            fHLLC = fR ; 
            uHLLC = uR ;  
        }
        #ifdef GRACE_ENABLE_B_FIELD_GLM
            int metric_comps[3] { 0, 3, 5} ; 
	        // the characteristic wavespeeds in GRMHD in coordinate frame are bound by
	        // Eq.(60) in Anton 2006: https://arxiv.org/pdf/astro-ph/0506063
            // cml_DC is a short name for "c_minus_left_divergence_cleaning"

            double cmin_DC = -1 ; 
            double cmax_DC =  1 ; 

            const double fl = pL[BXL+idir] ; 
            const double fr = pR[BXL+idir] ; 

            const double phi_glm_l = pL[PHI_GLML];
            const double phi_glm_r = pR[PHI_GLML];

            fHLLC[PHIG_GLML] = hlle_solver(fl,fr,phi_glm_l,phi_glm_r,cmin_DC,cmax_DC) ; 
        #endif // GRACE_ENABLE_B_FIELD_GLM

        //transform_fluxes_to_eulerian_frame(uHLLC,fHLLC) ; !TODO
        return fHLLC ; 
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    hllc_cons_with_magnetic_field( 
          grace::grmhd_cons_array_t const& fHLLE
        , grace::grmhd_cons_array_t const& uHLLE
        , grace::grmhd_cons_array_t const& fL 
        , grace::grmhd_cons_array_t const& fR 
        , grace::grmhd_cons_array_t const& uL
        , grace::grmhd_cons_array_t const& uR
        , grace::grmhd_cons_array_t& ucL
        , grace::grmhd_cons_array_t& ucR
        , grace::grmhd_prims_array_t const& pL 
        , grace::grmhd_prims_array_t const& pR
        , double const cmin
        , double const cmax
        , double const lambdaC ) 
    {
        using Kokkos::sqrt ; 
        int const bg_normal_idx = BGXL + idir;
        constexpr int t1 = (idir + 1) % 3;
        constexpr int t2 = (idir + 2) % 3;

        double const vt1 = (uHLLE[BGXL + t1]*lambdaC - fHLLE[BGXL + t1]) / uHLLE[BGXL+idir] ;
        double const vt2 = (uHLLE[BGXL + t2]*lambdaC - fHLLE[BGXL + t2]) / uHLLE[BGXL+idir] ;

        double const v2 = lambdaC*lambdaC + vt1*vt1 + vt2*vt2 ; 
        double const gamma_star = 1.0 / Kokkos::sqrt(1 - v2) ;
        double const v_star_B_star = lambdaC * uHLLE[BGXL+idir] + vt1 * uHLLE[BGXL + t1] + vt2 * uHLLE[BGXL + t2] ;

        double const p_star = fHLLE[BGXL + idir] + Kokkos::pow(uHLLE[BGXL+idir]/gamma_star, 2) \
                           - (fHLLE[TAUL]+fHLLE[DENSL] - uHLLE[BGXL+idir]*(v_star_B_star))*lambdaC;
                    
        ucL[BGXL+idir] = uHLLE[BGXL+idir] ;
        ucR[BGXL+idir] = uHLLE[BGXL+idir] ;

        ucL[BGXL + t1] = uHLLE[BGXL + t1] ;
        ucR[BGXL + t1] = uHLLE[BGXL + t1] ;

        ucL[BGXL + t2] = uHLLE[BGXL + t2] ;
        ucR[BGXL + t2] = uHLLE[BGXL + t2] ;

        ucL[DENSL] = uL[DENSL] * ( -cmin - pL[VXL+idir]) / ( -cmin - lambdaC ) ;
        ucR[DENSL] = uR[DENSL] * (  cmax - pR[VXL+idir]) / (  cmax - lambdaC ) ;

        ucL[TAUL] = (-cmin * uL[TAUL] - fL[TAUL] + p_star * lambdaC - v_star_B_star*uHLLE[BGXL+idir]) / ( -cmin - lambdaC ) ;
        ucR[TAUL] = ( cmax * uR[TAUL] + fR[TAUL] + p_star * lambdaC - v_star_B_star*uHLLE[BGXL+idir]) / (  cmax - lambdaC ) ;

        ucL[STXL+idir] = (ucL[TAUL] + ucL[DENSL] + p_star) * lambdaC - v_star_B_star*uHLLE[BGXL+idir] ;
        ucR[STXL+idir] = (ucR[TAUL] + ucR[DENSL] + p_star) * lambdaC - v_star_B_star*uHLLE[BGXL+idir] ;
        
        ucL[STXL+t1] = (-uHLLE[BGXL + idir] * (uHLLE[BGXL + t1]/Kokkos::pow(gamma_star, 2) + v_star_B_star * vt1) \
                      - fL[STXL + t1] + (-cmin) * uL[STXL+t1]) / ( -cmin - lambdaC ) ;
        ucR[STXL+t1] = (-uHLLE[BGXL + idir] * (uHLLE[BGXL + t1]/Kokkos::pow(gamma_star, 2) + v_star_B_star * vt1) \
                      - fR[STXL + t1] + ( cmax) * uR[STXL+t1]) / (  cmax - lambdaC ) ;

        ucL[STXL+t2] = (-uHLLE[BGXL + idir] * (uHLLE[BGXL + t2]/Kokkos::pow(gamma_star, 2) + v_star_B_star * vt2) \
                      - fL[STXL + t2] + (-cmin) * uL[STXL+t2]) / ( -cmin - lambdaC ) ;
        ucR[STXL+t2] = (-uHLLE[BGXL + idir] * (uHLLE[BGXL + t2]/Kokkos::pow(gamma_star, 2) + v_star_B_star * vt2) \
                      - fR[STXL + t2] + ( cmax) * uR[STXL+t2]) / (  cmax - lambdaC ) ;
        
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    hllc_cons_without_magnetic_field( 
          grace::grmhd_cons_array_t const& fHLLE
        , grace::grmhd_cons_array_t const& uHLLE
        , grace::grmhd_cons_array_t const& fL 
        , grace::grmhd_cons_array_t const& fR 
        , grace::grmhd_cons_array_t const& uL
        , grace::grmhd_cons_array_t const& uR
        , grace::grmhd_cons_array_t& ucL
        , grace::grmhd_cons_array_t& ucR
        , grace::grmhd_prims_array_t const& pL 
        , grace::grmhd_prims_array_t const& pR
        , double const cmin
        , double const cmax
        , double const lambdaC ) 
    {
        using Kokkos::sqrt ; 
        int const bg_normal_idx = BGXL + idir;
        constexpr int t1 = (idir + 1) % 3;
        constexpr int t2 = (idir + 2) % 3;
        
        double const p_star  = -lambdaC * (fHLLE[TAUL]+fHLLE[DENSL]) + fHLLE[STXL+idir] ;

        ucL[BGXL+idir] = uHLLE[BGXL+idir] ;
        ucR[BGXL+idir] = uHLLE[BGXL+idir] ;

        ucL[BGXL + t1] = uHLLE[BGXL + t1] ;
        ucR[BGXL + t1] = uHLLE[BGXL + t1] ;

        ucL[BGXL + t2] = uHLLE[BGXL + t2] ;
        ucR[BGXL + t2] = uHLLE[BGXL + t2] ;

        ucL[DENSL] = uL[DENSL] * ( -cmin - pL[VXL+idir] ) / ( -cmin - lambdaC ) ; 
        ucR[DENSL] = uR[DENSL] * (  cmax - pR[VXL+idir] ) / (  cmax - lambdaC ) ; 

        ucL[TAUL] = (-cmin * uL[TAUL] - fL[TAUL] + p_star * lambdaC) / ( -cmin - lambdaC ) ;
        ucR[TAUL] = ( cmax * uR[TAUL] + fR[TAUL] + p_star * lambdaC) / (  cmax - lambdaC ) ;

        ucL[STXL+idir] = (ucL[TAUL] + ucL[DENSL] + p_star) * lambdaC ;
        ucR[STXL+idir] = (ucR[TAUL] + ucR[DENSL] + p_star) * lambdaC ;
        
        ucL[STXL+t1] = uL[STXL+t1]*(-cmin - pL[VXL+idir])/(-cmin - lambdaC) ;
        ucR[STXL+t1] = uR[STXL+t1]*( cmax - pR[VXL+idir])/( cmax - lambdaC) ;

        ucL[STXL+t2] = uL[STXL+t2]*(-cmin - pL[VXL+idir])/(-cmin - lambdaC) ;
        ucR[STXL+t2] = uR[STXL+t2]*( cmax - pR[VXL+idir])/( cmax - lambdaC) ;

    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_interface_velocity() {
        int midx [] = { 0,3,5 } ; 
        static const double sqrt_gamma = Kokkos::sqrt(metric.gamma(midx[idir]));
        return metric.beta(idir) / sqrt_gamma / metric.alp();
    }
    /**
     * @brief Get the contact wave speed in the local frame.
     * 
     * @param cons Conserved variables in HLLE state. 
     * @param f    Fluxes in HLLE state.
     * @return double The contact wave speed.
     * 
     * This function solves the second order equation (3.70)
     * in Kiuchi+. Note that we always keep the solution with 
     * the minus sign as that's the one that is causal (see 
     * Mignone+ for a proof).
     */
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_contact_wave_speed(grace::grmhd_cons_array_t const& cons,
                          grace::grmhd_cons_array_t const& f) const
    {
        using Kokkos::sqrt;
        using Kokkos::max;

        constexpr int t1 = (idir + 1) % 3;
        constexpr int t2 = (idir + 2) % 3;

        // Magnetic field transverse components
        double const cons_bg_t1 = cons[BGXL + t1];
        double const cons_bg_t2 = cons[BGXL + t2];
        double const f_bg_t1    = f[BGXL + t1];
        double const f_bg_t2    = f[BGXL + t2];

        // Basic sums
        double const taul_densl_sum     = cons[TAUL] + cons[DENSL];
        double const f_taul_densl_sum   = f[TAUL] + f[DENSL];
        double const cross_term         = cons_bg_t1 * f_bg_t1 + cons_bg_t2 * f_bg_t2;

        // Magnetic field check
        double const bg_face = cons[BGXL + idir];
        double const has_magnetic_field = static_cast<double>(bg_face >= 1e-15); // 1.0 or 0.0

        // Transverse B^2
        double const cons_bt2 = cons_bg_t1 * cons_bg_t1 + cons_bg_t2 * cons_bg_t2;
        double const f_bt2    = f_bg_t1 * f_bg_t1 + f_bg_t2 * f_bg_t2;

        // Coefficients a, b, c with masking instead of conditionals
        double const a = f_taul_densl_sum - has_magnetic_field * cross_term;

        double const b = 
            -taul_densl_sum - f[STXL + idir]
            + has_magnetic_field * (cons_bt2 + f_bt2);

        double const c = cons[STXL + idir] - has_magnetic_field * cross_term;

        // Quadratic solution
        double const discriminant = max(0.0, b * b - 4.0 * a * c);

        // Avoid division by near-zero `a`
        double const eps = 1e-14;
        double const a_safe = (Kokkos::abs(a) > eps) ? a : ((a >= 0.0) ? eps : -eps);

        return -0.5 * (b + sqrt(discriminant)) / a_safe;
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    transform_fluxes_to_eulerian_frame( grace::grmhd_cons_array_t const& cons 
                                      , grace::grmhd_cons_array_t& f) 
    {
        int scalar_vars_indices[] = {DENSL,TAUL,YESL,ENTSL} ; 
        for (int ivar=0; ivar<4; ++ivar) {
            f[ivar] = (
                  inertial_tetrad[0][idir] * cons[ivar]
                + inertial_tetrad[idir][idir] * f[ivar] 
            ) ; 
        }
        // TODO ceck missing Minkowski metric.
        std::array<double,3> stilde = {cons[STXL], cons[STYL], cons[STZL]} ; 
        std::array<double,3> fstilde = {f[STXL], f[STYL], f[STZL]} ; 
        for( int ivdir=0; ivdir<3; ++ivdir) { 
            double const eS = metric.contract_vec_covec(
                  stilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ; 
            /* Contracting two lower indices, does not matter   */
            /* since it's only the spatial part of the locally  */
            /* flat tetrad.                                     */
            double const eF = metric.contract_vec_covec(
                  fstilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ;
            f[STXL+ivdir] = (
                  inertial_tetrad[0][idir] * eS 
                + inertial_tetrad[idir][idir] * eF 
            ) ; 
        }
    }



 private:
    //! Local metric
    grace::metric_array_t metric ; 
    /**
     * @brief Tetrad and cotetrad in which fluxes are evaluated.
     * NB: These matrices contain \f$M^\mu_{\hat{i}}\f$ and \f$M^{\hat{j}}_\mu\f$
     * i.e. their contraction is \f$\delta^{\hat{i}}_{\hat{j}}\f$.
     * For spatial vectors / tensors this makes no difference but when transforming 
     * back to GR frame \f$\eta_{\hat{i}\hat{j}}\f$ is needed to lower/raise 
     * tetrad indices.
     */
    tetrad_t inertial_tetrad, inertial_cotetrad ;

     
    void GRACE_HOST_DEVICE 
    get_tetrad_basis( grace::metric_array_t const& metric
                    , hllc_riemann_solver_t::tetrad_t& tetrad 
                    , hllc_riemann_solver_t::tetrad_t& cotetrad) {
        double beta[] = {metric.beta(0),metric.beta(1),metric.beta(2)} ; 
        double g[] = { metric.gamma(0),metric.gamma(1),metric.gamma(2)
                    , metric.gamma(3),metric.gamma(4),metric.gamma(5)} ;
        double gD[] = { metric.invgamma(0),metric.invgamma(1),metric.invgamma(2)
                    , metric.invgamma(3),metric.invgamma(4),metric.invgamma(5)} ;
        get_tetrad_basis_impl(
            metric.alp(),
            beta,g,gD,tetrad,cotetrad,idir
        ) ;
    };


    void GRACE_HOST_DEVICE
    get_tetrad_basis_impl( double const alp
                        , double const beta[3]
                        , double const g[6]
                        , double const gD[6]
                        , hllc_riemann_solver_t::tetrad_t& tetrad 
                        , hllc_riemann_solver_t::tetrad_t& cotetrad 
                        , int shift )
    {
        using Kokkos::sqrt ; 

        int midx[3][6] = {
            {0,1,2,3,4,5}
        , {3,4,1,5,2,0}
        , {5,2,4,0,1,3}
        } ; 

        int tidx[3][4] = {
            {0,1,2,3}
        , {0,2,3,1}
        , {0,3,1,2}
        } ; 

        int vidx[3][3] = {
            {0,1,2}
        , {1,2,0}
        , {2,0,1}
        } ; 

        double const one_over_alp = 1./alp ;
        double const Bhat = 1./sqrt(g[midx[shift][0]]) ;
        double const Chat = 1./sqrt(gD[midx[shift][5]]) ; 
        double const Dhat = 1./sqrt(gD[midx[shift][5]]*(gD[midx[shift][3]]*gD[midx[shift][5]] - math::int_pow<2>(gD[midx[shift][4]])));
        double const Ehat = -(beta[vidx[shift][1]]*g[midx[shift][0]]) + beta[vidx[shift][0]]*g[midx[shift][1]];
        double const Fhat = g[midx[shift][1]];
        double const Ghat = -(beta[vidx[shift][2]]*g[midx[shift][0]]) + beta[vidx[shift][0]]*g[midx[shift][2]];
        double const Hhat = g[midx[shift][2]];


        tetrad[0][tidx[shift][0]] = one_over_alp ; 
        tetrad[0][tidx[shift][1]] = -(beta[vidx[shift][0]]*one_over_alp) ; 
        tetrad[0][tidx[shift][2]] = -(beta[vidx[shift][1]]*one_over_alp) ; 
        tetrad[0][tidx[shift][3]] = -(beta[vidx[shift][2]]*one_over_alp) ; 


        tetrad[1][tidx[shift][0]] = 0                      ; 
        tetrad[1][tidx[shift][1]] = Bhat*g[midx[shift][0]] ; 
        tetrad[1][tidx[shift][2]] = Bhat*g[midx[shift][1]] ; 
        tetrad[1][tidx[shift][3]] = Bhat*g[midx[shift][2]] ;

        tetrad[2][tidx[shift][0]] = 0                         ; 
        tetrad[2][tidx[shift][1]] = 0                         ; 
        tetrad[2][tidx[shift][2]] = Dhat*gD[midx[shift][5]]   ; 
        tetrad[2][tidx[shift][3]] = -(Dhat*gD[midx[shift][4]]);


        tetrad[3][tidx[shift][0]] = 0    ; 
        tetrad[3][tidx[shift][1]] = 0    ; 
        tetrad[3][tidx[shift][2]] = 0    ; 
        tetrad[3][tidx[shift][3]] = Chat ;  


        cotetrad[0] = {alp, 0,0,0} ; 

        cotetrad[1][tidx[shift][0]] = beta[vidx[shift][0]]*Bhat; 
        cotetrad[1][tidx[shift][1]] = Bhat                     ; 
        cotetrad[1][tidx[shift][2]] = 0.                       ; 
        cotetrad[1][tidx[shift][3]] = 0                        ;


        cotetrad[2][tidx[shift][0]] = -(Ehat*pow(Bhat,2)/(Dhat*gD[midx[shift][5]])); 
        cotetrad[2][tidx[shift][1]] = -(Fhat*pow(Bhat,2)/(Dhat*gD[midx[shift][5]])); 
        cotetrad[2][tidx[shift][2]] = 1./(Dhat*gD[midx[shift][5]])                 ; 
        cotetrad[2][tidx[shift][3]] = 0                                            ; 
        
        cotetrad[3][tidx[shift][0]] =  -(Ehat*gD[midx[shift][4]] + gD[midx[shift][5]]*Ghat)*pow(Bhat,2)/(Chat*gD[midx[shift][5]]) ; 
        cotetrad[3][tidx[shift][1]] = -((Fhat*gD[midx[shift][4]] + gD[midx[shift][5]]*Hhat)*pow(Bhat,2)/(Chat*gD[midx[shift][5]]));
        cotetrad[3][tidx[shift][2]] = gD[midx[shift][4]]*1./(Chat*gD[midx[shift][5]])                                             ;
        cotetrad[3][tidx[shift][3]] = 1./Chat                                                                                     ; 

    }
} ; 


/**
 * @brief HLLD riemann solver in direction idir.
 * 
 * @tparam idir The direction of the riemann problem being solved. 
 */
template< int idir > 
struct hlld_riemann_solver_t {

    using tetrad_t = std::array<std::array<double,4>,4> ;
    
 public:
     

    GRACE_HOST_DEVICE
    hlld_riemann_solver_t( grace::metric_array_t const& _metric )
     : metric(_metric)
    {
        get_tetrad_basis(metric,inertial_tetrad,inertial_cotetrad) ; 
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    transform_velocities_to_tetrad_frame( double& u0
                                        , grace::grmhd_prims_array_t& prims
                                        , std::array<double,3>& uD ) const 
    {
        std::array<double,4> umu { 
              u0
            , prims[VXL]*u0 
            , prims[VYL]*u0
            , prims[VZL]*u0
        } ; 

        //u0 = metric.contract_4dvec_4dcovec(inertial_cotetrad[0],umu) ; !TODO
        for(int ii=0; ii<3;++ii) {
            uD[ii] = umu[ii+1] ; // metric.contract_4dvec_4dcovec(inertial_cotetrad[1+ii],umu) ; !TODO
            // In the tetrad frame lower and upper spatial indices are the same
            prims[VXL+ii]  = uD[ii] / u0 ; 
        }
    }
    #ifdef GRACE_DO_MHD
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    transform_magnetic_fields_to_tetrad_frame(grace::grmhd_prims_array_t& prims) const 
    {
        std::array<double,3> B { 
              prims[BXL] 
            , prims[BYL]
            , prims[BZL]
        } ; 

        //u0 = metric.contract_4dvec_4dcovec(inertial_cotetrad[0],umu) ; !TODO
        for(int ii=0; ii<3;++ii) {
            for (int jj=0; jj<3; ++jj) {
                prims[BXL+ii]  =  inertial_cotetrad[ii][jj]*B[jj]; 
            }
        }
    }
    #endif // GRACE_DO_MHD
    /**
     * @brief Compute HLLD fluxes for relativistic Hydro.
     * 
     * @param fL Left fluxes.
     * @param fR Right fluxes.
     * @param uL Left conserved state.
     * @param uR Right conserved state.
     * @param pL Left primitive state (in local frame).
     * @param pR Right primitive state (in local frame).
     * @param cmin lambdaL (this has \b positive sign).
     * @param cmax lambdaR (this has \b positive sign).
     * @return grace::grmhd_cons_array_t The HLLD flux.
     * 
     * Note that in GRACE notation the velocity is not the eulerian one but rather
     * \f$v^i=u^i/u^t\f$. Also, our conserved energy density \f$\tau\f$ follows the 
     * Valencia notation and is defined as \f$n^\mu n^\nu T_{\mu\nu} - D\f$. For this
     * reason some of the terms in this function are different from what appears in 
     * https://arxiv.org/abs/2205.04487 which is the main source followed in this 
     * implementation.
     */
    grace::grmhd_cons_array_t 
    GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    operator() ( 
          grace::grmhd_cons_array_t const& fL 
        , grace::grmhd_cons_array_t const& fR 
        , grace::grmhd_cons_array_t const& uL 
        , grace::grmhd_cons_array_t const& uR 
        , grace::grmhd_prims_array_t const& pL 
        , grace::grmhd_prims_array_t const& pR
        , double const cmin 
        , double const cmax )
    {
        grace::grmhd_cons_array_t uHLLE, fHLLE, uHLLD, fHLLD; 
        hll_riemann_solver_t hlle_solver ; 
        int var_indices[] = {DENSL, TAUL, STXL, STYL, STZL, BXL, BYL, BZL, YESL, ENTSL} ; 
       
        return fHLLD ; 
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    transform_fluxes_to_eulerian_frame( grace::grmhd_cons_array_t const& cons 
                                      , grace::grmhd_cons_array_t& f) 
    {
        int scalar_vars_indices[] = {DENSL,TAUL,YESL,ENTSL} ; 
        for (int ivar=0; ivar<4; ++ivar) {
            f[ivar] = (
                  inertial_tetrad[0][idir] * cons[ivar]
                + inertial_tetrad[idir][idir] * f[ivar] 
            ) ; 
        }
        // TODO ceck missing Minkowski metric.
        std::array<double,3> stilde = {cons[STXL], cons[STYL], cons[STZL]} ; 
        std::array<double,3> fstilde = {f[STXL], f[STYL], f[STZL]} ; 
        for( int ivdir=0; ivdir<3; ++ivdir) { 
            double const eS = metric.contract_vec_covec(
                  stilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ; 
            /* Contracting two lower indices, does not matter   */
            /* since it's only the spatial part of the locally  */
            /* flat tetrad.                                     */
            double const eF = metric.contract_vec_covec(
                  fstilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ;
            f[STXL+ivdir] = (
                  inertial_tetrad[0][idir] * eS 
                + inertial_tetrad[idir][idir] * eF 
            ) ; 
        }
    }



 private:
    //! Local metric
    grace::metric_array_t metric ; 
    /**
     * @brief Tetrad and cotetrad in which fluxes are evaluated.
     * NB: These matrices contain \f$M^\mu_{\hat{i}}\f$ and \f$M^{\hat{j}}_\mu\f$
     * i.e. their contraction is \f$\delta^{\hat{i}}_{\hat{j}}\f$.
     * For spatial vectors / tensors this makes no difference but when transforming 
     * back to GR frame \f$\eta_{\hat{i}\hat{j}}\f$ is needed to lower/raise 
     * tetrad indices.
     */
    tetrad_t inertial_tetrad, inertial_cotetrad ;

     
    void GRACE_HOST_DEVICE 
    get_tetrad_basis( grace::metric_array_t const& metric
                    , hlld_riemann_solver_t::tetrad_t& tetrad 
                    , hlld_riemann_solver_t::tetrad_t& cotetrad) {
        double beta[] = {metric.beta(0),metric.beta(1),metric.beta(2)} ; 
        double g[] = { metric.gamma(0),metric.gamma(1),metric.gamma(2)
                    , metric.gamma(3),metric.gamma(4),metric.gamma(5)} ;
        double gD[] = { metric.invgamma(0),metric.invgamma(1),metric.invgamma(2)
                    , metric.invgamma(3),metric.invgamma(4),metric.invgamma(5)} ;
        get_tetrad_basis_impl(
            metric.alp(),
            beta,g,gD,tetrad,cotetrad,idir
        ) ;
    };


    void GRACE_HOST_DEVICE
    get_tetrad_basis_impl( double const alp
                        , double const beta[3]
                        , double const g[6]
                        , double const gD[6]
                        , hlld_riemann_solver_t::tetrad_t& tetrad 
                        , hlld_riemann_solver_t::tetrad_t& cotetrad 
                        , int shift )
    {
        using Kokkos::sqrt ; 

        int midx[3][6] = {
            {0,1,2,3,4,5}
        , {3,4,1,5,2,0}
        , {5,2,4,0,1,3}
        } ; 

        int tidx[3][4] = {
            {0,1,2,3}
        , {0,2,3,1}
        , {0,3,1,2}
        } ; 

        int vidx[3][3] = {
            {0,1,2}
        , {1,2,0}
        , {2,0,1}
        } ; 

        double const one_over_alp = 1./alp ;
        double const Bhat = 1./sqrt(g[midx[shift][0]]) ;
        double const Chat = 1./sqrt(gD[midx[shift][5]]) ; 
        double const Dhat = 1./sqrt(gD[midx[shift][5]]*(gD[midx[shift][3]]*gD[midx[shift][5]] - math::int_pow<2>(gD[midx[shift][4]])));
        double const Ehat = -(beta[vidx[shift][1]]*g[midx[shift][0]]) + beta[vidx[shift][0]]*g[midx[shift][1]];
        double const Fhat = g[midx[shift][1]];
        double const Ghat = -(beta[vidx[shift][2]]*g[midx[shift][0]]) + beta[vidx[shift][0]]*g[midx[shift][2]];
        double const Hhat = g[midx[shift][2]];


        tetrad[0][tidx[shift][0]] = one_over_alp ; 
        tetrad[0][tidx[shift][1]] = -(beta[vidx[shift][0]]*one_over_alp) ; 
        tetrad[0][tidx[shift][2]] = -(beta[vidx[shift][1]]*one_over_alp) ; 
        tetrad[0][tidx[shift][3]] = -(beta[vidx[shift][2]]*one_over_alp) ; 


        tetrad[1][tidx[shift][0]] = 0                      ; 
        tetrad[1][tidx[shift][1]] = Bhat*g[midx[shift][0]] ; 
        tetrad[1][tidx[shift][2]] = Bhat*g[midx[shift][1]] ; 
        tetrad[1][tidx[shift][3]] = Bhat*g[midx[shift][2]] ;

        tetrad[2][tidx[shift][0]] = 0                         ; 
        tetrad[2][tidx[shift][1]] = 0                         ; 
        tetrad[2][tidx[shift][2]] = Dhat*gD[midx[shift][5]]   ; 
        tetrad[2][tidx[shift][3]] = -(Dhat*gD[midx[shift][4]]);


        tetrad[3][tidx[shift][0]] = 0    ; 
        tetrad[3][tidx[shift][1]] = 0    ; 
        tetrad[3][tidx[shift][2]] = 0    ; 
        tetrad[3][tidx[shift][3]] = Chat ;  


        cotetrad[0] = {alp, 0,0,0} ; 

        cotetrad[1][tidx[shift][0]] = beta[vidx[shift][0]]*Bhat; 
        cotetrad[1][tidx[shift][1]] = Bhat                     ; 
        cotetrad[1][tidx[shift][2]] = 0.                       ; 
        cotetrad[1][tidx[shift][3]] = 0                        ;


        cotetrad[2][tidx[shift][0]] = -(Ehat*pow(Bhat,2)/(Dhat*gD[midx[shift][5]])); 
        cotetrad[2][tidx[shift][1]] = -(Fhat*pow(Bhat,2)/(Dhat*gD[midx[shift][5]])); 
        cotetrad[2][tidx[shift][2]] = 1./(Dhat*gD[midx[shift][5]])                 ; 
        cotetrad[2][tidx[shift][3]] = 0                                            ; 
        
        cotetrad[3][tidx[shift][0]] =  -(Ehat*gD[midx[shift][4]] + gD[midx[shift][5]]*Ghat)*pow(Bhat,2)/(Chat*gD[midx[shift][5]]) ; 
        cotetrad[3][tidx[shift][1]] = -((Fhat*gD[midx[shift][4]] + gD[midx[shift][5]]*Hhat)*pow(Bhat,2)/(Chat*gD[midx[shift][5]]));
        cotetrad[3][tidx[shift][2]] = gD[midx[shift][4]]*1./(Chat*gD[midx[shift][5]])                                             ;
        cotetrad[3][tidx[shift][3]] = 1./Chat                                                                                     ; 

    }
} ; 



}

#endif 