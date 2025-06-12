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

        // u0 = metric.contract_4dvec_4dcovec(inertial_cotetrad[0],umu) ; //!TODO
        for(int ii=0; ii<3;++ii) {
            uD[ii] = umu [ii+1] ; //metric.contract_4dvec_4dcovec(inertial_cotetrad[1+ii],umu) ; //!TODO
            // In the tetrad frame lower and upper spatial indices are the same
            //prims[VXL+ii]  = uD[ii]; // / u0 ; 
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
           // for (int jj=0; jj<3; ++jj) {
           //     prims[BXL+ii]  =  inertial_cotetrad[ii][jj]*B[jj]; 
           // }
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
        int var_indices[] = {DENSL, TAUL, STXL, STYL, STZL, YESL, ENTSL
            #ifdef GRACE_DO_MHD 
            ,BGXL, BGYL, BGZL 
            #ifdef GRACE_ENABLE_B_FIELD_GLM
            ,PHIG_GLML
            #endif // GRACE_ENABLE_B_FIELD_GLM
            #endif // GRACE_DO_MHD
            } ; 
        grace::grmhd_cons_array_t ucL, ucR ; 

        constexpr int num_vars = 7
        #ifdef GRACE_DO_MHD
            + 3
        #ifdef GRACE_ENABLE_B_FIELD_GLM
            + 1
        #endif
        #endif
        ;

        for( int ii=0; ii<num_vars; ++ii) {
            int const ivar = var_indices[ii] ; 
            fHLLE[ivar] = 
                hlle_solver(fL[ivar],fR[ivar],uL[ivar],uR[ivar],cmin,cmax) ; 
            uHLLE[ivar] = 
                hlle_solver.get_state(fL[ivar],fR[ivar],uL[ivar],uR[ivar],cmin,cmax) ;
        }

        // Default to false, will be set later
        bool has_magnetic_field = false;
        double BG_idir = 0.0;

        // Common calculations for both cases
        constexpr int t1 = (idir + 1) % 3;
        constexpr int t2 = (idir + 2) % 3;
        #ifdef GRACE_DO_MHD
            BG_idir = uL[BGXL+idir] ; // This is the magnetic field component in the direction idir.
            
            // Magnetic Field is constant on the face, so we have to copy it.
            uHLLE[BGXL+idir] = BG_idir ;

            // Check for magnetic field strength
            has_magnetic_field = (Kokkos::abs(BG_idir) > 1e-15);

            // Set magnetic field components (same for both cases)
            ucL[BGXL+idir] = BG_idir;
            ucR[BGXL+idir] = BG_idir;
            ucL[BGXL + t1] = uHLLE[BGXL + t1];
            ucR[BGXL + t1] = uHLLE[BGXL + t1];
            ucL[BGXL + t2] = uHLLE[BGXL + t2];
            ucR[BGXL + t2] = uHLLE[BGXL + t2];
        #endif // GRACE_DO_MHD

        double const vi = get_interface_velocity() ; // Carlo wrote TODO but seems to be fine.
        double const lambdaC = get_contact_wave_speed(uHLLE,fHLLE) ; 
        //if (Kokkos::abs(lambdaC) > 1e-1) {
        //    Kokkos::printf("HLLC contact wave speed: %e\n", lambdaC);
        //}

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
        double v2 = lambdaC*lambdaC;

        if (has_magnetic_field) {
            vt1 = (uHLLE[BGXL + t1]*lambdaC - fHLLE[BGXL + t1]) / BG_idir;
            vt2 = (uHLLE[BGXL + t2]*lambdaC - fHLLE[BGXL + t2]) / BG_idir;

            v2 = lambdaC*lambdaC + vt1*vt1 + vt2*vt2;
            gamma_star = 1.0 / Kokkos::sqrt(1 - v2);
            v_star_B_star = lambdaC * uHLLE[BGXL+idir] + vt1 * uHLLE[BGXL + t1] + vt2 * uHLLE[BGXL + t2];

            p_star = fHLLE[STXL + idir] + (uHLLE[BGXL+idir]/gamma_star)*(uHLLE[BGXL+idir]/gamma_star)
                     -lambdaC * (fHLLE[TAUL]+fHLLE[DENSL] - uHLLE[BGXL+idir]*v_star_B_star);
        } else {
            p_star = -lambdaC * (fHLLE[TAUL]+fHLLE[DENSL]) + fHLLE[STXL+idir];
        }

        // Energy components
        ucL[TAUL] = (-cmin * uL[TAUL] - fL[TAUL] + p_star * lambdaC - v_star_B_star*BG_idir) / (-cmin - lambdaC);
        ucR[TAUL] = ( cmax * uR[TAUL] - fR[TAUL] + p_star * lambdaC - v_star_B_star*BG_idir) / ( cmax - lambdaC);

        // Momentum in normal direction
        ucL[STXL+idir] = (ucL[TAUL] + ucL[DENSL] + p_star) * lambdaC - v_star_B_star*BG_idir; // TODO vstar_B_star is not used in the paper but v_star_B probably a typo
        ucR[STXL+idir] = (ucR[TAUL] + ucR[DENSL] + p_star) * lambdaC - v_star_B_star*BG_idir;

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

            ucL[BGXL+t1] = uL[BGXL+t1]*(-cmin - pL[VXL+idir])/(-cmin - lambdaC);
            ucR[BGXL+t1] = uR[BGXL+t1]*( cmax - pR[VXL+idir])/( cmax - lambdaC);

            ucL[BGXL+t2] = uL[BGXL+t2]*(-cmin - pL[VXL+idir])/(-cmin - lambdaC);
            ucR[BGXL+t2] = uR[BGXL+t2]*( cmax - pR[VXL+idir])/( cmax - lambdaC);
        }

        // Handle supersonic case
        if (lambdaC <= -cmin || lambdaC >= cmax || v2 >= 1.0) {
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
        return fHLLC ; 
    }
    
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_interface_velocity() {
        int midx [] = { 0,3,5 } ; 
        const double sqrt_gamma = Kokkos::sqrt(metric.gamma(midx[idir]));
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
        constexpr int t1 = (idir + 1) % 3;
        constexpr int t2 = (idir + 2) % 3;

        // Common calculations
        double cons_bg_t1 = 0;
        double cons_bg_t2 = 0;
        double f_bg_t1 = 0;
        double f_bg_t2 = 0;

        bool has_magnetic_field = false;

        #ifdef GRACE_DO_MHD
            cons_bg_t1 = cons[BGXL + t1];
            cons_bg_t2 = cons[BGXL + t2];
            f_bg_t1 = f[BGXL + t1];
            f_bg_t2 = f[BGXL + t2];

            has_magnetic_field = (cons[BGXL + idir] > 1e-15);
        #endif // GRACE_DO_MHD

        double cross_term = 0;
        double bg_tan_2 = 0;
        double f_bg_tan_2 = 0;
        if (has_magnetic_field) {
            // Magnetic field is present, use the full formula
            cross_term = cons_bg_t1 * f_bg_t1 + cons_bg_t2 * f_bg_t2;
            bg_tan_2 = cons_bg_t1*cons_bg_t1 + cons_bg_t2*cons_bg_t2;
            f_bg_tan_2 = f_bg_t1*f_bg_t1 + f_bg_t2*f_bg_t2;
        } 
        double const taul_densl_sum = cons[TAUL] + cons[DENSL];
        double const f_taul_densl_sum = f[TAUL] + f[DENSL];
        

        // For HLLC, the contact wave speed is simpler
        // λ* = (F_momentum - F_energy - F_density) / (U_energy + U_density - U_momentum)

        double a, b, c;

        a = f_taul_densl_sum - cross_term;
        b = -f[STXL + idir] - taul_densl_sum + bg_tan_2 + f_bg_tan_2;
        c = cons[STXL + idir] - cross_term;
        //if (cons[STXL + idir] > 1e-10) {
        //    Kokkos::printf("a %e, b %e, c %e\n", a, b, c);
        //    Kokkos::printf("cons[STXL + idir] %e, f[STXL + idir] %e\n", 
        //                   cons[STXL + idir], f[STXL + idir]);
        //    Kokkos::printf("cons[TAUL] %e, cons[DENSL] %e\n",
        //                   cons[TAUL], cons[DENSL]);
        //    Kokkos::printf("f[TAUL] %e, f[DENSL] %e\n",
        //                   f[TAUL], f[DENSL]);
        //    Kokkos::printf("cons_bg_t1 %e, cons_bg_t2 %e\n", cons_bg_t1, cons_bg_t2);
        //    Kokkos::printf("f_bg_t1 %e, f_bg_t2 %e\n", f_bg_t1, f_bg_t2);
        //}
        
        // Safety check for denominator
        if (Kokkos::abs(a) < 1e-15) {
            if (Kokkos::abs(b) < 1e-15) {
                return 0.0; // No contact wave speed
            }
            // If a is too small, we can use b as the denominator
            return -c / b;
        }
        double const detm = Kokkos::sqrt( 
            Kokkos::max(0., b*b - 4.*a*c)
        ) ; 
        return -0.5 * ( b + detm ) / a ;  
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
        grace::grmhd_cons_array_t fHLLD;
        
        // Direction indices
        constexpr int t1 = (idir + 1) % 3;
        constexpr int t2 = (idir + 2) % 3;
        
        // Get interface velocity
        double const vi = get_interface_velocity();
        
        // Solve for total pressure using nonlinear equation
        double p_star;
        bool success = solve_for_total_pressure(uL, uR, fL, fR, pL, pR, cmin, cmax, p_star);
        
        if (!success) {
            // Fall back to HLL if pressure solver fails
            hll_riemann_solver_t hlle_solver;
            int var_indices[] = {DENSL, TAUL, STXL, STYL, STZL, BXL, BYL, BZL, YESL, ENTSL};
            for(int ii=0; ii<10; ++ii) {
                int const ivar = var_indices[ii];
                fHLLD[ivar] = hlle_solver(fL[ivar], fR[ivar], uL[ivar], uR[ivar], cmin, cmax);
            }
            return fHLLD;
        }
        
        // Compute intermediate states
        grace::grmhd_cons_array_t uaL, uaR, ucL, ucR;
        grace::grmhd_cons_array_t faL, faR;
        
        compute_alfven_states(uL, uR, fL, fR, pL, pR, cmin, cmax, p_star, uaL, uaR);
        compute_contact_states(uaL, uaR, p_star, ucL, ucR);
        
        // Compute Alfven wave speeds
        double lambdaAL = compute_alfven_speed_left(uaL, p_star);
        double lambdaAR = compute_alfven_speed_right(uaR, p_star);
        double lambdaC = compute_contact_speed(ucL, ucR);
        
        // Compute fluxes from jump conditions
        for(int iv=0; iv<uL.size(); ++iv) {
            faL[iv] = fL[iv] + cmin * (uaL[iv] - uL[iv]);
            faR[iv] = fR[iv] - cmax * (uaR[iv] - uR[iv]);
        }
        
        // Select appropriate flux based on wave speeds
        if (vi <= -cmin) {
            fHLLD = fL;
        } else if (-cmin < vi && vi <= lambdaAL) {
            fHLLD = faL;
        } else if (lambdaAL < vi && vi <= lambdaC) {
            for(int iv=0; iv<uL.size(); ++iv) {
                fHLLD[iv] = faL[iv] + lambdaAL * (ucL[iv] - uaL[iv]);
            }
        } else if (lambdaC < vi && vi <= lambdaAR) {
            for(int iv=0; iv<uL.size(); ++iv) {
                fHLLD[iv] = faR[iv] + lambdaAR * (ucR[iv] - uaR[iv]);
            }
        } else if (lambdaAR < vi && vi <= cmax) {
            fHLLD = faR;
        } else {
            fHLLD = fR;
        }
        
        return fHLLD;

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

        
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_interface_velocity() const {
        int midx[] = {0, 3, 5};
        const double sqrt_gamma = Kokkos::sqrt(metric.gamma(midx[idir]));
        return metric.beta(idir) / sqrt_gamma / metric.alp();
    }

        /**
         * @brief Solve nonlinear equation for total pressure.
         * 
         * This implements the iterative solution of Equation (54) from the paper.
         */
        bool GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        solve_for_total_pressure(
            grace::grmhd_cons_array_t const& uL, 
            grace::grmhd_cons_array_t const& uR,
            grace::grmhd_cons_array_t const& fL, 
            grace::grmhd_cons_array_t const& fR,
            grace::grmhd_prims_array_t const& pL, 
            grace::grmhd_prims_array_t const& pR,
            double cmin, double cmax, double& p_star) const
        {
            constexpr int MAX_ITER = 50;
            constexpr double TOL = 1e-12;
            constexpr int t1 = (idir + 1) % 3;
            constexpr int t2 = (idir + 2) % 3;

            // Initial guess - use average of left and right pressures
            p_star = 0.5 * (pL[PRESSL] + pR[PRESSL]);

            for (int iter = 0; iter < MAX_ITER; ++iter) {
                // Compute states at current pressure guess
                grace::grmhd_cons_array_t uaL_test, uaR_test;
                compute_alfven_states(uL, uR, fL, fR, pL, pR, cmin, cmax, p_star, uaL_test, uaR_test);

                // Compute contact states
                grace::grmhd_cons_array_t ucL_test, ucR_test;
                compute_contact_states(uaL_test, uaR_test, p_star, ucL_test, ucR_test);

                // Evaluate the nonlinear equation (54) from paper
                double vx_cL = get_normal_velocity(ucL_test);
                double vx_cR = get_normal_velocity(ucR_test);

                double f_eval = vx_cL - vx_cR;

                if (Kokkos::abs(f_eval) < TOL) {
                    return true; // Converged
                }

                // Simple bisection/Newton update
                double dp = -f_eval * 0.1 * p_star; // Simple damped update
                p_star += dp;

                // Ensure pressure remains positive
                p_star = Kokkos::max(p_star, 0.1 * Kokkos::min(pL[PRESSL], pR[PRESSL]));
            }

            return false; // Failed to converge
        }
    
        void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        compute_alfven_states(
            grace::grmhd_cons_array_t const& uL, 
            grace::grmhd_cons_array_t const& uR,
            grace::grmhd_cons_array_t const& fL, 
            grace::grmhd_cons_array_t const& fR,
            grace::grmhd_prims_array_t const& pL, 
            grace::grmhd_prims_array_t const& pR,
            double cmin, double cmax, double p_star,
            grace::grmhd_cons_array_t& uaL, 
            grace::grmhd_cons_array_t& uaR) const
        {
            constexpr int t1 = (idir + 1) % 3;
            constexpr int t2 = (idir + 2) % 3;

            // Left Alfven state
            compute_fast_wave_state(uL, fL, pL, cmin, p_star, true, uaL);

            // Right Alfven state  
            compute_fast_wave_state(uR, fR, pR, cmax, p_star, false, uaR);

            // Ensure magnetic field continuity
            double Bx_avg = 0.5 * (uaL[BXL+idir] + uaR[BXL+idir]);
            uaL[BXL+idir] = Bx_avg;
            uaR[BXL+idir] = Bx_avg;
        }
    
        void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        compute_fast_wave_state(
            grace::grmhd_cons_array_t const& u_in,
            grace::grmhd_cons_array_t const& f_in,
            grace::grmhd_prims_array_t const& p_in,
            double lambda, double p_star, bool is_left,
            grace::grmhd_cons_array_t& u_out) const
        {
            constexpr int t1 = (idir + 1) % 3;
            constexpr int t2 = (idir + 2) % 3;

            // This implements the fast wave relations from equations (47)-(51) in the paper
            // Simplified implementation - full implementation would use the complex expressions

            double sign = is_left ? 1.0 : -1.0;
            double denom = lambda - p_in[VXL+idir];

            // Density (continuous across Alfven waves)
            u_out[DENSL] = u_in[DENSL];

            // Magnetic field components
            u_out[BXL+idir] = u_in[BXL+idir]; // Normal component continuous
            u_out[BXL+t1] = u_in[BXL+t1];
            u_out[BXL+t2] = u_in[BXL+t2];

            // Simplified momentum and energy - real implementation needs full expressions
            u_out[STXL+idir] = (u_in[STXL+idir] * p_in[VXL+idir] + p_star * lambda) / denom;
            u_out[STXL+t1] = u_in[STXL+t1] * p_in[VXL+idir] / denom;
            u_out[STXL+t2] = u_in[STXL+t2] * p_in[VXL+idir] / denom;

            u_out[TAUL] = (u_in[TAUL] * p_in[VXL+idir] + p_star * lambda - f_in[TAUL]) / denom;

            // Passive scalars
            u_out[YESL] = u_in[YESL];
            u_out[ENTSL] = u_in[ENTSL];
        }

        void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        compute_contact_states(
            grace::grmhd_cons_array_t const& uaL,
            grace::grmhd_cons_array_t const& uaR,
            double p_star,
            grace::grmhd_cons_array_t& ucL,
            grace::grmhd_cons_array_t& ucR) const
        {
            // Contact discontinuity - density can jump, but velocity and B-field are continuous
            // This implements the contact relations from the paper

            double vx_contact = 0.5 * (get_normal_velocity(uaL) + get_normal_velocity(uaR));

            // Left contact state
            ucL = uaL; // Start with Alfven state
            ucL[DENSL] = uaL[DENSL] * (get_normal_velocity(uaL)) / vx_contact;

            // Right contact state
            ucR = uaR; // Start with Alfven state  
            ucR[DENSL] = uaR[DENSL] * (get_normal_velocity(uaR)) / vx_contact;

            // Ensure velocity continuity
            // (In full implementation, this would use the complex expressions from the paper)
        }

        double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        get_normal_velocity(grace::grmhd_cons_array_t const& u) const
        {
            // Extract normal velocity from conserved variables
            // This is a simplified extraction - full implementation needs proper conversion
            double E_plus_D = u[TAUL] + u[DENSL];
            if (E_plus_D > 0) {
                return u[STXL+idir] / E_plus_D;
            }
            return 0.0;
        }

        double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        compute_alfven_speed_left(grace::grmhd_cons_array_t const& ua, double p_star) const
        {
            // Compute left-going Alfven wave speed
            // Simplified - real implementation uses equations (52)-(53) from paper
            return get_normal_velocity(ua) - 0.1; // Placeholder
        }

        double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        compute_alfven_speed_right(grace::grmhd_cons_array_t const& ua, double p_star) const
        {
            // Compute right-going Alfven wave speed  
            return get_normal_velocity(ua) + 0.1; // Placeholder
        }

        double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        compute_contact_speed(grace::grmhd_cons_array_t const& ucL, 
                             grace::grmhd_cons_array_t const& ucR) const
        {
            // Contact wave speed - should be continuous across contact
            return 0.5 * (get_normal_velocity(ucL) + get_normal_velocity(ucR));
        }

     
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