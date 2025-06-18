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
            , prims[VXL]//*u0 
            , prims[VYL]//*u0
            , prims[VZL]//*u0
        } ; 

        u0 = metric.contract_4dvec_4dcovec(inertial_cotetrad[0],umu) ; //!TODO
        for(int ii=0; ii<3;++ii) {
            uD[ii] = metric.contract_4dvec_4dcovec(inertial_cotetrad[1+ii],umu) ; //!TODO
            // In the tetrad frame lower and upper spatial indices are the same
            prims[VXL+ii]  = uD[ii] / u0 ; 
        }
    }
    #ifdef GRACE_DO_MHD
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    transform_magnetic_fields_to_tetrad_frame(grace::grmhd_prims_array_t& prims) const 
    {
        std::array<double,4> B {
              0 
            , prims[BXL] 
            , prims[BYL]
            , prims[BZL]
        } ; 

        for(int ii=0; ii<3;++ii) {
            prims[BXL+ii] = metric.contract_4dvec_4dcovec(inertial_cotetrad[1+ii],B);
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
    std::pair<grace::grmhd_cons_array_t, grace::grmhd_cons_array_t> 
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
            #ifdef GRACE_ENABLE_B_FIELD_GLM
                BG_idir = uHLLE[BGXL+idir] ;
            #endif // GRACE_ENABLE_B_FIELD_GLM
            
            // Magnetic Field is constant on the face, so we have to copy it.
            uHLLE[BGXL+idir] = BG_idir ;

            // Check for magnetic field strength
            has_magnetic_field = (Kokkos::abs(BG_idir) > 1e-15);

            // Set magnetic field components (same for both cases)
            ucL[BGXL+idir] = BG_idir;
            ucR[BGXL+idir] = BG_idir;
            ucL[BGXL+t1] = uHLLE[BGXL+t1];
            ucR[BGXL+t1] = uHLLE[BGXL+t1];
            ucL[BGXL+t2] = uHLLE[BGXL+t2];
            ucR[BGXL+t2] = uHLLE[BGXL+t2];
            #ifdef GRACE_ENABLE_B_FIELD_GLM
                ucL[PHIG_GLML] = uHLLE[PHIG_GLML];
                ucR[PHIG_GLML] = uHLLE[PHIG_GLML];
            #endif // GRACE_ENABLE_B_FIELD_GLM
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
            vt1 = (uHLLE[BGXL + t1]*lambdaC - fHLLE[BGXL+t1]) / BG_idir;
            vt2 = (uHLLE[BGXL + t2]*lambdaC - fHLLE[BGXL+t2]) / BG_idir;

            v2 = lambdaC*lambdaC + vt1*vt1 + vt2*vt2;
            gamma_star = 1.0 / Kokkos::sqrt(1 - v2);
            v_star_B_star = lambdaC * uHLLE[BGXL+idir] + vt1 * uHLLE[BGXL+t1] + vt2 * uHLLE[BGXL+t2];

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
            ucL[STXL+t1] = (-uHLLE[BGXL+idir] * (uHLLE[BGXL+t1]/gamma_star/gamma_star + v_star_B_star * vt1)
                           - fL[STXL+t1] + (-cmin) * uL[STXL+t1]) / (-cmin - lambdaC);
            ucR[STXL+t1] = (-uHLLE[BGXL+idir] * (uHLLE[BGXL+t1]/gamma_star/gamma_star + v_star_B_star * vt1)
                           - fR[STXL+t1] + ( cmax) * uR[STXL+t1]) / ( cmax - lambdaC);

            ucL[STXL+t2] = (-uHLLE[BGXL+idir] * (uHLLE[BGXL+t2]/gamma_star/gamma_star + v_star_B_star * vt2)
                           - fL[STXL+t2] + (-cmin) * uL[STXL+t2]) / (-cmin - lambdaC);
            ucR[STXL+t2] = (-uHLLE[BGXL+idir] * (uHLLE[BGXL+t2]/gamma_star/gamma_star + v_star_B_star * vt2)
                           - fR[STXL+t2] + ( cmax) * uR[STXL+t2]) / ( cmax - lambdaC);
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

        // Handle supersonic case
        if (lambdaC <= -cmin || lambdaC >= cmax || v2 >= 1.0) {
            fHLLC = fHLLE;
            uHLLC = uHLLE;
        }
        #ifdef GRACE_ENABLE_B_FIELD_GLM
            fHLLC[BGXL+idir] = fHLLE[BGXL+idir] ; // Keep the magnetic field component in the direction idir.
        #endif // GRACE_ENABLE_B_FIELD_GLM
        return {fHLLC, uHLLC} ; 
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

            has_magnetic_field = (Kokkos::abs(cons[BGXL + idir]) > 1e-15);
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
        constexpr int num_scalar_vars = 4
        #ifdef GRACE_ENABLE_B_FIELD_GLM
            + 1
        #endif
        ;
        int scalar_vars_indices[] = {DENSL,TAUL,YESL,ENTSL
                                #ifdef GRACE_ENABLE_B_FIELD_GLM
                                    ,PHIG_GLML
                                #endif // GRACE_ENABLE_B_FIELD_GLM
                                    } ; 
        for (int ii=0; ii<num_scalar_vars; ++ii) {
            int const ivar = scalar_vars_indices[ii] ; 
            f[ivar] = (
                  inertial_tetrad[0][idir] * cons[ivar]
                + inertial_tetrad[idir][idir] * f[ivar] 
            ) ; 
        }
        // TODO ceck missing Minkowski metric.
        // TODO: check if first has to be done the tetrad transformation and then the cotetrad one, or if it does not matter.
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

        #ifdef GRACE_DO_MHD
        std::array<double,3> Btilde = {cons[BGXL], cons[BGYL], cons[BGZL]} ; 
        std::array<double,3> fBtilde = {f[BGXL], f[BGYL], f[BGZL]} ; 
        for( int ivdir=0; ivdir<3; ++ivdir) { 
            double const eS = metric.contract_vec_covec(
                  Btilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ; 
            /* Contracting two lower indices, does not matter   */
            /* since it's only the spatial part of the locally  */
            /* flat tetrad.                                     */
            double const eF = metric.contract_vec_covec(
                  fBtilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ;
            f[BGXL+ivdir] = (
                  inertial_tetrad[0][idir] * eS 
                + inertial_tetrad[idir][idir] * eF 
            ) ; 
        }
        #endif // GRACE_DO_MHD
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
        //get_tetrad_basis_impl(
        //    metric.alp(),
        //    beta,g,gD,tetrad,cotetrad,idir
        //) ;

        // This is a test to see if we can only use the 
        // tetrad in one direction.
        get_tetrad_basis_impl(
            metric.alp(),
            beta,g,gD,tetrad,cotetrad,0
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
            , prims[VXL]//*u0 
            , prims[VYL]//*u0
            , prims[VZL]//*u0
        } ; 

        u0 = metric.contract_4dvec_4dcovec(inertial_cotetrad[0],umu) ; //!TODO
        for(int ii=0; ii<3;++ii) {
            uD[ii] = metric.contract_4dvec_4dcovec(inertial_cotetrad[1+ii],umu) ; //!TODO
            // In the tetrad frame lower and upper spatial indices are the same
            prims[VXL+ii]  = uD[ii] / u0 ; 
        }
    }
    #ifdef GRACE_DO_MHD
    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    transform_magnetic_fields_to_tetrad_frame(grace::grmhd_prims_array_t& prims) const 
    {
        std::array<double,4> B {
              0 
            , prims[BXL] 
            , prims[BYL]
            , prims[BZL]
        } ; 

        for(int ii=0; ii<3;++ii) {
            prims[BXL+ii] = metric.contract_4dvec_4dcovec(inertial_cotetrad[1+ii],B);
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
    std::pair<grace::grmhd_cons_array_t, grace::grmhd_cons_array_t> 
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
        int var_indices[] = {DENSL, TAUL, STXL, STYL, STZL, YESL, ENTSL
            #ifdef GRACE_DO_MHD 
            ,BGXL, BGYL, BGZL 
            #ifdef GRACE_ENABLE_B_FIELD_GLM
            ,PHIG_GLML
            #endif // GRACE_ENABLE_B_FIELD_GLM
            #endif // GRACE_DO_MHD
            } ; 

        constexpr int num_vars = 7
        #ifdef GRACE_DO_MHD
            + 3
        #ifdef GRACE_ENABLE_B_FIELD_GLM
            + 1
        #endif
        #endif
        ;

        grace::grmhd_cons_array_t uHLLD, fHLLD; 
        grace::grmhd_cons_array_t ucL, ucR, uaL, uaR; 
        grace::grmhd_cons_array_t faL, faR;

        // define reduced primitive variables
        // PRESSL, VXL, VYL, VZL, ENTHALPIE
        constexpr int PRESSD = 0;
        constexpr int VIDIR = 1;
        constexpr int VT1 = 2;
        constexpr int VT2 = 3;
        constexpr int ENTHALPIE = 4;
        std::array<double, 5> pAL;
        std::array<double, 5> pCL;
        std::array<double, 5> pAR;
        std::array<double, 5> pCR;

        // compute transversal components
        constexpr int t1 = (idir + 1) % 3;
        constexpr int t2 = (idir + 2) % 3;

        // define the wave speeds
        double const vi = get_interface_velocity() ; // Carlo wrote TODO but seems to be fine.
        double const lambdaC = 0.0;

        //define calculations variables
        // 1. compute the R array for the momenta
        double const RS_idir_L = - cmin * uL[STXL+idir] - fL[STXL+idir] ; 
        double const RS_idir_R =   cmax * uR[STXL+idir] - fR[STXL+idir] ;
        double const RS_t1_L = - cmin * uL[STXL+t1] - fL[STXL+t1] ;
        double const RS_t1_R =   cmax * uR[STXL+t1] - fR[STXL+t1] ;
        double const RS_t2_L = - cmin * uL[STXL+t2] - fL[STXL+t2] ;
        double const RS_t2_R =   cmax * uR[STXL+t2] - fR[STXL+t2] ;

        // 2. compute the R array for the magnetic fields
        double const RB_idir_L = - cmin * uL[BGXL+idir] - fL[BGXL+idir] ;
        double const RB_idir_R =   cmax * uR[BGXL+idir] - fR[BGXL+idir] ;
        double const RB_t1_L = - cmin * uL[BGXL+t1] - fL[BGXL+t1] ;
        double const RB_t1_R =   cmax * uR[BGXL+t1] - fR[BGXL+t1] ;
        double const RB_t2_L = - cmin * uL[BGXL+t2] - fL[BGXL+t2] ;   
        double const RB_t2_R =   cmax * uR[BGXL+t2] - fR[BGXL+t2] ;

        // 3. compute the R array for tau and D
        double const RTau_L = - cmin * uL[TAUL] - fL[TAUL] ;
        double const RTau_R =   cmax * uR[TAUL] - fR[TAUL] ;
        double const RDens_L = - cmin * uL[DENSL] - fL[DENSL] ;
        double const RDens_R =   cmax * uR[DENSL] - fR[DENSL] ;


        // 4. compute various constants
        double const AL = RS_idir_L - (-cmin)*RTau_L + pL[PRESSL]*(1-cmin*cmin) ;
        double const AR = RS_idir_R + ( cmax)*RTau_R + pR[PRESSL]*(1-cmax*cmax) ;

        double const GL = RB_t1_L*RB_t1_L + RB_t2_L*RB_t2_L ;
        double const GR = RB_t1_R*RB_t1_R + RB_t2_R*RB_t2_R ;

        double const CL = RS_t1_L*RB_t1_L + RS_t2_L*RB_t2_L ;
        double const CR = RS_t1_R*RB_t1_R + RS_t2_R*RB_t2_R ;

        double const QL = - AL - GL + uL[BGXL+idir]*uL[BGXL+idir]*(1-cmin*cmin);
        double const QR = - AR - GR + uR[BGXL+idir]*uR[BGXL+idir]*(1-cmax*cmax);

        double const XL = uL[BGXL+idir]*(AL * (-cmin) * uL[BGXL+idir] + CL) - (AL + GL) * ((-cmin) * pL[PRESSL] + RTau_L) ;
        double const XR = uR[BGXL+idir]*(AR * ( cmax) * uR[BGXL+idir] + CR) - (AR + GR) * (( cmax) * pR[PRESSL] + RTau_R) ;

        // 5. compute the velocities in the fast wave
        double const v_aL = (uL[BGXL+idir] * (AL * uL[BGXL+idir] + (-cmin) * CL) - (AL + GL) * (pL[PRESSL] + RS_idir_L)) / XL ;
        double const v_aR = (uR[BGXL+idir] * (AR * uR[BGXL+idir] + ( cmax) * CR) - (AR + GR) * (pR[PRESSL] + RS_idir_R)) / XR ;

        double const vt1_aL = (QL * RS_t1_L + RB_t1_L * (CL + uL[BGXL+idir]*((-cmin)*RS_idir_L - RTau_L))) / XL ;
        double const vt1_aR = (QR * RS_t1_R + RB_t1_R * (CR + uR[BGXL+idir]*(( cmax)*RS_idir_R - RTau_R))) / XR ;
        double const vt2_aL = (QL * RS_t2_L + RB_t2_L * (CL + uL[BGXL+idir]*((-cmin)*RS_idir_L - RTau_L))) / XL ;
        double const vt2_aR = (QR * RS_t2_R + RB_t2_R * (CR + uR[BGXL+idir]*(( cmax)*RS_idir_R - RTau_R))) / XR ;

        double const Bt1_aL = (RB_t1_L - uL[BGXL+idir]*vt1_aL) / (-cmin - v_aL) ;
        double const Bt1_aR = (RB_t1_R - uR[BGXL+idir]*vt1_aR) / ( cmax - v_aR) ;
        double const Bt2_aL = (RB_t2_L - uL[BGXL+idir]*vt2_aL) / (-cmin - v_aL) ;
        double const Bt2_aR = (RB_t2_R - uR[BGXL+idir]*vt2_aR) / ( cmax - v_aR) ;

        double const v_RS_AL = v_aL*RS_idir_L + vt1_aL*RS_t1_L + vt2_aL*RS_t2_L ;
        double const v_RS_AR = v_aR*RS_idir_R + vt1_aR*RS_t1_R + vt2_aR*RS_t2_R ;

        double const w_aL = pL[PRESSL] + (RTau_L - pL[VXL]*v_RS_AL) / (-cmin - v_aL) ;
        double const w_aR = pR[PRESSL] + (RTau_R - pR[VXL]*v_RS_AR) / ( cmax - v_aR) ;

        double const Dens_aL = RDens_L / (-cmin - v_aL) ;
        double const Dens_aR = RDens_R / ( cmax - v_aR) ;

        double const v_B_aL = v_aL * uL[BGXL+idir] + vt1_aL * uL[BGXL+t1] + vt2_aL * uL[BGXL+t2] ;
        double const v_B_aR = v_aR * uR[BGXL+idir] + vt1_aR * uR[BGXL+t1] + vt2_aR * uR[BGXL+t2] ;

        double const Tau_aL = (RTau_L + pL[PRESSL] * v_aL - v_B_aL * uL[BGXL+idir]) / (-cmin - v_aL) ;
        double const Tau_aR = (RTau_R + pR[PRESSL] * v_aR - v_B_aR * uR[BGXL+idir]) / ( cmax - v_aR) ;

        double const S_aL = (Tau_aL + pL[PRESSL])*v_aL - v_B_aL * uL[BGXL+idir];
        double const S_aR = (Tau_aR + pR[PRESSL])*v_aR - v_B_aR * uR[BGXL+idir];
        double const St1_aL = (Tau_aL + pL[PRESSL])*vt1_aL - v_B_aL * uL[BGXL+t1];
        double const St1_aR = (Tau_aR + pR[PRESSL])*vt1_aR - v_B_aR * uR[BGXL+t1];
        double const St2_aL = (Tau_aL + pL[PRESSL])*vt2_aL - v_B_aL * uL[BGXL+t2];  
        double const St2_aR = (Tau_aR + pR[PRESSL])*vt2_aR - v_B_aR * uR[BGXL+t2];

        double S_L, S_R;
        if (uL[BGXL+idir] > 1e-15) {
            S_L = +1 ;
        } else if (uL[BGXL+idir] < -1e-15) {
            S_L = -1;
        } else {
            S_L = 0.0;
        }
        if (uR[BGXL+idir] > 1e-15) {
            S_R = +1 ;
        } else if (uR[BGXL+idir] < -1e-15) {
            S_R = -1;
        } else {
            S_R = 0.0;
        }
        // 6. compute the alfven wave speeds
                                          // Potentially wrong pressure TODO
        double const K_aL = ((RS_idir_L + pL[PRESSL] - RB_idir_L * S_L * Kokkos::sqrt(w_aL))) / ((-cmin) * pL[PRESSL] + RTau_L - uL[BGXL+idir] * S_L * Kokkos::sqrt(w_aL)) ; // TODO S is something else
        double const K_aR = ((RS_idir_R + pR[PRESSL] + RB_idir_R * S_R * Kokkos::sqrt(w_aR))) / (( cmax) * pR[PRESSL] + RTau_R + uR[BGXL+idir] * S_R * Kokkos::sqrt(w_aR)) ;

        double const Kt1_aL = ((RS_t1_L  - RB_t1_L * S_L * Kokkos::sqrt(w_aL))) / ((-cmin) * pL[PRESSL] + RTau_L - uL[BGXL+idir] * S_L * Kokkos::sqrt(w_aL)) ;
        double const Kt1_aR = ((RS_t1_R  + RB_t1_R * S_R * Kokkos::sqrt(w_aR))) / (( cmax) * pR[PRESSL] + RTau_R + uR[BGXL+idir] * S_R * Kokkos::sqrt(w_aR)) ;
        double const Kt2_aL = ((RS_t2_L  - RB_t2_L * S_L * Kokkos::sqrt(w_aL))) / ((-cmin) * pL[PRESSL] + RTau_L - uL[BGXL+idir] * S_L * Kokkos::sqrt(w_aL)) ;
        double const Kt2_aR = ((RS_t2_R  + RB_t2_R * S_R * Kokkos::sqrt(w_aR))) / (( cmax) * pR[PRESSL] + RTau_R + uR[BGXL+idir] * S_R * Kokkos::sqrt(w_aR)) ;

        double const lambda_aL = K_aL ; 
        double const lambda_aR = K_aR ;

        double const B_idir_c = (uR[BGXL+idir] * (lambda_aR - v_aR) + uR[BGXL+idir]*v_aR) / (lambda_aR - lambda_aL) 
                               -(uL[BGXL+idir] * (lambda_aL - v_aL) + uL[BGXL+idir]*v_aL) / (lambda_aR - lambda_aL) ;
        double const B_t1_c = (uR[BGXL+t1] * (lambda_aR - v_aR) + uR[BGXL+idir]*vt1_aR) / (lambda_aR - lambda_aL)
                             -(uL[BGXL+t1] * (lambda_aL - v_aL) + uL[BGXL+idir]*vt1_aL) / (lambda_aR - lambda_aL) ;
        double const B_t2_c = (uR[BGXL+t2] * (lambda_aR - v_aR) + uR[BGXL+idir]*vt2_aR) / (lambda_aR - lambda_aL)
                             -(uL[BGXL+t2] * (lambda_aL - v_aL) + uL[BGXL+idir]*vt2_aL) / (lambda_aR - lambda_aL) ;

        // Implement equation to solve for pressure
        GRACE_HOST_DEVICE
        double operator()(double p_star) const {
            
            double const KaL2 = K_aL * K_aL + Kt1_aL * Kt1_aL + Kt2_aL * Kt2_aL ;
            double const KaR2 = K_aR * K_aR + Kt1_aR * Kt1_aR + Kt2_aR * Kt2_aR ;
            double const KaL_Bc = K_aL * B_idir_c + Kt1_aL * B_t1_c + Kt2_aL * B_t2_c ;
            double const KaR_Bc = K_aR * B_idir_c + Kt1_aR * B_t1_c + Kt2_aR * B_t2_c ;

            double const rho_star_L eos::rho__press_cold_ye(p_star, pL[YEL]) ;
            double const eps_star_L = eps::eps_cold__rho_ye(rho_star_L, pL[YEL]) ;
            double const w_star_R = 1. + pR[EPSL] + p_star / pR[RHOL]
            double const w_star_L = 1. + pL[EPSL] + p_star / pL[RHOL];

            double const func = (1 - KaR2)/(S_R * Kokkos::sqrt(w_star_R) - KaR_Bc)
                               +(1 - KaL2)/(S_L * Kokkos::sqrt(w_star_L) + KaL_Bc)
            return func
        }


        if ( -cmin >= vi ) {
            fHLLD = fL ;
            uHLLD = uL ;  
        } else if (-cmin < vi and vi < lambda_aL) {
            fHLLD = faL ;
            uHLLD = uaL ;
        }else if ( lambda_aL < vi and vi < lambdaC ) {
            for( int iv=0; iv<uL.size(); ++iv) {
                fHLLD[iv] = fL[iv] - cmin * ( ucL[iv] - uL[iv] ) ; 
            }
            uHLLD = ucL ;
        } else if ( lambdaC <= vi and vi < lambda_aR  ) { 
            for( int iv=0; iv<uL.size(); ++iv) {
                fHLLD[iv] = fR[iv] + cmax * ( ucR[iv] - uR[iv] ) ; 
            }
            uHLLD = ucR ;  
        } else if ( lambda_aR <= vi and vi < cmax  ) { 
            fHLLD = faR ;
            uHLLD = uaR ; }
        else {
            fHLLD = fR ; 
            uHLLD = uR ;  
        }
        return {fHLLD, uHLLD} ; 
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

            has_magnetic_field = (Kokkos::abs(cons[BGXL + idir]) > 1e-15);
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
        constexpr int num_scalar_vars = 4
        #ifdef GRACE_ENABLE_B_FIELD_GLM
            + 1
        #endif
        ;
        int scalar_vars_indices[] = {DENSL,TAUL,YESL,ENTSL
                                #ifdef GRACE_ENABLE_B_FIELD_GLM
                                    ,PHIG_GLML
                                #endif // GRACE_ENABLE_B_FIELD_GLM
                                    } ; 
        for (int ii=0; ii<num_scalar_vars; ++ii) {
            int const ivar = scalar_vars_indices[ii] ; 
            f[ivar] = (
                  inertial_tetrad[0][idir] * cons[ivar]
                + inertial_tetrad[idir][idir] * f[ivar] 
            ) ; 
        }
        // TODO ceck missing Minkowski metric.
        // TODO: check if first has to be done the tetrad transformation and then the cotetrad one, or if it does not matter.
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

        #ifdef GRACE_DO_MHD
        std::array<double,3> Btilde = {cons[BGXL], cons[BGYL], cons[BGZL]} ; 
        std::array<double,3> fBtilde = {f[BGXL], f[BGYL], f[BGZL]} ; 
        for( int ivdir=0; ivdir<3; ++ivdir) { 
            double const eS = metric.contract_vec_covec(
                  Btilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ; 
            /* Contracting two lower indices, does not matter   */
            /* since it's only the spatial part of the locally  */
            /* flat tetrad.                                     */
            double const eF = metric.contract_vec_covec(
                  fBtilde
                , { inertial_cotetrad[1][ivdir]
                  , inertial_cotetrad[2][ivdir]
                  , inertial_cotetrad[3][ivdir] }
            ) ;
            f[BGXL+ivdir] = (
                  inertial_tetrad[0][idir] * eS 
                + inertial_tetrad[idir][idir] * eF 
            ) ; 
        }
        #endif // GRACE_DO_MHD
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
        //get_tetrad_basis_impl(
        //    metric.alp(),
        //    beta,g,gD,tetrad,cotetrad,idir
        //) ;

        // This is a test to see if we can only use the 
        // tetrad in one direction.
        get_tetrad_basis_impl(
            metric.alp(),
            beta,g,gD,tetrad,cotetrad,0
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