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
template< int idir> 
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
     * @brief Compute HLLC fluxes for relativistic Magneto-Hydro-Dynamics.
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
     * 
     * For the HLLC Implementation with magnetic fields, I (Keneth Miler) followed the
     * Paper of Mignone https://arxiv.org/pdf/2111.09369. 
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
        // TODO check missing Minkowski metric.
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
template< int idir> 
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
     * @brief Compute HLLD fluxes for relativistic Magneto-Hydro-Dynamics.
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
     * 
     * For the HLLD Implementation with magnetic fields, I (Keneth Miler) followed the
     * Paper of Mignone https://arxiv.org/pdf/2111.09369. 
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
        , double const cmax 
        , double const total_press)
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
        grace::grmhd_cons_array_t uHLLE, fHLLE; 
        hll_riemann_solver_t hlle_solver ; 
        for( int ii=0; ii<num_vars; ++ii) {
            int const ivar = var_indices[ii] ; 
            fHLLE[ivar] = 
                hlle_solver(fL[ivar],fR[ivar],uL[ivar],uR[ivar],cmin,cmax) ; 
            uHLLE[ivar] = 
                hlle_solver.get_state(fL[ivar],fR[ivar],uL[ivar],uR[ivar],cmin,cmax) ;
        }

        grace::grmhd_cons_array_t uHLLD, fHLLD; 
        grace::grmhd_cons_array_t ucL, ucR, uaL, uaR; 
        grace::grmhd_cons_array_t faL, faR;
        constexpr int max_iter=100;
        int err=15;

        constexpr int t1 = (idir + 1) % 3;
        constexpr int t2 = (idir + 2) % 3;

        // Create HLLD computation object
        HLLDComputation hlld_calc(fL, fR, uL, uR, pL, pR, cmin, cmax);
        double const p_star = utils::safe_secant_bisection(hlld_calc, total_press*0.1, total_press*10, 1e-12, err, max_iter) ; 
        
        //  Set the conserved states in the Alfven waves
        uaL[DENSL] = hlld_calc.Dens_aL ;
        uaR[DENSL] = hlld_calc.Dens_aR ;
        uaL[TAUL] = hlld_calc.Tau_aL ;
        uaR[TAUL] = hlld_calc.Tau_aR ;
        uaL[STXL+idir] = hlld_calc.S_aL ;
        uaR[STXL+idir] = hlld_calc.S_aR ;
        uaL[STXL+t1] = hlld_calc.St1_aL ;
        uaR[STXL+t1] = hlld_calc.St1_aR ;
        uaL[STXL+t2] = hlld_calc.St2_aL ;
        uaR[STXL+t2] = hlld_calc.St2_aR ;
        uaL[BGXL+idir] = uL[BGXL+idir] ; // Keep the magnetic field component in the direction idir.
        uaR[BGXL+idir] = uR[BGXL+idir] ; // Keep the magnetic field component in the direction idir.
        uaL[BGXL+t1] = hlld_calc.Bt1_aL ;
        uaR[BGXL+t1] = hlld_calc.Bt1_aR ;
        uaL[BGXL+t2] = hlld_calc.Bt2_aL ;
        uaR[BGXL+t2] = hlld_calc.Bt2_aR ;

        // Can also copy the magneticc field
        ucL[BGXL+idir] = hlld_calc.B_idir_c;
        ucR[BGXL+idir] = hlld_calc.B_idir_c;
        ucL[BGXL+t1] = hlld_calc.B_t1_c;
        ucR[BGXL+t1] = hlld_calc.B_t1_c;
        ucL[BGXL+t2] = hlld_calc.B_t2_c;
        ucR[BGXL+t2] = hlld_calc.B_t2_c;


        faL[DENSL] = fL[DENSL] + (-cmin)*(uaL[DENSL]-uL[DENSL]) ;
        faR[DENSL] = fR[DENSL] + ( cmax)*(uaR[DENSL]-uR[DENSL]) ;
        faL[TAUL] = fL[TAUL] + (-cmin)*(uaL[TAUL]-uL[TAUL]) ;
        faR[TAUL] = fR[TAUL] + ( cmax)*(uaR[TAUL]-uR[TAUL]) ;
        faL[STXL+idir] = fL[STXL+idir] + (-cmin)*(uaL[STXL+idir]-uL[STXL+idir]) ;
        faR[STXL+idir] = fR[STXL+idir] + ( cmax)*(uaR[STXL+idir]-uR[STXL+idir]) ;
        faL[STYL+t1] = fL[STYL+t1] + (-cmin)*(uaL[STYL+t1]-uL[STYL+t1]) ;
        faR[STYL+t1] = fR[STYL+t1] + ( cmax)*(uaR[STYL+t1]-uR[STYL+t1]) ;
        faL[STZL+t2] = fL[STZL+t2] + (-cmin)*(uaL[STZL+t2]-uL[STZL+t2]) ;
        faR[STZL+t2] = fR[STZL+t2] + ( cmax)*(uaR[STZL+t2]-uR[STZL+t2]) ;
        faL[BGXL+idir] = fL[BGXL+idir] + (-cmin)*(uaL[BGXL+idir]-uL[BGXL+idir]) ; // Keep the magnetic field component in the direction idir.
        faR[BGXL+idir] = fR[BGXL+idir] + ( cmax)*(uaR[BGXL+idir]-uR[BGXL+idir]) ; // Keep the magnetic field component in the direction idir.
        faL[BGXL+t1] = fL[BGXL+t1] + (-cmin)*(uaL[BGXL+t1]-uL[BGXL+t1]) ;
        faR[BGXL+t1] = fR[BGXL+t1] + ( cmax)*(uaR[BGXL+t1]-uR[BGXL+t1]) ;
        faL[BGXL+t2] = fL[BGXL+t2] + (-cmin)*(uaL[BGXL+t2]-uL[BGXL+t2]) ;
        faR[BGXL+t2] = fR[BGXL+t2] + ( cmax)*(uaR[BGXL+t2]-uR[BGXL+t2]) ;

        // define the wave speeds
        
        double const lambda_aL = hlld_calc.lambda_aL;
        double const lambda_aR = hlld_calc.lambda_aR;
        double const vi = get_interface_velocity() ; // Carlo wrote TODO but seems to be fine.
        

        double const v_idir_c = hlld_calc.K_aL - (uaL[BGXL+idir] * (1 - hlld_calc.KaL2)) / (hlld_calc.S_L*hlld_calc.w_aL - hlld_calc.KaL_Bc) ; 
        double const v_t1_c = hlld_calc.Kt1_aL * (uaL[BGXL+t1] * (1 - hlld_calc.KaL2)    / (hlld_calc.S_L*hlld_calc.w_aL - hlld_calc.KaL_Bc)) ;
        double const v_t2_c = hlld_calc.Kt2_aL * (uaL[BGXL+t2] * (1 - hlld_calc.KaL2)    / (hlld_calc.S_L*hlld_calc.w_aL - hlld_calc.KaL_Bc)) ;

        double const v_B = v_idir_c * uaL[BGXL+idir] + v_t1_c * uaL[BGXL+t1] + v_t2_c * uaL[BGXL+t2];

        double const lambdaC = v_idir_c;

        // Compute Right hand sides
        double const RDens_L = lambda_aL * uaL[DENSL] - faL[DENSL];
        double const RDens_R = lambda_aR * uaR[DENSL] - faR[DENSL]; 

        double const RTau_L = lambda_aL * uaL[TAUL] - faL[TAUL];
        double const RTau_R = lambda_aR * uaR[TAUL] - faR[TAUL];

        // Compute the left and right constact wave states
        ucL[DENSL] = RDens_L / ( lambda_aL - v_idir_c);
        ucR[DENSL] = RDens_R / ( lambda_aR - v_idir_c);

        ucL[TAUL] = (RTau_L + p_star * v_idir_c - v_B * ucL[BGXL+idir]) / ( lambda_aL - v_idir_c);
        ucR[TAUL] = (RTau_R + p_star * v_idir_c - v_B * ucR[BGXL+idir]) / ( lambda_aR - v_idir_c);

        ucL[STXL+idir] = (ucL[TAUL] + p_star)*v_idir_c - v_B * ucL[BGXL+idir];
        ucR[STXL+idir] = (ucR[TAUL] + p_star)*v_idir_c - v_B * ucR[BGXL+idir];
        ucL[STXL+t1] = (ucL[TAUL] + p_star)*v_t1_c - v_B * ucL[BGXL+t1];
        ucR[STXL+t1] = (ucR[TAUL] + p_star)*v_t1_c - v_B * ucR[BGXL+t1];
        ucL[STXL+t2] = (ucL[TAUL] + p_star)*v_t2_c - v_B * ucL[BGXL+t2];
        ucR[STXL+t2] = (ucR[TAUL] + p_star)*v_t2_c - v_B * ucR[BGXL+t2];


        // Now we compute the states for the entropy and ye
        double const RYe_aL = -cmin * uL[YESL] - fL[YESL];
        double const RYe_aR =  cmax * uR[YESL] - fR[YESL];
        uaL[YESL] = RYe_aL/((-cmin) - lambda_aL);
        uaR[YESL] = RYe_aR/(( cmax) - lambda_aR);
        faL[YESL] = fL[YESL] + (-cmin) * (uaL[YESL]-uL[YESL]);
        faR[YESL] = fR[YESL] + (-cmin) * (uaR[YESL]-uR[YESL]);
        double const RYe_cL = lambda_aL * uaL[YESL] - faL[YESL];
        double const RYe_cR = lambda_aR * uaR[YESL] - faR[YESL];
        ucL[YESL] = RYe_cL/(lambda_aL - lambdaC);
        ucR[YESL] = RYe_cR/(lambda_aR - lambdaC); 

        double const REnt_aL = -cmin * uL[ENTSL] - fL[ENTSL];
        double const REnt_aR =  cmax * uR[ENTSL] - fR[ENTSL];
        uaL[ENTSL] = REnt_aL/((-cmin) - lambda_aL);
        uaR[ENTSL] = REnt_aR/(( cmax) - lambda_aR);
        faL[ENTSL] = fL[ENTSL] + (-cmin) * (uaL[ENTSL]-uL[ENTSL]);
        faR[ENTSL] = fR[ENTSL] + (-cmin) * (uaR[ENTSL]-uR[ENTSL]);
        double const REnt_cL = lambda_aL * uaL[ENTSL] - faL[ENTSL];
        double const REnt_cR = lambda_aR * uaR[ENTSL] - faR[ENTSL];
        ucL[ENTSL] = REnt_cL/(lambda_aL - lambdaC);
        ucR[ENTSL] = REnt_cR/(lambda_aR - lambdaC); 

        #ifdef GRACE_ENABLE_B_FIELD_GLM
            // Now we calculate the states for phi
            double const RPhi_aL = -cmin * uL[PHIG_GLML] - fL[PHIG_GLML];
            double const RPhi_aR =  cmax * uR[PHIG_GLML] - fR[PHIG_GLML];
            uaL[PHIG_GLML] = RPhi_aL/((-cmin) - lambda_aL);
            uaR[PHIG_GLML] = RPhi_aR/(( cmax) - lambda_aR);
            faL[PHIG_GLML] = fL[PHIG_GLML] + (-cmin) * (uaL[PHIG_GLML]-uL[PHIG_GLML]);
            faR[PHIG_GLML] = fR[PHIG_GLML] + (-cmin) * (uaR[PHIG_GLML]-uR[PHIG_GLML]);
            double const RPhi_cL = lambda_aL * uaL[PHIG_GLML] - faL[PHIG_GLML];
            double const RPhi_cR = lambda_aR * uaR[PHIG_GLML] - faR[PHIG_GLML];
            ucL[PHIG_GLML] = RPhi_cL/(lambda_aL - lambdaC);
            ucR[PHIG_GLML] = RPhi_cR/(lambda_aR - lambdaC); 
        #endif //GRACE_ENABLE_B_FIELD_GLM

        if ( -cmin >= vi ) {
            fHLLD = fL ;
            uHLLD = uL ;  
        } else if (-cmin < vi and vi < lambda_aL) {
            fHLLD = faL ;
            uHLLD = uaL ;
        }else if ( lambda_aL < vi and vi < lambdaC ) {
            for( int iv=0; iv<uL.size(); ++iv) {
                fHLLD[iv] = faL[iv] + lambda_aL * ( ucL[iv] - uL[iv] ) ; 
            }
            uHLLD = ucL ;
        } else if ( lambdaC <= vi and vi < lambda_aR  ) { 
            for( int iv=0; iv<uL.size(); ++iv) {
                fHLLD[iv] = faR[iv] + lambda_aR * ( ucR[iv] - uR[iv] ) ; 
            }
            uHLLD = ucR ;  
        } else if ( lambda_aR <= vi and vi < cmax  ) { 
            fHLLD = faR ;
            uHLLD = uaR ; }
        else {
            fHLLD = fR ; 
            uHLLD = uR ;  
        }

        if (err>0) {
            fHLLD = fHLLE;
            uHLLD = uHLLE;
        }
        return {fHLLD, uHLLD} ; 
    }
    
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    get_interface_velocity() {
        int midx [] = { 0,3,5 } ; 
        const double sqrt_gamma = Kokkos::sqrt(metric.gamma(midx[idir]));
        return metric.beta(idir) / sqrt_gamma / metric.alp();
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    compute_b2( grace::grmhd_prims_array_t const& prims 
              , grace::metric_array_t const& metric ) const 
    {
        double const one_over_alp = 1./metric.alp() ;
        std::array<double,3> const vN {
              one_over_alp * ( prims[VXL] + metric.beta(0) )
            , one_over_alp * ( prims[VYL] + metric.beta(1) )
            , one_over_alp * ( prims[VZL] + metric.beta(2) )
        } ; 
        std::array<double,4> smallb ;
        get_smallb_from_eulerianB(metric, {prims[BXL],prims[BYL],prims[BZL]},
                                          {vN[0], vN[1], vN[2]},
                                          smallb
                                        );
        std::array<double,4> smallbD = metric.lower_4vec(smallb); 
        double const b2 = metric.contract_4dvec_4dcovec(smallb,smallbD);
        return b2;
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
    // HLLD computation struct with all intermediate variables
    struct HLLDComputation {
        // Input parameters (set once in constructor)
        grace::grmhd_cons_array_t const& fL;
        grace::grmhd_cons_array_t const& fR;
        grace::grmhd_cons_array_t const& uL;
        grace::grmhd_cons_array_t const& uR;
        grace::grmhd_prims_array_t const& pL;
        grace::grmhd_prims_array_t const& pR;
        double const cmin;
        double const cmax;
        int const t1;
        int const t2;

        // All intermediate variables (accessible from outside)
        double RS_idir_L, RS_idir_R, RS_t1_L, RS_t1_R, RS_t2_L, RS_t2_R;
        double RB_idir_L, RB_idir_R, RB_t1_L, RB_t1_R, RB_t2_L, RB_t2_R;
        double RTau_L, RTau_R, RDens_L, RDens_R;
        double AL, AR, GL, GR, CL, CR, QL, QR, XL, XR;
        double v_aL, v_aR, vt1_aL, vt1_aR, vt2_aL, vt2_aR;
        double Bt1_aL, Bt1_aR, Bt2_aL, Bt2_aR;
        double v_RS_AL, v_RS_AR, w_aL, w_aR;
        double Dens_aL, Dens_aR, v_B_aL, v_B_aR;
        double Tau_aL, Tau_aR;
        double S_aL, S_aR, St1_aL, St1_aR, St2_aL, St2_aR;
        double S_L, S_R;
        double K_aL, K_aR, Kt1_aL, Kt1_aR, Kt2_aL, Kt2_aR;
        double lambda_aL, lambda_aR;
        double B_idir_c, B_t1_c, B_t2_c;
        double KaL2, KaR2, KaL_Bc, KaR_Bc;

        // Constructor
        GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        HLLDComputation(
            grace::grmhd_cons_array_t const& fL_,
            grace::grmhd_cons_array_t const& fR_,
            grace::grmhd_cons_array_t const& uL_,
            grace::grmhd_cons_array_t const& uR_,
            grace::grmhd_prims_array_t const& pL_,
            grace::grmhd_prims_array_t const& pR_,
            double const cmin_,
            double const cmax_)
            : fL(fL_), fR(fR_), uL(uL_), uR(uR_), pL(pL_), pR(pR_)
            , cmin(cmin_), cmax(cmax_)
            , t1((idir + 1) % 3), t2((idir + 2) % 3)
        {
            // Compute the constant R arrays (independent of p_star)
            compute_r_arrays();
        }

        // Compute R arrays (called once in constructor)
        GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        void compute_r_arrays() {
            // 1. compute the R array for the momenta
            RS_idir_L = -cmin * uL[STXL+idir] - fL[STXL+idir];
            RS_idir_R =  cmax * uR[STXL+idir] - fR[STXL+idir];
            RS_t1_L = -cmin * uL[STXL+t1] - fL[STXL+t1];
            RS_t1_R =  cmax * uR[STXL+t1] - fR[STXL+t1];
            RS_t2_L = -cmin * uL[STXL+t2] - fL[STXL+t2];
            RS_t2_R =  cmax * uR[STXL+t2] - fR[STXL+t2];

            // 2. compute the R array for the magnetic fields
            RB_idir_L = -cmin * uL[BGXL+idir] - fL[BGXL+idir];
            RB_idir_R =  cmax * uR[BGXL+idir] - fR[BGXL+idir];
            RB_t1_L = -cmin * uL[BGXL+t1] - fL[BGXL+t1];
            RB_t1_R =  cmax * uR[BGXL+t1] - fR[BGXL+t1];
            RB_t2_L = -cmin * uL[BGXL+t2] - fL[BGXL+t2];
            RB_t2_R =  cmax * uR[BGXL+t2] - fR[BGXL+t2];

            // 3. compute the R array for tau and D
            RTau_L = -cmin * uL[TAUL] - fL[TAUL];
            RTau_R =  cmax * uR[TAUL] - fR[TAUL];
            RDens_L = -cmin * uL[DENSL] - fL[DENSL];
            RDens_R =  cmax * uR[DENSL] - fR[DENSL];

            // Compute S_L and S_R (independent of p_star)
            if (uL[BGXL+idir] > 1e-15) {
                S_L = +1.0;
            } else if (uL[BGXL+idir] < -1e-15) {
                S_L = -1.0;
            } else {
                S_L = 0.0;
            }

            if (uR[BGXL+idir] > 1e-15) {
                S_R = +1.0;
            } else if (uR[BGXL+idir] < -1e-15) {
                S_R = -1.0;
            } else {
                S_R = 0.0;
            }
        }

        // Compute all variables for given p_star
        GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        void compute_all_variables(double const p_star) {
            // 4. compute various constants
            AL = RS_idir_L - (-cmin)*RTau_L + p_star*(1-cmin*cmin);
            AR = RS_idir_R + ( cmax)*RTau_R + p_star*(1-cmax*cmax);

            GL = RB_t1_L*RB_t1_L + RB_t2_L*RB_t2_L;
            GR = RB_t1_R*RB_t1_R + RB_t2_R*RB_t2_R;

            CL = RS_t1_L*RB_t1_L + RS_t2_L*RB_t2_L;
            CR = RS_t1_R*RB_t1_R + RS_t2_R*RB_t2_R;

            QL = -AL - GL + uL[BGXL+idir]*uL[BGXL+idir]*(1-cmin*cmin);
            QR = -AR - GR + uR[BGXL+idir]*uR[BGXL+idir]*(1-cmax*cmax);

            XL = uL[BGXL+idir]*(AL * (-cmin) * uL[BGXL+idir] + CL) - (AL + GL) * ((-cmin) * p_star + RTau_L);
            XR = uR[BGXL+idir]*(AR * ( cmax) * uR[BGXL+idir] + CR) - (AR + GR) * (( cmax) * p_star + RTau_R);

            // 5. compute the velocities in the fast wave
            v_aL = (uL[BGXL+idir] * (AL * uL[BGXL+idir] + (-cmin) * CL) - (AL + GL) * (p_star + RS_idir_L)) / XL;
            v_aR = (uR[BGXL+idir] * (AR * uR[BGXL+idir] + ( cmax) * CR) - (AR + GR) * (p_star + RS_idir_R)) / XR;

            vt1_aL = (QL * RS_t1_L + RB_t1_L * (CL + uL[BGXL+idir]*((-cmin)*RS_idir_L - RTau_L))) / XL;
            vt1_aR = (QR * RS_t1_R + RB_t1_R * (CR + uR[BGXL+idir]*(( cmax)*RS_idir_R - RTau_R))) / XR;
            vt2_aL = (QL * RS_t2_L + RB_t2_L * (CL + uL[BGXL+idir]*((-cmin)*RS_idir_L - RTau_L))) / XL;
            vt2_aR = (QR * RS_t2_R + RB_t2_R * (CR + uR[BGXL+idir]*(( cmax)*RS_idir_R - RTau_R))) / XR;

            Bt1_aL = (RB_t1_L - uL[BGXL+idir]*vt1_aL) / (-cmin - v_aL);
            Bt1_aR = (RB_t1_R - uR[BGXL+idir]*vt1_aR) / ( cmax - v_aR);
            Bt2_aL = (RB_t2_L - uL[BGXL+idir]*vt2_aL) / (-cmin - v_aL);
            Bt2_aR = (RB_t2_R - uR[BGXL+idir]*vt2_aR) / ( cmax - v_aR);

            v_RS_AL = v_aL*RS_idir_L + vt1_aL*RS_t1_L + vt2_aL*RS_t2_L;
            v_RS_AR = v_aR*RS_idir_R + vt1_aR*RS_t1_R + vt2_aR*RS_t2_R;

            w_aL = p_star + (RTau_L - pL[VXL]*v_RS_AL) / (-cmin - v_aL);
            w_aR = p_star + (RTau_R - pR[VXL]*v_RS_AR) / ( cmax - v_aR);

            Dens_aL = RDens_L / (-cmin - v_aL);
            Dens_aR = RDens_R / ( cmax - v_aR);

            v_B_aL = v_aL * uL[BGXL+idir] + vt1_aL * uL[BGXL+t1] + vt2_aL * uL[BGXL+t2];
            v_B_aR = v_aR * uR[BGXL+idir] + vt1_aR * uR[BGXL+t1] + vt2_aR * uR[BGXL+t2];

            Tau_aL = (RTau_L + p_star * v_aL - v_B_aL * uL[BGXL+idir]) / (-cmin - v_aL);
            Tau_aR = (RTau_R + p_star * v_aR - v_B_aR * uR[BGXL+idir]) / ( cmax - v_aR);

            S_aL = (Tau_aL + p_star)*v_aL - v_B_aL * uL[BGXL+idir];
            S_aR = (Tau_aR + p_star)*v_aR - v_B_aR * uR[BGXL+idir];
            St1_aL = (Tau_aL + p_star)*vt1_aL - v_B_aL * uL[BGXL+t1];
            St1_aR = (Tau_aR + p_star)*vt1_aR - v_B_aR * uR[BGXL+t1];
            St2_aL = (Tau_aL + p_star)*vt2_aL - v_B_aL * uL[BGXL+t2];
            St2_aR = (Tau_aR + p_star)*vt2_aR - v_B_aR * uR[BGXL+t2];

            // 6. compute the alfven wave speeds
            K_aL = (RS_idir_L + p_star - RB_idir_L * S_L * Kokkos::sqrt(w_aL)) / 
                   ((-cmin) * p_star + RTau_L - uL[BGXL+idir] * S_L * Kokkos::sqrt(w_aL));
            K_aR = (RS_idir_R + p_star + RB_idir_R * S_R * Kokkos::sqrt(w_aR)) / 
                   (( cmax) * p_star + RTau_R + uR[BGXL+idir] * S_R * Kokkos::sqrt(w_aR));

            Kt1_aL = (RS_t1_L - RB_t1_L * S_L * Kokkos::sqrt(w_aL)) / 
                     ((-cmin) * p_star + RTau_L - uL[BGXL+idir] * S_L * Kokkos::sqrt(w_aL));
            Kt1_aR = (RS_t1_R + RB_t1_R * S_R * Kokkos::sqrt(w_aR)) / 
                     (( cmax) * p_star + RTau_R + uR[BGXL+idir] * S_R * Kokkos::sqrt(w_aR));
            Kt2_aL = (RS_t2_L - RB_t2_L * S_L * Kokkos::sqrt(w_aL)) / 
                     ((-cmin) * p_star + RTau_L - uL[BGXL+idir] * S_L * Kokkos::sqrt(w_aL));
            Kt2_aR = (RS_t2_R + RB_t2_R * S_R * Kokkos::sqrt(w_aR)) / 
                     (( cmax) * p_star + RTau_R + uR[BGXL+idir] * S_R * Kokkos::sqrt(w_aR));

            lambda_aL = K_aL;
            lambda_aR = K_aR;

            // Compute contact magnetic fields
            B_idir_c = (uR[BGXL+idir] * (lambda_aR - v_aR) + uR[BGXL+idir]*v_aR) / (lambda_aR - lambda_aL) 
                      -(uL[BGXL+idir] * (lambda_aL - v_aL) + uL[BGXL+idir]*v_aL) / (lambda_aR - lambda_aL);
            B_t1_c = (uR[BGXL+t1] * (lambda_aR - v_aR) + uR[BGXL+idir]*vt1_aR) / (lambda_aR - lambda_aL)
                    -(uL[BGXL+t1] * (lambda_aL - v_aL) + uL[BGXL+idir]*vt1_aL) / (lambda_aR - lambda_aL);
            B_t2_c = (uR[BGXL+t2] * (lambda_aR - v_aR) + uR[BGXL+idir]*vt2_aR) / (lambda_aR - lambda_aL)
                    -(uL[BGXL+t2] * (lambda_aL - v_aL) + uL[BGXL+idir]*vt2_aL) / (lambda_aR - lambda_aL);

            // Final pressure equation variables
            KaL2 = K_aL * K_aL + Kt1_aL * Kt1_aL + Kt2_aL * Kt2_aL;
            KaR2 = K_aR * K_aR + Kt1_aR * Kt1_aR + Kt2_aR * Kt2_aR;
            KaL_Bc = K_aL * B_idir_c + Kt1_aL * B_t1_c + Kt2_aL * B_t2_c;
            KaR_Bc = K_aR * B_idir_c + Kt1_aR * B_t1_c + Kt2_aR * B_t2_c;
        }

        // Operator for Brent solver
        GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
        double operator()(double const p_star) {
            // Compute all variables for this pressure
            compute_all_variables(p_star);

            // Return the pressure residual function
            // Fixed the typo: w_aL instead of w_aR in second term
            double const function = (1.0 - KaR2)/(S_R * Kokkos::sqrt(w_aR) - KaR_Bc)
                                  + (1.0 - KaL2)/(S_L * Kokkos::sqrt(w_aL) + KaL_Bc);
            return function;
        }
    };
} ; 



}

#endif 