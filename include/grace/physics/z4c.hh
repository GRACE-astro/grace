/**
 * @file z4c.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-12-20
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
#ifndef GRACE_Z4C_EQ_HH
#define GRACE_Z4C_EQ_HH
#include <grace_config.h>

#include <grace/utils/grace_utils.hh>

#include <grace/system/grace_system.hh>

#include <grace/data_structures/grace_data_structures.hh>

#include <grace/parallel/mpi_wrappers.hh>

#include <grace/utils/metric_utils.hh>
#include <grace/utils/fd_utils.hh>

#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/c2p.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <grace/evolution/fd_evolution_system.hh>

#include <grace/coordinates/coordinate_systems.hh>

#include <grace/amr/amr_functions.hh>
#include <grace/evolution/evolution_kernel_tags.hh>

#include "z4c_subexpressions.hh"
#include "fd_subexpressions.hh"

#include <Kokkos_Core.hpp>
namespace grace {

struct z4c_system_t 
    : public fd_evolution_system_t<z4c_system_t>
{
    private:
    using base_t = fd_evolution_system_t<z4c_system_t>  ;

    double k1,k2,eta,chi_safeguard,alp_min,epsdiss;
    double eta_ad_a, eta_ad_b, eta_ad_r, theta_ad_r, kappa_ad_r, chi_excise ; 
    bool adaptive_eta, is_vacuum ; 

    public:
    z4c_system_t( 
        var_array_t _state,
        var_array_t _aux,
        staggered_variable_arrays_t _sstate
    ) : base_t(_state,_aux,_sstate)
    {
        k1 = get_param<double>("z4c", "kappa_1") ; 
        k2 = get_param<double>("z4c", "kappa_2") ; 
        eta = get_param<double>("z4c", "eta") ; 
        epsdiss = get_param<double>("z4c", "eps_diss") ; 
        chi_safeguard = get_param<double>("z4c", "chi_floor") ; 
        adaptive_eta = get_param<bool>("z4c", "adaptive_eta") ; 
        eta_ad_a = get_param<double>("z4c", "adaptive_eta_a") ; 
        eta_ad_b = get_param<double>("z4c", "adaptive_eta_b") ; 
        eta_ad_r = get_param<double>("z4c", "eta_damp_radius") ; 
        theta_ad_r = get_param<double>("z4c", "theta_damp_radius") ; 
        kappa_ad_r = get_param<double>("z4c", "kappa_damp_radius") ; 
        alp_min = get_param<double>("z4c", "alp_min") ;
        is_vacuum =  get_param<bool>("z4c", "is_vacuum") ; 
        chi_excise = get_param<double>("z4c", "chi_excision") ;
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_update_impl( int const q 
                       , VEC( int const i 
                            , int const j 
                            , int const k)
                       , grace::scalar_array_t<GRACE_NSPACEDIM> const _idx 
                       , grace::var_array_t const state_new 
                       , grace::staggered_variable_arrays_t sstate_new
                       , double const dt 
                       , double const dtfact 
                       , grace::device_coordinate_system coords ) const
    {
        using namespace Kokkos ; 
        auto s = subview(this->_state,i,j,k,ALL(),q) ;
        auto a = subview(this->_aux,i,j,k,ALL(),q) ;
        impose_algebraic_constraints(this->_state,i,j,k,q) ;
        // get radius 
        double r; 
        {
            double xyz[3] ; 
            coords.get_physical_coordinates(i,j,k,q,xyz) ;
            r = sqrt(SQR(xyz[0])+SQR(xyz[1])+SQR(xyz[2])); 
        }
        // get local k1 
        double k1L = (r>kappa_ad_r) ? k1 * kappa_ad_r/r : k1 ; 

        // forward declare rhs 
        double dchi, dalp, dtheta, dKhat ; 
        double dgtdd[6], dAtdd[6], dGammat[3], dbetau[3], dBdr[3] ; 

        // Declare variables
        double alp{s(ALP_)}, theta{s(THETA_)}, chi{s(CHI_)}, Khat{s(KHAT_)};
        double Ktr{Khat+2*theta} ; 
        double beta[3] = {s(BETAX_), s(BETAY_), s(BETAZ_)} ; 
        double gtdd[6] = {
            s(GTXX_), s(GTXY_), s(GTXZ_), 
            s(GTYY_), s(GTYZ_), s(GTZZ_)
        } ; 
        double Atdd[6] = {
            s(ATXX_), s(ATXY_), s(ATXZ_), 
            s(ATYY_), s(ATYZ_), s(ATZZ_)
        } ;
        double Gammat[3] = {s(GAMMATX_), s(GAMMATY_), s(GAMMATZ_)} ; 


        // get metric inverse 
        double gtuu[6] ; 
        z4c_get_inverse_conf_metric(
            gtdd, 1.0, &gtuu
        ) ; 

        double Atuu[6] ; 
        z4c_get_Atuu(Atdd,gtuu,&Atuu) ; 

        double AA{0.} ; 
        z4c_get_Asqr(Atdd,Atuu,&AA) ; 

        double idx[3] = {_idx(0,q),_idx(1,q),_idx(2,q)} ; 

        // get beta deriv,
        // all rhs need it 
        // so we store it once
        double dbeta_dx[9] ;
        fill_deriv_vector(this->_state,i,j,k,BETAX_,q,dbeta_dx,idx[0]) ;

        { // chi rhs 
            double dchi_dx_upw ; 
            fill_deriv_scalar_upw(this->_state,i,j,k,CHI_,q,&dchi_dx_upw,beta,idx[0]) ; 

            z4c_get_chi_rhs(
                alp, chi, theta, Khat, dbeta_dx, dchi_dx_upw, &dchi
            ) ; 
            // let the derivative fall out of scope
        }

        { // gtdd rhs 
            double dgtdd_dx_upw[6] ; 
            fill_deriv_tensor_upw(this->_state,i,j,k,GTXX_,q,dgtdd_dx_upw,beta,idx[0]) ;

            z4c_get_gtdd_rhs(
                gtdd,Atdd,alp,dgtdd_dx_upw,dbeta_dx,&dgtdd
            ) ; 
            // let the derivative fall out of scope
        }

        // christoffel 
        double Gammatddd[18], Gammatudd[18], GammatDu[3] ; 
        {
            // get deriv
            double dgtdd_dx[18] ; 
            fill_deriv_tensor(this->_state,i,j,k,GTXX_,q,dgtdd_dx,idx[0]) ;
            
            z4c_get_first_Christoffel(
                dgtdd_dx, &Gammatddd
            ) ; 
            z4c_get_second_Christoffel(
                gtuu, Gammatddd, &Gammatudd
            ) ; 
            z4c_get_contracted_Christoffel(
                gtuu,Gammatudd,&GammatDu
            ) ; 
        }
        // compute DiDj alp
        double W2DiDjalp[6], DiDialp ; 
        // need some more derivs 
        double dchi_dx[3], dalp_dx[3] ;
        {
            // this is not user elsewhere
            // so we help the compiler 
            // get rid of it 
            double ddalp_dx2[6];
            fill_deriv_scalar(this->_state,i,j,k,ALP_,q,dalp_dx,idx[0]) ; 
            fill_deriv_scalar(this->_state,i,j,k,CHI_,q,dchi_dx,idx[0]) ;
            fill_second_deriv_scalar(this->_state,i,j,k,ALP_,q,ddalp_dx2,idx[0]) ; 
            
            z4c_get_DiDjalp(
                gtdd, chi, gtuu, Gammatudd, dchi_dx, dalp_dx, ddalp_dx2, 
                &W2DiDjalp
            ) ; 

            z4c_get_DiDialp(
                gtuu, W2DiDjalp, &DiDialp
            ) ; 
        }

        double rho{0}, Strace{0}, Si[3] = {0,0,0}, Sij[6] = {0,0,0,0,0,0};
        // Matter sources
        if (!is_vacuum) {
            // compute matter couplings 
            double rho0{a(RHO_)}, eps{a(EPS_)}, press{a(PRESS_)} ; 
            double z[3] = {a(ZVECX_),a(ZVECY_),a(ZVECZ_)} ; 
            double B[3] = {a(BX_),a(BY_),a(BZ_)} ; 
            z4c_get_matter_sources(
                gtdd, beta, alp, chi, gtuu, z, B, rho0, press, eps,
                &rho, &Strace, &Si, &Sij
            ) ; 
        }
        
        // Khat rhs
        {
            double dKhat_dx_upw ; 
            fill_deriv_scalar_upw(this->_state,i,j,k,KHAT_,q,&dKhat_dx_upw,beta,idx[0]) ;
            z4c_get_Khat_rhs(
                alp, theta, Ktr, Strace, rho, k1L, k2, AA, DiDialp, dKhat_dx_upw, &dKhat
            ) ; 
        }
        
        // Ricci 
        double Rtrace; 
        double W2Rdd[6] = {0.,0.,0.,0.,0.,0.};
        {
            // part 1 
            {
                double ddgtdd_dx2[36] ; 
                double dGammat_dx[9] ; 
                fill_second_deriv_tensor(this->_state,i,j,k,GTXX_,q,ddgtdd_dx2,idx[0]) ;
                fill_deriv_vector(this->_state,i,j,k,GAMMATX_,q,dGammat_dx,idx[0]) ;
                z4c_get_Ricci(
                    gtdd,chi, gtuu,Gammatddd,Gammatudd,GammatDu,dGammat_dx,ddgtdd_dx2, &W2Rdd
                ) ; 
            }
            // part 2
            {
                double ddchi_dx2[6];
                fill_second_deriv_scalar(this->_state,i,j,k,CHI_,q,ddchi_dx2,idx[0]) ; 
                z4c_get_Ricci_conf(
                    gtdd, chi, gtuu, Gammatudd, dchi_dx, ddchi_dx2, &W2Rdd
                ) ;
            }
                        
            
            z4c_get_Ricci_trace(
                gtuu, W2Rdd, &Rtrace
            ) ; 
        }   
        
        // theta rhs 
        {
            double theta_damp_fact = (theta_ad_r > 0) ? exp(-(r*r/(theta_ad_r*theta_ad_r))) : 1.0 ;
            double dtheta_dx_upw ; 
            fill_deriv_scalar_upw(this->_state,i,j,k,THETA_,q,&dtheta_dx_upw,beta,idx[0]) ;
            z4c_get_theta_rhs(
                alp, theta, Khat, rho, k1L, k2, theta_damp_fact, AA, Rtrace, dtheta_dx_upw, &dtheta 
            ) ; 
        }

        // Atdd rhs 
        {
            double dAtdd_dx_upw[6] ;   
            fill_deriv_tensor_upw(this->_state,i,j,k,ATXX_,q,dAtdd_dx_upw,beta,idx[0]) ; 
            z4c_get_Atdd_rhs(
                gtdd, Atdd, alp, chi, Ktr, Strace,
                Sij, gtuu, W2DiDjalp, DiDialp, W2Rdd, Rtrace, 
                dAtdd_dx_upw, dbeta_dx, 
                &dAtdd
            ) ; 
        }

        // lapse rhs 
        {
            double dalp_dx_upw ; 
            fill_deriv_scalar_upw(this->_state,i,j,k,ALP_,q,&dalp_dx_upw,beta,idx[0]) ; 

            z4c_get_alpha_rhs(
                alp, Khat, dalp_dx_upw,
                &dalp
            ) ; 
        }
        // shift rhs 
        double Bdriver[3] = {s(BDRIVERX_), s(BDRIVERY_), s(BDRIVERZ_) } ; 
        {
            double dbeta_dx_upw[3]; 
            fill_deriv_vector_upw(this->_state,i,j,k,BETAX_,q,dbeta_dx_upw,beta,idx[0]) ;
            z4c_get_beta_rhs(
                Bdriver, dbeta_dx_upw, &dbetau
            ) ; 
        }

        // Gammatilde rhs 
        double dGammat_dx_upw[3];
        {
            double dKhat_dx[3], ddbeta_dx2[18], dtheta_dx[3] ; 
            fill_deriv_scalar(this->_state,i,j,k,KHAT_,q,dKhat_dx,idx[0]) ;
            fill_deriv_scalar(this->_state,i,j,k,THETA_,q,dtheta_dx,idx[0]) ; 
            fill_second_deriv_vector(this->_state,i,j,k,BETAX_,q,ddbeta_dx2,idx[0]) ;
            fill_deriv_vector_upw(this->_state,i,j,k,GAMMATX_,q,dGammat_dx_upw,beta,idx[0]) ;
            

            z4c_get_Gammatilde_rhs(
                alp,chi,Gammat,Si,k1L,
                gtuu,Atuu,Gammatudd,GammatDu,
                dbeta_dx, dGammat_dx_upw, dKhat_dx,
                dchi_dx, dalp_dx, dtheta_dx, ddbeta_dx2,
                &dGammat
            ) ; 
        }

        // get adaptive eta if necessary 
        // compute local eta if adapted treatment enabled 
        double etaL = (r>eta_ad_r) ? eta*eta_ad_r/r : eta ; 
        if (adaptive_eta) {
            z4c_get_adaptive_eta(
                chi, etaL, gtuu, dchi_dx, eta_ad_a, eta_ad_b, 1e-15, &etaL
            ) ; 
        }

        // B driver rhs 
        {
            double dBdr_dx_upw[3] ; 
            fill_deriv_vector_upw(this->_state,i,j,k,BDRIVERX_,q,dBdr_dx_upw,beta,idx[0]) ;
            z4c_get_Bdriver_rhs(
                Bdriver, etaL, dGammat, dBdr_dx_upw, dGammat_dx_upw, &dBdr
            ) ; 
        }

        // add dissipation
        dchi   += epsdiss * kreiss_olinger_operator(i,j,k,q,CHI_,idx)   ;   
        dKhat  += epsdiss * kreiss_olinger_operator(i,j,k,q,KHAT_,idx)  ;
        dalp   += epsdiss * kreiss_olinger_operator(i,j,k,q,ALP_,idx)   ;
        dtheta += epsdiss * kreiss_olinger_operator(i,j,k,q,THETA_,idx) ;
        #pragma unroll 6
        for(int icomp=0; icomp<6; ++icomp) {
            dgtdd[icomp] += epsdiss * kreiss_olinger_operator(i,j,k,q,GTXX_+icomp,idx) ;  
            dAtdd[icomp] += epsdiss * kreiss_olinger_operator(i,j,k,q,ATXX_+icomp,idx) ;  
        } 
        # pragma unroll 3
        for(int icomp=0; icomp<3; ++icomp) {
            dBdr[icomp] += epsdiss * kreiss_olinger_operator(i,j,k,q,BDRIVERX_+icomp,idx) ;  
            dbetau[icomp]  += epsdiss * kreiss_olinger_operator(i,j,k,q,BETAX_+icomp,idx) ;  
            dGammat[icomp] += epsdiss * kreiss_olinger_operator(i,j,k,q,GAMMATX_+icomp,idx) ;  
        }

        // update 
        auto n = subview(state_new,i,j,k,ALL(),q) ; 
        n(CHI_)   += dt*dtfact*dchi   ; 
        n(ALP_)   += dt*dtfact*dalp   ; 
        n(THETA_) += dt*dtfact*dtheta ;
        n(KHAT_)  += dt*dtfact*dKhat  ;
        #pragma unroll 6 
        for( int ww=0; ww<6; ++ww) {
            n(GTXX_+ww) += dt*dtfact*dgtdd[ww] ; 
            n(ATXX_+ww) += dt*dtfact*dAtdd[ww] ; 
        }
        #pragma unroll 3
        for( int ww=0; ww<3; ++ww) {
            n(BDRIVERX_+ww) += dt*dtfact*dBdr[ww] ;
            n(BETAX_+ww)   += dt*dtfact*dbetau[ww] ;
            n(GAMMATX_+ww) += dt*dtfact*dGammat[ww] ; 
        }
        
        // apply constraints 
        impose_algebraic_constraints(state_new,i,j,k,q) ; 
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_auxiliaries( VEC( const int i
                            , const int j
                            , const int k)
                        , const int64_t q 
                        , grace::scalar_array_t<GRACE_NSPACEDIM> const _idx 
                        , grace::device_coordinate_system coords  ) const 
    {
        using namespace Kokkos ; 
        auto s = subview(this->_state,i,j,k,ALL(),q) ;
        auto a = subview(this->_aux,i,j,k,ALL(),q) ;
        // Declare variables
        double alp{s(ALP_)}, theta{s(THETA_)}, chi{fmax(chi_safeguard,s(CHI_))}, Khat{s(KHAT_)} ; 
        double beta[3] = {s(BETAX_), s(BETAY_), s(BETAZ_)} ; 
        double gtdd[6] = {
            s(GTXX_), s(GTXY_), s(GTXZ_), 
            s(GTYY_), s(GTYZ_), s(GTZZ_)
        } ; 
        double Atdd[6] = {
            s(ATXX_), s(ATXY_), s(ATXZ_), 
            s(ATYY_), s(ATYZ_), s(ATZZ_)
        } ;
        double Gammat[3] = {s(GAMMATX_), s(GAMMATY_), s(GAMMATZ_)} ; 
        
        // This should be 1 and the trace of A 
        // should be zero, but to be safe we 
        // spend an extra 20 FLOPs and enforce it
        // again.
        if( chi > chi_excise ) {
            // determinant
            double detg ; 
            z4c_get_det_conf_metric(
                gtdd, &detg 
            ) ; 

            // get metric inverse 
            double gtuu[6] ; 
            z4c_get_inverse_conf_metric(
                gtdd, detg, &gtuu
            ) ; 

            double Atuu[6] ; 
            z4c_get_Atuu(Atdd,gtuu,&Atuu) ; 

            double AA{0} ; 
            z4c_get_Asqr(
                Atdd,Atuu,&AA
            ) ; 

            // derivatives
            double dchi_dx[3], dKhat_dx[3], dtheta_dx[3], dAtdd_dx[18], dgtdd_dx[18];
            // inverse spacing 
            double idx[3] = {_idx(0,q),_idx(1,q),_idx(2,q)} ; 
            // fill derivatives 
            fill_deriv_scalar(this->_state,i,j,k,CHI_,q,dchi_dx,idx[0]) ; 
            fill_deriv_scalar(this->_state,i,j,k,KHAT_,q,dKhat_dx,idx[0]) ; 
            fill_deriv_scalar(this->_state,i,j,k,THETA_,q,dtheta_dx,idx[0]) ; 
            fill_deriv_tensor(this->_state,i,j,k,GTXX_,q,dgtdd_dx,idx[0]) ;
            fill_deriv_tensor(this->_state,i,j,k,ATXX_,q,dAtdd_dx,idx[0]) ;

            double Gammatddd[18], Gammatudd[18], GammatDu[3]; 
            z4c_get_first_Christoffel(
                dgtdd_dx, &Gammatddd
            ) ;
            z4c_get_second_Christoffel(
                gtuu, Gammatddd, &Gammatudd
            ) ;
            z4c_get_contracted_Christoffel(
                gtuu, Gammatudd, &GammatDu
            ) ;
            // Ricci 
            double Rtrace; 
            
            {
                double W2Rdd[6] = {0.,0.,0.,0.,0.,0.};
                // part 1 
                {
                    double ddgtdd_dx2[36] ; 
                    double dGammat_dx[9] ; 
                    fill_second_deriv_tensor(this->_state,i,j,k,GTXX_,q,ddgtdd_dx2,idx[0]) ;
                    fill_deriv_vector(this->_state,i,j,k,GAMMATX_,q,dGammat_dx,idx[0]) ;
                    z4c_get_Ricci(
                        gtdd,chi, gtuu,Gammatddd,Gammatudd,GammatDu,dGammat_dx,ddgtdd_dx2, &W2Rdd
                    ) ; 
                }
                // part 2
                {
                    double ddchi_dx2[6];
                    fill_second_deriv_scalar(this->_state,i,j,k,CHI_,q,ddchi_dx2,idx[0]) ; 
                    z4c_get_Ricci_conf(
                        gtdd, chi, gtuu, Gammatudd, dchi_dx, ddchi_dx2, &W2Rdd
                    ) ;
                }
                            
                
                z4c_get_Ricci_trace(
                    gtuu, W2Rdd, &Rtrace
                ) ; 
            }   
            // compute matter couplings 
            double rho{0}, Strace{0}, Si[3] = {0,0,0}, Sij[6] = {0,0,0,0,0,0};
            if (!is_vacuum) {
                // compute matter couplings 
                double rho0{a(RHO_)}, eps{a(EPS_)}, press{a(PRESS_)} ; 
                double z[3] = {a(ZVECX_),a(ZVECY_),a(ZVECZ_)} ; 
                double B[3] = {a(BX_),a(BY_),a(BZ_)} ; 
                
                z4c_get_matter_sources(
                    gtdd, beta, alp, chi, gtuu, z, B, rho0, press, eps,
                    &rho, &Strace, &Si, &Sij
                ) ; 
            }
            // get constraints 
            double H, Mi[3] ; 
            z4c_get_constraints(
                Atdd, chi, theta, Khat, rho, Si, gtuu, Atuu, AA,
                Gammatudd, GammatDu, Rtrace, dgtdd_dx, dAtdd_dx,
                dKhat_dx, dchi_dx, dtheta_dx, 
                &H, &Mi
            ) ; 
            // Store 
            a(HAM_) = H ; 
            a(MOMX_) = Mi[0] ; 
            a(MOMY_) = Mi[1] ; 
            a(MOMZ_) = Mi[2] ; 
        } else {
            a(HAM_)  = 0.0 ; 
            a(MOMX_) = 0.0 ; 
            a(MOMY_) = 0.0 ; 
            a(MOMZ_) = 0.0 ;
        }
        
        
    }

    void GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_psi4( VEC( const int i, const int j, const int k)
                , const int64_t q 
                , grace::scalar_array_t<GRACE_NSPACEDIM> const _idx 
                , grace::device_coordinate_system coords) const

    {
        using namespace Kokkos ;
        auto s = subview(this->_state,i,j,k,ALL(),q) ;
        auto a = subview(this->_aux,i,j,k,ALL(),q) ;
        // Declare variables
        double alp{s(ALP_)}, theta{s(THETA_)}, chi{s(CHI_)}, Khat{s(KHAT_)} ; 
        double gtdd[6] = {
            s(GTXX_), s(GTXY_), s(GTXZ_), 
            s(GTYY_), s(GTYZ_), s(GTZZ_)
        } ; 
        double Atdd[6] = {
            s(ATXX_), s(ATXY_), s(ATXZ_), 
            s(ATYY_), s(ATYZ_), s(ATZZ_)
        } ;
        // get metric inverse 
        double gtuu[6] ; 
        z4c_get_inverse_conf_metric(
            gtdd, 1.0, &gtuu
        ) ; 
        // coordinates
        double xyz[3] ; 
        coords.get_physical_coordinates(i,j,k,q,xyz) ;
        // derivatives 
        double dchi_dx[3], dKhat_dx[3], dtheta_dx[3] ; 
        double ddchi_dx2[6], ddgtdd_dx2[36] ; 
        double dgtdd_dx[18], dAtdd_dx[18] ;
        // -- 
        double idx[3] = {_idx(0,q),_idx(1,q),_idx(2,q)} ; 
        // -- 
        fill_deriv_scalar(this->_state,i,j,k,CHI_,q,dchi_dx,idx[0]) ;
        fill_deriv_scalar(this->_state,i,j,k,THETA_,q,dtheta_dx,idx[0]) ;
        fill_deriv_scalar(this->_state,i,j,k,KHAT_,q,dKhat_dx,idx[0]) ;
        // -- 
        fill_deriv_tensor(this->_state,i,j,k,GTXX_,q,dgtdd_dx,idx[0]) ;
        fill_deriv_tensor(this->_state,i,j,k,ATXX_,q,dAtdd_dx,idx[0]) ;
        // -- 
        fill_second_deriv_scalar(this->_state,i,j,k,CHI_,q,ddchi_dx2,idx[0]) ; 
        fill_second_deriv_tensor(this->_state,i,j,k,GTXX_,q,ddgtdd_dx2,idx[0]) ;

        double psi4re, psi4im ; 
        z4c_get_psi4(
            gtdd, Atdd, chi, theta, Khat, gtuu,
            dgtdd_dx, dAtdd_dx, dKhat_dx, dchi_dx,
            dtheta_dx, ddgtdd_dx2, ddchi_dx2, xyz,
            &psi4re, &psi4im
        ) ; 

        a(PSI4RE_) = psi4re ; 
        a(PSI4IM_) = psi4im ; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_max_eigenspeed( VEC( const int i
                               , const int j
                               , const int k)
                          , const int64_t q ) const
    {
        return 1. ; 
    } 

    #if 0
    // this is okay for 4th order FD, but 6th order is hard-coded as of now
    double KOKKOS_INLINE_FUNCTION
    kreiss_olinger_operator(int i, int j, int k, int q, int iv, double idx[3]) const 
    {
        // Eq 63 of https://arxiv.org/pdf/gr-qc/0610128 with r = 3 
        using namespace Kokkos ; 
        auto u = subview(this->_state,ALL(),ALL(),ALL(),iv,q) ; 
        double const dudx = (u(i-3,j,k) - 6*u(i-2,j,k) + 15*u(i-1,j,k) - 20*u(i,j,k) + 15*u(i+1,j,k) - 6*u(i+2,j,k) + u(i+3,j,k))*idx[0] ; 
        double const dudy = (u(i,j-3,k) - 6*u(i,j-2,k) + 15*u(i,j-1,k) - 20*u(i,j,k) + 15*u(i,j+1,k) - 6*u(i,j+2,k) + u(i,j+3,k))*idx[1] ; 
        double const dudz = (u(i,j,k-3) - 6*u(i,j,k-2) + 15*u(i,j,k-1) - 20*u(i,j,k) + 15*u(i,j,k+1) - 6*u(i,j,k+2) + u(i,j,k+3))*idx[2] ;
        return (dudx+dudy+dudz) * (1./64.) ;  
    }
    #else
    double KOKKOS_INLINE_FUNCTION
    kreiss_olinger_operator(int i, int j, int k, int q, int iv, double idx[3]) const 
    {
        // Eq 63 of https://arxiv.org/pdf/gr-qc/0610128 with r = 3 
        using namespace Kokkos ; 
        auto u = subview(this->_state,ALL(),ALL(),ALL(),iv,q) ; 
        double dudx, dudy, dudz; 
        fd_diss_x(u,i,j,k,idx[0],&dudx) ; 
        fd_diss_y(u,i,j,k,idx[1],&dudy) ; 
        fd_diss_z(u,i,j,k,idx[2],&dudz) ; 
        return -(dudx+dudy+dudz) * (1./256.) ;  
    }
    #endif

    void KOKKOS_INLINE_FUNCTION 
    impose_algebraic_constraints(grace::var_array_t state, VEC(int i, int j, int k), int q) const 
    {
        auto s = Kokkos::subview(state,i,j,k,Kokkos::ALL(),q) ;

        double gtdd[6] = {
            s(GTXX_), s(GTXY_), s(GTXZ_), 
            s(GTYY_), s(GTYZ_), s(GTZZ_)
        } ; 
        double Atdd[6] = {
            s(ATXX_), s(ATXY_), s(ATXZ_), 
            s(ATYY_), s(ATYZ_), s(ATZZ_)
        } ;

        // determinant
        double detg ; 
        z4c_get_det_conf_metric(
            gtdd, &detg 
        ) ; 

        double const inv_cbrt_detg = 1.0 / fabs(cbrt(detg));
        #pragma unroll 6
        for( int a=0; a<6; ++a ) gtdd[a] *= inv_cbrt_detg ; 

        // get metric inverse 
        double gtuu[6] ; 
        z4c_get_inverse_conf_metric(
            gtdd, 1.0, &gtuu
        ) ; 

        double Atr = gtuu[0] * Atdd[0] + gtuu[3] * Atdd[3] + gtuu[5] * Atdd[5]
                    + 2 * ( gtuu[1] * Atdd[1] + gtuu[2] * Atdd[2] + gtuu[4] * Atdd[4] );

        double const fixfact = - Atr / 3.;
        #pragma unroll 6
        for( int a=0; a<6; ++a ) Atdd[a] += gtdd[a] * fixfact ; 

        #pragma unroll 6
        for( int a=0; a<6; ++a ) {
            s(ATXX_+a) = Atdd[a] ; 
            s(GTXX_+a) = gtdd[a] ; 
        }

        // enforce lapse and conf fact positivity 
        auto chi = state(VEC(i,j,k),CHI_,q);
        if ( chi <= 0 ) state(VEC(i,j,k),CHI_,q) = chi_safeguard ; 

        state(VEC(i,j,k),ALP_,q) = fmax(state(VEC(i,j,k),ALP_,q), alp_min       ) ; 
    }

} ; 

}
#endif 