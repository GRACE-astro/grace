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

    double k1,k2,eta,chi_safeguard,epsdiss;

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
                       , double const dtfact ) const
    {
        using namespace Kokkos ; 
        auto s = subview(this->_state,i,j,k,ALL(),q) ;
        auto a = subview(this->_aux,i,j,k,ALL(),q) ;

        // Declare variables
        double alp{s(ALP_)}, theta{s(THETA_)}, chi{fmax(chi_safeguard,s(CHI_))}, Khat{s(KHAT_)}  ; 
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
        double Bdriver[3] = {s(BDRIVERX_), s(BDRIVERY_), s(BDRIVERZ_) } ; 

        // derivatives
        double dbeta_dx[9], dbeta_dx_upw[9], ddbeta_dx2[18] ; 
        double dBdr_dx_upw[9] ; 
        double dGammat_dx[9], dGammat_dx_upw[9] ; 
        double dalp_dx[3], dalp_dx_upw[3], ddalp_dx2[6];
        double dchi_dx[3], dchi_dx_upw[3], ddchi_dx2[6];
        double dKhat_dx[3], dKhat_dx_upw[3] ; 
        double dtheta_dx[3], dtheta_dx_upw[3] ; 
        double dgtdd_dx[18], dgtdd_dx_upw[18], ddgtdd_dx2[36] ;
        double dAtdd_dx_upw[18] ;   

        // inverse spacing 
        double idx[3] = {_idx(0,q),_idx(1,q),_idx(2,q)} ; 

        // fill derivatives 
        fill_deriv_scalar(this->_state,i,j,k,ALP_,q,dalp_dx,idx[0]) ; 
        fill_deriv_scalar_upw(this->_state,i,j,k,ALP_,q,dalp_dx_upw,beta,idx[0]) ; 
        fill_second_deriv_scalar(this->_state,i,j,k,ALP_,q,ddalp_dx2,idx[0]) ; 

        fill_deriv_scalar(this->_state,i,j,k,CHI_,q,dchi_dx,idx[0]) ; 
        fill_deriv_scalar_upw(this->_state,i,j,k,CHI_,q,dchi_dx_upw,beta,idx[0]) ; 
        fill_second_deriv_scalar(this->_state,i,j,k,CHI_,q,ddchi_dx2,idx[0]) ; 

        fill_deriv_scalar(this->_state,i,j,k,KHAT_,q,dKhat_dx,idx[0]) ; 
        fill_deriv_scalar_upw(this->_state,i,j,k,KHAT_,q,dKhat_dx_upw,beta,idx[0]) ;

        fill_deriv_scalar(this->_state,i,j,k,THETA_,q,dtheta_dx,idx[0]) ; 
        fill_deriv_scalar_upw(this->_state,i,j,k,THETA_,q,dtheta_dx_upw,beta,idx[0]) ;

        fill_deriv_vector(this->_state,i,j,k,BETAX_,q,dbeta_dx,idx[0]) ;
        fill_deriv_vector_upw(this->_state,i,j,k,BETAX_,q,dbeta_dx_upw,beta,idx[0]) ;
        fill_second_deriv_vector(this->_state,i,j,k,BETAX_,q,ddbeta_dx2,idx[0]) ;

        fill_deriv_vector(this->_state,i,j,k,GAMMATX_,q,dGammat_dx,idx[0]) ;
        fill_deriv_vector_upw(this->_state,i,j,k,GAMMATX_,q,dGammat_dx_upw,beta,idx[0]) ;

        fill_deriv_tensor(this->_state,i,j,k,GTXX_,q,dgtdd_dx,idx[0]) ;
        fill_deriv_tensor_upw(this->_state,i,j,k,GTXX_,q,dgtdd_dx_upw,beta,idx[0]) ;
        fill_second_deriv_tensor(this->_state,i,j,k,GTXX_,q,ddgtdd_dx2,idx[0]) ;

        fill_deriv_tensor_upw(this->_state,i,j,k,ATXX_,q,dAtdd_dx_upw,beta,idx[0]) ; 

        fill_deriv_vector_upw(this->_state,i,j,k,BDRIVERX_,q,dBdr_dx_upw,beta,idx[0]) ;

        // compute matter couplings 
        double rho0{a(RHO_)}, eps{a(EPS_)}, press{a(PRESS_)} ; 
        double z[3] = {a(ZVECX_),a(ZVECY_),a(ZVECZ_)} ; 
        double B[3] = {a(BX_),a(BY_),a(BZ_)} ; 

        // outputs 
        double rho, Strace, Si[3], Sij[6] ;
        z4c_get_matter_sources(
            gtdd,beta,alp,chi,z,B,rho0,press,eps,&rho,&Strace,&Si,&Sij
        );
        
        // compute rhs 
        double dchi, dalp, dtheta, dKhat ; 
        double dgtdd[6], dAtdd[6], dGammat[3], dbetau[3], dBdr[3] ; 
        z4c_get_rhs(
            gtdd,Atdd,beta,alp,chi,Bdriver,Gammat,
            theta,Khat,Strace,rho,Sij,Si,k1,k2,eta,
            dgtdd_dx,dgtdd_dx_upw,dAtdd_dx_upw,
            dbeta_dx,dbeta_dx_upw,dBdr_dx_upw,dGammat_dx,dGammat_dx_upw,
            dKhat_dx,dKhat_dx_upw,dchi_dx,dchi_dx_upw,dalp_dx,dalp_dx_upw,
            dtheta_dx,dtheta_dx_upw,ddgtdd_dx2,ddbeta_dx2,ddalp_dx2,ddchi_dx2,
            &dchi,&dgtdd,&dKhat,&dGammat,&dtheta,&dAtdd,&dalp,&dbetau,&dBdr
        ) ; 

        // add dissipation
        dchi   += epsdiss * kreiss_olinger_operator(i,j,k,q,CHI_,idx)   ;   
        dKhat  += epsdiss * kreiss_olinger_operator(i,j,k,q,KHAT_,idx)  ;
        dalp   += epsdiss * kreiss_olinger_operator(i,j,k,q,ALP_,idx)   ;
        dtheta += epsdiss * kreiss_olinger_operator(i,j,k,q,THETA_,idx) ;
        #if 1
        #pragma unroll 6
        for(int icomp=0; icomp<6; ++icomp) {
            dgtdd[icomp] += epsdiss * kreiss_olinger_operator(i,j,k,q,GTXX_+icomp,idx) ;  
            dAtdd[icomp] += epsdiss * kreiss_olinger_operator(i,j,k,q,ATXX_+icomp,idx) ;  
        } 
        #endif 
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
        double alp{s(ALP_)}, theta{s(THETA_)}, chi{s(CHI_)}, Khat{s(KHAT_)} ; 
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
        
        // derivatives
        double dGammat_dx[9], dGammat_dx_upw[9] ; 
        double dchi_dx[3], dchi_dx_upw[3], ddchi_dx2[6];
        double dKhat_dx[3], dKhat_dx_upw[3] ; 
        double dtheta_dx[3], dtheta_dx_upw[3] ; 
        double dgtdd_dx[18], dgtdd_dx_upw[18], ddgtdd_dx2[36] ;
        double dAtdd_dx[18] ;   
        // inverse spacing 
        double idx[3] = {_idx(0,q),_idx(1,q),_idx(2,q)} ; 
        // fill derivatives 
        fill_deriv_scalar(this->_state,i,j,k,CHI_,q,dchi_dx,idx[0]) ; 
        fill_deriv_scalar_upw(this->_state,i,j,k,CHI_,q,dchi_dx_upw,beta,idx[0]) ; 
        fill_second_deriv_scalar(this->_state,i,j,k,CHI_,q,ddchi_dx2,idx[0]) ; 

        fill_deriv_scalar(this->_state,i,j,k,KHAT_,q,dKhat_dx,idx[0]) ; 
        fill_deriv_scalar_upw(this->_state,i,j,k,KHAT_,q,dKhat_dx_upw,beta,idx[0]) ;

        fill_deriv_scalar(this->_state,i,j,k,THETA_,q,dtheta_dx,idx[0]) ; 
        fill_deriv_scalar_upw(this->_state,i,j,k,THETA_,q,dtheta_dx_upw,beta,idx[0]) ;

        fill_deriv_vector(this->_state,i,j,k,GAMMATX_,q,dGammat_dx,idx[0]) ;
        fill_deriv_vector_upw(this->_state,i,j,k,GAMMATX_,q,dGammat_dx_upw,beta,idx[0]) ;

        fill_deriv_tensor(this->_state,i,j,k,GTXX_,q,dgtdd_dx,idx[0]) ;
        fill_deriv_tensor_upw(this->_state,i,j,k,GTXX_,q,dgtdd_dx_upw,beta,idx[0]) ;
        fill_second_deriv_tensor(this->_state,i,j,k,GTXX_,q,ddgtdd_dx2,idx[0]) ;

        fill_deriv_tensor(this->_state,i,j,k,ATXX_,q,dAtdd_dx,idx[0]) ;

        // compute matter couplings 
        double rho0{a(RHO_)}, eps{a(EPS_)}, press{a(PRESS_)} ; 
        double z[3] = {a(ZVECX_),a(ZVECY_),a(ZVECZ_)} ; 
        double B[3] = {a(BX_),a(BY_),a(BZ_)} ; 
        // outputs 
        double rho, Strace, Si[3], Sij[6] ;
        z4c_get_matter_sources(
            gtdd,beta,alp,chi,z,B,rho0,press,eps,&rho,&Strace,&Si,&Sij
        );
        // get constraints 
        double H, Mi[3] ; 
        z4c_get_constraints(
            gtdd,Atdd,chi,theta,Khat,rho,Si,dgtdd_dx,
            dAtdd_dx,dGammat_dx,dKhat_dx,dchi_dx,dtheta_dx,
            ddgtdd_dx2,ddchi_dx2,&H,&Mi
        ) ; 
        // Store 
        a(HAM_) = H ; 
        a(MOMX_) = Mi[0] ; 
        a(MOMY_) = Mi[1] ; 
        a(MOMZ_) = Mi[2] ; 
        // psi4 
        double xyz[3] ; 
        coords.get_physical_coordinates(i,j,k,q,xyz) ; 
        if ( SQR(xyz[0]) + SQR(xyz[1]) < 1e-10 ) {
            xyz[0] += 1e-8 ; 
        }
        z4c_get_psi4(
            gtdd, Atdd, chi, theta, Khat,
            dgtdd_dx, dAtdd_dx, dKhat_dx, dchi_dx,
            dtheta_dx, ddgtdd_dx2, ddchi_dx2, xyz,
            &(a(PSI4RE_)), &(a(PSI4IM_))
        ) ;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    compute_max_eigenspeed( VEC( const int i
                               , const int j
                               , const int k)
                          , const int64_t q ) const
    {
        return 1. ; 
    } 

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

    void KOKKOS_INLINE_FUNCTION 
    impose_algebraic_constraints(grace::var_array_t state, VEC(int i, int j, int k), int q) const 
    {
        /* First impose the det(gtilde) = 1 constraint */
        double * gtxx = &(state(VEC(i,j,k),GTXX_+0,q));
        double * gtxy = &(state(VEC(i,j,k),GTXX_+1,q));
        double * gtxz = &(state(VEC(i,j,k),GTXX_+2,q));
        double * gtyy = &(state(VEC(i,j,k),GTXX_+3,q));
        double * gtyz = &(state(VEC(i,j,k),GTXX_+4,q));
        double * gtzz = &(state(VEC(i,j,k),GTXX_+5,q));

        double const detgt     = -((*gtxz)*(*gtxz)*(*gtyy)) + 2*(*gtxy)*(*gtxz)*(*gtyz) - (*gtxx)*((*gtyz)*(*gtyz)) - (*gtxy)*(*gtxy)*(*gtzz) + (*gtxx)*(*gtyy)*(*gtzz);
        double const cbrtdetgt = Kokkos::cbrt(detgt);

        (*gtxx) /= cbrtdetgt ; 
        (*gtxy) /= cbrtdetgt ; 
        (*gtxz) /= cbrtdetgt ; 
        (*gtyy) /= cbrtdetgt ; 
        (*gtyz) /= cbrtdetgt ; 
        (*gtzz) /= cbrtdetgt ; 

        /* And the trace-free Aij constraint next */
        double const gtXX=(-((*gtyz)*(*gtyz)) + (*gtyy)*(*gtzz)) ;
        double const gtXY=((*gtxz)*(*gtyz) - (*gtxy)*(*gtzz))    ;
        double const gtXZ=(-((*gtxz)*(*gtyy)) + (*gtxy)*(*gtyz)) ;
        double const gtYY=(-((*gtxz)*(*gtxz)) + (*gtxx)*(*gtzz)) ;
        double const gtYZ=((*gtxy)*(*gtxz) - (*gtxx)*(*gtyz))     ;
        double const gtZZ=(-((*gtxy)*(*gtxy)) + (*gtxx)*(*gtyy)) ; 

        double * Atxx = &(state(VEC(i,j,k),ATXX_+0,q));
        double * Atxy = &(state(VEC(i,j,k),ATXX_+1,q));
        double * Atxz = &(state(VEC(i,j,k),ATXX_+2,q));
        double * Atyy = &(state(VEC(i,j,k),ATXX_+3,q));
        double * Atyz = &(state(VEC(i,j,k),ATXX_+4,q));
        double * Atzz = &(state(VEC(i,j,k),ATXX_+5,q));

        double const ATR = (*Atxx)*gtXX + 2*(*Atxy)*gtXY + 2*(*Atxz)*gtXZ + (*Atyy)*gtYY + 2*(*Atyz)*gtYZ + (*Atzz)*gtZZ ; 
        double const corr = -1./3. * ATR ; 
        *Atxx += corr * (*gtxx) ; 
        *Atxy += corr * (*gtxy) ; 
        *Atxz += corr * (*gtxz) ; 
        *Atyy += corr * (*gtyy) ; 
        *Atyz += corr * (*gtyz) ; 
        *Atzz += corr * (*gtzz) ; 

    }

} ; 

}
#endif 