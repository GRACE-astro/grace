
/**
 * @file grmhd_c2p_kastaun.hh
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-11-16
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

#ifndef GRACE_PHYSICS_EOS_C2P_KASTAUN_HH
#define GRACE_PHYSICS_EOS_C2P_KASTAUN_HH

#include <grace_config.h>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
//#include <grace/utils/metric_utils.hh>
#include <grace/utils/numerics/metric_utils.hh>

#include <grace/utils/numerics/rootfinding.hh>
#include <grace/physics/eos/eos_base.hh>
#include <grace/physics/eos/hybrid_eos.hh>
#include <grace/physics/eos/piecewise_polytropic_eos.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace {



/**
 * @brief Implementation of conservative 
 *        to primitive conversion routine 
 *        for General Relativistic Magnetohydrodynamics
 *        following the implementation and nomenclature of https://arxiv.org/pdf/2312.11358
 *        Kastaun C2P
 * @tparam eos_t - the eos class
 */
template< typename eos_t >
struct grmhd_c2p_kastaun {

    /**
     * @brief Constructor.
     * 
     * @param _eos Equation of State.
     * @param _metric Metric array.
     * @param conservs Conservative variables.
     * NB: The conservatives are expected to be 
     *     undensitized when passed to the c2p.
     * TODO: Verify this class  design works when a tabulated EOS is used 
     * 
     */
    GRACE_HOST_DEVICE
    grmhd_c2p_kastaun( eos_t const& _eos
              , metric_array_t const& _metric 
              , grmhd_cons_array_t& conservs )
    : eos(_eos), metric(_metric)
    {   

        // first we make sure that momentum is casual 
        StildeU = metric.raise({conservs[STXL],conservs[STYL],conservs[STZL]}) ; 
        auto StildeNorm = 
            Kokkos::sqrt(conservs[STXL]*StildeU[0] + conservs[STYL]*StildeU[1] + conservs[STZL]*StildeU[2] ) ; 
        conservs[TAUL] = math::max(0, conservs[TAUL]) ;
        D  = conservs[DENSL] ; 

        BtildeU[0]=conservs[BGXL]/ Kokkos::sqrt(D);
        BtildeU[1]=conservs[BGXL]/ Kokkos::sqrt(D);
        BtildeU[2]=conservs[BGXL]/ Kokkos::sqrt(D);

        BtildeD = metric.lower({BtildeU[0],BtildeU[1],BtildeU[2]});
        BtildeNorm = Kokkos::sqrt(BtildeU[0]*BtildeD[0] + BtildeU[1]*BtildeD[1] + BtildeU[2]*BtildeD[2]);

        /* Acausal momentum */
        if ( StildeNorm > D+conservs[TAUL] ) {
            double const fact = 0.9999*(D+conservs[TAUL]) ; 
            conservs[STXL] *= fact/StildeNorm ; 
            conservs[STYL] *= fact/StildeNorm  ;
            conservs[STZL] *= fact/StildeNorm  ;
            StildeU = metric.raise({conservs[STXL],conservs[STYL],conservs[STZL]}) ; 
            StildeNorm = fact ; 
            //    Kokkos::sqrt(conservs[STXL]*StildeU[0] + conservs[STYL]*StildeU[1] + conservs[STZL]*StildeU[2] ) ; 
        } 

        // set up rescaled conserved variables 
        ye = conservs[YESL] / D ;
        q  = conservs[TAUL] / D ; 

        rtildeU[0] = StildeU[0]/ D;
        rtildeU[1] = StildeU[1]/ D;
        rtildeU[2] = StildeU[2]/ D;

        rtildeNorm = StildeNorm / D; // r = sqrt (r_i r^i) = sqrt(S_i S^i ) / D

        // finally, get parallel and perpendicular momentum:
        rD_BtildeU = (conservs[STXL] * BtildeU[0] + \
                           conservs[STYL] * BtildeU[1] + \
                           conservs[STZL] * BtildeU[2]      ) / D;

        rtildeU_par[0]= rD_BtildeU * BtildeU[0] / BtildeNorm;
        rtildeU_par[1]= rD_BtildeU * BtildeU[1] / BtildeNorm;
        rtildeU_par[2]= rD_BtildeU * BtildeU[2] / BtildeNorm;

        rtildeU_perp[0] = rtildeU[0] - rtildeU_par[0];
        rtildeU_perp[1] = rtildeU[1] - rtildeU_par[1];
        rtildeU_perp[2] = rtildeU[2] - rtildeU_par[2];

        v0 = math::int_pow<2>(rtildeNorm) / (math::int_pow<2>(rtildeNorm) +  eos.enthalpy_minimum()*eos.enthalpy_minimum());

    }

    /**
     * @brief Invert the primitive to conservative transformation
     *        and return primitive variables.
     * @param error c2p inversion residual.
     * @return grmhd_prims_array_t Primitives.
     * NB: When this function returns, the velocity portion 
     * of the prims array actually contains the z-vector, 
     * the pressure contains the lorentz factor and temperature
     * and entropy are left empty. This is later fixed by the 
     * calling function which will compute \f$v^i\f$, pressure, 
     * entropy and temperature by calling the EOS and adding 
     * the relevant metric components to the velocity.
     */
    grmhd_prims_array_t GRACE_HOST_DEVICE
    invert(double& error) {

        grmhd_prims_array_t prims ; 

        // under the assumption that the conservative variables enter the c2p routine undensitized,
        // magnetic field primitives are just simple copies: 
        prims[BXL]=conservs[BGXL];
        prims[BYL]=conservs[BGYL];
        prims[BZL]=conservs[BGZL];
    
        // prims[RHOL] = D/W ;
        // prims[YEL]  = ye ;
        unsigned long iter_max = 5000;  // change this to be determined elsewhere! 
        double const tolerance = 1e-12; // change this

        // first, we constrain the area of search by finding mu_plus
        double const mu_plus = utils::rootfind_newton_raphson(
                            0.0, 1.0/eos.enthalpy_minimum(),  // lower bound, upper bound
                            [this](double mu){return this -> fa_of_mu(mu);}, // function
                            [this](double mu){return this -> dfa_dmu(mu);}, //  derivative
                            tolerance, iter_max           // tolerance, iteration book-keeper
                            ) + tiny_number;
        
        // f_mu must be an lvalue to comply with brent in rootfinding.hh
        auto f_mu=[this](double lambda){return this->f_of_mu(lambda);};
        // now we look for the root of the master function in the (0, mu_plus] interval:
        auto mu = utils::brent(f_mu,
                                 0.0, mu_plus, tolerance);

        // once we have it, we start by recovering the velocity:
        //  call raise3_ixD(ix^D,myM,r_i(1:ndir),ri(1:ndir))
        //  do idir = 1, ndir
        //     vi_hat(idir) = mu * chi * ( ri(idir) + mu * r_dot_b * bi(idir) )
        //  end do
        std::array<double, 3> vhatU;
        auto chi =  1. / (1. + mu * math::int_pow<2>(BtildeNorm));
        for(size_t i=0; i<3; i++){
            vhatU[i]= mu * chi * (rtildeU[i] + mu * rD_BtildeU * BtildeU[i]);
        }

        // atmosphere check:
        double rho_atm = eos.rho_atmosphere();

//   ! adjust the results if it is invalid
//          if ( rho_hat <= small_rho_thr ) then
//                if (old_bhac_safety) then
//                   call usr_atmo_pt(ix^D,w(ix^D,1:nw),x(ix^D,1:ndim))
//                   cycle
//                endif
//                ! reset the primitive variables
//                W_hat          = 1.0d0
//                vi_hat(1:ndir) = 0.0d0
//                rho_hat        = small_rho
//                eps_hat        = small_eps  
//                ye_hat         = big_ye    
//                adjustment     = .True. 
//          else
//             ! limit the velocities
//             if ( W_hat > lfac_max ) then
//                if (usr_W_limit_scheme) then
//                else ! default one
//                  ! rescale the velocity such that v = v_max and keeping D constant
//                  rescale_factor = v_max / dsqrt( v_hat_sqr )
//                  vi_hat(1:ndir) = vi_hat(1:ndir) * rescale_factor
//                  !v_hat_sqr = v_max**2
//                  W_hat = lfac_max
//                  ! although D is kept constant, density is changed a bit
//                  rho_hat = cons_tmp(ix^D, D_) / W_hat
//                end if

//                !! caseI: should I this?
//                !! Since the rho_hat or sth else is changed
//                !! check if eps fails into the validity range
//                !call eos_get_eps_range(rho_hat, eps_min, eps_max, ye=ye_hat)
   
//                !if ( eps_hat < eps_min ) then
//                !   eps_hat = eps_min
//                !else if ( eps_hat > eps_max ) then
//                !   eps_hat = eps_max
//                !end if

//                ! case II: maybe I can just bound with eos_epsmin/max?
//                eps_hat = max( min( eos_epsmax, eps_hat), eos_epsmin)
//             end if !endif of lfac_max

//          end if !endif of small_rho_thr
//       end if ! end of con2prim or small_D checker



        // auto const func = [&] (double const& zeta) {
        //     return zeta - r / htilde(zeta) ; 
        // } ; 
        // double const zm{ 0.5*k/Kokkos::sqrt(1-math::int_pow<2>(0.5*k))} 
        //            , zp{ 1e-06 + k/Kokkos::sqrt(1-math::int_pow<2>(k))} ; 
        // double const zeta = utils::brent(func,zm,zp,1e-15) ; 
        // double const W = Wtilde(zeta) ; 
        // grmhd_prims_array_t prims ; 
        // prims[RHOL] = D/W ;

        // eos.eps_range__rho_ye(epsmin,epsmax,prims[RHOL],prims[YEL],err) ; 
        // prims[EPSL]   = math::min( epsmax
        //                          , math::max( epsmin
        //                                     , epstilde(W,zeta) ) ) ; 
        // prims[PRESSL] = W ; 
        // double const h = htilde(zeta) ; 
        // prims[VXL] = StildeU[0] / D / h / W; 
        // prims[VYL] = StildeU[1] / D / h / W; 
        // prims[VZL] = StildeU[2] / D / h / W; 
        // error = func(zeta) ;
        // return std::move(prims) ; 
    }
    
 private:
    //! Equation of state
    eos_t const& eos ; 
    //! Metric
    metric_array_t const& metric;
    //! Conserved density
    double D  ;
    //! Electron fraction
    double ye ; 
    //! Rescaled energy
    double q  ; 
    //! Momentum / Energy ratio
    double k  ; 
    //! Momentum with upper indices 
    std::array<double,3> StildeU ; 
    //! Rescaled momentum (with upper indices)
    std::array<double,3> rtildeU ; 
     //! Rescaled momentum norm 
    double rtildeNorm ; 
    //! Rescaled magnetic field (with upper indices)
    std::array<double,3> BtildeU ; 
    //! Rescaled magnetic field (with lower indices)
    std::array<double,3> BtildeD ; 
    //! Rescaled magnetic field norm 
    double BtildeNorm ; 
    
    //! Rescaled momentum, parallel to Btilde field (with upper indices)
    std::array<double,3> rtildeU_par ; 
    //! Rescaled momentum, perpendicular to Btilde field (with upper indices)
    std::array<double,3> rtildeU_perp ; 
    //! Rescaled momentum contracted with the Btilde field 
    double rD_BtildeU;

    // constant from eq. (25) 
    double v0;

    // will be conditionally returned from the f_of_mu call
    struct intermediate_variables{
        double What;
        double rhohat;
        double epsilonhat;
    }

    // small number 
    constexpr static const double tiny_number = std::numeric_limits<double>::epsilon() ;


    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    rbar2_of_mu(double const& mu) const  {   // eq 21
        auto chi =  1. / (1. + mu * math::int_pow<2>(BtildeNorm));
        return math::int_pow<2>(rtildeNorm*chi) + mu*chi*(1+chi)*rD_BtildeU;
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    fa_of_mu(double const& mu) const  {   // eq 21
        return mu*Kokkos::sqrt(eos.enthalpy_minimum()*eos.enthalpy_minimum() + math::int_pow<2>(rbar2_of_mu(mu))) - 1.; 
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    qbar_of_mu(double const& mu) const {
        auto rtildeD_perp = metric.lower({rtildeU_perp[0],rtildeU_perp[1], rtildeU_perp[2]});
        // r_\{\perp} 
        auto rtildeNorm_perp = Kokkos::sqrt(rtildeD_perp[0]*rtildeU_perp[0]+
                                            rtildeD_perp[1]*rtildeU_perp[1]+          
                                            rtildeD_perp[2]*rtildeU_perp[2]);
        auto chi =  1. / (1. + mu * math::int_pow<2>(BtildeNorm));
        return q - 0.5*math::int_pow<2>(BtildeNorm) - 0.5 * math::int_pow<2>(mu*chi*BtildeNorm*rtildeNorm_perp);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    dchi_dmu(double const& mu) const {
        return -1.0 * math::int_pow<2>(BtildeNorm) / math::int_pow<2>(1.0 + mu * math::int_pow<2>(BtildeNorm));
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    drbar2_dmu(double const& mu) const {
        auto dchidmu=dchi_dmu(mu);
        auto chi =  1. / (1. + mu * math::int_pow<2>(BtildeNorm));
        return 2.0 * math::int_pow<2>(rtildeNorm) * chi * dchidmu + \
               math::int_pow<2>(rD_BtildeU) * (chi*(1.0+chi) + mu*dchidmu*(1.0+chi) + mu*chi*dchidmu);
    }

    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    dfa_dmu(double const& mu) const {
        auto drbar2dmu=drbar2_dmu(mu);
        auto rbar2=rbar2_of_mu(mu);
        
        return Kokkos::sqrt(eos.enthalpy_minimum()*eos.enthalpy_minimum() + rbar2) + 0.5 * drbar2dmu / Kokkos::sqrt(eos.enthalpy_minimum()*eos.enthalpy_minimum() + rbar2);
    }

    // the master function of the Kastaun C2P scheme
    double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    f_of_mu(double const& mu) const {
        auto qbar = qbar_of_mu(mu);
        auto rbar2= rbar2_of_mu(mu);
        auto vhat2=math::min(mu*mu*rbar2 ,  v0*v0) ;
        auto What=1./Kokkos::sqrt(1- vhat2);
        auto rhohat0=D/What;
        auto rhohat=math::max(eos.density_minimum(), math::min(eos.density_maximum(),rhohat0));
        auto epsilonhat0=What*(qbar - mu*rbar2) + vhat2*What*What/(1.+What);
        // finally, here a call is made to the EOS table 
        // note that this will have to change based on the 
        double yel{ye} ; 
        unsigned int err; 
        double epslow, epshigh; 
       // double rho; 
        // call to the EOS
        eos.eps_range__rho_ye(epslow,epshigh,rhohat,yel,err) ; 
        auto epsilonhat = math::max(epslow,math::min(epshigh,epsilonhat0 )) ; 

        double h_, csnd2_, temp_, entropy_; // will be discarded 
        // call to the EOS
        auto phat =  eos.press_h_csnd2_temp_entropy__eps_rho_ye(h_,csnd2_,temp_,entropy_,
                                                epsilonhat, rhohat, yel, err);
        
        auto ahat = phat / rhohat / (1 + epsilonhat);
        auto nu_A = (1 + ahat) * (1 + epsilonhat) / What; 
        auto nu_B = (1 + ahat) * (1 + qbar - mu* rbar2);

        // finally, we return f(\mu)
        return mu - 1./(math::max(nu_A,nu_B) + mu * rbar2);
    }



    // double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    // Wtilde(double const& z) const {
    //     return Kokkos::sqrt(1 + math::int_pow<2>(z)) ; 
    // }

    // double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    // rhotilde(double const& W) const {
    //     return D/W ; 
    // }

    // double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
    // epstilde(double const& W, double const& z) const {
    //     return W*q - z*r + math::int_pow<2>(z)/(1+W) ; 
    // }

    // double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    // atilde(double& rho, double& eps) const {
    //     unsigned int err ;
    //     double yel{ye} ; 
    //     auto const press = eos.press__eps_rho_ye(eps,rho,yel,err) ; 
    //     return press / (rho * ( 1 + eps )) ; 
    // }

    // double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
    // htilde(double const& z) const {
    //     auto const W   = Wtilde(z) ; 
    //     auto rho = rhotilde(W) ; 
    //     double epsmin, epsmax; 
    //     double yel{ye} ; 
    //     unsigned int err; 
    //     eos.eps_range__rho_ye(epsmin,epsmax,rho,yel,err) ; 
    //     auto eps = math::max(epsmin,math::min(epsmax,epstilde(W,z))) ; 
    //     return (1+eps) * (1+atilde(rho,eps)) ; 
    // }
} ; 

}

#endif /* GRACE_PHYSICS_EOS_C2P_KASTAUN_HH */