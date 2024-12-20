
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
              , grmhd_cons_array_t& _conservs )
    : eos(_eos), metric(_metric), conservs(_conservs)
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

        
        inter_vars.What      =0.0;
        inter_vars.vhat2     =0.0;
        inter_vars.yehat     =0.0;
        inter_vars.rhohat    =0.0;
        inter_vars.epsilonhat=0.0;
       
    }

    /**
     * @brief Invert the primitive to conservative transformation
     *        and return primitive variables.
     * @param error c2p inversion residual.
     * @return grmhd_prims_array_t Primitives
     * Note: ignore the below NB, it's no longer valid 
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



        std::array<double, 3> vhatU;
        auto chi =  1. / (1. + mu * math::int_pow<2>(BtildeNorm));
        for(size_t i=0; i<3; i++){
            vhatU[i]= mu * chi * (rtildeU[i] + mu * rD_BtildeU * BtildeU[i]);
        }

        // atmosphere check:
        double rho_atm = eos.rho_atmosphere();

        auto f_mu_final=[this](double lambda, intermediate_variables& iv){
                            return this->f_of_mu(lambda,iv);};

        // call the f_mu function for the last time to get the inter_vars with the last mu:
        //f_mu_final(mu, inter_vars);
        f_mu_final(mu,inter_vars);

        if(inter_vars.rhohat <= rho_atm){
            inter_vars.What = 1.0;
            vhatU[0]=0.0;
            vhatU[1]=0.0;
            vhatU[2]=0.0;
            inter_vars.rhohat=rho_atm;
            inter_vars.yehat=eos.ye_atmosphere();
            inter_vars.epsilonhat=eos.eps_atmosphere();
        }

        // velocity limiter should be here:
        // 
        // if(inter_vars.What > max_lorentz_factor)

        // finally,  fill out the primitives:

        prims[RHOL]=inter_vars.rhohat;
        prims[YEL]=inter_vars.yehat;
        prims[EPSL]=inter_vars.epsilonhat;
        prims[VXL]=vhatU[0];
        prims[VYL]=vhatU[1];
        prims[VZL]=vhatU[2];
        double cs2_,entropy_;
        unsigned int err; 

        prims[PRESSL] =  eos.press_h_csnd2_temp_entropy__eps_rho_ye(prims[ENTL],cs2_,prims[TEMPL],entropy_,
                                                prims[EPSL], prims[RHOL], prims[YEL], err);


        return std::move(prims) ; 
    }
    
 private:
    //! Equation of state
    eos_t const& eos ; 
    //! Metric
    metric_array_t const& metric;
    //! array of conservative variables 
    grmhd_cons_array_t conservs;

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
    // when the required tolerance is reached 
    struct intermediate_variables{
        double What;
        double vhat2;
        double rhohat;
        double yehat;
        double epsilonhat;
    };

    intermediate_variables inter_vars;
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
    f_of_mu(double const& mu, std::optional<std::reference_wrapper<intermediate_variables>> inter_results = std::nullopt) const {
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

        if (inter_results) {
            inter_results->get().What = What;
            inter_results->get().rhohat = rhohat;
            inter_results->get().epsilonhat = epsilonhat;
            inter_results->get().yehat = yel;
            inter_results->get().vhat2 = vhat2;
        }

        // finally, we return f(\mu)
        return mu - 1./(math::max(nu_A,nu_B) + mu * rbar2);
    }

} ; 

}

#endif /* GRACE_PHYSICS_EOS_C2P_KASTAUN_HH */