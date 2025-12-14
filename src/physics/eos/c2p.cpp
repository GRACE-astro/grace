/**
 * @file c2p.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-10
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

#include <grace_config.h>

#include <grace/physics/eos/c2p.hh>
#include <grace/physics/eos/grhd_c2p.hh>
#include <grace/physics/eos/grmhd_c2p.hh>
#include <grace/physics/eos/kastaun_c2p.hh>

#include <Kokkos_Core.hpp>

#define C2P_TOLERANCE 10

namespace grace {

template< typename eos_t >
static void KOKKOS_INLINE_FUNCTION 
limit_conserved(
    grace::grmhd_cons_array_t&  cons,
    metric_array_t const& metric,
    eos_t const& eos,
    c2p_err_t& c2p_err
)
{
    // Tau cannot be lower than
    // D * (1 + eps_min) + B^2 / 2 - D
    // FIXME here we assume eps_min == 0
    double const tau_floor = metric.square_vec({cons[BSXL],cons[BSYL],cons[BSZL]}) ; 
    if ( cons[TAUL] < tau_floor ) {
        c2p_err.adjust_tau = true ; 
        cons[TAUL] = tau_floor ;
    }

    // Moreover, S^2 cannot be higher than
    // (tau + D)^2
    double const S2_max = SQR((cons[TAUL] + cons[DENSL])); 
    double const S2 = metric.square_covec({cons[STXL],cons[STYL],cons[STZL]}) ; 
    if ( S2 > S2_max ) {
        c2p_err.adjust_s = true ; 
        double const fact = 0.9999 * S2_max/S2 ; 
        cons[STXL] *= fact ; 
        cons[STYL] *= fact ; 
        cons[STZL] *= fact ; 
    }
}

template< typename eos_t >
void GRACE_HOST_DEVICE
conservs_to_prims(  grace::grmhd_cons_array_t&  cons
                  , grace::grmhd_prims_array_t& prims
                  , grace::metric_array_t const& metric
                  , eos_t const& eos 
                  , atmo_params_t const& atmo 
                  , excision_params_t const& excision 
                  , double * rtp
                  , c2p_err_t& c2p_err )
{
    
    using c2p_impl_t = kastaun_c2p_t<eos_t> ;
    bool c2p_failed{ false }             ;

    /* Undensitize conservs */
    for( auto& c: cons) c /= metric.sqrtg() ;
    
    /* Limit conserved vars  */
    limit_conserved<eos_t>(cons,metric,eos,c2p_err) ; 

    /* Get atmo rho and temp */
    auto const dens_atmo =  atmo.rho_fl * pow(rtp[0], atmo.rho_fl_scaling);
    auto const temp_atmo =  atmo.rho_fl * pow(rtp[0], atmo.rho_fl_scaling);
    
    /* If D>rho_atm we solve*/
    if( cons[DENSL] > dens_atmo ) {
        c2p_impl_t c2p(eos,metric,cons) ;
        double residual = c2p.invert(prims) ;
        c2p_failed = (math::abs(residual) > C2P_TOLERANCE) ;
    } else {
        c2p_failed = true ;
    }

    // now we check for atmo / excision 

    // excision criterion 
    bool excise = excision.excise_by_radius     
                ? rtp[0] <= excision.r_ex 
                : metric.alp() <= excision.alp_ex ; 
    if ( excise ) {
        prims[RHOL]  = excision.rho_ex  ; 
        prims[TEMPL] = excision.temp_ex ; 
        prims[VXL]   = 0. ;
        prims[VYL]   = 0. ;
        prims[VZL]   = 0. ;
        prims[YEL]   = 0   ;
        // get pressure, eps and entropy
        double csnd2 ;  
        unsigned int eos_err;  
        prims[PRESSL] = eos.press_eps_csnd2__temp_rho_ye(
            prims[EPSL], csnd2, prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // reset all conserved
        c2p_err.adjust_tau = c2p_err.adjust_d = c2p_err.adjust_s = true ; 
    } else if (prims[RHOL] < (1.+1e-03) * dens_atmo or c2p_failed ) {
        prims[RHOL]  = dens_atmo  ; 
        prims[TEMPL] = temp_atmo  ; 
        prims[VXL]   = 0. ;
        prims[VYL]   = 0. ;
        prims[VZL]   = 0. ;
        prims[YEL]   = 0   ;
        // get pressure, eps and entropy
        double csnd2 ;  
        unsigned int eos_err;  
        prims[PRESSL] = eos.press_eps_csnd2__temp_rho_ye(
            prims[EPSL], csnd2, prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // reset all conserved
        c2p_err.adjust_tau = c2p_err.adjust_d = c2p_err.adjust_s = true ; 
    } else if (prims[TEMPL] < temp_atmo) {
        // In this case we only reset 
        // T, eps and press 
        prims[TEMPL] = temp_atmo  ; 
        // get pressure, eps and entropy
        double csnd2 ;  
        unsigned int eos_err;  
        prims[PRESSL] = eos.press_eps_csnd2__temp_rho_ye(
            prims[EPSL], csnd2, prims[TEMPL], prims[RHOL], prims[YEL], eos_err
        ); 
        // reset all conserved
        c2p_err.adjust_tau = c2p_err.adjust_s = true ; 
    }

    /* The 3-velocity in grace is not in the */
    /* ZAMO frame.                           */
    prims[VXL] = metric.alp()*prims[VXL] - metric.beta(0) ;
    prims[VYL] = metric.alp()*prims[VYL] - metric.beta(1) ;
    prims[VZL] = metric.alp()*prims[VZL] - metric.beta(2) ;
    /* Re-compute conservative variables based  */
    /* on new primitives.                       */
    prims_to_conservs(prims,cons,metric) ;
    /* Re-densitize conservs */
    //for( auto& c: cons) c *= metric.sqrtg() ;
}

void GRACE_HOST_DEVICE
prims_to_conservs( grace::grmhd_prims_array_t& prims
                 , grace::grmhd_cons_array_t& cons 
                 , grace::metric_array_t const& metric )
{
    std::array<double,3> vZAMO {
          (prims[VXL]+metric.beta(0))/metric.alp()
        , (prims[VYL]+metric.beta(1))/metric.alp()
        , (prims[VZL]+metric.beta(2))/metric.alp()
    } ; 
    double const v2 = metric.square_vec(vZAMO) ; 
    double const W  = 1./Kokkos::sqrt(1-v2) ; 
    double const u0 = W / metric.alp();
    double const alp_sqrtgamma = metric.alp() * metric.sqrtg() ;

    cons[DENSL] = alp_sqrtgamma * u0 * prims[RHOL] ; 

    double b2{0.} ;
    std::array<double,4> smallb{0.,0.,0.,0.} ;
    std::array<double,3> const ui = { 
        prims[VXL] * u0,
        prims[VYL] * u0,
        prims[VZL] * u0,
    } ; 
    smallb[0] = metric.contract_vec_vec(vZAMO,{prims[BXL],prims[BYL],prims[BZL]}) * u0 ; 
    for( int i=0; i<3; ++i) {
        smallb[i+1] = (prims[BXL+i] + metric.alp() * smallb[0] * ui[i])/W ; 
    }
    b2 = ( metric.square_vec({prims[BXL],prims[BYL],prims[BZL]}) + metric.alp()*metric.alp()* smallb[0] * smallb[0] ) / W / W ; 
    auto smallbD = metric.lower_4vec(smallb) ; 

    double const one_over_alp2 = 1./math::int_pow<2>(metric.alp());
    double const rho0_h_plus_b2 = (prims[RHOL]*(1+prims[EPSL])) + prims[PRESSL] + b2 ;
    double const alp2_sqrtgamma = math::int_pow<2>(metric.alp()) * metric.sqrtg() ;
    double const g4uptt = -one_over_alp2 ; 
    
    double const P_plus_half_b2 = (prims[PRESSL] + 0.5*b2);
    double const Tuptt = rho0_h_plus_b2 * math::int_pow<2>(u0) + P_plus_half_b2 * g4uptt - math::int_pow<2>(smallb[0]) ; 
    cons[TAUL] = alp2_sqrtgamma * Tuptt - cons[DENSL] ;

     
    auto uD = metric.lower({prims[VXL]+metric.beta(0),prims[VYL]+metric.beta(1),prims[VZL]+metric.beta(2)}) ; 
    for(auto & uu: uD) uu *= u0 ; 

    cons[STXL] = alp_sqrtgamma * (rho0_h_plus_b2*u0*uD[0]-smallb[0]*smallbD[1]) ; 
    cons[STYL] = alp_sqrtgamma * (rho0_h_plus_b2*u0*uD[1]-smallb[0]*smallbD[2]) ; 
    cons[STZL] = alp_sqrtgamma * (rho0_h_plus_b2*u0*uD[2]-smallb[0]*smallbD[3]) ;

    cons[YESL] = cons[DENSL] * prims[YEL] ;
    cons[ENTSL] = cons[DENSL] * prims[ENTL] ;
    ////
    return ; 
}

#define INSTANTIATE_TEMPLATE(EOS) \
template \
void GRACE_HOST_DEVICE \
conservs_to_prims<EOS>( grace::grmhd_cons_array_t&  \
                      , grace::grmhd_prims_array_t&  \
                      , grace::metric_array_t const&  \
                      , EOS const& eos \
                      , atmo_params_t const& atmo \
                      , excision_params_t const& excision \
                      , double * rtp \
                      , c2p_err_t& c2p_err ) 
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE

}