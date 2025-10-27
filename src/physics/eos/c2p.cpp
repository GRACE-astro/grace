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
#include <grace/physics/eos/kastaun_c2p.hh>

#include <Kokkos_Core.hpp>

#define C2P_TOLERANCE 1e-10
#define SQR(a) (a)*(a)

#define BETA_FLOOR 1e-4
namespace grace {

template < typename eos_t > 
static void KOKKOS_FUNCTION 
limit_primitives(
  grmhd_prims_array_t& prims,
  eos_t const& eos,
  metric_array_t const& metric,
  atmo_params_t atmo_params,
  bool& recompute_cons,
  bool& adjust_tau,
  bool& adjust_s 
)
{
  // compute plasma beta 
  std::array<double,4> smallb ; 
  // NB here we assume that v == vZAMO 
  double const v2 = metric.square_vec({prims[VXL],prims[VYL],prims[VZL]}) ; 
  double const W  = 1./Kokkos::sqrt(1-v2) ; 
  double const u0 = W / metric.alp();
  std::array<double,3> const ui = { 
        (metric.alp() * prims[VXL] - metric.beta(0)) * u0,
        (metric.alp() * prims[VYL] - metric.beta(1)) * u0,
        (metric.alp() * prims[VZL] - metric.beta(2)) * u0,
  } ; 
  smallb[0] = metric.contract_vec_vec({prims[VXL],prims[VYL],prims[VZL]},{prims[BXL],prims[BYL],prims[BZL]}) * u0 ; 
  for( int i=0; i<3; ++i) {
      smallb[i+1] = (prims[BXL+i] + metric.alp() * smallb[0] * ui[i])/W ; 
  }
  double b2 = ( metric.square_vec({prims[BXL],prims[BYL],prims[BZL]}) + metric.alp()*metric.alp()* smallb[0] * smallb[0] ) / W / W ; 
  
  if ( prims[PRESSL] < 0.5 * BETA_FLOOR * b2) {
    adjust_s = adjust_tau = true ; 
    prims[PRESSL] = 1.001 * 0.5 * BETA_FLOOR * b2 ; 
    double h, csnd2 ; 
    unsigned int err ; 
    prims[EPSL] = eos.eps_h_csnd2_temp_entropy__press_rho_ye(h,csnd2, prims[TEMPL], prims[ENTL], prims[PRESSL],prims[RHOL],prims[YEL],err) ; 
  }

}

template< typename eos_t >
void GRACE_HOST_DEVICE
conservs_to_prims( grmhd_cons_array_t& cons 
                 , grmhd_prims_array_t& prims
                 , metric_array_t const& metric 
                 , eos_t const& eos
                 , std::array<double,3> const& xyz
                 , atmo_params_t atmo_params
                 , excision_params_t excision_params ) 
{
    using mhd_c2p_impl_t = kastaun_c2p_t<eos_t> ;
    using hd_c2p_impl_t = grhd_c2p_t<eos_t> ;

    bool recompute_cons{false}, adjust_tau{false}, adjust_s{false} ; 
    unsigned int err ;
    bool c2p_failed{ false }, is_atmo{false}            ;
    double W                             ;
    /* Undensitize conservs */
    for( auto& c: cons) c /= metric.sqrtg() ;
    /* First we check whether we are in the atmosphere */
    auto const rad = Kokkos::sqrt( xyz[0]*xyz[0]
                                + xyz[1]*xyz[1]
                                + xyz[2]*xyz[2]) ; 
    double const a = 0.9375;
    double r = fmax((sqrt( SQR(rad) - SQR(a) + sqrt(SQR(SQR(rad)-SQR(a))
                      + 4.0*SQR(a)*SQR(xyz[2])) ) / sqrt(2.0)), 1.0);
    double dens_atmo = atmo_params.rho_fl * Kokkos::pow(r,atmo_params.rho_fl_scaling) ; 
    double temp_atmo = atmo_params.temp_fl * Kokkos::pow(r,atmo_params.temp_fl_scaling) ;
    
    prims[BXL] = cons[BSXL] ; 
    prims[BYL] = cons[BSYL] ; 
    prims[BZL] = cons[BSZL] ;

    if( cons[DENSL] > dens_atmo ) {
        double const B2 = metric.square_vec({cons[BSXL],cons[BSYL], cons[BSZL]}) ; 
        double residual = 100 ; 
        if ( B2 / cons[DENSL] > 1e-15 ) {
          mhd_c2p_impl_t c2p(eos,metric,cons) ;
          residual = c2p.invert(prims,adjust_tau) ;
        } else {
          hd_c2p_impl_t c2p(eos,metric,cons) ;
          residual = c2p.invert(prims,adjust_tau) ;
          adjust_s = c2p.S_adjusted ; 
        }
        c2p_failed = (math::abs(residual) > C2P_TOLERANCE) ;
    } else {
        c2p_failed = true ;
    }


    bool excise = excision_params.excise_by_radius 
                ? r <= excision_params.r_ex 
                : metric.alp() <= excision_params.alp_ex ; 

    if(   prims[RHOL] < (1.+1e-03) * dens_atmo
      or  c2p_failed 
      or  excise ) // TODO excision
    {  
        prims[RHOL]  = excise ? excision_params.rho_ex : dens_atmo ;
        prims[YEL]   = atmo_params.ye_fl   ;
        prims[TEMPL] = excise ? excision_params.temp_ex : temp_atmo ; 
        prims[EPSL] = eos.eps__temp_rho_ye(prims[TEMPL],prims[RHOL],prims[YEL],err) ; 
        prims[VXL]   = 0. ;
        prims[VYL]   = 0. ;
        prims[VZL]   = 0. ;
        
	      recompute_cons = true ;
        is_atmo = true ; 
    }

    /* Set pressure entropy and temperature */
    double h, csnd2;
    
    prims[PRESSL] = eos.press_h_csnd2_temp_entropy__eps_rho_ye(
        h,csnd2,prims[TEMPL],prims[ENTL],prims[EPSL],prims[RHOL],prims[YEL], err
    ) ;

    if ( prims[TEMPL] < temp_atmo 
      and (not excise)
      and (not c2p_failed) 
    ) {
        prims[RHOL]  = dens_atmo ;
        prims[YEL]   = atmo_params.ye_fl   ;
        prims[TEMPL] = temp_atmo ; 
        prims[EPSL] = eos.eps__temp_rho_ye(prims[TEMPL],prims[RHOL],prims[YEL],err) ; 
        prims[VXL]   = 0. ;
        prims[VYL]   = 0. ;
        prims[VZL]   = 0. ;
        prims[PRESSL] = eos.press__eps_rho_ye(prims[EPSL],prims[RHOL],prims[YEL],err);
        recompute_cons = true ; 
        is_atmo = true ; 
    }

    if ( ! is_atmo ) {
      limit_primitives(prims,eos,metric,atmo_params,recompute_cons,adjust_tau,adjust_s) ; 
    }

    /* The 3-velocity in grace is not in the */
    /* ZAMO frame.                           */
    prims[VXL] = metric.alp()*prims[VXL] - metric.beta(0) ;
    prims[VYL] = metric.alp()*prims[VYL] - metric.beta(1) ;
    prims[VZL] = metric.alp()*prims[VZL] - metric.beta(2) ;
    /* re-densitize conservs */
    for(auto& c: cons) c*=metric.sqrtg() ; 
    /* Re-compute conservative variables based  */
    /* on new primitives, if needed.            */
    if (recompute_cons)
      prims_to_conservs(prims,cons,metric) ;
    if (adjust_tau) {
      grmhd_cons_array_t local_cons ;
      prims_to_conservs(prims,local_cons,metric) ;
      cons[TAUL] = local_cons[TAUL] ; 
    }
    if (adjust_s) {
      grmhd_cons_array_t local_cons ;
      prims_to_conservs(prims,local_cons,metric) ;
      cons[STXL] = local_cons[STXL] ; 
      cons[STYL] = local_cons[STYL] ; 
      cons[STZL] = local_cons[STZL] ; 
    }
    // no matter what, we reset the entropy 
    cons[ENTSL] = cons[DENSL] * prims[ENTL] ; 
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
                      , std::array<double,3> const& \
                      , atmo_params_t \
                      , excision_params_t \
                    ) 
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE

}
