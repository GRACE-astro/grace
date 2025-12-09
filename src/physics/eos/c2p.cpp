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
#include <grace/physics/eos/ent_based_c2p.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <Kokkos_Core.hpp>

#define C2P_TOLERANCE 1e-14

#define BETA_FLOOR 1e-4
#define WMAX 50 
#define zMax sqrt(SQR(WMAX)-1.)
namespace grace {

static double KOKKOS_FUNCTION 
compute_beta(
  double W,
  grmhd_prims_array_t const& prims,
  metric_array_t const& metric
)
{
  // compute plasma beta 
  std::array<double,4> smallb ; double b2 ; 
  compute_smallb(smallb,b2,W,prims,metric) ; 
  return 2. * prims[PRESSL] / fmax(b2,1e-100) ; 
}

// limit lorentz factor and maybe sigma
template < typename eos_t > 
static void KOKKOS_FUNCTION 
limit_primitives(
  double W,
  grmhd_prims_array_t& prims,
  grmhd_cons_array_t const& cons,
  eos_t const& eos,
  metric_array_t const& metric,
  atmo_params_t atmo_params,
  c2p_err_t& c2p_errors
)
{
  /*
  
  // Do we need to limit the Lorentz factor?
    if (lorentz > limits::lorentz_max) {
      const auto zL = sqrt(SQ(lorentz) - 1.);

      (*error_bits)[c2p_errors::V_MAX_EXCEEDED] = true;

      (*PRIMS)[ZVECX] *= limits::z_max / zL;
      (*PRIMS)[ZVECY] *= limits::z_max / zL;
      (*PRIMS)[ZVECZ] *= limits::z_max / zL;

      // Important we keep RHOSTAR constant so
      // this changes RHOB
      // Why can't we just do this further up when we compute
      // RHOB for the first time? The answer is that we need
      // a selfconsistent solution to obtain the correct velocities
      // from Stilde^2 since we only limit at the very end.
      (*PRIMS)[RHOB] = (*CONS)[RHOSTAR] / limits::lorentz_max;

      // 4. Compute a by making a pressure call
      // Update all vars here, cs2, temp, etc..
      (*PRIMS)[PRESSURE] = eos::press_h_csnd2_temp_entropy__eps_rho_ye(
          h, (*PRIMS)[CS2], (*PRIMS)[TEMP],
          (*PRIMS)[ENTROPY],  // out all but temp (inout)
          (*PRIMS)[EPS], (*PRIMS)[RHOB], (*PRIMS)[YE], error);  // in
    }
  */
  if ( W > WMAX ) {
    double const zL = sqrt(SQR(W)-1.) ; 
    prims[VXL] *= zMax / zL ; 
    prims[VYL] *= zMax / zL ;  
    prims[VZL] *= zMax / zL ; 

    prims[RHOL] = cons[DENSL] / WMAX ; 
    double h, csnd2 ; 
    unsigned int err ; 
    prims[PRESSL] = eos.press_h_csnd2_temp_entropy__eps_rho_ye(
      h,csnd2,prims[TEMPL],prims[ENTL],prims[EPSL],prims[RHOL],prims[YEL],err
    ) ;
    c2p_errors.adjust_s = c2p_errors.adjust_tau = true ; 
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
                 , excision_params_t excision_params 
                 , c2p_err_t& c2p_errors ) 
{
    using mhd_c2p_impl_t = kastaun_c2p_t<eos_t> ;
    using hd_c2p_impl_t = grhd_c2p_t<eos_t> ;
    using backup_c2p_impl_t = entropy_fix_c2p_t<eos_t> ; 

    unsigned int err ;
    bool c2p_failed{ false }, is_atmo{false}            ;
    double W ; 
    /* Undensitize conservs */
    for( auto& c: cons) c /= metric.sqrtg() ;
    /* First we check whether we are in the atmosphere */
    double r = xyz[0] ; 
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
          residual = c2p.invert(prims,W,c2p_errors) ;
        } else {
          hd_c2p_impl_t c2p(eos,metric,cons) ;
          residual = c2p.invert(prims,W,c2p_errors) ;
        }
        auto const beta = compute_beta(W,prims,metric) ; 
        c2p_failed = (math::abs(residual) > C2P_TOLERANCE) ;
        #if 1
        if ( c2p_failed or beta <= 1e-2 ) {
          // backup 
          c2p_errors.adjust_tau = true ; 
          backup_c2p_impl_t c2p(eos,metric,cons) ; 
          residual = c2p.invert(prims,W,c2p_errors) ; 
          c2p_failed = (math::abs(residual) > C2P_TOLERANCE) ;
        }
        #endif 
    } else {
        c2p_failed = true ;
    }


    bool excise = excision_params.excise_by_radius 
                ? r <= excision_params.r_ex 
                : metric.alp() <= excision_params.alp_ex ; 

    if(   prims[RHOL] < (1.+1e-03) * dens_atmo
      or  prims[TEMPL] < temp_atmo 
      or  c2p_failed 
      or  excise ) // TODO excision
    {  
        prims[RHOL]  = excise ? excision_params.rho_ex : dens_atmo ;
        prims[YEL]   = atmo_params.ye_fl   ;
        prims[TEMPL] = excise ? excision_params.temp_ex : temp_atmo ; 
        double csnd2 ; 
        prims[PRESSL] = eos.press_eps_csnd2_entropy__temp_rho_ye(prims[EPSL],csnd2,prims[ENTL],prims[TEMPL],prims[RHOL],prims[YEL],err) ; 
        prims[VXL]   = 0. ;
        prims[VYL]   = 0. ;
        prims[VZL]   = 0. ;
        c2p_errors.adjust_d = c2p_errors.adjust_tau = c2p_errors.adjust_s = true ; 
        is_atmo = true ; 
        W = 1. ;
    }

    limit_primitives(W,prims,cons,eos,metric,atmo_params,c2p_errors) ; 

    /* The 3-velocity in grace is not in the */
    /* ZAMO frame.                           */
    /* NB the c2p itself returns zvec        */
    prims[VXL] = metric.alp()*prims[VXL]/W - metric.beta(0) ;
    prims[VYL] = metric.alp()*prims[VYL]/W - metric.beta(1) ;
    prims[VZL] = metric.alp()*prims[VZL]/W - metric.beta(2) ;
    /* re-densitize conservs */
    for(auto& c: cons) c*=metric.sqrtg() ; 
    /* Re-compute conservative variables based  */
    /* on new primitives, if needed.            */
    grmhd_cons_array_t local_cons ;
    prims_to_conservs(prims,local_cons,metric) ;
    if ( c2p_errors.adjust_tau) {
      cons[TAUL] = local_cons[TAUL] ;
    }
    if ( c2p_errors.adjust_d) {
      cons[DENSL] = local_cons[DENSL] ;
      cons[YESL] = local_cons[YESL] ;
    }
    if ( c2p_errors.adjust_s) {
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

    double b2{0.} ;
    std::array<double,4> smallb{0.,0.,0.,0.} ;
    compute_smallb(smallb,b2,W,prims,metric) ; 


    double const u0 = W / metric.alp();
    double const alp_sqrtgamma = metric.alp() * metric.sqrtg() ;

    cons[DENSL] = alp_sqrtgamma * u0 * prims[RHOL] ; 


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
                      , c2p_err_t& \
                    ) 
INSTANTIATE_TEMPLATE(grace::hybrid_eos_t<grace::piecewise_polytropic_eos_t>) ;
#undef INSTANTIATE_TEMPLATE

}
