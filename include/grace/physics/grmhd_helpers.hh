/**
 * @file grmhd_helpers.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-06-17
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

#ifndef GRACE_PHYSICS_GRMHD_HELPERS_HH
#define GRACE_PHYSICS_GRMHD_HELPERS_HH

#include <grace_config.h> 
#include <array>
//**************************************************************************************************/
/* Auxiliaries */
//**************************************************************************************************/
/**
 * @brief Helper indices for prim arrays
 * \ingroup physics
 */
enum GRMHD_PRIMS_LOC_INDICES {
    RHOL = 0,
    PRESSL,
    VXL,
    VYL,
    VZL,
    YEL,
    TEMPL,
    EPSL,
    ENTL,
    #ifdef GRACE_DO_MHD
    BXL,
    BYL,
    BZL,
    #endif 
    NUM_PRIMS_LOC
} ; 
/**
 * @brief Helper indices for cons array.
 * \ingroup physics
 */
enum GRMHD_CONS_LOC_INDICES {
    DENSL=0,
    STXL,
    STYL,
    STZL,
    TAUL,
    YESL,
    ENTSL,
    NUM_CONS_LOC
} ; 
namespace grace {
/**
 * @brief Array of GRMHD primitives.
 * \ingroup physics
 */
using grmhd_prims_array_t = std::array<double,NUM_PRIMS_LOC> ; 
/**
 * @brief Array of GRMHD conservatives.
 * \ingroup physics
 */
using grmhd_cons_array_t  = std::array<double,NUM_CONS_LOC>  ;
} /* namespace grace */


#define FILL_PRIMS_ARRAY(primsarr,vview,q,...)        \
primsarr[RHOL] = vview(__VA_ARGS__,RHO_,q);      \
primsarr[PRESSL] = vview(__VA_ARGS__,PRESS_,q) ; \
primsarr[VXL] = vview(__VA_ARGS__,VELX_,q) ;     \
primsarr[VYL] = vview(__VA_ARGS__,VELY_,q) ;     \
primsarr[VZL] = vview(__VA_ARGS__,VELZ_,q) ;     \
primsarr[YEL] = vview(__VA_ARGS__,YE_,q) ;       \
primsarr[TEMPL] = vview(__VA_ARGS__,TEMP_,q) ;   \
primsarr[EPSL] = vview(__VA_ARGS__,EPS_,q) ;     \
primsarr[ENTL] = vview(__VA_ARGS__,ENTROPY_,q)

#define FILL_PRIMS_ARRAY_ZVEC(primsarr,vview,q,...)        \
primsarr[RHOL] = vview(__VA_ARGS__,RHO_,q);      \
primsarr[PRESSL] = vview(__VA_ARGS__,PRESS_,q) ; \
primsarr[VXL] = vview(__VA_ARGS__,ZVECX_,q) ;     \
primsarr[VYL] = vview(__VA_ARGS__,ZVECY_,q) ;     \
primsarr[VZL] = vview(__VA_ARGS__,ZVECZ_,q) ;     \
primsarr[YEL] = vview(__VA_ARGS__,YE_,q) ;       \
primsarr[TEMPL] = vview(__VA_ARGS__,TEMP_,q) ;   \
primsarr[EPSL] = vview(__VA_ARGS__,EPS_,q) ;     \
primsarr[ENTL] = vview(__VA_ARGS__,ENTROPY_,q)

#define FILL_CONS_ARRAY(consarr, vview,q,...)      \
consarr[DENSL] = vview(__VA_ARGS__,DENS_,q);       \
consarr[TAUL] = vview(__VA_ARGS__,TAU_,q);         \
consarr[STXL] = vview(__VA_ARGS__,SX_,q);          \
consarr[STYL] = vview(__VA_ARGS__,SY_,q);          \
consarr[STZL] = vview(__VA_ARGS__,SZ_,q);          \
consarr[YESL] = vview(__VA_ARGS__,YESTAR_,q);      \
consarr[ENTSL] = vview(__VA_ARGS__,ENTROPYSTAR_,q) 

double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
zvec_to_vel(
  grace::metric_array_t const& metric,
  double& zx, double& zy, double & zz
)
{
  double const alp = metric.alp() ;
  double const w   = Kokkos::sqrt(1. 
      + metric.square_vec({zx, zy, zz}));
  zx = alp * zx / w - metric.beta(0) ;
  zy = alp * zy / w - metric.beta(1) ;
  zz = alp * zz / w - metric.beta(2) ; 
  return w;
}


grace::grmhd_prims_array_t GRACE_HOST_DEVICE 
get_primitives_cell_corner(
  grace::var_array_t<GRACE_NSPACEDIM> const& state,
  grace::var_array_t<GRACE_NSPACEDIM> const& cstate,
  grace::metric_array_t const& metric,
  double& W
  VEC(int i, int j, int k),
  int64_t q
)
{
  using namespace Kokkos ;
  std::array<int, NUM_PRIMS_LOC> prim_idx {
    RHO_,PRESS_,ZVECX_,ZVECY_,ZVECZ_,YE_,TEMP_,EPS_,ENTROPY_
  } ; 

  grace::grmhd_prims_array_t prims ; 
  int _idx = 0 ;
  for( auto const& ivar: prim_idx ) {
    auto sview = subview(state,VEC(ALL(),ALL(),ALL()),ivar,q) ; 
    // interpolate 
    prims[_idx] = center_to_corner<2>::interpolate(sview,VEC(i,j,k)) ; 
    ++_idx ; 
  }
  
  // convert zvec to vel 
  W = zvec_to_vel(metric,prims[VXL],prims[VYL],primz[VZL]) ; 
  return prims ; 
}


struct grmhd_id_t {
  double rho;
  double press;
  double ye; 
  double vx, vy, vz;
  double gxx,gxy,gxz,gyy,gyz,gzz; 
  double kxx,kxy,kxz,kyy,kyz,kzz;
  double alp;  
  double betax, betay, betaz ;
} ; 
struct grmhd_id_tag_t {} ; 

struct metric_id_tag_t {} ; 

enum adm_metric_vars_t {
  _GXX_ = 0,
  _GXY_,
  _GXZ_,
  _GYY_,
  _GYZ_,
  _GZZ_,
  _BETAX_,
  _BETAY_,
  _BETAZ_,
  _ALP_,
  _KXX_,
  _KXY_,
  _KXZ_,
  _KYY_,
  _KYZ_,
  _KZZ_,
  _N_ADM_VARS_
} ; 

#endif /* GRACE_PHYSICS_GRMHD_HELPERS_HH */