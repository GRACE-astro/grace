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
    BXL,
    BYL,
    BZL,
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
    BSXL,
    BSYL,
    BSZL,
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

#define FILL_METRIC_ARRAY(g, view, q, ...)                    \
g = grace::metric_array_t{  { view(__VA_ARGS__,GXX_,q)   \
                          , view(__VA_ARGS__,GXY_,q)     \
                          , view(__VA_ARGS__,GXZ_,q)     \
                          , view(__VA_ARGS__,GYY_,q)     \
                          , view(__VA_ARGS__,GYZ_,q)     \
                          , view(__VA_ARGS__,GZZ_,q) }   \
                          , { view(__VA_ARGS__,BETAX_,q) \
                          , view(__VA_ARGS__,BETAY_,q)   \
                          , view(__VA_ARGS__,BETAZ_,q) } \
                          , view(__VA_ARGS__,ALP_,q) } 
                          
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

#define AM2 -0.0625
#define AM1  0.5625
#define A0   0.5625
#define A1  -0.0625
#ifdef GRACE_3D
#define COMPUTE_FCVAL_HELPER(mview,i,j,k,ivar,q,idir)                                          \
  AM2*mview(i-2*utils::delta(0,idir),j-2*utils::delta(1,idir),k-2*utils::delta(2,idir),ivar,q) \
+ AM1*mview(i-utils::delta(0,idir),j-utils::delta(1,idir),k-utils::delta(2,idir),ivar,q)       \
+ A0*mview(i,j,k,ivar,q)                                                                       \
+ A1*mview(i+utils::delta(0,idir),j+utils::delta(1,idir),k+utils::delta(2,idir),ivar,q)        
#define COMPUTE_FCVAL(g,mview,i,j,k,q,idir)                     \
g = grace::metric_array_t{                                      \
      {                                                         \
          COMPUTE_FCVAL_HELPER(mview,i,j,k,GXX_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GXY_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GXZ_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GYY_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GYZ_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GZZ_,q,idir)         \
      }                                                         \
    , {                                                         \
          COMPUTE_FCVAL_HELPER(mview,i,j,k,BETAX_,q,idir)       \
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,BETAY_,q,idir)       \
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,BETAZ_,q,idir)       \
      }                                                         \
    , COMPUTE_FCVAL_HELPER(mview,i,j,k,ALP_,q,idir)             \
}
#else
#define COMPUTE_FCVAL_HELPER(mview,i,j,ivar,q,idir)                   \
  AM2*mview(i-2*utils::delta(0,idir),j-2*utils::delta(1,idir),ivar,q) \
+ AM1*mview(i-utils::delta(0,idir),j-utils::delta(1,idir),ivar,q)     \
+ A0*mview(i,j,ivar,q)                                                \
+ A1*mview(i+utils::delta(0,idir),j+utils::delta(1,idir),ivar,q)        
#define COMPUTE_FCVAL(g,mview,i,j,q,idir)                     \
g = grace::metric_array_t{                                    \
      {                                                       \
          COMPUTE_FCVAL_HELPER(mview,i,j,GXX_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,GXY_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,GXZ_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,GYY_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,GYZ_,q,idir)         \
        , COMPUTE_FCVAL_HELPER(mview,i,j,GZZ_,q,idir)         \
      }                                                       \
    , {                                                       \
          COMPUTE_FCVAL_HELPER(mview,i,j,BETAX_,q,idir)       \
        , COMPUTE_FCVAL_HELPER(mview,i,j,BETAY_,q,idir)       \
        , COMPUTE_FCVAL_HELPER(mview,i,j,BETAZ_,q,idir)       \
      }                                                       \
    , COMPUTE_FCVAL_HELPER(mview,i,j,ALP_,q,idir)             \
}
#endif 

struct grmhd_id_t {
  double rho;
  double press;
  double ye;
  double gxx,gxy,gxz,gyy,gyz,gzz; 
  double kxx,kxy,kxz,kyy,kyz,kzz;
  double alp;  
  double betax, betay, betaz ; 
  double vx, vy, vz;
} ; 

#endif /* GRACE_PHYSICS_GRMHD_HELPERS_HH */