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

#include <grace/data_structures/variable_properties.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/utils/numerics/metric_utils.hh>
#include <grace/utils/numerics/grid_transfer.hh>
#include <Kokkos_Core.hpp>

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
    //#ifdef GRACE_DO_MHD
    BXL,
    BYL,
    BZL,
    //#endif 
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
    BGXL,
    BGYL,
    BGZL,
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
primsarr[ENTL] = vview(__VA_ARGS__,ENTROPY_,q) ;  \
primsarr[BXL] = vview(__VA_ARGS__,BX_,q);         \
primsarr[BYL] = vview(__VA_ARGS__,BY_,q);         \
primsarr[BZL] = vview(__VA_ARGS__,BZ_,q) ;

#define FILL_PRIMS_ARRAY_ZVEC(primsarr,vview,q,...)        \
primsarr[RHOL] = vview(__VA_ARGS__,RHO_,q);      \
primsarr[PRESSL] = vview(__VA_ARGS__,PRESS_,q) ; \
primsarr[VXL] = vview(__VA_ARGS__,ZVECX_,q) ;     \
primsarr[VYL] = vview(__VA_ARGS__,ZVECY_,q) ;     \
primsarr[VZL] = vview(__VA_ARGS__,ZVECZ_,q) ;     \
primsarr[YEL] = vview(__VA_ARGS__,YE_,q) ;       \
primsarr[TEMPL] = vview(__VA_ARGS__,TEMP_,q) ;   \
primsarr[EPSL] = vview(__VA_ARGS__,EPS_,q) ;     \
primsarr[ENTL] = vview(__VA_ARGS__,ENTROPY_,q)   \
primsarr[BXL] = vview(__VA_ARGS__,BX_,q) ;        \
primsarr[BYL] = vview(__VA_ARGS__,BY_,q) ;        \
primsarr[BZL] = vview(__VA_ARGS__,BZ_,q) ;

#define FILL_CONS_ARRAY(consarr, vview,q,...)      \
consarr[DENSL] = vview(__VA_ARGS__,DENS_,q);       \
consarr[TAUL] = vview(__VA_ARGS__,TAU_,q);         \
consarr[STXL] = vview(__VA_ARGS__,SX_,q);          \
consarr[STYL] = vview(__VA_ARGS__,SY_,q);          \
consarr[STZL] = vview(__VA_ARGS__,SZ_,q);          \
consarr[YESL] = vview(__VA_ARGS__,YESTAR_,q);      \
consarr[ENTSL] = vview(__VA_ARGS__,ENTROPYSTAR_,q); \
consarr[BGXL] = vview(__VA_ARGS__,BGX_,q) ;         \
consarr[BGYL] = vview(__VA_ARGS__,BGY_,q) ;         \
consarr[BGZL] = vview(__VA_ARGS__,BGZ_,q) ;         

/**
 * @brief Convert z^i to v^i
 * 
 * @param metric A `metric_array_t`
 * @param [inout] zx The x component of zvec
 * @param [inout] zy The y component of zvec 
 * @param [inout] zz The z component of zvec 
 * @return double The lorentz factor
 */
static double GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
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


// return co-moving magnetic field b^\mu components
/**
 * @brief return co-moving magnetic field b^\mu components from the Eulerian B^i field and Eulerian 3-velocity U^i
 * 
 * @param metric A `metric_array_t`
 * @param [in] std::array<double,3> components of the Eulerian B^i field ([0,1,2]=[x,y,z])
 * @param [in] std::array<double,3> components of the Eulerian U^i field ([0,1,2]=[x,y,z])
 * @return [inout] std::array<double,4> components of the co-moving magnetic field
 */ 
static void GRACE_HOST_DEVICE
get_smallb_from_eulerianB(grace::metric_array_t const& metric,
                          const std::array<double,3>& eulB, 
                          const std::array<double,3>& eulVel,
                          std::array<double, 4>& smallbU){
    std::array<double,4> normalvector{1./metric.alp(),
                                        -metric.beta(0)/metric.alp(),
                                        -metric.beta(1)/metric.alp(),
                                        -metric.beta(2)/metric.alp()
                                        };

    auto eulVelD   = metric.lower(eulVel);
    auto VelTimesB  = metric.contract_vec_covec(eulVelD,eulB);

    double const v2 = metric.square_vec({eulVel[0],eulVel[1],eulVel[2]}) ; 
    double const W  = 1./Kokkos::sqrt(1-v2) ; 
    // follow (6.108) from Gourgoulhon's book (Springer Verlag)
    // time-like component
    smallbU[0] = VelTimesB * W * ( normalvector[0] );
    // spatial components 
    for(int i=1; i<4; i++){ 
        smallbU[i] = VelTimesB * W * (normalvector[i] + eulVel[i-1]) + (1./W) * eulB[i-1];
    }
}


/**
 * @brief return Eulerian magnetic field B^\i components from the co-moving b^\mu field and Eulerian 3-velocity U^i
 * 
 * @param metric A `metric_array_t`
 * @param [in] std::array<double,4> components of the co-movin b^\mu field ([0,1,2,3]=[t,x,y,z])
 * @param [in] std::array<double,3> components of the Eulerian U^i field ([0,1,2]=[x,y,z])
 * @return [inout] std::array<double,3> components of the Eulerian magnetic field
 * @details IMPORTANT: the transformation between frames here (forward and back) only makes sense 
 *                     if the smallb vector is 
 */ 
static void GRACE_HOST_DEVICE
get_eulerianB_from_smallb(grace::metric_array_t const& metric,
                          const std::array<double,4>& smallb, 
                          const std::array<double,3>& eulVel,
                          std::array<double, 3>& eulB){

    std::array<double,4> normalvector{1./metric.alp(),
                                        -metric.beta(0)/metric.alp(),
                                        -metric.beta(1)/metric.alp(),
                                        -metric.beta(2)/metric.alp()
                                        };

    auto eulVelD   = metric.lower(eulVel);
    //auto VelTimesB  = metric.contract_vec_covec(eulVelD,eulB);
    double const v2 = metric.square_vec({eulVel[0],eulVel[1],eulVel[2]}) ; 
    double const W  = 1./Kokkos::sqrt(1-v2) ;

    // g_munu b^mu u^mu = 0 !!!

    assert(fabs(metric.contract_4dvec_4dcovec(smallb,
                                   metric.lower_4vec({W*(normalvector[0]),
                                                 W*(normalvector[1] + eulVel[0] ),
                                                 W*(normalvector[2] + eulVel[1] ),
                                                 W*(normalvector[3] + eulVel[2] )
                                   }) )
                                     ) < 1.e-10 );
      
    auto smallbD       = metric.lower_4vec(smallb);
    auto n_dot_smallb  = metric.contract_4dvec_4dcovec(normalvector,smallbD);

    // follow (6.107) from Gourgoulhon's book (Springer Verlag)
    // only spatial components 
    // B^i = W b^i + (n*b) * u^i   [ u^mu = W (n^mu + U^mu) ]
    for(int i=0; i<3; i++){ 
        eulB[i] = W*smallb[i+1] + n_dot_smallb * W * (normalvector[i+1] + eulVel[i]);
    }
}


/**
 * @brief Get the primitive variables at a cell corner.
 * 
 * @param aux Auxiliary variable array
 * @param cstate Corner staggered variable array
 * @param metric Metric array
 * @param [out] W Lorentz factor
 * @param i x cell index 
 * @param j y cell index 
 * @param k z cell index
 * @param q quadrant index
 * @return grace::grmhd_prims_array_t primitive variables at the i-1/2,j-1/2,k-1/2 corner. 
 */
static grace::grmhd_prims_array_t GRACE_HOST_DEVICE 
get_primitives_cell_corner(
  grace::var_array_t<GRACE_NSPACEDIM> const& aux,
  grace::var_array_t<GRACE_NSPACEDIM> const& cstate,
  grace::metric_array_t const& metric,
  double& W,
  VEC(int i, int j, int k),
  int64_t q
)
{
  using namespace Kokkos ;
  using namespace grace  ; 

  std::array<int, NUM_PRIMS_LOC> prim_idx {
    RHO_,PRESS_,ZVECX_,ZVECY_,ZVECZ_,YE_,TEMP_,EPS_,ENTROPY_,
    BX_,BY_,BZ_
  } ; 

  grace::grmhd_prims_array_t prims ; 
  int _idx = 0 ;
  for( auto const& ivar: prim_idx ) {
    auto sview = subview(aux,VEC(ALL(),ALL(),ALL()),ivar,q) ; 
    // interpolate 
    prims[_idx] = center_to_corner<2>::interpolate(sview,VEC(i,j,k)) ; 
    ++_idx ; 
  }
  
  // convert zvec to vel 
  W = zvec_to_vel(metric,prims[VXL],prims[VYL],prims[VZL]) ; 
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