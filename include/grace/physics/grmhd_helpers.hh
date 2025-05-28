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
#include <grace/utils/metric_utils.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/amr/grace_amr.hh>

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
    #ifdef GRACE_DO_MHD
    BXL,
    BYL,
    BZL,
    #ifdef GRACE_ENABLE_B_FIELD_GLM
    PHI_GLML,
    #endif 
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
    #ifdef GRACE_DO_MHD
    BGXL,
    BGYL,
    BGZL,
    #ifdef GRACE_ENABLE_B_FIELD_GLM
    PHIG_GLML,
    #endif 	
    #endif
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
primsarr[ENTL] = vview(__VA_ARGS__,ENTROPY_,q) ;   \
primsarr[BXL] = vview(__VA_ARGS__,BX_,q);         \
primsarr[BYL] = vview(__VA_ARGS__,BY_,q);         \
primsarr[BZL] = vview(__VA_ARGS__,BZ_,q) ;        \
primsarr[PHI_GLML] = vview(__VA_ARGS__,PHI_GLM_,q) 

#define FILL_PRIMS_ARRAY_ZVEC(primsarr,vview,q,...)        \
primsarr[RHOL] = vview(__VA_ARGS__,RHO_,q);      \
primsarr[PRESSL] = vview(__VA_ARGS__,PRESS_,q) ; \
primsarr[VXL] = vview(__VA_ARGS__,ZVECX_,q) ;     \
primsarr[VYL] = vview(__VA_ARGS__,ZVECY_,q) ;     \
primsarr[VZL] = vview(__VA_ARGS__,ZVECZ_,q) ;     \
primsarr[YEL] = vview(__VA_ARGS__,YE_,q) ;       \
primsarr[TEMPL] = vview(__VA_ARGS__,TEMP_,q) ;   \
primsarr[EPSL] = vview(__VA_ARGS__,EPS_,q) ;     \
primsarr[ENTL] = vview(__VA_ARGS__,ENTROPY_,q) ; \
primsarr[BXL] = vview(__VA_ARGS__,BX_,q) ;        \
primsarr[BYL] = vview(__VA_ARGS__,BY_,q) ;        \
primsarr[BZL] = vview(__VA_ARGS__,BZ_,q) ;       \
primsarr[PHI_GLML] = vview(__VA_ARGS__,PHI_GLM_,q) 


#define FILL_CONS_ARRAY(consarr, vview,q,...)      \
consarr[DENSL] = vview(__VA_ARGS__,DENS_,q);       \
consarr[TAUL] = vview(__VA_ARGS__,TAU_,q);         \
consarr[STXL] = vview(__VA_ARGS__,SX_,q);          \
consarr[STYL] = vview(__VA_ARGS__,SY_,q);          \
consarr[STZL] = vview(__VA_ARGS__,SZ_,q);          \
consarr[YESL] = vview(__VA_ARGS__,YESTAR_,q);      \
consarr[ENTSL] = vview(__VA_ARGS__,ENTROPYSTAR_,q) ;\
consarr[BGXL] = vview(__VA_ARGS__,BGX_,q) ;         \
consarr[BGYL] = vview(__VA_ARGS__,BGY_,q) ;         \
consarr[BGZL] = vview(__VA_ARGS__,BGZ_,q) ;         \
consarr[PHIG_GLML] = vview(__VA_ARGS__,PHIG_GLM_,q)  

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
    double const v2 = metric.square_vec({eulVel[0],eulVel[1],eulVel[2]}) ; 
    double const W  = 1./Kokkos::sqrt(1-v2) ;

    // g_munu b^mu u^mu = 0 !!!
    // Note left for clarity: 
    // It's the responsibility of the caller to make sure the orthogonality condition is satisfied 

    // assert(fabs(metric.contract_4dvec_4dcovec(smallb,
    //                                metric.lower_4vec({W*(normalvector[0]),
    //                                              W*(normalvector[1] + eulVel[0] ),
    //                                              W*(normalvector[2] + eulVel[1] ),
    //                                              W*(normalvector[3] + eulVel[2] )
    //                                }) )
    //                                  ) < 1.e-10 );
      
    auto smallbD       = metric.lower_4vec(smallb);
    auto n_dot_smallb  = metric.contract_4dvec_4dcovec(normalvector,smallbD);

    // follow (6.107) from Gourgoulhon's book (Springer Verlag)
    // only spatial components 
    // B^i = W b^i + (n*b) * u^i   [ u^mu = W (n^mu + U^mu) ]
    for(int i=0; i<3; i++){ 
        eulB[i] = W*smallb[i+1] + n_dot_smallb * W * (normalvector[i+1] + eulVel[i]);
    }
}


#ifdef GRACE_DO_MHD
#ifdef GRACE_ENABLE_B_FIELD_GLM

/**
 * @brief get_derivative
 * @tparam size_t fd_order finite differencing order
 * @tparam size_t dir direction of the derivative
 * @param idx array of inverse grid spacings
 * @param vars array of variables (could be state/aux)
 * @param in_var which variable is being differentiated
 * @param VEC(i,j,k) position
 * @param q quadrant number 
 * @note  helper function for computing finite-difference expressions in the routines below 
 * @returns fd-order derivative value at a point (double)
 */
template<size_t fd_order, size_t dir>
static double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
get_derivative(const grace::scalar_array_t<GRACE_NSPACEDIM>& idx,
                      const grace::var_array_t<GRACE_NSPACEDIM>& vars,
                       int in_var,
                       VEC(int i, int j, int k), int q){
    using namespace utils; 

    double derivative{0.}; 
    const double inv_dx = idx(dir,q);
  
    if constexpr(fd_order==2){
      derivative = ( vars(VEC(i+utils::delta(0,dir),j+utils::delta(1,dir),k+utils::delta(2,dir)), in_var,q) 
                  - vars(VEC(i-utils::delta(0,dir),j-utils::delta(1,dir),k-utils::delta(2,dir)), in_var,q)
                    ) * (1.0/2.0)  * inv_dx;
    }
    else if constexpr(fd_order==4){
      derivative = (vars(VEC(i-2*utils::delta(0,dir),j-2*utils::delta(1,dir),k-2*utils::delta(2,dir)), in_var,q) 
                  - 8.0 * vars(VEC(i-utils::delta(0,dir),j-utils::delta(1,dir),k-utils::delta(2,dir)), in_var,q)
                  + 8.0 * vars(VEC(i+utils::delta(0,dir),j+utils::delta(1,dir),k+utils::delta(2,dir)), in_var,q)
                  -vars(VEC(i+2*utils::delta(0,dir),j+2*utils::delta(1,dir),k+2*utils::delta(2,dir)), in_var,q) 
                    ) * (1.0/12.0)  * inv_dx;
    }
    else{
      static_assert(fd_order == 2 || fd_order == 4, "Unsupported finite difference order");
    }
    return derivative;
}



/**
 * @brief compute_B_field_from_Avec
 * @param state array of state variables 
 * @param aux array of auxiliary variables
 * @param in_var which variable is being differentiated
 * @note  computes the B as the curl of A in finite-order differencing
 * @warning for now, we make a working assumption that both the vector potential 
 *          and the magnetic field are defined at cell-centes;
 *          we opt for 2nd-order expressions
 * @warning this routine will have to change when a proper constraint-transport algorithm is implemented
 * @returns fd-order derivative value at a point (double)
 */
static void GRACE_HOST_DEVICE
compute_B_field_from_Avec(
    const grace::var_array_t<GRACE_NSPACEDIM>& state,
    grace::var_array_t<GRACE_NSPACEDIM>& aux, 
    const grace::scalar_array_t<GRACE_NSPACEDIM>& idx)
  {
    DECLARE_GRID_EXTENTS;
    using namespace grace  ;
    using namespace Kokkos ;

    constexpr int X=0;
    constexpr int Y=1;
    constexpr int Z=2;
    constexpr int FD_ORDER = 2 ; // set it to the number of ghostzones just for safety 
    // we initialize sqrtgamma * B^i = eps^ijk d_j A_k 
    // state contains the required metric components passed on from the id kernel already 
    // aux contains the AVEC components and the B-field components (to be filled)

    /*************************************************************************/
    /* loop fill everything in the interior points   */
    /*************************************************************************/
    auto policy = MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(ngz,ngz,ngz),0},{VEC(nx+1+ngz,ny+1+ngz,nz+1+ngz),nq}) ; 
    parallel_for( GRACE_EXECUTION_TAG("ID","magnetic_field_from_vector_potential_ID")
                , policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                  const double Bx_densitized =
                    get_derivative<FD_ORDER, Y>(idx, aux, AVECZ_, VEC(i,j,k), q) -
                    get_derivative<FD_ORDER, Z>(idx, aux, AVECY_, VEC(i,j,k), q);

                  const double By_densitized =
                    get_derivative<FD_ORDER, Z>(idx, aux, AVECX_, VEC(i,j,k), q) -
                    get_derivative<FD_ORDER, X>(idx, aux, AVECZ_, VEC(i,j,k), q);

                  const double Bz_densitized =
                    get_derivative<FD_ORDER, X>(idx, aux, AVECY_, VEC(i,j,k), q) -
                    get_derivative<FD_ORDER, Y>(idx, aux, AVECX_, VEC(i,j,k), q);

                  // Un-densitize
                  metric_array_t metric;
                  FILL_METRIC_ARRAY(metric, state, q, VEC(i,j,k));
                  const double SQRTG = metric.sqrtg();

                  aux(VEC(i,j,k), BX_, q) = Bx_densitized / SQRTG;
                  aux(VEC(i,j,k), BY_, q) = By_densitized / SQRTG;
                  aux(VEC(i,j,k), BZ_, q) = Bz_densitized / SQRTG;

                }
    );
}

/**
 * @brief compute_divB
 * @tparam fd_order order of finite differencing
 * @param state array of state variables 
 * @param aux array of auxiliary variables
 * @param VEC(i,j,k) position
 * @param q quadrant number 
 * @note  computes the divergence of B from the expression  
 *       \f$~\bigotimes_{i=1}^{N_d}~[a_i,b_i]\f$
 *       \ \div B = \frac{1}{\sqrt{\gamma}} \partial_i (\sqrt{\gamma}B^i)
 * @warning for now, an assumption is made that both the vector potential 
 *          and the magnetic field are defined at cell-centes;
 *          we opt for 2nd-order expressions also here 
 * @warning this routine will have to change when a proper constraint-transport algorithm is implemented;
 *          in particular, the expressions in divB and curlA have to be carefully crafted and match each other 
 *          to represent faithfully the notion of 'zero divergence' on a numerical level
 * @todo will this routine, combined with the one in compute_B_field_from_Avec, 
 *       yield divB = 0 for constraint-satisfying initial data? 
 * @returns divB value
 */
template <size_t fd_order> 
static decltype(auto) GRACE_HOST_DEVICE
compute_divB(
    const grace::var_array_t<GRACE_NSPACEDIM>& state,
    const grace::var_array_t<GRACE_NSPACEDIM>& aux,
    const grace::scalar_array_t<GRACE_NSPACEDIM>& idx,
    VEC(int i, int j, int k), int q)
    {

      using namespace grace  ;
      using namespace Kokkos ;
      // auto& idx   = grace::variable_list::get().getinvspacings()   ; 
      constexpr int X=0;
      constexpr int Y=1;
      constexpr int Z=2;
      // set it to the number of ghostzones just for safety 
      // we initialize sqrtgamma * B^i = eps^ijk d_j A_k 
      // state contains the required metric components passed on from the id kernel already 
      // state contains also the densitized B components 
      // aux contains the AVEC components and the B-field components (to be filled)

      //  divB = 1/sqrtgamma * \partial_i ( sqrtgamma B^i )
    
      const double dGBxdx = get_derivative<fd_order, X>(idx, state, BGX_, VEC(i,j,k), q);

      const double dGBydy = get_derivative<fd_order, Y>(idx, state, BGY_, VEC(i,j,k), q);

      const double dGBzdz = get_derivative<fd_order, Z>(idx, state, BGZ_, VEC(i,j,k), q);

      // Un-densitize
      metric_array_t metric;
      FILL_METRIC_ARRAY(metric, state, q, VEC(i,j,k));
      const double sqrtg = metric.sqrtg();

      const double divB  = (dGBxdx + dGBydy + dGBzdz) / sqrtg ;
      return divB ; 
}
#endif  // GRACE_ENABLE_B_FIELD_GLM
#endif  // GRACE_DO_MHD

struct grmhd_id_t {
  double rho;
  double press;
  double ye;
  double gxx,gxy,gxz,gyy,gyz,gzz; 
  double kxx,kxy,kxz,kyy,kyz,kzz;
  double alp;  
  double betax, betay, betaz ; 
  double vx, vy, vz;
  #ifdef GRACE_DO_MHD
  double ax, ay, az, phi_em;
  double bx, by, bz;
  #ifdef GRACE_ENABLE_B_FIELD_GLM
  double phi_glm ;
  #endif
  #endif
} ; 

#endif /* GRACE_PHYSICS_GRMHD_HELPERS_HH */
