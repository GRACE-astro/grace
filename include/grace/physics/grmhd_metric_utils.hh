
/**
 * @file grmhd_metric_utils.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-09-20
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

#ifndef GRACE_PHYSICS_GRMHD_METRIC_UTILS_HH
#define GRACE_PHYSICS_GRMHD_METRIC_UTILS_HH

#include <grace_config.h> 
#include <array>

#include <grace/utils/metric_utils.hh>
#include <grace/utils/inline.h>
// #include <grace/utils/device/device.h>
// #include <grace/utils/prolongation.hh>
// #include <grace/utils/lagrange_interpolators.hh>


#define AM2 -0.0625
#define AM1  0.5625
#define A0   0.5625
#define A1  -0.0625
#define COMPUTE_FCVAL_HELPER(mview,i,j,k,ivar,q,idir)                                               \
  AM2*mview(VEC(i-2*utils::delta(0,idir),j-2*utils::delta(1,idir),k-2*utils::delta(2,idir)),ivar,q) \
+ AM1*mview(VEC(i-utils::delta(0,idir),j-utils::delta(1,idir),k-utils::delta(2,idir)),ivar,q)       \
+ A0*mview(VEC(i,j,k),ivar,q)                                                                       \
+ A1*mview(VEC(i+utils::delta(0,idir),j+utils::delta(1,idir),k+utils::delta(2,idir)),ivar,q)             

// to be used in sources computation
#define BM2 (1./12.)
#define BM1 (-8./12.)
#define BP1 (8./12.)
#define BP2 (-1./12.)
#define COMPUTE_DERIV_4TH_ORDER_HELPER(mview,i,j,k,ivar,q,idir) \
  (BM2*(mview(VEC(i-2*utils::delta(0,idir),j-2*utils::delta(1,idir),k-2*utils::delta(2,idir)),ivar,q)) \
+  BM1*(mview(VEC(i-1*utils::delta(0,idir),j-1*utils::delta(1,idir),k-1*utils::delta(2,idir)),ivar,q)) \
+  BP1*(mview(VEC(i+1*utils::delta(0,idir),j+1*utils::delta(1,idir),k+1*utils::delta(2,idir)),ivar,q)) \ 
+  BP2*(mview(VEC(i+2*utils::delta(0,idir),j+2*utils::delta(1,idir),k+2*utils::delta(2,idir)),ivar,q)))

#define COMPUTE_DERIV_4TH_ORDER_HELPER_SIMPLE(XM2,XM1,XP1,XP2) \
  (BM2*XM2 + BM1*XM1 + BP1*XP1 + BP2*XP2)

namespace grace{

grace::metric_array_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
get_metric_array_cell_corner(
    grace::var_array_t const& state,
    VEC(int i, int j, int k),
    int64_t q
)
{
    #ifdef GRACE_ENABLE_COWLING_METRIC
    // need to interpolate
    #else
    return grace::metric_array_t {
        {
            state(VEC(i,j,k),GTXX_,q),
            state(VEC(i,j,k),GTXY_,q),
            state(VEC(i,j,k),GTXZ_,q),
            state(VEC(i,j,k),GTYY_,q),
            state(VEC(i,j,k),GTYZ_,q),
            state(VEC(i,j,k),GTZZ_,q)
        }, 
        state(VEC(i,j,k),PHI_,q), 
        {
            state(VEC(i,j,k),BETAX_,q),
            state(VEC(i,j,k),BETAY_,q),
            state(VEC(i,j,k),BETAZ_,q)  
        }, 
        state(VEC(i,j,k),ALP_,q)
    } ; 
    #endif 
}
grace::metric_array_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
get_metric_array_cell_center(
    grace::var_array_t const& state,
    VEC(int i, int j, int k),
    int64_t q
) 
{
    using namespace Kokkos ; 
    #ifdef GRACE_ENABLE_COWLING_METRIC
    return grace::metric_array_t {
        {state(VEC(i,j,k), GXX_, q), 
         state(VEC(i,j,k), GXY_, q),
         state(VEC(i,j,k), GXZ_, q),
         state(VEC(i,j,k), GYY_, q),
         state(VEC(i,j,k), GYZ_, q),
         state(VEC(i,j,k), GZZ_, q)},
        {state(VEC(i,j,k), BETAX_, q),
         state(VEC(i,j,k), BETAY_, q),
         state(VEC(i,j,k), BETAZ_, q)},
        state(VEC(i,j,k), ALP_, q)
    } ; 
    #elif defined(GRACE_ENABLE_BSSN_METRIC)
    /***************************************************************************************/
    std::array<int,6> gamma_indices {
        GTXX_,GTXY_,GTXZ_,GTYY_,GTYZ_,GTZZ_
    } ; 
    std::array<double,6> gamma ; 
    std::array<double,3> beta  ; 
    double alpha, phi ; 
    /***************************************************************************************/
    auto sview_phi = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), PHI_, q) ; 
    // Get phi at cell center 
    phi = sview_phi(VEC(i,j,k)) ;
    /***************************************************************************************/
    #pragma unroll 6 
    for( int i=0; i<6; ++i ) {
        auto sview = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), gamma_indices[i], q) ; 
        // Get gamma_{ij} component at cell center 
        gamma[i] = POW_CONFFACT(phi) * sview(VEC(i,j,k)) ; 
    }
    /***************************************************************************************/
    std::array<int,3> beta_indices {
        BETAX_,BETAY_,BETAZ_
    } ;
    #pragma unroll 3 
    for( int i=0; i<3; ++i ) {
        auto sview = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), beta_indices[i], q) ; 
        // Get beta^i component at cell center 
        beta[i] = sview(VEC(i,j,k)) ; 
    }
    /***************************************************************************************/
    auto sview_alp = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), ALP_, q) ; 
    // Get alp at cell center 
    alpha = sview_alp(VEC(i,j,k)) ;
    /***************************************************************************************/
    // Return the metric object 
    return grace::metric_array_t {
        gamma, phi, beta, alpha 
    } ; 
    #endif 
}

template< size_t idir >
grace::metric_array_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
get_metric_array_cell_face(
    grace::var_array_t const& state,
    VEC(int i, int j, int k),
    int64_t q
) 
{
    using namespace Kokkos ;  
    #ifdef GRACE_ENABLE_COWLING_METRIC
    return grace::metric_array_t{                                    
      {                                                         
          COMPUTE_FCVAL_HELPER(state,i,j,k,GXX_,q,idir)         
        , COMPUTE_FCVAL_HELPER(state,i,j,k,GXY_,q,idir)         
        , COMPUTE_FCVAL_HELPER(state,i,j,k,GXZ_,q,idir)         
        , COMPUTE_FCVAL_HELPER(state,i,j,k,GYY_,q,idir)         
        , COMPUTE_FCVAL_HELPER(state,i,j,k,GYZ_,q,idir)         
        , COMPUTE_FCVAL_HELPER(state,i,j,k,GZZ_,q,idir)         
      }                                                         
    , {                                                         
          COMPUTE_FCVAL_HELPER(state,i,j,k,BETAX_,q,idir)       
        , COMPUTE_FCVAL_HELPER(state,i,j,k,BETAY_,q,idir)       
        , COMPUTE_FCVAL_HELPER(state,i,j,k,BETAZ_,q,idir)       
      }                                                         
    , COMPUTE_FCVAL_HELPER(state,i,j,k,ALP_,q,idir)             
    };
    #elif defined(GRACE_ENABLE_BSSN_METRIC)
    static constexpr std::array<std::array<int,2>,3> dirs = {
        std::array<int,2>{1,2},
        std::array<int,2>{0,2},
        std::array<int,2>{0,1}
    } ;
    std::array<int,6> gammat_indices {
        GTXX_,GTXY_,GTXZ_,GTYY_,GTYZ_,GTZZ_
    } ; 
    std::array<int,3> beta_indices {
        BETAX_,BETAY_,BETAZ_
    } ;
    std::array<double,6> gamma ; 
    std::array<double,3> beta  ; 
    double alpha, phi ; 


    constexpr const size_t NPTS = 4 ; // points to interpolate with a 3-rd order expression

    // to recover the physical metric, we fill out the stencils first
    auto fill_stencil = [&](const grace::var_array_t& mview, int const& ivar){
            double UM2 = mview(VEC(i-2*utils::delta(0,idir),j-2*utils::delta(1,idir),k-2*utils::delta(2,idir)),ivar,q) ; 
            double UM1 = mview(VEC(i-utils::delta(0,idir),j-utils::delta(1,idir),k-utils::delta(2,idir)),ivar,q) ; 
            double U0  = mview(VEC(i,j,k),ivar,q);
            double UP1 = mview(VEC(i+utils::delta(0,idir),j+utils::delta(1,idir),k+utils::delta(2,idir)),ivar,q) ;
            std::array<double, NPTS> stencil_arr{UM2,UM1,U0,UP1};
            return stencil_arr;
    };

    auto weighted_sum = [](const std::array<double, NPTS>& arr){
        return AM2*arr[0] + AM1*arr[1] + A0*arr[2] + A1*arr[3];
    };


    std::array<double, NPTS> phi_stencil; 
    std::array<std::array<double,NPTS>, 6> gt_stencil; 
    std::array<std::array<double,NPTS>, 6> gp_stencil; 
    std::array<double, NPTS> alp_stencil; 
    std::array<std::array<double,NPTS>, 3> beta_stencil; 

    phi_stencil=fill_stencil(state, PHI_);
    alp_stencil=fill_stencil(state, ALP_);

    alpha = weighted_sum(alp_stencil);    

    for( int i=0; i<6; ++i ) {
       gt_stencil[i] =  fill_stencil(state, gammat_indices[i]);
    }

    for( int i=0; i<6; ++i ) {
        for (int j=0; j<NPTS; j++){
            gp_stencil[i][j] = POW_CONFFACT(phi_stencil[j]) * gt_stencil[i][j] ;
        }
        gamma[i] = weighted_sum(gp_stencil[i]);    
    }

    for( int i=0; i<3; ++i ) {
       beta_stencil[i] =  fill_stencil(state, beta_indices[i]);
       beta[i] = weighted_sum(beta_stencil[i]);    
    }

    return grace::metric_array_t {
        gamma, beta, alpha 
    } ; 
    #endif 
}

std::array<double,6> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
get_extrinsic_curvature_cell_center(
    grace::var_array_t const& state,
    VEC(int i,int j,int k),
    int64_t q,
    grace::metric_array_t const& gamma 
)
{
    using namespace Kokkos ; 
    #ifdef GRACE_ENABLE_COWLING_METRIC
    return std::array<double,6>{ 
              state(VEC(i,j,k),KXX_,q)
            , state(VEC(i,j,k),KXY_,q)
            , state(VEC(i,j,k),KXZ_,q)
            , state(VEC(i,j,k),KYY_,q)
            , state(VEC(i,j,k),KYZ_,q)
            , state (VEC(i,j,k),KZZ_,q)
        } ; 
    #elif defined(GRACE_ENABLE_BSSN_METRIC)
    // Get the trace at the cell center 
    auto sview = Kokkos::subview(state,VEC(ALL(),ALL(),ALL()), K_, q);
    double const K = sview(VEC(i,j,k)) ; 
    // Compute conformal factor from sqrtg 
    auto phi = SQRTG_TO_CONFFACT(gamma.sqrtg()) ;
    // Indices of conformal Aij
    std::array<int,6> A_indices {
        ATXX_,ATXY_,ATXZ_,ATYY_,ATYZ_,ATZZ_
    } ; 
    std::array<double,6> Kij ; 
    for( int ic=0; ic<6; ++ic ) {
        auto sview_A = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), A_indices[ic], q) ; 
        // Compute Kij = exp( 4 phi ) Aij + 1/3 gammaij K 
        Kij[ic] = POW_CONFFACT(phi)*(sview_A(VEC(i,j,k))) + 1./3. * gamma.gamma(ic) * K ; 
    }
     
    return Kij ; 
    #endif
}

grace::metric_array_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
get_metric_array(
    grace::var_array_t const& state,
    VEC(int i, int j, int k), int64_t q, 
    std::array<bool,GRACE_NSPACEDIM> stagger 
)
{
    auto& mview = state ; 

    // Decide what to do based on staggering 
    if ( EXPR((not stagger[0]), and (not stagger[1]), and (not stagger[2])) ) 
    {
        return get_metric_array_cell_center(mview,VEC(i,j,k),q) ;
    } else if (EXPR(( stagger[0]), and ( stagger[1]), and ( stagger[2])) ) {
        return get_metric_array_cell_corner(mview,VEC(i,j,k),q) ;
    } else if ( stagger[0] ) {
        return get_metric_array_cell_face<0>(mview,VEC(i,j,k),q) ;
    } else if ( stagger[1] ) {
        return get_metric_array_cell_face<1>(mview,VEC(i,j,k),q) ;
    } 
    #ifdef GRACE_3D 
    else if ( stagger[2] ) {
        return get_metric_array_cell_face<2>(mview,VEC(i,j,k),q) ;
    } 
    #endif 
    return grace::metric_array_t{
        {1,0,0,1,0,1}, {0,0,0}, 1
    } ; 
}

} /* namespace grace */

#endif /* GRACE_PHYSICS_GRMHD_METRIC_UTILS_HH */