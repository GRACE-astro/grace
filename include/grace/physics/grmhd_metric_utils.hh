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

#include <grace/utils/numerics/metric_utils.hh>
#include <grace/utils/inline.h>
#include <grace/utils/device/device.h>
#include <grace/utils/numerics/prolongation.hh>
#include <grace/utils/numerics/lagrange_interpolators.hh>

#define AM2 -0.0625
#define AM1  0.5625
#define A0   0.5625
#define A1  -0.0625
#define COMPUTE_FCVAL_HELPER(mview,i,j,k,ivar,q,idir)                                               \
  AM2*mview(VEC(i-2*utils::delta(0,idir),j-2*utils::delta(1,idir),k-2*utils::delta(2,idir)),ivar,q) \
+ AM1*mview(VEC(i-utils::delta(0,idir),j-utils::delta(1,idir),k-utils::delta(2,idir)),ivar,q)       \
+ A0*mview(VEC(i,j,k),ivar,q)                                                                       \
+ A1*mview(VEC(i+utils::delta(0,idir),j+utils::delta(1,idir),k+utils::delta(2,idir)),ivar,q)             

namespace grace{

grace::metric_array_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
get_metric_array_cell_center(
    grace::var_array_t<GRACE_NSPACEDIM> const& state,
    VEC(int i, int j, int k),
    int64_t q
) 
{
    using namespace Kokkos ; 
    #ifdef GRACE_ENABLE_COWLING_METRIC
    auto& state = grace::variable_list::get().getstate() ;
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
    using interp_t = utils::corner_staggered_lagrange_interp_t<2> ; 
    std::array<int,6> gamma_indices {
        GTXX_,GTXY_,GTXZ_,GTYY_,GTYZ_,GTZZ_
    } ; 
    std::array<double,6> gamma ; 
    std::array<double,3> beta  ; 
    double alpha, phi ; 
    for( int i=0; i<6; ++i ) {
        auto sview = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), gamma_indices[i], q) ; 
        gamma[i] = interp_t::threed_interp(sview, VEC(i,j,k)) ; 
    }
    std::array<int,3> beta_indices {
        BETAX_,BETAY_,BETAZ_
    } ;
    for( int i=0; i<3; ++i ) {
        auto sview = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), beta_indices[i], q) ; 
        beta[i] = interp_t::threed_interp(sview, VEC(i,j,k)) ; 
    }
    auto sview_alp = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), ALP_, q) ; 
    alpha = interp_t::threed_interp(sview_alp, VEC(i,j,k)) ;

    auto sview_phi = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), PHI_, q) ; 
    phi = interp_t::threed_interp(sview_phi, VEC(i,j,k)) ;

    return grace::metric_array_t {
        gamma, phi, beta, alpha 
    } ; 
    #endif 
}

template< size_t idir >
grace::metric_array_t GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE 
get_metric_array_cell_face(
    grace::var_array_t<GRACE_NSPACEDIM> const& state,
    VEC(int i, int j, int k),
    int64_t q
) 
{
    using namespace Kokkos ; 
    #ifdef GRACE_ENABLE_COWLING_METRIC
    return grace::metric_array_t{                                    
      {                                                         
          COMPUTE_FCVAL_HELPER(mview,i,j,k,GXX_,q,idir)         
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GXY_,q,idir)         
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GXZ_,q,idir)         
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GYY_,q,idir)         
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GYZ_,q,idir)         
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,GZZ_,q,idir)         
      }                                                         
    , {                                                         
          COMPUTE_FCVAL_HELPER(mview,i,j,k,BETAX_,q,idir)       
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,BETAY_,q,idir)       
        , COMPUTE_FCVAL_HELPER(mview,i,j,k,BETAZ_,q,idir)       
      }                                                         
    , COMPUTE_FCVAL_HELPER(mview,i,j,k,ALP_,q,idir)             
    };
    #elif defined(GRACE_ENABLE_BSSN_METRIC)
    static constexpr std::array<std::array<int,2>,3> dirs = {
        std::array<int,2>{1,2},
        std::array<int,2>{0,2},
        std::array<int,2>{0,1}
    } ;
    using interp_t = utils::corner_staggered_lagrange_interp_t<2> ; 
    std::array<int,6> gamma_indices {
        GTXX_,GTXY_,GTXZ_,GTYY_,GTYZ_,GTZZ_
    } ; 
    std::array<double,6> gamma ; 
    std::array<double,3> beta  ; 
    double alpha, phi ; 
    for( int i=0; i<6; ++i ) {
        auto sview = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), gamma_indices[i], q) ; 
        gamma[i] = interp_t::twod_interp<dirs[idir][0],dirs[idir][1]>(sview, VEC(i,j,k)) ; 
    }
    std::array<int,3> beta_indices {
        BETAX_,BETAY_,BETAZ_
    } ;
    for( int i=0; i<3; ++i ) {
        auto sview = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), beta_indices[i], q) ; 
        beta[i] = interp_t::twod_interp<dirs[idir][0],dirs[idir][1]>(sview, VEC(i,j,k)) ; 
    }
    auto sview_alp = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), ALP_, q) ; 
    alpha = interp_t::twod_interp<dirs[idir][0],dirs[idir][1]>(sview_alp, VEC(i,j,k)) ;

    auto sview_phi = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), PHI_, q) ; 
    phi = interp_t::twod_interp<dirs[idir][0],dirs[idir][1]>(sview_phi, VEC(i,j,k)) ;

    return grace::metric_array_t {
        gamma, phi, beta, alpha 
    } ; 
    #endif 
}

std::array<double,6> GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
get_extrinsic_curvature(
    grace::var_array_t<GRACE_NSPACEDIM> const& state,
    VEC(int i,int j,int k),
    int64_t q,
    grace::metric_array_t const& gamma 
)
{
    using namespace Kokkos ; 
    #ifdef GRACE_ENABLE_COWLING_METRIC
    return std::array<double,6>{ 
              this->_state(VEC(i,j,k),KXX_,q)
            , this->_state(VEC(i,j,k),KXY_,q)
            , this->_state(VEC(i,j,k),KXZ_,q)
            , this->_state(VEC(i,j,k),KYY_,q)
            , this->_state(VEC(i,j,k),KYZ_,q)
            , this->_state(VEC(i,j,k),KZZ_,q)
        } ; 
    #elif defined(GRACE_ENABLE_BSSN_METRIC)
    using interp_t = utils::corner_staggered_lagrange_interp_t<2> ; 
    auto sview = Kokkos::subview(state,VEC(ALL(),ALL(),ALL()), K_, q);
    double const K = interp_t::threed_interp(sview, VEC(i,j,k)) ; 
    auto phi = 1./Kokkos::cbrt(gamma.sqrtg()) ;

    std::array<int,6> A_indices {
        ATXX_,ATXY_,ATXZ_,ATYY_,ATYZ_,ATZZ_
    } ; 
    std::array<double,6> Kij ; 
    for( int i=0; i<6; ++i ) {
        auto sview = Kokkos::subview(state, VEC(ALL(),ALL(),ALL()), A_indices[i], q) ; 
        Kij[i] = interp_t::threed_interp(sview, VEC(i,j,k))/(phi*phi) + 1./3. * gamma.gamma(i) * K ; 
    }
     
    return Kij ; 
    #endif
}

} /* namespace grace */

#endif /* GRACE_PHYSICS_GRMHD_METRIC_UTILS_HH */