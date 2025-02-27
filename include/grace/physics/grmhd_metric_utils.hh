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
#include <grace/utils/numerics/fd_utils.hh>
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

/**
 * @brief This function computes a derivative using 
 *        a 2nd order accurate stencil. Given data on
 *        cell corners, the result will be at the cell
 *        centers. 
 * 
 * @tparam idir Direction of the derivative
 * @param mview View containing the data 
 * @param ivar  Variable index
 * @param q     Quadrant index
 * @return double  The derivative 
 */
template<size_t idir> 
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE 
fd_der_corner_to_center(
  grace::var_array_t<GRACE_NSPACEDIM> mview, int ivar, VEC(int i, int j, int k), int q
) ; 

template<>
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
fd_der_corner_to_center<0>(
  grace::var_array_t<GRACE_NSPACEDIM> mview, int ivar, VEC(int i, int j, int k), int q
) {
  return (
      mview(VEC(i+1,j,k),ivar,q) - mview(VEC(i,j,k),ivar,q)
    + mview(VEC(i+1,j+1,k),ivar,q) - mview(VEC(i,j+1,k),ivar,q) 
    #ifdef GRACE_NSPACEDIM 
    + mview(VEC(i+1,j,k+1),ivar,q) - mview(VEC(i,j,k+1),ivar,q) 
    + mview(VEC(i+1,j+1,k+1),ivar,q) - mview(VEC(i,j+1,k+1),ivar,q)
    #endif 
  ) * 1./((double)(1<<(GRACE_NSPACEDIM-1))) ;
}

template<>
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
fd_der_corner_to_center<1>(
  grace::var_array_t<GRACE_NSPACEDIM> mview, int ivar, VEC(int i, int j, int k), int q
) {
  return (
      mview(VEC(i,j+1,k),ivar,q) - mview(VEC(i,j,k),ivar,q)
    + mview(VEC(i+1,j+1,k),ivar,q) - mview(VEC(i+1,j,k),ivar,q) 
    #ifdef GRACE_NSPACEDIM 
    + mview(VEC(i,j+1,k+1),ivar,q) - mview(VEC(i,j,k+1),ivar,q) 
    + mview(VEC(i+1,j+1,k+1),ivar,q) - mview(VEC(i+1,j,k+1),ivar,q)
    #endif 
  ) * 1./((double)(1<<(GRACE_NSPACEDIM-1))) ;
}

template<>
double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
fd_der_corner_to_center<2>(
  grace::var_array_t<GRACE_NSPACEDIM> mview, int ivar, VEC(int i, int j, int k), int q
) {
  return (
      mview(VEC(i,j,k+1),ivar,q) - mview(VEC(i,j,k),ivar,q)
    + mview(VEC(i+1,j,k+1),ivar,q) - mview(VEC(i+1,j,k),ivar,q) 
    #ifdef GRACE_NSPACEDIM 
    + mview(VEC(i,j+1,k+1),ivar,q) - mview(VEC(i,j+1,k),ivar,q) 
    + mview(VEC(i+1,j+1,k+1),ivar,q) - mview(VEC(i+1,j+1,k),ivar,q)
    #endif 
  ) * 1./((double)(1<<(GRACE_NSPACEDIM-1))) ;
}

static GRACE_HOST_DEVICE 
void fill_metric_derivatives(
  grace::var_array_t<GRACE_NSPACEDIM> state, VEC(int i, int j, int k), int q,
  std::array<double,3>& dalp_dx,
  std::array<double,10>& dgmunu_dx1,
  std::array<double,10>& dgmunu_dx2,
  std::array<double,10>& dgmunu_dx3,
  std::array<double,3> const& idx
)
{
  auto read_at_cell_center = [&] (int ivar) {
    double out = 0 ; 
    EXPR(for(int ix=0; ix<2;++ix), for(int iy=0; iy<2; ++iy), for( int iz=0; iz<2; ++iz)) {
      out += state(VEC(i+ix,j+iy,k+iz),ivar,q) ;
    }
    return out / ((double)(1<<GRACE_NSPACEDIM)) ;
  } ; 
  double const gtxx = read_at_cell_center(GTXX_+0) ; 
  double const gtxy = read_at_cell_center(GTXX_+1) ; 
  double const gtxz = read_at_cell_center(GTXX_+2) ; 
  double const gtyy = read_at_cell_center(GTXX_+3) ; 
  double const gtyz = read_at_cell_center(GTXX_+4) ; 
  double const gtzz = read_at_cell_center(GTXX_+5) ; 

  double const phi = read_at_cell_center(PHI_) ; 

  double const alp = read_at_cell_center(ALP_) ; 

  double const betaX = read_at_cell_center(BETAX_+0) ; 
  double const betaY = read_at_cell_center(BETAX_+1) ; 
  double const betaZ = read_at_cell_center(BETAX_+2) ; 
  double const betax = (betaX*gtxx + betaY*gtxy + betaZ*gtxz)*exp(4.*phi);
  double const betay = (betaX*gtxy + betaY*gtyy + betaZ*gtyz)*exp(4.*phi);
  double const betaz = (betaX*gtxz + betaY*gtyz + betaZ*gtzz)*exp(4.*phi);

  double const gtXX=-(gtyz*gtyz) + gtyy*gtzz;
  double const gtXY=gtxz*gtyz - gtxy*gtzz;
  double const gtXZ=-(gtxz*gtyy) + gtxy*gtyz;
  double const gtYY=-(gtxz*gtxz) + gtxx*gtzz;
  double const gtYZ=gtxy*gtxz - gtxx*gtyz;
  double const gtZZ=-(gtxy*gtxy) + gtxx*gtyy;

  double const gtxxdx = fd_der_corner_to_center<0>(state,GTXX_+0, VEC(i,j,k),q) * idx[0 ];
  double const gtxxdy = fd_der_corner_to_center<1>(state,GTXX_+0, VEC(i,j,k),q) * idx[1 ];
  double const gtxxdz = fd_der_corner_to_center<2>(state,GTXX_+0, VEC(i,j,k),q) * idx[2 ];
  double const gtxydx = fd_der_corner_to_center<0>(state,GTXX_+1, VEC(i,j,k),q) * idx[0 ];
  double const gtxydy = fd_der_corner_to_center<1>(state,GTXX_+1, VEC(i,j,k),q) * idx[1 ];
  double const gtxydz = fd_der_corner_to_center<2>(state,GTXX_+1, VEC(i,j,k),q) * idx[2 ];
  double const gtxzdx = fd_der_corner_to_center<0>(state,GTXX_+2, VEC(i,j,k),q) * idx[0 ];
  double const gtxzdy = fd_der_corner_to_center<1>(state,GTXX_+2, VEC(i,j,k),q) * idx[1 ];
  double const gtxzdz = fd_der_corner_to_center<2>(state,GTXX_+2, VEC(i,j,k),q) * idx[2 ];
  double const gtyydx = fd_der_corner_to_center<0>(state,GTXX_+3, VEC(i,j,k),q) * idx[0 ];
  double const gtyydy = fd_der_corner_to_center<1>(state,GTXX_+3, VEC(i,j,k),q) * idx[1 ];
  double const gtyydz = fd_der_corner_to_center<2>(state,GTXX_+3, VEC(i,j,k),q) * idx[2 ];
  double const gtyzdx = fd_der_corner_to_center<0>(state,GTXX_+4, VEC(i,j,k),q) * idx[0 ];
  double const gtyzdy = fd_der_corner_to_center<1>(state,GTXX_+4, VEC(i,j,k),q) * idx[1 ];
  double const gtyzdz = fd_der_corner_to_center<2>(state,GTXX_+4, VEC(i,j,k),q) * idx[2 ];
  double const gtzzdx = fd_der_corner_to_center<0>(state,GTXX_+5, VEC(i,j,k),q) * idx[0 ];
  double const gtzzdy = fd_der_corner_to_center<1>(state,GTXX_+5, VEC(i,j,k),q) * idx[1 ];
  double const gtzzdz = fd_der_corner_to_center<2>(state,GTXX_+5, VEC(i,j,k),q) * idx[2 ];

  double const phidx = fd_der_corner_to_center<0>(state,PHI_,VEC(i,j,k),q) * idx[0 ];
  double const phidy = fd_der_corner_to_center<1>(state,PHI_,VEC(i,j,k),q) * idx[1 ];
  double const phidz = fd_der_corner_to_center<2>(state,PHI_,VEC(i,j,k),q) * idx[2 ];

  double const alpdx = fd_der_corner_to_center<0>(state,ALP_,VEC(i,j,k),q) * idx[0 ];
  double const alpdy = fd_der_corner_to_center<1>(state,ALP_,VEC(i,j,k),q) * idx[1 ];
  double const alpdz = fd_der_corner_to_center<2>(state,ALP_,VEC(i,j,k),q) * idx[2 ];

  double const betaXdx = fd_der_corner_to_center<0>(state,BETAX_+0, VEC(i,j,k),q)* idx[0 ];
  double const betaXdy = fd_der_corner_to_center<1>(state,BETAX_+0, VEC(i,j,k),q)* idx[1 ];
  double const betaXdz = fd_der_corner_to_center<2>(state,BETAX_+0, VEC(i,j,k),q)* idx[2 ];
  double const betaYdx = fd_der_corner_to_center<0>(state,BETAX_+1, VEC(i,j,k),q)* idx[0 ];
  double const betaYdy = fd_der_corner_to_center<1>(state,BETAX_+1, VEC(i,j,k),q)* idx[1 ];
  double const betaYdz = fd_der_corner_to_center<2>(state,BETAX_+1, VEC(i,j,k),q)* idx[2 ];
  double const betaZdx = fd_der_corner_to_center<0>(state,BETAX_+2, VEC(i,j,k),q)* idx[0 ];
  double const betaZdy = fd_der_corner_to_center<1>(state,BETAX_+2, VEC(i,j,k),q)* idx[1 ];
  double const betaZdz = fd_der_corner_to_center<2>(state,BETAX_+2, VEC(i,j,k),q)* idx[2 ];
  
  double const betaxdx = (betaXdx*gtxx + betaX*gtxxdx + betaYdx*gtxy + betaY*gtxydx + betaZdx*gtxz + betaZ*gtxzdx + 4*(betaX*gtxx + betaY*gtxy + betaZ*gtxz)*phidx)*exp(4.*phi);
  double const betaxdy = (betaXdy*gtxx + betaX*gtxxdy + betaYdy*gtxy + betaY*gtxydy + betaZdy*gtxz + betaZ*gtxzdy + 4*(betaX*gtxx + betaY*gtxy + betaZ*gtxz)*phidy)*exp(4.*phi);
  double const betaxdz = (betaXdz*gtxx + betaX*gtxxdz + betaYdz*gtxy + betaY*gtxydz + betaZdz*gtxz + betaZ*gtxzdz + 4*(betaX*gtxx + betaY*gtxy + betaZ*gtxz)*phidz)*exp(4.*phi);
  double const betaydx = (betaXdx*gtxy + betaX*gtxydx + betaYdx*gtyy + betaY*gtyydx + betaZdx*gtyz + betaZ*gtyzdx + 4*(betaX*gtxy + betaY*gtyy + betaZ*gtyz)*phidx)*exp(4.*phi);
  double const betaydy = (betaXdy*gtxy + betaX*gtxydy + betaYdy*gtyy + betaY*gtyydy + betaZdy*gtyz + betaZ*gtyzdy + 4*(betaX*gtxy + betaY*gtyy + betaZ*gtyz)*phidy)*exp(4.*phi);
  double const betaydz = (betaXdz*gtxy + betaX*gtxydz + betaYdz*gtyy + betaY*gtyydz + betaZdz*gtyz + betaZ*gtyzdz + 4*(betaX*gtxy + betaY*gtyy + betaZ*gtyz)*phidz)*exp(4.*phi);
  double const betazdx = (betaXdx*gtxz + betaX*gtxzdx + betaYdx*gtyz + betaY*gtyzdx + betaZdx*gtzz + betaZ*gtzzdx + 4*(betaX*gtxz + betaY*gtyz + betaZ*gtzz)*phidx)*exp(4.*phi);
  double const betazdy = (betaXdy*gtxz + betaX*gtxzdy + betaYdy*gtyz + betaY*gtyzdy + betaZdy*gtzz + betaZ*gtzzdy + 4*(betaX*gtxz + betaY*gtyz + betaZ*gtzz)*phidy)*exp(4.*phi);
  double const betazdz = (betaXdz*gtxz + betaX*gtxzdz + betaYdz*gtyz + betaY*gtyzdz + betaZdz*gtzz + betaZ*gtzzdz + 4*(betaX*gtxz + betaY*gtyz + betaZ*gtzz)*phidz)*exp(4.*phi);

  double const gttdx = -2*alp*alpdx + betaX*betaxdx + betax*betaXdx + betaY*betaydx + betay*betaYdx + betaZ*betazdx + betaz*betaZdx;
  double const gttdy = -2*alp*alpdy + betaX*betaxdy + betax*betaXdy + betaY*betaydy + betay*betaYdy + betaZ*betazdy + betaz*betaZdy;
  double const gttdz = -2*alp*alpdz + betaX*betaxdz + betax*betaXdz + betaY*betaydz + betay*betaYdz + betaZ*betazdz + betaz*betaZdz;
  double const gtxdx = betaxdx;
  double const gtxdy = betaxdy;
  double const gtxdz = betaxdz;
  double const gtydx = betaydx;
  double const gtydy = betaydy;
  double const gtydz = betaydz;
  double const gtzdx = betazdx;
  double const gtzdy = betazdy;
  double const gtzdz = betazdz;
  double const gxxdx = (gtxxdx + 4*gtxx*phidx)*exp(4.*phi);
  double const gxxdy = (gtxxdy + 4*gtxx*phidy)*exp(4.*phi);
  double const gxxdz = (gtxxdz + 4*gtxx*phidz)*exp(4.*phi);
  double const gxydx = (gtxydx + 4*gtxy*phidx)*exp(4.*phi);
  double const gxydy = (gtxydy + 4*gtxy*phidy)*exp(4.*phi);
  double const gxydz = (gtxydz + 4*gtxy*phidz)*exp(4.*phi);
  double const gxzdx = (gtxzdx + 4*gtxz*phidx)*exp(4.*phi);
  double const gxzdy = (gtxzdy + 4*gtxz*phidy)*exp(4.*phi);
  double const gxzdz = (gtxzdz + 4*gtxz*phidz)*exp(4.*phi);
  double const gyydx = (gtyydx + 4*gtyy*phidx)*exp(4.*phi);
  double const gyydy = (gtyydy + 4*gtyy*phidy)*exp(4.*phi);
  double const gyydz = (gtyydz + 4*gtyy*phidz)*exp(4.*phi);
  double const gyzdx = (gtyzdx + 4*gtyz*phidx)*exp(4.*phi);
  double const gyzdy = (gtyzdy + 4*gtyz*phidy)*exp(4.*phi);
  double const gyzdz = (gtyzdz + 4*gtyz*phidz)*exp(4.*phi);
  double const gzzdx = (gtzzdx + 4*gtzz*phidx)*exp(4.*phi);
  double const gzzdy = (gtzzdy + 4*gtzz*phidy)*exp(4.*phi);
  double const gzzdz = (gtzzdz + 4*gtzz*phidz)*exp(4.*phi);
  
  dalp_dx[0] = alpdx ; dalp_dx[1] = alpdy ; dalp_dx[2] = alpdz ;
  dgmunu_dx1[0] = gttdx ; dgmunu_dx1[1] = gtxdx ; dgmunu_dx1[2] = gtydx ;
  dgmunu_dx1[3] = gtzdx ; dgmunu_dx1[4] = gxxdx ; dgmunu_dx1[5] = gxydx ;
  dgmunu_dx1[6] = gxzdx ; dgmunu_dx1[7] = gyydx ; dgmunu_dx1[8] = gyzdx ;
  dgmunu_dx1[9] = gzzdx ;
  dgmunu_dx2[0] = gttdy ; dgmunu_dx2[1] = gtxdy ; dgmunu_dx2[2] = gtydy ;
  dgmunu_dx2[3] = gtzdy ; dgmunu_dx2[4] = gxxdy ; dgmunu_dx2[5] = gxydy ;
  dgmunu_dx2[6] = gxzdy ; dgmunu_dx2[7] = gyydy ; dgmunu_dx2[8] = gyzdy ;
  dgmunu_dx2[9] = gzzdy ;
  dgmunu_dx3[0] = gttdz ; dgmunu_dx3[1] = gtxdz ; dgmunu_dx3[2] = gtydz ;
  dgmunu_dx3[3] = gtzdz ; dgmunu_dx3[4] = gxxdz ; dgmunu_dx3[5] = gxydz ;
  dgmunu_dx3[6] = gxzdz ; dgmunu_dx3[7] = gyydz ; dgmunu_dx3[8] = gyzdz ;
  dgmunu_dx3[9] = gzzdz ;

}

#ifdef GRACE_ENABLE_COWLING_METRIC
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
#elif defined(GRACE_ENABLE_BSSN_METRIC)
#define FILL_METRIC_ARRAY(g, view, q, ...)                    \
g = grace::metric_array_t{  { view(__VA_ARGS__,GXX_,q)   \
                          , view(__VA_ARGS__,GXY_,q)     \
                          , view(__VA_ARGS__,GXZ_,q)     \
                          , view(__VA_ARGS__,GYY_,q)     \
                          , view(__VA_ARGS__,GYZ_,q)     \
                          , view(__VA_ARGS__,GZZ_,q) }   \
                          , { view(__VA_ARGS__,BETAXC_,q)   \
                            , view(__VA_ARGS__,BETAYC_,q)   \
                            , view(__VA_ARGS__,BETAZC_,q) } \
                            , view(__VA_ARGS__,ALPC_,q) } 
#endif 
#endif /* GRACE_PHYSICS_GRMHD_METRIC_UTILS_HH */