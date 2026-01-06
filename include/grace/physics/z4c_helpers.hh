/**
 * @file bssn_helpers.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-12-21
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

#ifndef GRACE_PHYSICS_Z4C_HELPERS_HH 
#define GRACE_PHYSICS_Z4C_HELPERS_HH

#include <grace_config.h> 

#include <grace/data_structures/variable_indices.hh>
#include <grace/system/print.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/utils/fd_utils.hh>

#include <array>

namespace grace {

static void GRACE_HOST_DEVICE
adm_to_z4c(
    grmhd_id_t const& id, 
    grace::var_array_t state,
    VEC(int i, int j, int k), int q
)
{
    std::array<double,6> const __g {
        id.gxx,id.gxy,id.gxz,id.gyy,id.gyz,id.gzz
    } ;
    std::array<double,3> const __beta {
        id.betax,id.betay,id.betaz 
    } ; 
    double const __alp {id.alp} ; 
    metric_array_t adm_metric {
        __g,__beta,__alp
    } ; 

    std::array<double,6> const __Kij {
        id.kxx,id.kxy,id.kxz,id.kyy,id.kyz,id.kzz
    } ; 

    double const sqrtgamma = adm_metric.sqrtg(); 
    double const one_over_cbrtgamma = 1./Kokkos::cbrt(math::int_pow<2>(sqrtgamma)) ;

    double const chi  = one_over_cbrtgamma ; 

    #pragma unroll 6 
    for( int icomp=0; icomp<6; ++icomp ) {
        state(VEC(i,j,k),GTXX_+icomp,q) = chi * __g[icomp] ; 
    }
    
    // Compute trace of extrinsic curvature 
    double const K =  adm_metric.trace_sym2tens_lower(__Kij) ;  //FIXME LINEAR GW 

    #pragma unroll 6
    for( int icomp=0; icomp<6; ++icomp ) {
        state(VEC(i,j,k),ATXX_+icomp,q) = chi * (__Kij[icomp] - 1./3. * __g[icomp] * K) ; 
    }

    state(VEC(i,j,k),CHI_  ,q) = chi ; 
    state(VEC(i,j,k),KHAT_ ,q) = K   ; 
    state(VEC(i,j,k),ALP_  ,q) = id.alp ; 
    state(VEC(i,j,k),BETAX_,q) = id.betax ; 
    state(VEC(i,j,k),BETAY_,q) = id.betay ; 
    state(VEC(i,j,k),BETAZ_,q) = id.betaz ; 
    // Theta is initially zero 
    state(VEC(i,j,k),THETA_,q) = 0.0 ; 

}

template< size_t der_order >
static void GRACE_HOST_DEVICE
compute_gamma_tilde(
    grace::var_array_t state,
    VEC(int i, int j, int k), int q, std::array<double,GRACE_NSPACEDIM> const& idx
    , VEC(int nx, int ny, int nz), int ngz
)
{
    using namespace grace ; 
    using namespace utils ; 

    double const gtxx = state(VEC(i,j,k),GTXX_+0,q);
    double const gtxy = state(VEC(i,j,k),GTXX_+1,q);
    double const gtxz = state(VEC(i,j,k),GTXX_+2,q);
    double const gtyy = state(VEC(i,j,k),GTXX_+3,q);
    double const gtyz = state(VEC(i,j,k),GTXX_+4,q);
    double const gtzz = state(VEC(i,j,k),GTXX_+5,q);

    double const gtxxdx = grace::fd_der_bnd_check<der_order,0>(state,GTXX_+0, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[0 ];
    double const gtxxdy = grace::fd_der_bnd_check<der_order,1>(state,GTXX_+0, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[1 ];
    double const gtxxdz = grace::fd_der_bnd_check<der_order,2>(state,GTXX_+0, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[2 ];
    double const gtxydx = grace::fd_der_bnd_check<der_order,0>(state,GTXX_+1, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[0 ];
    double const gtxydy = grace::fd_der_bnd_check<der_order,1>(state,GTXX_+1, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[1 ];
    double const gtxydz = grace::fd_der_bnd_check<der_order,2>(state,GTXX_+1, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[2 ];
    double const gtxzdx = grace::fd_der_bnd_check<der_order,0>(state,GTXX_+2, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[0 ];
    double const gtxzdy = grace::fd_der_bnd_check<der_order,1>(state,GTXX_+2, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[1 ];
    double const gtxzdz = grace::fd_der_bnd_check<der_order,2>(state,GTXX_+2, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[2 ];
    double const gtyydx = grace::fd_der_bnd_check<der_order,0>(state,GTXX_+3, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[0 ];
    double const gtyydy = grace::fd_der_bnd_check<der_order,1>(state,GTXX_+3, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[1 ];
    double const gtyydz = grace::fd_der_bnd_check<der_order,2>(state,GTXX_+3, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[2 ];
    double const gtyzdx = grace::fd_der_bnd_check<der_order,0>(state,GTXX_+4, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[0 ];
    double const gtyzdy = grace::fd_der_bnd_check<der_order,1>(state,GTXX_+4, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[1 ];
    double const gtyzdz = grace::fd_der_bnd_check<der_order,2>(state,GTXX_+4, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[2 ];
    double const gtzzdx = grace::fd_der_bnd_check<der_order,0>(state,GTXX_+5, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[0 ];
    double const gtzzdy = grace::fd_der_bnd_check<der_order,1>(state,GTXX_+5, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[1 ];
    double const gtzzdz = grace::fd_der_bnd_check<der_order,2>(state,GTXX_+5, VEC(i,j,k),q,VEC(nx,ny,nz),ngz) * idx[2 ];

    // \tilde{\Gamma}^i = - D_j \tilde{\gamma}^{ij}
    state(VEC(i,j,k),GAMMATX_+0, q) = gtxz*gtyydz - gtyz*(gtxydz + gtxzdy - 2*gtyzdx) - gtxz*gtyzdy - gtxy*gtyzdz + gtxydy*gtzz - gtyydx*gtzz + gtyy*(gtxzdz - gtzzdx) + gtxy*gtzzdy;
    state(VEC(i,j,k),GAMMATX_+1, q) = -(gtxy*gtxzdz) + (gtxxdz - gtxzdx)*gtyz - gtxz*(gtxydz - 2*gtxzdy + gtyzdx) + gtxx*gtyzdz - gtxxdy*gtzz + gtxydx*gtzz + gtxy*gtzzdx - gtxx*gtzzdy;
    state(VEC(i,j,k),GAMMATX_+2, q) = -(gtxydy*gtxz) + (-gtxxdz + gtxzdx)*gtyy + gtxz*gtyydx - gtxx*gtyydz + gtxxdy*gtyz - gtxydx*gtyz + gtxy*(2*gtxydz - gtxzdy - gtyzdx) + gtxx*gtyzdy;

}

}

#endif /* GRACE_PHYSICS_BSSN_HELPERS_HH */