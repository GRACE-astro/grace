/**
 * @file bssn_helpers.hh
 * @author  ()
 * @brief 
 * @date 2024-09-03
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

#ifndef GRACE_PHYSICS_BSSN_HELPERS_HH 
#define GRACE_PHYSICS_BSSN_HELPERS_HH

#include <grace_config.h> 
#include <array>

namespace grace {


enum BSSN_VARENUM_t {
    PHIL=0,
    GTXXL,
    GTXYL,
    GTXZL, 
    GTYYL,
    GTYZL,
    GTZZL,
    ATXXL,
    ATXYL,
    ATXZL,
    ATYYL,
    ATYZL,
    ATZZL,
    KL,
    GAMMAXL,
    GAMMAYL,
    GAMMAZL,
    NUM_BSSN_VARS
} ; 

using bssn_state_t = std::array<double, NUM_BSSN_VARS> ;

#define FILL_BSSN_STATE(sstate, vview, q, ...)\
do{                                      \
sstate[PHIL] = vview(__VA_ARGS__, PHI_     , q); \
sstate[GTXXL] = vview(__VA_ARGS__, GTXX_   , q); \
sstate[GTXYL] = vview(__VA_ARGS__, GTXY_   , q); \
sstate[GTXZL] = vview(__VA_ARGS__, GTXZ_   , q); \
sstate[GTYYL] = vview(__VA_ARGS__, GTYY_   , q); \
sstate[GTYZL] = vview(__VA_ARGS__, GTYZ_   , q); \
sstate[GTZZL] = vview(__VA_ARGS__, GTZZ_   , q); \
sstate[ATXXL] = vview(__VA_ARGS__, ATXX_   , q); \
sstate[ATXYL] = vview(__VA_ARGS__, ATXY_   , q); \
sstate[ATXZL] = vview(__VA_ARGS__, ATXZ_   , q); \
sstate[ATYYL] = vview(__VA_ARGS__, ATYY_   , q); \
sstate[ATYZL] = vview(__VA_ARGS__, ATYZ_   , q); \
sstate[ATZZL] = vview(__VA_ARGS__, ATZZ_   , q); \
sstate[KL]    = vview(__VA_ARGS__, K_      , q); \
sstate[GAMMAXL] = vview(__VA_ARGS__,GAMMAX_, q); \
sstate[GAMMAYL] = vview(__VA_ARGS__,GAMMAY_, q); \
sstate[GAMMAZL] = vview(__VA_ARGS__,GAMMAZ_, q); \
} while(false)


void adm_to_bssn(
    grmhd_id_t const& id, 
    grace::var_array_t<GRACE_NSPACEDIM>& state,
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

    double const sqrtgamma = amd_metric.sqrtg(); 
    
    double const phi  = 1./(sqrtgamma*sqrtgamma*sqrtgamma) ; 
    double const phi2 = phi*phi ; 

    #pragma unroll 6 
    for( int icomp=0; icomp<6; ++icomp ) {
        state(VEC(i,j,k),GTXX_+icomp,q) = phi2 * __g[icomp] ; 
    }
    
    // Compute trace of extrinsic curvature 
    double const K = adm_metric.trace_sym2tens_lower(__Kij) ; 

    #pragma unroll 6
    for( int icomp=0; icomp<6; ++icomp ) {
        state(VEC(i,j,k),ATXX_+icomp,q) = phi2 * (__Kij[icomp] - 1./3. * __g[icomp] * K) ; 
    }

    state(VEC(i,j,k),PHI_,q) = phi ; 
    state(VEC(i,j,k),K_  ,q) = K   ; 
}

template< size_t der_order >
void compute_gamma_tilde(
    grace::var_array_t<GRACE_NSPACEDIM>& state,
    VEC(int i, int j, int k), int q, std::array<double,GRACE_NSPACEDIM? const& idx
)
{
    using namespace grace ; 
    using namespace utils ; 

    int ww = 0 ; 
    double const gxx=state(VEC(i,j,k),GXX_+ww,q); ww++;
    double const gxy=state(VEC(i,j,k),GXX_+ww,q); ww++;
    double const gxz=state(VEC(i,j,k),GXX_+ww,q); ww++;
    double const gyy=state(VEC(i,j,k),GXX_+ww,q); ww++;
    double const gyz=state(VEC(i,j,k),GXX_+ww,q); ww++;
    double const gzz=state(VEC(i,j,k),GXX_+ww,q); ww++;

    // Determinant of gamma tilde is 1
    double const gUxUx=-(gyz*gyz) + gyy*gzz;
    double const gUxUy=gxz*gyz - gxy*gzz;
    double const gUxUz=-(gxz*gyy) + gxy*gyz;
    double const gUyUy=-(gxz*gxz) + gxx*gzz;
    double const gUyUz=gxy*gxz - gxx*gyz;
    double const gUzUz=-(gxy*gxy) + gxx*gyy;

    double const dgxxdx= 
        grace::fd_der<der_order,0>(state,GTXX_+0,VEC(i,j,k),q)*idx[0];
    double const dgxxdy= 
        grace::fd_der<der_order,1>(state,GTXX_+0,VEC(i,j,k),q)*idx[1];
    double const dgxxdz= 
        grace::fd_der<der_order,2>(state,GTXX_+0,VEC(i,j,k),q)*idx[2];
    double const dgxydx= 
        grace::fd_der<der_order,0>(state,GTXX_+1,VEC(i,j,k),q)*idx[0];
    double const dgxydy= 
        grace::fd_der<der_order,1>(state,GTXX_+1,VEC(i,j,k),q)*idx[1];
    double const dgxydz= 
        grace::fd_der<der_order,2>(state,GTXX_+1,VEC(i,j,k),q)*idx[2];
    double const dgxzdx= 
        grace::fd_der<der_order,0>(state,GTXX_+2,VEC(i,j,k),q)*idx[0];
    double const dgxzdy= 
        grace::fd_der<der_order,1>(state,GTXX_+2,VEC(i,j,k),q)*idx[1];
    double const dgxzdz= 
        grace::fd_der<der_order,2>(state,GTXX_+2,VEC(i,j,k),q)*idx[2];
    double const dgyydx= 
        grace::fd_der<der_order,0>(state,GTXX_+3,VEC(i,j,k),q)*idx[0];
    double const dgyydy= 
        grace::fd_der<der_order,1>(state,GTXX_+3,VEC(i,j,k),q)*idx[1];
    double const dgyydz= 
        grace::fd_der<der_order,2>(state,GTXX_+3,VEC(i,j,k),q)*idx[2];
    double const dgyzdx= 
        grace::fd_der<der_order,0>(state,GTXX_+4,VEC(i,j,k),q)*idx[0];
    double const dgyzdy= 
        grace::fd_der<der_order,1>(state,GTXX_+4,VEC(i,j,k),q)*idx[1];
    double const dgyzdz= 
        grace::fd_der<der_order,2>(state,GTXX_+4,VEC(i,j,k),q)*idx[2];
    double const dgzzdx= 
        grace::fd_der<der_order,0>(state,GTXX_+5,VEC(i,j,k),q)*idx[0];
    double const dgzzdy= 
        grace::fd_der<der_order,1>(state,GTXX_+5,VEC(i,j,k),q)*idx[1];
    double const dgzzdz= 
        grace::fd_der<der_order,2>(state,GTXX_+5,VEC(i,j,k),q)*idx[2];

    ww = 0 ; 
    state(VEC(i,j,k),GAMMAX_+ww,q)=((gUxUx + gUxUy + 
        gUxUz)*(dgxxdx*gUxUx + 2*dgxxdy*gUxUy + 2*dgxxdz*gUxUz + (2*dgxydy - 
        dgyydx)*gUyUy + 2*(dgxydz + dgxzdy - dgyzdx)*gUyUz + (2*dgxzdz - 
        dgzzdx)*gUzUz))/2.; ww++;
    state(VEC(i,j,k),GAMMAX_+ww,q)=((gUxUy + gUyUy + gUyUz)*(-((dgxxdy - 
        2*dgxydx)*gUxUx) + 2*dgyydx*gUxUy + 2*(dgxydz - dgxzdy + 
        dgyzdx)*gUxUz + dgyydy*gUyUy + 2*dgyydz*gUyUz + (2*dgyzdz - 
        dgzzdy)*gUzUz))/2.; ww++;
    state(VEC(i,j,k),GAMMAX_+ww,q)=((gUxUz + gUyUz + gUzUz)*(-((dgxxdz - 
        2*dgxzdx)*gUxUx) + 2*(-dgxydz + dgxzdy + dgyzdx)*gUxUy + 
        2*dgzzdx*gUxUz - (dgyydz - 2*dgyzdy)*gUyUy + 2*dgzzdy*gUyUz + 
        dgzzd*gUzUz))/2.; ww++;
}

template< typename id_kernel_t >
void init_bssn_metric( id_kernel_t id_kernel
                     , grace::var_array_t<GRACE_NSPACEDIM>& state
                     , grace::staggered_variable_arrays_& sstate)
{
    DECLARE_GRID_EXTENTS;

    using namespace grace  ;
    using namespace Kokkos ;

    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    grace::fill_physical_coordinates(pcoords, {VEC(true,true,true)}) ;

    /**************************************/
    /* First loop fill everything execpt  */
    /* for the Gammas                     */
    /**************************************/
    parallel_for( GRACE_EXECUTION_TAG("ID","metric_ID")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    auto const id = id_kernel(VEC(i,j,k), q) ; 

                    adm_to_bssn(id, sstate.corner_staggered_fields,VEC(i,j,k),q) ; 
                }
    );
    /**************************************/
    /* Second loop fill the Gammas        */
    /**************************************/
    parallel_for( GRACE_EXECUTION_TAG("ID","Gamma_ID")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    std::array<double,GRACE_NSPACEDIM> _idx {VEC(idx(0,q),idx(1,q),idx(2,q))} ; 

                    compute_gamma_tilde(sstate.corner_staggered_fields,VEC(i,j,k),q,_idx) ;  
                }
    );

    
}

}

#endif /* GRACE_PHYSICS_BSSN_HELPERS_HH */