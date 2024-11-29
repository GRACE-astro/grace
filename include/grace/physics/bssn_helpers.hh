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

#include <grace/data_structures/variable_indices.hh>
#include <grace/system/print.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/utils/numerics/fd_utils.hh>

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
    ALPL,
    BETAXL,
    BETAYL,
    BETAZL,
    BXL,
    BYL,
    BZL,
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
sstate[ALPL]    = vview(__VA_ARGS__,ALP_,q)    ; \
sstate[BETAXL]  = vview(__VA_ARGS__,BETAX_,q)  ; \
sstate[BETAYL]  = vview(__VA_ARGS__,BETAY_,q)  ; \
sstate[BETAZL]  = vview(__VA_ARGS__,BETAZ_,q)  ; \
sstate[BXL]     = vview(__VA_ARGS__,BX_,q)     ; \
sstate[BYL]     = vview(__VA_ARGS__,BY_,q)     ; \
sstate[BZL]     = vview(__VA_ARGS__,BZ_,q)     ; \
} while(false)


static void GRACE_HOST_DEVICE
adm_to_bssn(
    grmhd_id_t const& id, 
    grace::var_array_t<GRACE_NSPACEDIM> state,
    VEC(int i, int j, int k), int q
)
{
    #if 1
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

    double const phi  = SQRTG_TO_CONFFACT(sqrtgamma) ; 

    #pragma unroll 6 
    for( int icomp=0; icomp<6; ++icomp ) {
        state(VEC(i,j,k),GTXX_+icomp,q) = INVPOW_CONFFACT(phi) * __g[icomp] ; 
    }
    
    // Compute trace of extrinsic curvature 
    double const K = adm_metric.trace_sym2tens_lower(__Kij) ; 

    #pragma unroll 6
    for( int icomp=0; icomp<6; ++icomp ) {
        state(VEC(i,j,k),ATXX_+icomp,q) = INVPOW_CONFFACT(phi) * (__Kij[icomp] - 1./3. * __g[icomp] * K) ; 
    }

    state(VEC(i,j,k),PHI_,q) = phi ; 
    state(VEC(i,j,k),K_  ,q) = K   ; 
    state(VEC(i,j,k),ALP_,q) = id.alp ; 
    state(VEC(i,j,k),BETAX_,q) = id.betax ; 
    state(VEC(i,j,k),BETAY_,q) = id.betay ; 
    state(VEC(i,j,k),BETAZ_,q) = id.betaz ; 
    #else 
    state(VEC(i,j,k),PHI_,q)  = 1 ;
    state(VEC(i,j,k),K_  ,q) = 0 ;
    for( int i=0; i<6; ++i){
        state(VEC(i,j,k),GTXX_+i,q)  = 0 ;
        state(VEC(i,j,k),ATXX_+i,q)  = 0 ;
    } 
    state(VEC(i,j,k),GTXX_,q)  = 1 ;
    state(VEC(i,j,k),GTYY_,q)  = 1 ;
    state(VEC(i,j,k),GTZZ_,q)  = 1 ;
    #endif
}

template< size_t der_order >
static void GRACE_HOST_DEVICE
compute_gamma_tilde(
    grace::var_array_t<GRACE_NSPACEDIM> state,
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

    // Determinant of gamma tilde is 1
    double const gtXX=-(gtyz*gtyz) + gtyy*gtzz;
    double const gtXY=gtxz*gtyz - gtxy*gtzz;
    double const gtXZ=-(gtxz*gtyy) + gtxy*gtyz;
    double const gtYY=-(gtxz*gtxz) + gtxx*gtzz;
    double const gtYZ=gtxy*gtxz - gtxx*gtyz;
    double const gtZZ=-(gtxy*gtxy) + gtxx*gtyy;

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

    state(VEC(i,j,k),GAMMAX_+0,q) = (gtXX*gtXX*gtxxdx + gtXY*gtXY*(gtxydx - gtxydy + gtyydx + gtyydy) + gtXX*(gtXZ*(gtxxdx + gtxzdx + gtxzdz) + 2*gtxydy*gtYY - gtYY*gtyydx + gtYZ*(gtxydy + gtxydz + gtxzdy + gtxzdz - 2*gtyzdx) + 2*gtxzdz*gtZZ - gtZZ*gtzzdx) + gtXY*(gtXX*gtxxdx + gtXX*gtxydx + gtXX*gtxydy + gtYY*gtyydy + gtYZ*(gtyydy + gtyydz - gtyzdy + gtyzdz) + gtXZ*(gtxydx - gtxydz + gtxzdx - gtxzdy + 2*gtyzdx + gtyzdy + gtyzdz) + 2*gtyzdz*gtZZ - gtZZ*gtzzdy) + gtXZ*(-(gtYY*gtyydz) + 2*gtYY*gtyzdy + gtYZ*gtyzdy - gtYZ*gtyzdz + gtYZ*gtzzdy + gtYZ*gtzzdz + gtZZ*gtzzdz + gtXZ*(gtxzdx - gtxzdz + gtzzdx + gtzzdz)))/2.;
    state(VEC(i,j,k),GAMMAX_+1,q) = (gtXY*gtXY*(gtxxdx + gtxxdy - gtxydx + gtxydy) + gtxydx*gtXZ*gtYY + gtxydz*gtXZ*gtYY - 2*gtXZ*gtxzdy*gtYY + gtYY*gtYY*gtyydy + gtXZ*gtxzdx*gtYZ - gtXZ*gtxzdz*gtYZ + gtYY*gtyydy*gtYZ - gtXX*(gtxxdy*gtYY - 2*gtxydx*gtYY + (gtxxdz - 2*gtxzdx)*gtYZ) + gtXZ*gtYY*gtyzdx + gtYY*gtYZ*gtyzdy + gtYZ*gtYZ*gtyzdy + gtXZ*gtYY*gtyzdz + gtYY*gtYZ*gtyzdz - gtYZ*gtYZ*gtyzdz + 2*gtYY*gtyzdz*gtZZ + gtXZ*gtYZ*gtzzdx + gtXY*(gtXX*gtxxdx + gtXZ*(gtxxdx + gtxxdz - gtxzdx + gtxzdz) + gtxydx*gtYY + gtxydy*gtYY + gtYY*gtyydy + gtYZ*(gtxydy - gtxydz + gtxzdx + 2*gtxzdy + gtxzdz - gtyzdx + gtyzdy) + 2*gtxzdz*gtZZ - gtZZ*gtzzdx) + gtYZ*gtYZ*gtzzdy - gtYY*gtZZ*gtzzdy + gtXZ*gtYZ*gtzzdz + gtYZ*gtYZ*gtzzdz + gtYZ*gtZZ*gtzzdz)/2.;
    state(VEC(i,j,k),GAMMAX_+2,q) = (gtXZ*gtXZ*(gtxxdx + gtxxdz - gtxzdx + gtxzdz) + gtXY*gtxydx*gtYZ - gtXY*gtxydy*gtYZ + gtXY*gtyydx*gtYZ + gtXY*gtyydy*gtYZ + gtYY*gtyydy*gtYZ + gtyydy*(gtYZ*gtYZ) + gtyydz*(gtYZ*gtYZ) - gtYZ*gtYZ*gtyzdy + gtYZ*gtYZ*gtyzdz - 2*gtXY*gtxydz*gtZZ + gtXY*gtxzdx*gtZZ + gtXY*gtxzdy*gtZZ - gtYY*gtyydz*gtZZ + gtXY*gtyzdx*gtZZ + gtXY*gtyzdy*gtZZ + 2*gtYY*gtyzdy*gtZZ + gtYZ*gtyzdy*gtZZ + gtYZ*gtyzdz*gtZZ - gtXX*(gtxxdy*gtYZ - 2*gtxydx*gtYZ + (gtxxdz - 2*gtxzdx)*gtZZ) + gtYZ*gtZZ*gtzzdz + gtZZ*gtZZ*gtzzdz + gtXZ*(gtXX*gtxxdx + gtxxdx*gtXY + gtxxdy*gtXY - gtXY*gtxydx + gtXY*gtxydy + 2*gtxydy*gtYY - gtYY*gtyydx + gtYZ*(gtxydx + gtxydy + 2*gtxydz - gtxzdy + gtxzdz - gtyzdx + gtyzdz) + gtZZ*(gtxzdx + gtxzdz + gtzzdz)))/2.;


}

template< typename id_kernel_t >
static void init_bssn_metric( id_kernel_t id_kernel
                     , grace::var_array_t<GRACE_NSPACEDIM>& state
                     , grace::var_array_t<GRACE_NSPACEDIM>& cstate
                     , grace::scalar_array_t<GRACE_NSPACEDIM>& idx)
{
    DECLARE_GRID_EXTENTS;

    using namespace grace  ;
    using namespace Kokkos ;

    coord_array_t<GRACE_NSPACEDIM> pcoords ; 
    grace::fill_physical_coordinates(pcoords, {VEC(true,true,true)}) ;
    id_kernel._pcoords = pcoords ; 
 
    /**************************************/
    /* First loop fill everything execpt  */
    /* for the Gammas                     */
    /**************************************/
    auto policy = MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+1+2*ngz,ny+1+2*ngz,nz+1+2*ngz),nq}) ; 
    parallel_for( GRACE_EXECUTION_TAG("ID","metric_ID")
                , policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    auto const id = id_kernel(VEC(i,j,k), q) ; 
                    adm_to_bssn(id, cstate,VEC(i,j,k),q) ; 
                }
    );

    /**************************************/
    /* Second loop fill the Gammas        */
    /**************************************/
    parallel_for( GRACE_EXECUTION_TAG("ID","Gamma_ID")
                , policy
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q)
                {
                    std::array<double,GRACE_NSPACEDIM> _idx {VEC(idx(0,q),idx(1,q),idx(2,q))} ; 
                    compute_gamma_tilde<4>(cstate,VEC(i,j,k),q,_idx,VEC(nx,ny,nz),ngz) ;  
                }
    );

    
}

}

#endif /* GRACE_PHYSICS_BSSN_HELPERS_HH */