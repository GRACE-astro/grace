/**
 * @file spherical_coordinate_system.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *                                    
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public _License as published by
 * the Free Software Foundation, either version 3 of the _License, or
 * any later version.
 *   
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABI_LITY or FITNESS FOR A PARTICU_LAR PURPOSE.  See the
 * GNU General Public _License for more details.
 *   
 * You should have received a copy of the GNU General Public _License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */

#include <thunder/amr/thunder_amr.hh> 
#include <thunder/coordinates/cell_volume_kernels.h>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/coordinates/rotation_matrices.hh>
#include <thunder/coordinates/spherical_coordinate_systems.hh>
#include <thunder/coordinates/spherical_device_inlines.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/data_structures/thunder_data_structures.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/errors/error.hh> 

#include <array> 

namespace thunder { 

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_physical_coordinates(
      int const itree
    , std::array<double, THUNDER_NSPACEDIM> const& lcoords ) 
{
    if( itree==0 ){
        auto const tree_coords = amr::get_tree_vertex(itree,0UL) ; 
        auto const dx_tree     = amr::get_tree_spacing(itree) ;
        return {VEC(
            (lcoords[0] * 2. - 1.) * _L,
            (lcoords[1] * 2. - 1.) * _L,
            (lcoords[2] * 2. - 1.) * _L 
        )} ; 
    } else if ( (itree-1) / P4EST_FACES == 0 ) {
        auto const xi = (2*lcoords[1]-1); 
        #ifdef THUNDER_3D 
        auto const eta = (2*lcoords[2]-1) ; 
        #endif
        auto const one_over_rho = 1./sqrt(EXPR( 1
                                            , + math::int_pow<2>(xi)
                                            , + math::int_pow<2>(eta) )) ;
        auto const z = get_zeta(lcoords[0], one_over_rho, {_F0,_Fr}, {_S0,_Sr}, false);   
        std::array<double, THUNDER_NSPACEDIM> pcoords =
            { VEC( z
                 , z * xi
                 , z * eta )};
        return detail::apply_discrete_rotation(pcoords, (itree-1)%P4EST_FACES ) ; 
    } else if ( (itree-1) / P4EST_FACES == 1) {
        auto const xi = (2*lcoords[1]-1); 
        #ifdef THUNDER_3D 
        auto const eta = (2*lcoords[2]-1) ; 
        #endif
        auto const one_over_rho = 1./ sqrt(EXPR( 1
                                             , + math::int_pow<2>(xi)
                                             , + math::int_pow<2>(eta) )) ;
        auto const z = get_zeta(lcoords[0], one_over_rho, {_F1,_Fr1}, {_S1,_Sr1}, _use_logr);
        std::array<double, THUNDER_NSPACEDIM> pcoords =
            { VEC( z
                 , z * xi
                 , z * eta )};
        return detail::apply_discrete_rotation(pcoords, (itree-1)%P4EST_FACES ) ; 
    } else {
        ERROR("Logical coordinates failed sanity check.");
    }
}

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_physical_coordinates(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , int64_t q 
    , std::array<double, THUNDER_NSPACEDIM> const& cell_coordinates
    , bool use_ghostzones )
{
    using namespace thunder ; 
    int64_t itree = amr::get_quadrant_owner(q)   ; 
    return get_physical_coordinates(itree, get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones)) ; 
}


std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_physical_coordinates(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , int64_t q 
    , bool use_ghostzones )
{
    return get_physical_coordinates(ijk,q,{VEC(0.5,0.5,0.5)},use_ghostzones);
} 


std::array<double, THUNDER_NSPACEDIM>
THUNDER_HOST spherical_coordinate_system_impl_t::get_logical_coordinates(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , int64_t q 
    , std::array<double, THUNDER_NSPACEDIM> const& cell_coordinates
    , bool use_ghostzones)
{
    using namespace thunder ;

    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int64_t nq = amr::get_local_num_quadrants()      ;
    int ngz = amr::get_n_ghosts()                    ; 

    int64_t itree = amr::get_quadrant_owner(q)   ; 
    amr::quadrant_t quad = amr::get_quadrant(itree,q) ; 

    auto const dx_quad  = 1./(1<<quad.level()) ; 
    auto const qcoords = quad.qcoords()     ; 

    EXPR(
    auto const dx_cell = dx_quad / nx ;, 
    auto const dy_cell = dx_quad / ny ;,
    auto const dz_cell = dx_quad / nz ;
    ) 

    return {
        VEC(
            qcoords[0] * dx_quad + (ijk[0] + cell_coordinates[0] - use_ghostzones * ngz) * dx_cell, 
            qcoords[1] * dx_quad + (ijk[1] + cell_coordinates[1] - use_ghostzones * ngz) * dy_cell, 
            qcoords[2] * dx_quad + (ijk[2] + cell_coordinates[2] - use_ghostzones * ngz) * dz_cell 
        ) 
    } ; 
} 

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_logical_coordinates(
      int itree
    , std::array<double, THUNDER_NSPACEDIM> const& physical_coordinates )
{
    if( itree == 0 ){
        return {VEC(
            (physical_coordinates[0]/_L + 1.)/2.,
            (physical_coordinates[1]/_L + 1.)/2.,
            (physical_coordinates[2]/_L + 1.)/2.
        )};
    } else {
        auto l_coords = 
             detail::apply_discrete_rotation(physical_coordinates, (itree-1)%P4EST_FACES, true) ;
        auto const z = l_coords[0] ;
        auto const r = sqrt(EXPR( math::int_pow<2>(physical_coordinates[0]),
                                + math::int_pow<2>(physical_coordinates[1]),
                                + math::int_pow<2>(physical_coordinates[2]))) ;
        auto const fr =  ((itree-1) / P4EST_FACES == 0) * _Fr 
                      +  ((itree-1) / P4EST_FACES == 1) * _Fr1 ; 
        auto const f0 =  ((itree-1) / P4EST_FACES == 0) * _F0 
                      +  ((itree-1) / P4EST_FACES == 1) * _F1 ;
        auto const sr =  ((itree-1) / P4EST_FACES == 0) * _Sr 
                      +  ((itree-1) / P4EST_FACES == 1) * _Sr1 ; 
        auto const s0 =  ((itree-1) / P4EST_FACES == 0) * _S0 
                      +  ((itree-1) / P4EST_FACES == 1) * _S1 ;
        if( _use_logr and ((itree-1) / P4EST_FACES == 1)){
            l_coords[0] = 0.5*((log(r)-_S1)/_Sr1+1) ; 
        } else {
            auto const z_coeff = fr + sr * z / r ; 
            auto const z0      = f0 + s0 * z / r ;
            l_coords[0] = (z-z0)/z_coeff ;
        }
        EXPRD(
        l_coords[1] = 0.5*(1+l_coords[1]/z) ;,
        l_coords[2] = 0.5*(1+l_coords[2]/z) ; 
        )
        return l_coords ; 
        
    } 
}

std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_logical_coordinates(
    std::array<double, THUNDER_NSPACEDIM> const& physical_coordinates )
{
    
    if ( EXPR(
            physical_coordinates[0] > -_L 
        and physical_coordinates[0] <  _L,
        and physical_coordinates[1] > -_L 
        and physical_coordinates[1] <  _L,
        and physical_coordinates[2] > -_L 
        and physical_coordinates[2] <  _L)
    ){
        return get_logical_coordinates(0, physical_coordinates) ; 
    } else {
        double const r = sqrt(
            EXPR( math::int_pow<2>(physical_coordinates[0]),
                + math::int_pow<2>(physical_coordinates[1]),
                + math::int_pow<2>(physical_coordinates[2]))
        ) ; 
        #ifdef THUNDER_3D 
        double const rxy = sqrt( math::int_pow<2>(physical_coordinates[0])
                               + math::int_pow<2>(physical_coordinates[1])); 
        double const theta = asin(physical_coordinates[1]/rxy) ; 
        double const phi   = acos(physical_coordinates[2]/r) ; 
        if ( phi < M_PI/4 ) {
            if ( r > _Ri) {
                return get_logical_coordinates(12, physical_coordinates) ; 
            } else {
                return get_logical_coordinates(6, physical_coordinates) ;  ; 
            }
        } else if ( phi > 3*M_PI/4 ) {
            if ( r > _Ri) {
                return get_logical_coordinates(11, physical_coordinates) ;  ;
            } else {
                return get_logical_coordinates(5, physical_coordinates) ;  ; 
            }
        }
        #else 
        double const theta = asin(physical_coordinates[1]/r) ; 
        #endif  
        if( (theta > -M_PI/4) and (theta < M_PI / 4) and (physical_coordinates[0]<0.)) {
            if ( r > _Ri) {
                return get_logical_coordinates(1+P4EST_FACES, physical_coordinates) ;
            } else {
                return get_logical_coordinates(1, physical_coordinates); 
            }
        } else if ( (theta>=M_PI/4) )  {
            if ( r > _Ri) {
                return get_logical_coordinates(4+P4EST_FACES, physical_coordinates) ;
            } else {
                return get_logical_coordinates(4, physical_coordinates) ; 
            }
        } else if ( (theta > -M_PI/4) and (theta < M_PI / 4) and (physical_coordinates[0]>0.)) {
            if ( r > _Ri) {
                return get_logical_coordinates(2+P4EST_FACES, physical_coordinates) ;
            } else {
                return get_logical_coordinates(2, physical_coordinates) ; 
            }
        } else if ( (theta<=-M_PI/4) ){
            if ( r > _Ri) {
                return get_logical_coordinates(3+P4EST_FACES, physical_coordinates) ;
            } else {
                return get_logical_coordinates(3, physical_coordinates) ; 
            }
        } else {
            ERROR("Physical coordinates sanity check failed.") ; 
        }

    }

}

double
THUNDER_HOST spherical_coordinate_system_impl_t::get_cell_volume(
    std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
  , int64_t q
  , bool use_ghostzones)
{
    using namespace thunder ;
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int64_t itree = amr::get_quadrant_owner(q)   ; 
    amr::quadrant_t quad = amr::get_quadrant(itree,q) ; 

    auto const dx_quad  = 1./(1<<quad.level()) ; 
    auto const qcoords = quad.qcoords()     ; 

    EXPR(
    auto const dx_cell = dx_quad / nx ;, 
    auto const dy_cell = dx_quad / ny ;,
    auto const dz_cell = dx_quad / nz ;
    )
    return get_cell_volume(ijk,q,itree,{VEC(dx_cell,dy_cell,dz_cell)},use_ghostzones);
}

double
THUNDER_HOST spherical_coordinate_system_impl_t::get_cell_volume(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& dxl 
    , bool use_ghostzones) 
{
    auto lcoords = get_logical_coordinates(ijk,q,{VEC(0.,0.,0.)},use_ghostzones) ; 

    if( EXPR(
           lcoords[0] < 0 or lcoords[0] > 1,
        or lcoords[1] < 0 or lcoords[1] > 1,
        or lcoords[2] < 0 or lcoords[2] > 1
    )) {
        return get_cell_volume_buffer_zone(ijk,itree,lcoords,dxl);
    }
    if( itree == 0 ) {
        return math::int_pow<THUNDER_NSPACEDIM>(2.*_L) * EXPR(dxl[0],*dxl[1],*dxl[2]) ; 
    } else if( (itree-1)/P4EST_FACES == 0 ) {
        return dVol_sph(_L,_Ri, VECD(dxl[1],dxl[2]),dxl[0], VECD(lcoords[1],lcoords[2]),lcoords[0]) ; 
    } else {
        if( _use_logr ){
            return dVol_sph_log(_Ri,_Ro, VECD(dxl[1],dxl[2]),dxl[0], VECD(lcoords[1],lcoords[2]),lcoords[0]) ; 
        } else { 
            return dVol_sph_ext(_Ri,_Ro, VECD(dxl[1],dxl[2]),dxl[0], VECD(lcoords[1],lcoords[2]),lcoords[0]) ;
        }
    }


}

double
THUNDER_HOST spherical_coordinate_system_impl_t::get_cell_volume_buffer_zone(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& lcoords
    , std::array<double, THUNDER_NSPACEDIM> const& dxl ) 
{
    using namespace thunder ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int iface = EXPR(
          (ijk[0] < ngz) * 0 
        + (ijk[0] > nx + ngz-1) * 1,
        + (ijk[1] < ngz) * 2 
        + (ijk[1] > ny + ngz-1) * 3,
        + (ijk[2] < ngz) * 4 
        + (ijk[2] > nz + ngz-1) * 5) ;
    if( iface >= P4EST_FACES) iface = 0;   
    auto& conn = amr::connectivity::get();
    int    itree_b  = conn.tree_to_tree(itree, iface) ; 
    int8_t iface_b  = conn.tree_to_face(itree, iface) ; 
    int    polarity = conn.tree_to_tree_polarity(itree,iface) ;
    /****************************************/
    /* First index below is the distance    */
    /* in index space from tree boundary.   */
    /* The other two are simply the indices */
    /* orthogonal to the face.              */
    /****************************************/
    EXPR(
    int ig = EXPR( 
          (iface==0) * (ngz-1-ijk[0])
        + (iface==1) * (ijk[0]-nx-ngz),
        + (iface==2) * (ngz-1-ijk[1])
        + (iface==3) * (ijk[1]-ny-ngz),
        + (iface==4) * (ngz-1-ijk[2])
        + (iface==5) * (ijk[2]-nz-ngz)  ) ;, 
    int j  = EXPR( 
          (iface/2==0) * ijk[1],
        + (iface/2==1) * ijk[0],
        + (iface/2==2) * ijk[0] ) ;, 
    int k  = EXPR( 
          (iface/2==0) * ijk[2],
        + (iface/2==1) * ijk[2],
        + (iface/2==2) * ijk[1] ) ; )

    EXPR(
    int i_b = EXPR(
          (iface_b==0) * (
            (!polarity) * (ngz+ig)
          + (polarity)  * (2*ngz-1-ig) )
        + (iface_b==1) * (
            (!polarity) * (nx+ngz+ig)
          + (polarity)  * (nx+2*ngz-ig-1) ),
        + (iface_b/2==1) * j,
        + (iface_b/2==2) * j  );,
    int j_b = EXPR(
          (iface_b==2) * (
            (!polarity) * (ngz+ig)
          + (polarity)  * (2*ngz-1-ig) )
        + (iface_b==3) * (
            (!polarity) * (ny+ngz+ig)
          + (polarity)  * (ny+2*ngz-ig-1) ),
        + (iface_b/2==0) * j,
        + (iface_b/2==2) * k 
    );,
    int k_b = EXPR(
          (iface_b==4) * (
            (!polarity) * (ngz+ig)
          + (polarity)  * (2*ngz-1-ig) )
        + (iface_b==5) * (
            (!polarity) * (nz+ngz+ig)
          + (polarity)  * (nz+2*ngz-ig-1) ),
        + (iface_b/2==0) * k,
        + (iface_b/2==1) * k 
    ) ;
    )
    /******************************************/
    /* Now we find the logical coordinates    */
    /* of the appropriate cell in the         */
    /* neighbor tree.                         */
    /* We do this as follows:                 */
    /* First we find the physical coordinates */
    /* of the quadrant corner which sits on   */
    /* the tree boundary. Then we transform   */
    /* these to tree logical coordinates of   */
    /* the neighbor tree.                     */
    /******************************************/
    EXPR(
    double const x = lcoords[0]
        + (iface%2 == 1    ) * dxl[0] * nx;,
    double const y = lcoords[1]
        + ((iface/2)%2 == 1) * dxl[1] * ny;,
    double const z = lcoords[2]
        + ((iface/2)/2 == 1) * dxl[2] * nz;
    )
    auto pcoords = get_physical_coordinates(
          itree
        , lcoords 
    ) ; 
    auto lcoords_b = get_logical_coordinates(
          itree_b 
        , pcoords
    ) ; 
    /*******************************************/
    /* Now we can compute the cell coordinates */
    /*******************************************/
    EXPR(
    lcoords_b[0] += dxl[0] * i_b ;,
    lcoords_b[1] += dxl[1] * j_b ;,
    lcoords_b[2] += dxl[2] * k_b ;
    )
    if( itree_b == 0 ) {
        return math::int_pow<THUNDER_NSPACEDIM>(2.*_L) * EXPR(dxl[0],*dxl[1],*dxl[2]) ; 
    } else if( (itree_b-1)/P4EST_FACES == 0 ) {
        return dVol_sph(_L,_Ri 
                , VECD(dxl[1],dxl[2]),dxl[0]
                , VECD(lcoords_b[1],lcoords_b[1]),lcoords_b[0] ) ; 
    } else {
        if( _use_logr ){
            return dVol_sph_log(_Ri,_Ro
                    , VECD(dxl[1],dxl[2]),dxl[0]
                    , VECD(lcoords_b[1],lcoords_b[1]),lcoords_b[0] ) ; 
        } else { 
            return dVol_sph_ext(_Ri,_Ro
                    , VECD(dxl[1],dxl[2]),dxl[0]
                    , VECD(lcoords_b[1],lcoords_b[1]),lcoords_b[0] ) ;
        }
    }


}

double THUNDER_HOST 
spherical_coordinate_system_impl_t::get_zeta( double const& z
                                            , double const& one_over_rho
                                            , std::array<double,2> const& F
                                            , std::array<double,2> const& S
                                            , bool use_logr) const 
{
    if( use_logr ){
        return exp(S[0] + S[1]*(2*z-1)) * one_over_rho; 
    } else { 
        auto const z_coeff = F[1] + S[1]*one_over_rho ;   
        auto const z0      = F[0] + S[0]*one_over_rho ; 
        return z*z_coeff + z0 ; 
    }
} 

}