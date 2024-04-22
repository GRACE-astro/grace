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
#include <thunder/coordinates/spherical_coordinate_jacobian_utils.hh>
#include <thunder/coordinates/surface_elements/cell_surfaces_2D.hh>
#include <thunder/coordinates/surface_elements/cell_surfaces_3D.hh>
#include <thunder/coordinates/volume_elements/vol_sph_3D.hh>
#include <thunder/coordinates/volume_elements/vol_sph_3D_log.hh>
#include <thunder/coordinates/volume_elements/dVol_sph_3D_log_analytic.hh>
#include <thunder/coordinates/volume_elements/cell_volume_3D_helpers.hh>
#include <thunder/coordinates/surface_elements/cell_surface_3D_helpers.hh>
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
    auto lcoords_b = lcoords ; 
    auto itree_b   = itree   ; 
    /* First we handle buffer zones */
    if( is_outside_tree(lcoords) ) {
        /* Check if the boundary is internal */
        if( !is_physical_boundary(lcoords,itree) ) {
            /* In this case we just need to transfer the coordinates */
            /* and proceed as normal.                                */
            lcoords_b = get_logical_coordinates_buffer_zone(itree, lcoords);
            int8_t iface, iface_b ; 
            std::tie(itree_b,iface_b,iface) = get_neighbor_tree_and_face(itree,lcoords) ; 
            if ( itree_b == -1 ) {
                return {VEC(1,1,1)} ; 
            }
        }
    }
    if( itree_b==0 ){
        return get_physical_coordinates_cart(_L, lcoords_b) ; 
    } else if ( (itree_b-1) / P4EST_FACES == 0 ) {
        return get_physical_coordinates_sph((itree_b-1)%P4EST_FACES,_L,_Ri,{_F0,_Fr},{_S0,_Sr},lcoords_b,false) ; 
    } else if ( (itree_b-1) / P4EST_FACES == 1) {
        return get_physical_coordinates_sph((itree_b-1)%P4EST_FACES,_Ri,_Ro,{_F1,_Fr1},{_S1,_Sr1},lcoords_b,_use_logr) ;
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

double THUNDER_HOST 
spherical_coordinate_system_impl_t::get_jacobian(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) 
{
    return utils::det<THUNDER_NSPACEDIM>(
        get_jacobian_matrix(ijk,q,cell_coordinates,use_ghostzones) 
    ) ; 
}

double THUNDER_HOST 
spherical_coordinate_system_impl_t::get_jacobian(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) 
{
    return utils::det<THUNDER_NSPACEDIM>(
        get_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) 
    ) ; 
}

double THUNDER_HOST 
spherical_coordinate_system_impl_t::get_jacobian(
      int itree
    , std::array<double,THUNDER_NSPACEDIM> const& lcoords) 
{
    return utils::det<THUNDER_NSPACEDIM>(
        get_jacobian_matrix(itree,lcoords) 
    ) ; 
}

double THUNDER_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) 
{
    return utils::det<THUNDER_NSPACEDIM>(
        get_inverse_jacobian_matrix(ijk,q,cell_coordinates,use_ghostzones)
    ) ; 
}

double THUNDER_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) 
{
    return utils::det<THUNDER_NSPACEDIM>(
        get_inverse_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) 
    ) ; 
}

double THUNDER_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian(
      int itree
    , std::array<double,THUNDER_NSPACEDIM> const& lcoords) 
{
    return utils::det<THUNDER_NSPACEDIM>(
        get_inverse_jacobian_matrix(itree,lcoords) 
    ) ; 
}

std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_jacobian_matrix(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) 
{ 
    using namespace thunder ;
    int itree = amr::get_quadrant_owner(q) ;
    return get_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) ; 
}

std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_jacobian_matrix(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree 
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) 
{
    using namespace thunder ; 
    auto lcoords =  get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_jacobian_matrix(itree,lcoords) ; 
} 

std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_jacobian_matrix(
      int itree
    , std::array<double,THUNDER_NSPACEDIM> const& lcoords ) 
{
    using namespace thunder; 
    int itree_b = itree ; 
    auto lcoords_b = lcoords ; 
    if( is_outside_tree(lcoords) and !is_physical_boundary(lcoords,itree) ) {
        int8_t dummy1, dummy2 ; 
        std::tie(itree_b,dummy1,dummy2) =
            get_neighbor_tree_and_face(itree,lcoords) ; 
        
        lcoords_b = get_logical_coordinates_buffer_zone(
            itree,
            lcoords
        ) ; 
    }
    double si, so; 
    double r1,r2 ; 
    if( itree_b == 0 ) {
        auto J = utils::identity_matrix<THUNDER_NSPACEDIM>() ; 
        for( auto& x: J ) x /= (2.*_L) ; 
    } else if ( (itree_b-1)/P4EST_FACES == 0 ) { 
        si=0; so=1.; r1=_L; r2=_Ri;
    } else {
        si = 1.; so=1; 
        r1 = _Ri; r2=_Ro;
        if( _use_logr ) {
            #ifdef THUNDER_3D 
            return {
                Jac_sph_log_3D_00(r1,r2, lcoords_b[1], lcoords_b[2],lcoords_b[0]), Jac_sph_log_3D_01(r1,r2, lcoords_b[1], lcoords_b[2],lcoords_b[0]), Jac_sph_log_3D_02(r1,r2, lcoords_b[1], lcoords_b[2],lcoords_b[0]),
                Jac_sph_log_3D_10(r1,r2, lcoords_b[1], lcoords_b[2],lcoords_b[0]), Jac_sph_log_3D_11(r1,r2, lcoords_b[1], lcoords_b[2],lcoords_b[0]), Jac_sph_log_3D_12(r1,r2, lcoords_b[1], lcoords_b[2],lcoords_b[0]),
                Jac_sph_log_3D_20(r1,r2, lcoords_b[1], lcoords_b[2],lcoords_b[0]), Jac_sph_log_3D_21(r1,r2, lcoords_b[1], lcoords_b[2],lcoords_b[0]), Jac_sph_log_3D_22(r1,r2, lcoords_b[1], lcoords_b[2],lcoords_b[0])
            };
            #else 
            return  {
                Jac_sph_log_2D_00(r1,r2, lcoords_b[1],lcoords_b[0]), Jac_sph_log_2D_01(r1,r2, lcoords_b[1], lcoords_b[0]), 
                Jac_sph_log_2D_10(r1,r2, lcoords_b[1],lcoords_b[0]), Jac_sph_log_2D_11(r1,r2, lcoords_b[1], lcoords_b[0])
            };
            #endif
        } 
    }
    #ifdef THUNDER_3D 
    return {
        Jac_sph_3D_00(r1,r2, lcoords_b[1], si,so, lcoords_b[2]), Jac_sph_3D_01(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]), Jac_sph_3D_02(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]),
        Jac_sph_3D_10(r1,r2, lcoords_b[1], si,so, lcoords_b[2]), Jac_sph_3D_11(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]), Jac_sph_3D_12(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]),
        Jac_sph_3D_20(r1,r2, lcoords_b[1], si,so, lcoords_b[2]), Jac_sph_3D_21(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]), Jac_sph_3D_22(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0])
    };
    #else 
    return  {
        Jac_sph_2D_00(r1,r2, lcoords_b[1], si,so), Jac_sph_2D_01(r1,r2, lcoords_b[1], si,so, lcoords_b[0]), 
        Jac_sph_2D_10(r1,r2, lcoords_b[1], si,so), Jac_sph_2D_11(r1,r2, lcoords_b[1], si,so, lcoords_b[0])
    };
    #endif 
}


std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian_matrix(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) 
{ 
    using namespace thunder ;
    int itree = amr::get_quadrant_owner(q) ;
    return get_inverse_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) ; 
}

std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian_matrix(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree 
    , std::array<double,THUNDER_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) 
{
    using namespace thunder ; 
    auto lcoords =  get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_inverse_jacobian_matrix(itree,lcoords) ; 
} 

std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian_matrix(
      int itree
    , std::array<double,THUNDER_NSPACEDIM> const& lcoords ) 
{
    using namespace thunder; 
    int itree_b = itree ; 
    auto lcoords_b = lcoords ; 
    if( is_outside_tree(lcoords) and !is_physical_boundary(lcoords,itree) ) {
        int8_t dummy1, dummy2 ; 
        std::tie(itree_b,dummy1,dummy2) =
            get_neighbor_tree_and_face(itree,lcoords) ; 
        
        lcoords_b = get_logical_coordinates_buffer_zone(
            itree,
            lcoords
        ) ; 
    }
    double si, so; \
    double r1, r2 ;
    if( itree_b == 0 ) {
        auto J = utils::identity_matrix<THUNDER_NSPACEDIM>() ; 
        for( auto& x: J ) x *= (2.*_L) ; 
    } else if ( (itree_b-1)/P4EST_FACES == 0 ) { 
        si=0; so=1.; 
        r1 = _L; r2 = _Ri ; 
    } else {
        si = 1.; so=1; 
        r1 = _Ri; r2 = _Ro; 
        if( _use_logr ) {
            #ifdef THUNDER_3D 
            return {
                Jac_sph_log_inv_3D_00(r1,r2, lcoords_b[1],lcoords_b[2],lcoords_b[0]), Jac_sph_log_inv_3D_01(r1,r2, lcoords_b[1],lcoords_b[2],lcoords_b[0]), Jac_sph_log_inv_3D_02(r1,r2, lcoords_b[1],lcoords_b[2],lcoords_b[0]),
                Jac_sph_log_inv_3D_10(r1,r2, lcoords_b[1],lcoords_b[2],lcoords_b[0]), Jac_sph_log_inv_3D_11(r1,r2, lcoords_b[1],lcoords_b[2],lcoords_b[0]), Jac_sph_log_inv_3D_12(r1,r2, lcoords_b[1],lcoords_b[2],lcoords_b[0]),
                Jac_sph_log_inv_3D_20(r1,r2, lcoords_b[1],lcoords_b[2],lcoords_b[0]), Jac_sph_log_inv_3D_21(r1,r2, lcoords_b[1],lcoords_b[2],lcoords_b[0]), Jac_sph_log_inv_3D_22(r1,r2, lcoords_b[1],lcoords_b[2],lcoords_b[0])
            };
            #else 
            return  {
                Jac_sph_log_inv_2D_00(r1,r2, lcoords_b[1],lcoords_b[0]), Jac_sph_log_inv_2D_01(r1,r2, lcoords_b[1], lcoords_b[0]), 
                Jac_sph_log_inv_2D_10(r1,r2, lcoords_b[1],lcoords_b[0]), Jac_sph_log_inv_2D_11(r1,r2, lcoords_b[1], lcoords_b[0])
            };
            #endif
        }
    }
    #ifdef THUNDER_3D 
    return {
        Jac_sph_inv_3D_00(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]), Jac_sph_inv_3D_01(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]), Jac_sph_inv_3D_02(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]),
        Jac_sph_inv_3D_10(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]), Jac_sph_inv_3D_11(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]), 0.0,
        Jac_sph_inv_3D_20(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0]), 0.0                                                                      , Jac_sph_inv_3D_22(r1,r2, lcoords_b[1], si,so, lcoords_b[2],lcoords_b[0])
    };
    #else 
    return  {
        Jac_sph_inv_2D_00(r1,r2, lcoords_b[1], si,so, lcoords_b[0]), Jac_sph_inv_2D_01(r1,r2, lcoords_b[1], si,so, lcoords_b[0]), 
        Jac_sph_inv_2D_10(r1,r2, lcoords_b[1], si,so, lcoords_b[0]), Jac_sph_inv_2D_11(r1,r2, lcoords_b[1], si,so, lcoords_b[0])
    };
    #endif 
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
    int ngz = thunder::amr::get_n_ghosts();
    if( is_outside_tree(lcoords,true) and !is_physical_boundary(lcoords,itree,true) and use_ghostzones) {
        return get_cell_volume_buffer_zone(itree,q,lcoords,dxl);
    }
    if( itree == 0 ) {
        return math::int_pow<THUNDER_NSPACEDIM>(2.*_L) * EXPR(dxl[0],*dxl[1],*dxl[2]) ; 
    } else if( (itree-1)/P4EST_FACES == 0 ) {
        #ifndef THUNDER_3D 
        return dVol_sph(_L,_Ri, dxl[1],dxl[0], lcoords[1],lcoords[0]) ; 
        #else
        return detail::get_cell_volume_3D<5UL>(0.,1.,_L,_Ri,lcoords[0],lcoords[1],lcoords[2],dxl[0],dxl[1],dxl[2]) ;
        #endif 
    } else {
        if( _use_logr ){
            #ifndef THUNDER_3D 
            return dVol_sph_log(_Ri,_Ro, dxl[1],dxl[0], lcoords[1],lcoords[0]) ;
            #else  
            return detail::get_cell_volume_3D_log(_Ri,_Ro,lcoords[0],lcoords[1],lcoords[2],dxl[0],dxl[1],dxl[2]) ;
            #endif  
        } else { 
            #ifndef THUNDER_3D
            return dVol_sph_ext(_Ri,_Ro, dxl[1],dxl[0], lcoords[1],lcoords[0]) ;
            #else 
            return detail::get_cell_volume_3D<5UL>(1.,1.,_Ri,_Ro,lcoords[0],lcoords[1],lcoords[2],dxl[0],dxl[1],dxl[2]) ;
            #endif 
        }
    }
}
double
THUNDER_HOST spherical_coordinate_system_impl_t::get_cell_face_surface(
    std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
  , int64_t q
  , int8_t face 
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
    return get_cell_face_surface(ijk,q,face,itree,{VEC(dx_cell,dy_cell,dz_cell)},use_ghostzones);
}

double
THUNDER_HOST spherical_coordinate_system_impl_t::get_cell_face_surface(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
    , int64_t q
    , int8_t face 
    , int itree
    , std::array<double, THUNDER_NSPACEDIM> const& dxl 
    , bool use_ghostzones) 
{
    std::array<double, THUNDER_NSPACEDIM> cell_coordinates 
    {
        VEC( (face==1)*dxl[0]
           , (face==3)*dxl[1]
           , (face==5)*dxl[2] )
    } ; 
    auto lcoords = get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    int ngz = thunder::amr::get_n_ghosts();
    if( is_outside_tree(lcoords) and !is_physical_boundary(lcoords,itree) and use_ghostzones) {
        //return get_cell_face_surface_buffer_zone(ijk,q,face,itree,{VEC(dx_cell,dy_cell,dz_cell)}) ; 
    }
    if( itree == 0 ) {
        EXPRD(
        double const dh = EXPR(
               (face/2==0)*dxl[1],
             + (face/2==1)*dxl[0],
             + (face/2==2)*dxl[0]
        ) ;,
        double const dt = EXPR(
               (face/2==0)*dxl[2],
             + (face/2==1)*dxl[1],
             + (face/2==2)*dxl[1]
        ) ;
        )
        return math::int_pow<THUNDER_NSPACEDIM-1>(2.*_L) * EXPRD(dh,*dt) ; 
    } else if( (itree-1)/P4EST_FACES == 0 ) {
        if( face / 2 == 0 ) {
            auto const zeta = lcoords[0] ; 
            auto const integrand = [&] (VECD( double const& eta 
                                            , double const& xi) ) 
            { 
                return get_surface_element_sph(face,0.,1.,_L,_Ri,VEC(zeta,eta,xi) ) ; 
            } ; 
            std::array<double,THUNDER_NSPACEDIM-1> a{ VECD( lcoords[1],
                                                            lcoords[2] )}
                                                ,  b{ VECD( lcoords[1]+dxl[1],
                                                            lcoords[2]+dxl[2]) } ;
            return utils::nd_quadrature_integrate<THUNDER_NSPACEDIM-1,5>(a,b,integrand) ;
        } else if ( face/2 == 1 ) { 
            auto const eta = lcoords[1] ; 
            auto const integrand = [&] (VECD( double const& zeta 
                                            , double const& xi) ) 
            { 
                return get_surface_element_sph(face,0.,1.,_L,_Ri,VEC(zeta,eta,xi) ) ; 
            } ; 
            std::array<double,THUNDER_NSPACEDIM-1> a{ VECD( lcoords[0],
                                                            lcoords[2] )}
                                                ,  b{ VECD( lcoords[0]+dxl[0],
                                                            lcoords[2]+dxl[2]) } ;
            return utils::nd_quadrature_integrate<THUNDER_NSPACEDIM-1,5>(a,b,integrand) ;
            #ifdef THUNDER_3D
        } else if ( face/2 == 2) {
            auto const xi = lcoords[2] ; 
            auto const integrand = [&] (VECD( double const& zeta 
                                            , double const& eta) ) 
            { 
                return get_surface_element_sph(face,0.,1.,_L,_Ri,VEC(zeta,eta,xi) ) ; 
            } ; 
            std::array<double,THUNDER_NSPACEDIM-1> a{ VECD( lcoords[0],
                                                            lcoords[1] )}
                                                ,  b{ VECD( lcoords[0]+dxl[0],
                                                            lcoords[1]+dxl[1]) } ;
            return utils::nd_quadrature_integrate<THUNDER_NSPACEDIM-1,5>(a,b,integrand) ;
            #endif 
        }
        
    } else {
        if( _use_logr ){
            if( face / 2 == 0 ) {
                auto const zeta = lcoords[0] ; 
                auto const integrand = [&] (VECD( double const& eta 
                                                , double const& xi) ) 
                { 
                    return get_surface_element_sph_log(face,_Ri,_Ro,VEC(zeta,eta,xi) ) ; 
                } ; 
                std::array<double,THUNDER_NSPACEDIM-1> a{ VECD( lcoords[1],
                                                                lcoords[2] )}
                                                    ,  b{ VECD( lcoords[1]+dxl[1],
                                                                lcoords[2]+dxl[2]) } ;
                return utils::nd_quadrature_integrate<THUNDER_NSPACEDIM-1,5>(a,b,integrand) ;
            } else if ( face/2 == 1 ) { 
                auto const eta = lcoords[1] ; 
                auto const integrand = [&] (VECD( double const& zeta 
                                                , double const& xi) ) 
                { 
                    return get_surface_element_sph_log(face,_Ri,_Ro,VEC(zeta,eta,xi) ) ; 
                } ; 
                std::array<double,THUNDER_NSPACEDIM-1> a{ VECD( lcoords[0],
                                                                lcoords[2] )}
                                                    ,  b{ VECD( lcoords[0]+dxl[0],
                                                                lcoords[2]+dxl[2]) } ;
                return utils::nd_quadrature_integrate<THUNDER_NSPACEDIM-1,5>(a,b,integrand) ;
            #ifdef THUNDER_3D 
            } else if ( face/2 == 2) {
                auto const xi = lcoords[2] ; 
                auto const integrand = [&] (VECD( double const& zeta 
                                                , double const& eta) ) 
                { 
                    return get_surface_element_sph_log(face,_Ri,_Ro,VEC(zeta,eta,xi) ) ; 
                } ; 
                std::array<double,THUNDER_NSPACEDIM-1> a{ VECD( lcoords[0],
                                                                lcoords[1] )}
                                                    ,  b{ VECD( lcoords[0]+dxl[0],
                                                                lcoords[1]+dxl[1]) } ;
                return utils::nd_quadrature_integrate<THUNDER_NSPACEDIM-1,5>(a,b,integrand) ;
            #endif
            }
        } else { 
            if( face / 2 == 0 ) {
                auto const zeta = lcoords[0] ; 
                auto const integrand = [&] (VECD( double const& eta 
                                                , double const& xi) ) 
                { 
                    return get_surface_element_sph(face,1.,1.,_Ri,_Ro,VEC(zeta,eta,xi) ) ; 
                } ; 
                std::array<double,THUNDER_NSPACEDIM-1> a{ VECD( lcoords[1],
                                                                lcoords[2] )}
                                                    ,  b{ VECD( lcoords[1]+dxl[1],
                                                                lcoords[2]+dxl[2]) } ;
                return utils::nd_quadrature_integrate<THUNDER_NSPACEDIM-1,5>(a,b,integrand) ;
            } else if ( face/2 == 1 ) { 
                auto const eta = lcoords[1] ; 
                auto const integrand = [&] (VECD( double const& zeta 
                                                , double const& xi) ) 
                { 
                    return get_surface_element_sph(face,1.,1.,_Ri,_Ro,VEC(zeta,eta,xi) ) ; 
                } ; 
                std::array<double,THUNDER_NSPACEDIM-1> a{ VECD( lcoords[0],
                                                                lcoords[2] )}
                                                    ,  b{ VECD( lcoords[0]+dxl[0],
                                                                lcoords[2]+dxl[2]) } ;
                return utils::nd_quadrature_integrate<THUNDER_NSPACEDIM-1,5>(a,b,integrand) ;
            #ifdef THUNDER_3D
            } else if ( face/2 == 2) {
                auto const xi = lcoords[2] ; 
                auto const integrand = [&] (VECD( double const& zeta 
                                                , double const& eta) ) 
                { 
                    return get_surface_element_sph(face,1.,1.,_Ri,_Ro,VEC(zeta,eta,xi) ) ; 
                } ; 
                std::array<double,THUNDER_NSPACEDIM-1> a{ VECD( lcoords[0],
                                                                lcoords[1] )}
                                                    ,  b{ VECD( lcoords[0]+dxl[0],
                                                                lcoords[1]+dxl[1]) } ;
                return utils::nd_quadrature_integrate<THUNDER_NSPACEDIM-1,5>(a,b,integrand) ;
            #endif
            }
        }
    }
}
bool 
THUNDER_HOST spherical_coordinate_system_impl_t::is_outside_tree(
    std::array<double, THUNDER_NSPACEDIM> const& lcoords, bool check_exact_boundary
) 
{
    return EXPR(
           lcoords[0] < 0 or (check_exact_boundary and lcoords[0] >= 1) or (!check_exact_boundary and lcoords[0] > 1), 
        or lcoords[1] < 0 or (check_exact_boundary and lcoords[1] >= 1) or (!check_exact_boundary and lcoords[1] > 1), 
        or lcoords[2] < 0 or (check_exact_boundary and lcoords[2] >= 1) or (!check_exact_boundary and lcoords[2] > 1) 
    ) ; 
}

bool 
THUNDER_HOST spherical_coordinate_system_impl_t::is_physical_boundary(
    std::array<double,THUNDER_NSPACEDIM> const& lcoords, int itree, bool check_exact_boundary) 
{
    ASSERT_DBG(is_outside_tree(lcoords,check_exact_boundary), "In is_physical_boundary: lcoords not in buffer zone"); 
    int itree_b; int8_t iface,iface_b ; 
    std::tie(itree_b,iface_b,iface) = get_neighbor_tree_and_face(itree,lcoords,check_exact_boundary) ; 
    return (itree_b==itree) and (iface_b==iface) ; 
}

std::tuple<int, int8_t, int8_t>
THUNDER_HOST spherical_coordinate_system_impl_t::get_neighbor_tree_and_face(
      int itree
    , std::array<size_t,THUNDER_NSPACEDIM> const& ijk ) 
{
    using namespace thunder ; 
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ;
    int ngz = amr::get_n_ghosts() ; 

    int iface ; 
    int nghost = 0 ; 
    if ( ijk[0] < ngz ) {
        iface = 0 ; 
        ++nghost ; 
    }
    if (ijk[0] > nx+ngz-1 ){
        iface = 1 ;
        ++nghost ; 
    }
    if ( ijk[1] < ngz ) {
        iface = 2 ;
        ++nghost ; 
    }
    if (ijk[1] >= nx+ngz-1) {
        iface = 3 ; 
        ++nghost ; 
    }
    #ifdef THUNDER_3D 
    if ( ijk[2] < ngz ) {
        iface = 4 ;
        ++nghost ; 
    }
    if (ijk[2] > nx+ngz-1) {
        iface = 5 ; 
        ++nghost ; 
    }
    #endif 
    if(nghost != 1) { // corner neighbor or no neighbor at all 
        return std::make_tuple(-1,-1,-1) ; 
    }
    auto& conn = amr::connectivity::get() ;
    return std::make_tuple(
        conn.tree_to_tree(itree,iface),
        conn.tree_to_face(itree,iface),
        iface
    ) ; 
}

std::tuple<int, int8_t, int8_t>
THUNDER_HOST spherical_coordinate_system_impl_t::get_neighbor_tree_and_face(
      int itree
    , std::array<double,THUNDER_NSPACEDIM> const& lcoords 
    , bool check_exact_boundary) 
{
    using namespace thunder ; 
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ;
    int ngz = amr::get_n_ghosts() ; 
    ASSERT_DBG(is_outside_tree(lcoords,check_exact_boundary), "In get neighbor tree: lcoords not outside tree.") ; 
    int iface ; 
    int nghost = 0 ; 
    if ( lcoords[0] < 0 ) {
        iface = 0 ; 
        ++nghost ; 
    }
    if ((check_exact_boundary and lcoords[0] >= 1) or (!check_exact_boundary and lcoords[0] > 1)){
        iface = 1 ;
        ++nghost ; 
    }
    if ( lcoords[1] < 0 ) {
        iface = 2 ;
        ++nghost ; 
    }
    if ((check_exact_boundary and lcoords[1] >= 1) or (!check_exact_boundary and lcoords[1] > 1)) {
        iface = 3 ; 
        ++nghost ; 
    }
    #ifdef THUNDER_3D 
    if ( lcoords[2] < 0 ) {
        iface = 4 ;
        ++nghost ; 
    }
    if ((check_exact_boundary and lcoords[2] >= 1) or (!check_exact_boundary and lcoords[2] > 1)) {
        iface = 5 ; 
        ++nghost ; 
    }
    #endif 
    if(nghost > 1) { // corner neighbor
        return std::make_tuple(-1,-1,-1) ; 
    }
    auto& conn = amr::connectivity::get() ;
    return std::make_tuple(
        conn.tree_to_tree(itree,iface),
        conn.tree_to_face(itree,iface),
        iface
    ) ; 
}

std::array<double, THUNDER_NSPACEDIM> 
THUNDER_HOST spherical_coordinate_system_impl_t::get_logical_coordinates_buffer_zone(
      int itree
    , std::array<double, THUNDER_NSPACEDIM> const& lcoords ) 
{
    using namespace thunder ;
    int itree_b ; 
    int8_t iface, iface_b ;
    std::tie(itree_b,iface_b,iface) =
        get_neighbor_tree_and_face(itree, lcoords, true) ; 
    // corner neighbor -- unused 
    if(itree_b==-1){
        return {VEC(1,1,1)} ; 
    }
    
    std::array<double, THUNDER_NSPACEDIM> lcoords_b ; 
    
    EXPR(
    double const dl = 
        	(iface==0) * std::fabs(lcoords[0]) 
         +  (iface==1) * (lcoords[0] - 1.),
         +  (iface==2) * std::fabs(lcoords[1]) 
         +  (iface==3) * (lcoords[1] - 1.),
         +  (iface==4) * std::fabs(lcoords[2]) 
         +  (iface==5) * (lcoords[2] - 1.) 
    ) ;
    EXPR(
    double const dh = 
            (iface/2==0) * lcoords[1],
          + (iface/2==1) * lcoords[0],
          + (iface/2==2) * lcoords[0]
    ) ;
    #ifdef THUNDER_3D 
    EXPR(
    double const dt =
            (iface/2==0) * lcoords[2], 
          + (iface/2==1) * lcoords[2],
          + (iface/2==2) * lcoords[1]
    ) ;
    #endif
    EXPR(
    lcoords_b[0] = 
          (iface_b==0) * dl 
        + (iface_b==1) * (1.-dl),
        + (iface_b/2==1) * dh, 
        + (iface_b/2==2) * dh 
    ) ;
    EXPR(
    lcoords_b[1] = 
          (iface_b==2) * dl 
        + (iface_b==3) * (1.-dl),
        + (iface_b/2==0) * dh, 
        + (iface_b/2==2) * dt 
    ) ;
    #ifdef THUNDER_3D
    EXPR(
    lcoords_b[2] = 
          (iface_b==4) * dl 
        + (iface_b==5) * (1.-dl),
        + (iface_b/2==0) * dt, 
        + (iface_b/2==1) * dt 
    ) ;
    #endif 
    for(int idir=0; idir<THUNDER_NSPACEDIM; ++idir){
        ASSERT_DBG(
            lcoords_b[idir] >=0 and lcoords_b[idir] <=1,
            "Out of bounds logical coordinates "
            EXPR(<< lcoords[0], << ", " << lcoords[1] ,<< ", " << lcoords[2])<< '\n'
            EXPR(<< lcoords_b[0], << ", " << lcoords_b[1] ,<< ", " << lcoords_b[2])<< '\n'
            EXPR(<< dl, << ", " << dh ,<< ", " << dt)<< '\n'
            << iface << ", " << iface_b 
        ) ; 
    }
    return lcoords_b ; 
}

double
THUNDER_HOST spherical_coordinate_system_impl_t::get_cell_volume_buffer_zone(
      int itree
    , int64_t q
    , std::array<double, THUNDER_NSPACEDIM> const& lcoords
    , std::array<double, THUNDER_NSPACEDIM> const& dxl ) 
{
    using namespace thunder ;

    int    itree_b  ;
    int8_t dummy1, dummy2 ; 
    std::tie(itree_b,dummy1,dummy2) =
        get_neighbor_tree_and_face(itree,lcoords,true) ; 
    if(itree_b==-1){ return 1. ;}
    auto lcoords_b = get_logical_coordinates_buffer_zone(
        itree,
        lcoords
    ) ; 
    double Vol = 0;
    if( itree_b == 0 ) {
        Vol = math::int_pow<THUNDER_NSPACEDIM>(2.*_L) * EXPR(dxl[0],*dxl[1],*dxl[2]) ; 
    } else if( (itree_b-1)/P4EST_FACES == 0 ) {
        #ifndef THUNDER_3D 
        Vol = dVol_sph(_L,_Ri 
                , VECD(dxl[1],dxl[2]),dxl[0]
                , VECD(lcoords_b[1],lcoords_b[2]),lcoords_b[0] ) ; 
        #else 
        Vol = detail::get_cell_volume_3D<5UL>(0.,1.,_L,_Ri,lcoords_b[0],lcoords_b[1],lcoords_b[2],dxl[0],dxl[1],dxl[2]) ;
        #endif 
    } else {
        if( _use_logr ){
            #ifndef THUNDER_3D 
            Vol = dVol_sph_log(_Ri,_Ro
                    , VECD(dxl[1],dxl[2]),dxl[0]
                    , VECD(lcoords_b[1],lcoords_b[2]),lcoords_b[0] ) ; 
            #else 
            Vol = detail::get_cell_volume_3D_log(_Ri,_Ro,lcoords_b[0],lcoords_b[1],lcoords_b[2],dxl[0],dxl[1],dxl[2]) ;
            #endif 
        } else { 
            #ifndef THUNDER_3D 
            Vol = dVol_sph_ext(_Ri,_Ro
                    , VECD(dxl[1],dxl[2]),dxl[0]
                    , VECD(lcoords_b[1],lcoords_b[2]),lcoords_b[0] ) ;
            #else 
            Vol = detail::get_cell_volume_3D<5UL>(1.,1.,_Ri,_Ro,lcoords_b[0],lcoords_b[1],lcoords_b[2],dxl[0],dxl[1],dxl[2]) ;
            #endif 
        }
    }
    return Vol ; 

}


std::array<double, THUNDER_NSPACEDIM>
THUNDER_HOST spherical_coordinate_system_impl_t::get_physical_coordinates_cart(
    double L,
    std::array<double, THUNDER_NSPACEDIM> const& lcoords )
{
    return {VEC(
            (lcoords[0] * 2. - 1.) * L,
            (lcoords[1] * 2. - 1.) * L,
            (lcoords[2] * 2. - 1.) * L 
        )} ;
}

std::array<double, THUNDER_NSPACEDIM>
THUNDER_HOST spherical_coordinate_system_impl_t::get_physical_coordinates_sph(
      int irot   
    , double Ri
    , double Ro 
    , std::array<double,2> const& F 
    , std::array<double,2> const& S
    , std::array<double, THUNDER_NSPACEDIM> const& lcoords
    , bool logr )
{
    auto const eta = (2*lcoords[1]-1); 
    #ifdef THUNDER_3D 
    auto const xi = (2*lcoords[2]-1) ; 
    #endif
    auto const one_over_rho = 1./sqrt(EXPR( 1
                                        , + math::int_pow<2>(eta)
                                        , + math::int_pow<2>(xi) )) ;
    auto const z = get_zeta(lcoords[0], one_over_rho, F, S, logr);   
    std::array<double, THUNDER_NSPACEDIM> pcoords =
        { VEC( z
             , z * eta
             , z * xi )};
    return detail::apply_discrete_rotation(pcoords, irot ) ; 
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

#if 0
double THUNDER_HOST
spherical_coordinate_system_impl_t::get_jacobian_determinant_sph(
      double const& si, double const& so 
    , double const& r1, double const& r2
    , VEC(double const& zeta, double const& eta, double const& xi) )
{
    using namespace thunder; 
    #ifdef THUNDER_3D 
    /* 
    std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> const J { 
        Jac_sph_3D_00(r1,r2, eta, si,so, xi), Jac_sph_3D_01(r1,r2, eta, si,so, xi,zeta), Jac_sph_3D_02(r1,r2, eta, si,so, xi,zeta),
        Jac_sph_3D_10(r1,r2, eta, si,so, xi), Jac_sph_3D_11(r1,r2, eta, si,so, xi,zeta), Jac_sph_3D_12(r1,r2, eta, si,so, xi,zeta),
        Jac_sph_3D_20(r1,r2, eta, si,so, xi), Jac_sph_3D_21(r1,r2, eta, si,so, xi,zeta), Jac_sph_3D_22(r1,r2, eta, si,so, xi,zeta)
    };
    */
    #else 
    std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> const J { 
        Jac_sph_2D_00(r1,r2, eta, si,so), Jac_sph_2D_01(r1,r2, eta, si,so, zeta), 
        Jac_sph_2D_10(r1,r2, eta, si,so), Jac_sph_2D_11(r1,r2, eta, si,so, zeta)
    };
    #endif 
    //return utils::det<THUNDER_NSPACEDIM>(J) ; 
    return J_sph_3D(r1,r2,eta,si,so,xi,zeta) ; 
}



double THUNDER_HOST
spherical_coordinate_system_impl_t::get_jacobian_determinant_sph_log(
      double const& r1, double const& r2
    , VEC(double const& zeta, double const& eta, double const& xi) )
{
    using namespace thunder;
    #ifdef THUNDER_3D 
    /*
    std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> const J {
                Jac_sph_log_3D_00(r1,r2, eta, xi,zeta), Jac_sph_log_3D_01(r1,r2, eta, xi,zeta), Jac_sph_log_3D_02(r1,r2, eta, xi,zeta),
                Jac_sph_log_3D_10(r1,r2, eta, xi,zeta), Jac_sph_log_3D_11(r1,r2, eta, xi,zeta), Jac_sph_log_3D_12(r1,r2, eta, xi,zeta),
                Jac_sph_log_3D_20(r1,r2, eta, xi,zeta), Jac_sph_log_3D_21(r1,r2, eta, xi,zeta), Jac_sph_log_3D_22(r1,r2, eta, xi,zeta)
            };*/
    #else 
    std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> const J {
                Jac_sph_log_2D_00(r1,r2, eta,zeta), Jac_sph_log_2D_01(r1,r2, eta, zeta), 
                Jac_sph_log_2D_10(r1,r2, eta,zeta), Jac_sph_log_2D_11(r1,r2, eta, zeta)
            };
    #endif 
    //return utils::det<THUNDER_NSPACEDIM>(J) ; 
    return J_sph_log_3D(r1,r2,eta,xi,zeta) ; 
}
#endif 
#ifdef THUNDER_3D
double THUNDER_HOST
spherical_coordinate_system_impl_t::get_volume_element_sph(
      double const& r1, double const& r2
    , double const& zeta0, double const& dzeta 
    , double const& xi0, double const& dxi 
    , double const& eta )
{
    using namespace thunder; 
    #ifdef THUNDER_3D 
    /* 
    std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> const J { 
        Jac_sph_3D_00(r1,r2, eta, si,so, xi), Jac_sph_3D_01(r1,r2, eta, si,so, xi,zeta), Jac_sph_3D_02(r1,r2, eta, si,so, xi,zeta),
        Jac_sph_3D_10(r1,r2, eta, si,so, xi), Jac_sph_3D_11(r1,r2, eta, si,so, xi,zeta), Jac_sph_3D_12(r1,r2, eta, si,so, xi,zeta),
        Jac_sph_3D_20(r1,r2, eta, si,so, xi), Jac_sph_3D_21(r1,r2, eta, si,so, xi,zeta), Jac_sph_3D_22(r1,r2, eta, si,so, xi,zeta)
    };
    */
    #endif 
    //return utils::det<THUNDER_NSPACEDIM>(J) ; 
    return dVol_sph_3D(r1,r2,dxi,dzeta,eta,xi0,zeta0) ; 
}

double THUNDER_HOST
spherical_coordinate_system_impl_t::get_volume_element_sph_ext(
      double const& r1, double const& r2
    , double const& zeta0, double const& dzeta 
    , double const& xi0, double const& dxi 
    , double const& eta )
{
    using namespace thunder; 
    #ifdef THUNDER_3D 
    /* 
    std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> const J { 
        Jac_sph_3D_00(r1,r2, eta, si,so, xi), Jac_sph_3D_01(r1,r2, eta, si,so, xi,zeta), Jac_sph_3D_02(r1,r2, eta, si,so, xi,zeta),
        Jac_sph_3D_10(r1,r2, eta, si,so, xi), Jac_sph_3D_11(r1,r2, eta, si,so, xi,zeta), Jac_sph_3D_12(r1,r2, eta, si,so, xi,zeta),
        Jac_sph_3D_20(r1,r2, eta, si,so, xi), Jac_sph_3D_21(r1,r2, eta, si,so, xi,zeta), Jac_sph_3D_22(r1,r2, eta, si,so, xi,zeta)
    };
    */
    #endif 
    //return utils::det<THUNDER_NSPACEDIM>(J) ; 
    return dVol_sph_3D_ext(r1,r2,dxi,dzeta,eta,xi0,zeta0) ; 
}

double THUNDER_HOST
spherical_coordinate_system_impl_t::get_volume_element_sph_ext_log(
      double const& r1, double const& r2
    , double const& zeta0, double const& dzeta 
    , double const& xi0, double const& dxi 
    , double const& eta )
{
    using namespace thunder;
    #ifdef THUNDER_3D 
    /*
    std::array<double,THUNDER_NSPACEDIM*THUNDER_NSPACEDIM> const J {
                Jac_sph_log_3D_00(r1,r2, eta, xi,zeta), Jac_sph_log_3D_01(r1,r2, eta, xi,zeta), Jac_sph_log_3D_02(r1,r2, eta, xi,zeta),
                Jac_sph_log_3D_10(r1,r2, eta, xi,zeta), Jac_sph_log_3D_11(r1,r2, eta, xi,zeta), Jac_sph_log_3D_12(r1,r2, eta, xi,zeta),
                Jac_sph_log_3D_20(r1,r2, eta, xi,zeta), Jac_sph_log_3D_21(r1,r2, eta, xi,zeta), Jac_sph_log_3D_22(r1,r2, eta, xi,zeta)
            };*/
    #endif 
    //return utils::det<THUNDER_NSPACEDIM>(J) ; 
    return dVol_sph_3D_log(r1,r2,dxi,dzeta,eta,xi0,zeta0) ; 
}
#endif 
double THUNDER_HOST
spherical_coordinate_system_impl_t::get_surface_element_sph(
      int8_t iface
    , double const& si, double const& so 
    , double const& ri, double const& ro
    , VEC(double const& zeta, double const& eta, double const& xi) )
{
    #ifdef THUNDER_3D 
    std::array<double,4> J ;
    switch(iface/2) {
        case 0:
        J[0 + 2*0] =  dA_3D_zeta_00(ri,ro,eta,si,so,xi,zeta) ; 
        J[1 + 2*0] =  dA_3D_zeta_01(ri,ro,eta,si,so,xi,zeta) ; 
        J[0 + 2*1] =  dA_3D_zeta_10(ri,ro,eta,si,so,xi,zeta) ; 
        J[1 + 2*1] =  dA_3D_zeta_11(ri,ro,eta,si,so,xi,zeta) ; 
        break ;
        case 1:
        J[0 + 2*0] =  dA_3D_eta_00(ri,ro,eta,si,so,xi,zeta) ; 
        J[1 + 2*0] =  dA_3D_eta_01(ri,ro,eta,si,so,xi,zeta) ; 
        J[0 + 2*1] =  dA_3D_eta_10(ri,ro,eta,si,so,xi,zeta) ; 
        J[1 + 2*1] =  dA_3D_eta_11(ri,ro,eta,si,so,xi,zeta) ; 
        break ; 
        case 2:
        J[0 + 2*0] =  dA_3D_xi_00(ri,ro,eta,si,so,xi,zeta) ; 
        J[1 + 2*0] =  dA_3D_xi_01(ri,ro,eta,si,so,xi,zeta) ; 
        J[0 + 2*1] =  dA_3D_xi_10(ri,ro,eta,si,so,xi,zeta) ; 
        J[1 + 2*1] =  dA_3D_xi_11(ri,ro,eta,si,so,xi,zeta) ; 
        break ; 
    }
    return sqrt(utils::det<THUNDER_NSPACEDIM-1>(J)) ; 
    #else 
    switch(iface/2) {
        case 0:
        return dA_2D_zeta(ri,ro,eta,si,so,zeta) ; 
        case 1:
        return dA_2D_eta(ri,ro,eta,si,so,zeta) ; 
    }
    #endif 
}



double THUNDER_HOST
spherical_coordinate_system_impl_t::get_surface_element_sph_log(
      int8_t iface 
    , double const& ri, double const& ro
    , VEC(double const& zeta, double const& eta, double const& xi) )
{
    #ifdef THUNDER_3D 
    std::array<double,4> J ;
    switch(iface/2) {
        case 0:
        J[0 + 2*0] =  dA_3D_zeta_log_00(ri,ro,eta,xi,zeta) ; 
        J[1 + 2*0] =  dA_3D_zeta_log_01(ri,ro,eta,xi,zeta) ; 
        J[0 + 2*1] =  dA_3D_zeta_log_10(ri,ro,eta,xi,zeta) ; 
        J[1 + 2*1] =  dA_3D_zeta_log_11(ri,ro,eta,xi,zeta) ; 
        break ;
        case 1:
        J[0 + 2*0] =  dA_3D_eta_log_00(ri,ro,eta,xi,zeta) ; 
        J[1 + 2*0] =  dA_3D_eta_log_01(ri,ro,eta,xi,zeta) ; 
        J[0 + 2*1] =  dA_3D_eta_log_10(ri,ro,eta,xi,zeta) ; 
        J[1 + 2*1] =  dA_3D_eta_log_11(ri,ro,eta,xi,zeta) ; 
        break ; 
        case 2:
        J[0 + 2*0] =  dA_3D_xi_log_00(ri,ro,eta,xi,zeta) ; 
        J[1 + 2*0] =  dA_3D_xi_log_01(ri,ro,eta,xi,zeta) ; 
        J[0 + 2*1] =  dA_3D_xi_log_10(ri,ro,eta,xi,zeta) ; 
        J[1 + 2*1] =  dA_3D_xi_log_11(ri,ro,eta,xi,zeta) ; 
        break ; 
    }
    return sqrt(utils::det<THUNDER_NSPACEDIM-1>(J)) ; 
    #else 
    switch(iface/2) {
        case 0:
        return dA_2D_zeta_log(ri,ro,eta,zeta) ; 
        case 1:
        return dA_2D_eta_log(ri,ro,eta,zeta) ; 
    }
    #endif 
}



}