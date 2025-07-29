/**
 * @file spherical_coordinate_system.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Differences
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

#include <grace/amr/grace_amr.hh> 
#include <grace/coordinates/cell_volume_kernels.h>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/coordinates/rotation_matrices.hh>
#include <grace/coordinates/spherical_coordinate_systems.hh>
#include <grace/coordinates/spherical_device_inlines.hh>
#include <grace/coordinates/spherical_coordinate_jacobian_utils.hh>
#include <grace/coordinates/surface_elements/cell_surfaces_2D.hh>
#include <grace/coordinates/surface_elements/cell_surfaces_3D.hh>
#include <grace/coordinates/volume_elements/vol_sph_3D.hh>
#include <grace/coordinates/volume_elements/vol_sph_3D_log.hh>
#include <grace/coordinates/volume_elements/dVol_sph_3D_log_analytic.hh>
#include <grace/coordinates/volume_elements/cell_volume_helpers.hh>
#include <grace/coordinates/surface_elements/cell_surface_helpers.hh>
#include <grace/coordinates/line_elements/line_element_helpers.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh> 

#include <array> 

namespace grace { 
    
std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_physical_coordinates(
      int const itree
    , std::array<double, GRACE_NSPACEDIM> const& lcoords ) const
{
    auto lcoords_b = lcoords ; 
    auto itree_b   = itree   ; 
    /* First we handle buffer zones */
    if( spherical_coordinate_system_impl_t::is_outside_tree(lcoords) ) {
        /* Check if the boundary is internal */
        if( !spherical_coordinate_system_impl_t::is_physical_boundary(lcoords,itree) ) {
            /* In this case we just need to transfer the coordinates */
            /* and proceed as normal.                                */
            int8_t iface, iface_b ; 
            std::tie(itree_b,iface_b,iface) = get_neighbor_tree_and_face(itree,lcoords) ; 
            lcoords_b = spherical_coordinate_system_impl_t::get_logical_coordinates_buffer_zone(itree, itree_b,iface,iface_b,lcoords);
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

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_physical_coordinates(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
    , int64_t q 
    , std::array<double, GRACE_NSPACEDIM> const& cell_coordinates
    , bool use_ghostzones ) const
{
    using namespace grace ; 
    int64_t itree = amr::get_quadrant_owner(q)   ; 
    return get_physical_coordinates(itree, get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones)) ; 
}


std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_physical_coordinates(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
    , int64_t q 
    , bool use_ghostzones ) const
{
    return get_physical_coordinates(ijk,q,{VEC(0.5,0.5,0.5)},use_ghostzones);
} 


std::array<double, GRACE_NSPACEDIM>
GRACE_HOST spherical_coordinate_system_impl_t::get_logical_coordinates(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
    , int64_t q 
    , std::array<double, GRACE_NSPACEDIM> const& cell_coordinates
    , bool use_ghostzones) const
{
    using namespace grace ;

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

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_logical_coordinates(
      int itree
    , std::array<double, GRACE_NSPACEDIM> const& physical_coordinates ) const
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

double GRACE_HOST 
spherical_coordinate_system_impl_t::get_jacobian(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    return utils::det<GRACE_NSPACEDIM>(
        get_jacobian_matrix(ijk,q,cell_coordinates,use_ghostzones) 
    ) ; 
}

double GRACE_HOST 
spherical_coordinate_system_impl_t::get_jacobian(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    return utils::det<GRACE_NSPACEDIM>(
        get_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) 
    ) ; 
}

double GRACE_HOST 
spherical_coordinate_system_impl_t::get_jacobian(
      int itree
    , std::array<double,GRACE_NSPACEDIM> const& lcoords) const
{
    return utils::det<GRACE_NSPACEDIM>(
        get_jacobian_matrix(itree,lcoords) 
    ) ; 
}

double GRACE_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    return utils::det<GRACE_NSPACEDIM>(
        get_inverse_jacobian_matrix(ijk,q,cell_coordinates,use_ghostzones)
    ) ; 
}

double GRACE_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    return utils::det<GRACE_NSPACEDIM>(
        get_inverse_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) 
    ) ; 
}

double GRACE_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian(
      int itree
    , std::array<double,GRACE_NSPACEDIM> const& lcoords) const
{
    return utils::det<GRACE_NSPACEDIM>(
        get_inverse_jacobian_matrix(itree,lcoords) 
    ) ; 
}

std::array<double,GRACE_NSPACEDIM*GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_jacobian_matrix(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{ 
    using namespace grace ;
    int itree = amr::get_quadrant_owner(q) ;
    return get_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) ; 
}

std::array<double,GRACE_NSPACEDIM*GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_jacobian_matrix(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    using namespace grace ; 
    auto lcoords =  get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_jacobian_matrix(itree,lcoords) ; 
} 

std::array<double,GRACE_NSPACEDIM*GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_jacobian_matrix(
      int itree
    , std::array<double,GRACE_NSPACEDIM> const& lcoords ) const
{
    using namespace grace; 
    int itree_b = itree ; 
    auto lcoords_b = lcoords ; 
    if( spherical_coordinate_system_impl_t::is_outside_tree(lcoords) and !spherical_coordinate_system_impl_t::is_physical_boundary(lcoords,itree) ) {
        int8_t iface_b, iface ; 
        std::tie(itree_b,iface_b,iface) =
            get_neighbor_tree_and_face(itree,lcoords) ; 
        
        lcoords_b = spherical_coordinate_system_impl_t::get_logical_coordinates_buffer_zone(
            itree,itree_b,iface,iface_b,
            lcoords
        ) ; 
    }
    double si, so; 
    double r1,r2 ; 
    if( itree_b == 0 ) {
        auto J = utils::identity_matrix<GRACE_NSPACEDIM>() ; 
        for( auto& x: J ) x /= (2.*_L) ; 
    } else if ( (itree_b-1)/P4EST_FACES == 0 ) { 
        si=0; so=1.; r1=_L; r2=_Ri;
    } else {
        si = 1.; so=1; 
        r1 = _Ri; r2=_Ro;
        if( _use_logr ) {
            #ifdef GRACE_3D 
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
    #ifdef GRACE_3D 
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


std::array<double,GRACE_NSPACEDIM*GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian_matrix(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{ 
    using namespace grace ;
    int itree = amr::get_quadrant_owner(q) ;
    return get_inverse_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) ; 
}

std::array<double,GRACE_NSPACEDIM*GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian_matrix(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    using namespace grace ; 
    auto lcoords =  get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_inverse_jacobian_matrix(itree,lcoords) ; 
} 

std::array<double,GRACE_NSPACEDIM*GRACE_NSPACEDIM> GRACE_HOST 
spherical_coordinate_system_impl_t::get_inverse_jacobian_matrix(
      int itree
    , std::array<double,GRACE_NSPACEDIM> const& lcoords ) const
{
    using namespace grace; 
    int itree_b = itree ; 
    auto lcoords_b = lcoords ; 
    if( spherical_coordinate_system_impl_t::is_outside_tree(lcoords) and !spherical_coordinate_system_impl_t::is_physical_boundary(lcoords,itree) ) {
        int8_t iface_b, iface ; 
        std::tie(itree_b,iface_b,iface) =
            get_neighbor_tree_and_face(itree,lcoords) ; 
        
        lcoords_b = spherical_coordinate_system_impl_t::get_logical_coordinates_buffer_zone(
            itree,itree_b,iface,iface_b,
            lcoords
        ) ; 
    }
    double si, so; \
    double r1, r2 ;
    if( itree_b == 0 ) {
        auto J = utils::identity_matrix<GRACE_NSPACEDIM>() ; 
        for( auto& x: J ) x *= (2.*_L) ; 
    } else if ( (itree_b-1)/P4EST_FACES == 0 ) { 
        si=0; so=1.; 
        r1 = _L; r2 = _Ri ; 
    } else {
        si = 1.; so=1; 
        r1 = _Ri; r2 = _Ro; 
        if( _use_logr ) {
            #ifdef GRACE_3D 
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
    #ifdef GRACE_3D 
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
GRACE_HOST spherical_coordinate_system_impl_t::get_cell_volume(
    std::array<size_t, GRACE_NSPACEDIM> const& ijk 
  , int64_t q
  , bool use_ghostzones) const
{
    using namespace grace ;
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
GRACE_HOST spherical_coordinate_system_impl_t::get_cell_volume(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones)  const
{
    auto lcoords = get_logical_coordinates(ijk,q,{VEC(0.,0.,0.)},use_ghostzones) ; 
    return  get_cell_volume(lcoords,itree,dxl,use_ghostzones) ; 
}  

double
GRACE_HOST spherical_coordinate_system_impl_t::get_cell_volume(
      std::array<double, GRACE_NSPACEDIM> const& lcoords
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones)  const
{
    using namespace grace ;
    using namespace grace::detail  ; 
    int ngz = grace::amr::get_n_ghosts();
    if( spherical_coordinate_system_impl_t::is_outside_tree(lcoords,true) and !spherical_coordinate_system_impl_t::is_physical_boundary(lcoords,itree,true) and use_ghostzones) {
        return get_cell_volume_buffer_zone(itree,lcoords,dxl);
    }
    if( itree == 0 ) {
        return math::int_pow<GRACE_NSPACEDIM>(2.*_L) * EXPR(dxl[0],*dxl[1],*dxl[2]) ; 
    } else if( (itree-1)/P4EST_FACES == 0 ) {
        return detail::get_cell_volume_int<5UL>(_L,_Ri,VEC(lcoords[0],lcoords[1],lcoords[2])
                                              ,VEC(dxl[0],dxl[1],dxl[2])  ) ; 
    } else {
        if( _use_logr ){  
            return detail::get_cell_volume_log(_Ri,_Ro,VEC(lcoords[0],lcoords[1],lcoords[2]),VEC(dxl[0],dxl[1],dxl[2]));
        } else { 
             return detail::get_cell_volume_ext( _Ri,_Ro
                                               , VEC(lcoords[0],lcoords[1],lcoords[2])
                                               , VEC(dxl[0],dxl[1],dxl[2])  ) ; 
        }
    }
}

double
GRACE_HOST spherical_coordinate_system_impl_t::get_cell_face_surface(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q
    , int8_t face 
    , bool use_ghostzones) const 
{
    using namespace grace ;
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int itree = amr::get_quadrant_owner(q)   ; 
    amr::quadrant_t quad = amr::get_quadrant(itree,q) ; 

    auto const dx_quad  = 1./(1<<quad.level()) ; 
    auto const qcoords = quad.qcoords()     ; 

    EXPR(
    auto const dx_cell = dx_quad / nx ;, 
    auto const dy_cell = dx_quad / ny ;,
    auto const dz_cell = dx_quad / nz ;
    )
    return get_cell_face_surface(ijk,q,face,itree,{VEC(dx_cell,dy_cell,dz_cell)},use_ghostzones) ; 
}

double
GRACE_HOST spherical_coordinate_system_impl_t::get_cell_face_surface(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q
    , int8_t face 
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const 
{
    std::array<double, GRACE_NSPACEDIM> cell_coordinates 
    { VEC( 0.,0.,0.) } ;
    auto lcoords = get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_cell_face_surface(lcoords,face,itree,dxl,use_ghostzones) ; 
}
/* NB: by convention index i,j,k and face idx if corresponds to face */
/*     i-delta_{i,if}/2, j-delta_{j,if}/2, k-delta_{k,if}/2          */
double
GRACE_HOST spherical_coordinate_system_impl_t::get_cell_face_surface(
      std::array<double, GRACE_NSPACEDIM> const& lcoords
    , int8_t face 
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const 
{
    using namespace grace::detail ;
    int ngz = grace::amr::get_n_ghosts();
    if( spherical_coordinate_system_impl_t::is_outside_tree(lcoords) and !spherical_coordinate_system_impl_t::is_physical_boundary(lcoords,itree) and use_ghostzones) {
        return get_cell_face_surface_buffer_zone(face,itree,lcoords,dxl); 
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
        return math::int_pow<GRACE_NSPACEDIM-1>(2.*_L) * EXPRD(dh,*dt) ; 
    } else if( (itree-1)/P4EST_FACES == 0 ) {
        if( face / 2 == 0 ) {
            return get_cell_surface_zeta_int<5UL>(_L,_Ri
                                       ,VEC(lcoords[0],lcoords[1],lcoords[2])
                                       ,VECD(dxl[1],dxl[2]));

        } else if ( face/2 == 1 ) { 
            return get_cell_surface_eta_int(_L,_Ri
                                       ,VEC(lcoords[0],lcoords[1],lcoords[2])
                                       ,VECD(dxl[0],dxl[2]));                        
        } 
        #ifdef GRACE_3D 
        else if ( face/2 == 2) {
            return get_cell_surface_xi_int(_L,_Ri
                                       ,VEC(lcoords[0],lcoords[1],lcoords[2])
                                       ,VECD(dxl[0],dxl[1]));
        }
        #endif 
        else {
            ERROR("Invalid face code " << face ) ; 
        }
    } else {
        if( _use_logr ){
            if( face / 2 == 0 ) {
            return get_cell_surface_zeta_log(_Ri,_Ro
                                       ,VEC(lcoords[0],lcoords[1],lcoords[2])
                                       ,VECD(dxl[1],dxl[2]));

            } else if ( face/2 == 1 ) { 
                return get_cell_surface_eta_log(_Ri,_Ro
                                        ,VEC(lcoords[0],lcoords[1],lcoords[2])
                                        ,VECD(dxl[0],dxl[2]));                        
            } 
            #ifdef GRACE_3D 
            else if ( face/2 == 2) {
                return get_cell_surface_xi_log(_Ri,_Ro
                                        ,VEC(lcoords[0],lcoords[1],lcoords[2])
                                        ,VECD(dxl[0],dxl[1]));
            }
            #endif 
            else {
                ERROR("Invalid face code " << face ) ; 
            }
        } else { 
            if( face / 2 == 0 ) {
            return get_cell_surface_zeta_ext(_Ri,_Ro
                                       ,VEC(lcoords[0],lcoords[1],lcoords[2])
                                       ,VECD(dxl[1],dxl[2]));

            } else if ( face/2 == 1 ) { 
                return get_cell_surface_eta_ext(_Ri,_Ro
                                        ,VEC(lcoords[0],lcoords[1],lcoords[2])
                                        ,VECD(dxl[0],dxl[2]));                        
            } 
            #ifdef GRACE_3D 
            else if ( face/2 == 2) {
                return get_cell_surface_xi_ext(_Ri,_Ro
                                        ,VEC(lcoords[0],lcoords[1],lcoords[2])
                                        ,VECD(dxl[0],dxl[1]));
            }
            #endif 
            else {
                ERROR("Invalid face code " << face ) ; 
            }
        }
    }
} 
#ifdef GRACE_3D 
double 
GRACE_HOST 
spherical_coordinate_system_impl_t::get_cell_edge_length(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q
    , int8_t edge
    , bool use_ghostzones) const
{
    using namespace grace ;
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int itree = amr::get_quadrant_owner(q)   ; 
    amr::quadrant_t quad = amr::get_quadrant(itree,q) ; 

    auto const dx_quad  = 1./(1<<quad.level()) ; 
    auto const qcoords = quad.qcoords()     ; 

    EXPR(
    auto const dx_cell = dx_quad / nx ;, 
    auto const dy_cell = dx_quad / ny ;,
    auto const dz_cell = dx_quad / nz ;
    )
 
    return get_cell_edge_length(ijk,q,edge,itree,{VEC(dx_cell,dy_cell,dz_cell)},use_ghostzones) ; 
}

double 
GRACE_HOST 
spherical_coordinate_system_impl_t::get_cell_edge_length(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q
    , int8_t edge 
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const
{
    std::array<double, GRACE_NSPACEDIM> cell_coordinates {
        VEC(0.,0.,0.)
    } ; 
    auto lcoords = get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_cell_edge_length(lcoords,edge,itree,dxl,use_ghostzones) ; 
}
/* NB: by convention index i,j,k and edge idx ie corresponds to edge    */
/*     i-1/2+delta_{i,ie}/2, j-1/2+delta_{j,ie}/2, k-1/2+delta_{k,ie}/2 */
double 
GRACE_HOST 
spherical_coordinate_system_impl_t::get_cell_edge_length(
      std::array<double, GRACE_NSPACEDIM> const & lcoords 
    , int8_t edge 
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const
{
    using namespace grace::detail ;
    if( spherical_coordinate_system_impl_t::is_outside_tree(lcoords) and !spherical_coordinate_system_impl_t::is_physical_boundary(lcoords,itree) and use_ghostzones) {
        return get_cell_edge_length_buffer_zone(edge,itree,lcoords,dxl); 
    }
    if ( itree == 0 ) {
        return 2.*_L * EXPR( (edge==0) * dxl[0], + (edge==1) * dxl[1], + (edge==2) * dxl[2] ) ;
    } else if ( (itree-1)/P4EST_FACES==0 ) {
        switch (edge) {
            case 0:
            return get_line_element_zeta_int(_L,_Ri,VEC(lcoords[0],lcoords[1],lcoords[2]),dxl[0]); 
            case 1:
            return get_line_element_eta_int<5UL>(_L,_Ri,VEC(lcoords[0],lcoords[1],lcoords[2]),dxl[1]);
            case 2:
            return get_line_element_xi_int<5UL>(_L,_Ri,VEC(lcoords[0],lcoords[1],lcoords[2]),dxl[2]);
            default:
            ERROR("Invalid edge code " << edge ) ; 
            return 0 ; // Compiler is throwing a warning without this.
        } 
    } else if ( (itree-1)/P4EST_FACES==1 ) {
        if( _use_logr ) {
            switch (edge) {
            case 0:
            return get_line_element_zeta_log(_Ri,_Ro,VEC(lcoords[0],lcoords[1],lcoords[2]),dxl[0]); 
            case 1:
            return get_line_element_eta_log(_Ri,_Ro,VEC(lcoords[0],lcoords[1],lcoords[2]),dxl[1]);
            case 2:
            return get_line_element_xi_log(_Ri,_Ro,VEC(lcoords[0],lcoords[1],lcoords[2]),dxl[2]);
            default:
            ERROR("Invalid edge code " << edge ) ; 
            return 0 ; // Compiler is throwing a warning without this.
            } 
        } else {
            switch (edge) {
            case 0:
            return get_line_element_zeta_ext(_Ri,_Ro,VEC(lcoords[0],lcoords[1],lcoords[2]),dxl[0]); 
            case 1:
            return get_line_element_eta_ext(_Ri,_Ro,VEC(lcoords[0],lcoords[1],lcoords[2]),dxl[1]);
            case 2:
            return get_line_element_xi_ext(_Ri,_Ro,VEC(lcoords[0],lcoords[1],lcoords[2]),dxl[2]);
            default:
            ERROR("Invalid edge code " << edge ) ; 
            return 0 ; // Compiler is throwing a warning without this.
            } 
        }
    } else {
        ERROR("Invalid tree index in get_edge_length.") ; 
        return 0 ; // Compiler is throwing a warning without this.
    }
}
#endif 
bool 
GRACE_HOST spherical_coordinate_system_impl_t::is_outside_tree(
    std::array<double, GRACE_NSPACEDIM> const& lcoords, bool check_exact_boundary
)
{
    return EXPR(
           lcoords[0] < 0 or (check_exact_boundary and lcoords[0] >= 1) or (!check_exact_boundary and lcoords[0] > 1), 
        or lcoords[1] < 0 or (check_exact_boundary and lcoords[1] >= 1) or (!check_exact_boundary and lcoords[1] > 1), 
        or lcoords[2] < 0 or (check_exact_boundary and lcoords[2] >= 1) or (!check_exact_boundary and lcoords[2] > 1) 
    ) ; 
}

bool 
GRACE_HOST spherical_coordinate_system_impl_t::is_physical_boundary(
    std::array<double,GRACE_NSPACEDIM> const& lcoords, int itree, bool check_exact_boundary) const 
{
    ASSERT_DBG(spherical_coordinate_system_impl_t::is_outside_tree(lcoords,check_exact_boundary), "In is_physical_boundary: lcoords not in buffer zone"); 
    int itree_b; int8_t iface,iface_b ; 
    std::tie(itree_b,iface_b,iface) = get_neighbor_tree_and_face(itree,lcoords,check_exact_boundary) ; 
    return (itree_b==itree) and (iface_b==iface) ; 
}


std::tuple<int, int8_t, int8_t>
GRACE_HOST spherical_coordinate_system_impl_t::get_neighbor_tree_and_face(
      int itree
    , std::array<size_t,GRACE_NSPACEDIM> const& ijk ) const
{
    using namespace grace ; 
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
    #ifdef GRACE_3D 
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
GRACE_HOST spherical_coordinate_system_impl_t::get_neighbor_tree_and_face(
      int itree
    , std::array<double,GRACE_NSPACEDIM> const& lcoords 
    , bool check_exact_boundary) const
{
    using namespace grace ; 
    ASSERT_DBG(spherical_coordinate_system_impl_t::is_outside_tree(lcoords,check_exact_boundary), "In get neighbor tree: lcoords not outside tree.") ; 
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
    #ifdef GRACE_3D 
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

std::array<double, GRACE_NSPACEDIM> 
GRACE_HOST spherical_coordinate_system_impl_t::get_logical_coordinates_buffer_zone(
      int itree, int itree_b, int8_t iface, int8_t iface_b
    , std::array<double, GRACE_NSPACEDIM> const& lcoords )
{
    using namespace grace ; 
    // corner neighbor -- unused 
    if(itree_b==-1){
        return {VEC(1,1,1)} ; 
    }
    std::array<double, GRACE_NSPACEDIM> lcoords_b ; 
    
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
    #ifdef GRACE_3D 
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
    #ifdef GRACE_3D
    EXPR(
    lcoords_b[2] = 
          (iface_b==4) * dl 
        + (iface_b==5) * (1.-dl),
        + (iface_b/2==0) * dt, 
        + (iface_b/2==1) * dt 
    ) ;
    #endif 
    for(int idir=0; idir<GRACE_NSPACEDIM; ++idir){
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
GRACE_HOST spherical_coordinate_system_impl_t::get_cell_volume_buffer_zone(
      int itree
    , std::array<double, GRACE_NSPACEDIM> const& lcoords
    , std::array<double, GRACE_NSPACEDIM> const& dxl ) const
{
    using namespace grace ;
    using namespace grace::detail ;
    int    itree_b  ;
    int8_t iface_b, iface ; 
    std::tie(itree_b,iface_b,iface) =
        get_neighbor_tree_and_face(itree,lcoords,true) ; 
    if(itree_b==-1){ return 1. ;}
    auto lcoords_b = spherical_coordinate_system_impl_t::get_logical_coordinates_buffer_zone(
        itree,itree_b,iface,iface_b,lcoords
    ) ; 
    int polarity = amr::connectivity::get().tree_to_tree_polarity(itree,iface);
    EXPR(
    lcoords_b[0] += polarity * dxl[0]*( (iface_b==0)-(iface_b==1) );,
    lcoords_b[1] += polarity * dxl[1]*( (iface_b==2)-(iface_b==3) );,
    lcoords_b[2] += polarity * dxl[2]*( (iface_b==4)-(iface_b==5) );
    )
    double Vol = 0;
    if( itree_b == 0 ) {
        Vol = math::int_pow<GRACE_NSPACEDIM>(2.*_L) * EXPR(dxl[0],*dxl[1],*dxl[2]) ; 
    } else if( (itree_b-1)/P4EST_FACES == 0 ) {
        Vol = detail::get_cell_volume_int<5UL>(_L,_Ri,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                              ,VEC(dxl[0],dxl[1],dxl[2])  ) ; 
    } else {
        if( _use_logr ){
            Vol = detail::get_cell_volume_log(_Ri,_Ro,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),VEC(dxl[0],dxl[1],dxl[2])); 
        } else { 
            Vol =  detail::get_cell_volume_ext( _Ri,_Ro
                                                , VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                                , VEC(dxl[0],dxl[1],dxl[2])  ) ; 
        }
    }
    return Vol ; 

}

double
GRACE_HOST spherical_coordinate_system_impl_t::get_cell_face_surface_buffer_zone(
      int8_t icell_face 
    , int itree 
    , std::array<double, GRACE_NSPACEDIM> const& lcoords
    , std::array<double, GRACE_NSPACEDIM> const& dxl ) const
{
    using namespace grace ;
    using namespace grace::detail ;
    int    itree_b  ;
    int8_t iface_b, iface ; 
    std::tie(itree_b,iface_b,iface) =
        get_neighbor_tree_and_face(itree,lcoords,true) ; 
    if(itree_b==-1){ return 1. ;}
    auto lcoords_b = spherical_coordinate_system_impl_t::get_logical_coordinates_buffer_zone(
        itree,itree_b,iface,iface_b,
        lcoords
    ) ; 
    int polarity = amr::connectivity::get().tree_to_tree_polarity(itree,iface);
    EXPR(
    lcoords_b[0] += polarity * dxl[0]*( (iface_b==0)-(iface_b==1) ) * (icell_face!=0);,
    lcoords_b[1] += polarity * dxl[1]*( (iface_b==2)-(iface_b==3) ) * (icell_face!=1);,
    lcoords_b[2] += polarity * dxl[2]*( (iface_b==4)-(iface_b==5) ) * (icell_face!=2);
    )
    double Surf = 0;
    if( itree_b == 0 ) {
        EXPRD(
        double const dh = EXPR(
               (icell_face/2==0)*dxl[1],
             + (icell_face/2==1)*dxl[0],
             + (icell_face/2==2)*dxl[0]
        ) ;,
        double const dt = EXPR(
               (icell_face/2==0)*dxl[2],
             + (icell_face/2==1)*dxl[1],
             + (icell_face/2==2)*dxl[1]
        ) ;
        )
        Surf = math::int_pow<GRACE_NSPACEDIM-1>(2.*_L) * EXPRD(dh,*dt) ; 
    } else if( (itree_b-1)/P4EST_FACES == 0 ) {
        if( icell_face / 2 == 0 ) {
            Surf = get_cell_surface_zeta_int<5UL>(_L,_Ri
                                       ,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                       ,VECD(dxl[1],dxl[2]));

        } else if ( icell_face/2 == 1 ) { 
            Surf = get_cell_surface_eta_int(_L,_Ri
                                       ,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                       ,VECD(dxl[0],dxl[2]));                        
        } 
        #ifdef GRACE_3D 
        else if ( icell_face/2 == 2) {
            Surf = get_cell_surface_xi_int(_L,_Ri
                                       ,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                       ,VECD(dxl[0],dxl[1]));
        }
        #endif
        else {
            ERROR("Invalid face code in get_surface_buffer_zone") ; 
        }
    } else {
        if( _use_logr ){
            if( icell_face / 2 == 0 ) {
                Surf= get_cell_surface_zeta_log(_Ri,_Ro
                                       ,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                       ,VECD(dxl[1],dxl[2]));

            } else if ( icell_face/2 == 1 ) { 
                Surf= get_cell_surface_eta_log(_Ri,_Ro
                                        ,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                        ,VECD(dxl[0],dxl[2]));                        
            } 
            #ifdef GRACE_3D 
            else if ( icell_face/2 == 2) {
                Surf= get_cell_surface_xi_log(_Ri,_Ro
                                        ,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                        ,VECD(dxl[0],dxl[1]));
            }
            #endif 
            else {
                ERROR("Invalid face code " << icell_face ) ; 
            }
        } else { 
            if( icell_face / 2 == 0 ) {
                Surf = get_cell_surface_zeta_ext(_Ri,_Ro
                                       ,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                       ,VECD(dxl[1],dxl[2]));

            } else if ( icell_face/2 == 1 ) { 
                Surf = get_cell_surface_eta_ext(_Ri,_Ro
                                        ,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                        ,VECD(dxl[0],dxl[2]));                        
            } 
            #ifdef GRACE_3D 
            else if ( icell_face/2 == 2) {
                Surf= get_cell_surface_xi_ext(_Ri,_Ro
                                        ,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2])
                                        ,VECD(dxl[0],dxl[1]));
            }
            #endif 
            else {
                ERROR("Invalid face code " << icell_face ) ; 
            }
        }
    }
    return Surf ; 

}
#ifdef GRACE_3D 
double
GRACE_HOST spherical_coordinate_system_impl_t::get_cell_edge_length_buffer_zone(
      int8_t icell_edge
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& lcoords
    , std::array<double, GRACE_NSPACEDIM> const& dxl ) const
{
    using namespace grace::detail ;
    int    itree_b  ;
    int8_t iface_b, iface ; 
    std::tie(itree_b,iface_b,iface) =
        get_neighbor_tree_and_face(itree,lcoords,true) ; 
    if(itree_b==-1){ return 1. ;}
    auto lcoords_b = spherical_coordinate_system_impl_t::get_logical_coordinates_buffer_zone(
        itree,itree_b,iface,iface_b,lcoords
    ) ; 
    int polarity = amr::connectivity::get().tree_to_tree_polarity(itree,iface);
    EXPR(
    lcoords_b[0] += polarity * dxl[0]*( (iface_b==0)-(iface_b==1) ) * (icell_edge==0);,
    lcoords_b[1] += polarity * dxl[1]*( (iface_b==2)-(iface_b==3) ) * (icell_edge==1);,
    lcoords_b[2] += polarity * dxl[2]*( (iface_b==4)-(iface_b==5) ) * (icell_edge==2);
    )
    double Length{0.};
    if ( itree_b == 0 ) {
        Length= 2.*_L * EXPR( (icell_edge==0) * dxl[0], + (icell_edge==1) * dxl[1], + (icell_edge==2) * dxl[2] ) ;
    } else if ( (itree_b-1)/P4EST_FACES==0 ) {
        switch (icell_edge) {
            case 0:
            Length = get_line_element_zeta_int(_L,_Ri,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),dxl[0]); 
            case 1:
            Length = get_line_element_eta_int<5UL>(_L,_Ri,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),dxl[1]);
            case 2:
            Length = get_line_element_xi_int<5UL>(_L,_Ri,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),dxl[2]);
        } 
    } else if ( (itree_b-1)/P4EST_FACES==1 ) {
        if( _use_logr ) {
            switch (icell_edge) {
            case 0:
            Length = get_line_element_zeta_log(_Ri,_Ro,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),dxl[0]); 
            case 1:
            Length = get_line_element_eta_log(_Ri,_Ro,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),dxl[1]);
            case 2:
            Length = get_line_element_xi_log(_Ri,_Ro,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),dxl[2]);
            } 
        } else {
            switch (icell_edge) {
            case 0:
            Length = get_line_element_zeta_ext(_Ri,_Ro,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),dxl[0]); 
            case 1:
            Length = get_line_element_eta_ext(_Ri,_Ro,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),dxl[1]);
            case 2:
            Length = get_line_element_xi_ext(_Ri,_Ro,VEC(lcoords_b[0],lcoords_b[1],lcoords_b[2]),dxl[2]);
            } 
        }
    } else {
        ERROR("Invalid tree index in get_edge_length.") ; 
    }
    return Length ; 
}
#endif 
std::array<double, GRACE_NSPACEDIM>
GRACE_HOST spherical_coordinate_system_impl_t::get_physical_coordinates_cart(
    double L,
    std::array<double, GRACE_NSPACEDIM> const& lcoords ) const
{
    return {VEC(
            (lcoords[0] * 2. - 1.) * L,
            (lcoords[1] * 2. - 1.) * L,
            (lcoords[2] * 2. - 1.) * L 
        )} ;
}

std::array<double, GRACE_NSPACEDIM>
GRACE_HOST spherical_coordinate_system_impl_t::get_physical_coordinates_sph(
      int irot   
    , double Ri
    , double Ro 
    , std::array<double,2> const& F 
    , std::array<double,2> const& S
    , std::array<double, GRACE_NSPACEDIM> const& lcoords
    , bool logr ) const
{
    auto const eta = (2*lcoords[1]-1); 
    #ifdef GRACE_3D 
    auto const xi = (2*lcoords[2]-1) ; 
    #endif
    auto const one_over_rho = 1./sqrt(EXPR( 1
                                        , + math::int_pow<2>(eta)
                                        , + math::int_pow<2>(xi) )) ;
    auto const z = get_zeta(lcoords[0], one_over_rho, F, S, logr);   
    std::array<double, GRACE_NSPACEDIM> pcoords =
        { VEC( z
             , z * eta
             , z * xi )};
    return detail::apply_discrete_rotation(pcoords, irot ) ; 
}


double GRACE_HOST 
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