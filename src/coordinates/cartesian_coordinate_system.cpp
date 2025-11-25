/**
 * @file coordinate_systems.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-26
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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

#include <grace/amr/grace_amr.hh> 
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/errors/assert.hh>
#include <grace/coordinates/cartesian_coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/data_structures/macros.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/error.hh> 

#include <array> 
#include <cstring>

namespace grace { 


double GRACE_HOST 
cartesian_coordinate_system_impl_t::get_spacing(size_t const& q) const {
    DECLARE_GRID_EXTENTS ; 
    int64_t itree = amr::get_quadrant_owner(q)   ; 
    amr::quadrant_t quad = amr::get_quadrant(itree,q) ; 
    auto const dx_quad  = quad.spacing() ; 
    return dx_quad/nx ; 
}; 

double GRACE_HOST 
cartesian_coordinate_system_impl_t::get_inv_spacing(size_t const& q) const {
    return 1./get_spacing(q) ;  
}; 

cartesian_coordinate_system_impl_t::cartesian_coordinate_system_impl_t()
{
    using namespace grace ;
    using namespace Kokkos ; 
    DECLARE_GRID_EXTENTS;

    qx_h.reserve(nq); qdx_h.reserve(nq) ; 
    for( int i=0; i<nq; ++i) {
        auto itree = amr::get_quadrant_owner(i) ; 
        auto quad = amr::get_quadrant(itree,i) ; 
        auto dx = quad.spacing() ; 
        auto dx_tree = amr::get_tree_spacing(itree)[0] ; 
        qdx_h.push_back(dx_tree*dx/nx) ; 

        auto qx_i = quad.qcoords() ; 
        auto vx = amr::get_tree_vertex(itree,0UL) ; 

        qx_h.push_back({
            vx[0] + qx_i[0] * dx * dx_tree,
            vx[1] + qx_i[1] * dx * dx_tree,
            vx[2] + qx_i[2] * dx * dx_tree
        }) ; 
    }


    is_cks = get_param<bool>("coordinate_system", "is_kerr_schild") ; 
    bh_spin = get_param<double>("coordinate_system", "bh_spin"); 
}

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
cartesian_coordinate_system_impl_t::cart_to_sph(
    std::array<double, GRACE_NSPACEDIM> const& xyz 
) const 
{
    double rad = sqrt(SQR(xyz[0]) + SQR(xyz[1]) + SQR(xyz[2]));
    if ( is_cks ) {
        double r = fmax((sqrt( SQR(rad) - SQR(bh_spin) + sqrt(SQR(SQR(rad)-SQR(bh_spin))
                 + 4.0*SQR(bh_spin)*SQR(xyz[2])) ) / sqrt(2.0)), 1.0);
        return std::array<double,GRACE_NSPACEDIM>{{
            r,
            (fabs(xyz[2]/r) < 1.0) ? acos(xyz[2]/r) : acos(copysign(1.0, xyz[2])),
            atan2(r*xyz[1]-bh_spin*xyz[0], bh_spin*xyz[1]+r*xyz[0]) - bh_spin*r/(SQR(r)-2.0*r+SQR(bh_spin))
        }};
        
    } else {
        
        return std::array<double,GRACE_NSPACEDIM>{{
            rad,
            acos(xyz[2]/(rad+1e-50)),
            atan2(xyz[1],xyz[0])
        }};
    }
}

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
cartesian_coordinate_system_impl_t::sph_to_cart(
    std::array<double, GRACE_NSPACEDIM> const& rtp 
) const 
{
    if ( is_cks ) {
        return std::array<double,GRACE_NSPACEDIM>{{
            (rtp[0] * cos(rtp[2]) - bh_spin * sin(rtp[2]))*sin(rtp[1]),
            (rtp[0] * sin(rtp[2]) + bh_spin * cos(rtp[2]))*sin(rtp[1]),
            rtp[0] * cos(rtp[1])
        }} ;
    } else {
        return std::array<double,GRACE_NSPACEDIM>{{
            rtp[0] * cos(rtp[2])*sin(rtp[1]),
            rtp[0] * sin(rtp[2])*sin(rtp[1]),
            rtp[0] * cos(rtp[1])
        }} ;
    }
}

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
cartesian_coordinate_system_impl_t::get_physical_coordinates_sph(
           std::array<size_t, GRACE_NSPACEDIM> const& ijk
        , int64_t q 
        , bool use_ghostzones 
    ) const
{
    auto xyz = get_physical_coordinates(ijk,q,use_ghostzones) ; 
    return cart_to_sph(xyz) ; 
}

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
cartesian_coordinate_system_impl_t::get_physical_coordinates_sph(
         std::array<size_t, GRACE_NSPACEDIM> const& ijk
        , int64_t q 
        , std::array<double, GRACE_NSPACEDIM> const& cell_coordinates
        , bool use_ghostzones 
    ) const
{
    auto xyz = get_physical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return cart_to_sph(xyz) ; 
}

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
cartesian_coordinate_system_impl_t::get_physical_coordinates(
      int const itree
    , std::array<double, GRACE_NSPACEDIM> const& logical_coordinates ) const
{
    auto const tree_coords = amr::get_tree_vertex(itree,0UL) ; 
    auto const dx_tree     = amr::get_tree_spacing(itree) ;
    return {VEC(
        logical_coordinates[0] * dx_tree[0] + tree_coords[0],
        logical_coordinates[1] * dx_tree[1] + tree_coords[1],
        logical_coordinates[2] * dx_tree[2] + tree_coords[2]
    )} ; 
}

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
cartesian_coordinate_system_impl_t::get_physical_coordinates(
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
cartesian_coordinate_system_impl_t::get_physical_coordinates(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
    , int64_t q 
    , bool use_ghostzones ) const
{   
    return get_physical_coordinates(ijk,q,{VEC(0.5,0.5,0.5)},use_ghostzones);
} 

std::array<double, GRACE_NSPACEDIM>
GRACE_HOST cartesian_coordinate_system_impl_t::get_logical_coordinates(
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

    auto const dx_quad  = quad.spacing() ; 
    auto const qcoords = quad.qcoords()     ; 

    EXPR(
    auto const dx_cell = dx_quad / nx ;, 
    auto const dy_cell = dx_quad / ny ;,
    auto const dz_cell = dx_quad / nz ;
    ) 

    EXPR(
    int const i = use_ghostzones ? static_cast<int>(ijk[0]) - ngz : static_cast<int>(ijk[0]);,
    int const j = use_ghostzones ? static_cast<int>(ijk[1]) - ngz : static_cast<int>(ijk[1]);,
    int const k = use_ghostzones ? static_cast<int>(ijk[2]) - ngz : static_cast<int>(ijk[2]);
    )

    std::array<double,GRACE_NSPACEDIM> coords = {
        VEC(
            qcoords[0] * dx_quad 
                + (static_cast<double>(i) + cell_coordinates[0]) * dx_cell, 
            qcoords[1] * dx_quad 
                + (static_cast<double>(j) + cell_coordinates[1]) * dy_cell, 
            qcoords[2] * dx_quad 
                + (static_cast<double>(k) + cell_coordinates[2]) * dz_cell
        )
    } ; 

    return coords ; 
}

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
cartesian_coordinate_system_impl_t::get_logical_coordinates(
      int itree
    , std::array<double, GRACE_NSPACEDIM> const& physical_coordinates ) const
{ 
    auto const tree_coords = amr::get_tree_vertex(itree,0UL) ; 
    auto const dx_tree     = amr::get_tree_spacing(itree) ;
    return {VEC(
        (physical_coordinates[0] - tree_coords[0])/dx_tree[0],
        (physical_coordinates[1] - tree_coords[1])/dx_tree[1],
        (physical_coordinates[2] - tree_coords[2])/dx_tree[2]
    )} ; 
}

std::array<double, GRACE_NSPACEDIM> GRACE_HOST 
cartesian_coordinate_system_impl_t::get_logical_coordinates(
    std::array<double, GRACE_NSPACEDIM> const& physical_coordinates ) const
{
    int ntrees = amr::connectivity::get().get()->num_trees; 

    for( int itree=0; itree<ntrees; ++itree)
    {
        auto const tree_000 = amr::get_tree_vertex(itree,0UL) ; 
        auto const tree_100 = amr::get_tree_vertex(itree,1UL) ; 
        auto const tree_010 = amr::get_tree_vertex(itree,2UL) ; 
        #ifdef GRACE_3D 
        auto const tree_001 = amr::get_tree_vertex(itree,4UL) ; 
        #endif 
        if(
                physical_coordinates[0] > tree_000[0]  
            and physical_coordinates[0] < tree_100[0]  
            and physical_coordinates[1] > tree_000[1]  
            and physical_coordinates[1] < tree_010[1]  
            #ifdef GRACE_3D 
            and physical_coordinates[2] > tree_000[2]  
            and physical_coordinates[2] < tree_001[2]  
            #endif 
        ){
            return {VEC(
            (physical_coordinates[0] - tree_000[0])/(tree_100[0]-tree_000[0]),
            (physical_coordinates[1] - tree_000[1])/(tree_010[1]-tree_000[1]),
            (physical_coordinates[2] - tree_000[2])/(tree_001[2]-tree_000[2])
            )} ; 
        }
    }
    ERROR("Point (" << physical_coordinates[0] 
    << "," << physical_coordinates[1] 
    #ifdef GRACE_3D 
    << physical_coordinates[2] 
    #endif 
    << ") is ouside the grid."
    << "If you're seeking the coordinates "
    << "of a point in the ghost-zones "
    << "please use the version of this function "
    << "that takes the tree index as input.") ; 
}

double
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_jacobian(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const 
{
    using namespace grace ;
    auto const itree = amr::get_quadrant_owner(q) ;  
    return get_jacobian(ijk,q,itree,cell_coordinates,use_ghostzones) ; 
}

double
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_jacobian(
        std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    using namespace grace ; 
    auto const lcoords = get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_jacobian(itree,lcoords) ; 
}
double
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_jacobian(
      int itree
    , std::array<double,GRACE_NSPACEDIM> const& lcoords ) const
{
    auto dx_tree = amr::get_tree_spacing(itree) ;
    return EXPR(dx_tree[0], * dx_tree[1], * dx_tree[2]) ; 
}

double
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_inverse_jacobian(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const 
{
    using namespace grace ;
    auto const itree = amr::get_quadrant_owner(q) ;  
    return get_inverse_jacobian(ijk,q,itree,cell_coordinates,use_ghostzones) ;
}

double
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_inverse_jacobian(
        std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    using namespace grace ; 
    auto const lcoords = get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_inverse_jacobian(itree,lcoords) ; 
}
double
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_inverse_jacobian(
      int itree
    , std::array<double,GRACE_NSPACEDIM> const& lcoords ) const
{
    // For cartesian coordinates the situation is rather
    // simple! 
    return 1./get_jacobian(itree,lcoords) ;  
}

std::array<double, 9>
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_jacobian_matrix(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    auto const itree = grace::amr::get_quadrant_owner(q) ; 
    return get_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) ; 
}

std::array<double, 9>
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_jacobian_matrix(
        std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    auto const lcoords = get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_jacobian_matrix(itree,lcoords) ; 
}

std::array<double, 9>
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_jacobian_matrix(
        int itree
    , std::array<double,GRACE_NSPACEDIM> const& lcoords ) const
{
    auto const dx_tree = amr::get_tree_spacing(itree) ; 
    return std::array<double,9>{
          dx_tree[0], 0         , 0           
        , 0         , dx_tree[0], 0          
        , 0         , 0         , dx_tree[0] 
        
    } ; 
}

std::array<double, 9>
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_inverse_jacobian_matrix(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    auto const itree = grace::amr::get_quadrant_owner(q) ; 
    return get_inverse_jacobian_matrix(ijk,q,itree,cell_coordinates,use_ghostzones) ; 
}

std::array<double, 9>
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_inverse_jacobian_matrix(
        std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q 
    , int itree
    , std::array<double,GRACE_NSPACEDIM> const& cell_coordinates 
    , bool use_ghostzones ) const
{
    auto const lcoords = get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_inverse_jacobian_matrix(itree,lcoords) ; 
}

std::array<double, 9>
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_inverse_jacobian_matrix(
        int itree
    , std::array<double,GRACE_NSPACEDIM> const& lcoords ) const
{
    auto const dx_tree = amr::get_tree_spacing(itree) ; 
    return std::array<double,9>{
          1./dx_tree[0], 0            , 0           
        , 0            , 1./dx_tree[0], 0          
        , 0            , 0            , 1./dx_tree[0] 
        
    } ; 
}

double
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_cell_volume(
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
GRACE_HOST cartesian_coordinate_system_impl_t::get_cell_volume(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk 
    , int64_t q
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const
{
    using namespace grace ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto lcoords = get_logical_coordinates(ijk,q,{VEC(0.5,0.5,0.5)},use_ghostzones) ; 
    return get_cell_volume(lcoords,itree,dxl,use_ghostzones) ; 
}

double
GRACE_HOST cartesian_coordinate_system_impl_t::get_cell_volume(
      std::array<double, GRACE_NSPACEDIM> const& lcoords 
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const
{
    using namespace grace ; 
    int ngz = amr::get_n_ghosts() ; 
    int64_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto& conn = amr::connectivity::get();
    ASSERT_DBG(
            itree < amr::connectivity::get().get()->num_trees,
            "Tree out of bounds " << itree  
        ) ;
    auto dx_tree = amr::get_tree_spacing(itree) ; 
    
    if( EXPR(
           lcoords[0] < 0 or lcoords[0] >= 1,
        or lcoords[1] < 0 or lcoords[1] >= 1,
        or lcoords[2] < 0 or lcoords[2] >= 1
    )) {
        int iface = 0 + EXPR(
          (lcoords[0] < 0) * 0 
        + (lcoords[0] >= 1) * 1,
        + (lcoords[1] < 0) * 2 
        + (lcoords[1] >= 1) * 3,
        + (lcoords[2] < 0) * 4 
        + (lcoords[2] >= 1) * 5) ;
        if( iface >= P4EST_FACES ) {
            iface = 0 ; // in corner ghostzones we can put anything 
        }
        int    itree_b  = conn.tree_to_tree(itree, iface) ;
        dx_tree = amr::get_tree_spacing(itree_b) ;  
    }
    
    return EXPR(
          dx_tree[0] * dxl[0], 
        * dx_tree[1] * dxl[1],
        * dx_tree[2] * dxl[2]
    ) ; 

}

double 
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_cell_face_surface(
    std::array<size_t, GRACE_NSPACEDIM> const& ijk 
  , int64_t q
  , int8_t face 
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
    return get_cell_face_surface(ijk,q,face,itree,{VEC(dx_cell,dy_cell,dz_cell)},use_ghostzones) ; 
}; 

double 
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_cell_face_surface(
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
}; 

double 
GRACE_HOST 
cartesian_coordinate_system_impl_t::get_cell_face_surface(
      std::array<double, GRACE_NSPACEDIM> const& lcoords 
    , int8_t face 
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const 
{
    using namespace grace ;
    auto& conn = amr::connectivity::get();
    ASSERT_DBG(
            itree < amr::connectivity::get().get()->num_trees,
            "Tree out of bounds " << itree ) ;
    auto dx_tree = amr::get_tree_spacing(itree) ; 
    
    if( EXPR(
           lcoords[0] < 0 or lcoords[0] > 1,
        or lcoords[1] < 0 or lcoords[1] > 1,
        or lcoords[2] < 0 or lcoords[2] > 1
    )) {
        int iface = 0 + EXPR(
          (lcoords[0] < 0) * 0 
        + (lcoords[0] > 1) * 1,
        + (lcoords[1] < 0) * 2 
        + (lcoords[1] > 1) * 3,
        + (lcoords[2] < 0) * 4 
        + (lcoords[2] > 1) * 5) ;
        if( iface >= P4EST_FACES ) {
            iface = 0 ; // in corner ghostzones we can put anything 
        }
        int    itree_b  = conn.tree_to_tree(itree, iface) ;
        dx_tree = amr::get_tree_spacing(itree_b) ;  
    }
    EXPRD(
    int const idir = (face==0) * 1 + (face==1) * 0 + (face==2) * 0 ;,
    int const jdir = (face==0) * 2 + (face==1) * 2 + (face==2) * 1 ;
    )
    return EXPRD(
          dx_tree[idir] * dxl[idir], 
        * dx_tree[jdir] * dxl[jdir]
    ) ; 
}; 

double GRACE_HOST 
cartesian_coordinate_system_impl_t::get_cell_edge_length(
    std::array<size_t, GRACE_NSPACEDIM> const& ijk
  , int64_t q 
  , int8_t edge
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
    return get_cell_edge_length(ijk,q,edge,itree,{VEC(dx_cell,dy_cell,dz_cell)},use_ghostzones) ;
};

double GRACE_HOST 
cartesian_coordinate_system_impl_t::get_cell_edge_length(
      std::array<size_t, GRACE_NSPACEDIM> const& ijk
    , int64_t q 
    , int8_t edge
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const 
{
    std::array<double, GRACE_NSPACEDIM> cell_coordinates 
    { VEC( 0.,0.,0.) } ;
    auto lcoords = get_logical_coordinates(ijk,q,cell_coordinates,use_ghostzones) ; 
    return get_cell_edge_length(lcoords,edge,itree,dxl,use_ghostzones) ;
}

double GRACE_HOST 
cartesian_coordinate_system_impl_t::get_cell_edge_length(
      std::array<double, GRACE_NSPACEDIM> const& lcoords
    , int8_t edge
    , int itree
    , std::array<double, GRACE_NSPACEDIM> const& dxl 
    , bool use_ghostzones) const 
{
    using namespace grace ;
    auto& conn = amr::connectivity::get();
    ASSERT_DBG(
            itree < amr::connectivity::get().get()->num_trees,
            "Tree out of bounds " << itree ) ;
    auto dx_tree = amr::get_tree_spacing(itree) ; 
    
    if( EXPR(
           lcoords[0] < 0 or lcoords[0] > 1,
        or lcoords[1] < 0 or lcoords[1] > 1,
        or lcoords[2] < 0 or lcoords[2] > 1
    )) {
        int iface = 0 + EXPR(
          (lcoords[0] < 0) * 0 
        + (lcoords[0] > 1) * 1,
        + (lcoords[1] < 0) * 2 
        + (lcoords[1] > 1) * 3,
        + (lcoords[2] < 0) * 4 
        + (lcoords[2] > 1) * 5) ;
        if( iface >= P4EST_FACES ) {
            iface = 0 ; // in corner ghostzones we can put anything 
        }
        int    itree_b  = conn.tree_to_tree(itree, iface) ;
        dx_tree = amr::get_tree_spacing(itree_b) ;  
    }

    return dx_tree[edge] * dxl[edge] ;
}


} /* namespace grace */ 