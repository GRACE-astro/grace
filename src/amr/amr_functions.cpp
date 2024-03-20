/**
 * @file amr_functions.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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

#include <thunder/amr/amr_functions.hh>

#include <thunder/data_structures/macros.hh>
#include <thunder/amr/tree.hh>
#include <thunder/amr/connectivity.hh>
#include <thunder/amr/forest.hh> 

#include <thunder/config/config_parser.hh>

namespace thunder { namespace amr {

std::tuple<size_t,size_t,size_t>
get_quadrant_extents()
{
    auto& config = thunder::config_parser::get() ; 
    auto const nx = config["amr"]["npoints_block_x"].as<size_t>() ; 
    auto const ny = config["amr"]["npoints_block_y"].as<size_t>() ; 
    auto const nz = config["amr"]["npoints_block_z"].as<size_t>() ; 
    return std::make_tuple(nx,ny,nz) ;  
}

int 
get_n_ghosts()
{
    auto& config = thunder::config_parser::get() ; 
    return config["amr"]["n_ghostzones"].as<int>() ; 
}

size_t 
get_local_num_quadrants()
{
    return thunder::amr::forest::get().local_num_quadrants() ; 
}

size_t 
get_quadrant_owner(size_t iquad)
{
    auto& forest = thunder::amr::forest::get() ;
    for(size_t itree=forest.first_local_tree();
        itree <= forest.last_local_tree(); 
        itree+=1UL)
    {
        auto tree = forest.tree(itree) ; 
        int iquad_loc = iquad - tree.quadrants_offset() ; 
        if(     (iquad_loc >= 0)
            and (iquad_loc < tree.num_quadrants() ) ){
            return itree ; 
        }
    }
    ASSERT_DBG(0, 
    "In get_quadrant_owner: " << iquad << " is not owned by any local tree.") ;
    return -1 ; 
}

quadrant_t  
get_quadrant(size_t which_tree, size_t iquad)
{
    tree_t tree = thunder::amr::forest::get().tree(which_tree) ;
    return tree.quadrant(iquad-tree.quadrants_offset()) ; 
}

quadrant_t  
get_quadrant(size_t iquad)
{
    tree_t tree = 
        thunder::amr::forest::get().tree(get_quadrant_owner(iquad)) ;
    return tree.quadrant(iquad-tree.quadrants_offset()) ; 
}

std::array<double,THUNDER_NSPACEDIM> 
get_tree_vertex(size_t which_tree, size_t which_vertex)
{
    return thunder::amr::connectivity::get().vertex_coordinates(which_tree,which_vertex);
}

std::array<double,THUNDER_NSPACEDIM> 
get_tree_spacing(size_t which_tree)
{
    return thunder::amr::connectivity::get().tree_coordinate_exents(which_tree);
}

std::array<double, THUNDER_NSPACEDIM> 
get_physical_coordinates( std::array<size_t, THUNDER_NSPACEDIM> const& ijk 
                        , size_t iq 
                        , std::array<double,THUNDER_NSPACEDIM> const& local_coords 
                        , bool include_gzs) 
{
    using namespace detail ; 
    auto& params = thunder::config_parser::get() ; 

    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    int ngz = thunder::amr::get_n_ghosts() ;    

    std::string coord_system = params["amr"]["physical_coordinates"].as<std::string>() ; 

    double L,R,Rl ; 
    if( coord_system == "spherical" ) {
        L = params["amr"]["inner_region_side"].as<double>() ;
        R = params["amr"]["inner_region_radius"].as<double>()   ;
        Rl = params["amr"]["outer_region_radius"].as<double>()  ;
    }
    bool use_logr = params["amr"]["use_logarithmic_radial_zone"].as<bool>() ;
    size_t itree              = thunder::amr::get_quadrant_owner(iq) ; 
    thunder::amr::quadrant_t const quadrant = thunder::amr::get_quadrant(itree, iq) ;  
    auto const dx_quad        = 1.0 / ( 1<<quadrant.level()) ; 
    auto const qcoords        = quadrant.qcoords() ; 
    EXPR(
    auto const dx_cell        = dx_quad / nx ;,
    auto const dy_cell        = dx_quad / ny ;, 
    auto const dz_cell        = dx_quad / nz ; 
    ) 
    /* return physical coordinates of point within cell */ 
    auto lcoords = std::array<double,THUNDER_NSPACEDIM> {
        VEC(
        qcoords[0] * dx_quad + (ijk[0] + local_coords[0] - include_gzs * ngz) * dx_cell, 
        qcoords[1] * dx_quad + (ijk[1] + local_coords[1] - include_gzs * ngz) * dy_cell, 
        qcoords[2] * dx_quad + (ijk[2] + local_coords[2] - include_gzs * ngz) * dz_cell 
        )
    } ; 
    std::array<double,THUNDER_NSPACEDIM> pcoords ;
    if( coord_system == "cartesian" or itree == CARTESIAN_TREE ){ 
        auto const tree_coords    = thunder::amr::get_tree_vertex(itree, 0UL) ; 
        auto const dx_tree        = thunder::amr::get_tree_spacing(itree)[0]  ;
        for(int idir=0; idir<THUNDER_NSPACEDIM; ++idir) {
            pcoords[idir] = tree_coords[idir] + lcoords[idir] * dx_tree ;
        }
    } else {
        #ifdef THUNDER_3D 
        auto const H  = tan(M_PI/4. * (2.*lcoords[1]-1)) ; 
        auto const XI = tan(M_PI/4. * (2.*lcoords[2]-1)) ; 
        auto const rho = sqrt( 1 + math::int_pow<2>(H) + math::int_pow<2>(XI) ) ; 
        auto const zeta     = ((1.-lcoords[0]) * L + lcoords[0]*R/rho)  ;
        auto const zeta_log = 
            use_logr  * sqrt( std::pow(R, 2*(1-lcoords[0])) * std::pow(Rl, 2*lcoords[0]) ) / rho 
        + (!use_logr) * ((1.-lcoords[0]) * R/rho + lcoords[0]*Rl/rho) ;
        switch( itree )
        {
            case MX_TREE: 
                pcoords[0] = -zeta     ; 
                pcoords[1] = zeta * XI ;
                pcoords[2] = zeta * H  ;
                break ; 
            case PX_TREE: 
                pcoords[0] = zeta      ; 
                pcoords[1] = zeta * H  ;
                pcoords[2] = zeta * XI ;
                break; 
            case MY_TREE:
                pcoords[0] =  zeta * H  ;
                pcoords[1] = -zeta      ;
                pcoords[2] =  zeta * XI ;
                break;
            case PY_TREE:
                pcoords[0] =  zeta * XI  ;
                pcoords[1] =  zeta       ;
                pcoords[2] =  zeta * H   ; 
                break;
            case MZ_TREE:
                pcoords[0] =  zeta * XI ;
                pcoords[1] =  zeta * H  ;
                pcoords[2] = -zeta      ;
                break;
            case PZ_TREE:
                pcoords[0] =  zeta * H  ;
                pcoords[1] =  zeta * XI ;
                pcoords[2] =  zeta      ;
                break;
            case MXL_TREE: 
                pcoords[0] = -zeta_log     ; 
                pcoords[1] = zeta_log * XI ;
                pcoords[2] = zeta_log * H  ;
                break ; 
            case PXL_TREE: 
                pcoords[0] = zeta_log      ; 
                pcoords[1] = zeta_log * H  ;
                pcoords[2] = zeta_log * XI ;
                break; 
            case MYL_TREE:
                pcoords[0] =  zeta_log * H  ;
                pcoords[1] = -zeta_log      ;
                pcoords[2] =  zeta_log * XI ;
                break;
            case PYL_TREE:
                pcoords[0] =  zeta_log * XI  ;
                pcoords[1] =  zeta_log       ;
                pcoords[2] =  zeta_log * H   ; 
                break;
            case MZL_TREE:
                pcoords[0] =  zeta_log * XI ;
                pcoords[1] =  zeta_log * H  ;
                pcoords[2] = -zeta_log      ;
                break;
            case PZL_TREE:
                pcoords[0] =  zeta_log * H  ;
                pcoords[1] =  zeta_log * XI ;
                pcoords[2] =  zeta_log      ;
                break;
        }
        #else 
        auto const H  = tan(M_PI/4. * (2.*lcoords[1]-1)) ; 
        auto const rho = sqrt( 1 + math::int_pow<2>(H) ) ; 
        auto const zeta     = ((1.-lcoords[0]) * L + lcoords[0]*R/rho)  ;
        auto const zeta_log = sqrt( std::pow(R, 2*(1-lcoords[0])) * std::pow(Rl, 2*lcoords[0]) ) / rho ;
        switch( itree )
        {
            case MX_TREE: 
                pcoords[0] = -zeta     ; 
                pcoords[1] = zeta * H  ;
                break ; 
            case PX_TREE: 
                pcoords[0] = zeta      ; 
                pcoords[1] = zeta * H  ;
                break; 
            case MY_TREE:
                pcoords[0] =  zeta * H  ;
                pcoords[1] = -zeta      ;
                break;
            case PY_TREE:
                pcoords[0] =  zeta * H   ;
                pcoords[1] =  zeta       ;
                break;
            case MXL_TREE: 
                pcoords[0] = -zeta_log     ; 
                pcoords[1] = zeta_log * H  ;
                break ; 
            case PXL_TREE: 
                pcoords[0] = zeta_log      ; 
                pcoords[1] = zeta_log * H  ; 
                break; 
            case MYL_TREE:
                pcoords[0] =  zeta_log * H  ;
                pcoords[1] = -zeta_log      ;
                break;
            case PYL_TREE:
                pcoords[0] =  zeta_log * H   ;
                pcoords[1] =  zeta_log       ; 
                break;
        }
        #endif 
    }
    return pcoords ; 
};

std::array<double, THUNDER_NSPACEDIM> 
get_physical_coordinates( size_t icell
                        , std::array<double,THUNDER_NSPACEDIM> const& local_coords 
                        , bool include_gzs)
{
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    int ngz = thunder::amr::get_n_ghosts() ; 
    
    size_t const ix = icell%(nx + include_gzs*2*ngz) ; 
    size_t const iy = (icell/(nx + include_gzs*2*ngz)) % (ny + include_gzs*2*ngz) ;
    #ifdef THUNDER_3D 
    size_t const iz = 
        (icell/(nx + include_gzs*2*ngz)/(ny + include_gzs*2*ngz)) % (nz + include_gzs*2*ngz) ; 
    size_t const iq = 
        (icell/(nx + include_gzs*2*ngz)/(ny + include_gzs*2*ngz)/(nz + include_gzs*2*ngz)) ;
    #else 
    size_t const iq = (icell/(nx + include_gzs*2*ngz)/(nx + include_gzs*2*ngz)) ; 
    #endif 
    return get_physical_coordinates({VEC(ix,iy,iz)}, iq, local_coords, include_gzs) ; 
}; 

}} /* namespace thunder::amr */ 