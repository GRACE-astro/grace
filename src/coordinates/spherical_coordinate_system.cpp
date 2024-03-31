/**
 * @file coordinate_systems.cpp
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

    std::array<double,THUNDER_NSPACEDIM> lcoords {
        VEC(
            qcoords[0] * dx_quad + (ijk[0] + cell_coordinates[0] - use_ghostzones * ngz) * dx_cell, 
            qcoords[1] * dx_quad + (ijk[1] + cell_coordinates[1] - use_ghostzones * ngz) * dy_cell, 
            qcoords[2] * dx_quad + (ijk[2] + cell_coordinates[2] - use_ghostzones * ngz) * dz_cell 
        ) 
    } ; 

    return get_physical_coordinates(itree, lcoords) ; 
}


std::array<double, THUNDER_NSPACEDIM> THUNDER_HOST 
spherical_coordinate_system_impl_t::get_physical_coordinates(
      std::array<size_t, THUNDER_NSPACEDIM> const& ijk
    , int64_t q 
    , bool use_ghostzones )
{
    return get_physical_coordinates(ijk,q,{VEC(0.5,0.5,0.5)},use_ghostzones);
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
            l_coords[0] = (log(r)-_S1)/_Sr1 ; 
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
        return exp(S[0] + S[1]*z) * one_over_rho; 
    } else { 
        auto const z_coeff = F[1] + S[1]*one_over_rho ;   
        auto const z0      = F[0] + S[0]*one_over_rho ; 
        return z*z_coeff + z0 ; 
    }
} 

}