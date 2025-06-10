/**
 * @file connectivity.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-02-29
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#include <grace_config.h>

#include <grace/utils/inline.h>

#include <grace/amr/connectivities_impl.hh>
#include <grace/amr/connectivity.hh>
#include <grace/utils/affine_transformation.hh>


#include <grace/config/config_parser.hh>

#include <cstdlib>

namespace grace{ namespace amr { 

namespace detail{

#ifdef GRACE_3D 
//**************************************************************************************************
#ifdef GRACE_3D 
//**************************************************************************************************
p4est_connectivity_t*
new_cartesian_connectivity( double xmin, double xmax, bool periodic_x
                            , double ymin, double ymax, bool periodic_y 
                            , double zmin, double zmax, bool periodic_z )
{
    auto x_ext{xmax-xmin}, y_ext{ymax-ymin}, z_ext{zmax-zmin} ; 

    // Determine the minimum extent to base the grid size on
    double min_ext = std::min({x_ext, y_ext, z_ext});

    uint32_t nx = static_cast<uint32_t>(x_ext / min_ext);
    uint32_t ny = static_cast<uint32_t>(y_ext / min_ext);
    uint32_t nz = static_cast<uint32_t>(z_ext / min_ext);

    // Adjust the extents to fit the grid
    x_ext = nx * min_ext;
    y_ext = ny * min_ext;
    z_ext = nz * min_ext;
    double x_tree { min_ext } , y_tree { min_ext }, z_tree { min_ext } ; 
    auto conn = p4est_connectivity_new_brick( nx,ny,nz, periodic_x,periodic_y,periodic_z ) ;
    // We manually set the vertices' coordinates to their physical value  
    auto vertices = conn->vertices; 
    auto n_vertices      = conn->num_vertices; 

    for( uint32_t nv=0; nv<n_vertices; ++nv) {
        vertices[ 3*nv     ] = utils::affine_transformation(vertices[ 3*nv   ], 0.0, static_cast<double>(nx), xmin, xmax);
        vertices[ 3*nv + 1 ] = utils::affine_transformation(vertices[ 3*nv+1 ], 0.0, static_cast<double>(ny), ymin, ymax);
        vertices[ 3*nv + 2 ] = utils::affine_transformation(vertices[ 3*nv+2 ], 0.0, static_cast<double>(nz), zmin, zmax);
    }   
      
    
  return conn ; 
} ; 
//**************************************************************************************************
//**************************************************************************************************
p4est_connectivity_t*
new_spherical_connectivity( double L, double R, double Rlog)
{
    const p4est_topidx_t num_vertices =  16 + 8 ;
    const p4est_topidx_t num_trees    =  7  + 6 ;
    const p4est_topidx_t num_edges    =  0;
    const p4est_topidx_t num_corners  =  0;
    const p4est_topidx_t ctt_offset   =  0;
    const p4est_topidx_t ett_offset   =  0;

    p4est_connectivity_t* conn { nullptr } ; 

    double const isqrt2 = 1./sqrt(2) ; 
    const double vertices [ 24 * 3 ] = { 
        - L, - L, - L,            // 0 
          L, - L, - L,            // 1
        - L,   L, - L,            // 2
          L,   L, - L,            // 3
        - L, - L,   L,            // 4 
          L, - L,   L,            // 5
        - L,   L,   L,            // 6
          L,   L,   L,            // 7
        - R * isqrt2, - R * isqrt2, - R * isqrt2,         // 8 
          R * isqrt2, - R * isqrt2, - R * isqrt2,         // 9
        - R * isqrt2,   R * isqrt2, - R * isqrt2,         // 10
          R * isqrt2,   R * isqrt2, - R * isqrt2,         // 11
        - R * isqrt2, - R * isqrt2,   R * isqrt2,         // 12 
          R * isqrt2, - R * isqrt2,   R * isqrt2,         // 13
        - R * isqrt2,   R * isqrt2,   R * isqrt2,         // 14
          R * isqrt2,   R * isqrt2,   R * isqrt2,         // 15
        - Rlog * isqrt2, - Rlog * isqrt2, - Rlog * isqrt2,         // 8 
          Rlog * isqrt2, - Rlog * isqrt2, - Rlog * isqrt2,         // 9
        - Rlog * isqrt2,   Rlog * isqrt2, - Rlog * isqrt2,         // 10
          Rlog * isqrt2,   Rlog * isqrt2, - Rlog * isqrt2,         // 11
        - Rlog * isqrt2, - Rlog * isqrt2,   Rlog * isqrt2,         // 12 
          Rlog * isqrt2, - Rlog * isqrt2,   Rlog * isqrt2,         // 13
        - Rlog * isqrt2,   Rlog * isqrt2,   Rlog * isqrt2,         // 14
          Rlog * isqrt2,   Rlog * isqrt2,   Rlog * isqrt2,         // 15
    }; 

    const p4est_topidx_t tree_to_vertex [ 13 * 8 ] = 
    { 
        0,1,2,3,4,5,6,7,     // 0
        0,8,4,12,2,10,6,14,  // 1
        1,9,3,11,5,13,7,15,  // 2 
        0,8,1,9,4,12,5,13,   // 3
        2,10,6,14,3,11,7,15, // 4
        0,8,2,10,1,9,3,11,   // 5
        4,12,5,13,6,14,7,15,  // 6
        8,16,12,20,10,18,14,22, // 7 
        9,17,11,19,13,21,15,23,  // 8
        8,16,9,17,12,20,13,21,  // 9
        10,18,14,22,11,19,15,23, // 10 
        8,16,10,18,9,17,11,19,
        12,20,13,21,14,22,15,23

    } ;

    const p4est_topidx_t tree_to_tree [ 13 * 6 ] = 
    { 
        1,2,3,4,5,6, // 0
        0,7,5,6,3,4, // 1
        0,8,3,4,5,6, // 2
        0,9,1,2,5,6, // 3
        0,10,5,6,1,2, // 4
        0,11,3,4,1,2, // 5
        0,12,1,2,3,4,  // 6
        1,7,11,12,9,10, // 7
        2,8,9,10,11,12, // 8 
        3,9,7,8,11,12, // 9
        4,10,11,12,7,8, // 10
        5,11,9,10,7,8, // 11
        6,12,7,8,9,10  // 12

    } ; 

    const int8_t tree_to_face [ 13 * 6 ] =
    { 
        0,0,0,0,0,0, // 0 
        0,0,4,2,2,4, // 1
        1,0,3,5,5,3, // 2
        2,0,4,2,2,4, // 3
        3,0,3,5,5,3, // 4
        4,0,4,2,2,4, // 5
        5,0,3,5,5,3,  // 6 
        1,1,4,2,2,4,
        1,1,3,5,5,3,
        1,1,4,2,2,4,
        1,1,3,5,5,3,
        1,1,4,2,2,4,
        1,1,3,5,5,3

    } ;  
  conn = p4est_connectivity_new_copy (num_vertices, num_trees,
                                  num_edges, num_corners,
                                  vertices, tree_to_vertex,
                                  tree_to_tree, tree_to_face,
                                  NULL, &ett_offset,
                                  NULL, NULL,
                                  NULL, &ctt_offset, NULL, NULL);
    
    ASSERT( p4est_connectivity_is_valid(conn), 
                                        "Sanity check on connectivity failed "
                                        "while setting up polar grid in 3D.") ;
    return conn ; 
} ;
//**************************************************************************************************
//**************************************************************************************************
#else 
p4est_connectivity_t*
new_spherical_connectivity( double L, double R, double Rlog)
{
    int32_t const num_trees    = 1 + 4 + 4 ; 
    int32_t const num_vertices = 4 + 4 + 4 ; 
    int32_t const num_ctt      = 0  ; 
    p4est_connectivity_t * conn{ nullptr } ; 

    auto const isqrt2 = 1./sqrt(2) ; 
    
      const double vertices [ 12 * 3 ] = {  
      - L, - L, 0,                                // 0 
        L, - L, 0,                                // 1
      - L,   L, 0,                                // 2
        L,   L, 0,                                // 3
      - R * isqrt2,   - R * isqrt2, 0,            // 4
        R * isqrt2,   - R * isqrt2, 0,            // 5
      - R * isqrt2,     R * isqrt2, 0,            // 6
        R * isqrt2,     R * isqrt2, 0,            // 7
      - Rlog * isqrt2,   - Rlog * isqrt2, 0,      // 8
        Rlog * isqrt2,   - Rlog * isqrt2, 0,      // 9
      - Rlog * isqrt2,     Rlog * isqrt2, 0,      // 10
        Rlog * isqrt2,     Rlog * isqrt2, 0,      // 11  
    } ;

    const p4est_topidx_t tree_to_vertex [ 9 * 4 ] = 
    { 
        0,1,2,3,  // 0
        0,2,4,6,  // 1
        1,5,3,7,  // 2 
        0,4,1,5,  // 3
        3,7,2,6,  // 4
        4,8,6,10, // 5
        5,9,7,11, // 6
        4,8,5,9,  // 7
        6,10,7,11 // 8
    } ; 

    const p4est_topidx_t tree_to_tree [9 * 4 ] = 
    { 
        1,2,3,4, // 0 
        0,5,3,4, // 1
        0,6,3,4, // 2
        0,7,1,2, // 3
        0,8,1,2, // 4
        1,5,7,8, // 5
        2,6,7,8, // 6
        3,7,5,6, // 7
        4,8,5,6  // 8
    } ; 

    const int8_t tree_to_face [ 9 * 4 ] =
    {
        0, 0, 0, 0, // 0
        0, 0, 2, 2, // 1
        1, 0, 3, 3, // 2
        2, 0, 2, 2, // 3
        3, 0, 3, 3, // 4
        1, 1, 2, 2, // 5
        1, 1, 3, 3, // 6
        1, 1, 2, 2, // 7
        1, 1, 3, 3  // 8
    } ;  
    conn =  p4est_connectivity_new_copy (num_vertices, num_trees, 0,
                                        vertices, tree_to_vertex,
                                        tree_to_tree, tree_to_face,
                                        NULL, &num_ctt, NULL, NULL) ;
    
    
    ASSERT( p4est_connectivity_is_valid(conn), 
                                        "Sanity check on connectivity failed "
                                        "while setting up polar grid in 2D.")  ;
    return conn ; 
}
//**************************************************************************************************
//**************************************************************************************************
p4est_connectivity_t*
new_cartesian_connectivity( double xmin, double xmax, bool periodic_x
                            , double ymin, double ymax, bool periodic_y )
{
    auto x_ext{xmax-xmin}, y_ext{ymax-ymin} ; 

    // Determine the minimum extent to base the grid size on
    double min_ext = std::min({x_ext, y_ext, z_ext});

    uint32_t nx = static_cast<uint32_t>(x_ext / min_ext);
    uint32_t ny = static_cast<uint32_t>(y_ext / min_ext);

    // Adjust the extents to fit the grid
    x_ext = nx * min_ext;
    y_ext = ny * min_ext;
    double x_tree { min_ext } , y_tree { min_ext }  ; 
    auto conn = p4est_connectivity_new_brick( nx,ny, periodic_x,periodic_y ) ;
    // We manually set the vertices' coordinates to their physical value  
    auto vertices = conn->vertices; 
    auto n_vertices      = conn->num_vertices; 

    for( uint32_t nv=0; nv<n_vertices; ++nv) {
        vertices[ 3*nv     ] = utils::affine_transformation(vertices[ 3*nv   ], 0.0, static_cast<double>(nx), xmin, xmax);
        vertices[ 3*nv + 1 ] = utils::affine_transformation(vertices[ 3*nv+1 ], 0.0, static_cast<double>(ny), ymin, ymax);
        vertices[ 3*nv + 2 ] = 0 ;
    }   
      
    
  return conn ; 
} ;
#endif 


} /* namespace detail */

//**************************************************************************************************
/**
 * @brief Construct a new connectivity object
 * 
 */
connectivity_impl_t::connectivity_impl_t() {
  auto& params = grace::config_parser::get() ; 
  #if defined(GRACE_CARTESIAN_COORDINATES)
  double xmin{ params["amr"]["xmin"].as<double>() } ,
          xmax{ params["amr"]["xmax"].as<double>() } ,
          ymin{ params["amr"]["ymin"].as<double>() } ,
          ymax{ params["amr"]["ymax"].as<double>() } , 
          zmin{ params["amr"]["zmin"].as<double>() } ,
          zmax{ params["amr"]["zmax"].as<double>() }   ;
  bool periodic_x{ params["amr"]["periodic_x"].as<bool>() } ,
        periodic_y{ params["amr"]["periodic_y"].as<bool>() } , 
        periodic_z{ params["amr"]["periodic_z"].as<bool>() } ; 

  #ifndef GRACE_3D 
    pconn_ = detail::new_cartesian_connectivity(xmin, xmax, periodic_x
                                                ,ymin, ymax, periodic_y) ; 
  #else 
    pconn_ = detail::new_cartesian_connectivity( xmin, xmax, periodic_x
                                                ,ymin, ymax, periodic_y
                                                ,zmin, zmax, periodic_z) ; 
  #endif 
  t2t_polarity_.resize(pconn_->num_trees * P4EST_FACES) ; 
  for( auto& x: t2t_polarity_ ) x = 0 ; 
  #elif defined(GRACE_SPHERICAL_COORDINATES)
  double  L{ params["amr"]["inner_region_side"].as<double>() }
        , R{ params["amr"]["inner_region_radius"].as<double>() }
        , Rl{ params["amr"]["outer_region_radius"].as<double>() } ; 
  bool use_log_r { params["amr"]["use_logarithmic_radial_zone"].as<bool>() } ; 

  #ifndef GRACE_3D 
    pconn_ = detail::new_spherical_connectivity(L, R, Rl) ; 
    t2t_polarity_ = {
      1, 0, 1, 0, // 0
      1, 0, 1, 0, // 1
      0, 0, 0, 1, // 2
      1, 0, 1, 0, // 3 
      0, 0, 0, 1, // 4
      0, 0, 1, 0, // 5
      0, 0, 0, 1, // 6 
      0, 0, 1, 0, // 7 
      0, 0, 0, 1  // 8 
  } ;
  #else 
    pconn_ = detail::new_spherical_connectivity(L, R, Rl) ;  
    t2t_polarity_ = {
      1,0,1,0,1,0, // 0
      1,0,0,0,1,0, // 1
      0,0,0,1,0,1, // 2
      1,0,1,0,1,0, // 3
      0,0,0,1,0,1, // 4
      1,0,1,0,0,0, // 5
      0,0,0,1,0,1, // 6
      0,0,0,0,1,0, // 7
      0,0,0,1,0,1, // 8
      0,0,1,0,1,0, // 9
      0,0,0,1,0,1, // 10
      0,0,1,0,0,0, // 11
      0,0,0,1,0,1, // 12
    };
  #endif 
  #else 
  ERROR("Unknown coordinate system.") ; 
  #endif 
  for( unsigned itree=0; itree<pconn_->num_trees; ++itree) {
    for( unsigned iface=0; iface<P4EST_FACES; ++iface){
      int jtree = pconn_->tree_to_tree[itree*P4EST_FACES+iface] ; 
      int jface = pconn_->tree_to_face[itree*P4EST_FACES+iface];
      ASSERT(
        t2t_polarity_[ itree*P4EST_FACES + iface] 
        ==
        t2t_polarity_[ jtree*P4EST_FACES + jface],
        "Polarity not symmetric at " << itree << ", " << iface << ".\n"
      ) ; 
    }
  }
}
//**************************************************************************************************

}} /* namespace grace::amr */