/**
 * @file connectivity.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-02-29
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

#include <thunder_config.h>

#include <thunder/utils/inline.h>

#include <thunder/amr/connectivities_impl.hh>
#include <thunder/amr/connectivity.hh>

#include <thunder/config/config_parser.hh>

#include <cstdlib>

namespace thunder{ namespace amr { 

namespace detail{

#ifdef THUNDER_3D 
//**************************************************************************************************
p4est_connectivity_t*
new_cartesian_connectivity( double xmin, double xmax, bool periodic_x
                            , double ymin, double ymax, bool periodic_y 
                            , double zmin, double zmax, bool periodic_z )
{
    auto x_ext{xmax-xmin}, y_ext{ymax-ymin}, z_ext{zmax-zmin} ; 

    uint32_t nx,ny,nz ; 
    if( x_ext < y_ext ) { 
        if( x_ext < z_ext ) { 
            nx = 1 ; 
            ny = static_cast<uint32_t>(y_ext / x_ext) ;
            nz = static_cast<uint32_t>(z_ext / x_ext) ;
            y_ext = ny * x_ext ; 
            z_ext = nz * x_ext ; 
        } else { 
            nz = 1 ; 
            nx = static_cast<uint32_t>(x_ext / z_ext) ;
            ny = static_cast<uint32_t>(y_ext / z_ext) ;
            x_ext = nx * z_ext ; 
            y_ext = ny * z_ext ;
        }
    } else { 
        if ( y_ext < z_ext ) {
            ny = 1 ; 
            nz = static_cast<uint32_t>(z_ext / y_ext) ;
            nx = static_cast<uint32_t>(x_ext / y_ext) ;
            x_ext = nx * y_ext ; 
            z_ext = nz * y_ext ;
        } else { 
            nz = 1 ; 
            nx = static_cast<uint32_t>(x_ext / z_ext) ;
            ny = static_cast<uint32_t>(y_ext / z_ext) ;
            x_ext = nx * z_ext ; 
            y_ext = ny * z_ext ;
        }
    }
    double x_tree { x_ext / nx } , y_tree { y_ext / ny }, z_tree { z_ext / nz } ; 
    auto conn = p4est_connectivity_new_brick( nx,ny,nz, periodic_x,periodic_y,periodic_z ) ;
    // We manually set the vertices' coordinates to their physical value  
    auto vertices = conn->vertices; 
    auto t2v      = conn->tree_to_vertex ; 
    size_t nt = 0 ; 
    for( uint32_t k=0; k<nz; ++k) for(uint32_t j=0; j<ny; ++j) for( uint32_t i=0; i<nx; ++i) { 
        for( uint32_t v=0; v<8; ++v) {
            size_t nv = t2v[ 8 * nt + v ] ; 
            vertices[ 3*nv ]     = ( i + (uint32_t)(v%2U)          ) * x_tree + xmin; 
            vertices[ 3*nv + 1 ] = ( j + (uint32_t)((v>>1U) & 1U)  ) * y_tree + ymin;
            vertices[ 3*nv + 2 ] = ( k + (uint32_t)((v>>2U) & 1U)  ) * z_tree + zmin; 
        }   
        ++nt ; 
    }
  return conn ; 
} ; 
//**************************************************************************************************
//**************************************************************************************************
p4est_connectivity_t*
new_spherical_connectivity( double rmin, double rmax, double rmax_log, bool extend_with_logr )
{
    const p4est_topidx_t num_vertices =  16 + extend_with_logr * 8 ;
    const p4est_topidx_t num_trees    =  7  + extend_with_logr * 6 ;
    const p4est_topidx_t num_edges    =  0;
    const p4est_topidx_t num_corners  =  0;
    const p4est_topidx_t ctt_offset   =  0;
    const p4est_topidx_t ett_offset   =  0;

    p4est_connectivity_t* conn { nullptr } ; 

    if( extend_with_logr ) {
       double const isqrt2 = 1./sqrt(2) ; 
        const double vertices [ 24 * 3 ] = { 
            - rmin * isqrt2, - rmin * isqrt2, - rmin * isqrt2,            // 0 
              rmin * isqrt2, - rmin * isqrt2, - rmin * isqrt2,            // 1
            - rmin * isqrt2,   rmin * isqrt2, - rmin * isqrt2,            // 2
              rmin * isqrt2,   rmin * isqrt2, - rmin * isqrt2,            // 3
            - rmin * isqrt2, - rmin * isqrt2,   rmin * isqrt2,            // 4 
              rmin * isqrt2, - rmin * isqrt2,   rmin * isqrt2,            // 5
            - rmin * isqrt2,   rmin * isqrt2,   rmin * isqrt2,            // 6
              rmin * isqrt2,   rmin * isqrt2,   rmin * isqrt2,            // 7
            - rmax  * isqrt2, - rmax  * isqrt2, - rmax  * isqrt2,         // 8 
              rmax  * isqrt2, - rmax  * isqrt2, - rmax  * isqrt2,         // 9
            - rmax  * isqrt2,   rmax  * isqrt2, - rmax  * isqrt2,         // 10
              rmax  * isqrt2,   rmax  * isqrt2, - rmax  * isqrt2,         // 11
            - rmax  * isqrt2, - rmax  * isqrt2,   rmax  * isqrt2,         // 12 
              rmax  * isqrt2, - rmax  * isqrt2,   rmax  * isqrt2,         // 13
            - rmax  * isqrt2,   rmax  * isqrt2,   rmax  * isqrt2,         // 14
              rmax  * isqrt2,   rmax  * isqrt2,   rmax  * isqrt2,         // 15
            - rmax_log  * isqrt2, - rmax_log  * isqrt2, - rmax_log  * isqrt2,         // 8 
              rmax_log  * isqrt2, - rmax_log  * isqrt2, - rmax_log  * isqrt2,         // 9
            - rmax_log  * isqrt2,   rmax_log  * isqrt2, - rmax_log  * isqrt2,         // 10
              rmax_log  * isqrt2,   rmax_log  * isqrt2, - rmax_log  * isqrt2,         // 11
            - rmax_log  * isqrt2, - rmax_log  * isqrt2,   rmax_log  * isqrt2,         // 12 
              rmax_log  * isqrt2, - rmax_log  * isqrt2,   rmax_log  * isqrt2,         // 13
            - rmax_log  * isqrt2,   rmax_log  * isqrt2,   rmax_log  * isqrt2,         // 14
              rmax_log  * isqrt2,   rmax_log  * isqrt2,   rmax_log  * isqrt2,         // 15
        }; 

        const p4est_topidx_t tree_to_vertex [ 13 * 8 ] = 
        { 
            0, 1, 2 , 3 , 4 , 5 , 6 , 7 , // 0
            8, 9, 10, 11, 0 , 1 , 2 , 3 , // 1
            1, 9, 3 , 11, 5 , 13, 7 , 15, // 2 
            4, 5, 6 , 7 , 12, 13, 14, 15, // 3
            8, 0, 10, 2 , 12, 4 , 14, 6 , // 4
            8, 9, 0 , 1 , 12, 13, 4 , 5 , // 5
            2, 3, 10, 11, 6 , 7 , 14, 15, // 6
            16, 17, 18, 19, 8, 9, 10, 11, // 7 
            9,  17, 
        } ;

        const p4est_topidx_t tree_to_tree [ 13 * 6 ] = 
        { 
            4, 2, 5, 6, 1, 3, // 0 
            4, 2, 5, 6, 1, 0, // 1
            0, 2, 5, 6, 1, 3, // 2
            4, 2, 5, 6, 0, 3, // 3
            4, 0, 5, 6, 1, 3, // 4
            4, 2, 5, 0, 1, 3, // 5
            4, 2, 0, 6, 1, 3  // 6
        } ; 

        const int8_t tree_to_face [ 13 * 6 ] =
        { 
            1, 0, 3, 2, 5, 4,
            4, 10, 4, 16, 4, 4, 
            1, 1, 7, 1, 7, 1,
            11, 5, 17, 5, 5, 5,
            0, 0, 0, 6, 0, 6, 
            2, 8, 2, 2, 2, 14, 
            9, 3, 3, 3, 15, 3

        } ; 
      conn = p4est_connectivity_new_copy (num_vertices, num_trees,
                                      num_edges, num_corners,
                                      vertices, tree_to_vertex,
                                      tree_to_tree, tree_to_face,
                                      NULL, &ett_offset,
                                      NULL, NULL,
                                      NULL, &ctt_offset, NULL, NULL);
    } else { 
        double const isqrt2 = 1./sqrt(2) ; 
        const double vertices [ 16 * 3 ] = { 
            - rmin * isqrt2, - rmin * isqrt2, - rmin * isqrt2,            // 0 
              rmin * isqrt2, - rmin * isqrt2, - rmin * isqrt2,            // 1
            - rmin * isqrt2,   rmin * isqrt2, - rmin * isqrt2,            // 2
              rmin * isqrt2,   rmin * isqrt2, - rmin * isqrt2,            // 3
            - rmin * isqrt2, - rmin * isqrt2,   rmin * isqrt2,            // 4 
              rmin * isqrt2, - rmin * isqrt2,   rmin * isqrt2,            // 5
            - rmin * isqrt2,   rmin * isqrt2,   rmin * isqrt2,            // 6
              rmin * isqrt2,   rmin * isqrt2,   rmin * isqrt2,            // 7
            - rmax  * isqrt2, - rmax  * isqrt2, - rmax  * isqrt2,         // 8 
              rmax  * isqrt2, - rmax  * isqrt2, - rmax  * isqrt2,         // 9
            - rmax  * isqrt2,   rmax  * isqrt2, - rmax  * isqrt2,         // 10
              rmax  * isqrt2,   rmax  * isqrt2, - rmax  * isqrt2,         // 11
            - rmax  * isqrt2, - rmax  * isqrt2,   rmax  * isqrt2,         // 12 
              rmax  * isqrt2, - rmax  * isqrt2,   rmax  * isqrt2,         // 13
            - rmax  * isqrt2,   rmax  * isqrt2,   rmax  * isqrt2,         // 14
              rmax  * isqrt2,   rmax  * isqrt2,   rmax  * isqrt2,         // 15
        }; 

        const p4est_topidx_t tree_to_vertex [ 7 * 8 ] = 
        { 
            0, 1, 2 , 3 , 4 , 5 , 6 , 7 , // 0
            8, 9, 10, 11, 0 , 1 , 2 , 3 , // 1
            1, 9, 3 , 11, 5 , 13, 7 , 15, // 2 
            4, 5, 6 , 7 , 12, 13, 14, 15, // 3
            8, 0, 10, 2 , 12, 4 , 14, 6 , // 4
            8, 9, 0 , 1 , 12, 13, 4 , 5 , // 5
            2, 3, 10, 11, 6 , 7 , 14, 15  // 6
        } ;

        const p4est_topidx_t tree_to_tree [ 7 * 6 ] = 
        { 
            4, 2, 5, 6, 1, 3, // 0 
            4, 2, 5, 6, 1, 0, // 1
            0, 2, 5, 6, 1, 3, // 2
            4, 2, 5, 6, 0, 3, // 3
            4, 0, 5, 6, 1, 3, // 4
            4, 2, 5, 0, 1, 3, // 5
            4, 2, 0, 6, 1, 3  // 6
        } ; 

        const int8_t tree_to_face [ 7 * 6 ] =
        { 
            1, 0, 3, 2, 5, 4,
            4, 10, 4, 16, 4, 4, 
            1, 1, 7, 1, 7, 1,
            11, 5, 17, 5, 5, 5,
            0, 0, 0, 6, 0, 6, 
            2, 8, 2, 2, 2, 14, 
            9, 3, 3, 3, 15, 3

        } ; 
      conn = p4est_connectivity_new_copy (num_vertices, num_trees,
                                      num_edges, num_corners,
                                      vertices, tree_to_vertex,
                                      tree_to_tree, tree_to_face,
                                      NULL, &ett_offset,
                                      NULL, NULL,
                                      NULL, &ctt_offset, NULL, NULL);
    }
    ASSERT( p4est_connectivity_is_valid(conn), 
                                        "Sanity check on connectivity failed "
                                        "while setting up polar grid in 3D.") ;
    return conn ; 
} ;
//**************************************************************************************************
//**************************************************************************************************
#else 
p4est_connectivity_t*
new_spherical_connectivity( double rmin, double rmax, double rmax_log, bool extend_with_logr )
{
    int32_t const num_trees    = 1 + 4 + 4 * extend_with_logr ; 
    int32_t const num_vertices = 4 + 4 + 4 * extend_with_logr ; 
    int32_t const num_ctt      = 0  ; 
    p4est_connectivity_t * conn{ nullptr } ; 

    auto const isqrt2 = 1./sqrt(2) ; 
    if ( extend_with_logr ) {
        double const isqrt2 = 1./sqrt(2) ; 
        const double vertices [ 12 * 3 ] = {  
            - rmin * isqrt2, - rmin * isqrt2, 0,            // 0 
              rmin * isqrt2, - rmin * isqrt2, 0,            // 1
            - rmin * isqrt2,   rmin * isqrt2, 0,            // 2
              rmin * isqrt2,   rmin * isqrt2, 0,            // 3
            - rmax * isqrt2, - rmax * isqrt2, 0,            // 4
              rmax * isqrt2, - rmax * isqrt2, 0,            // 5
            - rmax * isqrt2,   rmax * isqrt2, 0,            // 6
              rmax * isqrt2,   rmax * isqrt2, 0,            // 7
            - rmax_log  * isqrt2, - rmax_log  * isqrt2, 0,  // 8
              rmax_log  * isqrt2, - rmax_log  * isqrt2, 0,  // 9
            - rmax_log  * isqrt2,   rmax_log  * isqrt2, 0,  // 10
              rmax_log  * isqrt2,   rmax_log  * isqrt2, 0,  // 11 
        } ;

        const p4est_topidx_t tree_to_vertex [ 9 * 4 ] = 
        { 
            0, 1, 2, 3, // 0
            4, 5, 0, 1, // 1
            1, 5, 3, 7, // 2 
            2, 3, 6, 7, // 3
            4, 0, 6, 2, // 4
            8, 9, 4, 5, // 5
            5, 9, 7, 11,// 6
            6, 7, 10,11,// 7
            8, 4, 10, 6 // 8 
        } ; 

        const p4est_topidx_t tree_to_tree [ 9 * 4 ] = 
        { 
            4, 2, 1, 3, // 0 
            4, 2, 5, 0, // 1
            0, 6, 1, 3, // 2
            4, 2, 0, 7, // 3
            8, 0, 1, 3, // 4
            8, 6, 5, 1, // 5
            2, 6, 5, 7, // 6
            8, 6, 3, 7, // 7 
            8, 4, 5, 7  // 8 
        } ; 

        const int8_t tree_to_face [ 9 * 4 ] =
        {
            1, 0, 3, 2,
            2, 6, 3, 2,
            1, 0, 5, 1,
            7, 3, 3, 2,
            1, 0, 0, 4,
            2, 6, 2, 2,
            1, 1, 5, 1,
            7, 3, 3, 3,
            0, 0, 0, 4 
        } ; 
        conn =  p4est_connectivity_new_copy (num_vertices, num_trees, 0,
                                            vertices, tree_to_vertex,
                                            tree_to_tree, tree_to_face,
                                            NULL, &num_ctt, NULL, NULL) ; 
    } else { 
        double const isqrt2 = 1./sqrt(2) ; 
        const double vertices [ 8 * 3 ] = {  
            - rmin * isqrt2, - rmin * isqrt2, 0,            // 0 
              rmin * isqrt2, - rmin * isqrt2, 0,            // 1
            - rmin * isqrt2,   rmin * isqrt2, 0,            // 2
              rmin * isqrt2,   rmin * isqrt2, 0,            // 3
            - rmax * isqrt2, - rmax * isqrt2, 0,            // 4
              rmax * isqrt2, - rmax * isqrt2, 0,            // 5
            - rmax * isqrt2,   rmax * isqrt2, 0,            // 6
              rmax * isqrt2,   rmax * isqrt2, 0,            // 7
        } ;

        const p4est_topidx_t tree_to_vertex [ 5 * 4 ] = 
        { 
            0, 1, 2, 3, // 0
            4, 5, 0, 1, // 1
            1, 5, 3, 7, // 2 
            2, 3, 6, 7, // 3
            4, 0, 6, 2  // 4
        } ; 

        const p4est_topidx_t tree_to_tree [ 5 * 4 ] = 
        { 
            4, 2, 1, 3, // 0 
            4, 2, 5, 0, // 1
            0, 6, 1, 3, // 2
            4, 2, 0, 7, // 3
            8, 0, 1, 3  // 4
        } ; 

        const int8_t tree_to_face [ 5 * 4 ] =
        {
            1, 0, 3, 2,
            2, 6, 2, 2,
            1, 1, 5, 1,
            7, 3, 3, 3,
            0, 0, 0, 4 
        } ;  
        conn =  p4est_connectivity_new_copy (num_vertices, num_trees, 0,
                                            vertices, tree_to_vertex,
                                            tree_to_tree, tree_to_face,
                                            NULL, &num_ctt, NULL, NULL) ; 
    }
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
    auto x_ext{xmax-xmin}, y_ext{ymax-ymin}; 

    uint32_t nx,ny; 
    if( x_ext < y_ext ) { 
        nx = 1 ; 
        ny = static_cast<uint32_t>(y_ext / x_ext) ;
        y_ext = ny * x_ext ;
    } else { 
        ny = 1 ; 
        nx = static_cast<uint32_t>(x_ext / y_ext) ;
        x_ext = nx * y_ext ; 
    }
    double x_tree { x_ext / nx } , y_tree { y_ext / ny }  ; 
    auto conn = p4est_connectivity_new_brick( nx,ny, periodic_x,periodic_y ) ;
    // We manually set the vertices' coordinates to their physical value  
    auto vertices = conn->vertices; 
    auto t2v      = conn->tree_to_vertex ; 
    size_t nt = 0 ; 
    for(uint32_t j=0; j<ny; ++j) for( uint32_t i=0; i<nx; ++i) { 
        for( uint32_t v=0; v<4; ++v) {
            size_t nv = t2v[ 4 * nt + v ] ; 
            vertices[ 3*nv ]     = ( i + (uint32_t)(v%2U)          ) * x_tree + xmin; 
            vertices[ 3*nv + 1 ] = ( j + (uint32_t)((v>>1U) & 1U)  ) * y_tree + ymin;
            vertices[ 3*nv + 2 ] = 0;
        }   
        ++nt ; 
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
  auto& params = thunder::config_parser::get() ; 
  std::string coord_system( params["amr"]["physical_coordinates"].as<std::string>() ) ; 

  if ( coord_system == "cartesian" ) { 
    double xmin{ params["amr"]["xmin"].as<double>() } ,
           xmax{ params["amr"]["xmax"].as<double>() } ,
           ymin{ params["amr"]["ymin"].as<double>() } ,
           ymax{ params["amr"]["ymax"].as<double>() } , 
           zmin{ params["amr"]["zmin"].as<double>() } ,
           zmax{ params["amr"]["zmax"].as<double>() }   ;
    bool periodic_x{ params["amr"]["periodic_x"].as<bool>() } ,
         periodic_y{ params["amr"]["periodic_y"].as<bool>() } , 
         periodic_z{ params["amr"]["periodic_z"].as<bool>() } ; 

    
    #ifndef THUNDER_3D 
      pconn_ = detail::new_cartesian_connectivity(xmin, xmax, periodic_x
                                                  ,ymin, ymax, periodic_y) ; 
    #else 
      pconn_ = detail::new_cartesian_connectivity(xmin, xmax, periodic_x
                                                  ,ymin, ymax, periodic_y
                                                  ,zmin, zmax, periodic_z) ; 
    #endif 
  } else if ( coord_system == "spherical" ) { 
    double  rmin{ params["amr"]["inner_region_radius"].as<double>() }
          , rmax{ params["amr"]["outer_region_radius"].as<double>() }
          , rlog{ params["amr"]["logarithmic_outer_radius"].as<double>() } ; 
    bool use_log_r { params["amr"]["use_logarithmic_radial_zone"].as<bool>() } ; 

    #ifndef THUNDER_3D 
      pconn_ = detail::new_spherical_connectivity(rmin, rmax, rlog, use_log_r) ; 
    #else 
      pconn_ = detail::new_spherical_connectivity(rmin, rmax, rlog, use_log_r) ;  
    #endif 
  } else { 
    ERROR("Unknown coordinate system.") ; 
  }
}
//**************************************************************************************************

}} /* namespace thunder::amr */