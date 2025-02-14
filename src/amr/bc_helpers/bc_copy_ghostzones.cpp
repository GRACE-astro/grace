/**
 * @file bc_helpers.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Index fiesta.
 * @date 2024-03-21
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

#include <grace/amr/bc_copy_ghostzones.hh>
#include <grace/amr/prolongation_kernels.tpp> 
#include <grace/amr/restriction_kernels.tpp> 
#include <grace/utils/numerics/prolongation.hh>
#include <grace/utils/numerics/limiters.hh> 
#include <grace/utils/numerics/restriction.hh>
#include <grace/amr/boundary_conditions.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/bc_helpers.tpp>
#include <grace/amr/grace_amr.hh> 
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/numerics/interpolators.hh>
#include <grace/data_structures/grace_data_structures.hh>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

namespace grace { namespace amr { 


namespace detail {

template< bool stagger_x, bool stagger_y, bool stagger_z > 
void copy_interior_ghostzones_impl(
        grace::var_array_t<GRACE_NSPACEDIM>& vars
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::device_vector<simple_face_info_t>& interior_faces
    , grace::device_vector<simple_corner_info_t>& interior_corners
    #ifdef GRACE_3D
    , grace::device_vector<simple_edge_info_t>& interior_edges
    #endif 
)
{
    using namespace grace; 
    using namespace Kokkos ; 

    int nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = amr::get_n_ghosts() ;
    int nq  = amr::get_local_num_quadrants() ;
    int nvars  = variables::get_n_evolved()      ;
    size_t const n_faces = interior_faces.size()   ;
    size_t const n_corners = interior_corners.size()   ;
    #ifdef GRACE_3D
    size_t const n_edges = interior_edges.size()   ;
    #endif 
    if( EXPR( n_faces == 0, and n_corners == 0, and n_edges==0) ) {
        return ; 
    }
    #define LOW_DST_GZS_X Kokkos::pair<int,int>{0,ngz+static_cast<int>(stagger_x)} 
    #define LOW_DST_GZS_Y Kokkos::pair<int,int>{0,ngz+static_cast<int>(stagger_y)} 
    #define LOW_DST_GZS_Z Kokkos::pair<int,int>{0,ngz+static_cast<int>(stagger_z)}

    #define HIGH_DST_GZS_X Kokkos::pair<int,int>{nx+ngz,nx+2*ngz+static_cast<int>(stagger_x)}
    #define HIGH_DST_GZS_Y Kokkos::pair<int,int>{ny+ngz,ny+2*ngz+static_cast<int>(stagger_y)}
    #define HIGH_DST_GZS_Z Kokkos::pair<int,int>{nz+ngz,nz+2*ngz+static_cast<int>(stagger_z)}

    #define LOW_SRC_GZS_X Kokkos::pair<int,int>{ngz,2*ngz+static_cast<int>(stagger_x)}
    #define LOW_SRC_GZS_Y Kokkos::pair<int,int>{ngz,2*ngz+static_cast<int>(stagger_y)}
    #define LOW_SRC_GZS_Z Kokkos::pair<int,int>{ngz,2*ngz+static_cast<int>(stagger_z)}

    #define HIGH_SRC_GZS_X Kokkos::pair<int,int>{nx,nx+ngz+static_cast<int>(stagger_x)}
    #define HIGH_SRC_GZS_Y Kokkos::pair<int,int>{ny,ny+ngz+static_cast<int>(stagger_y)}
    #define HIGH_SRC_GZS_Z Kokkos::pair<int,int>{nz,nz+ngz+static_cast<int>(stagger_z)}

    #define INNER_DOMAIN_X Kokkos::pair<int,int>{ngz,nx+ngz+static_cast<int>(stagger_x)}
    #define INNER_DOMAIN_Y Kokkos::pair<int,int>{ngz,ny+ngz+static_cast<int>(stagger_y)}
    #define INNER_DOMAIN_Z Kokkos::pair<int,int>{ngz,nz+ngz+static_cast<int>(stagger_z)}

    using array_of_pairs_t = std::array<Kokkos::pair<int,int>,GRACE_NSPACEDIM> ;
    const std::array<array_of_pairs_t,P4EST_FACES>
        dst_face_mask {
            // Face 0 --> LOW X 
            array_of_pairs_t{
                VEC(LOW_DST_GZS_X, INNER_DOMAIN_Y, INNER_DOMAIN_Z)
            },
            // Face 1 --> HIGH X 
            array_of_pairs_t{
                VEC(HIGH_DST_GZS_X, INNER_DOMAIN_Y, INNER_DOMAIN_Z)
            },
            // Face 2 --> LOW Y
            array_of_pairs_t{
                VEC(INNER_DOMAIN_X, LOW_DST_GZS_Y, INNER_DOMAIN_Z)
            },
            // Face 3 --> HIGH Y
            array_of_pairs_t{
                VEC(INNER_DOMAIN_X, HIGH_DST_GZS_Y, INNER_DOMAIN_Z)
            },
            // Face 4 --> LOW Z
            array_of_pairs_t{
                VEC(INNER_DOMAIN_X, INNER_DOMAIN_Y, LOW_DST_GZS_Z)
            },
            // Face 5 --> HIGH Z
            array_of_pairs_t{
                VEC(INNER_DOMAIN_X, INNER_DOMAIN_Y, HIGH_DST_GZS_Z)
            }
        } ; 
    const std::array<array_of_pairs_t,P4EST_FACES>
        src_face_mask {
            // Face 0 --> LOW X 
            array_of_pairs_t{
                VEC(LOW_SRC_GZS_X, INNER_DOMAIN_Y, INNER_DOMAIN_Z)
            },
            // Face 1 --> HIGH X 
            array_of_pairs_t{
                VEC(HIGH_SRC_GZS_X, INNER_DOMAIN_Y, INNER_DOMAIN_Z)
            },
            // Face 2 --> LOW Y
            array_of_pairs_t{
                VEC(INNER_DOMAIN_X, LOW_SRC_GZS_Y, INNER_DOMAIN_Z)
            },
            // Face 3 --> HIGH Y
            array_of_pairs_t{
                VEC(INNER_DOMAIN_X, HIGH_SRC_GZS_Y, INNER_DOMAIN_Z)
            },
            // Face 4 --> LOW Z
            array_of_pairs_t{
                VEC(INNER_DOMAIN_X, INNER_DOMAIN_Y, LOW_SRC_GZS_Z)
            },
            // Face 5 --> HIGH Z
            array_of_pairs_t{
                VEC(INNER_DOMAIN_X, INNER_DOMAIN_Y, HIGH_SRC_GZS_Z)
            }
        } ;
    const std::array<array_of_pairs_t, P4EST_CHILDREN> 
        dst_corner_mask {
            // CORNER 0 --> low xyz 
            array_of_pairs_t{
                VEC(LOW_DST_GZS_X, LOW_DST_GZS_Y, LOW_DST_GZS_Z)
            },
            // CORNER 1 --> high x low yz 
            array_of_pairs_t{
                VEC(HIGH_DST_GZS_X,  LOW_DST_GZS_Y, LOW_DST_GZS_Z)
            },
            // CORNER 2 --> high y low xz 
            array_of_pairs_t{
                VEC(LOW_DST_GZS_X,  HIGH_DST_GZS_Y, LOW_DST_GZS_Z)
            },
            // CORNER 3 --> high xy low z
            array_of_pairs_t{
                VEC(HIGH_DST_GZS_X,  HIGH_DST_GZS_Y, LOW_DST_GZS_X)
            },
            // CORNER 4 --> low xy high z 
            array_of_pairs_t{
                VEC(LOW_DST_GZS_X, LOW_DST_GZS_Y, HIGH_DST_GZS_Z)
            },
            // CORNER 5 --> high xz low y 
            array_of_pairs_t{
                VEC(HIGH_DST_GZS_X, LOW_DST_GZS_Y, HIGH_DST_GZS_Z)
            },
            // CORNER 6 --> high yz low x 
            array_of_pairs_t{
                VEC(LOW_DST_GZS_X, HIGH_DST_GZS_Y, HIGH_DST_GZS_Z)
            },
            // CORNER 7 --> high all 
            array_of_pairs_t{
                VEC(HIGH_DST_GZS_X, HIGH_DST_GZS_Y, HIGH_DST_GZS_Z)
            }
        } ;
    const std::array<array_of_pairs_t, P4EST_CHILDREN> 
        src_corner_mask {
            // CORNER 0 --> low xyz 
            array_of_pairs_t{
                VEC(LOW_SRC_GZS_X, LOW_SRC_GZS_Y, LOW_SRC_GZS_Z)
            },
            // CORNER 1 --> high x low yz 
            array_of_pairs_t{
                VEC(HIGH_SRC_GZS_X,  LOW_SRC_GZS_Y, LOW_SRC_GZS_Z)
            },
            // CORNER 2 --> high y low xz 
            array_of_pairs_t{
                VEC(LOW_SRC_GZS_X,  HIGH_SRC_GZS_Y, LOW_SRC_GZS_Z)
            },
            // CORNER 3 --> high xy low z
            array_of_pairs_t{
                VEC(HIGH_SRC_GZS_X,  HIGH_SRC_GZS_Y, LOW_SRC_GZS_Z)
            },
            // CORNER 4 --> low xy high z 
            array_of_pairs_t{
                VEC(LOW_SRC_GZS_X, LOW_SRC_GZS_Y, HIGH_SRC_GZS_Z)
            },
            // CORNER 5 --> high xz low y 
            array_of_pairs_t{
                VEC(HIGH_SRC_GZS_X, LOW_SRC_GZS_Y, HIGH_SRC_GZS_Z)
            },
            // CORNER 6 --> high yz low x 
            array_of_pairs_t{
                VEC(LOW_SRC_GZS_X, HIGH_SRC_GZS_Y, HIGH_SRC_GZS_Z)
            },
            // CORNER 7 --> high all 
            array_of_pairs_t{
                VEC(HIGH_SRC_GZS_X, HIGH_SRC_GZS_Y, HIGH_SRC_GZS_Z)
            }
        } ; 
    #ifdef GRACE_3D 
    const std::array<array_of_pairs_t, 12> 
        dst_edge_mask {
            // EDGE 0 --> low yz, along x  
            array_of_pairs_t {
                INNER_DOMAIN_X, LOW_DST_GZS_Y, LOW_DST_GZS_Z
            } , 
            // EDGE 1 --> high y low z along x 
            array_of_pairs_t {
                INNER_DOMAIN_X, HIGH_DST_GZS_Y, LOW_DST_GZS_Z
            } , 
            // EDGE 2 --> high z low y along x 
            array_of_pairs_t {
                INNER_DOMAIN_X, LOW_DST_GZS_Y, HIGH_DST_GZS_Z
            } , 
            // EDGE 3 --> high yz along x
            array_of_pairs_t {
                INNER_DOMAIN_X, HIGH_DST_GZS_Y, HIGH_DST_GZS_Z
            } ,
            // EDGE 4 --> low xz along y 
            array_of_pairs_t {
                LOW_DST_GZS_X, INNER_DOMAIN_Y, LOW_DST_GZS_Z
            } ,
            // EDGE 5 --> high x low z along y 
            array_of_pairs_t {
                HIGH_DST_GZS_X, INNER_DOMAIN_Y, LOW_DST_GZS_Z
            } ,
            // EDGE 6 --> high z low x along y 
            array_of_pairs_t {
                LOW_DST_GZS_X, INNER_DOMAIN_Y, HIGH_DST_GZS_Z
            } ,
            // EDGE 7 --> high xz along y
            array_of_pairs_t {
                HIGH_DST_GZS_X, INNER_DOMAIN_Y, HIGH_DST_GZS_Z
            } ,
            // EDGE 8 --> low xy along z 
            array_of_pairs_t {
                LOW_DST_GZS_X, LOW_DST_GZS_Y, INNER_DOMAIN_Z
            } ,
            // EDGE 9 --> high x low y along z 
            array_of_pairs_t {
                HIGH_DST_GZS_X, LOW_DST_GZS_Y, INNER_DOMAIN_Z
            } ,
            // EDGE 10 --> low x high y along z 
            array_of_pairs_t {
                LOW_DST_GZS_X, HIGH_DST_GZS_Y, INNER_DOMAIN_Z
            } ,
            // EDGE 10 --> high xy along z 
            array_of_pairs_t {
                HIGH_DST_GZS_X, HIGH_DST_GZS_Y, INNER_DOMAIN_Z
            } ,
    }; 
    const std::array<array_of_pairs_t, 12> 
        src_edge_mask {
            // EDGE 0 --> low yz, along x  
            array_of_pairs_t {
                INNER_DOMAIN_X, LOW_SRC_GZS_Y, LOW_SRC_GZS_Z
            } , 
            // EDGE 1 --> high y low z along x 
            array_of_pairs_t {
                INNER_DOMAIN_X, HIGH_SRC_GZS_Y, LOW_SRC_GZS_Z
            } , 
            // EDGE 2 --> high z low y along x 
            array_of_pairs_t {
                INNER_DOMAIN_X, LOW_SRC_GZS_Y, HIGH_SRC_GZS_Z
            } , 
            // EDGE 3 --> high yz along x
            array_of_pairs_t {
                INNER_DOMAIN_X, HIGH_SRC_GZS_Y, HIGH_SRC_GZS_Z
            } ,
            // EDGE 4 --> low xz along y 
            array_of_pairs_t {
                LOW_SRC_GZS_X, INNER_DOMAIN_Y, LOW_SRC_GZS_Z
            } ,
            // EDGE 5 --> high x low z along y 
            array_of_pairs_t {
                HIGH_SRC_GZS_X, INNER_DOMAIN_Y, LOW_SRC_GZS_Z
            } ,
            // EDGE 6 --> high z low x along y 
            array_of_pairs_t {
                LOW_SRC_GZS_X, INNER_DOMAIN_Y, HIGH_SRC_GZS_Z
            } ,
            // EDGE 7 --> high xz along y
            array_of_pairs_t {
                HIGH_SRC_GZS_X, INNER_DOMAIN_Y, HIGH_SRC_GZS_Z
            } ,
            // EDGE 8 --> low xy along z 
            array_of_pairs_t {
                LOW_SRC_GZS_X, LOW_SRC_GZS_Y, INNER_DOMAIN_Z
            } ,
            // EDGE 9 --> high x low y along z 
            array_of_pairs_t {
                HIGH_SRC_GZS_X, LOW_SRC_GZS_Y, INNER_DOMAIN_Z
            } ,
            // EDGE 10 --> low x high y along z 
            array_of_pairs_t {
                LOW_SRC_GZS_X, HIGH_SRC_GZS_Y, INNER_DOMAIN_Z
            } ,
            // EDGE 10 --> high xy along z 
            array_of_pairs_t {
                HIGH_SRC_GZS_X, HIGH_SRC_GZS_Y, INNER_DOMAIN_Z
            } ,
    };
    #endif 

    auto const perform_copy = [](
        auto const& src, 
        auto const& dest, 
        auto const& src_mask_arr,
        auto const& dst_mask_arr,
        int64_t qid_src,
        int64_t qid_dst,
        int dir_src,
        int dir_dst
    ) {
        auto dst_view =
            Kokkos::subview(
                dest,
                VEC(
                    dst_mask_arr[dir_dst][0],
                    dst_mask_arr[dir_dst][1],
                    dst_mask_arr[dir_dst][2]
                ),
                Kokkos::ALL(),
                qid_dst
            ) ; 
        auto src_view = 
            Kokkos::subview(
                src, 
                VEC(
                    src_mask_arr[dir_src][0],
                    src_mask_arr[dir_src][1],
                    src_mask_arr[dir_src][2]
                ),
                Kokkos::ALL(),
                qid_src
            ) ; 
        Kokkos::deep_copy(
            grace::default_execution_space{}, dst_view, src_view 
        ) ; 
    } ; 

    for( int iface = 0; iface < n_faces; ++iface) {
        auto src_view = interior_faces[iface].is_ghost ? halo : vars ;
        perform_copy(
            src_view, 
            vars,
            src_face_mask,
            dst_face_mask,
            interior_faces[iface].qid_b,
            interior_faces[iface].qid_a,
            interior_faces[iface].which_face_b,
            interior_faces[iface].which_face_a
        ) ; 
        if ( not interior_faces[iface].is_ghost ) {
            perform_copy(
                vars,
                vars,
                src_face_mask,
                dst_face_mask,
                interior_faces[iface].qid_a,
                interior_faces[iface].qid_b,
                interior_faces[iface].which_face_a,
                interior_faces[iface].which_face_b
            ) ; 
        }
    }

    for( int icorner=0; icorner< n_corners; ++icorner ) {
        auto src_view_proxy = interior_corners[icorner].is_ghost ? halo : vars ; 
        perform_copy(
            src_view_proxy,
            vars,
            src_corner_mask,
            dst_corner_mask,
            interior_corners[icorner].qid_b,
            interior_corners[icorner].qid_a,
            interior_corners[icorner].which_corner_b,
            interior_corners[icorner].which_corner_a
        ) ; 
        if ( not interior_corners[icorner].is_ghost ) {
            perform_copy(
                vars,
                vars,
                src_corner_mask,
                dst_corner_mask,
                interior_corners[icorner].qid_a,
                interior_corners[icorner].qid_b,
                interior_corners[icorner].which_corner_a,
                interior_corners[icorner].which_corner_b
            ) ;
        }
    }
    #ifdef GRACE_3D 
    for( int iedge=0; iedge< n_edges; ++iedge ) {
        auto src_view_proxy = interior_edges[iedge].is_ghost ? halo : vars ; 
        perform_copy(
            src_view_proxy,
            vars,
            src_edge_mask,
            dst_edge_mask,
            interior_edges[iedge].qid_b,
            interior_edges[iedge].qid_a,
            interior_edges[iedge].which_edge_b,
            interior_edges[iedge].which_edge_a
        ) ; 
        if ( not interior_edges[iedge].is_ghost ) {
            perform_copy(
                vars,
                vars,
                src_edge_mask,
                dst_edge_mask,
                interior_edges[iedge].qid_a,
                interior_edges[iedge].qid_b,
                interior_edges[iedge].which_edge_a,
                interior_edges[iedge].which_edge_b
            ) ;
        }
    }
    #endif 
}

}

void copy_interior_ghostzones(
      grace::var_array_t<GRACE_NSPACEDIM>& vars
    , grace::var_array_t<GRACE_NSPACEDIM>& halo 
    , grace::staggered_variable_arrays_t& staggered_state
    , grace::staggered_variable_arrays_t& staggered_halo
    , grace::device_vector<simple_face_info_t>& interior_faces
    , grace::device_vector<simple_corner_info_t>& interior_corners
    #ifdef GRACE_3D
    , grace::device_vector<simple_edge_info_t>& interior_edges
    #endif 
)
{
    /******************************************************/
    /*                CELL CENTERS                        */
    /******************************************************/
    detail::copy_interior_ghostzones_impl<false,false,false>
    (
          vars
        , halo 
        , interior_faces 
        , interior_corners 
        #ifdef GRACE_3D 
        , interior_edges 
        #endif 
    ) ; 
    Kokkos::fence() ; 
    GRACE_VERBOSE("Done cell centers start corners.") ; 
    /******************************************************/
    /*                CELL CORNERS                        */
    /******************************************************/
    #if 1
    detail::copy_interior_ghostzones_impl<true,true,true>(
          staggered_state.corner_staggered_fields
        , staggered_halo.corner_staggered_fields
        , interior_faces 
        , interior_corners 
        #ifdef GRACE_3D 
        , interior_edges 
        #endif 
    ) ;
    #endif 
    Kokkos::fence() ;
    GRACE_VERBOSE("All done in copy.") ; 
}

}} /* namespace grace::amr */
