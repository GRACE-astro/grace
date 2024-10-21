/**
 * @file bc_helpers.cpp
 * @author  Carlo Musolino
 * @brief 
 * @date 2024-09-13
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

#include <grace_config.h>

#include <grace/amr/bc_helpers.hh>
#include <grace/amr/boundary_conditions.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/bc_helpers.tpp>
#include <grace/amr/grace_amr.hh> 
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/numerics/interpolators.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>

namespace grace { namespace amr { 

void grace_init_halo_transfer(
    grace_transfer_context_t& context       ,
    p4est_ghost_t*                     halos          ,
    sc_array_view_t<p4est_quadrant_t>& halo_quads     , 
    sc_array_view_t<p4est_quadrant_t>& mirror_quads   ,
    grace::var_array_t<GRACE_NSPACEDIM>& halo          , 
    grace::staggered_variable_arrays_t& staggered_halo,
    cell_vol_array_t<GRACE_NSPACEDIM>&  halo_vols     ,
    grace::var_array_t<GRACE_NSPACEDIM>& vars          , 
    grace::staggered_variable_arrays_t& staggered_vars,
    cell_vol_array_t<GRACE_NSPACEDIM>&  vols          ,
    bool exchange_cell_volumes
) 
{
    DECLARE_GRID_EXTENTS ; 
    /******************************************************/
    /*                  Fetch nvars                       */
    /******************************************************/
    size_t nvars = variables::get_n_evolved() ;
    size_t nvars_face = variables::get_n_evolved_face_staggered() ; 
    size_t nvars_edge = variables::get_n_evolved_edge_staggered() ; 
    size_t nvars_corner = variables::get_n_evolved_corner_staggered() ; 
    GRACE_TRACE("Inside init_halo transfer nvars {}, nvars_corner {}", nvars, nvars_corner) ; 
    GRACE_TRACE("Nhalo {} nmirror {}", halo_quads.size(), mirror_quads.size()) ; 
    GRACE_TRACE("halo view shape {} {} {} {} {}", halo.extent(0), halo.extent(1), halo.extent(2), halo.extent(3), halo.extent(4)) ; 
    GRACE_TRACE("staggered halo (corner) view shape {} {} {} {} {}", staggered_halo.corner_staggered_fields.extent(0), staggered_halo.corner_staggered_fields.extent(1), staggered_halo.corner_staggered_fields.extent(2), staggered_halo.corner_staggered_fields.extent(3), staggered_halo.corner_staggered_fields.extent(4)) ; 
    /******************************************************/
    /*          Compute send / receive buffer sizes       */
    /******************************************************/
    size_t const send_size_vol = EXPR((nx+2*ngz), *(ny+2*ngz), *(nz+2*ngz)) ; 
    size_t const send_size = send_size_vol * nvars ; 
    size_t const send_size_face_staggered_x = 
        EXPR((nx+1+2*ngz),*(ny+2*ngz), *(nz+2*ngz)) * nvars_face; 
    size_t const send_size_face_staggered_y = 
        EXPR((nx+2*ngz), *(ny+1+2*ngz), *(nz+2*ngz)) * nvars_face;
    size_t const send_size_face_staggered_z = 
        EXPR((nx+2*ngz), *(ny+2*ngz), *(nz+1+2*ngz)) * nvars_face;
    size_t const send_size_edge_staggered_xy = 
        EXPR((nx+1+2*ngz), *(ny+1+2*ngz), *(nz+2*ngz)) * nvars_edge;
    size_t const send_size_edge_staggered_xz = 
        EXPR((nx+1+2*ngz), *(ny+2*ngz), *(nz+1+2*ngz)) * nvars_edge;
    size_t const send_size_edge_staggered_yz = 
        EXPR((nx+2*ngz), *(ny+1+2*ngz), *(nz+1+2*ngz)) * nvars_edge;
    size_t const send_size_corner_staggered = 
        EXPR((nx+1+2*ngz), *(ny+1+2*ngz), *(nz+1+2*ngz)) * nvars_corner;
    /******************************************************/
    /*                Receive halo data                   */
    /******************************************************/
    size_t rank = parallel::mpi_comm_rank() ; 
    for(int iproc=0; iproc<parallel::mpi_comm_size(); ++iproc){
        int tag = parallel::GRACE_HALO_EXCHANGE_TAG ;
        size_t first_halo  = halos->proc_offsets[iproc]   ; 
        size_t last_halo   = halos->proc_offsets[iproc+1] ;
        int const n_halo = last_halo - first_halo ; 
        GRACE_TRACE("Receiving {} ghost quadrants ({}-{}) from procid {}", n_halo,first_halo,last_halo,iproc) ; 
        if ( n_halo > 0 ) {
            /* Receive variables */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hsview = Kokkos::subview(
                      halo
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , first_halo) ;
            parallel::mpi_irecv(
                  hsview.data()
                , send_size * n_halo 
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
            if ( send_size_face_staggered_x > 0 ) {
                /* X-face */
                context._requests.push_back(sc_MPI_Request{}) ; 
                auto hsview_fx = Kokkos::subview(
                      staggered_halo.face_staggered_fields_x
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , first_halo) ; 
                parallel::mpi_irecv(
                      hsview_fx.data()
                    , send_size_face_staggered_x * n_halo 
                    , iproc
                    , parallel::GRACE_HALO_EXCHANGE_TAG+1
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ;
            }
            if ( send_size_face_staggered_y > 0 ) {
                /* Y-face */
                context._requests.push_back(sc_MPI_Request{}) ; 
                auto hsview_fy = Kokkos::subview(
                      staggered_halo.face_staggered_fields_y
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , first_halo) ; 
                parallel::mpi_irecv(
                      hsview_fy.data()
                    , send_size_face_staggered_y * n_halo 
                    , iproc
                    , parallel::GRACE_HALO_EXCHANGE_TAG+2
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ;
            }
            if ( send_size_face_staggered_z > 0 ) {
                /* Z-face */
                context._requests.push_back(sc_MPI_Request{}) ; 
                auto hsview_fz = Kokkos::subview(
                      staggered_halo.face_staggered_fields_z
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , first_halo) ; 
                parallel::mpi_irecv(
                      hsview_fz.data()
                    , send_size_face_staggered_z * n_halo 
                    , iproc
                    , parallel::GRACE_HALO_EXCHANGE_TAG+3
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ;
            }
            if ( send_size_edge_staggered_xy > 0 ) {
                /* XY-edge */
                context._requests.push_back(sc_MPI_Request{}) ; 
                auto hsview_exy = Kokkos::subview(
                      staggered_halo.edge_staggered_fields_xy 
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , first_halo) ; 
                parallel::mpi_irecv(
                      hsview_exy.data()
                    , send_size_edge_staggered_xy * n_halo  
                    , iproc
                    , parallel::GRACE_HALO_EXCHANGE_TAG+4
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ;
            }
            if ( send_size_edge_staggered_xz > 0 ) {
                /* XZ-edge */
                context._requests.push_back(sc_MPI_Request{}) ; 
                auto hsview_exz = Kokkos::subview(
                      staggered_halo.edge_staggered_fields_xz
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , first_halo) ; 
                parallel::mpi_irecv(
                      hsview_exz.data()
                    , send_size_edge_staggered_xz * n_halo 
                    , iproc
                    , parallel::GRACE_HALO_EXCHANGE_TAG+5
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ;
            }
            if ( send_size_edge_staggered_yz > 0 ) {
                /* YZ-edge */
                context._requests.push_back(sc_MPI_Request{}) ; 
                auto hsview_eyz = Kokkos::subview(
                      staggered_halo.edge_staggered_fields_yz
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , first_halo) ; 
                parallel::mpi_irecv(
                      hsview_eyz.data()
                    , send_size_edge_staggered_yz * n_halo 
                    , iproc
                    , parallel::GRACE_HALO_EXCHANGE_TAG+6
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ;
            }
            if ( send_size_corner_staggered > 0 ) {
                /* corner */
                context._requests.push_back(sc_MPI_Request{}) ; 
                auto hsview_c = Kokkos::subview(
                      staggered_halo.corner_staggered_fields
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , first_halo ) ; 
                parallel::mpi_irecv(
                      hsview_c.data()
                    , send_size_corner_staggered * n_halo 
                    , iproc
                    , parallel::GRACE_HALO_EXCHANGE_TAG+7
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ;
            }
        }
    }
    /******************************************************/
    /*                Send halo data                      */
    /******************************************************/
     
    for( int iproc=0; iproc<parallel::mpi_comm_size(); ++iproc) {
        int tag = parallel::GRACE_HALO_EXCHANGE_TAG ;
        size_t first_mirror = halos->mirror_proc_offsets[iproc]   ; 
        size_t last_mirror  = halos->mirror_proc_offsets[iproc+1] ; 
        int const n_mirrors = last_mirror - first_mirror ; 
        if ( n_mirrors <= 0 ) {
            continue ; 
        }
        grace::var_array_t<GRACE_NSPACEDIM> mirror(
            "mirror_data", VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz), nvars, n_mirrors
        ) ; 
        grace::staggered_variable_arrays_t staggered_mirror{};
        staggered_mirror.realloc(VEC(nx,ny,nz),ngz,n_mirrors,nvars_face,nvars_edge,nvars_corner) ; 

        context._buffs.push_back(
            mirror
        ) ; 
        context._staggered_buffs.push_back(
            staggered_mirror
        ) ; 
        GRACE_TRACE("Send procid {} first {} last {}", iproc, first_mirror, last_mirror) ; 
        for( int imirror=0; imirror<n_mirrors; ++imirror){
            size_t iq_loc = 
                (mirror_quads[halos->mirror_proc_mirrors[imirror+first_mirror]]).p.piggy3.local_num ; 
            // Do some range checking 
            ASSERT(
                iq_loc < vars.extent(GRACE_NSPACEDIM+1), 
                "Out of bounds access for vars in mpi_isend."
            ) ; 
            ASSERT(
                imirror < mirror.extent(GRACE_NSPACEDIM+1),
                "Out of bounds access for mirrors in mpi_isend."
            ) ; 
            auto sview = Kokkos::subview(
                      vars
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , iq_loc ) ;
            auto msview = Kokkos::subview(
                      mirror
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , imirror ) ;
            GRACE_TRACE("Copying {} to {}", reinterpret_cast<void*>(sview.data()) ,reinterpret_cast<void*>(msview.data()));
            Kokkos::fence() ; 
            Kokkos::deep_copy( msview, sview ) ; 
            Kokkos::fence() ; 
            GRACE_TRACE("Sending data ptr {}", reinterpret_cast<void*>(msview.data())) ; 
            if( send_size_face_staggered_x > 0 ) {
                /* X-Face */
                auto sview = Kokkos::subview(
                      staggered_vars.face_staggered_fields_x
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , iq_loc ) ;
                auto msview = Kokkos::subview(
                          staggered_mirror.face_staggered_fields_x
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , imirror ) ;
                Kokkos::deep_copy( msview, sview ) ; 
            } 
            if( send_size_face_staggered_y > 0 ) {
                /* Y-Face */
                auto sview = Kokkos::subview(
                      staggered_vars.face_staggered_fields_y
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , iq_loc ) ;
                auto msview = Kokkos::subview(
                          staggered_mirror.face_staggered_fields_y
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , imirror ) ;
                Kokkos::deep_copy( msview, sview ) ; 
            } 
            if( send_size_face_staggered_y > 0 ) {
                /* Z-Face */
                auto sview = Kokkos::subview(
                      staggered_vars.face_staggered_fields_z
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , iq_loc ) ;
                auto msview = Kokkos::subview(
                          staggered_mirror.face_staggered_fields_z
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , imirror ) ;
                Kokkos::deep_copy( msview, sview ) ; 
            }  
            if( send_size_edge_staggered_xy > 0 ) {
                /* XY-Edge */
                auto sview = Kokkos::subview(
                      staggered_vars.edge_staggered_fields_xy
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , iq_loc ) ;
                auto msview = Kokkos::subview(
                          staggered_mirror.edge_staggered_fields_xy
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , imirror ) ;
                Kokkos::deep_copy( msview, sview ) ; 
            } 
            if( send_size_edge_staggered_xz > 0 ) {
                /* XZ-Edge */
                auto sview = Kokkos::subview(
                      staggered_vars.edge_staggered_fields_xz
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , iq_loc ) ;
                auto msview = Kokkos::subview(
                          staggered_mirror.edge_staggered_fields_xz
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , imirror ) ;
                Kokkos::deep_copy( msview, sview ) ; 
            } 
            if( send_size_edge_staggered_yz > 0 ) {
                /* YZ-Edge */
                auto sview = Kokkos::subview(
                      staggered_vars.edge_staggered_fields_yz
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , iq_loc ) ;
                auto msview = Kokkos::subview(
                          staggered_mirror.edge_staggered_fields_yz
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , imirror ) ;
                Kokkos::deep_copy( msview, sview ) ; 
            }     
            if( send_size_corner_staggered > 0 ) {
                /* Corner */
                auto sview = Kokkos::subview(
                      staggered_vars.corner_staggered_fields
                    , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                    , Kokkos::ALL()
                    , iq_loc ) ;
                auto msview = Kokkos::subview(
                          staggered_mirror.corner_staggered_fields
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , imirror ) ;
                Kokkos::deep_copy( msview, sview ) ; 
            }  
        } // Loop over imirror 
            
        /* Send variables */
        context._requests.push_back(sc_MPI_Request{}) ; 
        parallel::mpi_isend(
              mirror.data() 
            , send_size * n_mirrors 
            , iproc
            , parallel::GRACE_HALO_EXCHANGE_TAG
            , parallel::get_comm_world()
            , &(context._requests.back())
        ) ; 
        if ( send_size_face_staggered_x > 0 ) {
            /* X-face */
            context._requests.push_back(sc_MPI_Request{}) ; 
            parallel::mpi_isend(
                  staggered_mirror.face_staggered_fields_x.data()
                , send_size_face_staggered_x * n_mirrors 
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG + 1 
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
        }
        if ( send_size_face_staggered_y > 0 ) {
            /* Y-face */
            context._requests.push_back(sc_MPI_Request{}) ; 
            parallel::mpi_isend(
                  staggered_mirror.face_staggered_fields_y.data()
                , send_size_face_staggered_y * n_mirrors 
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG + 2 
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
        }
        if ( send_size_face_staggered_z > 0 ) {
            /* Z-face */
            context._requests.push_back(sc_MPI_Request{}) ; 
            parallel::mpi_isend(
                  staggered_mirror.face_staggered_fields_z.data()
                , send_size_face_staggered_z * n_mirrors 
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG + 3 
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
        }
        if ( send_size_edge_staggered_xy > 0 ) {
            /* XY-edge */
            context._requests.push_back(sc_MPI_Request{}) ; 
            parallel::mpi_isend(
                  staggered_mirror.edge_staggered_fields_xy.data()
                , send_size_edge_staggered_xy * n_mirrors 
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG + 4 
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
        }
        if ( send_size_edge_staggered_xz > 0 ) {
            /* XZ-edge */
            context._requests.push_back(sc_MPI_Request{}) ; 
            parallel::mpi_isend(
                  staggered_mirror.edge_staggered_fields_xz.data()
                , send_size_edge_staggered_xz * n_mirrors 
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG + 5
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
        }
        if ( send_size_edge_staggered_yz > 0 ) {
            /* YZ-edge */
            context._requests.push_back(sc_MPI_Request{}) ; 
            parallel::mpi_isend(
                  staggered_mirror.edge_staggered_fields_yz.data()
                , send_size_edge_staggered_yz * n_mirrors 
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG + 6
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
        }
        if ( send_size_corner_staggered > 0 ) {
            GRACE_TRACE("sending corner.") ; 
            /* Corner */
            context._requests.push_back(sc_MPI_Request{}) ; 
            parallel::mpi_isend(
                  staggered_mirror.corner_staggered_fields.data()
                , send_size_corner_staggered * n_mirrors 
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG + 7
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
        }
    } 
}
 
void grace_init_halo_transfer_custom(
    grace_transfer_context_t& context       ,
    std::vector<int64_t> const& snd_quadid            ,
    std::vector<int64_t> const& rcv_quadid            ,  
    std::vector<std::set<int>> const& snd_procid      , 
    std::vector<int>     const& rcv_procid            ,
    grace::var_array_t<GRACE_NSPACEDIM>& halo          , 
    grace::staggered_variable_arrays_t& staggered_halo,
    cell_vol_array_t<GRACE_NSPACEDIM>&  halo_vols     ,
    grace::var_array_t<GRACE_NSPACEDIM>& vars          , 
    grace::staggered_variable_arrays_t& staggered_vars,
    cell_vol_array_t<GRACE_NSPACEDIM>&  vols          ,
    bool exchange_cell_volumes
)
{
    DECLARE_GRID_EXTENTS ; 
    /******************************************************/
    /*                  Fetch nvars                       */
    /******************************************************/
    size_t nvars = variables::get_n_evolved() ;
    size_t nvars_face = variables::get_n_evolved_face_staggered() ; 
    size_t nvars_edge = variables::get_n_evolved_edge_staggered() ; 
    size_t nvars_corner = variables::get_n_evolved_corner_staggered() ;  
    /******************************************************/
    /*          Compute send / receive buffer sizes       */
    /******************************************************/
    size_t const send_size_vol = EXPR((nx+2*ngz), *(ny+2*ngz), *(nz+2*ngz)) ; 
    size_t const send_size = send_size_vol * nvars ; 
    size_t const send_size_face_staggered_x = 
        EXPR((nx+1+2*ngz),*(ny+2*ngz), *(nz+2*ngz)) * nvars_face; 
    size_t const send_size_face_staggered_y = 
        EXPR((nx+2*ngz), *(ny+1+2*ngz), *(nz+2*ngz)) * nvars_face;
    size_t const send_size_face_staggered_z = 
        EXPR((nx+2*ngz), *(ny+2*ngz), *(nz+1+2*ngz)) * nvars_face;
    size_t const send_size_edge_staggered_xy = 
        EXPR((nx+1+2*ngz), *(ny+1+2*ngz), *(nz+2*ngz)) * nvars_edge;
    size_t const send_size_edge_staggered_xz = 
        EXPR((nx+1+2*ngz), *(ny+2*ngz), *(nz+1+2*ngz)) * nvars_edge;
    size_t const send_size_edge_staggered_yz = 
        EXPR((nx+2*ngz), *(ny+1+2*ngz), *(nz+1+2*ngz)) * nvars_edge;
    size_t const send_size_corner_staggered = 
        EXPR((nx+1+2*ngz), *(ny+1+2*ngz), *(nz+1+2*ngz)) * nvars_corner;

    /******************************************************/
    /*                Receive halo data                   */
    /******************************************************/
    for(int ircv=0; ircv<rcv_procid.size(); ++ircv) 
    {
        int tag = parallel::GRACE_HALO_EXCHANGE_TAG ;
        int ihalo = rcv_quadid[ircv] ; 
        int iproc = rcv_procid[ircv] ; 
        GRACE_VERBOSE("Receive iproc {} ihalo {}", iproc,ihalo) ; 
        int tag = parallel::GRACE_HALO_EXCHANGE_TAG; 
        /* Receive variables */
        context._requests.push_back(sc_MPI_Request{}) ; 
        auto hsview = Kokkos::subview(
                halo
            , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
            , Kokkos::ALL()
            , ihalo) ; 
        parallel::mpi_irecv(
                hsview.data()
            , send_size
            , iproc
            , tag
            , parallel::get_comm_world()
            , &(context._requests.back())
        ) ; 
        //tag++
        /* Receive face-staggered vars */
        if( nvars_face > 0 ) {
            /* X-face */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hsview_fx = Kokkos::subview(
                    staggered_halo.face_staggered_fields_x
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                    hsview_fx.data()
                , send_size_face_staggered_x
                , iproc
                , tag
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ;
            //tag++
            /* Y-face */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hsview_fy = Kokkos::subview(
                    staggered_halo.face_staggered_fields_y
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                    hsview_fy.data()
                , send_size_face_staggered_y
                , iproc
                , tag
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ;
            //tag++
            /* Z-face */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hsview_fz = Kokkos::subview(
                    staggered_halo.face_staggered_fields_z
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                    hsview_fz.data()
                , send_size_face_staggered_z
                , iproc
                , tag
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ;
            //tag++
        }
        /* Receive edge-staggered vars */
        #ifdef GRACE_3D 
        if (nvars_edge > 0) {
            /* XY-edge */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hsview_exy = Kokkos::subview(
                    staggered_halo.edge_staggered_fields_xy
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                    hsview_exy.data()
                , send_size_edge_staggered_xy
                , iproc
                , tag
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ;
            //tag++
            /* XZ-edge */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hsview_exz = Kokkos::subview(
                    staggered_halo.edge_staggered_fields_xz
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                    hsview_exz.data()
                , send_size_edge_staggered_xz
                , iproc
                , tag
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ;
            //tag++
            /* YZ-edge */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hsview_eyz = Kokkos::subview(
                    staggered_halo.edge_staggered_fields_yz
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                    hsview_eyz.data()
                , send_size_edge_staggered_xy
                , iproc
                , tag
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ;
            //tag++
        }
        #endif 
        /* Receive corner-staggered vars */
        if ( nvars_corner > 0 ) {
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hsview_c = Kokkos::subview(
                    staggered_halo.corner_staggered_fields
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                    hsview_c.data()
                , send_size_corner_staggered
                , iproc
                , tag
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ;
            //tag++
        }
        if (exchange_cell_volumes) {
            /* Receive cell volumes */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hvsview = Kokkos::subview(
                  halo_vols
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , ihalo 
            ) ; 
            parallel::mpi_irecv(
                  hvsview.data()
                , send_size_vol 
                , iproc
                , tag
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
            //tag++
        }
    }
    /******************************************************/
    /*                Send halo data                      */
    /******************************************************/
    for( int isend=0; isend<snd_quadid.size(); ++isend){
        int64_t iq_loc = snd_quadid[isend] ; 
        auto sview = Kokkos::subview(
                  vars
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , iq_loc ) ;
        for( auto const& iproc: snd_procid[isend] ) {
            GRACE_VERBOSE("Send iproc {} iq {}", iproc, iq_loc) ; 
            int tag = parallel::GRACE_HALO_EXCHANGE_TAG ; 
            /* Send variables */
            context._requests.push_back(sc_MPI_Request{}) ; 
            parallel::mpi_isend(
                  sview.data()
                , send_size
                , iproc
                , tag
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
            tag++ ; 
            /* Send face-staggered vars */
            if (nvars_face > 0 ) {
                auto sview_fx = Kokkos::subview(
                        staggered_vars.face_staggered_fields_x
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , iq_loc ) ;
                auto sview_fy = Kokkos::subview(
                        staggered_vars.face_staggered_fields_y
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , iq_loc ) ;
                auto sview_fz = Kokkos::subview(
                        staggered_vars.face_staggered_fields_z
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , iq_loc ) ; 
                /* X-face */
                context._requests.push_back(sc_MPI_Request{}) ; 
                parallel::mpi_isend(
                    sview_fx.data()
                    , send_size_face_staggered_x
                    , iproc
                    , tag
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ; 
                tag++ ; 
                /* Y-face */
                context._requests.push_back(sc_MPI_Request{}) ; 
                parallel::mpi_isend(
                    sview_fy.data()
                    , send_size_face_staggered_y
                    , iproc
                    , tag
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ; 
                tag++ ; 
                /* Z-face */
                context._requests.push_back(sc_MPI_Request{}) ; 
                parallel::mpi_isend(
                    sview_fz.data()
                    , send_size_face_staggered_z
                    , iproc
                    , tag
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ; 
                tag++ ; 
            }
            /* Send edge-staggered vars */
            #ifdef GRACE_3D
            if( nvars_edge > 0 ){
                auto sview_exy = Kokkos::subview(
                        staggered_vars.edge_staggered_fields_xy
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , iq_loc ) ;
                auto sview_exz = Kokkos::subview(
                        staggered_vars.edge_staggered_fields_xz
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , iq_loc ) ;
                auto sview_eyz = Kokkos::subview(
                        staggered_vars.edge_staggered_fields_yz
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , iq_loc ) ;
                /* XY edge */
                context._requests.push_back(sc_MPI_Request{}) ; 
                parallel::mpi_isend(
                    sview_exy.data()
                    , send_size_edge_staggered_xy
                    , iproc
                    , tag
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ; 
                tag++ ;  
                /* XZ edge */
                context._requests.push_back(sc_MPI_Request{}) ; 
                parallel::mpi_isend(
                    sview_exz.data()
                    , send_size_edge_staggered_xz
                    , iproc
                    , tag
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ; 
                tag++ ;
                /* YZ edge */
                context._requests.push_back(sc_MPI_Request{}) ; 
                parallel::mpi_isend(
                    sview_exy.data()
                    , send_size_edge_staggered_yz
                    , iproc
                    , tag
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ; 
                tag++ ;
            }
            #endif 
            /* Send corner-staggered vars */
            if( nvars_corner > 0 ) {
                auto sview_c = Kokkos::subview(
                        staggered_vars.corner_staggered_fields
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , iq_loc ) ;
                
                context._requests.push_back(sc_MPI_Request{}) ; 
                parallel::mpi_isend(
                    sview_c.data()
                    , send_size_corner_staggered
                    , iproc
                    , tag
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ; 
                tag++ ;
            }
            if( exchange_cell_volumes) {
                auto svview = Kokkos::subview(
                        vols
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , iq_loc ) ;
                /* Send cell volumes */
                context._requests.push_back(sc_MPI_Request{}) ; 
                parallel::mpi_isend(
                    svview.data()
                    , send_size_vol
                    , iproc
                    , tag
                    , parallel::get_comm_world()
                    , &(context._requests.back())
                ) ; 
                //tag++
            }
        }
    } 
}; 


void grace_finalize_halo_transfer(grace_transfer_context_t& context) 
{
    parallel::mpi_waitall(context._requests) ;
    // Now we free the space 
    context.reset() ; 
}; 

} } /* namespace grace::amr */