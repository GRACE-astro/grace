/**
 * @file boundary_conditions.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
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

#include <grace_config.h>

#include <Kokkos_Core.hpp>

#include <grace/amr/grace_amr.hh>
#include <grace/amr/bc_helpers.hh> 
#include <grace/amr/bc_helpers.tpp> 
#include <grace/amr/bc_kernels.tpp>
#include <grace/coordinates/coordinates.hh>
#include <grace/system/grace_system.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/prolongation.hh>
#include <grace/utils/limiters.hh>
#include <grace/data_structures/macros.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/config/config_parser.hh>

#include <spdlog/stopwatch.h>

namespace grace { namespace amr {

void apply_boundary_conditions() {
    auto& vars = variable_list::get().getstate() ;
    apply_boundary_conditions(vars)              ; 
}

void apply_boundary_conditions(grace::var_array_t<GRACE_NSPACEDIM>& vars) {
    Kokkos::Profiling::pushRegion("BC") ; 
    using namespace grace ;
    /******************************************************/
    /* First step:                                        */
    /* Asynchronous data exchange for quadrants in the    */
    /* halo.                                              */
    /******************************************************/
    spdlog::stopwatch sw ; 
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = get_quadrant_extents() ;
    int64_t ngz = get_n_ghosts() ;
    int64_t nq  = get_local_num_quadrants()  ;  
    size_t nvars = variables::get_n_evolved() ;
    /******************************************************/
    auto& qcoords = variable_list::get().getcoords() ;
    auto& vols   = variable_list::get().getvolumes() ; 
    auto& halo = variable_list::get().gethalo()      ;
    /******************************************************/
    auto& params = config_parser::get() ;  
    /******************************************************/
    /*                Create ghost layer                  */
    /******************************************************/
    p4est_ghost_t * halos = p4est_ghost_new( 
          forest::get().get() 
        , P4EST_CONNECT_FULL // CHANGED 
    ) ; 
    sc_array_view_t<p4est_quadrant_t> 
          halo_quads{ &(halos->ghosts) }
        , mirror_quads{ &(halos->mirrors) }  ;
    Kokkos::realloc(halo, VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nvars,halo_quads.size());  
    cell_vol_array_t<GRACE_NSPACEDIM> halo_vols("halo cell volumes", VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),halo_quads.size()) ; 
    scalar_array_t<GRACE_NSPACEDIM> halo_coords("halo quadrant coordinates",GRACE_NSPACEDIM, halo_quads.size() ); 
    /******************************************************/
    /*                Receive halo data                   */
    /******************************************************/
    GRACE_VERBOSE( "Shipping halo quadrants with {}" 
                     " total quadrants and {} halo quadrants.", nq, halo_quads.size()) ; 
    size_t send_size_coords = GRACE_NSPACEDIM ; 
    size_t send_size_vol = EXPR((nx+2*ngz), *(ny+2*ngz), *(nz+2*ngz)) ; 
    size_t send_size = send_size_vol * nvars ; 
    parallel::grace_transfer_context_t context ;
    size_t rank = parallel::mpi_comm_rank() ; 
    for(int iproc=0; iproc<parallel::mpi_comm_size(); ++iproc){
        size_t first_halo  = halos->proc_offsets[iproc]   ; 
        size_t last_halo   = halos->proc_offsets[iproc+1] ;
        for( int ihalo=first_halo; ihalo<last_halo; ++ihalo ) {
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
                , parallel::GRACE_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
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
                , parallel::GRACE_HALO_EXCHANGE_TAG+1
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
            #if 0 
            /* Receive quadrant coordinates */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto hcsview = Kokkos::subview(
                  halo_coords
                , Kokkos::ALL()
                , ihalo 
            ) ; 
            parallel::mpi_irecv(
                  hcsview.data()
                , send_size_coords 
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG+2
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
            #endif 
        }
    }
    /******************************************************/
    /*                Send halo data                      */
    /******************************************************/
    for( int iproc=0; iproc<parallel::mpi_comm_size(); ++iproc){
        size_t first_mirror = halos->mirror_proc_offsets[iproc]   ; 
        size_t last_mirror  = halos->mirror_proc_offsets[iproc+1] ; 
        for( int imirror=first_mirror; imirror<last_mirror; ++imirror){
            size_t iq_loc = 
                (mirror_quads[halos->mirror_proc_mirrors[imirror]]).p.piggy3.local_num ;
            /* Send variables */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto sview = Kokkos::subview(
                  vars
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , iq_loc ) ; 
            parallel::mpi_isend(
                  sview.data()
                , send_size
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
            /* Send cell volumes */
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto svview = Kokkos::subview(
                  vols
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , iq_loc ) ;
            parallel::mpi_isend(
                  svview.data()
                , send_size_vol
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG+1
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
            /* Send quadrant coordinates */
            #if 0
            context._requests.push_back(sc_MPI_Request{}) ; 
            auto scview = Kokkos::subview(
                  qcoords
                , Kokkos::ALL()
                , iq_loc ) ;
            parallel::mpi_isend(
                  scview.data()
                , send_size_coords
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG+2
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ; 
            #endif 
        }
    } 
    /******************************************************/
    /* Second step:                                       */
    /* Iterate over all quadrant faces and store face     */
    /* information.                                       */
    /******************************************************/
    spdlog::stopwatch sw1 ; 
    grace_neighbor_info_t neighbor_info{} ;
    p4est_iterate(
          forest::get().get()
        , halos 
        , reinterpret_cast<void*>( &neighbor_info )//! TODO neighbors_info is a grace_neighbor_info_t object  
        , nullptr
        , grace_iterate_faces 
        #ifdef GRACE_3D 
        , grace_iterate_edges 
        #endif 
        , grace_iterate_corners ) ;
    GRACE_TRACE("Iter faces took {} s.", sw1) ; 
    GRACE_VERBOSE("After iter-faces: obtained\n" 
          " {:d} simple faces of which " 
          " {:d} cross processor boundaries,\n"
          " {:d} hanging faces of which " 
          " {:d} cross processor boundaries,\n"
          " {:d} faces on a physical boundary.\n"
          "Second ghost exchange will send/receive {:d}" 
          "/{:d} coarse quadrants."
        , neighbor_info.face_info.simple_interior_info.size()
        , neighbor_info.face_info.n_simple_ghost_faces
        , neighbor_info.face_info.hanging_interior_info.size() 
        , neighbor_info.face_info.n_hanging_ghost_faces 
        , neighbor_info.face_info.phys_boundary_info.size()
        , neighbor_info.face_info.coarse_hanging_quads_info.snd_quadid.size()  
        , neighbor_info.face_info.coarse_hanging_quads_info.rcv_quadid.size() ) ; 
    GRACE_VERBOSE( "Applying physical boundary conditions on " 
    " {:d} quadrants.", neighbor_info.face_info.phys_boundary_info.size() ) ; 
    
    /******************************************************/
    /* Third step:                                        */
    /* Copy and prolongate/restrict face data from        */
    /* internal boundaries.                               */
    /******************************************************/
    std::string interp = params["amr"]["prolongation_interpolator_type"].as<std::string>(); 
    std::string limiter = params["amr"]["prolongation_limiter_type"].as<std::string>();
    /******************************************************/
    /*                       Copy                         */
    /******************************************************/
    parallel::mpi_waitall(context) ;
    GRACE_VERBOSE( "Copying interior ghostzones across simple boundaries on " 
    " {:d} quadrants.", neighbor_info.face_info.simple_interior_info.size() ) ;
    auto simple_interior_face_info = neighbor_info.face_info.simple_interior_info ;
    auto simple_interior_corner_info = neighbor_info.corner_info.simple_interior_info ;
    #ifdef GRACE_3D 
    auto simple_interior_edge_info = neighbor_info.edge_info.simple_interior_info ;
    #endif 
    simple_interior_face_info.host_to_device() ;
    simple_interior_corner_info.host_to_device() ; 
    #ifdef GRACE_3D 
    simple_interior_edge_info.host_to_device() ;
    #endif 
    copy_interior_ghostzones( vars,halo
                            , simple_interior_face_info
                            , simple_interior_corner_info
                            #ifdef GRACE_3D 
                            , simple_interior_edge_info
                            #endif 
                            ) ; 
    /******************************************************/
    /*       Restrict and prolongate hanging faces        */
    /******************************************************/
    GRACE_VERBOSE( "Restricting and prolongating data on "
    "interior ghostzones across hanging boundaries on " 
    "{:d} quadrants.", neighbor_info.face_info.hanging_interior_info.size() ) ;
    auto hanging_interior_face_info = neighbor_info.face_info.hanging_interior_info ; 
    auto hanging_interior_corner_info = neighbor_info.corner_info.hanging_interior_info ; 
    #ifdef GRACE_3D 
    auto hanging_interior_edge_info = neighbor_info.edge_info.hanging_interior_info ; 
    #endif 
    hanging_interior_face_info.host_to_device() ;
    hanging_interior_corner_info.host_to_device() ;
    #ifdef GRACE_3D 
    hanging_interior_edge_info.host_to_device() ; 
    #endif 
    /******************************************************/
    /*       1) Restriction                               */
    /******************************************************/
    restrict_hanging_ghostzones(
              vars 
            , halo 
            , vols 
            , halo_vols
            , hanging_interior_face_info) ;
    Kokkos::fence() ; 
    /******************************************************/
    /*       2) Exchange coarse quadrants again           */
    /******************************************************/
    auto coarse_hanging_info = neighbor_info.face_info.coarse_hanging_quads_info ; 
    context.reset() ; 
    for(int ircv=0; ircv<coarse_hanging_info.rcv_quadid.size(); ++ircv){
        int ihalo = coarse_hanging_info.rcv_quadid[ircv] ; 
        int iproc = coarse_hanging_info.rcv_procid[ircv] ; 
        GRACE_VERBOSE("Receive iproc {}", iproc);
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
            , parallel::GRACE_HALO_EXCHANGE_TAG
            , parallel::get_comm_world()
            , &(context._requests.back())
        ) ; 
    }
    for( int isend=0; isend<coarse_hanging_info.snd_quadid.size(); ++isend){
        /* Send variables */
        int64_t iq_loc =  coarse_hanging_info.snd_quadid[isend] ; 
        auto sview = Kokkos::subview( 
                          vars
                        , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                        , Kokkos::ALL()
                        , iq_loc ) ; 
        for( auto const& iproc: coarse_hanging_info.snd_procid[isend] ) {
            GRACE_VERBOSE("Send iproc {}", iproc);
            context._requests.push_back(sc_MPI_Request{}) ; 
            parallel::mpi_isend(
                  sview.data()
                , send_size
                , iproc
                , parallel::GRACE_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._requests.back())
            ) ;
        }         
    }
    /******************************************************/
    /*       3) Prolongation                              */
    /******************************************************/
    parallel::mpi_waitall(context) ;
    GRACE_VERBOSE("Initiating prolongation on {} quadrants.", hanging_interior_face_info.size());
    if( interp == "linear" ){
        if( limiter == "minmod" ) {
            prolongate_hanging_ghostzones<utils::linear_prolongator_t<grace::minmod>>(
                      vars 
                    , halo 
                    , vols 
                    , halo_vols
                    , hanging_interior_face_info) ; 
        } else if ( limiter == "monotonized-central") {
            prolongate_hanging_ghostzones<utils::linear_prolongator_t<grace::MCbeta>>(
                      vars 
                    , halo 
                    , vols 
                    , halo_vols
                    , hanging_interior_face_info) ;
        } else {
            ERROR("Unsupported limiter in ghost-zone exchange.") ; 
        }
    } else {
        ERROR("Unsupported interpolator in ghost-zone exchange.") ; 
    }
    Kokkos::fence() ; 
    /******************************************************/
    /* Fourth step:                                       */
    /* Apply physical boundary conditions.                */
    /******************************************************/
    auto phys_boundary_info = neighbor_info.face_info.phys_boundary_info ; 
    phys_boundary_info.host_to_device() ; 
    for(int ivar=0; ivar<nvars; ++ivar){
        auto bc_type = variables::get_bc_type(ivar) ; 
        if( bc_type == "outgoing" )
        {
            auto var = Kokkos::subview( vars
                                      , VEC( Kokkos::ALL() 
                                           , Kokkos::ALL() 
                                           , Kokkos::ALL() )
                                      , ivar 
                                      , Kokkos::ALL() ) ; 
            apply_phys_bc<outgoing_bc_t>(
                  var
                , phys_boundary_info
            ) ; 
        } else if (bc_type == "none" ) {
            /* Nothing to do here */
        } else {
            ERROR("Unrecognized bc type for variable " << ivar << ".\n") ;
        }
    }
    /******************************************************/
    /* De-allocate halo quadrant data                     */
    /******************************************************/
    Kokkos::realloc(halo, VEC(0,0,0), 0,0);
    parallel::mpi_barrier() ; 
    GRACE_TRACE("All done in BC. Total number of cells processed: {}.\n"
                  "Total time elapsed {} s.\n"\
                , EXPR((nx+2*ngz), *(ny+2*ngz), *(nz+2*ngz)) * nq * nvars 
                , sw ) ; 
    Kokkos::Profiling::popRegion() ; 
}

}} /* namespace grace::amr */
