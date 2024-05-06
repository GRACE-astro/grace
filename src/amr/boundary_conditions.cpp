/**
 * @file boundary_conditions.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-21
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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

#include <Kokkos_Core.hpp>

#include <thunder/amr/thunder_amr.hh>
#include <thunder/amr/bc_helpers.hh> 
#include <thunder/amr/bc_helpers.tpp> 
#include <thunder/amr/bc_kernels.tpp>
#include <thunder/coordinates/coordinates.hh>
#include <thunder/system/thunder_system.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/utils/prolongation.hh>
#include <thunder/utils/limiters.hh>
#include <thunder/data_structures/macros.hh>
#include <thunder/data_structures/memory_defaults.hh>
#include <thunder/data_structures/variable_indices.hh>
#include <thunder/data_structures/variable_properties.hh>
#include <thunder/data_structures/variables.hh>
#include <thunder/data_structures/variable_utils.hh>
#include <thunder/config/config_parser.hh>

namespace thunder { namespace amr {

void apply_boundary_conditions() {
    using namespace thunder ;
    /******************************************************/
    /* First step:                                        */
    /* Asynchronous data exchange for quadrants in the    */
    /* halo.                                              */
    /******************************************************/
    size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = get_quadrant_extents() ;
    int64_t ngz = get_n_ghosts() ;
    int64_t nq  = get_local_num_quadrants()  ;  
    size_t nvars = variables::get_n_evolved() ;
    /******************************************************/
    auto& vars = variable_list::get().getstate()     ;
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
        , P4EST_CONNECT_FACE  
    ) ; 
    sc_array_view_t<p4est_quadrant_t> 
          halo_quads{ &(halos->ghosts) }
        , mirror_quads{ &(halos->mirrors) }  ;
    Kokkos::realloc(halo, VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nvars,halo_quads.size());  
    cell_vol_array_t<THUNDER_NSPACEDIM> halo_vols("halo cell volumes", VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),halo_quads.size()) ; 
    scalar_array_t<THUNDER_NSPACEDIM> halo_coords("halo quadrant coordinates",THUNDER_NSPACEDIM, halo_quads.size() ); 
    /******************************************************/
    /*                Receive halo data                   */
    /******************************************************/
    THUNDER_INFO(VERBOSE, "AMR-BC", "Shipping halo quadrants with " 
        << nq << " total quadrants and " 
        <<  halo_quads.size() << " halo quadrants.") ; 
    std::cout << "About to ship " << halo_quads.size() << " halo quadrants.\n" ; 
    size_t send_size_coords = THUNDER_NSPACEDIM ; 
    size_t send_size_vol = EXPR(nx, *ny, *nz) ; 
    size_t send_size = send_size_vol * nvars ; 
    parallel::thunder_transfer_context_t context ;
    size_t rank = parallel::mpi_comm_rank() ; 
    for(int iproc=0; iproc<parallel::mpi_comm_size(); ++iproc){
        size_t first_halo  = halos->proc_offsets[iproc]   ; 
        size_t last_halo   = halos->proc_offsets[iproc+1] ;
        for( int ihalo=first_halo; ihalo<last_halo; ++ihalo ) {
            /* Receive variables */
            context._rcv_rq.push_back(sc_MPI_Request{}) ; 
            auto hsview = Kokkos::subview(
                  halo
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                  hsview.data()
                , send_size
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._rcv_rq.back())
            ) ; 
            /* Receive cell volumes */
            context._rcv_rq.push_back(sc_MPI_Request{}) ; 
            auto hvsview = Kokkos::subview(
                  halo_vols
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , ihalo 
            ) ; 
            parallel::mpi_irecv(
                  hvsview.data()
                , send_size_vol 
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG+1
                , parallel::get_comm_world()
                , &(context._rcv_rq.back())
            ) ; 
            /* Receive quadrant coordinates */
            context._rcv_rq.push_back(sc_MPI_Request{}) ; 
            auto hcsview = Kokkos::subview(
                  halo_coords
                , Kokkos::ALL()
                , ihalo 
            ) ; 
            parallel::mpi_irecv(
                  hcsview.data()
                , send_size_coords 
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG+2
                , parallel::get_comm_world()
                , &(context._rcv_rq.back())
            ) ; 
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
            context._snd_rq.push_back(sc_MPI_Request{}) ; 
            auto sview = Kokkos::subview(
                  vars
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                ,Kokkos::ALL()
                , iq_loc ) ; 
            parallel::mpi_isend(
                  sview.data()
                , send_size
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._snd_rq.back())
            ) ; 
            /* Send cell volumes */
            context._snd_rq.push_back(sc_MPI_Request{}) ; 
            auto svview = Kokkos::subview(
                  vols
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , iq_loc ) ;
            parallel::mpi_isend(
                  svview.data()
                , send_size_vol
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG+1
                , parallel::get_comm_world()
                , &(context._snd_rq.back())
            ) ; 
            /* Send quadrant coordinates */
            context._snd_rq.push_back(sc_MPI_Request{}) ; 
            auto scview = Kokkos::subview(
                  qcoords
                , Kokkos::ALL()
                , iq_loc ) ;
            parallel::mpi_isend(
                  svview.data()
                , send_size_coords
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG+2
                , parallel::get_comm_world()
                , &(context._snd_rq.back())
            ) ; 
        }
    } 
    /******************************************************/
    /* Second step:                                       */
    /* Iterate over all quadrant faces and store face     */
    /* information.                                       */
    /******************************************************/
    parallel::mpi_waitall(context) ;
    thunder_face_info_t face_info{} ;
    p4est_iterate(
          forest::get().get()
        , halos 
        , reinterpret_cast<void*>( &face_info )
        , nullptr
        , thunder_iterate_faces 
        #ifdef THUNDER_3D 
        , nullptr 
        #endif 
        , nullptr) ;
    THUNDER_INFO(VERBOSE,"AMR-BC", "After iter-faces: obtained\n" 
        << face_info.simple_interior_info.size() << " simple faces of which " 
        << face_info.n_simple_ghost_faces << " cross processor boundaries,\n"
        << face_info.hanging_interior_info.size() << " hanging faces of which " 
        << face_info.n_hanging_ghost_faces << " cross processor boundaries,\n"
        << face_info.phys_boundary_info.size() << " faces on a physical boundary.\n"
        << "Second ghost exchange will send/receive " << face_info.coarse_hanging_quads_info.snd_quadid.size() 
        << "/" << face_info.coarse_hanging_quads_info.rcv_quadid.size() << " coarse quadrants."  ) ; 
    THUNDER_INFO(VERBOSE, "AMR-BC", "Applying physical boundary conditions on " 
    << face_info.phys_boundary_info.size() << " quadrants." ) ; 
    /******************************************************/
    /* Third step:                                        */
    /* Apply physical boundary conditions.                */
    /******************************************************/
    auto phys_boundary_info = face_info.phys_boundary_info ; 
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
    /* Physical boundary conditions can also be applied   */
    /* to auxiliary variables (e.g. primitives).          */
    /******************************************************/
    size_t nvars_aux = variables::get_n_auxiliary() ;
    auto& aux = variable_list::get().getaux()       ; 
    for(int ivar=0; ivar<nvars_aux; ++ivar){
        auto bc_type = variables::get_bc_type(ivar, thunder::variables::AUXILIARY) ; 
        if( bc_type == "outgoing" )
        {
            auto var = Kokkos::subview( aux
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
    /* Fourth step:                                       */
    /* Copy and prolongate/restrict face data from        */
    /* internal boundaries.                               */
    /******************************************************/
    std::string interp = params["amr"]["prolongation_interpolator_type"].as<std::string>(); 
    std::string limiter = params["amr"]["prolongation_limiter_type"].as<std::string>();
    //std::cout << "In halo exchange: got " << simple_interior_info.size() << " simple interior faces " << std::endl ; 
    /******************************************************/
    /*                       Copy                         */
    /******************************************************/
    THUNDER_INFO(VERBOSE, "AMR-BC", "Copying interior ghostzones across simple boundaries on " 
    << face_info.simple_interior_info.size() << " quadrants." ) ;
    auto simple_interior_info = face_info.simple_interior_info ;
    simple_interior_info.host_to_device() ;
    copy_interior_ghostzones(vars,halo,simple_interior_info) ; 
    /******************************************************/
    /*       Restrict and prolongate hanging faces        */
    /******************************************************/
    THUNDER_INFO(VERBOSE, "AMR-BC", "Restricting and prolongating data on "
    "interior ghostzones across hanging boundaries on " 
    << face_info.hanging_interior_info.size() << " quadrants." ) ;
    auto hanging_interior_info = face_info.hanging_interior_info ; 
    hanging_interior_info.host_to_device() ;
    /******************************************************/
    /*       1) Restriction                               */
    /******************************************************/
    restrict_hanging_ghostzones(
              vars 
            , halo 
            , vols 
            , halo_vols
            , hanging_interior_info) ;
    Kokkos::fence() ; 
    /******************************************************/
    /*       2) Exchange coarse quadrants again           */
    /******************************************************/
    auto coarse_hanging_info = face_info.coarse_hanging_quads_info ; 
    context.reset() ; 
    for(int iproc=0; iproc<parallel::mpi_comm_size(); ++iproc){
        size_t first_halo  = halos->proc_offsets[iproc]   ; 
        size_t last_halo   = halos->proc_offsets[iproc+1] ;
        for( int ihalo=first_halo; ihalo<last_halo; ++ihalo ) {
            /* Receive variables */
            if(  std::find(coarse_hanging_info.rcv_quadid.begin(), coarse_hanging_info.rcv_quadid.end(), ihalo) == coarse_hanging_info.rcv_quadid.end() ) {
                continue ; 
            }
            context._rcv_rq.push_back(sc_MPI_Request{}) ; 
            auto hsview = Kokkos::subview(
                  halo
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , ihalo) ; 
            parallel::mpi_irecv(
                  hsview.data()
                , send_size
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._rcv_rq.back())
            ) ; 
        }
    }
    for( int iproc=0; iproc<parallel::mpi_comm_size(); ++iproc){
        size_t first_mirror = halos->mirror_proc_offsets[iproc]   ; 
        size_t last_mirror  = halos->mirror_proc_offsets[iproc+1] ; 
        for( int imirror=first_mirror; imirror<last_mirror; ++imirror){
            size_t iq_loc = 
                (mirror_quads[halos->mirror_proc_mirrors[imirror]]).p.piggy3.local_num ;
            if( std::find(coarse_hanging_info.snd_quadid.begin(), coarse_hanging_info.snd_quadid.end(), iq_loc) == coarse_hanging_info.snd_quadid.end() ) {
                continue ; 
            }
            /* Send variables */
            context._snd_rq.push_back(sc_MPI_Request{}) ; 
            auto sview = Kokkos::subview(
                  vars
                , VEC(Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL())
                , Kokkos::ALL()
                , iq_loc ) ; 
            parallel::mpi_isend(
                  sview.data()
                , send_size
                , iproc
                , parallel::THUNDER_HALO_EXCHANGE_TAG
                , parallel::get_comm_world()
                , &(context._snd_rq.back())
            ) ; 
        }
    }
    /******************************************************/
    /*       3) Prolongation                              */
    /******************************************************/
    parallel::mpi_waitall(context) ;
    if( interp == "linear" ){
        if( limiter == "minmod" ) {
            prolongate_hanging_ghostzones<utils::linear_prolongator_t<thunder::minmod>>(
                      vars 
                    , halo 
                    , qcoords 
                    , halo_coords 
                    , vols 
                    , halo_vols
                    , hanging_interior_info) ; 
        } else if ( limiter == "monotonized-central") {
            prolongate_hanging_ghostzones<utils::linear_prolongator_t<thunder::MCbeta>>(
                      vars 
                    , halo 
                    , qcoords 
                    , halo_coords 
                    , vols 
                    , halo_vols
                    , hanging_interior_info) ;
        } else {
            ERROR("Unsupported limiter in ghost-zone exchange.") ; 
        }
    } else {
        ERROR("Unsupported interpolator in ghost-zone exchange.") ; 
    }
    /******************************************************/
    /* Transform vector and tensor components             */
    /* across tree boundaries (where applicable)          */
    /******************************************************/
    
    /******************************************************/
    /* De-allocate halo quadrant data                     */
    /******************************************************/
    Kokkos::realloc(halo, VEC(0,0,0), 0,0);
}

}} /* namespace thunder::amr */