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
#include <grace/amr/bc_iterate.hh> 
#include <grace/amr/bc_copy_ghostzones.hh> 
#include <grace/amr/bc_restrict_ghostzones.hh> 
#include <grace/amr/bc_prolongate_ghostzones.hh> 
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
    auto& staggered_vars = variable_list::get().getstaggeredstate() ; 
    apply_boundary_conditions(vars, staggered_vars) ; 
}

void apply_boundary_conditions( grace::var_array_t<GRACE_NSPACEDIM>& vars
                              , grace::staggered_variable_arrays_t& staggered_vars) {
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
    size_t nvars_face = variables::get_n_evolved_face_staggered() ; 
    size_t nvars_edge = variables::get_n_evolved_edge_staggered() ; 
    size_t nvars_corner = variables::get_n_evolved_corner_staggered() ; 
    /******************************************************/
    auto& qcoords = variable_list::get().getcoords() ;
    auto& vols   = variable_list::get().getvolumes() ; 
    auto& halo = variable_list::get().gethalo()      ;
    staggered_variable_arrays_t staggered_halo{}     ; 
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
    staggered_halo.realloc(VEC(nx,ny,nz),ngz,halo_quads.size(),nvars_face,nvars_edge,nvars_corner) ; 
    /******************************************************/
    /*     Allocate space to hold remote data on GPU      */
    /******************************************************/
    cell_vol_array_t<GRACE_NSPACEDIM> halo_vols(
        "halo cell volumes", 
        VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),
        halo_quads.size()
    ) ; 
    scalar_array_t<GRACE_NSPACEDIM> halo_coords(
        "halo quadrant coordinates",
        GRACE_NSPACEDIM, halo_quads.size() 
    ) ; 
    /******************************************************/
    /*                Halo transfer                       */
    /******************************************************/
    GRACE_VERBOSE( "Shipping halo quadrants with {}" 
                     " total quadrants and {} halo quadrants.", nq, halo_quads.size()) ; 
    parallel::grace_transfer_context_t context{} ; 
    grace_init_halo_transfer(
        context, 
        halos, 
        halo_quads, 
        mirror_quads, 
        halo, 
        staggered_halo,
        halo_vols,
        vars,
        staggered_vars,
        vols,
        #ifdef GRACE_CARTESIAN_COORDINATES
        false
        #else
        true
        #endif
    ) ; 
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
        , neighbor_info.coarse_hanging_quads_info.snd_quadid.size()  
        , neighbor_info.coarse_hanging_quads_info.rcv_quadid.size() ) ; 
    GRACE_VERBOSE("After iter-edges: obtained\n"
        "{:d} edges in total of which "
        "{:d} edges facing the exterior\n"
        "{:d} simple edge pairs of which "
        "{:d} cross processor boundaries\n"
        "{:d} hanging edges of which "
        "{:d} cross processor boundaries."
        , neighbor_info.edge_info.n_edges_total
        , neighbor_info.edge_info.n_exterior_edges
        , neighbor_info.edge_info.simple_interior_info.size()
        , neighbor_info.edge_info.n_simple_ghost_edges
        , neighbor_info.edge_info.hanging_interior_info.size() 
        , neighbor_info.edge_info.n_hanging_ghost_edges) ; 
    GRACE_VERBOSE("After iter-corners: obtained\n"
        "{:d} simple corner pairs of which "
        "{:d} cross processor boundaries\n"
        "{:d} hanging corners of which "
        "{:d} cross processor boundaries."
        , neighbor_info.corner_info.simple_interior_info.size()
        , neighbor_info.corner_info.n_simple_ghost_corners
        , neighbor_info.corner_info.hanging_interior_info.size() 
        , neighbor_info.corner_info.n_hanging_ghost_corners) ; 

    GRACE_VERBOSE( "Applying physical boundary conditions on " 
    " {:d} quadrants.", neighbor_info.face_info.phys_boundary_info.size() ) ; 
    
    /******************************************************/
    /* Third step:                                        */
    /* Copy and data from internal boundaries.            */
    /******************************************************/
    std::string interp = params["amr"]["prolongation_interpolator_type"].as<std::string>(); 
    std::string limiter = params["amr"]["prolongation_limiter_type"].as<std::string>();
    /******************************************************/
    /*                       Copy                         */
    /******************************************************/
    grace_finalize_halo_transfer(context) ; 
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
    copy_interior_ghostzones( vars, halo
                            , staggered_vars 
                            , staggered_halo
                            , simple_interior_face_info
                            , simple_interior_corner_info
                            #ifdef GRACE_3D 
                            , simple_interior_edge_info
                            #endif 
                            ) ; 
    /******************************************************/
    /* Fourth step:                                       */
    /* Restrict data on internal hanging boundaries.      */
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
    /*          Restriction                               */
    /******************************************************/
    restrict_hanging_ghostzones(
              vars 
            , halo 
            , staggered_vars 
            , stagered_halo
            , vols 
            , halo_vols
            , hanging_interior_face_info
            , hanging_interior_corner_info 
            #ifdef GRACE_3D 
            , hanging_interior_edge_info
            #endif 
            ) ;
    /******************************************************/
    /* Fifth step:                                        */
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
    Kokkos::fence() ; 
    /******************************************************/
    /* Sixth step:                                        */
    /* Exchange coarse quadrants again                    */
    /******************************************************/
    auto coarse_hanging_info = neighbor_info.coarse_hanging_quads_info ; 
    context.reset() ; 
    grace_init_halo_transfer_custom(
        context, 
        coarse_hanging_info.snd_quadid, 
        coarse_hanging_info.rcv_quadid, 
        coarse_hanging_info.snd_procid, 
        coarse_hanging_info.rcv_procid, 
        halo,
        staggered_halo,
        halo_vols,
        vars,
        staggered_vars,
        vols,
        false
    );
    grace_finalize_halo_transfer(context) ; 
    /******************************************************/
    /* Seventh step:                                      */
    /* Exchange coarse quadrants again                    */
    /******************************************************/
    GRACE_VERBOSE("Initiating prolongation on {} quadrants.", hanging_interior_face_info.size()) ;
    prolongate_hanging_ghostzones(
              vars 
            , halo
            , staggered_vars
            , staggered_halo 
            , vols 
            , halo_vols
            , hanging_interior_face_info
            , hanging_interior_corner_info 
            #ifdef GRACE_3D 
            , hanging_interior_edge_info
            #endif 
            ) ; 
    
    Kokkos::fence() ;
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
