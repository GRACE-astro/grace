/**
 * @file amr_ghosts.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Index fiesta.
 * @date 2025-09-05
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

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_utils.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/assert.hh>

#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <grace/amr/amr_ghosts.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/ghostzone_kernels/copy_kernels.hh>
#include <grace/amr/ghostzone_kernels/phys_bc_kernels.hh>
#include <grace/amr/ghostzone_kernels/pack_unpack_kernels.hh>
#include <grace/amr/ghostzone_kernels/task_factories.hh>


#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/profiling/profiling.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <numeric>
#include <unordered_set>

//#define INSERT_FENCE_DEBUG_TASKS_ 

namespace grace {

/***************************************************/
// BC Task tree (deps from top to bottom)
//
// |--> MPI Recv 
// |-----> Unpack 
// |-----> Unpack Cbufs 
// |-------> Prolongate 
//
// |--> Pack
// |----> MPI Send 


void amr_ghosts_impl_t::update() {
    GRACE_PROFILING_PUSH_REGION("GHOST_UPDATE") ; 
    // empty everything first 
    reset() ; 

    auto nvar = variables::get_n_evolved() ; 
    var_bc_kind = Kokkos::View<bc_t*>{
        "BC_types", static_cast<size_t>(nvar) 
    } ;
    auto var_bc_kind_h = Kokkos::create_mirror_view(var_bc_kind) ; 
    for(int ivar=0; ivar<nvar; ++ivar){
        auto bc_type = variables::get_bc_type(ivar) ; 
        bc_t kind ; 
        if (bc_type=="outgoing") {
            kind = bc_t::BC_OUTFLOW ; 
        } else if ( bc_type == "lagrange_extrap") {
            kind = bc_t::BC_LAGRANGE_EXTRAP ; 
        } else if ( bc_type == "none") {
            kind = bc_t::BC_NONE ; 
        }
        var_bc_kind_h(ivar) = kind ;
    }
    Kokkos::deep_copy(var_bc_kind,var_bc_kind_h) ; 

    // Destroy old ghost layer if present
    if (p4est_ghost_layer) {
        p4est_ghost_destroy(p4est_ghost_layer);
    }

    // Rebuild ghost layer from scratch
    p4est_ghost_layer = p4est_ghost_new(grace::amr::forest::get().get(), P4EST_CONNECT_FULL);

    // Clear and re-alloc the ghost_layer to the correct size
    auto nq = amr::get_local_num_quadrants() ; 
    ghost_layer.clear() ; 
    ghost_layer.resize(nq) ; 

    // Register neighbor faces into ghost_layer
    p4est_iterate(grace::amr::forest::get().get(),      /*forest*/
                  p4est_ghost_layer,                    /*ghost layer*/
                  static_cast<void*>(&ghost_layer),     /*user data*/
                  nullptr,                              /*volume*/
                  &grace_iterate_faces,                 /*face*/
                  #ifdef GRACE_3D 
                  &grace_iterate_edges,                 /*edge*/
                  #endif 
                  &grace_iterate_corners );             /*corner*/
    parallel::mpi_barrier() ; // FIXME 

    std::unordered_set<size_t> cbuf_qid ; 
    //build_flux_buffers()     ;
    build_coarse_buffers(cbuf_qid)   ; 
     
    bucket_t phys_bc_kernels, copy_kernels, copy_to_cbuf_kernels, prolong_kernels; 
    hang_bucket_t copy_from_cbuf_kernels ;
    std::vector<bucket_t> pack_kernels, unpack_kernels
                        , pack_to_cbuf_kernels  
                        , unpack_to_cbuf_kernels ;
    std::vector<hang_bucket_t>  unpack_from_cbuf_kernels ; 

    build_remote_buffers(
        phys_bc_kernels, copy_kernels,
        copy_from_cbuf_kernels, copy_to_cbuf_kernels, 
        pack_kernels, unpack_kernels, pack_to_cbuf_kernels,
        unpack_to_cbuf_kernels, unpack_from_cbuf_kernels, prolong_kernels
    )   ; 
    parallel::mpi_barrier() ; // FIXME 
     
    build_task_list(
        phys_bc_kernels, copy_kernels,
        copy_from_cbuf_kernels, copy_to_cbuf_kernels, 
        pack_kernels, unpack_kernels, pack_to_cbuf_kernels,
        unpack_to_cbuf_kernels, unpack_from_cbuf_kernels,
        prolong_kernels, cbuf_qid
    )        ; 
    build_executor_runtime() ;
     
    GRACE_PROFILING_POP_REGION ; 
}

void amr_ghosts_impl_t::build_coarse_buffers(
    std::unordered_set<size_t> & cbuf_qid
) {
    /****************************************************/
    auto nq = amr::get_local_num_quadrants() ; 
    std::size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts() ; 
    // get n vars 
    std::size_t nvars = variables::get_n_evolved() ; 
    /****************************************************/

    auto needs_cbuf = [&] (quad_neighbors_descriptor_t const& desc) {
        for( int f=0; f<P4EST_FACES; ++f) {
            if (desc.faces[f].level_diff == level_diff_t::COARSER ) 
                return true ;
        } 
        for( int e=0; e<12; ++e) {
            if (desc.edges[e].level_diff == level_diff_t::COARSER ) 
                return true ;
        } 
        for( int c=0; c<P4EST_CHILDREN; ++c) {
            if (desc.corners[c].level_diff == level_diff_t::COARSER ) 
                return true ;
        } 
        return false ; 
    } ; 

    /****************************************************/
    size_t cur_idx{0UL} ; 
    for( size_t iq=0UL; iq<nq; iq+=1UL ) {
        if ( needs_cbuf(ghost_layer[iq]) ) {
            ghost_layer[iq].cbuf_id = cur_idx ++ ; 
            cbuf_qid.insert(iq) ; 
        } 
    }   
    /****************************************************/
    _coarse_buffers = var_array_t<GRACE_NSPACEDIM>(
        "coarse_buffers", VEC(nx/2+2*ngz, ny/2+2*ngz, nz/2+2*ngz), nvars, cur_idx 
    ) ; 
    /****************************************************/
}

void amr_ghosts_impl_t::build_executor_runtime() {
    task_queue.clear() ; 
    task_queue.reserve(task_list.size()) ; 

    for( auto& t: task_list) {
        
        runtime_task_view rtv ; 
        rtv.t = t.get() ; 
        ASSERT( rtv.t != nullptr, "Dangling pointer! ") ; 
        rtv.pending = t->_dependencies.size() ; 
        if ( rtv.pending == 0 ) {
            t -> status = status_id_t::READY ;
            task_queue.ready.push_back(t -> task_id) ;  
        }

        task_queue.rt.push_back(std::move(rtv)) ; 
    }
    GRACE_TRACE("Task queue constructed. {} tasks are ready to run.", task_queue.ready.size()) ; 
}

void amr_ghosts_impl_t::build_task_list(
    bucket_t& phys_bc_kernels,
    bucket_t& copy_kernels,
    hang_bucket_t& copy_from_cbuf_kernels,
    bucket_t& copy_to_cbuf_kernels,
    std::vector<bucket_t>& pack_kernels, 
    std::vector<bucket_t>& unpack_kernels, 
    std::vector<bucket_t>& pack_to_cbuf_kernels,
    std::vector<bucket_t>& unpack_to_cbuf_kernels,
    std::vector<hang_bucket_t>& unpack_from_cbuf_kernels,
    bucket_t& prolong_kernels,
    std::unordered_set<size_t> const& cbuf_qid
) {
    /***********************************************************************/
    task_id_t task_counter{0UL} ; 
    /***********************************************************************/
    // Get variables 
    auto& vars = grace::variable_list::get() ; 
    auto& state = vars.getstate() ; 
    /***********************************************************************/
    // MPI info 
    auto rank = parallel::mpi_comm_rank() ; 
    auto nproc= parallel::mpi_comm_size() ;
    /***********************************************************************/
    // Grid info 
    auto nq = amr::get_local_num_quadrants() ; 
    std::size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts() ; 
    std::size_t nvars = variables::get_n_evolved() ;
    /***********************************************************************/
    // First we construct the mpi tasks 
    std::vector<task_id_t> send_task_id, recv_task_id ; 
    send_task_id.resize(nproc) ; recv_task_id.resize(nproc) ; 
    for( size_t r=0UL; r<nproc; r+=1UL) {
        if( send_rank_sizes[r] > 0 ){
            task_list.push_back(
                std::make_unique<mpi_task_t>(
                    make_mpi_send_task(r, _send_buffer, send_rank_offsets, send_rank_sizes, task_counter)
                )
            ) ; 
            send_task_id[r] = task_list.back()->task_id ; 
        }
        if (recv_rank_sizes[r] > 0 ){
            task_list.push_back(
                std::make_unique<mpi_task_t>(
                    make_mpi_recv_task(r, _recv_buffer, recv_rank_offsets, recv_rank_sizes, task_counter)
                ) 
            ) ; 
            recv_task_id[r] = task_list.back()->task_id ; 
        }   
    }
    /***********************************************************************/
    // now we set up the kernels 
    /***********************************************************************/
    // First decide on streams 
    auto& stream_pool = device_stream_pool::get();
    auto& copy_stream = stream_pool.next() ; 
    auto& pup_stream = stream_pool.next() ; 
    auto& phys_bc_stream = stream_pool.next() ; 
    auto& interp_stream = stream_pool.next() ; 
    /***********************************************************************/
    /***********************************************************************/
    task_id_t restrict_tid{UNSET_TASK_ID}; 
    if(cbuf_qid.size()>0) {
        restrict_tid = insert_restriction_tasks(
            cbuf_qid,
            ghost_layer,
            state, 
            _coarse_buffers, 
            interp_stream,
            VEC(nx,ny,nz), ngz, nvars, 
            task_counter, task_list 
        ) ; 
    }
    /***********************************************************************/
    /***********************************************************************/
    GRACE_TRACE("Counter {} Size {}", task_counter, task_list.size()) ; 
    insert_copy_tasks(
        ghost_layer,
        copy_kernels,
        copy_from_cbuf_kernels,
        copy_to_cbuf_kernels,
        state,
        _coarse_buffers,
        copy_stream,
        VEC(nx,ny,nz), ngz, nvars, 
        task_counter,restrict_tid, task_list 
    ) ; 
    /***********************************************************************/
    /***********************************************************************/
    GRACE_TRACE("Counter {} Size {}", task_counter, task_list.size()) ; 
    insert_pup_tasks(
        ghost_layer,
        pack_kernels,
        unpack_kernels,
        pack_to_cbuf_kernels,
        unpack_to_cbuf_kernels,
        unpack_from_cbuf_kernels,
        state, _coarse_buffers,
        _send_buffer,
        _recv_buffer,
        send_task_id,
        recv_task_id,
        restrict_tid,
        pup_stream,
        VEC(nx,ny,nz), ngz, nvars,
        task_counter, task_list
    ) ; 
    /***********************************************************************/
    /***********************************************************************/
    GRACE_TRACE("Inserting gz-restriction tasks.") ; 
    GRACE_TRACE("Counter {} Size {}", task_counter, task_list.size()) ; 
    insert_ghost_restriction_tasks(
        cbuf_qid, ghost_layer,
        state, _coarse_buffers,
        interp_stream,
        VEC(nx,ny,nz),ngz,nvars,
        task_counter, task_list
    ) ; 
    /***********************************************************************/
    /***********************************************************************/
    #if 1
    GRACE_TRACE("Inserting phys-bc tasks.") ; 
    GRACE_TRACE("Counter {} Size {}", task_counter, task_list.size()) ; 
    auto const deferred_phys_bc_kernels = 
        insert_phys_bc_tasks(
                phys_bc_kernels, ghost_layer,
                state, _coarse_buffers, var_bc_kind,
                phys_bc_stream, VEC(nx,ny,nz),ngz,nvars,
                task_counter,task_list
        ) ;
    /***********************************************************************/
    /***********************************************************************/
    GRACE_TRACE("Inserting prolongation tasks.") ; 
    GRACE_TRACE("Counter {} Size {}", task_counter, task_list.size()) ; 
    insert_prolongation_tasks(
        prolong_kernels, ghost_layer,
        state, _coarse_buffers, interp_stream,
        VEC(nx,ny,nz), ngz, nvars, task_counter, task_list 
    ) ; 
    /***********************************************************************/
    /***********************************************************************/
    #if 1
    GRACE_TRACE("Inserting deferred phys-bc tasks.") ; 
    insert_deferred_phys_bc_tasks(
        deferred_phys_bc_kernels, ghost_layer,
        state, _coarse_buffers, var_bc_kind, phys_bc_stream,
        VEC(nx,ny,nz),ngz,nvars, task_counter, task_list
    ); 
    #endif 
    /***********************************************************************/
    /***********************************************************************/
}

} /* namespace grace */
