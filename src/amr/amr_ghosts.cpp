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
#include <grace/amr/ghostzone_kernels/pr_helpers.hh>
#include <grace/amr/ghostzone_kernels/pr_ho_coeffs.hh>


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
    GRACE_VERBOSE("Building task list for ghostzone exchange.") ; 
    // empty everything first 
    reset() ; 

    auto nvar = variables::get_n_evolved() ; 

    // collect bc types of all variables 
    high_order_interp_varlist.clear() ; 
    low_order_interp_varlist.clear() ; 
    Kokkos::realloc(var_bc_kind, static_cast<size_t>(nvar)) ; 
    auto var_bc_kind_h = Kokkos::create_mirror_view(var_bc_kind) ; 
    for(int ivar=0; ivar<nvar; ++ivar){
        var_bc_kind_h(ivar) = variables::get_bc_type(ivar) ; 
        auto interp_kind = variables::get_interp_type(ivar) ; 
        if ( interp_kind ==  var_amr_interp_t::INTERP_SECOND_ORDER) {
            low_order_interp_varlist.push_back(ivar) ; 
        } else if (  interp_kind ==  var_amr_interp_t::INTERP_FOURTH_ORDER ) {
            high_order_interp_varlist.push_back(ivar) ; 
        } else {
            ERROR("Unrecognised prolongation/restriction operator requested for var " << ivar) ; 
        }
    }
    Kokkos::deep_copy(var_bc_kind,var_bc_kind_h) ; 

    auto nvar_f = variables::get_n_evolved_face_staggered() ; 
    Kokkos::realloc(var_bc_kind_f,static_cast<size_t>(nvar_f)); 
    auto var_bc_kind_f_h = Kokkos::create_mirror_view(var_bc_kind_f) ; 
    for(int ivar=0; ivar<nvar_f; ++ivar){
        var_bc_kind_f_h(ivar) = variables::get_bc_type(ivar,STAG_FACEX) ;
    }
    Kokkos::deep_copy(var_bc_kind_f,var_bc_kind_f_h) ;

    // collect parity factors for reflection symmetries
    Kokkos::realloc(var_reflect_parity, static_cast<size_t>(nvar) ) ; 
    auto var_reflect_parity_h = Kokkos::create_mirror_view(var_reflect_parity) ; 
    for(int ivar=0; ivar<nvar; ++ivar){
        var_reflect_parity_h(ivar,0) = var_reflect_parity_h(ivar,1) = var_reflect_parity_h(ivar,2) = 1.0;

        auto props = variables::detail::_varprops[variables::detail::_varnames[ivar]] ; 
        if ( props.is_vector ) {
            var_reflect_parity_h(ivar,props.comp_num) = -1.0 ; 
        } else if ( props.is_tensor ) {
            if (props.comp_num==1) {
                var_reflect_parity_h(ivar,0) = var_reflect_parity_h(ivar,1) = -1. ;
            } else if (props.comp_num==2) {
                var_reflect_parity_h(ivar,0) = var_reflect_parity_h(ivar,2) = -1. ;
            } else if (props.comp_num==4) {
                var_reflect_parity_h(ivar,1) = var_reflect_parity_h(ivar,2) = -1. ;
            }
        }
    }
    deep_copy(var_reflect_parity,var_reflect_parity_h) ; 

    // initialize weights for 5th order restrict/prolong 
    grace::detail::fill_fifth_order_prolongation_coefficients(ho_prolong_coefficients) ; 
    grace::detail::fill_fifth_order_restriction_coefficients(ho_restrict_coefficients) ; 

    // Rebuild ghost layer from scratch
    p4est_ghost_layer = p4est_ghost_new(grace::amr::forest::get().get(), P4EST_CONNECT_FULL);

    // Clear and re-alloc the ghost_layer to the correct size
    auto nq = amr::get_local_num_quadrants() ; 
    ghost_layer.clear() ; 
    ghost_layer.resize(nq) ; 

    p4est_iter_data_t iter_data {
        &ghost_layer,
        &_reflux_face_descs,
        &_reflux_coarse_face_descs,
        &_reflux_edge_descs,
        &_reflux_coarse_edge_descs
    } ; 
    //**************************************************************************************************
    // Register neighbor faces into ghost_layer
    p4est_iterate(grace::amr::forest::get().get(),      /*forest*/
                  p4est_ghost_layer,                    /*ghost layer*/
                  static_cast<void*>(&iter_data),     /*user data*/
                  nullptr,                              /*volume*/
                  &grace_iterate_faces,                 /*face*/
                  #ifdef GRACE_3D 
                  &grace_iterate_edges,                 /*edge*/
                  #endif 
                  &grace_iterate_corners );             /*corner*/
    //**************************************************************************************************
    //**************************************************************************************************
    std::unordered_set<size_t> cbuf_qid ; 
    build_coarse_buffers(cbuf_qid)   ; 
    GRACE_TRACE("Constructed coarse buffers, we have {} quadrants", cbuf_qid.size()) ; 
    //**************************************************************************************************
    build_remote_buffers()   ; 
    //**************************************************************************************************
    task_id_t task_counter{0UL} ; 
    build_task_list<STAG_CENTER>(
        var_bc_kind, cbuf_qid, task_counter 
    ) ;
    build_task_list_face_stag(
        var_bc_kind_f, cbuf_qid, task_counter
    ) ; 
    build_executor_runtime() ;
    parallel::mpi_barrier() ;
    build_reflux_buffers() ; 
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
    std::size_t nvars_f = variables::get_n_evolved_face_staggered() ; 
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
    _coarse_buffers = var_array_t(
        "coarse_buffers", VEC(nx/2+2*ngz, ny/2+2*ngz, nz/2+2*ngz), nvars, cur_idx 
    ) ; 
    _stag_coarse_buffers.realloc(
        VEC(nx/2, ny/2, nz/2), ngz, cur_idx, nvars_f, 0, 0 
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
    GRACE_VERBOSE("Task queue constructed. {} tasks are ready to run.", task_queue.ready.size()) ; 
}

template< grace::var_staggering_t stag >
void amr_ghosts_impl_t::build_task_list(
    Kokkos::View<bc_t*>& vbc,
    std::unordered_set<size_t> const& cbuf_qid,
    task_id_t& task_counter 
) {
    /***********************************************************************/
    grace::var_array_t dummy("placeholder", VEC(0,0,0),0,0) ; 
    /***********************************************************************/
    auto& cbuf_view = get_coarse_buffers<stag>() ; 
    /***********************************************************************/
    // Get variables 
    auto& vars = grace::variable_list::get() ; 
    auto& state = vars.getstate() ;
    auto& stag_state = vars.getstaggeredstate() ;  
    /***********************************************************************/
    // MPI info 
    auto rank = parallel::mpi_comm_rank() ; 
    auto nproc= parallel::mpi_comm_size() ;
    /***********************************************************************/
    // Grid info 
    auto ngz = amr::get_n_ghosts() ; 
    auto nq = amr::get_local_num_quadrants() ; 
    std::size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ;
    auto nv =  variables::get_n_evolved() ;
    /***********************************************************************/
    // First we construct the mpi tasks 
    std::vector<task_id_t> send_task_id, recv_task_id ; 
    send_task_id.resize(nproc) ; recv_task_id.resize(nproc) ; 
    for( size_t r=0UL; r<nproc; r+=1UL) {
        if( send_rank_sizes[stag][r] > 0 ){
            task_list.push_back(
                std::make_unique<mpi_task_t>(
                    make_mpi_send_task(r, _send_buffer[stag], send_rank_offsets[stag], send_rank_sizes[stag], task_counter, parallel::GRACE_HALO_EXCHANGE_TAG_CC)
                )
            ) ; 
            send_task_id[r] = task_list.back()->task_id ; 
        }
        if (recv_rank_sizes[stag][r] > 0 ){
            task_list.push_back(
                std::make_unique<mpi_task_t>(
                    make_mpi_recv_task(r, _recv_buffer[stag], recv_rank_offsets[stag], recv_rank_sizes[stag], task_counter, parallel::GRACE_HALO_EXCHANGE_TAG_CC)
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
    auto& copy_stream    = stream_pool.next() ; 
    auto& pup_stream     = stream_pool.next() ; 
    auto& phys_bc_stream = stream_pool.next() ; 
    auto& interp_stream  = stream_pool.next() ; 
    /***********************************************************************/
    /***********************************************************************/
    task_id_t restrict_tid{UNSET_TASK_ID}; 

    if(cbuf_qid.size()>0) {
        restrict_tid = insert_restriction_tasks<stag>(
            cbuf_qid,
            ghost_layer,
            dummy, 
            cbuf_view, 
            low_order_interp_varlist,
            high_order_interp_varlist,
            ho_restrict_coefficients,
            interp_stream,
            VEC(nx,ny,nz), ngz, nv, 
            task_counter, task_list 
        ) ; 
    }
    /***********************************************************************/
    /***********************************************************************/
    insert_copy_tasks<stag>(
        ghost_layer,
        cbuf_qid,
        copy_kernels,
        cbuf_p2p_copy_kernels,
        copy_from_cbuf_kernels,
        copy_to_cbuf_kernels,
        dummy,
        cbuf_view,
        copy_stream,
        VEC(nx,ny,nz), ngz, nv, 
        task_counter,restrict_tid, task_list 
    ) ; 
    /***********************************************************************/
    /***********************************************************************/
    insert_pup_tasks<stag>(
        ghost_layer,
        pack_kernels,
        cbuf_p2p_pack_kernels,
        unpack_kernels,
        cbuf_p2p_unpack_kernels,
        pack_finer_kernels,
        pack_to_cbuf_kernels,
        unpack_to_cbuf_kernels,
        unpack_from_cbuf_kernels,
        dummy, cbuf_view,
        _send_buffer[stag],
        _recv_buffer[stag],
        send_task_id,
        recv_task_id,
        restrict_tid,
        pup_stream,
        VEC(nx,ny,nz), ngz, nv,
        task_counter, task_list
    ) ; 

    /***********************************************************************/
    /***********************************************************************/
    #if 0
    insert_ghost_restriction_tasks<stag>(
        cbuf_qid, ghost_layer,
        dummy, cbuf_view,
        interp_stream,
        VEC(nx,ny,nz),ngz,nv,
        task_counter, task_list
    ) ; 
    #endif 
    tag_bcs_in_cbuf<stag>(
        cbuf_qid, ghost_layer
    ) ; 
    /***********************************************************************/
    /***********************************************************************/
    auto const deferred_phys_bc_kernels = 
        insert_phys_bc_tasks<stag>(
                phys_bc_kernels, ghost_layer,
                dummy, cbuf_view, vbc, var_reflect_parity,
                phys_bc_stream, VEC(nx,ny,nz),ngz,nv,
                restrict_tid, task_counter,task_list
        ) ;
    /***********************************************************************/
    /***********************************************************************/
    insert_prolongation_tasks<stag>(
        prolong_kernels, ghost_layer,
        low_order_interp_varlist, high_order_interp_varlist, ho_prolong_coefficients,
        dummy, cbuf_view, interp_stream, 
        VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
    ) ;
   
    /***********************************************************************/
    /***********************************************************************/
    insert_deferred_phys_bc_tasks<stag>(
        deferred_phys_bc_kernels, ghost_layer,
        dummy, cbuf_view, vbc, var_reflect_parity, phys_bc_stream, 
        VEC(nx,ny,nz),ngz,nv, task_counter, task_list
    ); 
    /***********************************************************************/
    /***********************************************************************/
}

void amr_ghosts_impl_t::build_task_list_face_stag(
    Kokkos::View<bc_t*>& vbc,
    std::unordered_set<size_t> const& cbuf_qid,
    task_id_t& task_counter 
) {
    /***********************************************************************/
    grace::var_array_t dummy("placeholder", VEC(0,0,0),0,0) ; 
    /***********************************************************************/
    /***********************************************************************/
    // Get variables 
    auto& vars = grace::variable_list::get() ; 
    /***********************************************************************/
    // MPI info 
    auto rank = parallel::mpi_comm_rank() ; 
    auto nproc= parallel::mpi_comm_size() ;
    /***********************************************************************/
    // Grid info 
    auto ngz = amr::get_n_ghosts() ; 
    auto nq = amr::get_local_num_quadrants() ; 
    std::size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ;
    auto nv = variables::get_n_evolved_face_staggered()  ;
    /***********************************************************************/
    auto& stream_pool = device_stream_pool::get(); 
    auto& stream = stream_pool.next() ; 
    /***********************************************************************/
    std::array<bucket_t,N_VAR_STAGGERINGS> deferred_phys_bc_kernels; 
    /***********************************************************************/
    #define INSERT_TASKS_UP_TO_PROLONG_IMPL(stag,tag) \
    do {\
    auto& cbuf_view = get_coarse_buffers<stag>() ; \
    std::vector<task_id_t> send_task_id, recv_task_id ; \
    send_task_id.resize(nproc) ; recv_task_id.resize(nproc) ; \
    for( size_t r=0UL; r<nproc; r+=1UL) { \
        if( send_rank_sizes[stag][r] > 0 ){ \
            task_list.push_back( \
                std::make_unique<mpi_task_t>( \
                    make_mpi_send_task(r, _send_buffer[stag], send_rank_offsets[stag], send_rank_sizes[stag], task_counter, tag) \
                ) \
            ) ; \
            send_task_id[r] = task_list.back()->task_id ; \
        } \
        if (recv_rank_sizes[stag][r] > 0 ){ \
            task_list.push_back( \
                std::make_unique<mpi_task_t>( \
                    make_mpi_recv_task(r, _recv_buffer[stag], recv_rank_offsets[stag], recv_rank_sizes[stag], task_counter, tag) \
                ) \
            ) ; \
            recv_task_id[r] = task_list.back()->task_id ; \
        }   \
    } \
    task_id_t restrict_tid{UNSET_TASK_ID};  \
    if(cbuf_qid.size()>0) { \
        restrict_tid = insert_div_preserving_restriction_tasks<stag>( \
            cbuf_qid, ghost_layer, \
            dummy, cbuf_view, \
            stream, \
            VEC(nx,ny,nz),ngz,nv, \
            task_counter, task_list \
        ) ; \
    } \
    insert_copy_tasks<stag>(\
        ghost_layer, \
        cbuf_qid, \
        copy_kernels, \
        cbuf_p2p_copy_kernels, \
        copy_from_cbuf_kernels, \
        copy_to_cbuf_kernels, \
        dummy, \
        cbuf_view, \
        stream, \
        VEC(nx,ny,nz), ngz, nv, \
        task_counter,restrict_tid, task_list \
    ) ;  \
    insert_pup_tasks<stag>( \
        ghost_layer, \
        pack_kernels, \
        cbuf_p2p_pack_kernels,\
        unpack_kernels, \
        cbuf_p2p_unpack_kernels,\
        pack_finer_kernels, \
        pack_to_cbuf_kernels, \
        unpack_to_cbuf_kernels, \
        unpack_from_cbuf_kernels, \
        dummy, cbuf_view, \
        _send_buffer[stag], \
        _recv_buffer[stag], \
        send_task_id, \
        recv_task_id, \
        restrict_tid, \
        stream, \
        VEC(nx,ny,nz), ngz, nv, \
        task_counter, task_list \
    ) ; \
    deferred_phys_bc_kernels[stag] = \
        insert_phys_bc_tasks<stag>( \
                phys_bc_kernels, ghost_layer,\
                dummy, cbuf_view, vbc, var_reflect_parity,\
                stream, VEC(nx,ny,nz),ngz,nv,\
                restrict_tid,task_counter,task_list\
        ) ;\
    } while (false)
    /***********************************************************************/
    /***********************************************************************/
    INSERT_TASKS_UP_TO_PROLONG_IMPL(STAG_FACEX,parallel::GRACE_HALO_EXCHANGE_TAG_FX);
    INSERT_TASKS_UP_TO_PROLONG_IMPL(STAG_FACEY,parallel::GRACE_HALO_EXCHANGE_TAG_FY);
    INSERT_TASKS_UP_TO_PROLONG_IMPL(STAG_FACEZ,parallel::GRACE_HALO_EXCHANGE_TAG_FZ);
    /***********************************************************************/
    /***********************************************************************/
    insert_div_preserving_prolongation_tasks(
        prolong_kernels, ghost_layer,
        dummy,_stag_coarse_buffers,
        stream, VEC(nx,ny,nz), ngz, nv,
        task_counter, task_list 
    ) ; 
    /***********************************************************************/
    /***********************************************************************/
    insert_deferred_phys_bc_tasks<STAG_FACEX>(
        deferred_phys_bc_kernels[STAG_FACEX], ghost_layer,
        dummy, get_coarse_buffers<STAG_FACEX>(), vbc, var_reflect_parity, stream,
        VEC(nx,ny,nz),ngz,nv, task_counter, task_list
    ); 
    insert_deferred_phys_bc_tasks<STAG_FACEY>(
        deferred_phys_bc_kernels[STAG_FACEY], ghost_layer,
        dummy, get_coarse_buffers<STAG_FACEY>(), vbc, var_reflect_parity, stream,
        VEC(nx,ny,nz),ngz,nv, task_counter, task_list
    ); 
    insert_deferred_phys_bc_tasks<STAG_FACEZ>(
        deferred_phys_bc_kernels[STAG_FACEZ], ghost_layer,
        dummy, get_coarse_buffers<STAG_FACEZ>(), vbc, var_reflect_parity, stream,
        VEC(nx,ny,nz),ngz,nv, task_counter, task_list
    ); 
    /***********************************************************************/
    /***********************************************************************/
}
#if 0
void amr_ghosts_impl_t::build_reflux_buffers() {
    // goals: figure out how much data needs to be 
    // send and received from / to whom in reflux.
    // Allocate buffers

    // the vectors _reflux_face_descs and _reflux_edge_descs
    // were filled during iterate 
    DECLARE_GRID_EXTENTS ; 
    // first: fluxes through faces 
    // each recorded face has a hanging side. 
    // If both are local we append to a list
    // If coarse is remote we record a send 
    // If fine is remote we record a receive 
    /****************************************************/
    // get mpi info
    auto rank = parallel::mpi_comm_rank() ; 
    auto nproc= parallel::mpi_comm_size() ;

    auto nvars_hrsc = variables::get_n_hrsc() ; 
    // sends and receives must be uniquely ordered 
    struct comm_key_t {
        size_t qid ;
        int8_t elem_id ; // face or edge 
        bool operator==(const comm_key_t & other) const {
        return (qid == other.qid) && 
               (elem_id == other.elem_id);
        }
    } ; 

    struct key_cmp {
        bool operator() (
            comm_key_t const& a,
            comm_key_t const& b
        ) const
        {
            if ( a.qid < b.qid ) return true ; 
            if ( b.qid < a.qid ) return false ; 
            return (a.elem_id < b.elem_id) ; 
        }
    };

    struct comm_key_hash {
        std::size_t operator()(comm_key_t const& k) const noexcept {
            std::size_t h1 = std::hash<size_t>{}(k.qid);
            std::size_t h2 = std::hash<int8_t>{}(k.elem_id);

            // Combine hashes (boost-like method)
            std::size_t seed = h1;
            seed ^= h2 + 0x9e3779b97f4a7c15ULL + (seed<<6) + (seed>>2);
            return seed;
        }
    };

    struct comm_patt_builder {
        int nproc ;

        auto sort_and_dedup(
            std::vector<std::vector<comm_key_t>>& keys 
        ) const 
        {
            std::vector<std::unordered_map<comm_key_t, size_t, comm_key_hash>> lookup(nproc);
            std::vector<size_t> counts(nproc, 0);
            for (int r = 0; r < nproc; ++r) {
                auto& vec = keys[r];
                std::sort(vec.begin(), vec.end(), key_cmp{});
                vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
                counts[r] = vec.size();
                
                std::unordered_map<comm_key_t, size_t, comm_key_hash> index;
                index.reserve(vec.size());
                for (size_t k = 0; k < vec.size(); ++k)
                    index[vec[k]] = k;
                lookup[r] = std::move(index);
            }
            return std::make_pair(lookup, counts);
        }

        auto get_offsets_and_sizes(
            std::vector<size_t> const& counts,
            size_t size 
        ) const 
        {
            std::vector<size_t> sizes(nproc,0), offsets(nproc,0) ; 
            size_t total = 0;
            for (size_t r = 0; r < nproc; ++r) {
                size_t _loc_size = counts[r] * size ; 
                total+=_loc_size ; 
                sizes[r] = _loc_size ;
            }
            for( size_t r=1; r<nproc; ++r) {
                offsets[r] = offsets[r-1] + sizes[r-1];
            }
            return std::make_tuple(total,sizes,offsets) ; 
        }
    } ; 

    comm_patt_builder cpb{nproc} ; 

    std::vector<std::vector<comm_key_t>> snd_keys(nproc), rcv_keys(nproc) ; 
    // loop over 
    for( int i=0; i<_reflux_face_descs.size(); ++i) {
        auto const& dsc =  _reflux_face_descs[i] ; 
        if ( dsc.coarse_is_remote ) { 
            // if coarse is remote we just need a send 
            for( int ic=0; ic<P4EST_CHILDREN/2; ++ic) {
                if ( dsc.fine_is_remote[ic]) continue ; // both remote 
                // select face id from the smaller of the two ranks, to keep ordering consistent 
                int8_t elem_id = dsc.coarse_owner_rank < rank ? dsc.coarse_face_id : dsc.fine_face_id ; 
                snd_keys[dsc.coarse_owner_rank].push_back(
                    comm_key_t{
                        dsc.fine_qid[ic], elem_id
                    }
                ) ; 
            }
        } else {
            // if coarse is local it will need correction
            for( int ic=0; ic<P4EST_CHILDREN/2; ++ic) {
                if ( !dsc.fine_is_remote[ic]) continue ; // both local, nothing to report
                int8_t elem_id = dsc.fine_owner_rank[ic] < rank ? dsc.fine_face_id : dsc.coarse_face_id ; 
                rcv_keys[dsc.fine_owner_rank[ic]].push_back(
                    comm_key_t{
                        dsc.fine_qid[ic], elem_id
                    }
                ); 
            }
        }
    }
    // sort keys
    std::vector<std::unordered_map<comm_key_t, size_t, comm_key_hash>> send_lookup, recv_lookup ; 
    std::vector<size_t> rank_send_counts, rank_recv_counts ; 
    std::tie(send_lookup,rank_send_counts) = cpb.sort_and_dedup(snd_keys) ;
    std::tie(recv_lookup,rank_recv_counts) = cpb.sort_and_dedup(rcv_keys) ;

    // -- 
    std::vector<hanging_face_reflux_desc_t> local_interfaces ; 
    _reflux_face_snd.clear() ;  

    for (int i = 0; i < _reflux_face_descs.size(); ++i) {
        auto &dsc = _reflux_face_descs[i];
        if (dsc.coarse_is_remote) {
            GRACE_TRACE("Coarse remote") ; 
            // coarse remote → we send to coarse owner
            for (int ic = 0; ic < P4EST_CHILDREN/2; ++ic) {
                if (dsc.fine_is_remote[ic]) {
                    continue;
                }
                hanging_remote_reflux_desc_t snd_desc{} ; 
                int r = dsc.coarse_owner_rank;
                int8_t elem = (dsc.coarse_owner_rank < rank ?
                            dsc.coarse_face_id : dsc.fine_face_id);

                comm_key_t key{ dsc.fine_qid[ic], elem };
                snd_desc.qid = dsc.fine_qid[ic] ; 
                snd_desc.rank = r ; 
                snd_desc.elem_id =  dsc.fine_face_id ; // note 
                snd_desc.buf_id = send_lookup[r][key];
                _reflux_face_snd.push_back(snd_desc) ;
            }
        }
        else {
            // coarse local → receiver side
            for (int ic = 0; ic < P4EST_CHILDREN/2; ++ic) {
                if (!dsc.fine_is_remote[ic]) {
                    continue;
                }
                GRACE_TRACE("Fine remote") ; 
                int r = dsc.fine_owner_rank[ic];
                int8_t elem = (dsc.fine_owner_rank[ic] < rank ?
                            dsc.fine_face_id : dsc.coarse_face_id);

                comm_key_t key{ dsc.fine_qid[ic], elem };
                // replace fine qid with the buf id 
                dsc.fine_qid[ic] = recv_lookup[r][key];
            }
            // since coarse is local we need 
            // to keep track of this
            local_interfaces.push_back( dsc ); 
        }
    }
    _reflux_face_descs.swap(local_interfaces) ; // note, we discard the total array here and only keep the locals
    // we need to dedup 
    // note: at this stage order does not matter, all the necessary info is 
    // stored inside the remote descriptors 
    std::sort(_reflux_face_snd.begin(), _reflux_face_snd.end(), 
        [](const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) {
            if (a.rank < b.rank) return true;
            if (b.rank < a.rank) return false;
            if (a.buf_id < b.buf_id) return true;
            if (b.buf_id < a.buf_id) return false;
            if (a.qid < b.qid) return true;
            if (b.qid < a.qid) return false;
            return (a.elem_id < b.elem_id);
        }
    );

    // Then remove consecutive duplicates
    _reflux_face_snd.erase(
        std::unique(_reflux_face_snd.begin(), _reflux_face_snd.end(),
            [](const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) {
                return a.rank == b.rank && a.buf_id == b.buf_id && 
                    a.qid == b.qid && a.elem_id == b.elem_id;
            }
        ),
        _reflux_face_snd.end()
    );
    GRACE_VERBOSE("We have {} local faces which need refluxing", _reflux_face_descs.size() ) ; 
    
    // allocate buffers 
    // we know counts, now we need per-rank sizes and per-rank offsets 
    // first fluxes: size of each is n/2 x n/2 x nvars 
    size_t send_size_flux = (nx/2)*(nx/2) * nvars_hrsc ; 
    // then emfs: size of each is (n/2+1) x (n/2+1) x 2 (the 2 because two edge dirs per face)
    size_t send_size_emf = (nx/2)*(nx/2) * 2  ;

    size_t total_snd_flux, total_rcv_flux, total_snd_emf, total_rcv_emf ; 
    std::tie(total_snd_flux,_reflux_snd_size,_reflux_snd_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_flux) ; 
    std::tie(total_rcv_flux,_reflux_rcv_size,_reflux_rcv_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_flux) ; 

    std::tie(total_snd_emf,_reflux_snd_emf_size,_reflux_snd_emf_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_emf) ; 
    std::tie(total_rcv_emf,_reflux_rcv_emf_size,_reflux_rcv_emf_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_emf) ;

    // Construct the actual buffers
    _reflux_snd_buf = amr::reflux_array_t("reflux_flux_send") ; 
    _reflux_recv_buf = amr::reflux_array_t("reflux_flux_receive") ;
    _reflux_emf_snd_buf = amr::reflux_array_t("reflux_emf_send") ; 
    _reflux_emf_recv_buf = amr::reflux_array_t("reflux_emf_receive") ; 

    _reflux_snd_buf.set_strides(nx/2, nvars_hrsc) ; 
    _reflux_recv_buf.set_strides(nx/2, nvars_hrsc) ; 
    _reflux_emf_snd_buf.set_strides(nx/2, 2) ; 
    _reflux_emf_recv_buf.set_strides(nx/2, 2) ; 

    _reflux_snd_buf.set_offsets(_reflux_snd_off)          ; 
    _reflux_recv_buf.set_offsets(_reflux_rcv_off)         ; 
    _reflux_emf_snd_buf.set_offsets(_reflux_snd_emf_off)  ; 
    _reflux_emf_recv_buf.set_offsets(_reflux_rcv_emf_off) ; 

    _reflux_snd_buf.realloc(total_snd_flux)     ; 
    _reflux_recv_buf.realloc(total_rcv_flux)    ;
    _reflux_emf_snd_buf.realloc(total_snd_emf)  ; 
    _reflux_emf_recv_buf.realloc(total_rcv_emf) ; 

    GRACE_VERBOSE("[REFLUX]: Face buffers constructed, total size send flux: {} emf: {}, total size receive flux: {} emf: {}", total_snd_flux, total_snd_emf, total_rcv_flux, total_rcv_emf) ;
    // Detailed per-rank breakdown
    for( int r=0; r<nproc; ++r) {
        if (_reflux_snd_size[r] > 0 || _reflux_rcv_size[r] > 0) {
            GRACE_VERBOSE("[REFLUX]: Fluxes: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_off[r], _reflux_snd_size[r],
                _reflux_rcv_off[r], _reflux_rcv_size[r]);
            GRACE_VERBOSE("[REFLUX]: EMFs face: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_emf_off[r], _reflux_snd_emf_size[r],
                _reflux_rcv_emf_off[r], _reflux_rcv_emf_size[r]);
        }
    }
    // now: coarse faces 
    snd_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    rcv_keys = std::vector<std::vector<comm_key_t>>(nproc) ;
    // loop over coarse
    for( int i=0; i<_reflux_coarse_face_descs.size(); ++i) {
        auto const& dsc =  _reflux_coarse_face_descs[i] ; 
        for( int is=0; is<2; ++is ) {
            if ( dsc.is_remote[is] ) {
                // Note they can't both be remote 
                // or we wouldn't be here
                ASSERT(!dsc.is_remote[1-is], "Both sides of a face remote!") ; 
                // send and receive 
                int8_t elem_id = dsc.owner_rank[is] < rank ? dsc.face_id[is] : dsc.face_id[1-is] ;
                // we send the other qid 
                snd_keys[dsc.owner_rank[is]].push_back(
                    comm_key_t(dsc.qid[1-is], elem_id)
                ) ; 
                // we receive this qid 
                rcv_keys[dsc.owner_rank[is]].push_back(
                    comm_key_t(dsc.qid[is], elem_id)
                ) ; 
            }
        }
    }
    // sort and deduplicate the send and receive keys 
    std::tie(send_lookup,rank_send_counts) = cpb.sort_and_dedup(snd_keys) ;
    std::tie(recv_lookup,rank_recv_counts) = cpb.sort_and_dedup(rcv_keys) ;

    _reflux_coarse_face_snd.clear() ; 
    // loop over coarse
    for( int i=0; i<_reflux_coarse_face_descs.size(); ++i) {
        auto& dsc =  _reflux_coarse_face_descs[i] ; 
        for( int is=0; is<2; ++is ) {
            if ( dsc.is_remote[is] ) {
                auto r = dsc.owner_rank[is] ; 
                // send and receive 
                int8_t elem_id = r < rank ? dsc.face_id[is] : dsc.face_id[1-is] ;

                // send: we need to register into the send list 
                comm_key_t snd_key{
                    dsc.qid[1-is], elem_id
                } ; 
                hanging_remote_reflux_desc_t snd_desc{} ;
                snd_desc.qid = dsc.qid[1-is] ;
                snd_desc.rank = r ; 
                snd_desc.elem_id =  dsc.face_id[1-is] ;
                snd_desc.buf_id = send_lookup[r][snd_key];
                _reflux_coarse_face_snd.push_back(snd_desc) ;

                // receive: we write back the bufid into the desc
                comm_key_t rcv_key{
                    dsc.qid[is], elem_id
                } ; 
                dsc.qid[is] = recv_lookup[r][rcv_key] ; 
            }
        }
    }
    #if 1
    std::sort(_reflux_coarse_face_snd.begin(), _reflux_coarse_face_snd.end(), 
        [](const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) {
            if (a.rank < b.rank) return true;
            if (b.rank < a.rank) return false;
            if (a.buf_id < b.buf_id) return true;
            if (b.buf_id < a.buf_id) return false;
            if (a.qid < b.qid) return true;
            if (b.qid < a.qid) return false;
            return (a.elem_id < b.elem_id);
        }
    );

    // Then remove consecutive duplicates
    _reflux_coarse_face_snd.erase(
        std::unique(_reflux_coarse_face_snd.begin(), _reflux_coarse_face_snd.end(),
            [](const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) {
                return a.rank == b.rank && a.buf_id == b.buf_id && 
                    a.qid == b.qid && a.elem_id == b.elem_id;
            }
        ),
        _reflux_coarse_face_snd.end()
    );
    #endif 
    send_size_emf = nx*nx*2 ; 

    size_t total_snd_emf_coarse, total_recv_emf_coarse ; 
    std::tie(total_snd_emf_coarse,_reflux_snd_emf_coarse_size,_reflux_snd_emf_coarse_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_emf) ; 
    std::tie(total_recv_emf_coarse,_reflux_rcv_emf_coarse_size,_reflux_rcv_emf_coarse_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_emf) ; 

    // allocate buffers 
    _reflux_emf_coarse_snd_buf = amr::reflux_array_t("reflux_emf_coarse_send") ; 
    _reflux_emf_coarse_recv_buf = amr::reflux_array_t("reflux_emf_coarse_receive") ; 

    _reflux_emf_coarse_snd_buf.set_strides(nx, 2) ; 
    _reflux_emf_coarse_recv_buf.set_strides(nx, 2) ;
    _reflux_emf_coarse_snd_buf.set_offsets(_reflux_snd_emf_coarse_off)  ; 
    _reflux_emf_coarse_recv_buf.set_offsets(_reflux_rcv_emf_coarse_off) ; 
    _reflux_emf_coarse_snd_buf.realloc(total_snd_emf_coarse)  ; 
    _reflux_emf_coarse_recv_buf.realloc(total_recv_emf_coarse) ; 

    //GRACE_VERBOSE("Coarse faces: total send {} total receive {}", total_snd_emf_coarse,total_recv_emf_coarse) ; 
    GRACE_VERBOSE("[REFLUX]: Coarse face buffers constructed, total size send: {} receive: {}", total_snd_emf_coarse,total_recv_emf_coarse) ; 
    // Detailed per-rank breakdown
    for( int r=0; r<nproc; ++r) {
        if (_reflux_snd_emf_coarse_size[r] > 0 || _reflux_rcv_emf_coarse_size[r] > 0) {
            GRACE_VERBOSE("[REFLUX]: Coarse faces: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_emf_coarse_off[r], _reflux_snd_emf_coarse_size[r],
                _reflux_rcv_emf_coarse_off[r], _reflux_rcv_emf_coarse_size[r]);
        }
    }
    // now: edges 
    // the convention is that the sender sets the element id 
    snd_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    rcv_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    for( int i=0; i<_reflux_edge_descs.size(); ++i) {
        auto& dsc = _reflux_edge_descs[i] ; 

        // only loop over fine, these are the ones we send and receive 
        for( int iside=0; iside<dsc.n_sides; ++iside) {
            auto const& dsc_this = dsc.sides[iside] ; 
            if ( !dsc_this.is_fine ) continue ; 

            for( int ic=0; ic<2; ++ic) {
                if ( dsc_this.octants.fine.is_remote[ic] ) {
                    // fine remote --> receive 
                    rcv_keys[dsc_this.octants.fine.owner_rank[ic]].push_back(
                        comm_key_t{
                            dsc_this.octants.fine.quad_id[ic], dsc_this.edge_id
                        }
                    ) ;
                } else {
                    // fine local --> send 
                    for( int jside=0; jside<dsc.n_sides; ++jside){ 
                        if ( jside==iside ) continue ; 
                        auto const& dsc_other = dsc.sides[jside] ; 
                        if ( dsc_other.is_fine) { // other is fine
                            for( int icj=0; icj<2; ++icj) {
                                if (!dsc_other.octants.fine.is_remote[icj]) continue ; 
                                // note, in send we use **our** qid
                                snd_keys[dsc_other.octants.fine.owner_rank[icj]].push_back(
                                    comm_key_t{
                                        dsc_this.octants.fine.quad_id[ic], dsc_this.edge_id
                                    }
                                ) ;
                            }
                        } else { // other is coarse
                            if ( !dsc_other.octants.coarse.is_remote) continue ; 
                            snd_keys[dsc_other.octants.coarse.owner_rank].push_back(
                                    comm_key_t{
                                        dsc_this.octants.fine.quad_id[ic], dsc_this.edge_id
                                    }
                                ) ;
                        }
                    }
                }
            }
            
        }
    } // for loop 
    // note: we need to dedup since if multiple remotes are on the same rank 
    // we send the data to each octant, which is redundant. 
    std::tie(send_lookup,rank_send_counts) = cpb.sort_and_dedup(snd_keys) ;
    std::tie(recv_lookup,rank_recv_counts) = cpb.sort_and_dedup(rcv_keys) ;

    // fixme, we are not filling recv here cause I don't think it's necessary, if true come back and remove it altogether
    _reflux_edge_snd.clear() ;     
    // goals of this loop: 
    // 2 record send info 
    // 3 write back recv ids into the original struct 
    for( int i=0; i<_reflux_edge_descs.size(); ++i) {
        auto& dsc = _reflux_edge_descs[i] ; 

        // only loop over fine, these are the ones we send and receive 
        for( int iside=0; iside<dsc.n_sides; ++iside) {
            auto& dsc_this = dsc.sides[iside] ; 
            if (!dsc_this.is_fine) continue ; 

            for( int ic=0; ic<2; ++ic) {
                if ( dsc_this.octants.fine.is_remote[ic] ) {
                    // fine remote --> receive 
                    comm_key_t key {
                            dsc_this.octants.fine.quad_id[ic], dsc_this.edge_id
                        } ; 
                    // fixme, ensure this is never used again 
                    auto r = dsc_this.octants.fine.owner_rank[ic] ; 
                    dsc_this.octants.fine.quad_id[ic] = recv_lookup[r][key];
                } else {
                    // fine local --> send 
                    for( int jside=0; jside<dsc.n_sides; ++jside){ 
                        if ( jside==iside ) continue ; 
                        auto const& dsc_other = dsc.sides[jside] ; 
                        if ( dsc_other.is_fine) { // other is fine
                            for( int icj=0; icj<2; ++icj) {
                                if (!dsc_other.octants.fine.is_remote[icj]) continue ; 
                                // note, in send we use **our** qid
                                comm_key_t key {
                                    dsc_this.octants.fine.quad_id[ic], dsc_this.edge_id
                                } ; 
                                hanging_remote_reflux_desc_t snd_desc{} ; 
                                snd_desc.qid = dsc_this.octants.fine.quad_id[ic] ; 
                                auto r = dsc_other.octants.fine.owner_rank[icj] ;
                                snd_desc.rank = r; 
                                snd_desc.elem_id = dsc_this.edge_id ; 
                                snd_desc.buf_id = send_lookup[r][key] ; 
                                _reflux_edge_snd.push_back(snd_desc) ; 
                            }
                        } else { // other is coarse
                            if ( !dsc_other.octants.coarse.is_remote) continue ; 
                            comm_key_t key {
                                        dsc_this.octants.fine.quad_id[ic], dsc_this.edge_id
                            } ; 
                            hanging_remote_reflux_desc_t snd_desc{} ;
                            snd_desc.qid = dsc_this.octants.fine.quad_id[ic] ; 
                            auto r = dsc_other.octants.coarse.owner_rank ;
                            snd_desc.rank = r; 
                            snd_desc.elem_id = dsc_this.edge_id ; 
                            snd_desc.buf_id = send_lookup[r][key] ;
                            _reflux_edge_snd.push_back(snd_desc) ; 
                        }
                    }
                }
            }

        } // for desc 
    }    
    // we need to dedup _reflux_edge_snd 
    std::sort(_reflux_edge_snd.begin(), _reflux_edge_snd.end(), 
        [](const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) {
            if (a.rank < b.rank) return true;
            if (b.rank < a.rank) return false;
            if (a.buf_id < b.buf_id) return true;
            if (b.buf_id < a.buf_id) return false;
            if (a.qid < b.qid) return true;
            if (b.qid < a.qid) return false;
            return (a.elem_id < b.elem_id);
        }
    );
    _reflux_edge_snd.erase(
        std::unique(_reflux_edge_snd.begin(), _reflux_edge_snd.end(),
            [](const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) {
                return a.rank == b.rank && a.buf_id == b.buf_id && 
                    a.qid == b.qid && a.elem_id == b.elem_id;
            }
        ),
        _reflux_edge_snd.end()
    );

    // finally allocate 
    // here we always send the emf along the edge --> no stagger
    // also we always send just one --> send size simply nx 
    send_size_emf = (nx) ; 

    size_t total_snd_emf_edge, total_rcv_emf_edge;
    std::tie(total_snd_emf_edge,_reflux_snd_emf_edge_size,_reflux_snd_emf_edge_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_emf) ; 
    std::tie(total_rcv_emf_edge,_reflux_rcv_emf_edge_size,_reflux_rcv_emf_edge_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_emf) ; 

    _reflux_emf_edge_snd_buf = amr::reflux_edge_array_t("reflux_emf_edge_send") ; 
    _reflux_emf_edge_recv_buf = amr::reflux_edge_array_t("reflux_emf_edge_receive") ;
    
    _reflux_emf_edge_snd_buf.set_strides(nx) ; 
    _reflux_emf_edge_recv_buf.set_strides(nx) ; 

    _reflux_emf_edge_snd_buf.set_offsets(_reflux_snd_emf_edge_off) ; 
    _reflux_emf_edge_recv_buf.set_offsets(_reflux_rcv_emf_edge_off) ; 

    _reflux_emf_edge_snd_buf.realloc(total_snd_emf_edge) ; 
    _reflux_emf_edge_recv_buf.realloc(total_rcv_emf_edge) ; 


    GRACE_VERBOSE("[REFLUX]: EMF-edge buffers constructed, total size send emf: {}, total size receive: {}", total_snd_emf_edge, total_rcv_emf_edge) ;
    
    GRACE_VERBOSE("Total amount of edges to be corrected {}", _reflux_edge_descs.size()) ; 
    _reflux_emf_edge_accumulation_buf = Kokkos::View<double***, grace::default_space>(
        "reflux_emf_edge_local_buffer", nx, 2, _reflux_edge_descs.size()
    ) ; 

    // Detailed per-rank breakdown
    for( int r=0; r<nproc; ++r) {
        if (_reflux_snd_emf_edge_size[r] > 0 || _reflux_rcv_emf_edge_size[r] > 0) {
            GRACE_VERBOSE("[REFLUX]: EMFs edge: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_emf_edge_off[r], _reflux_snd_emf_edge_size[r],
                _reflux_rcv_emf_edge_off[r], _reflux_rcv_emf_edge_size[r]);
        }
    }

    // now the coarse edge
    snd_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    rcv_keys = std::vector<std::vector<comm_key_t>>(nproc) ; 
    for( int i=0; i<_reflux_coarse_edge_descs.size(); ++i) {
        auto& dsc = _reflux_coarse_edge_descs[i] ;
        
        for( int iside=0; iside<dsc.n_sides; ++iside) {
            auto const& dsc_this = dsc.sides[iside] ; 
            ASSERT(!dsc_this.is_fine, "In coarse edges got fine side.") ; 
            if ( dsc_this.octants.coarse.is_remote ) {
                GRACE_TRACE("Receive coarse, quadid {} rank {} edge {}",dsc_this.octants.coarse.quad_id,dsc_this.octants.coarse.owner_rank,dsc_this.edge_id );
                // receive 
                rcv_keys[dsc_this.octants.coarse.owner_rank].push_back(
                        comm_key_t{
                            dsc_this.octants.coarse.quad_id, dsc_this.edge_id
                        }
                    ) ;
            } else {
                // send 
                for( int jside=0; jside<dsc.n_sides; ++jside){ 
                    if ( jside==iside ) continue ;
                    auto const& dsc_other = dsc.sides[jside] ; 
                    if ( !dsc_other.octants.coarse.is_remote ) continue ; 
                    GRACE_TRACE("Send coarse, quadid {} rank {} edge {}",dsc_this.octants.coarse.quad_id,dsc_other.octants.coarse.owner_rank,dsc_this.edge_id );
                    snd_keys[dsc_other.octants.coarse.owner_rank].push_back(
                                    comm_key_t{
                                        dsc_this.octants.coarse.quad_id, dsc_this.edge_id
                                    }
                                ) ;
                }
            }
        }
    }
    std::tie(send_lookup,rank_send_counts) = cpb.sort_and_dedup(snd_keys) ;
    std::tie(recv_lookup,rank_recv_counts) = cpb.sort_and_dedup(rcv_keys) ;
    _reflux_coarse_edge_snd.clear() ;
    for( int i=0; i<_reflux_coarse_edge_descs.size(); ++i) {
        auto& dsc = _reflux_coarse_edge_descs[i] ;

        for( int iside=0; iside<dsc.n_sides; ++iside) {
            auto& dsc_this = dsc.sides[iside] ; 
            if ( dsc_this.octants.coarse.is_remote ) {
                // receive 
                comm_key_t key {
                            dsc_this.octants.coarse.quad_id, dsc_this.edge_id
                        } ;
                // fixme, ensure this is never used again 
                auto r = dsc_this.octants.coarse.owner_rank ; 
                GRACE_TRACE("Receive coarse, quadid {} rank {} edge {} buf idx {}",dsc_this.octants.coarse.quad_id,dsc_this.octants.coarse.owner_rank,dsc_this.edge_id, recv_lookup[r][key] );
                dsc_this.octants.coarse.quad_id = recv_lookup[r][key];
            } else {
                // send 
                for( int jside=0; jside<dsc.n_sides; ++jside){ 
                    if ( jside==iside ) continue ;
                    auto const& dsc_other = dsc.sides[jside] ; 
                    if ( !dsc_other.octants.coarse.is_remote ) continue ; 
                    comm_key_t key {
                                    dsc_this.octants.coarse.quad_id, dsc_this.edge_id
                                } ; 
                    hanging_remote_reflux_desc_t snd_desc{} ; 
                    snd_desc.qid = dsc_this.octants.coarse.quad_id ; 
                    auto r = dsc_other.octants.coarse.owner_rank ;
                    snd_desc.rank = r; 
                    snd_desc.elem_id = dsc_this.edge_id ; 
                    snd_desc.buf_id = send_lookup[r][key] ; 
                    _reflux_coarse_edge_snd.push_back(snd_desc) ;
                    GRACE_TRACE("Send coarse, quadid {} rank {} edge {} buf idx {}",dsc_this.octants.coarse.quad_id,dsc_other.octants.coarse.owner_rank,dsc_this.edge_id, send_lookup[r][key]  );
                }
            }
        }
    }
    #if 1
    // we need to dedup _reflux_coarse_edge_snd 
    std::sort(_reflux_coarse_edge_snd.begin(), _reflux_coarse_edge_snd.end(), 
        [](const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) {
            if (a.rank < b.rank) return true;
            if (b.rank < a.rank) return false;
            if (a.buf_id < b.buf_id) return true;
            if (b.buf_id < a.buf_id) return false;
            if (a.qid < b.qid) return true;
            if (b.qid < a.qid) return false;
            return (a.elem_id < b.elem_id);
        }
    );
    _reflux_coarse_edge_snd.erase(
        std::unique(_reflux_coarse_edge_snd.begin(), _reflux_coarse_edge_snd.end(),
            [](const hanging_remote_reflux_desc_t& a, const hanging_remote_reflux_desc_t& b) {
                return a.rank == b.rank && a.buf_id == b.buf_id && 
                    a.qid == b.qid && a.elem_id == b.elem_id;
            }
        ),
        _reflux_coarse_edge_snd.end()
    );
    #endif
    // finally allocate 
    // here we always send the emf along the edge --> no stagger
    // also we always send just one --> send size simply nx 
    send_size_emf = (nx) ; 

    size_t total_snd_emf_coarse_edge, total_rcv_emf_coarse_edge;
    std::tie(total_snd_emf_coarse_edge,_reflux_snd_emf_coarse_edge_size,_reflux_snd_emf_coarse_edge_off) = cpb.get_offsets_and_sizes(rank_send_counts,send_size_emf) ; 
    std::tie(total_rcv_emf_coarse_edge,_reflux_rcv_emf_coarse_edge_size,_reflux_rcv_emf_coarse_edge_off) = cpb.get_offsets_and_sizes(rank_recv_counts,send_size_emf) ; 

    ASSERT(total_snd_emf_coarse_edge/send_size_emf==_reflux_coarse_edge_snd.size(), "Mismatch send element count in coarse edge descriptor") ;
    _reflux_emf_coarse_edge_snd_buf = amr::reflux_edge_array_t("reflux_emf_coarse_edge_send") ; 
    _reflux_emf_coarse_edge_recv_buf = amr::reflux_edge_array_t("reflux_emf_coarse_edge_receive") ;
    
    _reflux_emf_coarse_edge_snd_buf.set_strides(nx) ; 
    _reflux_emf_coarse_edge_recv_buf.set_strides(nx) ; 

    _reflux_emf_coarse_edge_snd_buf.set_offsets(_reflux_snd_emf_coarse_edge_off) ; 
    _reflux_emf_coarse_edge_recv_buf.set_offsets(_reflux_rcv_emf_coarse_edge_off) ; 

    _reflux_emf_coarse_edge_snd_buf.realloc(total_snd_emf_coarse_edge) ; 
    _reflux_emf_coarse_edge_recv_buf.realloc(total_rcv_emf_coarse_edge) ;

    GRACE_VERBOSE("[REFLUX]: EMF-coarse-edge buffers constructed, total size send emf: {}, total size receive: {}", total_snd_emf_coarse_edge, total_rcv_emf_coarse_edge) ;

    _reflux_emf_coarse_edge_accumulation_buf = Kokkos::View<double**, grace::default_space>(
        "reflux_emf_coarse_edge_local_buffer", nx, _reflux_coarse_edge_descs.size()
    ) ; 

    // Detailed per-rank breakdown
    for( int r=0; r<nproc; ++r) {
        if (_reflux_snd_emf_coarse_edge_size[r] > 0 || _reflux_rcv_emf_coarse_edge_size[r] > 0) {
            GRACE_VERBOSE("[REFLUX]: EMFs coarse-edge: Rank {}: send[off={}, size={}] recv[off={}, size={}]", 
                r, 
                _reflux_snd_emf_coarse_edge_off[r], _reflux_snd_emf_coarse_edge_size[r],
                _reflux_rcv_emf_coarse_edge_off[r], _reflux_rcv_emf_coarse_edge_size[r]);
        }
    }
}
#endif 
} /* namespace grace */

