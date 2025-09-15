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
#include <grace/amr/bc_helpers.tpp>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/ghostzone_kernels/copy_kernels.hh>
#include <grace/amr/ghostzone_kernels/phys_bc_kernels.hh>
#include <grace/amr/ghostzone_kernels/pack_unpack_kernels.hh>

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <numeric>

//#define INSERT_FENCE_DEBUG_TASKS_ 

namespace grace {

void register_simple_face(
    p4est_iter_face_side_t const& quad,
    p4est_iter_face_side_t const& neighbor,
    std::vector<quad_neighbors_descriptor_t>& neighbors) 
{
    auto qid = quad.is.full.quadid + amr::get_local_quadrants_offset(quad.treeid) ; 

    int8_t const f = quad.face ;
    auto& desc = neighbors[qid].faces[f] ; 

    
    desc.level_diff = 0 ; 
    desc.kind = interface_kind_t::INTERNAL ; 
    desc.face = neighbor.face ; 

    desc.data.full.is_remote = neighbor.is.full.is_ghost ; 
    auto offset = neighbor.is.full.is_ghost  ? 0 : amr::get_local_quadrants_offset(neighbor.treeid) ;
    desc.data.full.quad_id = neighbor.is.full.quadid + offset ;
    // TODO figure out where the rank is  
    if ( desc.data.full.is_remote ) {
        desc.data.full.owner_rank = 
            p4est_comm_find_owner(
                amr::forest::get().get(), 
                neighbor.treeid, 
                neighbor.is.full.quad, 
                0 /* safe guess */
            );
    }
}

void grace_iterate_faces(
    p4est_iter_face_info_t * info,
    void* user_data 
) 
{
    auto ghosts = reinterpret_cast<std::vector<quad_neighbors_descriptor_t>*>(user_data) ; 
    sc_array_view_t<p4est_iter_face_side_t> sides{
        &(info->sides)
    } ;
    auto const& s0 = sides[0] ; auto const& s1 = sides[1] ; 

    /* Grid boundary case first */
    if (sides.size() == 1) {
        auto offset = amr::get_local_quadrants_offset(s0.treeid) ; 
        auto& desc = ghosts->at(s0.is.full.quadid + offset); 
        uint8_t f = s0.face ;
        desc.faces[f].kind = interface_kind_t::PHYS ; 
        return ; 
    }

    if ( s0.is_hanging ) {
        //! TODO 
    } else if ( s1.is_hanging ) {

    } else {
        if ( s0.is.full.is_ghost ) {
            register_simple_face(s1, s0, *ghosts) ; 
        } else if (s1.is.full.is_ghost ) {
            register_simple_face(s0, s1, *ghosts) ; 
        } else {
            register_simple_face(s0, s1, *ghosts) ; 
            register_simple_face(s1, s0, *ghosts) ; 
        }
    }
}

void amr_ghosts_impl_t::update() {
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
    p4est_ghost_layer = p4est_ghost_new(grace::amr::forest::get().get(), P4EST_CONNECT_FACE);

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
                  nullptr,                              /*edge*/
                  nullptr );                            /*corner*/
    
    build_remote_buffers() ; 
    build_task_list() ; 
    build_executor_runtime() ; 
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

void amr_ghosts_impl_t::build_remote_buffers() {
    /**************************/
    /*  These are per-rank    */
    /**************************/
    auto rank = parallel::mpi_comm_rank() ; 
    auto nproc= parallel::mpi_comm_size() ;

    auto nq = amr::get_local_num_quadrants() ; 
    std::size_t nx,ny,nz ; 
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    auto ngz = amr::get_n_ghosts() ; 

    std::size_t nvars = variables::get_n_evolved() ; 

    std::vector<int> rank_send_count_faces(nproc,0)
                   , rank_recv_count_faces(nproc,0) ; 
    send_rank_sizes.resize(nproc); 
    recv_rank_sizes.resize(nproc); 

    std::size_t face_size = nx*nx * ngz * nvars ; 

    struct face_key_t {
        face_descriptor_t * desc; 
        size_t rank;
        size_t key ;
    } ;    
    std::vector<face_key_t> halo_face_keys ; 
    std::vector<face_key_t> mirror_face_keys ; 
    std::size_t total_send_size{0UL}, total_recv_size{0UL} ; 
    for( size_t iq=0UL; iq<nq; iq+=1UL) {
        for (uint8_t f = 0; f < P4EST_FACES; ++f) {
            auto& face = ghost_layer[iq].faces[f] ; 
            if ( face.kind ==  interface_kind_t::PHYS ) continue ; 
            if ( face.level_diff != 0 ) {
                // not supported yet 
            } else {
                if ( !face.data.full.is_remote) continue ; 
                
                auto r = face.data.full.owner_rank ; 
                // face-id in "canonical order"
                auto f_c =  r < rank ? face.face : f ; 

                // replace index on the other side to buffer index 
                // send_buffer_id: where to pack the local face data (to send to rank r)
                // here we just use our local quad_id ordering which should be consistent 
                // with the **mirror** index ordering (ENSURE THIS IS TRUE!)
                {
                    face_key_t key ;
                    key.desc = &face ; 
                    key.rank = r ;
                    key.key = iq * P4EST_FACES + f_c ; 
                    mirror_face_keys.push_back(key) ; 
                } 

                // for receives we need to be careful and ensure that 
                // the ordering is consistent. Therefore we store here 
                // a key == P4EST_FACES * halo_id + face_id 
                // and later compute receive buffer indices by sorting those keys 
                {
                    face_key_t key ;
                    key.desc = &face ; 
                    key.rank = r ;
                    key.key = face.data.full.quad_id * P4EST_FACES + f_c ; 
                    halo_face_keys.push_back(key) ; 
                }

            } /* face not hanging */
        } /* for f .. nfaces */
    } /* for iq .. nquads */
    struct compare_keys {
        bool inline operator() (face_key_t const& a, face_key_t const& b) const {
            return a.key < b.key ;
        }
    } ; 

    std::sort(halo_face_keys.begin(), halo_face_keys.end(), compare_keys{});
    std::sort(mirror_face_keys.begin(), mirror_face_keys.end(), compare_keys{});

    // assign recv buffer indices
    for (auto const& hkey : halo_face_keys) {
        GRACE_TRACE("Register halo q {} f {} from rank {} recv_id {}",
                     hkey.key / P4EST_FACES, hkey.key % P4EST_FACES, hkey.rank, rank_recv_count_faces[hkey.rank]) ;
        auto r = hkey.rank;
        hkey.desc->data.full.recv_buffer_id = rank_recv_count_faces[r]++;
        recv_rank_sizes[r] += face_size;
        total_recv_size += face_size;
    }

    // assign send buffer indices
    for (auto const& mkey : mirror_face_keys) {
        GRACE_TRACE("Register mirror q {} f {} to rank {} send_id {}",
            mkey.key/P4EST_FACES, mkey.key%P4EST_FACES, mkey.rank, rank_send_count_faces[mkey.rank]) ; 
        auto r = mkey.rank;
        mkey.desc->send_buffer_id = rank_send_count_faces[r]++;
        send_rank_sizes[r] += face_size;
        total_send_size += face_size;
    }

    GRACE_TRACE("Total send size {}, total receive size {}", total_send_size, total_recv_size) ;
    // exclusive scan for rank offsets 
    send_rank_offsets.resize(nproc) ; 
    std::exclusive_scan( send_rank_sizes.begin(), send_rank_sizes.end()
                       , send_rank_offsets.begin(), 0) ; 
    recv_rank_offsets.resize(nproc) ; 
    std::exclusive_scan( recv_rank_sizes.begin(), recv_rank_sizes.end()
                       , recv_rank_offsets.begin(), 0) ;
    GRACE_TRACE("Was it really here??");
    Kokkos::fence() ; 
    // Allocate memory for MPI buffers
    Kokkos::realloc(_send_buffer, total_send_size) ; 
    Kokkos::realloc(_recv_buffer, total_recv_size) ;
    Kokkos::fence()  ;
    GRACE_TRACE("Checking send size {}, total receive size {}", _send_buffer.extent(0), _recv_buffer.extent(0)) ;
    for( int r=0; r<nproc; ++r) {
        GRACE_TRACE("Send rank {}, offset {} size {}", r, send_rank_offsets[r], send_rank_sizes[r]) ; 
        GRACE_TRACE("Recv rank {}, offset {} size {}", r, recv_rank_offsets[r], recv_rank_sizes[r]) ; 
    }
}


struct gpu_task_descriptor_t {
    int8_t face_src, face_dst ; 
    std::size_t dst_offset, src_offset; 
    std::size_t qid_src, qid_dst ; 
    bool dst_is_remote, src_is_remote ;  
} ; 

using task_bucket_t = std::vector<
    gpu_task_descriptor_t 
> ; 

#ifndef USE_DUMMY_GPU_TASK
template< typename view_t >
gpu_task_t make_gpu_copy_task(
      task_bucket_t const& bucket
    , view_t data 
    , device_stream_t& stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv 
    , task_id_t& task_counter 

) 
{
    using namespace grace::amr ; 
    GRACE_TRACE("Recording GPU-copy task (tid {}). Number of faces processed {}.", task_counter, bucket.size()) ; 
    Kokkos::View<size_t*> src_qid{"src_qid", bucket.size()}
                        , dst_qid{"dst_qid", bucket.size()} ; 
    Kokkos::View<uint8_t*> src_face{"src_face", bucket.size()}
                     , dst_face{"dst_face", bucket.size()}  ;
    auto src_qid_h = Kokkos::create_mirror_view(src_qid) ; 
    auto dst_qid_h = Kokkos::create_mirror_view(dst_qid) ; 
    auto src_face_h = Kokkos::create_mirror_view(src_face) ; 
    auto dst_face_h = Kokkos::create_mirror_view(dst_face) ; 
    int i{0} ; 
    for( auto const& d: bucket ) {
        src_qid_h(i) = d.qid_src; dst_qid_h(i) = d.qid_dst ; 
        src_face_h(i) = d.face_src; dst_face_h(i) = d.face_dst ; 
        i++ ; 
    }
    Kokkos::deep_copy(src_qid,src_qid_h) ; 
    Kokkos::deep_copy(dst_qid,dst_qid_h) ; 
    Kokkos::deep_copy(src_face,src_face_h) ; 
    Kokkos::deep_copy(dst_face,dst_face_h) ; 

    gpu_task_t task{} ;

    face_copy_k<true,decltype(data),decltype(data)> functor{
        data, data, src_qid, dst_qid, src_face, dst_face, VEC(nx,ny,nz), ngz
    } ; 
    
    Kokkos::DefaultExecutionSpace exec_space{stream} ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, {ngz, nx,nx,nv, bucket.size()}
        } ; 
    
    // TODO more informative name for the kernel launch 
    // will help in debug

    task._run = [functor, policy] () {
        
        GRACE_TRACE("Copy start.") ; 
        Kokkos::parallel_for("fill_ghostzones", policy, functor) ; 
        // TODO remove
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ;
        #endif 
        GRACE_TRACE("Copy done.") ; 
    };
    task.stream = &stream; 
    task.task_id = task_counter++ ; 

    return std::move(task) ; 

}


template< typename view_t >
gpu_task_t make_gpu_phys_bc_task(
      task_bucket_t const& bucket
    , view_t data 
    , Kokkos::View<bc_t*> var_bc_kind 
    , device_stream_t& stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 

) 
{
    GRACE_TRACE("Recording phys_BC task (tid {}). Number of faces processed {}.", task_counter, bucket.size()) ; 
    using namespace grace::amr ; 
    Kokkos::DefaultExecutionSpace exec_space{stream} ; 

    Kokkos::View<size_t*> qid{"qid", bucket.size()}; 
    Kokkos::View<uint8_t*> face{"face", bucket.size()} ; 
    auto qid_h = Kokkos::create_mirror_view(qid) ; 
    auto face_h = Kokkos::create_mirror_view(face) ; 
    int i{0} ; 
    for( auto const& d: bucket ) {
        qid_h(i) = d.qid_src; 
        face_h(i) = d.face_src; 
        i++ ; 
    }
    Kokkos::deep_copy(qid,qid_h) ;
    Kokkos::deep_copy(face,face_h) ; 

    gpu_task_t task{} ;

    face_phys_bc_k functor{
        data, data, qid, var_bc_kind, face, VEC(nx,ny,nz), ngz
    } ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, {ngz, nx,nx,nv, bucket.size()}
        } ;
    
    task._run = [functor, policy] () {
        GRACE_TRACE("Fill phys start") ; 
        Kokkos::parallel_for("fill_phys_ghostzones", policy, functor) ; 
        // TODO remove 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        #endif 
        GRACE_TRACE("Fill phys done") ; 
    };
    task.stream = &stream ; 
    task.task_id = task_counter++ ; 

    return std::move(task) ; 
}

template< typename view_t >
gpu_task_t make_pack_task(
      task_bucket_t const& sb
    , size_t rank 
    , view_t data 
    , Kokkos::View<double*> send_buf 
    , std::vector<std::size_t> const& send_rank_offsets
    , std::vector<task_id_t> const& send_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    // construct pack task
    auto const send_offset = send_rank_offsets[rank] ;
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> pack_src_qid{"pack_src_qid", sb.size()}
                        , pack_dest_qid{"pack_dst_qid", sb.size()} ; 
    Kokkos::View<uint8_t*> pack_src_face{"pack_src_fid", sb.size()}  ;
    auto pack_src_qid_h = Kokkos::create_mirror_view(pack_src_qid) ; 
    auto pack_dst_qid_h = Kokkos::create_mirror_view(pack_dest_qid) ; 
    auto pack_src_face_h =  Kokkos::create_mirror_view(pack_src_face) ; 
    size_t i{0UL} ; 
    for( auto const& d: sb ) {
        pack_src_qid_h(i) = d.qid_src ; 
        pack_dst_qid_h(i) = d.qid_dst ; 
        pack_src_face_h(i) = d.face_src ; 
        i += 1UL ; 
    }
    Kokkos::deep_copy(pack_src_qid,pack_src_qid_h)   ; 
    Kokkos::deep_copy(pack_dest_qid,pack_dst_qid_h)  ;  
    Kokkos::deep_copy(pack_src_face,pack_src_face_h) ;

    gpu_task_t pack_task{} ;

    amr::face_pack_k pack_functor {
        data, send_buf, pack_src_qid, pack_dest_qid, pack_src_face, VEC(nx,ny,nz), ngz, nv, send_offset
    } ; 

    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
    pack_policy{
        exec_space, {0,0,0,0,0}, {ngz, nx,nx, nv, sb.size()}
    } ; 
    
    pack_task._run = [pack_functor, pack_policy] () {
        GRACE_TRACE("Pack start.") ; 
        Kokkos::parallel_for("pack_ghostzones", pack_policy, pack_functor) ; 
        // TODO remove 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        #endif 
        GRACE_TRACE("Pack done.") ;
    } ; 
    pack_task.stream = &pup_stream ; 
    pack_task.task_id = task_counter ++ ; 
    // send depends on this 
    pack_task._dependents.push_back(send_task_id[rank]) ; 
    task_list[send_task_id[rank]] -> _dependencies.push_back(pack_task.task_id); 
    return pack_task ; 
}

template< typename view_t >
gpu_task_t make_unpack_task(
      task_bucket_t const& rb
    , size_t rank
    , view_t data 
    , Kokkos::View<double*> recv_buf 
    , std::vector<std::size_t> const& recv_rank_offsets
    , std::vector<task_id_t> const& recv_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    // construct unpack task 
    auto const recv_offset = recv_rank_offsets[rank] ;
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> unpack_src_qid{"unpack_src_qid", rb.size()}
                            , unpack_dest_qid{"unpack_dst_qid", rb.size()} ; 
    Kokkos::View<uint8_t*> unpack_dest_face{"unpack_src_fid", rb.size()}  ;
    auto unpack_src_qid_h = Kokkos::create_mirror_view(unpack_src_qid) ; 
    auto unpack_dst_qid_h = Kokkos::create_mirror_view(unpack_dest_qid) ; 
    auto unpack_dest_face_h =  Kokkos::create_mirror_view(unpack_dest_face) ; 
    size_t i = 0UL; 
    for( auto const& d: rb ) {
        GRACE_TRACE("{}", d.qid_src);
        unpack_src_qid_h(i) = d.qid_src ; 
        unpack_dst_qid_h(i) = d.qid_dst ; 
        unpack_dest_face_h(i) = d.face_dst ; 
        i += 1UL ; 
    }
    Kokkos::deep_copy(unpack_src_qid,unpack_src_qid_h)   ; 
    Kokkos::deep_copy(unpack_dest_qid,unpack_dst_qid_h)  ;  
    Kokkos::deep_copy(unpack_dest_face,unpack_dest_face_h) ;

    
    gpu_task_t unpack_task{} ;

    amr::face_unpack_k unpack_functor {
        recv_buf, data, unpack_src_qid, unpack_dest_qid, unpack_dest_face, VEC(nx,ny,nz), ngz, nv, recv_offset
    } ; 

    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
    unpack_policy{
        exec_space, {0,0,0,0,0}, {ngz, nx,nx,nv, rb.size()}
    } ; 
    
    unpack_task._run = [unpack_functor, unpack_policy] () {
        GRACE_TRACE("Unpack start.") ; 
        Kokkos::parallel_for("unpack_ghostzones", unpack_policy, unpack_functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        GRACE_TRACE("Unpack done.") ;
        #endif 
    } ; 
    unpack_task.stream = &pup_stream ; 
    unpack_task.task_id = task_counter ++ ; 

    // this depends on receive 
    unpack_task._dependencies.push_back(recv_task_id[rank]) ; 
    task_list[recv_task_id[rank]] -> _dependents.push_back(unpack_task.task_id) ; 
    return unpack_task ; 
}
#else
// START 
template< typename view_t >
gpu_task_t make_gpu_copy_task(
      task_bucket_t const& bucket
    , view_t data 
    , device_stream_t& stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv 
    , task_id_t& task_counter 

) 
{
    gpu_task_t task{} ;
    task.task_id = task_counter ++ ; 
    task._run = [] () {} ; 
    return task ;
}

template< typename view_t >
gpu_task_t make_gpu_phys_bc_task(
      task_bucket_t const& bucket
    , view_t data 
    , Kokkos::View<bc_t*> var_bc_kind 
    , device_stream_t& stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 

) 
{
    gpu_task_t task{} ;
    task.task_id = task_counter ++ ; 
    task._run = [] () {} ; 
    return task ;
}

template< typename view_t >
gpu_task_t make_pack_task(
      task_bucket_t const& sb
    , size_t rank 
    , view_t data 
    , Kokkos::View<double*> send_buf 
    , std::vector<std::size_t> const& send_rank_offsets
    , std::vector<task_id_t> const& send_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    gpu_task_t task{} ;
    task.task_id = task_counter ++ ; 
    task._run = [] () {} ; 
    return task ;
}

template< typename view_t >
gpu_task_t make_unpack_task(
      task_bucket_t const& rb
    , size_t rank
    , view_t data 
    , Kokkos::View<double*> recv_buf 
    , std::vector<std::size_t> const& recv_rank_offsets
    , std::vector<task_id_t> const& recv_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    gpu_task_t task{} ;
    task.task_id = task_counter ++ ; 
    task._run = [] () {} ; 
    return task ;
}
#endif 

template< typename view_t >
void populate_pup_tasks(
      std::vector<task_bucket_t> const& send_bucket
    , std::vector<task_bucket_t> const& recv_bucket
    , view_t data
    , Kokkos::View<double*> send_buf 
    , Kokkos::View<double*> recv_buf 
    , std::vector<std::size_t> const& send_rank_offsets
    , std::vector<std::size_t> const& recv_rank_offsets
    , std::vector<task_id_t> const& send_task_id
    , std::vector<task_id_t> const& recv_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
) 
{
    using namespace grace::amr ; 

    for( int r=0; r<parallel::mpi_comm_size(); ++r) {
        auto& sb = send_bucket[r] ; auto& rb = recv_bucket[r] ;
        
        
        if( sb.size() > 0 ) {
            GRACE_TRACE("Recording GPU-pack task (tid {}). Number of faces processed {}.", task_counter, sb.size()) ; 
            task_list.push_back(
                std::make_unique<gpu_task_t>(
                    make_pack_task(
                        sb, r, data, send_buf, send_rank_offsets, send_task_id, pup_stream, 
                        VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                    )
                )
            ) ;
        } 
        if (rb.size() > 0 ) {
            GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of faces processed {}.", task_counter+1, rb.size()) ; 
            task_list.push_back(
                std::make_unique<gpu_task_t>(
                    make_unpack_task(
                        rb, r, data, recv_buf, recv_rank_offsets, recv_task_id, pup_stream, 
                        VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                    )
                )
            ) ;
        }
    }
}; 


#ifndef USE_DUMMY_MPI_TASK
mpi_task_t make_mpi_send_task(
      std::size_t rr 
    , Kokkos::View<double*> send_buf 
    , std::vector<std::size_t> const& send_rank_offsets 
    , std::vector<std::size_t> const& send_rank_sizes
    , task_id_t& task_counter 
)
{
    GRACE_TRACE("Registering MPI Send task (tid {}).\n    Send to Rank {} {} elements, offset {}", task_counter, rr, send_rank_sizes[rr], send_rank_offsets[rr]) ; 

    ASSERT(send_rank_offsets[rr] + send_rank_sizes[rr] <= send_buf.extent(0), "Send out-of-bounds" ) ; 

    mpi_task_t task ; 
    task._run = [&, send_buf, rr] (MPI_Request* req) {
        parallel::mpi_isend(
              send_buf.data() + send_rank_offsets[rr]
            , send_rank_sizes[rr]
            , rr 
            , 0
            , MPI_COMM_WORLD
            , req
        ) ;
    } ; 
    task.task_id = task_counter++; 
    return task ; 
}

mpi_task_t make_mpi_recv_task(
      std::size_t rr 
    , Kokkos::View<double*> recv_buf 
    , std::vector<std::size_t> const& recv_rank_offsets 
    , std::vector<std::size_t> const& recv_rank_sizes
    , task_id_t& task_counter 
)
{
    GRACE_TRACE("Registering MPI Receive task (tid {}).\n    Receive from Rank {} {} elements, offset {}", task_counter, rr, recv_rank_sizes[rr], recv_rank_offsets[rr]) ; 

    ASSERT(recv_rank_offsets[rr] + recv_rank_sizes[rr] <= recv_buf.extent(0), "Receive out-of-bounds" ) ; 

    mpi_task_t task ; 
    task._run = [&, recv_buf, rr] (MPI_Request* req) {
        parallel::mpi_irecv(
              recv_buf.data() + recv_rank_offsets[rr]
            , recv_rank_sizes[rr]
            , rr 
            , 0
            , MPI_COMM_WORLD
            , req
        ) ;
    } ; 
    task.task_id = task_counter++; 
    return task ; 
}
#else 
mpi_task_t make_mpi_send_task(
      std::size_t rr 
    , Kokkos::View<double*> send_buf 
    , std::vector<std::size_t> const& send_rank_offsets 
    , std::vector<std::size_t> const& send_rank_sizes
    , task_id_t& task_counter 
)
{
    mpi_task_t task {} ;
    task.task_id = task_counter ++ ; 
    task._run = [] () {} ; 
    return task ; 
}
mpi_task_t make_mpi_recv_task(
      std::size_t rr 
    , Kokkos::View<double*> recv_buf 
    , std::vector<std::size_t> const& recv_rank_offsets 
    , std::vector<std::size_t> const& recv_rank_sizes
    , task_id_t& task_counter 
)
{
    mpi_task_t task {} ;
    task.task_id = task_counter ++ ; 
    task._run = [] () {} ; 
    return task ; 
}
#endif 

void amr_ghosts_impl_t::build_task_list() {
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
            #if 0
            std::unique_ptr<task_t> send = std::make_unique<mpi_task_t>(
                make_mpi_send_task(r, _send_buffer, send_rank_offsets, send_rank_sizes, task_counter)
            ) ; 
            send_task_id[r] = send->task_id ; 
            task_list.push_back(std::move(send)) ; 
            #endif 
            task_list.push_back(
                std::make_unique<mpi_task_t>(
                    make_mpi_send_task(r, _send_buffer, send_rank_offsets, send_rank_sizes, task_counter)
                )
            ) ; 
            send_task_id[r] = task_list.back()->task_id ; 
        }
        if (recv_rank_sizes[r] > 0 ){
            #if 0
            std::unique_ptr<task_t> recv = std::make_unique<mpi_task_t>(
                make_mpi_recv_task(r, _recv_buffer, recv_rank_offsets, recv_rank_sizes, task_counter)
            ) ; 
            recv_task_id[r] = recv->task_id ; 
            task_list.push_back(std::move(recv)) ;
            #endif 
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

    /***********************************************************************/
    task_bucket_t copy_kernels, phys_bc_kernels ; 
    std::vector<task_bucket_t> pack_kernels, unpack_kernels ; 
    pack_kernels.resize(nproc);
    unpack_kernels.resize(nproc) ;  

    for( size_t iq=0UL; iq<nq; iq+=1UL) {
        for (uint8_t f = 0; f < P4EST_FACES; ++f) {
            auto& face = ghost_layer[iq].faces[f] ; 
            if ( face.kind ==  interface_kind_t::PHYS ) {
                /* register phys bc kernel */
                gpu_task_descriptor_t desc{ } ; 
                desc.face_src = f ;
                desc.src_offset = 0 ;
                desc.src_is_remote = false ;
                desc.qid_src = iq ;
                // following info is not relevant here
                desc.dst_offset = 0;
                desc.face_dst = 0 ; 
                desc.dst_is_remote = false ;
                desc.qid_dst = 0 ;
                phys_bc_kernels.push_back(desc) ;
                if (phys_bc_kernels.size() >= BATCH_N_KERNELS) {
                    task_list.push_back(
                        std::make_unique<gpu_task_t>(
                                make_gpu_phys_bc_task(
                                  phys_bc_kernels
                                , state 
                                , var_bc_kind
                                , phys_bc_stream 
                                , VEC(nx,ny,nz), (size_t) ngz, (size_t) nvars 
                                , task_counter
                            )
                        ) 
                    ) ; 
                    phys_bc_kernels.clear(); 
                }
                continue ; 
            }
            if ( face.level_diff != 0 ) {
                // not supported yet 
            } else {
                if (face.data.full.is_remote) {
                    /* remote */
                    // pack for send 
                    gpu_task_descriptor_t desc_snd{ } ; 
                    desc_snd.face_src = f ;
                    /* in the buffers we always use face 0 for ordering */ 
                    desc_snd.face_dst = 0 ; 
                    desc_snd.dst_offset = send_rank_offsets[face.data.full.owner_rank] ;
                    desc_snd.src_offset = 0 ;
                    desc_snd.dst_is_remote = true ;
                    desc_snd.src_is_remote = false ;
                    desc_snd.qid_src = iq ;
                    desc_snd.qid_dst = face.send_buffer_id ;
                    pack_kernels[face.data.full.owner_rank].push_back(desc_snd) ; 
                    
                    // unpack from receive 
                    gpu_task_descriptor_t desc_rcv{ } ; 
                    /* in the buffers we always use face 0 for ordering */ 
                    desc_rcv.face_src = 0 ;
                    desc_rcv.face_dst = f ; 
                    desc_rcv.dst_offset = 0 ;
                    desc_rcv.src_offset = recv_rank_offsets[face.data.full.owner_rank] ;
                    desc_rcv.dst_is_remote = false ;
                    desc_rcv.src_is_remote = true ;
                    desc_rcv.qid_src = face.data.full.recv_buffer_id ;
                    desc_rcv.qid_dst = iq ;
                    unpack_kernels[face.data.full.owner_rank].push_back(desc_rcv) ; 

                } else {
                    /* local */
                    gpu_task_descriptor_t desc{ } ; 
                    desc.face_src = f ;
                    desc.face_dst = face.face ; 
                    desc.dst_offset = 0 ;
                    desc.src_offset = 0 ;
                    desc.dst_is_remote = false ;
                    desc.src_is_remote = false ;
                    desc.qid_src = iq ;
                    desc.qid_dst = face.data.full.quad_id ;
                    copy_kernels.push_back(desc) ; 
                    
                    if (copy_kernels.size() >= BATCH_N_KERNELS) {
                        task_list.push_back(
                            std::make_unique<gpu_task_t>(
                                make_gpu_copy_task(
                                    copy_kernels
                                    , state 
                                    , copy_stream 
                                    , VEC(nx,ny,nz), (size_t) ngz, (size_t) nvars
                                    , task_counter
                                )
                            )
                        ) ; 
                        copy_kernels.clear(); 
                    }
                }
            }
        }
    }

    populate_pup_tasks(
        pack_kernels, 
        unpack_kernels, 
        state, 
        _send_buffer, 
        _recv_buffer, 
        send_rank_offsets, 
        recv_rank_offsets, 
        send_task_id,
        recv_task_id,
        pup_stream, 
        VEC(nx,ny,nz), ngz, nvars, 
        task_counter, task_list 
    ) ; 


    // flush any remaining kernels 
    // that did not make the cut 
    if (!copy_kernels.empty()) {
        task_list.push_back(
            std::make_unique<gpu_task_t>(
                make_gpu_copy_task(
                    copy_kernels
                    , state 
                    , copy_stream 
                    , VEC(nx,ny,nz), (size_t) ngz, (size_t) nvars
                    , task_counter
                )
            )
        ) ; 
        copy_kernels.clear();
    }

    if (!phys_bc_kernels.empty()) {
        task_list.push_back(
            std::make_unique<gpu_task_t>(
                    make_gpu_phys_bc_task(
                        phys_bc_kernels
                    , state 
                    , var_bc_kind
                    , phys_bc_stream 
                    , VEC(nx,ny,nz), (size_t) ngz, (size_t) nvars 
                    , task_counter
                )
            ) 
        ) ;
        phys_bc_kernels.clear();
    }

}

} /* namespace grace */