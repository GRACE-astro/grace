/**
 * @file task_factories.hh
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
#include <grace/amr/ghostzone_kernels/pack_unpack_kernels.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/amr/ghostzone_kernels/index_helpers.hh>


#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <unordered_set>
#include <vector>
#include <numeric>

#ifndef GRACE_AMR_GHOSTZONE_PUP_KERNELS_TASK_FACTORY_HH
#define GRACE_AMR_GHOSTZONE_PUP_KERNELS_TASK_FACTORY_HH



namespace grace {
/**
 * @brief Create a task that packs data into send buffers.
 * @ingroup amr
 * @tparam elem_kind Kind of interface where data is taken from.
 * @param sb List of task descriptors.
 * @param ghost_array Array of neighbor descriptors.
 * @param rank Rank where the data will be sent.
 * @param data Data.
 * @param send_buf Send buffer.
 * @param send_task_id Task identifier of MPI send.
 * @param pup_stream Device stream.
 * @param ngz Number of ghost-cells.
 * @param nv Number of variables.
 * @param task_counter Current task counter.
 * @param task_list List of tasks.
 * @return gpu_task_t Task encapsulating the copy of data to send buffers.
 */
template< amr::element_kind_t elem_kind >
gpu_task_t make_pack_task(
      std::vector<gpu_task_desc_t> const& sb
    , std::vector<quad_neighbors_descriptor_t>& ghost_array 
    , size_t rank 
    , grace::var_array_t<GRACE_NSPACEDIM> data 
    , amr::ghost_array_t send_buf 
    , std::vector<task_id_t> const& send_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    GRACE_TRACE("Recording pack task (tid {}), # of elements {}, send_task_id {}", task_counter, sb.size(), send_task_id[rank]) ; 
    // construct pack task
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> pack_src_qid{"pack_src_qid", sb.size()}
                        , pack_dest_qid{"pack_dst_qid", sb.size()} ; 
    Kokkos::View<uint8_t*> pack_src_elem{"unpack_dst_eid", sb.size()}  ;
    auto pack_src_qid_h = Kokkos::create_mirror_view(pack_src_qid) ; 
    auto pack_dst_qid_h = Kokkos::create_mirror_view(pack_dest_qid) ; 
    auto pack_src_elem_h =  Kokkos::create_mirror_view(pack_src_elem) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d) -> std::tuple<size_t, size_t, uint8_t>  {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            return {std::get<0>(d),face.data.full.send_buffer_id, std::get<1>(d)} ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {std::get<0>(d), edge.data.full.send_buffer_id, std::get<1>(d)} ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {std::get<0>(d), corner.data.send_buffer_id, std::get<1>(d)} ;
        }
    } ; 


    size_t i{0UL} ; 
    for( auto const& d: sb ) {
        auto [qid_src, qid_dst, elem_src] = get_interface_info(d) ; 
        pack_src_qid_h(i) = qid_src ; 
        pack_dst_qid_h(i) = qid_dst ; 
        pack_src_elem_h(i) = elem_src ; 
        i += 1UL ; 
    }
    Kokkos::deep_copy(pack_src_qid,pack_src_qid_h)   ; 
    Kokkos::deep_copy(pack_dest_qid,pack_dst_qid_h)  ;  
    Kokkos::deep_copy(pack_src_elem,pack_src_elem_h) ;

    gpu_task_t pack_task{} ;

    amr::pack_op<elem_kind,decltype(data)> pack_functor {
        data, send_buf, pack_src_qid, pack_dest_qid, pack_src_elem, VEC(nx,ny,nz), ngz, nv, rank
    } ; 

    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
    pack_policy{
        exec_space, {0,0,0,0,0}, amr::get_iter_range<elem_kind>(ngz,nx,nv,sb.size())
    } ; 
    
    pack_task._run = [pack_functor, pack_policy] (view_alias_t alias) mutable {
        pack_functor.set_data_ptr(alias) ; 
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
/**
 * @brief Create a task that packs data into send buffers from coarse buffers.
 * @ingroup amr
 * @tparam elem_kind Kind of interface where data is taken from.
 * @param sb List of task descriptors.
 * @param ghost_array Array of neighbor descriptors.
 * @param rank Rank where the data will be sent.
 * @param cbuf Coarse buffers.
 * @param send_buf Send buffer.
 * @param send_task_id Task identifier of MPI send.
 * @param pup_stream Device stream.
 * @param ngz Number of ghost-cells.
 * @param nv Number of variables.
 * @param task_counter Current task counter.
 * @param task_list List of tasks.
 * @return gpu_task_t Task encapsulating the copy of data to send buffers.
 */
template< amr::element_kind_t elem_kind >
gpu_task_t make_pack_to_cbuf_task(
      std::vector<gpu_task_desc_t> const& sb
    , std::vector<quad_neighbors_descriptor_t>& ghost_array 
    , size_t rank 
    , grace::var_array_t<GRACE_NSPACEDIM> cbuf
    , amr::ghost_array_t send_buf 
    , std::vector<task_id_t> const& send_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , task_id_t const& restrict_task_id 
    , std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    GRACE_TRACE("Recording pack-to-cbuf task (tid {}), # of elements {}", task_counter, sb.size()) ; 
    // construct pack task
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> pack_src_qid{"pack_src_qid", sb.size()}
                        , pack_dest_qid{"pack_dst_qid", sb.size()} ; 
    Kokkos::View<uint8_t*> pack_src_elem{"unpack_dst_eid", sb.size()}  ;
    auto pack_src_qid_h = Kokkos::create_mirror_view(pack_src_qid) ; 
    auto pack_dst_qid_h = Kokkos::create_mirror_view(pack_dest_qid) ; 
    auto pack_src_elem_h =  Kokkos::create_mirror_view(pack_src_elem) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d) -> std::tuple<size_t, size_t, uint8_t>  {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            return {ghost_array[std::get<0>(d)].cbuf_id,face.data.full.send_buffer_id, std::get<1>(d)} ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {ghost_array[std::get<0>(d)].cbuf_id, edge.data.full.send_buffer_id, std::get<1>(d)} ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {ghost_array[std::get<0>(d)].cbuf_id, corner.data.send_buffer_id, std::get<1>(d)} ;
        }
    } ;  


    size_t i{0UL} ; 
    for( auto const& d: sb ) {
        auto [qid_src, qid_dst, elem_src] = get_interface_info(d) ; 
        pack_src_qid_h(i) = qid_src ; 
        pack_dst_qid_h(i) = qid_dst ; 
        pack_src_elem_h(i) = elem_src ; 
        i += 1UL ; 
    }
    Kokkos::deep_copy(pack_src_qid,pack_src_qid_h)   ; 
    Kokkos::deep_copy(pack_dest_qid,pack_dst_qid_h)  ;  
    Kokkos::deep_copy(pack_src_elem,pack_src_elem_h) ;

    gpu_task_t pack_task{} ;

    amr::pack_to_cbuf_op<elem_kind,decltype(cbuf)> pack_functor {
        cbuf, send_buf, pack_src_qid, pack_dest_qid, pack_src_elem, VEC(nx,ny,nz), ngz, nv, rank
    } ; 

    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
    pack_policy{
        exec_space, {0,0,0,0,0}, amr::get_iter_range<elem_kind>(ngz,nx/2,nv,sb.size())
    } ; 
    
    pack_task._run = [pack_functor, pack_policy] (view_alias_t alias) mutable {
        // don't change data ptr here!! 
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
    // this depends on restrict
    pack_task._dependencies.push_back(restrict_task_id) ; 
    task_list[restrict_task_id] -> _dependents.push_back(pack_task.task_id); 
    return pack_task ; 
}
/**
 * @brief 
 * 
 * @tparam elem_kind 
 * @param rb 
 * @param ghost_array 
 * @param rank 
 * @param data 
 * @param recv_buf 
 * @param recv_task_id 
 * @param pup_stream 
 * @param ngz 
 * @param nv 
 * @param task_counter 
 * @param task_list 
 * @return gpu_task_t 
 */
template< amr::element_kind_t elem_kind >
gpu_task_t make_unpack_task(
      std::vector<gpu_task_desc_t> const& rb
    , std::vector<quad_neighbors_descriptor_t>& ghost_array 
    , size_t rank
    , grace::var_array_t<GRACE_NSPACEDIM> data 
    , amr::ghost_array_t recv_buf 
    , std::vector<task_id_t> const& recv_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    GRACE_TRACE("Recording unpack task (tid {}), # of elements {}", task_counter, rb.size()) ; 
    // construct unpack task 
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> unpack_src_qid{"unpack_src_qid", rb.size()}
                            , unpack_dest_qid{"unpack_dst_qid", rb.size()} ; 
    Kokkos::View<uint8_t*> unpack_dest_elem{"unpack_src_eid", rb.size()}  ;
    auto unpack_src_qid_h = Kokkos::create_mirror_view(unpack_src_qid) ; 
    auto unpack_dst_qid_h = Kokkos::create_mirror_view(unpack_dest_qid) ; 
    auto unpack_dest_elem_h =  Kokkos::create_mirror_view(unpack_dest_elem) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d)-> std::tuple<size_t, size_t, uint8_t>  {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            return {face.data.full.recv_buffer_id,std::get<0>(d), std::get<1>(d)} ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {edge.data.full.recv_buffer_id, std::get<0>(d) , std::get<1>(d)} ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return { corner.data.recv_buffer_id, std::get<0>(d), std::get<1>(d)} ;
        }
    } ; 
    auto const set_task_id = [&] (gpu_task_desc_t const& d)
    {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.full.task_id = task_counter ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.full.task_id = task_counter ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id = task_counter ;
        }
    } ; 

    size_t i = 0UL; 
    for( auto const& d: rb ) {
        auto [qid_src, qid_dst, elem_dst] = get_interface_info(d) ; 
        unpack_src_qid_h(i) = qid_src ; 
        unpack_dst_qid_h(i) = qid_dst ; 
        unpack_dest_elem_h(i) = elem_dst ;
        set_task_id(d) ;  
        i += 1UL ; 
    }
    Kokkos::deep_copy(unpack_src_qid,unpack_src_qid_h)   ; 
    Kokkos::deep_copy(unpack_dest_qid,unpack_dst_qid_h)  ;  
    Kokkos::deep_copy(unpack_dest_elem,unpack_dest_elem_h) ;

    
    gpu_task_t unpack_task{} ;

    amr::unpack_op<elem_kind,decltype(data)> unpack_functor {
        recv_buf, data, unpack_src_qid, unpack_dest_qid, unpack_dest_elem, VEC(nx,ny,nz), ngz, nv, rank
    } ; 

    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
    unpack_policy{
        exec_space, {0,0,0,0,0}, amr::get_iter_range<elem_kind>(ngz,nx,nv,rb.size())
    } ; 
    
    unpack_task._run = [unpack_functor, unpack_policy] (view_alias_t alias) mutable {
        unpack_functor.set_data_ptr(alias) ; 
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

template< amr::element_kind_t elem_kind >
gpu_task_t make_unpack_to_cbuf_task(
      std::vector<gpu_task_desc_t> const& rb
    , std::vector<quad_neighbors_descriptor_t>& ghost_array 
    , size_t rank
    , grace::var_array_t<GRACE_NSPACEDIM> cbuf 
    , amr::ghost_array_t recv_buf 
    , std::vector<task_id_t> const& recv_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    GRACE_TRACE("Recording unpack-to-cbuf task (tid {}), # of elements {}", task_counter, rb.size()) ; 
    // construct unpack task 
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> cbuf_qid_d{"unpack_to_cbuf_cbuf_qid", rb.size()}
                        , ghost_qid_d{"unpack_to_cbuf_ghost_qid", rb.size()} ; 
    Kokkos::View<uint8_t*> cbuf_elem_d{"unpack_to_cbuf_cbuf_eid", rb.size()}
                        , child_id_d{"unpack_to_cbuf_cid", rb.size()} ;

    auto cbuf_qid_h = Kokkos::create_mirror_view(cbuf_qid_d) ; 
    auto ghost_qid_h = Kokkos::create_mirror_view(ghost_qid_d) ; 
    auto cbuf_elem_h =  Kokkos::create_mirror_view(cbuf_elem_d) ; 
    auto child_id_h =  Kokkos::create_mirror_view(child_id_d) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d) -> std::tuple<size_t, size_t, uint8_t, uint8_t>  {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            // return cbuf_id, ghost_qid, face_id, child_id 
            return {ghost_array[std::get<0>(d)].cbuf_id, face.data.full.recv_buffer_id, std::get<1>(d), face.child_id } ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {ghost_array[std::get<0>(d)].cbuf_id, edge.data.full.recv_buffer_id, std::get<1>(d), edge.child_id } ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {ghost_array[std::get<0>(d)].cbuf_id, corner.data.recv_buffer_id, std::get<1>(d), 0 } ;
        }
    } ; 
    auto const set_task_id = [&] (gpu_task_desc_t const& d)
    {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.full.task_id = task_counter ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.full.task_id = task_counter ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id = task_counter ;
        }
    } ; 
    size_t i = 0UL; 
    for( auto const& d: rb ) {
        auto [cbuf_id, ghost_qid, cbuf_eid, child_id] = get_interface_info(d) ; 
        cbuf_qid_h(i) = cbuf_id ; 
        ghost_qid_h(i) = ghost_qid ; 
        cbuf_elem_h(i) = cbuf_eid ; 
        child_id_h(i) = child_id ; 
        set_task_id(d) ; 
        i += 1UL ; 
    }
    Kokkos::deep_copy(cbuf_qid_d,cbuf_qid_h)   ; 
    Kokkos::deep_copy(ghost_qid_d,ghost_qid_h)  ;  
    Kokkos::deep_copy(cbuf_elem_d,cbuf_elem_h) ;
    Kokkos::deep_copy(child_id_d,child_id_h) ;

    
    gpu_task_t unpack_task{} ;

    amr::unpack_to_cbuf_op<elem_kind,decltype(cbuf)> unpack_functor {
        recv_buf, cbuf, ghost_qid_d, cbuf_qid_d, cbuf_elem_d, child_id_d, VEC(nx,ny,nz), ngz, nv, rank
    } ; 

    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
    unpack_policy{
        exec_space, {0,0,0,0,0}, amr::get_iter_range<elem_kind>(ngz,nx/2,nv,rb.size(), true /*extend loops by ngz*/)
    } ; 
    
    unpack_task._run = [unpack_functor, unpack_policy] (view_alias_t alias) mutable {
        //don't change data ptr here! 
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

template< amr::element_kind_t elem_kind >
gpu_task_t make_unpack_from_cbuf_task(
      std::vector<gpu_hanging_task_desc_t> const& rb
    , std::vector<quad_neighbors_descriptor_t>& ghost_array 
    , size_t rank
    , grace::var_array_t<GRACE_NSPACEDIM> data 
    , amr::ghost_array_t recv_buf 
    , std::vector<task_id_t> const& recv_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    GRACE_TRACE("Recording unpack-from-cbuf task (tid {}), # of elements {}", task_counter, rb.size()) ; 
    // construct unpack task 
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> view_qid_d{"unpack_from_cbuf_view_qid", rb.size()}
                        , ghost_qid_d{"unpack_from_cbuf_ghost_qid", rb.size()} ; 
    Kokkos::View<uint8_t*> view_elem_d{"unpack_from_cbuf_view_eid", rb.size()}
                        , child_id_d{"unpack_from_cbuf_cid", rb.size()} ;

    auto view_qid_h = Kokkos::create_mirror_view(view_qid_d) ; 
    auto ghost_qid_h = Kokkos::create_mirror_view(ghost_qid_d) ; 
    auto view_elem_h =  Kokkos::create_mirror_view(view_elem_d) ; 
    auto child_id_h =  Kokkos::create_mirror_view(child_id_d) ; 

    auto const get_interface_info = [&] (gpu_hanging_task_desc_t const& d) -> std::tuple<size_t, size_t, uint8_t, uint8_t> {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            // return cbuf_id, ghost_qid, face_id, child_id 
            return {std::get<0>(d), face.data.hanging.recv_buffer_id[std::get<2>(d)], std::get<1>(d), std::get<2>(d) } ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {std::get<0>(d), edge.data.hanging.recv_buffer_id[std::get<2>(d)], std::get<1>(d), std::get<2>(d) } ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {std::get<0>(d), corner.data.recv_buffer_id, std::get<1>(d), 0 } ;
        }
    } ; 
    auto const set_task_id = [&] (gpu_hanging_task_desc_t const& d)
    {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.hanging.task_id[std::get<2>(d)] = task_counter ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.hanging.task_id[std::get<2>(d)] = task_counter ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id = task_counter ;
        }
    } ; 
    size_t i = 0UL; 
    for( auto const& d: rb ) {
        auto [qid, ghost_qid, eid, child_id] = get_interface_info(d) ; 
        view_qid_h(i) = qid ; 
        ghost_qid_h(i) = ghost_qid ; 
        view_elem_h(i) = eid ; 
        child_id_h(i) = child_id ; 
        set_task_id(d) ; 
        i += 1UL ; 
    }
    Kokkos::deep_copy(view_qid_d,view_qid_h)   ; 
    Kokkos::deep_copy(ghost_qid_d,ghost_qid_h)  ;  
    Kokkos::deep_copy(view_elem_d,view_elem_h) ;
    Kokkos::deep_copy(child_id_d,child_id_h) ;

    
    gpu_task_t unpack_task{} ;

    amr::unpack_from_cbuf_op<elem_kind,decltype(data)> unpack_functor {
        recv_buf, data, ghost_qid_d, view_qid_d, view_elem_d, child_id_d, VEC(nx,ny,nz), ngz, nv, rank
    } ; 

    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
    unpack_policy{
        exec_space, {0,0,0,0,0}, amr::get_iter_range<elem_kind>(ngz,nx/2,nv,rb.size(), false /*extend loops by ngz*/)
    } ; 
    
    unpack_task._run = [unpack_functor, unpack_policy] (view_alias_t alias) mutable {
        unpack_functor.set_data_ptr(alias) ; 
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

void insert_pup_tasks(
      std::vector<quad_neighbors_descriptor_t> & ghost_array
    , std::vector<bucket_t> const& pack_kernels
    , std::vector<bucket_t> const& unpack_kernels
    , std::vector<bucket_t> const& pack_to_cbuf_kernels
    , std::vector<bucket_t> const& unpack_to_cbuf_kernels
    , std::vector<hang_bucket_t> const&  unpack_from_cbuf_kernels
    , grace::var_array_t<GRACE_NSPACEDIM> data
    , grace::var_array_t<GRACE_NSPACEDIM> coarse_buffers 
    , amr::ghost_array_t send_buf 
    , amr::ghost_array_t recv_buf 
    , std::vector<task_id_t> const& send_task_id
    , std::vector<task_id_t> const& recv_task_id
    , task_id_t const& restrict_task_id
    , device_stream_t& pup_stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv
    , task_id_t& task_counter 
    , std::vector<std::unique_ptr<task_t>>& task_list 
) 
{
    using namespace grace::amr ; 

    #define MAKE_PACK(r, kind) \
    if(pack_kernels[r][static_cast<size_t>(kind)].size()>0)\
    task_list.push_back( \
        std::make_unique<gpu_task_t>( \
            make_pack_task<kind>( \
                pack_kernels[r][static_cast<size_t>(kind)], \
                ghost_array, r, data, \
                send_buf, send_task_id, pup_stream, \
                VEC(nx,ny,nz), ngz, nv, \
                task_counter, task_list \
            ) \
        ) \
    )
    #define MAKE_PACK_TO_CBUF(r, kind) \
    if(pack_to_cbuf_kernels[r][static_cast<size_t>(kind)].size()>0)\
    task_list.push_back( \
        std::make_unique<gpu_task_t>( \
            make_pack_to_cbuf_task<kind>( \
                pack_to_cbuf_kernels[r][static_cast<size_t>(kind)], \
                ghost_array, r, coarse_buffers, \
                send_buf, send_task_id, pup_stream, \
                VEC(nx,ny,nz), ngz, nv, \
                task_counter, restrict_task_id, task_list \
            ) \
        ) \
    )

    #define MAKE_UNPACK(r, kind) \
    if (unpack_kernels[r][static_cast<size_t>(kind)].size()>0)\
    task_list.push_back( \
        std::make_unique<gpu_task_t>( \
            make_unpack_task<kind>( \
                unpack_kernels[r][static_cast<size_t>(kind)], \
                ghost_array, r, data, \
                recv_buf, recv_task_id, pup_stream, \
                VEC(nx,ny,nz), ngz, nv, \
                task_counter, task_list \
            ) \
        ) \
    )

    #define MAKE_UNPACK_TO_CBUF(r, kind) \
    if (unpack_to_cbuf_kernels[r][static_cast<size_t>(kind)].size()>0) \
    task_list.push_back( \
        std::make_unique<gpu_task_t>( \
            make_unpack_to_cbuf_task<kind>( \
                unpack_to_cbuf_kernels[r][static_cast<size_t>(kind)], \
                ghost_array, r, coarse_buffers, \
                recv_buf, recv_task_id, pup_stream, \
                VEC(nx,ny,nz), ngz, nv, \
                task_counter, task_list \
            ) \
        ) \
    )

    #define MAKE_UNPACK_FROM_CBUF(r, kind) \
    if(unpack_from_cbuf_kernels[r][static_cast<size_t>(kind)].size()>0) \
    task_list.push_back( \
        std::make_unique<gpu_task_t>( \
            make_unpack_from_cbuf_task<kind>( \
                unpack_from_cbuf_kernels[r][static_cast<size_t>(kind)], \
                ghost_array, r, data, \
                recv_buf, recv_task_id, pup_stream, \
                VEC(nx,ny,nz), ngz, nv, \
                task_counter, task_list \
            ) \
        ) \
    )

    #define MAKE_TASKS(r, kind)\
    MAKE_PACK(r,kind);\
    MAKE_PACK_TO_CBUF(r,kind);\
    MAKE_UNPACK(r,kind);\
    MAKE_UNPACK_TO_CBUF(r,kind);\
    MAKE_UNPACK_FROM_CBUF(r,kind)

     

    for( int r=0; r<parallel::mpi_comm_size(); ++r) {
        GRACE_TRACE("Inserting pup tasks, rank {}", r) ;
        MAKE_TASKS(r,amr::FACE);
        MAKE_TASKS(r,amr::EDGE);
        MAKE_TASKS(r,amr::CORNER);
    } // for r in ranks 

    #undef MAKE_TASKS
    #undef MAKE_PACK 
    #undef MAKE_UNPACK
    #undef MAKE_PACK_TO_CBUF
    #undef MAKE_UNPACK_TO_CBUF
    #undef MAKE_UNPACK_FROM_CBUF
}; 

}

#endif 