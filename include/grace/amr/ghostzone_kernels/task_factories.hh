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
#include <grace/amr/ghostzone_kernels/copy_kernels.hh>
#include <grace/amr/ghostzone_kernels/phys_bc_kernels.hh>
#include <grace/amr/ghostzone_kernels/pack_unpack_kernels.hh>

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <numeric>

#ifndef GRACE_AMR_GHOSTZONE_KERNELS_TASK_FACTORY_HH
#define GRACE_AMR_GHOSTZONE_KERNELS_TASK_FACTORY_HH

namespace grace {

using gpu_task_desc_t = std::pair<size_t, uint8_t> ; 
using gpu_hanging_task_desc_t = std::tuple<size_t,uint8_t,uint8_t> ; 

using bucket_t = std::array<std::vector<gpu_task_desc_t>,3> ; 
using hang_bucket_t = std::array<std::vector<gpu_hanging_task_desc_t>,3> ; 

template< amr::element_kind_t elem_kind >
Kokkos::Array<int64_t, 5> get_iter_range(size_t ngz,size_t _nx, size_t nv, size_t nq,  bool offset=false) {
    int64_t const nx = offset ? static_cast<int64_t>(_nx + ngz) : static_cast<int64_t>( _nx) ; 
    if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
        return Kokkos::Array<int64_t, 5>{static_cast<int64_t>(ngz),nx,nx,static_cast<int64_t>(nv),static_cast<int64_t>(nq)} ; 
    } else if constexpr  ( elem_kind == amr::element_kind_t::EDGE ) {
        return Kokkos::Array<int64_t, 5>{static_cast<int64_t>(ngz),static_cast<int64_t>(ngz),nx,static_cast<int64_t>(nv),static_cast<int64_t>(nq)} ; 
    } else {
        return Kokkos::Array<int64_t, 5>{static_cast<int64_t>(ngz),static_cast<int64_t>(ngz),static_cast<int64_t>(ngz),static_cast<int64_t>(nv),static_cast<int64_t>(nq)} ; 
    }
}

template< amr::element_kind_t elem_kind >
gpu_task_t make_gpu_copy_task(
      std::vector<gpu_task_desc_t> const& bucket
    , std::vector<quad_neighbors_descriptor_t>& ghost_layer
    , grace::var_array_t<GRACE_NSPACEDIM> data 
    , device_stream_t& stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv 
    , task_id_t& task_counter 

) 
{
    using namespace grace::amr ;
    
    GRACE_TRACE("Recording GPU-copy task (tid {}). Number of elements processed {}.", task_counter, bucket.size()) ; 
    Kokkos::View<size_t*> src_qid{"copy_src_qid", bucket.size()}
                        , dst_qid{"copy_dst_qid", bucket.size()} ; 
    Kokkos::View<uint8_t*> src_elem{"copy_src_elem_id", bucket.size()}
                         , dst_elem{"copy_dst_elem_id", bucket.size()}  ;
    auto src_qid_h = Kokkos::create_mirror_view(src_qid) ; 
    auto dst_qid_h = Kokkos::create_mirror_view(dst_qid) ; 
    auto src_face_h = Kokkos::create_mirror_view(src_elem) ; 
    auto dst_face_h = Kokkos::create_mirror_view(dst_elem) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d) -> std::tuple<size_t,size_t,uint8_t,uint8_t> {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            return {face.data.full.quad_id, std::get<0>(d), face.face, std::get<1>(d)} ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {edge.data.full.quad_id, std::get<0>(d), edge.edge, std::get<1>(d)} ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {corner.data.quad_id, std::get<0>(d), corner.corner, std::get<1>(d)} ;
        }
    } ; 

    auto const set_task_id = [&] (gpu_task_desc_t const& d)
    {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.full.task_id = task_counter ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.full.task_id = task_counter ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id = task_counter ;
        }
    } ; 

    int i{0} ; 
    for( auto const& d: bucket ) {
        auto const qid = d.first() ; 
        auto const eid = d.second() ; 
        auto [src_qid,dst_qid,src_eid,dst_eid] = get_interface_info(d) ; 
        src_qid_h(i) = src_qid; dst_qid_h(i) = dst_qid ; 
        src_elem_h(i) = src_eid; dst_elem_h(i) = dst_eid ; 
        set_task_id(d) ; 
        i++ ; 
    }

    Kokkos::deep_copy(src_qid,src_qid_h) ; 
    Kokkos::deep_copy(dst_qid,dst_qid_h) ; 
    Kokkos::deep_copy(src_elem,src_elem_h) ; 
    Kokkos::deep_copy(dst_elem,dst_elem_h) ; 

    gpu_task_t task{} ;

    copy_op<elem_kind,decltype(data),decltype(data)> functor{
        data, src_qid, dst_qid, src_elem, dst_elem, VEC(nx,ny,nz), ngz
    } ; 
    
    Kokkos::DefaultExecutionSpace exec_space{stream} ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx,nv,bucket.size())
        } ; 
 
    task._run = [functor, policy] (view_alias_t alias) mutable {
        functor.set_data_ptr(alias) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE("Copy start.") ; 
        #endif 
        Kokkos::parallel_for("copy_ghostzones", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        GRACE_TRACE("Copy end") ; 
        #endif 
    };
    task.stream = &stream; 
    task.task_id = task_counter++ ; 

    return std::move(task) ; 

}

template< amr::element_kind_t elem_kind >
gpu_task_t make_gpu_copy_to_cbuf_task(
      std::vector<gpu_task_desc_t> const& bucket
    , std::vector<quad_neighbors_descriptor_t>& ghost_layer
    , grace::var_array_t<GRACE_NSPACEDIM> data 
    , grace::var_array_t<GRACE_NSPACEDIM> cbuf 
    , device_stream_t& stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv 
    , task_id_t& task_counter 

) 
{
    using namespace grace::amr ;
    
    GRACE_TRACE("Recording GPU-copy-to-cbuf task (tid {}). Number of elements processed {}.", task_counter, bucket.size()) ; 
    Kokkos::View<size_t*> src_qid{"copy_v2c_src_qid", bucket.size()}
                        , dst_qid{"copy_v2c_dst_qid", bucket.size()} ; 
    Kokkos::View<uint8_t*> src_elem{"copy_v2c_src_elem_id", bucket.size()}
                         , dst_elem{"copy_v2c_dst_elem_id", bucket.size()}  
                         , ic{"copy_v2c_child_id", bucket.size()};
    auto src_qid_h = Kokkos::create_mirror_view(src_qid) ; 
    auto dst_qid_h = Kokkos::create_mirror_view(dst_qid) ; 
    auto src_face_h = Kokkos::create_mirror_view(src_elem) ; 
    auto dst_face_h = Kokkos::create_mirror_view(dst_elem) ; 
    auto ic_h = Kokkos::create_mirror_view(ic) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d) -> std::tuple<size_t, size_t,uint8_t,uint8_t, uint8_t> {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            //return src qid dst qid src face dst face child id
            return {face.data.full.quad_id, ghost_layer[std::get<0>(d)].cbuf_id,  face.face, std::get<1>(d), face.child_id} ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {edge.data.full.quad_id, ghost_layer[std::get<0>(d)].cbuf_id,  edge.edge, std::get<1>(d), edge.child_id} ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {edge.data.quad_id, ghost_layer[std::get<0>(d)].cbuf_id,  corner.corner, std::get<1>(d), corner.child_id} ;
        }
    } ; 

    auto const set_task_id = [&] (gpu_task_desc_t const& d)
    {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.full.task_id = task_counter ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.full.task_id = task_counter ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id = task_counter ;
        }
    } ; 

    int i{0} ; 
    for( auto const& d: bucket ) {
        auto const qid = d.first() ; 
        auto const eid = d.second() ; 
        auto [src_qid, dst_qid, src_eid, dst_eid, child_id] = get_interface_info(d) ; 
        src_qid_h(i) = src_qid; dst_qid_h(i) = dst_qid ; 
        src_elem_h(i) = src_eid ; dst_elem_h(i) = dst_eid ;
        ic_h(i) = child_id ; 
        set_task_id(d) ; 
        i++ ; 
    }

    Kokkos::deep_copy(src_qid,src_qid_h) ; 
    Kokkos::deep_copy(dst_qid,dst_qid_h) ; 
    Kokkos::deep_copy(src_elem,src_elem_h) ; 
    Kokkos::deep_copy(dst_elem,dst_elem_h) ; 
    Kokkos::deep_copy(ic, ic_h) ; 

    gpu_task_t task{} ;

    copy_to_cbuf_op<elem_kind,decltype(data),decltype(cbuf)> functor{
        data, cbuf, src_qid, dst_qid, src_elem, dst_elem, ic, VEC(nx,ny,nz), ngz
    } ; 
    
    Kokkos::DefaultExecutionSpace exec_space{stream} ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx/2,nv,bucket.size(),true /*add +ngz to nx ranges*/)
        } ; 
 
    task._run = [functor, policy] (view_alias_t alias) mutable {
        functor.set_data_ptr(alias) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE("Copy start.") ; 
        #endif 
        Kokkos::parallel_for("copy_ghostzones_v2c", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        GRACE_TRACE("Copy end") ; 
        #endif 
    };
    task.stream = &stream; 
    task.task_id = task_counter++ ; 

    return std::move(task) ; 

}

template< amr::element_kind_t elem_kind >
gpu_task_t make_gpu_copy_from_cbuf_task(
      std::vector<gpu_hanging_task_desc_t> const& bucket
    , std::vector<quad_neighbors_descriptor_t>& ghost_layer
    , grace::var_array_t<GRACE_NSPACEDIM> data 
    , grace::var_array_t<GRACE_NSPACEDIM> cbuf 
    , device_stream_t& stream
    , VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv 
    , task_id_t& task_counter 
    , task_id_t const& restrict_task_id 
    , std::vector<std::unique_ptr<task_t>>& task_list 

) 
{
    using namespace grace::amr ;
    
    GRACE_TRACE("Recording GPU-copy-from-cbuf task (tid {}). Number of elements processed {}.", task_counter, bucket.size()) ; 
    Kokkos::View<size_t*> src_qid{"copy_c2v_src_qid", bucket.size()}
                        , dst_qid{"copy_c2v_dst_qid", bucket.size()} ; 
    Kokkos::View<uint8_t*> src_elem{"copy_c2v_src_elem_id", bucket.size()}
                         , dst_elem{"copy_c2v_dst_elem_id", bucket.size()}  
                         , ic{"copy_c2v_child_id", bucket.size()};
    auto src_qid_h = Kokkos::create_mirror_view(src_qid) ; 
    auto dst_qid_h = Kokkos::create_mirror_view(dst_qid) ; 
    auto src_face_h = Kokkos::create_mirror_view(src_elem) ; 
    auto dst_face_h = Kokkos::create_mirror_view(dst_elem) ; 
    auto ic_h = Kokkos::create_mirror_view(ic) ; 

    auto const get_interface_info = [&] (gpu_hanging_task_desc_t const& d) -> std::tuple<size_t, size_t,uint8_t,uint8_t, uint8_t> {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ;
            auto qid =  face.data.hanging.quad_id[std::get<2>(d)] ; 
            // return src_id, dst_id, src_e, dst_e, child_id 
            return {ghost_layer[qid].cbuf_id,std::get<0>(d), face.face, std::get<1>(d), std::get<2>(d)} ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            auto qid =  edge.data.hanging.quad_id[std::get<2>(d)] ; 
            return {ghost_layer[qid].cbuf_id,std::get<0>(d), edge.edge, std::get<1>(d), std::get<2>(d)} ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            auto qid = corner.data.quad_id ; 
            return {ghost_layer[qid].cbuf_id, std::get<0>(d), edge.edge, std::get<1>(d), 0} ;
        }
    } ; 

    auto const set_task_id = [&] (gpu_hanging_task_desc_t const& d)
    {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.hanging.task_id[std::get<2>(d)] = task_counter ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.hanging.task_id[std::get<2>(d)] = task_counter ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id = task_counter ;
        }
    } ; 

    int i{0} ; 
    for( auto const& d: bucket ) {
        auto [src_qid, dst_qid, src_eid, dst_eid ,child_id] = get_interface_info(d) ; 
        src_qid_h(i) = src_qid; dst_qid_h(i) = dst_qid ; 
        src_elem_h(i) = src_eid ; dst_elem_h(i) = dst_eid ;
        ic_h(i) = child_id ; 
        set_task_id(d) ; 
        i++ ; 
    }

    Kokkos::deep_copy(src_qid,src_qid_h) ; 
    Kokkos::deep_copy(dst_qid,dst_qid_h) ; 
    Kokkos::deep_copy(src_elem,src_elem_h) ; 
    Kokkos::deep_copy(dst_elem,dst_elem_h) ; 
    Kokkos::deep_copy(ic, ic_h) ; 

    gpu_task_t task{} ;

    copy_from_cbuf_op<elem_kind,decltype(data),decltype(cbuf)> functor{
        cbuf, data, 
        /*view qid*/dst_qid, /*cbuf qid*/src_qid, 
        /*view elem*/ dst_elem, /*cbuf elem*/ src_elem,  
        ic, VEC(nx,ny,nz), ngz
    } ; 
    
    Kokkos::DefaultExecutionSpace exec_space{stream} ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx/2,nv,bucket.size(),false /*add +ngz to nx ranges*/)
        } ; 
 
    task._run = [functor, policy] (view_alias_t alias) mutable {
        functor.set_data_ptr(alias) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE("Copy start.") ; 
        #endif 
        Kokkos::parallel_for("copy_ghostzones_c2v", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        GRACE_TRACE("Copy end") ; 
        #endif 
    };
    task.stream = &stream; 
    task.task_id = task_counter++ ; 

    task._dependencies.push_back(restrict_task_id) ; 
    task_list[restrict_task_id]->_dependents.push_back(task.task_id) ; 

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
    
    task._run = [functor, policy] (view_alias_t alias) mutable {
        functor.set_data_ptr(alias) ; 
        GRACE_TRACE("Fill phys start") ; 
        Kokkos::parallel_for("fill_phys_ghostzones", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        #endif 
        GRACE_TRACE("Fill phys done") ; 
    };
    task.stream = &stream ; 
    task.task_id = task_counter++ ; 

    return std::move(task) ; 
}

template< amr::element_kind_t elem_kind >
gpu_task_t make_pack_task(
      bucket_t const& sb
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
    // construct pack task
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> pack_src_qid{"pack_src_qid", sb.size()}
                        , pack_dest_qid{"pack_dst_qid", sb.size()} ; 
    Kokkos::View<uint8_t*> pack_src_elem{"unpack_dst_eid", sb.size()}  ;
    auto pack_src_qid_h = Kokkos::create_mirror_view(pack_src_qid) ; 
    auto pack_dst_qid_h = Kokkos::create_mirror_view(pack_dest_qid) ; 
    auto pack_src_elem_h =  Kokkos::create_mirror_view(pack_src_elem) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d) -> std::tuple<size_t, size_t, uint8_t>  {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            return {std::get<0>(d),face.data.full.send_buffer_id, std::get<1>(d)} ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {std::get<0>(d), edge.data.full.send_buffer_id, std::get<1>(d)} ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
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
        exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx,nv,sb.size())
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
    // construct pack task
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> pack_src_qid{"pack_src_qid", sb.size()}
                        , pack_dest_qid{"pack_dst_qid", sb.size()} ; 
    Kokkos::View<uint8_t*> pack_src_elem{"unpack_dst_eid", sb.size()}  ;
    auto pack_src_qid_h = Kokkos::create_mirror_view(pack_src_qid) ; 
    auto pack_dst_qid_h = Kokkos::create_mirror_view(pack_dest_qid) ; 
    auto pack_src_elem_h =  Kokkos::create_mirror_view(pack_src_elem) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d) -> -> std::tuple<size_t, size_t, uint8_t>  {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            return {ghost_layer[std::get<0>(d)].cbuf_id,face.data.full.send_buffer_id, std::get<1>(d)} ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {ghost_layer[std::get<0>(d)].cbuf_id, edge.data.full.send_buffer_id, std::get<1>(d)} ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {ghost_layer[std::get<0>(d)].cbuf_id, corner.data.send_buffer_id, std::get<1>(d)} ;
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
        exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx/2,nv,sb.size())
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
    // this depends on restrict
    pack_task._dependencies.push_back(restrict_task_id) ; 
    task_list[restrict_task_id] -> _dependents.push_back(pack_task.task_id); 
    return pack_task ; 
}

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
    // construct unpack task 
    Kokkos::DefaultExecutionSpace exec_space{pup_stream} ; 
    Kokkos::View<size_t*> unpack_src_qid{"unpack_src_qid", rb.size()}
                            , unpack_dest_qid{"unpack_dst_qid", rb.size()} ; 
    Kokkos::View<uint8_t*> unpack_dest_elem{"unpack_src_eid", rb.size()}  ;
    auto unpack_src_qid_h = Kokkos::create_mirror_view(unpack_src_qid) ; 
    auto unpack_dst_qid_h = Kokkos::create_mirror_view(unpack_dest_qid) ; 
    auto unpack_dest_elem_h =  Kokkos::create_mirror_view(unpack_dest_elem) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d)-> std::tuple<size_t, size_t, uint8_t>  {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            return {face.data.full.recv_buffer_id,std::get<0>(d), std::get<1>(d)} ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {edge.data.full.recv_buffer_id, std::get<0>(d) , std::get<1>(d)} ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return { corner.data.recv_buffer_id, std::get<0>(d), std::get<1>(d)} ;
        }
    } ; 
    auto const set_task_id = [&] (gpu_task_desc_t const& d)
    {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[d.first()].faces[d.second()] ; 
            face.data.full.task_id = task_counter ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[d.first()].edges[d.second()] ; 
            edge.data.full.task_id = task_counter ;
        } else {
            auto& corner = ghost_layer[d.first()].corners[d.second()] ; 
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
        exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx,nv,rb.size())
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
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            // return cbuf_id, ghost_qid, face_id, child_id 
            return {ghost_layer[std::get<0>(d)].cbuf_id, face.data.full.recv_buffer_id, std::get<1>(d), face.child_id } ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {ghost_layer[std::get<0>(d)].cbuf_id, edge.data.full.recv_buffer_id, std::get<1>(d), edge.child_id } ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {ghost_layer[std::get<0>(d)].cbuf_id, corner.data.recv_buffer_id, std::get<1>(d), 0 } ;
        }
    } ; 
    auto const set_task_id = [&] (gpu_task_desc_t const& d)
    {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[d.first()].faces[d.second()] ; 
            face.data.full.task_id = task_counter ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[d.first()].edges[d.second()] ; 
            edge.data.full.task_id = task_counter ;
        } else {
            auto& corner = ghost_layer[d.first()].corners[d.second()] ; 
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
        exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx/2,nv,rb.size(), true /*extend loops by ngz*/)
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

template< amr::element_kind_t elem_kind, typename view_t >
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
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            // return cbuf_id, ghost_qid, face_id, child_id 
            return {std::get<0>(d), face.data.hanging.recv_buffer_id[std::get<2>(d)], std::get<1>(d), std::get<2>(d) } ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {std::get<0>(d), edge.data.hanging.recv_buffer_id[std::get<2>(d)], std::get<1>(d), std::get<2>(d) } ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {std::get<0>(d), corner.data.recv_buffer_id, std::get<1>(d), 0 } ;
        }
    } ; 
    auto const set_task_id = [&] (gpu_hanging_task_desc_t const& d)
    {
        if constexpr ( elem_kind == element_kind_t::FACE ) {
            auto& face = ghost_layer[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.hanging.task_id[std::get<2>(d)] = task_counter ;
        } else if constexpr (elem_kind == element_kind_t::EDGE) {
            auto& edge = ghost_layer[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.hanging.task_id[std::get<2>(d)] = task_counter ;
        } else {
            auto& corner = ghost_layer[std::get<0>(d)].corners[std::get<1>(d)] ; 
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
        exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx/2,nv,rb.size(), false /*extend loops by ngz*/)
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

template<amr::element_kind_t EK, typename task_bucket_t, typename KernelFactory>
void insert_task_for_kind(task_bucket_t& copy_kernels,
                          KernelFactory&& make_task,
                          std::vector<std::unique_ptr<task_t>>& task_list)
{
    task_list.push_back(
        std::make_unique<gpu_task_t>(
            make_task(copy_kernels[static_cast<size_t>(EK)])
        )
    );
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

    for( int r=0; r<parallel::mpi_comm_size(); ++r) {
        // Pack tasks
        std::apply([&](auto... EK){
            (insert_task_for_kind<EK>(pack_kernels[r],
                [&](auto const& bucket){
                    return make_pack_task<EK>(bucket, ghost_array, r, data,
                                              send_buf, send_task_id, pup_stream,
                                              VEC(nx,ny,nz), ngz, nv,
                                              task_counter, task_list);
                }, task_list), ...);
        }, elem_kinds);
        // Pack to coarse buffer
        std::apply([&](auto... EK){
            (insert_task_for_kind<EK>(pack_to_cbuf_kernels[r],
                [&](auto const& bucket){
                    return make_pack_to_cbuf_task<EK>(bucket, ghost_array, r,
                                                      coarse_buffers, send_buf,
                                                      send_task_id, pup_stream,
                                                      VEC(nx,ny,nz), ngz, nv,
                                                      task_counter, restrict_task_id,
                                                      task_list);
                }, task_list), ...);
        }, elem_kinds);
        { 
            auto& b = pack_kernels[r] ; 
        
            // simple pack 
            if( b[element_kind_t::FACE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-pack task (tid {}). Number of faces processed {}.", task_counter, b[element_kind_t::FACE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_pack_task<element_kind_t::FACE>(
                            b[element_kind_t::FACE], ghost_array, r, data, send_buf, send_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            } 
            if( b[element_kind_t::EDGE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-pack task (tid {}). Number of edges processed {}.", task_counter, b[element_kind_t::EDGE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_pack_task<element_kind_t::EDGE>(
                            b[element_kind_t::EDGE], ghost_array, r, data, send_buf, send_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
            if( b[element_kind_t::CORNER].size() > 0 ) {
                GRACE_TRACE("Recording GPU-pack task (tid {}). Number of corners processed {}.", task_counter, b[element_kind_t::CORNER].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_pack_task<element_kind_t::CORNER>(
                            b[element_kind_t::CORNER], ghost_array, r, data, send_buf, send_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
        }
        { // pack_to_cbuf 
            auto& b = pack_to_cbuf_kernels[r] ; 
        
            
            if( b[element_kind_t::FACE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-pack task (tid {}). Number of faces processed {}.", task_counter, b[element_kind_t::FACE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_pack_to_cbuf_task<element_kind_t::FACE>(
                            b[element_kind_t::FACE], ghost_array, r, coarse_buffers, send_buf, send_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, restrict_task_id, task_list 
                        )
                    )
                ) ;
            } 
            if( b[element_kind_t::EDGE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-pack task (tid {}). Number of edges processed {}.", task_counter, b[element_kind_t::EDGE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_pack_to_cbuf_task<element_kind_t::EDGE>(
                            b[element_kind_t::EDGE], ghost_array, r, coarse_buffers, send_buf, send_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, restrict_task_id, task_list 
                        )
                    )
                ) ;
            }
            if( b[element_kind_t::CORNER].size() > 0 ) {
                GRACE_TRACE("Recording GPU-pack task (tid {}). Number of corners processed {}.", task_counter, b[element_kind_t::CORNER].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_pack_to_cbuf_task<element_kind_t::CORNER>(
                            b[element_kind_t::CORNER], ghost_array, r, coarse_buffers, send_buf, send_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, restrict_task_id, task_list 
                        )
                    )
                ) ;
            }
        }
        { // simple unpack 
            auto& b = unpack_kernels[r] ;  
            if ( b[element_kind_t::FACE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of faces processed {}.", task_counter, b[element_kind_t::FACE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_unpack_task<element_kind_t::FACE>(
                            b[element_kind_t::FACE], ghost_array, r, data, recv_buf, recv_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
            if ( b[element_kind_t::EDGE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of edges processed {}.", task_counter, b[element_kind_t::EDGE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_unpack_to_cbuf_task<element_kind_t::EDGE>(
                            b[element_kind_t::EDGE], ghost_array, r, data, recv_buf, recv_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
            if ( b[element_kind_t::CORNER].size() > 0 ) {
                GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of corners processed {}.", task_counter, b[element_kind_t::CORNER].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_unpack_to_cbuf_task<element_kind_t::CORNER>(
                            b[element_kind_t::CORNER], ghost_array, r, data, recv_buf, recv_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
        }
        { // unpack to cbuf 
            auto& b = unpack_to_cbuf_kernels[r] ;  
            if ( b[element_kind_t::FACE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of faces processed {}.", task_counter, b[element_kind_t::FACE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_unpack_to_cbuf_task<element_kind_t::FACE>(
                            b[element_kind_t::FACE], ghost_array, r, coarse_buffers, recv_buf, recv_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
            if ( b[element_kind_t::EDGE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of edges processed {}.", task_counter, b[element_kind_t::EDGE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_unpack_to_cbuf_task<element_kind_t::EDGE>(
                            b[element_kind_t::EDGE], ghost_array, r, coarse_buffers, recv_buf, recv_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
            if ( b[element_kind_t::CORNER].size() > 0 ) {
                GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of corners processed {}.", task_counter, b[element_kind_t::CORNER].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_unpack_to_cbuf_task<element_kind_t::CORNER>(
                            b[element_kind_t::CORNER], ghost_array, r, coarse_buffers, recv_buf, recv_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
        }
        { // unpack from cbuf 
            auto& b = unpack_from_cbuf_kernels[r] ;  
            if ( b[element_kind_t::FACE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of faces processed {}.", task_counter, b[element_kind_t::FACE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_unpack_from_cbuf_task<element_kind_t::FACE>(
                            b[element_kind_t::FACE], ghost_array, r, coarse_buffers, recv_buf, recv_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
            if ( b[element_kind_t::EDGE].size() > 0 ) {
                GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of edges processed {}.", task_counter, b[element_kind_t::EDGE].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_unpack_from_cbuf_task<element_kind_t::EDGE>(
                            b[element_kind_t::EDGE], ghost_array, r, coarse_buffers, recv_buf, recv_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
            if ( b[element_kind_t::CORNER].size() > 0 ) {
                GRACE_TRACE("Recording GPU-unpack task (tid {}). Number of corners processed {}.", task_counter, b[element_kind_t::CORNER].size()) ; 
                task_list.push_back(
                    std::make_unique<gpu_task_t>(
                        make_unpack_from_cbuf_task<element_kind_t::CORNER>(
                            b[element_kind_t::CORNER], ghost_array, r, coarse_buffers, recv_buf, recv_task_id, pup_stream, 
                            VEC(nx,ny,nz), ngz, nv, task_counter, task_list 
                        )
                    )
                ) ;
            }
        }
        
    } // for r in ranks 
}; 

mpi_task_t make_mpi_send_task(
      std::size_t rr 
    , amr::ghost_array_t send_buf 
    , std::vector<std::size_t> const& send_rank_offsets 
    , std::vector<std::size_t> const& send_rank_sizes
    , task_id_t& task_counter 
)
{
    GRACE_TRACE("Registering MPI Send task (tid {}).\n    Send to Rank {} {} elements, offset {}", task_counter, rr, send_rank_sizes[rr], send_rank_offsets[rr]) ; 

    ASSERT(send_rank_offsets[rr] + send_rank_sizes[rr] <= send_buf.size(), "Send out-of-bounds" ) ; 

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
    , amr::ghost_array_t recv_buf 
    , std::vector<std::size_t> const& recv_rank_offsets 
    , std::vector<std::size_t> const& recv_rank_sizes
    , task_id_t& task_counter 
)
{
    GRACE_TRACE("Registering MPI Receive task (tid {}).\n    Receive from Rank {} {} elements, offset {}", task_counter, rr, recv_rank_sizes[rr], recv_rank_offsets[rr]) ; 

    ASSERT(recv_rank_offsets[rr] + recv_rank_sizes[rr] <= recv_buf.size(), "Receive out-of-bounds" ) ; 

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


// FIXME (?) right now this creates a single task 
task_id_t insert_restriction_tasks(
    std::unordered_set<size_t> const& cbuf_qid,
    std::vector<quad_neighbors_descriptor_t>& ghost_layer,
    grace::var_array_t<GRACE_NSPACEDIM> state, 
    grace::var_array_t<GRACE_NSPACEDIM> coarse_buffers,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    Kokkos::View<size_t*> quad_id_d("restrict_qid", cbuf_qid.size())
                        , cbuf_id_d("restrict_cbufid", cbuf_qid.size()) ; 
    auto quad_id_h = Kokkos::create_mirror_view(quad_id_d) ; 
    auto cbuf_id_h = Kokkos::create_mirror_view(cbuf_id_d) ; 

    size_t i{0UL} ; 
    for( auto const& qid: cbuf_qid) {
        quad_id_h(i) = qid ; 
        cbuf_id_h(i) = ghost_layer[qid].cbuf_id ; 
        i+=1UL ; 
    }
    Kokkos::deep_copy(quad_id_d,quad_id_h) ;
    Kokkos::deep_copy(cbuf_id_d,cbuf_id_h) ;

    gpu_task_t task{} ;

    restrict_op functor(
        state, coarse_buffers, quad_id_d, cbuf_id_d, ngz
    ) ; 

    Kokkos::DefaultExecutionSpace exec_space{stream} ;

    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, {nx,nx,nx, nv, cbuf_qid.size()}
        } ;

    task._run = [functor, policy] (view_alias_t alias) mutable {
        functor.set_data_ptr(alias) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE("Restrict start.") ; 
        #endif 
        Kokkos::parallel_for("restrict_to_cbufs", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        GRACE_TRACE("Copy end") ; 
        #endif 
    };
    task.stream = &stream; 
    task.task_id = task_counter++ ; 

    taks_list.push_back(
        std::make_unique<gpu_task_t>(std::move(task))
    ) ; 
    return task.task_id ; 
}


void insert_copy_tasks(
    std::vector<quad_neighbors_descriptor_t>& ghost_layer,
    bucket_t& copy_kernels,
    hang_bucket_t& copy_from_cbuf_kernels,
    hang_bucket_t& copy_to_cbuf_kernels,
    grace::var_array_t<GRACE_NSPACEDIM> state, 
    grace::var_array_t<GRACE_NSPACEDIM> coarse_buffers, 
    device_stream_t& stream,
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    task_id_t const& restrict_task_id,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    constexpr amr::element_kind_t kinds[] = {
        element_kind_t::FACE,
        element_kind_t::EDGE,
        element_kind_t::CORNER
    };

    std::apply([&](auto... EK){
        // fold over element kinds
        ((insert_task_for_kind<EK>(
            copy_kernels,
            [&](auto const& bucket){
                return make_gpu_copy_task<EK>(bucket, ghost_layer, state, stream, VEC(nx,ny,nz), ngz, nv, task_counter);
            },
            task_list
        )), ...);
    }, kinds);
    std::apply([&](auto... EK){
        // fold over element kinds
        ((insert_task_for_kind<EK>(
            copy_to_cbuf_kernels,
            [&](auto const& bucket){
                return make_gpu_copy_to_cbuf_task<EK>(bucket, ghost_layer, state, coarse_buffers, stream, VEC(nx,ny,nz), ngz, nv, task_counter);
            },
            task_list
        )), ...);
    }, kinds);
    std::apply([&](auto... EK){
        // fold over element kinds
        ((insert_task_for_kind<EK>(
            copy_from_cbuf_kernels,
            [&](auto const& bucket){
                return make_gpu_copy_from_cbuf_task<EK>(bucket, ghost_layer, state, coarse_buffers,stream, VEC(nx,ny,nz), ngz, nv, task_counter, restrict_task_id, task_list);
            },
            task_list
        )), ...);
    }, kinds);
}


} /* namespace grace */

#endif /*GRACE_AMR_GHOSTZONE_KERNELS_TASK_FACTORY_HH*/ 