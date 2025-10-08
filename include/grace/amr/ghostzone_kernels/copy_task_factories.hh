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
 * GNU General Public License for more grace::detail s.
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
#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <unordered_set>
#include <vector>
#include <numeric>

#ifndef GRACE_AMR_GHOSTZONE_COPY_KERNELS_TASK_FACTORY_HH
#define GRACE_AMR_GHOSTZONE_COPY_KERNELS_TASK_FACTORY_HH

namespace grace {

/**
 * @brief Construct a <code>gpu_task_t</code> that copies data from physical cells to
 *        ghostzone across an element of type <code>elem_kind</code>.
 * 
 * @tparam elem_kind Kind of element where data is copied.
 * @param bucket Descriptors of processed interfaces.
 * @param ghost_array Array of neighbor descriptors.
 * @param data Data.
 * @param stream Device stream.
 * @param ngz Number of ghost-cells.
 * @param nv Number of variables.
 * @param task_counter Current task counter.
 * @return gpu_task_t encapsulating a kernel that copies data across interfaces at the same ref-level.
 */
template< amr::element_kind_t elem_kind, var_staggering_t stag >
gpu_task_t make_gpu_copy_task(
      std::vector<gpu_task_desc_t> const& bucket
    , std::vector<quad_neighbors_descriptor_t>& ghost_array
    , grace::var_array_t data 
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
    auto src_elem_h = Kokkos::create_mirror_view(src_elem) ; 
    auto dst_elem_h = Kokkos::create_mirror_view(dst_elem) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d) -> std::tuple<size_t,size_t,uint8_t,uint8_t> {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            return {face.data.full.quad_id, std::get<0>(d), face.face, std::get<1>(d)} ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {edge.data.full.quad_id, std::get<0>(d), edge.edge, std::get<1>(d)} ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {corner.data.quad_id, std::get<0>(d), corner.corner, std::get<1>(d)} ;
        }
    } ; 

    auto const set_task_id = [&] (gpu_task_desc_t const& d)
    {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.full.task_id[stag] = task_counter ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.full.task_id[stag] = task_counter ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id[stag] = task_counter ;
        }
    } ; 
    
    int i{0} ; 
    for( auto const& d: bucket ) { 
        if (elem_kind == amr::EDGE and std::get<0>(d) == 5 and std::get<1>(d) == 5) {
            GRACE_TRACE("Here! Tid {}", task_counter ) ; 
        }
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

    amr::copy_op<elem_kind,decltype(data)> functor{
        data, src_qid, dst_qid, src_elem, dst_elem, VEC(nx,ny,nz), ngz
    } ; 
    
    Kokkos::DefaultExecutionSpace exec_space{stream} ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx,nv,bucket.size())
        } ; 
 
    task._run = [functor, policy] (view_alias_t alias) mutable {
        functor.template set_data_ptr<stag>(alias) ; 
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
/**
 * @brief Create a task to copy data from a coarse quadrant to a coarse buffer.
 * 
 * @tparam elem_kind Element kind across which data is copied.
 * @param bucket Descriptors of processed interfaces.
 * @param ghost_array Array of neighbor descriptors.
 * @param data Data.
 * @param cbuf Coarse buffers.
 * @param stream Device stream.
 * @param ngz Number of ghost-zones.
 * @param nv Number of variables.
 * @param task_counter Current task count.
 * @return gpu_task_t Encapsulating a kernel that copies data to coarse buffers.
 */
template< amr::element_kind_t elem_kind, var_staggering_t stag >
gpu_task_t make_gpu_copy_to_cbuf_task(
      std::vector<gpu_task_desc_t> const& bucket
    , std::vector<quad_neighbors_descriptor_t>& ghost_array
    , grace::var_array_t data 
    , grace::var_array_t cbuf 
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
    auto src_elem_h = Kokkos::create_mirror_view(src_elem) ; 
    auto dst_elem_h = Kokkos::create_mirror_view(dst_elem) ; 
    auto ic_h = Kokkos::create_mirror_view(ic) ; 

    auto const get_interface_info = [&] (gpu_task_desc_t const& d) -> std::tuple<size_t, size_t,uint8_t,uint8_t, uint8_t> {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            //return src qid dst qid src face dst face child id
            return {face.data.full.quad_id, ghost_array[std::get<0>(d)].cbuf_id,  face.face, std::get<1>(d), face.child_id} ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            return {edge.data.full.quad_id, ghost_array[std::get<0>(d)].cbuf_id,  edge.edge, std::get<1>(d), edge.child_id} ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            return {corner.data.quad_id, ghost_array[std::get<0>(d)].cbuf_id,  corner.corner, std::get<1>(d), 0} ;
        }
    } ; 

    auto const set_task_id = [&] (gpu_task_desc_t const& d)
    {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.full.task_id[stag] = task_counter ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.full.task_id[stag] = task_counter ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id[stag] = task_counter ;
        }
    } ; 

    int i{0} ; 
    for( auto const& d: bucket ) {
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

    amr::copy_to_cbuf_op<elem_kind,decltype(data),decltype(cbuf)> functor{
        data, cbuf, src_qid, dst_qid, src_elem, dst_elem, ic, VEC(nx,ny,nz), ngz
    } ; 
    
    Kokkos::DefaultExecutionSpace exec_space{stream} ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, get_iter_range<elem_kind>(ngz,nx/2,nv,bucket.size(),true /*add +ngz to nx ranges*/)
        } ; 
 
    task._run = [functor, policy] (view_alias_t alias) mutable {
        functor.template set_data_ptr<stag>(alias) ; 
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
/**
 * @brief Create a task to copy data from a coarse buffer quadrant to a coarse quadrant.
 * 
 * @tparam elem_kind Element kind across which data is copied.
 * @param bucket Descriptors of processed interfaces.
 * @param ghost_array Array of neighbor descriptors.
 * @param data Data.
 * @param cbuf Coarse buffers.
 * @param stream Device stream.
 * @param ngz Number of ghost-zones.
 * @param nv Number of variables.
 * @param task_counter Current task count.
 * @param restrict_task_id Task index of restriction.
 * @param task_list List of tasks.
 * @return gpu_task_t A task encapsulating a kernel that copies data from coarse buffers.
 * NB: this task depends on the restriction of data in coarse buffers.
 */
template< amr::element_kind_t elem_kind, var_staggering_t stag >
gpu_task_t make_gpu_copy_from_cbuf_task(
      std::vector<gpu_hanging_task_desc_t> const& bucket
    , std::vector<quad_neighbors_descriptor_t>& ghost_array
    , grace::var_array_t data 
    , grace::var_array_t cbuf 
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
    auto src_elem_h = Kokkos::create_mirror_view(src_elem) ; 
    auto dst_elem_h = Kokkos::create_mirror_view(dst_elem) ; 
    auto ic_h = Kokkos::create_mirror_view(ic) ; 

    auto const get_interface_info = [&] (gpu_hanging_task_desc_t const& d) -> std::tuple<size_t, size_t,uint8_t,uint8_t, uint8_t> {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ;
            auto qid =  face.data.hanging.quad_id[std::get<2>(d)] ; 
            // return src_id, dst_id, src_e, dst_e, child_id 
            return {ghost_array[qid].cbuf_id,std::get<0>(d), face.face, std::get<1>(d), std::get<2>(d)} ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            auto qid =  edge.data.hanging.quad_id[std::get<2>(d)] ; 
            return {ghost_array[qid].cbuf_id,std::get<0>(d), edge.edge, std::get<1>(d), std::get<2>(d)} ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            auto qid = corner.data.quad_id ; 
            return {ghost_array[qid].cbuf_id, std::get<0>(d), corner.corner, std::get<1>(d), 0} ;
        }
    } ; 

    auto const set_task_id = [&] (gpu_hanging_task_desc_t const& d)
    {
        if constexpr ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.hanging.task_id[std::get<2>(d)][stag] = task_counter ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.hanging.task_id[std::get<2>(d)][stag] = task_counter ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id[stag] = task_counter ;
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

    amr::copy_from_cbuf_op<elem_kind,decltype(data),decltype(cbuf)> functor{
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
        functor.template set_data_ptr<stag>(alias) ; 
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
/**
 * @brief Insert copy tasks in the task list. These are all local copies amongst quadrants at the same 
 *        refinement level and/or coarse quadrants and coarse buffers. 
 * @ingroup amr
 * @param ghost_array Array of neighbor descriptors.
 * @param copy_kernels List of descriptor for simple copy tasks.
 * @param copy_from_cbuf_kernels List of descriptor for copy-from-cbuf tasks.
 * @param copy_to_cbuf_kernels List of descriptor for copy-to-cbuf tasks.
 * @param state Data.
 * @param coarse_buffers Coarse buffers.
 * @param stream Device streams.
 * @param ngz Number of ghost-cells.
 * @param nv Number of variables.
 * @param task_counter Current task counter.
 * @param restrict_task_id Identifier of restriction task.
 * @param task_list List of tasks.
 */
template< var_staggering_t stag >
void insert_copy_tasks(
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    bucket_t& copy_kernels,
    hang_bucket_t& copy_from_cbuf_kernels,
    bucket_t& copy_to_cbuf_kernels,
    grace::var_array_t state, 
    grace::var_array_t coarse_buffers, 
    device_stream_t& stream,
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    task_id_t const& restrict_task_id,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{

    #define MAKE_COPY(kind)\
    if(copy_kernels[static_cast<size_t>(kind)].size()>0)\
    task_list.push_back( \
        std::make_unique<gpu_task_t>( \
            make_gpu_copy_task<kind,stag>(copy_kernels[static_cast<size_t>(kind)], ghost_array, state, stream, VEC(nx,ny,nz), ngz, nv, task_counter) \
        ) \
    ) 
    #define MAKE_COPY_TO_CBUF(kind)\
    if(copy_to_cbuf_kernels[static_cast<size_t>(kind)].size()>0)\
    task_list.push_back( \
        std::make_unique<gpu_task_t>( \
            make_gpu_copy_to_cbuf_task<kind,stag>(copy_to_cbuf_kernels[static_cast<size_t>(kind)], ghost_array, state, coarse_buffers, stream, VEC(nx,ny,nz), ngz, nv, task_counter) \
        ) \
    ) 

    #define MAKE_COPY_FROM_CBUF(kind)\
    if(copy_from_cbuf_kernels[static_cast<size_t>(kind)].size()>0)\
    task_list.push_back( \
        std::make_unique<gpu_task_t>( \
            make_gpu_copy_from_cbuf_task<kind,stag>(copy_from_cbuf_kernels[static_cast<size_t>(kind)], ghost_array, state, coarse_buffers,stream, VEC(nx,ny,nz), ngz, nv, task_counter, restrict_task_id, task_list) \
        ) \
    ) 

    #define MAKE_TASKS(kind) \
    MAKE_COPY(kind); \
    MAKE_COPY_TO_CBUF(kind) ; \
    MAKE_COPY_FROM_CBUF(kind) 

    MAKE_TASKS(amr::FACE) ; 
    MAKE_TASKS(amr::EDGE) ; 
    MAKE_TASKS(amr::CORNER) ; 

    #undef MAKE_COPY 
    #undef MAKE_COPY_TO_CBUF
    #undef MAKE_COPY_FROM_CBUF
    #undef MAKE_TASKS
}

} /* namespace grace */

#endif 