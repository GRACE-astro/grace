/**
 * @file restriction_task_factories.hh
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
#include <grace/amr/ghostzone_kernels/restrict_kernels.hh>
#include <grace/amr/ghostzone_kernels/pack_unpack_kernels.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>
#include <grace/amr/ghostzone_kernels/pr_helpers.hh>

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <unordered_set>
#include <vector>
#include <numeric>

#ifndef GRACE_AMR_GHOSTZONE_KERNELS_RESTRICTION_TASK_FACTORY_HH
#define GRACE_AMR_GHOSTZONE_KERNELS_RESTRICTION_TASK_FACTORY_HH

namespace grace {

// FIXME (?) right now this creates a single task 
template< var_staggering_t stag >
task_id_t insert_restriction_tasks(
    std::unordered_set<size_t> const& cbuf_qid,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t state, 
    grace::var_array_t coarse_buffers,
    std::vector<size_t> const& varlist_lo,
    std::vector<size_t> const& varlist_ho,
    std::vector<double> const& ho_restrict_coeffs,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    using restrict_lo_op = second_order_restrict_op;
    using restrict_ho_op = fourth_order_restrict_op;


    GRACE_TRACE("Recording GPU-restrict task (tid {}) number of quadrants {}", task_counter, cbuf_qid.size()) ;
    Kokkos::View<size_t*> quad_id_d("restrict_qid", cbuf_qid.size())
                        , cbuf_id_d("restrict_cbufid", cbuf_qid.size()) 
                        , varlist_lo_d("restrict_vlist_lo", varlist_lo.size())
                        , varlist_ho_d("restrict_vlist_ho", varlist_ho.size()); 
    Kokkos::View<double*> ho_restrict_coeffs_d("ho_restrict_coeffs", ho_restrict_coeffs.size());
    deep_copy_vec_to_view(ho_restrict_coeffs_d,ho_restrict_coeffs);

    deep_copy_vec_to_view(varlist_lo_d,varlist_lo);
    deep_copy_vec_to_view(varlist_ho_d,varlist_ho);

    auto quad_id_h = Kokkos::create_mirror_view(quad_id_d) ; 
    auto cbuf_id_h = Kokkos::create_mirror_view(cbuf_id_d) ; 
    
    size_t i{0UL} ; 
    for( auto const& qid: cbuf_qid) {
        quad_id_h(i) = qid ; 
        cbuf_id_h(i) = ghost_array[qid].cbuf_id ; 
        i+=1UL ; 
    }
    Kokkos::deep_copy(quad_id_d,quad_id_h) ;
    Kokkos::deep_copy(cbuf_id_d,cbuf_id_h) ;

    gpu_task_t task{} ;

    amr::restrict_op<restrict_lo_op,decltype(state)> 
    functor(
        state, coarse_buffers, quad_id_d, cbuf_id_d, varlist_lo_d, restrict_lo_op{}, ngz
    ) ; 

    amr::restrict_op<restrict_ho_op,decltype(state)> 
    functor_ho(
        state, coarse_buffers, quad_id_d, cbuf_id_d, varlist_ho_d, restrict_ho_op(ho_restrict_coeffs_d,nx,ny,nz,ngz), ngz
    ) ;

    Kokkos::DefaultExecutionSpace exec_space{stream} ;

    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, {nx/2,nx/2,nx/2, varlist_lo.size(), cbuf_qid.size()}
        } ;
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy_ho{
            exec_space, {0,0,0,0,0}, {nx/2,nx/2,nx/2, varlist_ho.size(), cbuf_qid.size()}
        } ;
        
    task._run = [functor, functor_ho, policy, policy_ho] (view_alias_t alias) mutable {
        functor.template set_data_ptr<stag>(alias) ; 
        functor_ho.template set_data_ptr<stag>(alias) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE_DBG("Restrict start.") ; 
        #endif 
        Kokkos::parallel_for("restrict_to_cbufs", policy, functor) ;
        Kokkos::parallel_for("restrict_to_cbufs_high_order", policy_ho, functor_ho) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        GRACE_TRACE_DBG("Restrict end") ; 
        #endif 
    };
    task.stream = &stream; 
    task.task_id = task_counter++ ; 

    task_list.push_back(
        std::make_unique<gpu_task_t>(std::move(task))
    ) ; 
    return task_list.back()->task_id ; 
}

template< var_staggering_t stag >
task_id_t insert_div_preserving_restriction_tasks(
    std::unordered_set<size_t> const& cbuf_qid,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t state, 
    grace::var_array_t coarse_buffers,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    static_assert( stag == STAG_FACEX or stag == STAG_FACEY or stag == STAG_FACEZ,
                   "Invalid staggering in insert_div_preserving_restriction_tasks, only face-staggerings supported."  );
    GRACE_TRACE("Recording GPU-restrict task (tid {}) number of quadrants {}", task_counter, cbuf_qid.size()) ;
    Kokkos::View<size_t*> quad_id_d("restrict_qid", cbuf_qid.size())
                        , cbuf_id_d("restrict_cbufid", cbuf_qid.size()) ; 
    auto quad_id_h = Kokkos::create_mirror_view(quad_id_d) ; 
    auto cbuf_id_h = Kokkos::create_mirror_view(cbuf_id_d) ; 

    size_t i{0UL} ; 
    for( auto const& qid: cbuf_qid) {
        quad_id_h(i) = qid ; 
        cbuf_id_h(i) = ghost_array[qid].cbuf_id ; 
        i+=1UL ; 
    }
    Kokkos::deep_copy(quad_id_d,quad_id_h) ;
    Kokkos::deep_copy(cbuf_id_d,cbuf_id_h) ;

    gpu_task_t task{} ;
    constexpr int stag_dir = (stag == STAG_FACEX ? 0 : (stag == STAG_FACEY ? 1 : 2)) ; 
    amr::div_free_restrict_op<stag_dir, decltype(state)> functor(
        state, coarse_buffers, quad_id_d, cbuf_id_d, ngz
    ) ; 

    Kokkos::DefaultExecutionSpace exec_space{stream} ;
    int loff[3] = {stag_dir==0,stag_dir==1,stag_dir==2} ; 
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, {nx/2+loff[0],nx/2+loff[1],nx/2+loff[2], nv, cbuf_qid.size()}
        } ;

    task._run = [functor, policy] (view_alias_t alias) mutable {
        functor.template set_data_ptr<stag>(alias) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE_DBG("Restrict start.") ; 
        #endif 
        Kokkos::parallel_for("restrict_to_cbufs_stag", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        GRACE_TRACE_DBG("Restrict end") ; 
        #endif 
    };
    task.stream = &stream; 
    task.task_id = task_counter++ ; 

    task_list.push_back(
        std::make_unique<gpu_task_t>(std::move(task))
    ) ; 
    return task_list.back()->task_id ; 
}

/**
 * @brief Get iter policy for gz-restrict
 * \cond grace_detail
 * @tparam elem_kind Kind of element being filled
 * @param stream Device stream
 * @param n Number of cells
 * @param nv Number of variables
 * @param nq Number of quadrants
 * @return auto 
 */
template< amr::element_kind_t elem_kind >
auto get_iter_policy(
    device_stream_t& stream, size_t n, size_t nv, size_t nq, bool stag_loop=false
) {
    using namespace amr ; 
    using namespace Kokkos ; 
    int off = static_cast<int>(stag_loop) ; 
    if constexpr ( elem_kind == FACE ) {
        return MDRangePolicy<Rank<4>, ghost_restrict_face_tag>(
            DefaultExecutionSpace{stream},
            {0,0,0,0}, {n/2+off,n/2+off,nv,nq}
        ) ; 
    } else if constexpr (elem_kind == EDGE) {
        return MDRangePolicy<Rank<3>, ghost_restrict_edge_tag>(
            DefaultExecutionSpace{stream},
            {0,0,0}, {n/2+off,nv,nq}
        ) ; 
    } else {
        return MDRangePolicy<Rank<2>, ghost_restrict_corner_tag>(
            DefaultExecutionSpace{stream},
            {0,0}, {nv,nq}
        ) ; 
    }
} 


template< amr::element_kind_t elem_kind, var_staggering_t stag >
void make_gpu_restrict_gz_task(
    std::vector<gpu_task_desc_t> const& bucket,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t state, 
    grace::var_array_t coarse_buffers,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
) {
    if (bucket.size()==0) return ;  
    GRACE_TRACE("Recording GPU-ghostzone-restrict task (tid {}), number of elements {}", task_counter, bucket.size()) ; 
    using namespace amr ;
    Kokkos::View<size_t*> qid("qid_restrict", bucket.size())
                        , cbuf_qid("cbuf_qid_restrict", bucket.size()) ; 
    Kokkos::View<uint8_t*> eid("eid_restrict", bucket.size()) ; 

    auto qid_h = Kokkos::create_mirror_view(qid) ; 
    auto cbuf_qid_h = Kokkos::create_mirror_view(cbuf_qid) ; 
    auto eid_h = Kokkos::create_mirror_view(eid) ; 
    
    std::unordered_set<task_id_t> dependencies ; 
    auto insert_dependency = [&] (task_id_t tid) {
        if ( tid == UNSET_TASK_ID ) {
            GRACE_TRACE_DBG("Unset task-id in gz_restrict. Element kind: {}", static_cast<int>(elem_kind)) ; 
            ERROR("Unset task_id") ; 
        }
        if ( tid != task_counter ) {
            dependencies.insert(tid) ; 
        } 
    } ; 
    auto write_back_tid = [&](gpu_task_desc_t const& d) {
        if constexpr (elem_kind == FACE) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            if ( face.level_diff == level_diff_t::FINER ) {
                for(int ic=0; ic<P4EST_CHILDREN/2; ++ic) face.data.hanging.task_id[ic][stag] = task_counter; 
            } else {
                face.data.full.task_id[stag] = task_counter; 
            }
        } else if constexpr (elem_kind == EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            if ( edge.level_diff == level_diff_t::FINER ) {
                for(int ic=0; ic<2; ++ic) edge.data.hanging.task_id[ic][stag] = task_counter; 
            } else {
                edge.data.full.task_id[stag] = task_counter; 
            }
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ;
            corner.data.task_id[stag] = task_counter;  
        }
    } ; 

    auto get_info = [&] (gpu_task_desc_t const& d) -> std::tuple<size_t, size_t, uint8_t > {
        if constexpr (elem_kind == FACE) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            GRACE_TRACE_DBG("Inserting dependency FACE, qid {} fid {}", std::get<0>(d), std::get<1>(d)) ;
            ASSERT(face.level_diff != FINER, "In gz-restrict, FINER interfaces are forbidden by 2:1 balance.") ; 
            insert_dependency(face.data.full.task_id[stag]) ; 
            return {std::get<0>(d), ghost_array[std::get<0>(d)].cbuf_id, std::get<1>(d) } ; 
        } else if constexpr (elem_kind == EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            GRACE_TRACE_DBG("Inserting dependency EDGE, qid {} eid {}, edge kind {} level diff {} filled {}", std::get<0>(d), std::get<1>(d), static_cast<int>(edge.kind), static_cast<int>(edge.level_diff), edge.filled) ;
            ASSERT(edge.filled, "Edge passed to restrict_gz is virtual.") ; 
            ASSERT(edge.level_diff != FINER, "In gz-restrict, FINER interfaces are forbidden by 2:1 balance.") ; 
            insert_dependency(edge.data.full.task_id[stag]) ;
            return {std::get<0>(d), ghost_array[std::get<0>(d)].cbuf_id, std::get<1>(d) } ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            GRACE_TRACE_DBG("Inserting dependency CORNER, qid {} cid {}", std::get<0>(d), std::get<1>(d)) ;
            ASSERT(corner.filled, "Corner passed to restrict_gz is virtual.") ; 
            ASSERT(corner.level_diff != FINER, "In gz-restrict, FINER interfaces are forbidden by 2:1 balance.") ; 
            insert_dependency(corner.data.task_id[stag]) ; 
            return {std::get<0>(d), ghost_array[std::get<0>(d)].cbuf_id, std::get<1>(d) } ;
        }
    } ; 

    size_t i{0UL} ; 
    for( auto const& d: bucket ) {
        auto [_qid,_cid,_eid] = get_info(d) ;
        if ( _qid == 30 ) {
            GRACE_TRACE_DBG("qid 30 cid {}", _cid) ; 
        }
        qid_h(i) = _qid ; 
        cbuf_qid_h(i) = _cid ; 
        eid_h(i) = _eid ; 
 
        write_back_tid(d) ; 
        i+= 1UL ;
    }

    Kokkos::deep_copy(qid, qid_h) ; 
    Kokkos::deep_copy(cbuf_qid, cbuf_qid_h) ; 
    Kokkos::deep_copy(eid,eid_h ) ; 
  
    // here we could try to split deps, might be a good 
    // optimization knob. For now simplest thing is to 
    // create a single task (FIXME?)
    gpu_task_t task {} ; 
    if constexpr ( stag == STAG_CENTER ) {
        ghost_restrict_op functor{
            state, coarse_buffers, qid, cbuf_qid, eid, nx, ngz
        } ; 

        // the rank of iterations depends on the element kind 
        auto policy = get_iter_policy<elem_kind>(stream,nx,nv,bucket.size()) ; 

        task._run = [functor,policy] (view_alias_t alias) mutable {
            GRACE_TRACE_DBG("GZ restrict start") ; 
            functor.template set_data_ptr<stag>(alias) ; 
            Kokkos::parallel_for("ghostzone_restrict", policy, functor) ; 
            #ifdef INSERT_FENCE_DEBUG_TASKS_
            Kokkos::fence() ; 
            GRACE_TRACE_DBG("GZ restrict done") ; 
            #endif 
        } ; 
    } else {
        constexpr int stag_dir = (stag == STAG_FACEX ? 0 : (stag == STAG_FACEY ? 1 : 2)) ; 
        div_free_ghost_restrict_op<stag_dir, decltype(state)> functor{
            state,coarse_buffers,qid,cbuf_qid,eid,nx,ngz,stag
        } ; 
        // the rank of iterations depends on the element kind 
        auto policy = get_iter_policy<elem_kind>(stream,nx,nv,bucket.size(),true /*stag_loop*/) ; 

        task._run = [functor,policy] (view_alias_t alias) mutable {
            GRACE_TRACE_DBG("GZ restrict start") ; 
            functor.template set_data_ptr<stag>(alias) ; 
            Kokkos::parallel_for("ghostzone_restrict_stag", policy, functor) ; 
            #ifdef INSERT_FENCE_DEBUG_TASKS_
            Kokkos::fence() ; 
            GRACE_TRACE_DBG("GZ restrict done") ; 
            #endif 
        } ; 
    }
    

    task.stream = &stream ; 
    task.task_id = task_counter++ ; 

    for ( auto const& tid: dependencies) {
        task._dependencies.push_back(tid) ; 
        task_list[tid]->_dependents.push_back(task.task_id) ; 
    }

    task_list.push_back(
        std::make_unique<gpu_task_t>(
            std::move(task)
        )
    ) ; 
}


/**
 * @brief Insert ghostzone restriction tasks in the task list.
 * 
 * @param cbuf_qid Quad-id to cbuf-id lookup table.
 * @param ghost_array Array of neighbor descriptors.
 * @param state State view.
 * @param coarse_buffers Coarse buffer view.
 * @param stream Device stream.
 * @param nx Number of cells per block.
 * @param ngz Number of ghost-zones.
 * @param nv Number of variables.
 * @param task_counter Current task counter.
 * @param task_list Task list.
 */
template< var_staggering_t stag >
void insert_ghost_restriction_tasks(
    std::unordered_set<size_t> const& cbuf_qid,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t state, 
    grace::var_array_t coarse_buffers,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
) {
    std::vector<gpu_task_desc_t> restrict_faces, restrict_edges, restrict_corners ;
    

    // this loop collects all the interfaces 
    // where restriction in the ghostzones needs
    // to happen. This corresponds to all elements 
    // adjacent to a prolongation interface which are 
    // not themselves being prolonged or touch a physical boundary.
    for (auto const& qid : cbuf_qid) {
        for(int8_t f=0; f<P4EST_FACES; ++f) {
            auto& face = ghost_array[qid].faces[f] ; 
            if (face.kind == interface_kind_t::PHYS) continue ;  
            if (!(face.level_diff == level_diff_t::COARSER)) continue ;  
            for( int ie=0; ie<4; ++ie) {
                auto e = amr::detail::f2e[f][ie] ; 
                auto& edge = ghost_array[qid].edges[e] ; 
                if ( ! edge.filled ) continue; 
                if ( edge.kind == interface_kind_t::PHYS) {
                    edge.data.phys.in_cbuf = true ;  
                    continue ; 
                }
                if (!(edge.level_diff == level_diff_t::COARSER)) {
                    restrict_edges.emplace_back(qid,e) ; 
                }
            } 
        }
        
        for( int8_t e=0; e<12; ++e){
            auto& edge = ghost_array[qid].edges[e] ; 
            bool need_neighbor_restrict = (!edge.filled) ; 
            if ( edge.filled) {
                need_neighbor_restrict |= (edge.level_diff == level_diff_t::COARSER) and (edge.kind != interface_kind_t::PHYS);
            }
            if ( qid == 30 and e == 11 ) {
                GRACE_TRACE_DBG("From in gz_restrict ctor, we need it! did we get it? {}", need_neighbor_restrict) ; 
            }
            if (!need_neighbor_restrict) continue ; 
            for( int iface=0; iface<2; ++iface) {
                auto f = amr::detail::e2f[e][iface] ; 
                auto& face = ghost_array[qid].faces[f] ; 
                if ( face.kind == interface_kind_t::PHYS) {
                    face.data.phys.in_cbuf=true ;  
                    continue ; 
                }
                if (!(face.level_diff == level_diff_t::COARSER)) restrict_faces.emplace_back(qid,f) ;
                
            }
            for( int ic=0; ic<2; ++ic) {
                auto c = amr::detail::e2c[e][ic] ; 
                auto& corner = ghost_array[qid].corners[c] ; 
                if ( ! corner.filled ) continue; 
                if ( corner.kind == interface_kind_t::PHYS) {
                    corner.phys.in_cbuf=true ;  
                    continue ; 
                }
                if (!(corner.level_diff == level_diff_t::COARSER)) restrict_corners.emplace_back(qid,c) ; 
            }
        }
        for( int8_t c=0; c<P4EST_CHILDREN; ++c){
            auto& corner = ghost_array[qid].corners[c] ; 
            bool need_neighbor_restrict = (!corner.filled) ; 
            if (corner.filled) {
                need_neighbor_restrict |= (corner.level_diff == level_diff_t::COARSER) and (corner.kind != interface_kind_t::PHYS);
            }
            if (!need_neighbor_restrict) continue ; 
            for( int ie=0; ie<3; ++ie) {
                auto e = amr::detail::c2e[c][ie] ; 
                auto& edge = ghost_array[qid].edges[e] ; 
                if ( ! edge.filled ) continue; 
                if ( edge.kind == interface_kind_t::PHYS) {
                    edge.data.phys.in_cbuf = true ;
                    continue ; 
                }  
                if (!(edge.level_diff == level_diff_t::COARSER)) restrict_edges.emplace_back(qid,e) ; 
            }
        }
    }
    // dedup buckets 
    auto const dedup = [&] (std::vector<gpu_task_desc_t>& vec) {
        if (vec.empty()) return ; 
        std::set<gpu_task_desc_t> s( vec.begin(), vec.end() );
        vec.assign( s.begin(), s.end() );
    } ; 
    dedup(restrict_faces) ; dedup(restrict_edges) ; dedup(restrict_corners) ;
    // make and append tasks 
    make_gpu_restrict_gz_task<amr::FACE, stag>(
        restrict_faces,
        ghost_array,
        state,
        coarse_buffers,
        stream,
        VEC(nx,ny,nz),ngz,nv,
        task_counter,
        task_list
    ) ; 

    make_gpu_restrict_gz_task<amr::EDGE, stag>(
        restrict_edges,
        ghost_array,
        state,
        coarse_buffers,
        stream,
        VEC(nx,ny,nz),ngz,nv,
        task_counter,
        task_list
    ) ; 

    make_gpu_restrict_gz_task<amr::CORNER, stag>(
        restrict_corners,
        ghost_array,
        state,
        coarse_buffers,
        stream,
        VEC(nx,ny,nz),ngz,nv,
        task_counter,
        task_list
    ) ; 

}

}

#endif 