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
task_id_t insert_restriction_tasks(
    std::unordered_set<size_t> const& cbuf_qid,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
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
        cbuf_id_h(i) = ghost_array[qid].cbuf_id ; 
        i+=1UL ; 
    }
    Kokkos::deep_copy(quad_id_d,quad_id_h) ;
    Kokkos::deep_copy(cbuf_id_d,cbuf_id_h) ;

    gpu_task_t task{} ;

    amr::restrict_op<decltype(state)> functor(
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

    task_list.push_back(
        std::make_unique<gpu_task_t>(std::move(task))
    ) ; 
    return task.task_id ; 
}

template< amr::element_kind_t elem_kind >
auto get_iter_policy(
    device_stream_t& stream, size_t n, size_t ngz, size_t nv, size_t nq 
) {
    using namespace amr ; 
    using namespace Kokkos ; 
    if constexpr ( elem_kind == FACE ) {
        return MDRangePolicy<Rank<4>, ghost_restrict_face_tag>(
            DefaultExecutionSpace{stream},
            {0,0,0,0}, {n/2,ngz/2,nv,nq}
        ) ; 
    } else if constexpr (elem_kind == EDGE) {
        return MDRangePolicy<Rank<3>, ghost_restrict_edge_tag>(
            DefaultExecutionSpace{stream},
            {0,0,0}, {ngz/2,nv,nq}
        ) ; 
    } else {
        return MDRangePolicy<Rank<2>, ghost_restrict_corner_tag>(
            DefaultExecutionSpace{stream},
            {0,0}, {nv,nq}
        ) ; 
    }
} 


template< amr::element_kind_t elem_kind >
void make_gpu_restrict_gz_task(
    std::vector<gpu_task_desc_t> const& bucket,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t<GRACE_NSPACEDIM> state, 
    grace::var_array_t<GRACE_NSPACEDIM> coarse_buffers,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
) {
    using namespace amr ;
    Kokkos::View<size_t*> qid("qid_restrict", bucket.size())
                        , cbuf_qid("cbuf_qid_restrict", bucket.size()) ; 
    Kokkos::View<uint8_t*> eid("eid_restrict", bucket.size())
                         , cbuf_eid("cbuf_eid_restrict", bucket.size()) ; 

    auto qid_h = Kokkos::create_mirror_view(qid) ; 
    auto cbuf_qid_h = Kokkos::create_mirror_view(cbuf_qid) ; 
    auto eid_h = Kokkos::create_mirror_view(eid) ; 
    auto cbuf_eid_h = Kokkos::create_mirror_view(cbuf_eid) ; 
    
    std::unordered_set<task_id_t> dependencies ; 
    auto insert_dependency = [&] (task_id_t tid) {
        if ( tid != UNSET_TASK_ID ) {
            dependencies.insert(tid) ; 
        } else {
            // this should not happen here 
            ERROR("Unset task_id") ; 
        }
    } ; 
    auto write_back_tid = [&](gpu_task_desc_t const& d) {
        if constexpr (elem_kind == FACE) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            if ( face.level_diff == level_diff_t::FINER ) {
                for(int ic=0; ic<P4EST_CHILDREN/2; ++ic) face.data.hanging.task_id[ic] = task_counter; 
            } else {
                face.data.full.task_id = task_counter; 
            }
        } else if constexpr (elem_kind == EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edge[std::get<1>(d)] ; 
            if ( edge.level_diff == level_diff_t::FINER ) {
                for(int ic=0; ic<2; ++ic) edge.data.hanging.task_id[ic] = task_counter; 
            } else {
                edge.data.full.task_id = task_counter; 
            }
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ;
            corner.data.task_id = task_counter;  
        }
    }

    auto get_info = [&] (gpu_task_desc_t const& d) -> std::tuple<> {
        if constexpr (elem_kind == FACE) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            if ( face.level_diff == level_diff_t::FINER ) {
                for( int ic=0; ic<P4EST_CHILDREN/2; ++ic) 
                    insert_dependency(face.data.hanging.task_id[ic]) ; 
            } else {
                insert_dependency(face.data.full.task_id) ; 
            }
             
            return {std::get<0>(d), ghost_array[std::get<0>(d)].cbuf_id, std::get<1>(d), face.face } ; 
        } else if constexpr (elem_kind == EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            ASSERT(edge.filled, "Edge passed to restrict_gz is virtual.") ; 
            if ( edge.level_diff = FINER) {
                for( int ic=0; ic<2; ++ic)
                    insert_dependency(edge.data.hanging.task_id[ic]) ; 
            } else {
                insert_dependency(edge.data.full.task_id) ;
            } 
            
            return {std::get<0>(d), ghost_array[std::get<0>(d)].cbuf_id, std::get<1>(d), edge.edge } ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            ASSERT(corner.filled, "Corner passed to restrict_gz is virtual.") ; 
            insert_dependency(corner.data.task_id) ; 
            
            return {std::get<0>(d), ghost_array[std::get<0>(d)].cbuf_id, std::get<1>(d), corner.corner } ;
        }
    } ; 

    size_t i{0UL} ; 
    for( auto const& d: bucket ) {
        auto [_qid,_cid,_eid,_ceid] = get_info(d) ; 
        qid_h(i) = _qid ; 
        cbuf_id_h(i) = _cid ; 
        eid_h(i) = _eid ; 
        cbuf_eid_h(i) = _ceid ; 
        write_back_tid(d) ; 
        i+= 1UL ;
    }

    // here we could try to split deps, might be agood 
    // optimization knob. For now simplest thing is to 
    // create a single task 
    gpu_task_t task {} ; 

    ghost_restrict_op functor{
        state, coarse_buffers, qid, cbuf_id, eid, cbuf_eid, VEC(nx,ny,nz), ngz
    } ; 

    // the rank of iterations depends on the element kind 
    auto policy = get_iter_policy<elem_kind>(stream,nx,ngz,nv,bucket.size()) ; 

    task._run = [functor,policy] (view_alias_t alias) mutable {
        functor.set_data_ptr(alias) ; 
        Kokkos::parallel_for("ghostzone_restrict", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        GRACE_TRACE("GZ restrict done") ; 
        #endif 
    }

    task.stream = &stream ; 
    task.task_id = task_counter++ ; 

    for ( auto const& tid: dependencies) {
        task._dependencies.push_back(tid) ; 
        task_list[tid]->_dependents.push_back(task.task_id) ; 
    }
}

void insert_ghost_restriction_tasks(
    std::unordered_set<size_t> const& cbuf_qid,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t<GRACE_NSPACEDIM> state, 
    grace::var_array_t<GRACE_NSPACEDIM> coarse_buffers,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
) {
    std::vector<std::tuple<size_t,uint8_t>> restrict_faces, restrict_edges, restrict_corners ;
    
    const uint8_t f2e[P4EST_FACES][4] = {
        {4,6,8,10}  , // face 0 
        {5,7,9,11}  , // face 1
        {0,2,8,9}   , // face 2
        {1,3,10,11} , // face 3 
        {0,1,4,5}   , // face 4
        {2,3,6,7}     // face 5 
    } ; 

    const uint8_t f2c[P4EST_FACES][4] = {
        {0,2,4,6},
        {1,3,5,7},
        {0,1,4,5},
        {2,3,6,7},
        {0,1,2,3},
        {4,5,6,7}
    } ; 

    const uint8_t e2f[12][2] = {
        {2,4},//0
        {3,4},//1
        {2,5},//2
        {3,5},//3
        {0,4},//4
        {1,4},//5
        {0,5},//6
        {1,5},//7
        {0,2},//8
        {1,2},//9
        {0,3},//10
        {1,3}//11
    } ; 

    const uint8_t e2c[12][2] = {
        {0,1},//0
        {2,3},//1
        {4,5},//2
        {6,7},//3
        {0,2},//4
        {1,3},//5
        {4,6},//6
        {5,7},//7
        {0,4},//8
        {1,5},//9
        {2,6},//10
        {3,7}//11
    } ; 

    const uint8_t c2f[P4EST_CHILDREN][3] = {
        {0,2,4},//0
        {1,2,4},//1
        {0,3,4},//2
        {1,3,4},//3
        {0,2,5},//4
        {1,2,5},//5
        {0,3,5},//6
        {1,3,5} //7
    } ; 

    const uint8_t c2e[P4EST_CHILDREN][3] = {
        {0,4,8},//0
        {0,5,9},//1
        {1,4,10},//2
        {1,5,11},//3
        {2,6,8},//4
        {2,7,9},//5
        {3,6,10},//6
        {3,7,11} //7
    } ; 

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
                auto e = f2e[f][ie] ; 
                auto& edge = ghost_array[qid].edges[e] ; 
                if ( ! edge.filled ) continue; 
                if ( edge.kind == interface_kind_t::PHYS) edge.data.phys.in_cbuf = true ;  
                if (!(edge.level_diff == level_diff_t::COARSER)) restrict_edges.emplace_back(qid,e) ; 
            } 
        }
        
        for( int8_t e=0; e<12; ++e){
            auto& edge = ghost_array[qid].edges[e] ; 
            if (!(edge.filled)) continue ; 
            if (edge.kind == interface_kind_t::PHYS) continue ;  
            if (!(edge.level_diff == level_diff_t::COARSER)) continue ;  
            for( int iface=0; iface<2; ++iface) {
                auto f = e2f[e][iface] ; 
                auto& face = ghost_array[qid].faces[f] ; 
                if ( face.kind == interface_kind_t::PHYS) face.data.phys.in_cbuf=true ;  
                if (!(face.level_diff == level_diff_t::COARSER)) restrict_faces.emplace_back(qid,f) ;
            }
            for( int ic=0; ic<2; ++ic) {
                auto c = e2c[e][ic] ; 
                auto& corner = ghost_array[qid].corners[c] ; 
                if ( ! corner.filled ) continue; 
                if ( corner.kind == interface_kind_t::PHYS) corner.phys.in_cbuf=true ;  
                if (!(corner.level_diff == level_diff_t::COARSER)) restrict_corners.emplace_back(qid,c) ; 
            }
        }
        for( int8_t c=0; c<P4EST_CHILDREN; ++c){
            auto& corner = ghost_array[qid].corners[c] ; 
            if (!(corner.filled)) continue ; 
            if (corner.kind == interface_kind_t::PHYS) continue ;  
            if (!(corner.level_diff == level_diff_t::COARSER)) continue ;  
            for( int ie=0; ie<3; ++ie) {
                auto e = c2e[c][ie] ; 
                auto& edge = ghost_array[qid].edges[e] ; 
                if ( ! edge.filled ) continue; 
                if ( edge.kind == interface_kind_t::PHYS) edge.data.phys.in_cbuf = true ;  
                if (!(edge.level_diff == level_diff_t::COARSER)) restrict_edges.emplace_back(qid,e) ; 
            }
        }
    }

    // make and append tasks 
    make_gpu_restrict_gz_task<FACE>(
        restrict_faces,
        ghost_array,
        state,
        coarse_buffers,
        stream,
        VEC(nx,ny,nz),ngz,nv,
        task_counter,
        task_list
    ) ; 

    make_gpu_restrict_gz_task<EDGE>(
        restrict_edges,
        ghost_array,
        state,
        coarse_buffers,
        stream,
        VEC(nx,ny,nz),ngz,nv,
        task_counter,
        task_list
    ) ; 

    make_gpu_restrict_gz_task<CORNER>(
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