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
#include <grace/amr/ghostzone_kernels/phys_bc_kernels.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <unordered_set>
#include <vector>
#include <numeric>

#ifndef GRACE_AMR_GHOSTZONE_KERNELS_PHYS_BC_TASK_FACTORY_HH
#define GRACE_AMR_GHOSTZONE_KERNELS_PHYS_BC_TASK_FACTORY_HH

namespace grace {

template< amr::element_kind_t elem_kind 
        , amr::element_kind_t bc_kind >
task_id_t 
make_gpu_phys_bc_task(
    std::vector<size_t> const& qid_h,
    std::vector<uint8_t> const& eid_h,
    std::vector<std::array<int8_t,3>> const& dir_h, 
    std::unordered_set<task_id_t> const& deps,
    Kokkos::View<bc_t*> var_bc,
    device_stream_t& stream, 
    task_id_t& task_counter,
    grace::var_array_t<GRACE_NSPACEDIM> data_array,
    size_t n, size_t nv, size_t ngz,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    if ( qid_h.size() == 0 ) return ; 
    Kokkos::View<size_t*> qid_d{"qid", qid_h.size()}; 
    Kokkos::View<uint8_t*> eid_d{"eid", qid_h.size()} ; 
    Kokkos::View<int8_t*[3]> dir_d{"dir", qid_h.size()} ; 

    grace::deep_copy_vec_to_view(qid_d,qid_h) ; 
    grace::deep_copy_vec_to_view(eid_d,eid_h) ; 
    grace::deep_copy_vec_to_2D_view(dir_d,dir_h) ;

    auto exec_space = Kokkos::DefaultExecutionSpace{stream} ; 

    gpu_task_t task{} ;

    amr::phys_bc_op<elem_kind,bc_kind,decltype(data_array)> functor{
       data_array, qid_d, eid_d, dir_d, var_bc, VEC(n,n,n),ngz  
    } ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0}, {nv, qid_h.size()}
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
    auto tid = task_counter++ ;
    task.task_id = tid ; 

    // set deps 
    for( auto const dep_id : deps ) {
        task._dependencies.push_back(dep_id) ; 
        task_list[dep_id]->_dependents.push_back(tid) ; 
    }

    task_list.push_back(
        std::make_unique<gpu_task_t>(task)
    ) ; 
    return tid; 
}

namespace detail {

std::tuple<bool, size_t, uint8_t, int8_t, int8_t, int8_t, amr::element_kind_t>
get_phys_bc_info(
    int _kind, 
    std::vector<quad_neighbors_descriptor_t> const & ghost_array, 
    gpu_task_desc_t const& d )
{
    // input here is element kind, descriptor 
    // output here is quad_id, e_id, grid normal, BC type 
    using namespace amr ; 
    amr::element_kind_t kind = static_cast<amr::element_kind_t>(_kind) ; 

    if ( kind == amr::element_kind_t::FACE ) {
        auto const& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
        bool is_cbuf = face.data.phys.in_cbuf ; 
        // face has no deps 
        return {  is_cbuf, is_cbuf ? ghost_array[std::get<0>(d)].cbuf_id : std::get<0>(d)
                , std::get<1>(d)
                , face.data.phys.dir[0]
                , face.data.phys.dir[1]
                , face.data.phys.dir[2] 
                , face.data.phys.type } ; 
    } else if (kind == amr::element_kind_t::EDGE) {
        auto const& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
        bool is_cbuf = edge.data.phys.in_cbuf ; 
        return {  is_cbuf, is_cbuf ? ghost_array[std::get<0>(d)].cbuf_id : std::get<0>(d)
                , std::get<1>(d)
                , edge.data.phys.dir[0]
                , edge.data.phys.dir[1]
                , edge.data.phys.dir[2] 
                , edge.data.phys.type } ; 
    } else {
        auto const& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
        bool is_cbuf = corner.phys.in_cbuf ; 
        return {  is_cbuf, is_cbuf ? ghost_array[std::get<0>(d)].cbuf_id : std::get<0>(d)
                , std::get<1>(d)
                , corner.phys.dir[0]
                , corner.phys.dir[1]
                , corner.phys.dir[2] 
                , corner.phys.type } ;
    }
} ;

std::tuple<uint8_t, uint8_t> inline 
get_cids_edge_face(uint8_t iface, const int dir[3])
{
    uint8_t idir ; 
    uint8_t isign ; 
    for( uint8_t i=0; i<3; ++i) {
        if ( dir[i] ) {
            idir = i ;
            isign = static_cast<uint8_t>(dir[i] > 0) ;
            break ;  
        }
    }
    if ((iface/2)==0) {
        return {
            (idir==1) ? (isign ? 1 : 0) : (isign ? 2 : 0),
            (idir==1) ? (isign ? 3 : 2) : (isign ? 3 : 1),
        } ; 
    } else {
        return {
            (idir==0) ? (isign ? 1 : 0) : (isign ? 2 : 0),
            (idir==0) ? (isign ? 3 : 2) : (isign ? 3 : 1),
        } ; 
    } 
};

inline uint8_t 
get_cid_corner_face(const int dir[3])
{
    uint8_t idir ; 
    uint8_t isign ; 
    for( uint8_t i=0; i<3; ++i) {
        if ( dir[i] ) {
            idir = i ;
            isign = static_cast<uint8_t>(dir[i] > 0) ;
            break ;  
        }
    }
    return static_cast<int8_t>(isign > 0) ;
};

// this returns a boolean indicating whether 
// this task needs to be deferred due to unresolved 
// dependencies 
template< typename F >
inline bool unpack_dependencies(
    int const& kind, 
    gpu_task_desc_t const& d, 
    std::vector<quad_neighbors_descriptor_t> const & ghost_array, 
    bool is_cbuf,
    F&& insert_dependencies 
) 
{
    if (kind == FACE) {
        // nothing to do here 
        return false ; 
    } else if (kind == EDGE) {
        auto const& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
        auto type = edge.data.phys.type ; 
        if ( type == FACE ) {
            // this depends on the face underneath
            // get adjacent face's id 
            auto face_idx = get_adjacent_idx<EDGE>(std::get<1>(d), edge.data.phys.dir);
            auto const& face = ghost_array[std::get<0>(d)].faces[face_idx] ;
            if ( face.level_diff == FINER ) {
                // fixme check this 
                auto [c0,c1] = get_cids_edge_face(face_idx, )
                insert_dependencies(EDGE,FACE,face.data.hanging.task_id[c0], is_cbuf) ; 
                insert_dependencies(EDGE,FACE,face.data.hanging.task_id[c1], is_cbuf) ; 
            } else {
                insert_dependencies(EDGE,FACE,face.data.full.task_id, is_cbuf) ; 
            }
        }
    } else {
        auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
        auto type = corner.phys.type ; 
        if ( type == FACE ) { // type stands for BC type 
            auto edge_idx = get_adjacent_idx<CORNER>(std::get<1>(d), corner.phys.dir);
            auto const& edge = ghost_array[std::get<0>(d)].edges[edge_idx] ;
            if ( edge.level_diff == FINER ) {
                // fixme check this 
                auto c0 = get_cid_corner_face(edge_idx) ; 
                insert_dependencies(CORNER,FACE,edge.data.hanging.task_id[c0], is_cbuf) ;
            } else {
                insert_dependencies(CORNER,FACE,edge.full.hanging.task_id[0], is_cbuf) ;
            }
        } else if ( type == EDGE ) {
            // we want to check if nearby edges are in cbufs, if so, this needs
            // to be deferred 
            for( int i=0; i<3; ++i) {
                auto const& edge = ghost_array[std::get<0>(d)].edges[amr::detail::c2e[std::get<1>(d)][i]] ; 
                if ( edge.data.phys.in_cbuf ) {
                    return true ;
                } 
            }
        }
    }
    return false ; 
} ; 

} /* namespace detail */



void insert_phys_bc_tasks(
    bucket_t phys_bc_tasks,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t<GRACE_NSPACEDIM> state, 
    grace::var_array_t<GRACE_NSPACEDIM> coarse_buffers,
    Kokkos::View<bc_t*> var_bc, 
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
) 
{
    using namespace amr ;

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

    bucket_t deferred_phys_bcs ; 

    // we have faces (ff) edges in faces (ef)
    // corners in faces (cf), edges in edges (ee)
    // corners in edges (ce), corners in corners (cc)
    // quad_id 
    std::array<std::array<std::vector<size_t>,3>,3> qid, qid_cbuf  ;
    std::array<std::array<std::vector<uint8_t>,3>,3> eid, eid_cbuf ;
    std::array<std::array<std::vector<std::array<int8_t,3>>,3>,3> dir, dir_cbuf ; 
    std::array<std::array<std::unordered_set<task_id_t>,3>,3> dependencies, dependencies_cbuf ; 

    auto insert_dependencies = [&] (int elem, int bc, task_id_t const& tid, bool is_cbuf) {
        if ( tid == UNSET_TASK_ID ) {
            ERROR("Unset task_id") ; 
        } else {
            if ( is_cbuf ) {
                dependencies_cbuf[elem][bc].insert(tid) ; 
            } else {
                dependencies[elem][bc].insert(tid) ; 
            }   
        }
    };

    auto const set_task_id = [&] (element_kind_t elem_kind, gpu_hanging_task_desc_t const& d)
    {
        if ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.phys.task_id = task_counter ;
        } else if (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.phys.task_id = task_counter ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.phys.task_id = task_counter ;
        }
    } ;  

    // loop through bucket, fill
    for( int kind=0; kind<3 ; ++kind) { // element kind 
        for( auto const& d: phys_bc_tasks[kind]) { 
            // find dependencies here ! 
            // for EDGE, FACE we need to look at faces underneath 
            // for CORNER, FACE we need to look at edges 
            // for EDGE EDGE we depend on face BCs
            // for CORNER, EDGE we depend on EDGE FACE BCs 
            // for CORNER CORNER we depend on EDGE BCs 
            auto [is_cbuf,_qid,_eid,dx,dy,dz,type] = detail::get_phys_bc_info(kind, ghost_array, d) ; 
            auto is_deferred = detail::unpack_dependencies(kind, d, ghost_array, is_cbuf, insert_dependencies);
            if ( is_cbuf ) {
                qid_cbuf[kind][type].push_back(_qid) ; 
                eid_cbuf[kind][type].push_back(_eid) ; 
                dir_cbuf[kind][type].emplace_back(dx,dy,dz) ; 
            } else if (! is_deferred ) {
                qid[kind][type].push_back(_qid) ; 
                eid[kind][type].push_back(_eid) ; 
                dir[kind][type].emplace_back(dx,dy,dz) ; 
            }
            if (is_cbuf or is_deferred) {
                // will be processed later 
                deferred_phys_bcs[kind].push_back(d) ;  
            }
            set_task_id(static_cast<element_kind_t>(kind),d) ; 
        }
    }
    // face face is safe to schedule
    task_id_t tid ; 
    tid =make_gpu_phys_bc_task<FACE,FACE>(
        qid[FACE][FACE],
        eid[FACE][FACE],
        dir[FACE][FACE],
        dependencies[FACE][FACE],
        var_bc, stream, task_counter,
        state,nx,nv,ngz
    ) ; 
    // BC edges depend on BC faces 
    dependencies[EDGE][EDGE].insert(tid) ; 
    // edges in faces are also fine 
    tid = make_gpu_phys_bc_task<EDGE,FACE>(
        qid[EDGE][FACE],
        eid[EDGE][FACE],
        dir[EDGE][FACE],
        dependencies[EDGE][FACE],
        var_bc, stream, task_counter,
        state,nx,nv,ngz
    ) ; 
    dependencies[CORNER][EDGE].insert(tid) ; 
    // and corners in faces 
    tid =make_gpu_phys_bc_task<CORNER,FACE>(
        qid[CORNER][FACE],
        eid[CORNER][FACE],
        dir[CORNER][FACE],
        dependencies[CORNER][FACE],
        var_bc, stream, task_counter,
        state,nx,nv,ngz
    ) ; 
    // nothing depends on this 
    // edges in edges are fine 
    tid = make_gpu_phys_bc_task<EDGE,EDGE>(
        qid[EDGE][EDGE],
        eid[EDGE][EDGE],
        dir[EDGE][EDGE],
        dependencies[EDGE][EDGE],
        var_bc, stream, task_counter,
        state,nx,nv,ngz
    ) ;
    dependencies[CORNER][CORNER].insert(tid) ; 
    // and corners in corners 
    tid = make_gpu_phys_bc_task<CORNER,CORNER>(
        qid[CORNER][CORNER],
        eid[CORNER][CORNER],
        dir[CORNER][CORNER],
        dependencies[CORNER][CORNER],
        var_bc, stream, task_counter,
        state,nx,nv,ngz
    ) ;
    // some corners in edges depend on 
    // edges in faces, some of which 
    // might now only be filled in 
    // cbufs, so we need to wait
    tid = make_gpu_phys_bc_task<CORNER,EDGE>(
        qid[CORNER][EDGE],
        eid[CORNER][EDGE],
        dir[CORNER][EDGE],
        dependencies[CORNER][EDGE],
        var_bc, stream, task_counter,
        state,nx,nv,ngz
    ) ; 

    // now for cbufs 
    ASSERT(qid_cbuf[FACE][FACE].size() == 0, "No faces can be in cbufs") ; 
    ASSERT(qid_cbuf[EDGE][EDGE].size() == 0, "No edges can be in cbufs") ; 
    ASSERT(qid_cbuf[CORNER][EDGE].size() == 0, "No corners can be in cbufs") ; 
    ASSERT(qid_cbuf[CORNER][CORNER].size() == 0, "No corners can be in cbufs") ; 

    tid = make_gpu_phys_bc_task<EDGE,FACE>(
        qid_cbuf[EDGE][FACE],
        eid_cbuf[EDGE][FACE],
        dir_cbuf[EDGE][FACE],
        dependencies_cbuf[EDGE][FACE],
        var_bc, stream, task_counter,
        coarse_buffers,nx/2,nv,ngz
    ) ; 
    tid = make_gpu_phys_bc_task<CORNER,FACE>(
        qid_cbuf[CORNER][FACE],
        eid_cbuf[CORNER][FACE],
        dir_cbuf[CORNER][FACE],
        dependencies_cbuf[CORNER][FACE],
        var_bc, stream, task_counter,
        coarse_buffers,nx/2,nv,ngz
    ) ; 
}

void insert_deferred_phys_bc_tasks(
    bucket_t phys_bc_tasks,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t<GRACE_NSPACEDIM> state, 
    grace::var_array_t<GRACE_NSPACEDIM> coarse_buffers,
    Kokkos::View<bc_t*> var_bc, 
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
) 
{
    using namespace amr ;

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

    std::array<std::array<std::vector<size_t>,3>,3> qid  ;
    std::array<std::array<std::vector<uint8_t>,3>,3> eid ;
    std::array<std::array<std::vector<std::array<int8_t,3>>,3>,3> dir ; 
    std::array<std::array<std::unordered_set<task_id_t>,3>,3> dependencies ; 

    auto insert_dependencies = [&] (int elem, int bc, task_id_t const& tid, bool dummy) {
        if ( tid == UNSET_TASK_ID ) {
            ERROR("Unset task_id") ; 
        } else {
            dependencies[elem][bc].insert(tid) ; 
        }
    };

    auto const set_task_id = [&] (element_kind_t elem_kind, gpu_hanging_task_desc_t const& d)
    {
        if ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.phys.task_id = task_counter ;
        } else if (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.phys.task_id = task_counter ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.phys.task_id = task_counter ;
        }
    } ;

    // loop through bucket, fill
    for( int kind=0; kind<3 ; ++kind) { // element kind 
        for( auto const& d: phys_bc_tasks[kind]) { 
            // find dependencies here ! 
            // for EDGE, FACE we need to look at faces underneath 
            // for CORNER, FACE we need to look at edges 
            // for EDGE EDGE we depend on face BCs
            // for CORNER, EDGE we depend on EDGE FACE BCs 
            // for CORNER CORNER we depend on EDGE BCs 
            auto [dummy,_qid,_eid,dx,dy,dz,type] = detail::get_phys_bc_info(kind, ghost_array, d) ; 
            auto dummy2 = detail::unpack_dependencies(kind, d, ghost_array, false, insert_dependencies);

            qid[kind][type].push_back(_qid) ; 
            eid[kind][type].push_back(_eid) ; 
            dir[kind][type].emplace_back(dx,dy,dz) ; 
            
            set_task_id(static_cast<element_kind_t>(kind),d) ; 
        }
    }

    task_id_t tid ; 
    tid = make_gpu_phys_bc_task<EDGE,FACE>(
        qid[EDGE][FACE],
        eid[EDGE][FACE],
        dir[EDGE][FACE],
        dependencies[EDGE][FACE],
        var_bc, stream, task_counter,
        state,nx,nv,ngz
    ) ; 
    dependencies[CORNER][EDGE].push_back(tid) ;

    tid = make_gpu_phys_bc_task<CORNER,FACE>(
        qid[CORNER][FACE],
        eid[CORNER][FACE],
        dir[CORNER][FACE],
        dependencies[CORNER][FACE],
        var_bc, stream, task_counter,
        state,nx,nv,ngz
    ) ; 

    tid = make_gpu_phys_bc_task<CORNER,EDGE>(
        qid[CORNER][EDGE],
        eid[CORNER][EDGE],
        dir[CORNER][EDGE],
        dependencies[CORNER][EDGE],
        var_bc, stream, task_counter,
        state,nx,nv,ngz
    ) ; 

    ASSERT(qid[FACE][FACE].size() == 0, "No faces can be deferred") ; 
    ASSERT(qid[EDGE][EDGE].size() == 0, "No edges can be deferred") ; 
    ASSERT(qid[CORNER][CORNER].size() == 0, "No corners can deferred") ; 

}

} /* namespace grace */
#endif 