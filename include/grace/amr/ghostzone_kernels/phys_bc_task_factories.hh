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
#include <unordered_map>

#ifndef GRACE_AMR_GHOSTZONE_KERNELS_PHYS_BC_TASK_FACTORY_HH
#define GRACE_AMR_GHOSTZONE_KERNELS_PHYS_BC_TASK_FACTORY_HH

namespace grace {

template< amr::element_kind_t elem_kind 
        , amr::element_kind_t bc_kind 
        , var_staggering_t stag >
task_id_t 
make_gpu_phys_bc_task(
    std::vector<size_t> const& qid_h,
    std::vector<uint8_t> const& eid_h,
    std::vector<std::array<int8_t,3>> const& dir_h, 
    std::unordered_set<task_id_t> const& deps,
    Kokkos::View<bc_t*> var_bc,
    device_stream_t& stream, 
    task_id_t& task_counter,
    grace::var_array_t data_array,
    size_t nx, size_t ny, size_t nz, size_t nv, size_t ngz,
    std::vector<std::unique_ptr<task_t>>& task_list, bool is_cbuf=false
)
{

    std::unordered_map<amr::element_kind_t,std::string> const name = {
        {amr::FACE,"face"}, {amr::EDGE,"edge"}, {amr::CORNER,"corner"}
    } ; 

    GRACE_TRACE("Registering phys-bc task ({}-{}, tid {}) number of elements {}", 
        detail::elem_kind_names[static_cast<int>(elem_kind)], 
        detail::elem_kind_names[static_cast<int>(bc_kind)], task_counter, qid_h.size()) ; 
    Kokkos::View<size_t*> qid_d{"qid", qid_h.size()}; 
    Kokkos::View<uint8_t*> eid_d{"eid", qid_h.size()} ; 
    Kokkos::View<int8_t*[3]> dir_d{"dir", qid_h.size()} ; 

    grace::deep_copy_vec_to_view(qid_d,qid_h) ; 
    grace::deep_copy_vec_to_view(eid_d,eid_h) ; 
    grace::deep_copy_vec_to_2D_view(dir_d,dir_h) ;

    auto exec_space = Kokkos::DefaultExecutionSpace{stream} ; 

    gpu_task_t task{} ;

    auto const off = get_index_staggerings(stag) ; 
    amr::phys_bc_op<elem_kind,bc_kind,decltype(data_array)> functor{
       data_array, qid_d, eid_d, dir_d, var_bc, VEC(nx+off[0],ny+off[1],nz+off[2]),ngz, is_cbuf
    } ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0}, {nv, qid_h.size()}
        } ;
    
    task._run = [functor, policy] (view_alias_t alias) mutable {
        functor.template set_data_ptr<stag>(alias) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE("Fill phys start") ; 
        #endif 
        Kokkos::parallel_for("fill_phys_ghostzones", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence() ; 
        GRACE_TRACE("Fill phys done") ; 
        #endif 
    };

    task.stream = &stream ; 
    auto tid = task_counter++ ;
    task.task_id = tid ; 

    // set deps 
    for( auto const dep_id : deps ) {
        ASSERT(dep_id < task_list.size(), "Dep-id out-of-range") ; 
        task._dependencies.push_back(dep_id) ; 
        task_list[dep_id]->_dependents.push_back(tid) ; 
    }

    task_list.push_back(
        std::make_unique<gpu_task_t>(std::move(task))
    ) ; 
    return tid; 
}

namespace detail {

std::tuple<bool, size_t, size_t, uint8_t, int8_t, int8_t, int8_t, amr::element_kind_t>
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
        return {  is_cbuf, ghost_array[std::get<0>(d)].cbuf_id, std::get<0>(d)
                , std::get<1>(d)
                , face.data.phys.dir[0]
                , face.data.phys.dir[1]
                , face.data.phys.dir[2] 
                , face.data.phys.type } ; 
    } else if (kind == amr::element_kind_t::EDGE) {
        auto const& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
        bool is_cbuf = edge.data.phys.in_cbuf ; 
        return {  is_cbuf, ghost_array[std::get<0>(d)].cbuf_id, std::get<0>(d)
                , std::get<1>(d)
                , edge.data.phys.dir[0]
                , edge.data.phys.dir[1]
                , edge.data.phys.dir[2] 
                , edge.data.phys.type } ; 
    } else {
        auto const& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
        bool is_cbuf = corner.phys.in_cbuf ; 
        return {  is_cbuf, ghost_array[std::get<0>(d)].cbuf_id, std::get<0>(d)
                , std::get<1>(d)
                , corner.phys.dir[0]
                , corner.phys.dir[1]
                , corner.phys.dir[2] 
                , corner.phys.type } ;
    }
} ;

std::tuple<uint8_t, uint8_t> inline 
get_cids_edge_face(uint8_t iface, const int8_t dir[3])
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
get_cid_corner_face(const int8_t dir[3])
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
template< var_staggering_t stag, typename F >
inline bool unpack_dependencies(
    int const& kind, 
    gpu_task_desc_t const& d, 
    std::vector<quad_neighbors_descriptor_t> const & ghost_array, 
    bool is_cbuf,
    F&& insert_dependencies 
) 
{
    using namespace amr ;
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
                auto [c0,c1] = get_cids_edge_face(face_idx, edge.data.phys.dir) ;                 
                insert_dependencies(EDGE,FACE,face.data.hanging.task_id[c0][stag], is_cbuf) ; 
                insert_dependencies(EDGE,FACE,face.data.hanging.task_id[c1][stag], is_cbuf) ; 
            } else { 

                insert_dependencies(EDGE,FACE,face.data.full.task_id[stag], is_cbuf) ; 
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
                auto c0 = get_cid_corner_face(corner.phys.dir) ; 
                insert_dependencies(CORNER,FACE,edge.data.hanging.task_id[c0][stag], is_cbuf) ;
            } else {
                insert_dependencies(CORNER,FACE,edge.data.full.task_id[stag], is_cbuf) ;
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


template< var_staggering_t stag> 
bucket_t insert_phys_bc_tasks(
    bucket_t phys_bc_tasks,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t state, 
    grace::var_array_t coarse_buffers,
    Kokkos::View<bc_t*> var_bc, 
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
) 
{
    using namespace amr ;

    bucket_t deferred_phys_bcs ; 

    // we have faces (ff) edges in faces (ef)
    // corners in faces (cf), edges in edges (ee)
    // corners in edges (ce), corners in corners (cc)
    // quad_id 
    std::array<std::array<std::vector<size_t>,3>,3> qid, qid_cbuf, cid_cbuf  ;
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

    auto const set_task_id = [&] (
        element_kind_t elem_kind, 
        std::vector<size_t> const& qid, 
        std::vector<uint8_t> const& eid,
        task_id_t tid )
    {
        ASSERT_DBG(qid.size() == eid.size(), "Mismatched array sizes in tid writeback.") ;
        for( int i=0; i<qid.size(); ++i) {
            auto _qid = qid[i] ; 
            auto _eid = eid[i] ; 
            if ( elem_kind == amr::element_kind_t::FACE ) {
                auto& face = ghost_array[_qid].faces[_eid] ; 
                face.data.phys.task_id[stag] = tid ;
            } else if (elem_kind == amr::element_kind_t::EDGE) {
                auto& edge = ghost_array[_qid].edges[_eid] ; 
                edge.data.phys.task_id[stag] = tid ;
            } else {
                auto& corner = ghost_array[_qid].corners[_eid] ; 
                corner.phys.task_id[stag] = tid ;
            }
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
            auto [is_cbuf,_cid,_qid,_eid,dx,dy,dz,type] = grace::detail::get_phys_bc_info(kind, ghost_array, d) ; 
            auto is_deferred = grace::detail::unpack_dependencies<stag>(kind, d, ghost_array, is_cbuf, insert_dependencies);

            if ( is_cbuf ) {
                cid_cbuf[kind][type].push_back(_cid) ; 
                qid_cbuf[kind][type].push_back(_qid) ; 
                eid_cbuf[kind][type].push_back(_eid) ; 
                dir_cbuf[kind][type].emplace_back(std::array{dx,dy,dz}) ; 
            } else if (! is_deferred ) {
                qid[kind][type].push_back(_qid) ; 
                eid[kind][type].push_back(_eid) ; 
                dir[kind][type].emplace_back(std::array{dx,dy,dz}) ; 
            }
            if (is_cbuf or is_deferred) {
                // will be processed later 
                deferred_phys_bcs[kind].push_back(d) ;  
            }
            
        }
    }
    // face face is safe to schedule
    task_id_t tid ; 
    if ( qid[FACE][FACE].size() >0 ) {
        tid =make_gpu_phys_bc_task<FACE,FACE,stag>(
            qid[FACE][FACE],
            eid[FACE][FACE],
            dir[FACE][FACE],
            dependencies[FACE][FACE],
            var_bc, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(FACE,qid[FACE][FACE],eid[FACE][FACE],tid) ; 
        // BC edges depend on BC faces 
        dependencies[EDGE][EDGE].insert(tid) ; 
    }
    if( qid[EDGE][FACE].size() > 0 ) {
        // edges in faces are also fine 
        tid = make_gpu_phys_bc_task<EDGE,FACE,stag>(
            qid[EDGE][FACE],
            eid[EDGE][FACE],
            dir[EDGE][FACE],
            dependencies[EDGE][FACE],
            var_bc, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(EDGE,qid[EDGE][FACE],eid[EDGE][FACE],tid) ;
        dependencies[CORNER][EDGE].insert(tid) ; 
    }
    if (qid[CORNER][FACE].size() > 0) {
        // and corners in faces 
        tid =make_gpu_phys_bc_task<CORNER,FACE,stag>(
            qid[CORNER][FACE],
            eid[CORNER][FACE],
            dir[CORNER][FACE],
            dependencies[CORNER][FACE],
            var_bc, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid[CORNER][FACE],eid[CORNER][FACE],tid) ;
        // nothing depends on this 
    }
    if (qid[EDGE][EDGE].size() > 0 ) {
        // edges in edges are fine 
        tid = make_gpu_phys_bc_task<EDGE,EDGE,stag>(
            qid[EDGE][EDGE],
            eid[EDGE][EDGE],
            dir[EDGE][EDGE],
            dependencies[EDGE][EDGE],
            var_bc, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ;
        // write back tid 
        set_task_id(EDGE,qid[EDGE][EDGE],eid[EDGE][EDGE],tid) ;
        dependencies[CORNER][CORNER].insert(tid) ; 
    }
    if ( qid[CORNER][CORNER].size() > 0 ) { 
        // and corners in corners 
        tid = make_gpu_phys_bc_task<CORNER,CORNER,stag>(
            qid[CORNER][CORNER],
            eid[CORNER][CORNER],
            dir[CORNER][CORNER],
            dependencies[CORNER][CORNER],
            var_bc, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ;
        // write back tid 
        set_task_id(CORNER,qid[CORNER][CORNER],eid[CORNER][CORNER],tid) ;
    }
    if (qid[CORNER][EDGE].size() > 0) {
        // some corners in edges depend on 
        // edges in faces, some of which 
        // might now only be filled in 
        // cbufs, so we need to wait
        tid = make_gpu_phys_bc_task<CORNER,EDGE,stag>(
            qid[CORNER][EDGE],
            eid[CORNER][EDGE],
            dir[CORNER][EDGE],
            dependencies[CORNER][EDGE],
            var_bc, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid[CORNER][EDGE],eid[CORNER][EDGE],tid) ;
    }
    
    // now for cbufs 
    ASSERT(qid_cbuf[FACE][FACE].size() == 0, "No faces can be in cbufs") ; 
    ASSERT(qid_cbuf[EDGE][EDGE].size() == 0, "No edges can be in cbufs") ; 
    ASSERT(qid_cbuf[CORNER][EDGE].size() == 0, "No corners can be in cbufs") ; 
    ASSERT(qid_cbuf[CORNER][CORNER].size() == 0, "No corners can be in cbufs") ; 
    if (qid_cbuf[EDGE][FACE].size() > 0 ) {
        // == 
        tid = make_gpu_phys_bc_task<EDGE,FACE,stag>(
            cid_cbuf[EDGE][FACE],
            eid_cbuf[EDGE][FACE],
            dir_cbuf[EDGE][FACE],
            dependencies_cbuf[EDGE][FACE],
            var_bc, stream, task_counter,
            coarse_buffers,nx/2,ny/2,nz/2,nv,ngz,task_list,true
        ) ; 
        // write back tid 
        set_task_id(EDGE,qid_cbuf[EDGE][FACE],eid_cbuf[EDGE][FACE],tid) ;
    }
    if (qid_cbuf[CORNER][FACE].size() > 0) {
        // == 
        tid = make_gpu_phys_bc_task<CORNER,FACE,stag>(
            cid_cbuf[CORNER][FACE],
            eid_cbuf[CORNER][FACE],
            dir_cbuf[CORNER][FACE],
            dependencies_cbuf[CORNER][FACE],
            var_bc, stream, task_counter,
            coarse_buffers,nx/2,ny/2,nz/2,nv,ngz,task_list,true
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid_cbuf[CORNER][FACE],eid_cbuf[CORNER][FACE],tid) ;
    }   
    


    return deferred_phys_bcs ;
}

template< var_staggering_t stag >
void insert_deferred_phys_bc_tasks(
    bucket_t phys_bc_tasks,
    std::vector<quad_neighbors_descriptor_t>& ghost_array,
    grace::var_array_t state, 
    grace::var_array_t coarse_buffers,
    Kokkos::View<bc_t*> var_bc, 
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
) 
{
    using namespace amr ;

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

    auto const set_task_id = [&] (
        element_kind_t elem_kind, 
        std::vector<size_t> const& qid, 
        std::vector<uint8_t> const& eid,
        task_id_t tid )
    {
        ASSERT_DBG(qid.size() == eid.size(), "Mismatched array sizes in tid writeback.") ;
        for( int i=0; i<qid.size(); ++i) {
            auto _qid = qid[i] ; 
            auto _eid = eid[i] ; 
            if ( elem_kind == amr::element_kind_t::FACE ) {
                auto& face = ghost_array[_qid].faces[_eid] ; 
                face.data.phys.task_id[stag] = tid ;
            } else if (elem_kind == amr::element_kind_t::EDGE) {
                auto& edge = ghost_array[_qid].edges[_eid] ; 
                edge.data.phys.task_id[stag] = tid ;
            } else {
                auto& corner = ghost_array[_qid].corners[_eid] ; 
                corner.phys.task_id[stag] = tid ;
            }
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
            auto [dummy,dummy3,_qid,_eid,dx,dy,dz,type] = grace::detail::get_phys_bc_info(kind, ghost_array, d) ; 
            auto dummy2 = grace::detail::unpack_dependencies<stag>(kind, d, ghost_array, false, insert_dependencies);

            qid[kind][type].push_back(_qid) ; 
            eid[kind][type].push_back(_eid) ; 
            dir[kind][type].emplace_back(std::array{dx,dy,dz}) ; 
        }
    }

    task_id_t tid ; 
    if (qid[EDGE][FACE].size() > 0) {
        tid = make_gpu_phys_bc_task<EDGE,FACE,stag>(
            qid[EDGE][FACE],
            eid[EDGE][FACE],
            dir[EDGE][FACE],
            dependencies[EDGE][FACE],
            var_bc, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        dependencies[CORNER][EDGE].insert(tid) ;
        // write back tid 
        set_task_id(EDGE,qid[EDGE][FACE],eid[EDGE][FACE],tid) ;
    }
    
    
    if( qid[CORNER][FACE].size() > 0 ) {
        tid = make_gpu_phys_bc_task<CORNER,FACE,stag>(
            qid[CORNER][FACE],
            eid[CORNER][FACE],
            dir[CORNER][FACE],
            dependencies[CORNER][FACE],
            var_bc, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid[CORNER][FACE],eid[CORNER][FACE],tid) ;
    }
    if (qid[CORNER][EDGE].size()>0){
        tid = make_gpu_phys_bc_task<CORNER,EDGE,stag>(
            qid[CORNER][EDGE],
            eid[CORNER][EDGE],
            dir[CORNER][EDGE],
            dependencies[CORNER][EDGE],
            var_bc, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid[CORNER][EDGE],eid[CORNER][EDGE],tid) ;
    }

    ASSERT(qid[FACE][FACE].size() == 0, "No faces can be deferred") ; 
    ASSERT(qid[EDGE][EDGE].size() == 0, "No edges can be deferred") ; 
    ASSERT(qid[CORNER][CORNER].size() == 0, "No corners can deferred") ; 

}

} /* namespace grace */
#endif 