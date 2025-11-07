/**
 * @file prolongation_task_factories.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
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
#include <grace/amr/ghostzone_kernels/prolongation_kernels.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/utils/prolongation.hh>
#include <grace/utils/limiters.hh>

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <unordered_set>
#include <vector>
#include <numeric>


#ifndef GRACE_AMR_BC_PROLONG_FACTORIES_HH
#define GRACE_AMR_BC_PROLONG_FACTORIES_HH 

namespace grace {

template< amr::element_kind_t elem_kind > 
task_id_t 
make_div_preserving_prolongation_task( 
    std::vector<size_t> const& qid, 
    std::vector<size_t> const& cid, 
    std::vector<uint8_t> const& eid,
    std::vector<std::array<std::array<int,2>,3>> const& have_fine,
    std::unordered_set<task_id_t> const& deps,
    device_stream_t& stream,
    task_id_t& task_counter,
    grace::var_array_t data,
    grace::staggered_variable_arrays_t coarse_buffers,
    size_t n, size_t nv, size_t ngz,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    GRACE_TRACE("Recording GPU-div-preserving-prolong task (tid {}), number of elements {}", task_counter, qid.size()) ; 
    Kokkos::View<size_t*> qid_d{"qid", qid.size()}; 
    Kokkos::View<size_t*> cid_d{"qid", cid.size()}; 
    Kokkos::View<uint8_t*> eid_d{"eid", eid.size()} ; 
    Kokkos::View<int8_t***> have_fine_data_d{"have_fine_data", 3,2,eid.size()} ; 
    auto have_fine_data_h = Kokkos::create_mirror_view(have_fine_data_d) ; 
    grace::deep_copy_vec_to_view(qid_d,qid) ;
    grace::deep_copy_vec_to_view(cid_d,cid) ;
    grace::deep_copy_vec_to_view(eid_d,eid) ;
    for( int i=0; i<qid.size(); ++i) {
        for( int j=0; j<3; ++j) {
            have_fine_data_h(j,0,i) = have_fine[i][j][0] ; 
            have_fine_data_h(j,1,i) = have_fine[i][j][1] ; 
        }
    }
    Kokkos::deep_copy(have_fine_data_d,have_fine_data_h) ; 

    auto exec_space = Kokkos::DefaultExecutionSpace{stream} ; 

    gpu_task_t task{} ;

    amr::div_free_prolong_op<elem_kind,decltype(data)> 
    functor{
       data, data, data,
       coarse_buffers.face_staggered_fields_x,
       coarse_buffers.face_staggered_fields_y,
       coarse_buffers.face_staggered_fields_z,
       qid_d, cid_d, eid_d, 
       have_fine_data_d,
       n, ngz
    } ; 

    Kokkos::TeamPolicy 
        policy{
            exec_space, static_cast<int>(qid.size()), Kokkos::AUTO
        } ;

    task._run = [functor,policy] (view_alias_t alias) mutable {
        functor.set_data_ptr(alias) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE_DBG("Prolong start.") ; 
        #endif 
        Kokkos::parallel_for("prolong_ghostzones", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence(); 
        GRACE_TRACE_DBG("Prolong end.");
        #endif 
    } ; 

    auto tid = task_counter ++ ;
    task.task_id = tid ; 
    task.stream = &stream ; 
    
    for( auto const& t : deps){
        task._dependencies.push_back(t) ; 
        task_list[t]->_dependents.push_back(tid) ;
    }

    task_list.push_back(std::make_unique<gpu_task_t>(std::move(task))) ; 

    return tid ;
}

template< amr::element_kind_t elem_kind, var_staggering_t stag > 
task_id_t 
make_prolongation_task(
    std::vector<size_t> const& qid, 
    std::vector<size_t> const& cid, 
    std::vector<uint8_t> const& eid,
    std::unordered_set<task_id_t> const& deps,
    device_stream_t& stream,
    task_id_t& task_counter,
    grace::var_array_t data,
    grace::var_array_t coarse_buffers,
    size_t n, size_t nv, size_t ngz,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    GRACE_TRACE("Recording GPU-prolong task (tid {}), number of elements {}", task_counter, qid.size()) ; 
    Kokkos::View<size_t*> qid_d{"qid", qid.size()}; 
    Kokkos::View<size_t*> cid_d{"qid", cid.size()}; 
    Kokkos::View<uint8_t*> eid_d{"eid", eid.size()} ; 

    grace::deep_copy_vec_to_view(qid_d,qid) ;
    grace::deep_copy_vec_to_view(cid_d,cid) ;
    grace::deep_copy_vec_to_view(eid_d,eid) ;

    auto exec_space = Kokkos::DefaultExecutionSpace{stream} ; 

    gpu_task_t task{} ;

    amr::prolong_op<utils::linear_prolongator_t<grace::minmod>,elem_kind,decltype(coarse_buffers)> 
    functor{
       data,
       coarse_buffers,
       qid_d, cid_d, eid_d, 
       n, ngz
    } ; 
    
    Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>   
        policy{
            exec_space, {0,0,0,0,0}, amr::get_iter_range<elem_kind>(ngz,n,nv,qid.size())
        } ;

    task._run = [functor,policy] (view_alias_t alias) mutable {
        functor.template set_data_ptr<stag>(alias) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE_DBG("Prolong start.") ; 
        #endif 
        Kokkos::parallel_for("prolong_ghostzones", policy, functor) ; 
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence(); 
        GRACE_TRACE_DBG("Prolong end.");
        #endif 
    } ; 

    auto tid = task_counter ++ ;
    task.task_id = tid ; 
    task.stream = &stream ; 
    
    for( auto const& t : deps){
        task._dependencies.push_back(t) ; 
        task_list[t]->_dependents.push_back(tid) ;
    }

    task_list.push_back(std::make_unique<gpu_task_t>(std::move(task))) ; 

    return tid ;

}

template < var_staggering_t stag >
void insert_prolongation_tasks(
    bucket_t const & prolong_tasks,
    std::vector<quad_neighbors_descriptor_t> & ghost_array, 
    grace::var_array_t state, 
    grace::var_array_t coarse_buffers,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    using namespace amr ; 
    std::array<std::vector<size_t>,3> qid, cid ; 
    std::array<std::vector<uint8_t>,3> eid ; 
    std::array<std::unordered_set<task_id_t>,3> deps ;

    auto insert_dep = [&] (int elem, task_id_t const& tid) {
        if ( tid == UNSET_TASK_ID ) {
            ERROR("Unset task_id") ; 
        } else {
            deps[elem].insert(tid) ;        
        }
    };

    auto const get_info = [&] (int _kind, gpu_task_desc_t const& d) -> std::tuple<size_t,size_t,uint8_t> {
        using namespace amr ; 
        amr::element_kind_t kind = static_cast<amr::element_kind_t>(_kind) ; 

        return {std::get<0>(d), ghost_array[std::get<0>(d)].cbuf_id, std::get<1>(d)} ; 
    } ; 

    // we depend on all nearby elements 
    // nothing across here can be FINER 
    // due to 2:1 balance 
    auto const unpack_dependencies = [&] (int _kind, gpu_task_desc_t const& d) {
        using namespace amr ; 
        amr::element_kind_t kind = static_cast<amr::element_kind_t>(_kind) ; 

        if (kind == FACE) {
            insert_dep(FACE, ghost_array[std::get<0>(d)].faces[std::get<1>(d)].data.full.task_id[stag]) ; 
            for( auto eid: amr::detail::f2e[std::get<1>(d)] ) {
                auto& edge = ghost_array[std::get<0>(d)].edges[eid] ;
                if ( !edge.filled ) continue ; 
                if ( edge.kind == interface_kind_t::PHYS ) {
                    insert_dep(FACE, edge.data.phys.task_id[stag]) ;
                } else {
                    insert_dep(FACE, edge.data.full.task_id[stag]) ;
                }   
            }
        } else if (kind == EDGE) {
            insert_dep(EDGE, ghost_array[std::get<0>(d)].edges[std::get<1>(d)].data.full.task_id[stag]) ; 
            for( auto fid: amr::detail::e2f[std::get<1>(d)] ) {
                auto& face = ghost_array[std::get<0>(d)].faces[fid] ;
                if ( face.kind == interface_kind_t::PHYS ) {
                    insert_dep(EDGE,face.data.phys.task_id[stag]) ; 
                } else {
                    insert_dep(EDGE,face.data.full.task_id[stag]) ; 
                }
            }
            for( auto cid: amr::detail::e2c[std::get<1>(d)] ) {
                auto& corner = ghost_array[std::get<0>(d)].corners[cid] ; 
                if ( !corner.filled ) continue ; 
                if ( corner.kind == interface_kind_t::PHYS ) {
                    insert_dep(EDGE,corner.phys.task_id[stag]) ; 
                } else {
                    insert_dep(EDGE,corner.data.task_id[stag]) ; 
                }
                
            }
        } else {
            insert_dep(CORNER, ghost_array[std::get<0>(d)].corners[std::get<1>(d)].data.task_id[stag]) ; 
            for( auto eid: amr::detail::c2e[std::get<1>(d)] ) {
                auto& edge = ghost_array[std::get<0>(d)].edges[eid] ;
                if ( !edge.filled ) continue ; 
                if ( edge.kind == interface_kind_t::PHYS ) {
                    insert_dep(CORNER, edge.data.phys.task_id[stag]) ;
                } else {
                    insert_dep(CORNER, edge.data.full.task_id[stag]) ;
                } 
            }
        }
    } ; 

    auto const set_task_id = [&] (
        amr::element_kind_t elem_kind, 
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
                face.data.full.task_id[stag] = tid ;
            } else if (elem_kind == amr::element_kind_t::EDGE) {
                auto& edge = ghost_array[_qid].edges[_eid] ; 
                edge.data.full.task_id[stag] = tid ;
            } else {
                auto& corner = ghost_array[_qid].corners[_eid] ; 
                corner.data.task_id[stag] = tid ;
            }
        }
        
    } ; 

    // loop through bucket, fill
    for( int kind=0; kind<3 ; ++kind) { // element kind 
        for( auto const& d: prolong_tasks[kind]) { 
            auto [_qid,_cid,_eid] = get_info(kind,d) ; 
            unpack_dependencies(kind,d) ; 

            qid[kind].push_back(_qid) ; 
            cid[kind].push_back(_cid) ; 
            eid[kind].push_back(_eid) ; 
        }
    }

    task_id_t tid ; 
    if ( qid[FACE].size() > 0 ) 
    {
        tid = make_prolongation_task<FACE, stag>(
            qid[FACE], cid[FACE], eid[FACE], deps[FACE], 
            stream, task_counter, state, coarse_buffers,
            nx, nv, ngz, task_list 
        ) ; 
        set_task_id(FACE,qid[FACE],eid[FACE],tid) ; 
    }
    if ( qid[EDGE].size() > 0 ){
        tid = make_prolongation_task<EDGE, stag>(
            qid[EDGE], cid[EDGE], eid[EDGE], deps[EDGE], 
            stream, task_counter, state, coarse_buffers,
            nx, nv, ngz, task_list 
        ) ; 
        set_task_id(EDGE,qid[EDGE],eid[EDGE],tid) ; 
    }
    if ( qid[CORNER].size() > 0 ) 
    {
        tid = make_prolongation_task<CORNER, stag>(
            qid[CORNER], cid[CORNER], eid[CORNER], deps[CORNER], 
            stream, task_counter, state, coarse_buffers,
            nx, nv, ngz, task_list 
        ) ; 
        set_task_id(CORNER,qid[CORNER],eid[CORNER],tid) ;
    }
}




void insert_div_preserving_prolongation_tasks(
    bucket_t const & prolong_tasks,
    std::vector<quad_neighbors_descriptor_t> & ghost_array, 
    grace::var_array_t state, 
    grace::staggered_variable_arrays_t coarse_buffers,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{
    using namespace amr ; 


    std::array<std::vector<size_t>,3> qid, cid ; 
    std::array<std::vector<uint8_t>,3> eid ; 
    std::array<std::unordered_set<task_id_t>,3> deps ;
    std::array<std::vector<std::array<std::array<int,2>,3>>,3> has_fine ; 

    auto insert_dep = [&] (int elem, std::array<task_id_t,N_VAR_STAGGERINGS> const& _tids) {
        int stags[] = {static_cast<int>(STAG_FACEX),static_cast<int>(STAG_FACEY),static_cast<int>(STAG_FACEZ)} ; 
        for ( int i=0; i<3; ++i){
            auto tid = _tids[stags[i]]; 
            if ( tid == UNSET_TASK_ID ) {
                GRACE_TRACE_DBG("Stag {} unset", stags[i]) ; 
                ERROR("Unset task_id") ; 
            } else {
                deps[elem].insert(tid) ;        
            }
        }
        
    };

    auto const get_info = [&] (int _kind, gpu_task_desc_t const& d) -> std::tuple<size_t,size_t,uint8_t,std::array<std::array<int,2>,3>> {
        using namespace amr ; 
        amr::element_kind_t kind = static_cast<amr::element_kind_t>(_kind) ; 
        std::array<std::array<int,2>,3> has_fine_data {{
            {{0,0}}, {{0,0}}, {{0,0}}
        }}; 
        if ( kind == FACE ) {
            int8_t side = std::get<1>(d)%2 ;
            int8_t normal = std::get<1>(d)/2 ; 
            // if upper face the lower end is always fine
            // vice versa for lower face
            if ( side ) {
                has_fine_data[normal][0] = 1 ;
                has_fine_data[normal][1] = 0 ;
            } else {
                has_fine_data[normal][1] = 1 ;
                has_fine_data[normal][0] = 0 ;
            }
            // for the other two directions we need to check adjacent elements.
            // in f2e the edge codes are in z order
            int other_dirs[2] ; 
            if ( normal == 0 ) {
                // 1 2 
                other_dirs[0] = 1 ;
                other_dirs[1] = 2 ; 
            } else if ( normal == 1 ) {
                other_dirs[0] = 0 ;
                other_dirs[1] = 2 ; 
            } else {
                other_dirs[0] = 0 ;
                other_dirs[1] = 1 ; 
            }

            for(int idir=0; idir<2; ++idir) {
                for( int j=0; j<2; ++j) {
                    auto ie = grace::amr::detail::f2e[std::get<1>(d)][2*idir+j] ; 
                    auto& edge = ghost_array[std::get<0>(d)].edges[ie] ; 
                    if ( !edge.filled ) continue ; 
                    if (edge.level_diff == SAME) {has_fine_data[other_dirs[idir]][j] = 1 ;}
                    else {has_fine_data[other_dirs[idir]][j] = 0 ;}
                }
            }
            
        } else if ( kind == EDGE ) {
            int edge_dir = (std::get<1>(d)<4 ? 0 : (std::get<1>(d) < 8 ? 1 : 2)) ; 
            int edge_side[2] = { (std::get<1>(d)>>0)&1, (std::get<1>(d)>>1)&1} ; 
            int other_dirs[2] ; 
            if( edge_dir == 0 ) {
                other_dirs[0] = 1 ; 
                other_dirs[1] = 2 ; 
            } else if ( edge_dir == 1) {
                other_dirs[0] = 0 ;
                other_dirs[1] = 2 ;
            } else {
                other_dirs[0] = 0 ; 
                other_dirs[1] = 1 ; 
            }
            // along edge we need to check two corners 
            for( int j=0; j<2; ++j) {
                auto ic = grace::amr::detail::e2c[std::get<1>(d)][j] ;
                auto& corner = ghost_array[std::get<0>(d)].corners[ic] ; 
                if (!corner.filled) continue ; 
                if ( corner.level_diff == SAME) has_fine_data[edge_dir][j] = 1; 
            }

            // across it's faces
            for(int idir=0; idir<2; ++idir) {
                auto iface = grace::amr::detail::e2f[std::get<1>(d)][idir] ;
                int dir = other_dirs[idir] ; 
                int side = edge_side[idir] ; 

                auto& face = ghost_array[std::get<0>(d)].faces[iface] ;
                int j = side ? 0 : 1 ; // if upper edge, write to lower 
                if ( face.level_diff == SAME ) has_fine_data[dir][j] = 1; 
            }

        } else {
            // a corner has 4 edges to check, one per direction. 
            // as always if the corner is in the "upper" ghostzones
            // and the neighbor edge has fine data we mark has_fine_data[lower] = 1 
            int corner_side[3] = { (std::get<1>(d)>>0)&1, (std::get<1>(d)>>1)&1, (std::get<1>(d)>>2)&1} ;
            for ( int idir=0; idir<3; ++idir) {
                int ie = grace::amr::detail::c2e[std::get<1>(d)][idir] ; 
                auto& edge = ghost_array[std::get<0>(d)].edges[ie] ; 
                int j = corner_side[idir] ? 0 : 1 ; 
                if (!edge.filled) continue ; 
                if ( edge.level_diff == SAME ) has_fine_data[idir][j] = 1 ;
            } 
        }
        return {std::get<0>(d), ghost_array[std::get<0>(d)].cbuf_id, std::get<1>(d), has_fine_data} ; 
    } ; 

    // we depend on all nearby elements 
    // nothing across here can be FINER 
    // due to 2:1 balance 
    auto const unpack_dependencies = [&] (int _kind, gpu_task_desc_t const& d) {
        using namespace amr ; 
        amr::element_kind_t kind = static_cast<amr::element_kind_t>(_kind) ; 

        if (kind == FACE) {
            GRACE_TRACE_DBG("Inserting prolong dep FACE qid {} fid {}", std::get<0>(d), std::get<1>(d)) ; 
            insert_dep(FACE, ghost_array[std::get<0>(d)].faces[std::get<1>(d)].data.full.task_id) ; 
            for( auto eid: amr::detail::f2e[std::get<1>(d)] ) {
                auto& edge = ghost_array[std::get<0>(d)].edges[eid] ;
                if ( !edge.filled ) {
                    GRACE_TRACE_DBG("Inserting prolong dep FACE from virtual EDGE qid {} fid {} eid {}", std::get<0>(d), std::get<1>(d), eid) ; 
                    insert_dep(FACE, edge.data.full.task_id) ;
                } else if ( edge.kind == interface_kind_t::PHYS ) {
                    GRACE_TRACE_DBG("Inserting prolong dep FACE from phys EDGE qid {} fid {} eid {}", std::get<0>(d), std::get<1>(d), eid) ; 
                    insert_dep(FACE, edge.data.phys.task_id) ;
                } else {
                    GRACE_TRACE_DBG("Inserting prolong dep FACE from EDGE qid {} fid {} eid {}", std::get<0>(d), std::get<1>(d), eid) ; 
                    insert_dep(FACE, edge.data.full.task_id) ;
                }   
            }
        } else if (kind == EDGE) {
            GRACE_TRACE_DBG("Inserting prolong dep EDGE qid {} eid {}", std::get<0>(d), std::get<1>(d)) ; 
            insert_dep(EDGE, ghost_array[std::get<0>(d)].edges[std::get<1>(d)].data.full.task_id) ; 
            for( auto fid: amr::detail::e2f[std::get<1>(d)] ) {
                auto& face = ghost_array[std::get<0>(d)].faces[fid] ;
                if ( face.kind == interface_kind_t::PHYS ) {
                    GRACE_TRACE_DBG("Inserting prolong dep EDGE from FACE qid {} eid {} fid {}", std::get<0>(d), std::get<1>(d), fid) ; 
                    insert_dep(EDGE,face.data.phys.task_id) ; 
                } else {
                    GRACE_TRACE_DBG("Inserting prolong dep EDGE from FACE qid {} eid {} fid {}", std::get<0>(d), std::get<1>(d), fid) ; 
                    insert_dep(EDGE,face.data.full.task_id) ;  
                }
            }
            for( auto cid: amr::detail::e2c[std::get<1>(d)] ) {
                auto& corner = ghost_array[std::get<0>(d)].corners[cid] ; 
                if ( !corner.filled ) {
                    GRACE_TRACE_DBG("Inserting prolong dep EDGE from virtual CORNER qid {} eid {} cid {}", std::get<0>(d), std::get<1>(d), cid) ; 
                    insert_dep(EDGE,corner.data.task_id) ; 
                } else if ( corner.kind == interface_kind_t::PHYS ) {
                    GRACE_TRACE_DBG("Inserting prolong dep EDGE from phys CORNER qid {} eid {} cid {}", std::get<0>(d), std::get<1>(d), cid) ; 
                    insert_dep(EDGE,corner.phys.task_id) ; 
                } else {
                    GRACE_TRACE_DBG("Inserting prolong dep EDGE from CORNER qid {} eid {} cid {}", std::get<0>(d), std::get<1>(d), cid) ; 
                    insert_dep(EDGE,corner.data.task_id) ; 
                }
                
            }
        } else {
            GRACE_TRACE_DBG("Inserting prolong dep CORNER qid {} cid {}", std::get<0>(d), std::get<1>(d)) ; 
            insert_dep(CORNER, ghost_array[std::get<0>(d)].corners[std::get<1>(d)].data.task_id) ; 
            for( auto eid: amr::detail::c2e[std::get<1>(d)] ) {
                auto& edge = ghost_array[std::get<0>(d)].edges[eid] ;
                if ( !edge.filled ) {
                    GRACE_TRACE_DBG("Inserting prolong dep CORNER from virtual EDGE qid {} cid {} eid {}", std::get<0>(d), std::get<1>(d), eid) ; 
                    insert_dep(CORNER, edge.data.full.task_id) ;
                } else if ( edge.kind == interface_kind_t::PHYS ) {
                    GRACE_TRACE_DBG("Inserting prolong dep CORNER from phys EDGE qid {} cid {} eid {}", std::get<0>(d), std::get<1>(d), eid) ; 
                    insert_dep(CORNER, edge.data.phys.task_id) ;
                } else {
                    GRACE_TRACE_DBG("Inserting prolong dep CORNER from EDGE qid {} cid {} eid {}", std::get<0>(d), std::get<1>(d), eid) ; 
                    insert_dep(CORNER, edge.data.full.task_id) ;
                } 
            }
        }
    } ; 

    auto const set_task_id = [&] (
        amr::element_kind_t elem_kind, 
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
                for( int istag=STAG_FACEX; istag<=STAG_FACEZ; ++istag) 
                    face.data.full.task_id[istag] = tid ;
            } else if (elem_kind == amr::element_kind_t::EDGE) {
                auto& edge = ghost_array[_qid].edges[_eid] ; 
                for( int istag=STAG_FACEX; istag<=STAG_FACEZ; ++istag) 
                    edge.data.full.task_id[istag] = tid ;
            } else {
                auto& corner = ghost_array[_qid].corners[_eid] ;
                for( int istag=STAG_FACEX; istag<=STAG_FACEZ; ++istag)  
                    corner.data.task_id[istag] = tid ;
            }
        }
        
    } ; 

    // loop through bucket, fill
    for( int kind=0; kind<3 ; ++kind) { // element kind 
        for( auto const& d: prolong_tasks[kind]) { 
            auto [_qid,_cid,_eid,_has_fine] = get_info(kind,d) ; 
            unpack_dependencies(kind,d) ; 

            qid[kind].push_back(_qid) ; 
            cid[kind].push_back(_cid) ; 
            eid[kind].push_back(_eid) ; 
            has_fine[kind].push_back(_has_fine) ; 
            if ( kind == 2 and std::get<0>(d) == 0 and std::get<1>(d) == 7 ) {
                GRACE_TRACE("Here! has_fine? lower: {} {} {},  upper: {} {} {}"
                           , _has_fine[0][0], _has_fine[0][1], _has_fine[0][2]
                           , _has_fine[1][0], _has_fine[1][1], _has_fine[1][2]) ; 
            }
        }
    }

    task_id_t tid ; 
    if ( qid[FACE].size() > 0 ) 
    {
        tid = make_div_preserving_prolongation_task<FACE>(
            qid[FACE], cid[FACE], eid[FACE], has_fine[FACE], deps[FACE], 
            stream, task_counter, state, coarse_buffers,
            nx, nv, ngz, task_list 
        ) ; 
        set_task_id(FACE,qid[FACE],eid[FACE],tid) ; 
    }
    if ( qid[EDGE].size() > 0 ){
        tid = make_div_preserving_prolongation_task<EDGE>(
            qid[EDGE], cid[EDGE], eid[EDGE], has_fine[EDGE], deps[EDGE], 
            stream, task_counter, state, coarse_buffers,
            nx, nv, ngz, task_list 
        ) ; 
        set_task_id(EDGE,qid[EDGE],eid[EDGE],tid) ; 
    }
    if ( qid[CORNER].size() > 0 ) 
    {
        tid = make_div_preserving_prolongation_task<CORNER>(
            qid[CORNER], cid[CORNER], eid[CORNER], has_fine[CORNER], deps[CORNER], 
            stream, task_counter, state, coarse_buffers,
            nx, nv, ngz, task_list 
        ) ; 
        set_task_id(CORNER,qid[CORNER],eid[CORNER],tid) ;
    }
}
} /* namespace grace*/
#endif 