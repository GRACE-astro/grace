/**
 * @file restrict_kernels.hh
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


void insert_prolongation_tasks(
    bucket_t const & prolong_tasks,
    std::vector<quad_neighbors_descriptor_t> & ghost_array, 
    grace::var_array_t<GRACE_NSPACEDIM> state, 
    grace::var_array_t<GRACE_NSPACEDIM> coarse_buffers,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t& task_counter,
    std::vector<std::unique_ptr<task_t>>& task_list 
)
{

    std::array<std::vector<size_t>,3> qid, cid ; 
    std::array<std::vector<uint8_t>,3> qid, cid ; 
    std::array<std::unordered_set<task_id_t>,3> deps ;

    auto insert_dep = [&] (int elem, task_id_t const& tid) {
        if ( tid == UNSET_TASK_ID ) {
            ERROR("Unset task_id") ; 
        } else {
            deps[elem].insert(tid) ;        
        }
    };

    auto const get_info = [&] (int _kind, gpu_task_desc_t& d) {
        using namespace amr ; 
        amr::element_kind_t kind = static_cast<amr::element_kind_t>(_kind) ; 

        return {std::get<0>(d), ghost_array[std::get<0>(d)].cbuf_id, std::get<1>(d)} ; 
    } ; 

    // we depend on all nearby elements 
    // nothing across here can be FINER 
    // due to 2:1 balance 
    auto const unpack_dependencies = [&] (int _kind, gpu_task_desc_t& d) {
        using namespace amr ; 
        amr::element_kind_t kind = static_cast<amr::element_kind_t>(_kind) ; 

        if (kind == FACE) {
            for( auto eid: amr::detail::f2e[std::get<1>(d)] ) {
                insert_dep(FACE, ghost_array[std::get<0>(d)].edges[eid].data.full.task_id) ; 
            }
        } else if (kind == EDGE) {
            for( auto fid: amr::detail::e2f[std::get<1>(d)] ) {
                insert_dep(EDGE,ghost_array[std::get<0>(d)].faces[fid].data.full.task_id)
            }
            for( auto cid: amr::detail::e2c[std::get<1>(d)] ) {
                insert_dep(EDGE,ghost_array[std::get<0>(d)].corners[cid].data.task_id) ; 
            }
        } else {
            for( auto eid: amr::detail::c2e[std::get<1>(d)] ) {
                insert_dep(CORNER,ghost_array[std::get<0>(d)].edges[eid].data.full.task_id) ; 
            }
        }
    } ; 

    auto const set_task_id = [&] (element_kind_t elem_kind, gpu_task_desc_t& d) {
        if ( elem_kind == amr::element_kind_t::FACE ) {
            auto& face = ghost_array[std::get<0>(d)].faces[std::get<1>(d)] ; 
            face.data.full.task_id = task_counter ;
        } else if (elem_kind == amr::element_kind_t::EDGE) {
            auto& edge = ghost_array[std::get<0>(d)].edges[std::get<1>(d)] ; 
            edge.data.full.task_id = task_counter ;
        } else {
            auto& corner = ghost_array[std::get<0>(d)].corners[std::get<1>(d)] ; 
            corner.data.task_id = task_counter ;
        }
    }

    // loop through bucket, fill
    for( int kind=0; kind<3 ; ++kind) { // element kind 
        for( auto const& d: prolong_tasks[kind]) { 
            auto [_qid,_cid,_eid] = get_info(kind,d) ; 
            unpack_dependencies(kind,d) ; 

            qid[kind].push_back(_qid) ; 
            cid[kind].push_back(_cid) ; 
            eid[kind].push_back(_eid) ; 
            // write back tid 
            set_task_id(static_cast<element_kind_t>(kind),d) ; 
        }
    }
}

}

#endif 