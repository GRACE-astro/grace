/**
 * @file reflux_task.hh
 * @author Marie Cassing (mcassing@itp.uni-frankfurt.de)
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
//#include <grace/amr/ghostzone_kernels/copy_kernels.hh>
//#include <grace/amr/ghostzone_kernels/phys_bc_kernels.hh>
//#include <grace/amr/ghostzone_kernels/restrict_kernels.hh>
//#include <grace/amr/ghostzone_kernels/pack_unpack_kernels.hh>
#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/reflux_kernels.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <grace/data_structures/memory_defaults.hh>
#include <grace/data_structures/variables.hh>

#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <unordered_set>
#include <vector>
#include <numeric>
//#include <memory>
//#include <tuple>

#ifndef GRACE_AMR_GHOSTZONE_KERNELS_REFLUXING_TASK_FACTORY_HH
#define GRACE_AMR_GHOSTZONE_KERNELS_REFLUXING_TASK_FACTORY_HH
namespace grace { 
    

inline gpu_task_t make_gpu_sum_flux_faces_task(
      std::vector<gpu_task_desc_t> const& bucket,
      std::vector<quad_neighbors_descriptor_t>& ghost_array,
      grace::var_array_t data_fine,   // fine face flux registers (3D view)
      grace::var_array_t data_coarse, // coarse face flux registers (3D view)
      device_stream_t& stream,
      VEC(std::size_t nx, std::size_t ny, std::size_t nz), std::size_t ngz, std::size_t nv,
      task_id_t& task_counter,
      std::vector<std::unique_ptr<task_t>>& /*task_list*/)
{
    using namespace grace::amr ;

    GRACE_TRACE("Recording GPU-reflux task (tid {}). Number of faces {}", task_counter, bucket.size());

    // Per-interface metadata
    Kokkos::View<std::size_t*> src_qid("reflux_src_qid", bucket.size());
    Kokkos::View<std::size_t*> dst_qid("reflux_dst_qid", bucket.size());
    Kokkos::View<uint8_t*>     src_face("reflux_src_face", bucket.size());
    Kokkos::View<uint8_t*>     dst_face("reflux_dst_face", bucket.size());
    Kokkos::View<uint8_t*>     child_id("reflux_child_id", bucket.size()); // which quarter on the coarse face

    auto h_src_q = Kokkos::create_mirror_view(src_qid);
    auto h_dst_q = Kokkos::create_mirror_view(dst_qid);
    auto h_src_f = Kokkos::create_mirror_view(src_face);
    auto h_dst_f = Kokkos::create_mirror_view(dst_face);
    auto h_chid  = Kokkos::create_mirror_view(child_id);

    std::size_t i = 0;
    for (auto const& d : bucket) {
        const auto qidx = std::get<0>(d);
        const auto fidx = std::get<1>(d);
        auto& fdesc = ghost_array[qidx].faces[fidx];

        ASSERT(fdesc.level_diff == level_diff_t::FINER, "Only Coarse Cells need buffer for Refluxing");

        const auto c = fdesc.child_id;                 // child position relative to parent face (0..3 on that face)
        h_chid(i)  = static_cast<uint8_t>(c);

        // fine child quadrant -> coarse parent quadrant
        h_src_q(i) = fdesc.data.hanging.quad_id[c];
        h_dst_q(i) = fdesc.data.full.quad_id;

        // orientation (face IDs 0..5) on each side
        // if you do not store per-child face_id on the hanging side, you can use fdesc.face_id_hanging

        // !!!!What Marie had!!!
        //h_src_f(i) = fdesc.data.hanging.face_id[c];
        //h_dst_f(i) = fdesc.data.full.face_id;
        
        // Had to change it to this KEN
        // I am not sure about this logic. I tried to change as little as possible
        h_src_f(i) = fdesc.face;
        h_dst_f(i) = fdesc.face;

        ++i;
    }

    Kokkos::deep_copy(src_qid, h_src_q);
    Kokkos::deep_copy(dst_qid, h_dst_q);
    Kokkos::deep_copy(src_face, h_src_f);
    Kokkos::deep_copy(dst_face, h_dst_f);
    Kokkos::deep_copy(child_id, h_chid);

    // Functor: sums 2x2 fine subfaces into the correct quarter of the coarse face plane
    reflux_sum_faces_op<decltype(data_fine)> functor{
        data_fine, data_coarse, src_qid, dst_qid, src_face, dst_face, child_id,
        VEC(nx,ny,nz), ngz
    };

    Kokkos::DefaultExecutionSpace exec_space{stream};

    // Iterate over a *coarse* face plane: nx/2 × nx/2 (2:1), with ngz depth along the normal
    auto ext = get_iter_range<element_kind_t::FACE>(ngz, nx/2, nv, bucket.size());
    using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<5, Kokkos::Iterate::Left>>;
    Policy policy(exec_space,
                  {0, 0, 0, 0, 0},                  // ig, j, k, iv, iq
                  {ext[0], ext[1], ext[2], ext[3], ext[4]});

    gpu_task_t task{};
    task._run = [functor, policy](view_alias_t alias) mutable {
        functor.set_data_ptr(alias);
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        GRACE_TRACE("Reflux sum start.");
        #endif
        Kokkos::parallel_for("reflux_sum_faces", policy, functor);
        #ifdef INSERT_FENCE_DEBUG_TASKS_
        Kokkos::fence();
        GRACE_TRACE("Reflux sum end.");
        #endif
    };
    task.stream  = &stream;
    task.task_id = task_counter++;
    return task;
}

} // namespace grace

#endif // GRACE_AMR_GHOSTZONE_KERNELS_REFLUXING_TASK_FACTORY_HH