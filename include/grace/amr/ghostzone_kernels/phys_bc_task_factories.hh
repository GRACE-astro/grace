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

#include <algorithm>
#include <unordered_set>
#include <vector>
#include <numeric>
#include <unordered_map>

#ifndef GRACE_AMR_GHOSTZONE_KERNELS_PHYS_BC_TASK_FACTORY_HH
#define GRACE_AMR_GHOSTZONE_KERNELS_PHYS_BC_TASK_FACTORY_HH

namespace grace {

// Forward decl — definition below, used inside make_gpu_phys_bc_task.
inline void compute_bounds_classical_host(
    amr::element_kind_t elem_kind,
    amr::element_kind_t bc_kind,
    bool extended,
    int8_t const dir[3],
    uint8_t eid,
    size_t nx, size_t ny, size_t nz, size_t ngz,
    int lmin[3], int idir[3], int extents[3],
    int pdim[3], int npdim[3]) ;

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
    Kokkos::View<double*[3]> var_refl,
    device_stream_t& stream,
    task_id_t& task_counter,
    grace::var_array_t data_array,
    size_t nx, size_t ny, size_t nz, size_t nv, size_t ngz,
    std::vector<std::unique_ptr<task_t>>& task_list, bool is_cbuf=false
)
{
    auto& idx = grace::variable_list::get().getinvspacings() ;
    auto coords = grace::coordinate_system::get().get_device_coord_system();

    bool rx = get_param<bool>("amr","reflection_symmetries", "x") ;
    bool ry = get_param<bool>("amr","reflection_symmetries", "y") ;
    bool rz = get_param<bool>("amr","reflection_symmetries", "z") ;

    GRACE_TRACE("Registering phys-bc task ({}-{}, tid {}) number of elements {} staggering {}",
        detail::elem_kind_names[static_cast<int>(elem_kind)],
        detail::elem_kind_names[static_cast<int>(bc_kind)], task_counter, qid_h.size(), static_cast<int>(stag)) ;
    int const N_elems = static_cast<int>(qid_h.size()) ;
    Kokkos::View<size_t*> qid_d{"qid", qid_h.size()};
    Kokkos::View<uint8_t*> eid_d{"eid", qid_h.size()} ;
    Kokkos::View<int8_t*[3]> dir_d{"dir", qid_h.size()} ;
    // Deprecated kernel-side views — still populated as empty placeholders
    // so the `phys_bc_op` ctor signature keeps taking them, but the MDRange
    // operators don't read them (the cbuf EDGE-FACE extension is baked into
    // `bnd_*` at setup time below).
    Kokkos::View<int*[3]> ext_d{"extension", qid_h.size()} ;
    Kokkos::View<int*[3]> off_d{"offset", qid_h.size()} ;

    grace::deep_copy_vec_to_view(qid_d,qid_h) ;
    grace::deep_copy_vec_to_view(eid_d,eid_h) ;
    grace::deep_copy_vec_to_2D_view(dir_d,dir_h) ;

    // ---- Precompute per-element bounds on host ----
    auto const off = get_index_staggerings(stag) ;
    size_t const snx = nx + off[0] ;
    size_t const sny = ny + off[1] ;
    size_t const snz = nz + off[2] ;

    Kokkos::View<int*[3]> bnd_lmin_d   {"bnd_lmin",    qid_h.size()} ;
    Kokkos::View<int*[3]> bnd_idir_d   {"bnd_idir",    qid_h.size()} ;
    Kokkos::View<int*[3]> bnd_extents_d{"bnd_extents", qid_h.size()} ;
    Kokkos::View<int*[3]> bnd_pdim_d   {"bnd_pdim",    qid_h.size()} ;
    Kokkos::View<int*[3]> bnd_npdim_d  {"bnd_npdim",   qid_h.size()} ;

    auto bnd_lmin_h    = Kokkos::create_mirror_view(bnd_lmin_d) ;
    auto bnd_idir_h    = Kokkos::create_mirror_view(bnd_idir_d) ;
    auto bnd_extents_h = Kokkos::create_mirror_view(bnd_extents_d) ;
    auto bnd_pdim_h    = Kokkos::create_mirror_view(bnd_pdim_d) ;
    auto bnd_npdim_h   = Kokkos::create_mirror_view(bnd_npdim_d) ;

    // Per-element `max_ext_par` is the max over the parallel axes — used
    // to size the MDRange policy.  FACE has 2 parallel axes, EDGE 1,
    // CORNER none; the value is only consulted for FACE / EDGE.
    int max_ext_par = 0 ;

    for (int iq = 0; iq < N_elems; ++iq) {
        int8_t const d[3] = {dir_h[iq][0], dir_h[iq][1], dir_h[iq][2]} ;
        uint8_t const _eid = eid_h[iq] ;
        int lmin[3], idir[3], extents[3], pdim[3] = {0,0,0}, npdim[3] = {0,0,0} ;
        compute_bounds_classical_host(
            elem_kind, bc_kind, /*extended*/ false,
            d, _eid, snx, sny, snz, ngz,
            lmin, idir, extents, pdim, npdim
        ) ;

        // Cbuf EDGE-FACE along-edge extension — was previously done
        // per-thread inside the kernel via `extents += exloop / lmin += offloop`.
        // Folded into the precomputed `bnd_*` here so the kernel stays
        // cbuf-agnostic and all launches share one code path.
        if constexpr (elem_kind == amr::element_kind_t::EDGE
                      && bc_kind == amr::element_kind_t::FACE) {
            if (is_cbuf) {
                int const ax = _eid / 4 ;
                extents[ax] += 2 * static_cast<int>(ngz) ;
                lmin[ax]    -= static_cast<int>(ngz) ;
            }
        }

        for (int ii = 0; ii < 3; ++ii) {
            bnd_lmin_h(iq,ii)    = lmin[ii] ;
            bnd_idir_h(iq,ii)    = idir[ii] ;
            bnd_extents_h(iq,ii) = extents[ii] ;
            bnd_pdim_h(iq,ii)    = pdim[ii] ;
            bnd_npdim_h(iq,ii)   = npdim[ii] ;
        }

        if constexpr (elem_kind == amr::element_kind_t::FACE) {
            max_ext_par = std::max(max_ext_par, extents[pdim[0]]) ;
            max_ext_par = std::max(max_ext_par, extents[pdim[1]]) ;
        } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
            max_ext_par = std::max(max_ext_par, extents[pdim[0]]) ;
        }
    }
    Kokkos::deep_copy(bnd_lmin_d,    bnd_lmin_h) ;
    Kokkos::deep_copy(bnd_idir_d,    bnd_idir_h) ;
    Kokkos::deep_copy(bnd_extents_d, bnd_extents_h) ;
    Kokkos::deep_copy(bnd_pdim_d,    bnd_pdim_h) ;
    Kokkos::deep_copy(bnd_npdim_d,   bnd_npdim_h) ;

    auto exec_space = grace::make_exec_space(stream) ;

    gpu_task_t task{} ;

    amr::phys_bc_op<elem_kind,bc_kind,decltype(data_array)> functor{
       data_array, data_array, idx, coords, qid_d, eid_d, dir_d,
       ext_d, off_d, var_refl, var_bc,
       VEC(snx, sny, snz), ngz, nv, is_cbuf, stag, rx, ry, rz
    } ;
    functor.bnd_lmin    = bnd_lmin_d ;
    functor.bnd_idir    = bnd_idir_d ;
    functor.bnd_extents = bnd_extents_d ;
    functor.bnd_pdim    = bnd_pdim_d ;
    functor.bnd_npdim   = bnd_npdim_d ;

    int const N_vars = static_cast<int>(nv) ;

    #ifdef GRACE_ENABLE_Z4C_METRIC
    constexpr bool need_z4c_constr = (stag == STAG_CENTER) ;
    #else
    constexpr bool need_z4c_constr = false ;
    #endif

    // One MDRange policy per elem_kind.  `bc_kind` affects bounds (already
    // baked into bnd_*) but never changes the iteration-space shape.
    if constexpr (elem_kind == amr::element_kind_t::FACE) {
        Kokkos::MDRangePolicy<Kokkos::Rank<4>, amr::phys_bc_md_face_tag> bc_policy(
            exec_space,
            {0, 0, 0, 0},
            {max_ext_par, max_ext_par, N_vars, N_elems}
        ) ;
        if constexpr (need_z4c_constr) {
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, amr::phys_bc_md_face_z4c_tag> constr_policy(
                exec_space,
                {0, 0, 0},
                {max_ext_par, max_ext_par, N_elems}
            ) ;
            task._run = [functor, bc_policy, constr_policy] (view_alias_t alias) mutable {
                functor.template set_data_ptr<stag>(alias) ;
                Kokkos::parallel_for("fill_phys_ghostzones_face",      bc_policy,     functor) ;
                Kokkos::parallel_for("fill_phys_ghostzones_face_z4c",  constr_policy, functor) ;
            } ;
        } else {
            task._run = [functor, bc_policy] (view_alias_t alias) mutable {
                functor.template set_data_ptr<stag>(alias) ;
                Kokkos::parallel_for("fill_phys_ghostzones_face", bc_policy, functor) ;
            } ;
        }
    } else if constexpr (elem_kind == amr::element_kind_t::EDGE) {
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, amr::phys_bc_md_edge_tag> bc_policy(
            exec_space,
            {0, 0, 0},
            {max_ext_par, N_vars, N_elems}
        ) ;
        if constexpr (need_z4c_constr) {
            Kokkos::MDRangePolicy<Kokkos::Rank<2>, amr::phys_bc_md_edge_z4c_tag> constr_policy(
                exec_space,
                {0, 0},
                {max_ext_par, N_elems}
            ) ;
            task._run = [functor, bc_policy, constr_policy] (view_alias_t alias) mutable {
                functor.template set_data_ptr<stag>(alias) ;
                Kokkos::parallel_for("fill_phys_ghostzones_edge",     bc_policy,     functor) ;
                Kokkos::parallel_for("fill_phys_ghostzones_edge_z4c", constr_policy, functor) ;
            } ;
        } else {
            task._run = [functor, bc_policy] (view_alias_t alias) mutable {
                functor.template set_data_ptr<stag>(alias) ;
                Kokkos::parallel_for("fill_phys_ghostzones_edge", bc_policy, functor) ;
            } ;
        }
    } else {
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, amr::phys_bc_md_corner_tag> bc_policy(
            exec_space,
            {0, 0},
            {N_vars, N_elems}
        ) ;
        if constexpr (need_z4c_constr) {
            Kokkos::RangePolicy<amr::phys_bc_md_corner_z4c_tag> constr_policy(
                exec_space, 0, N_elems
            ) ;
            task._run = [functor, bc_policy, constr_policy] (view_alias_t alias) mutable {
                functor.template set_data_ptr<stag>(alias) ;
                Kokkos::parallel_for("fill_phys_ghostzones_corner",     bc_policy,     functor) ;
                Kokkos::parallel_for("fill_phys_ghostzones_corner_z4c", constr_policy, functor) ;
            } ;
        } else {
            task._run = [functor, bc_policy] (view_alias_t alias) mutable {
                functor.template set_data_ptr<stag>(alias) ;
                Kokkos::parallel_for("fill_phys_ghostzones_corner", bc_policy, functor) ;
            } ;
        }
    }

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

// Host-side mirror of `phys_bc_op::compute_bounds` + `compute_zero_dir`,
// dispatched on runtime element/BC kind.  Used by every phys-BC task
// factory to precompute per-element geometry so the GPU kernel can skip
// `compute_bounds` per-thread and instead read small per-element Views.
//
// Handles all combinations of (elem_kind, bc_kind, extended) that the GPU
// operator() family supports:
//   - classical (extended=false) FACE-FACE:     interior slab  [ngz, n+ngz)
//   - classical (extended=false) EDGE-FACE:     along-edge full, perp ghost
//   - classical (extended=false) CORNER-*:      ghost-only per axis
//   - classical (extended=false) EDGE-EDGE:     ghost per ghost dir
//   - extended (FACE-FACE only):                full extent  [0, n+2*ngz)
//
// The *classical* FACE-FACE sweep deliberately *skips* the ghost zones at
// face corners/edges — those are written by separate EDGE-*/CORNER-* kernels
// after FACE-FACE, ordered by task-graph dependencies.  The extended variant
// absorbs those regions into the face sweep (FACE_EXT fused kernel only).
//
// Caller passes the staggered (nx, ny, nz) — the same shape the functor
// sees.  `eid` matters only for elem_kind != FACE and for EDGE+FACE.
inline void compute_bounds_classical_host(
    amr::element_kind_t elem_kind,
    amr::element_kind_t bc_kind,
    bool extended,
    int8_t const dir[3],
    uint8_t eid,
    size_t nx, size_t ny, size_t nz, size_t ngz,
    int lmin[3], int idir[3], int extents[3],
    int pdim[3], int npdim[3])
{
    size_t const ncells[3] = {nx, ny, nz};
    int npc = 0, pc = 0;
    for (int ii = 0; ii < 3; ++ii) {
        int8_t const d = dir[ii];
        size_t const n = ncells[ii];
        if (d < 0) {
            lmin[ii]    = static_cast<int>(ngz) - 1;
            idir[ii]    = -1;
            extents[ii] = static_cast<int>(ngz);
        } else if (d > 0) {
            lmin[ii]    = static_cast<int>(n + ngz);
            idir[ii]    = +1;
            extents[ii] = static_cast<int>(ngz);
        } else {
            // Mirror of phys_bc_op::compute_zero_dir — runtime dispatch
            // because elem/bc are compile-time template params inside the
            // kernel but runtime enums here.
            if (elem_kind == amr::element_kind_t::CORNER) {
                lmin[ii]    = ((eid >> ii) & 1) ? static_cast<int>(n + ngz) : 0;
                idir[ii]    = +1;
                extents[ii] = static_cast<int>(ngz);
            } else if (elem_kind == amr::element_kind_t::EDGE
                       && bc_kind == amr::element_kind_t::FACE) {
                if (eid / 4 == ii) {
                    // along-edge direction -> full interior sweep
                    lmin[ii]    = static_cast<int>(ngz);
                    idir[ii]    = +1;
                    extents[ii] = static_cast<int>(n);
                } else {
                    int side_bit;
                    if (eid < 4) {
                        side_bit = (eid >> ((ii + 1) % 2)) & 1;
                    } else if (eid < 8) {
                        side_bit = (eid >> (ii / 2)) & 1;
                    } else {
                        side_bit = (eid >> ii) & 1;
                    }
                    lmin[ii]    = side_bit ? static_cast<int>(n + ngz) : 0;
                    idir[ii]    = +1;
                    extents[ii] = static_cast<int>(ngz);
                }
            } else if (extended
                       && elem_kind == amr::element_kind_t::FACE
                       && bc_kind   == amr::element_kind_t::FACE) {
                // FACE_EXT (fused) — covers full face + adjacent ghost regions
                lmin[ii]    = 0;
                idir[ii]    = +1;
                extents[ii] = static_cast<int>(n + 2 * ngz);
            } else {
                // Classical interior-only face sweep — skips ghost zones.
                // EDGE-* and CORNER-* kernels fill those separately.
                lmin[ii]    = static_cast<int>(ngz);
                idir[ii]    = +1;
                extents[ii] = static_cast<int>(n);
            }
        }
        if (d != 0) {
            npdim[npc++] = ii;
        } else {
            pdim[pc++] = ii;
        }
    }
}

// Fused FACE_EXT kernel: handles (FACE-FACE) + absorbed (EDGE-FACE of type=FACE)
// + absorbed (CORNER-FACE of type=FACE) in a single launch.  The per-quad
// `guard_mask_h` decides which adjacent edge/corner regions are written by
// extending the face-sweep; bits are laid out per `phys_bc_op::guard_mask`.
template< var_staggering_t stag >
task_id_t
make_gpu_phys_bc_face_ext_task(
    std::vector<size_t> const& qid_h,
    std::vector<uint8_t> const& eid_h,
    std::vector<std::array<int8_t,3>> const& dir_h,
    std::vector<uint8_t> const& guard_mask_h,
    std::unordered_set<task_id_t> const& deps,
    Kokkos::View<bc_t*> var_bc,
    Kokkos::View<double*[3]> var_refl,
    device_stream_t& stream,
    task_id_t& task_counter,
    grace::var_array_t data_array,
    size_t nx, size_t ny, size_t nz, size_t nv, size_t ngz,
    std::vector<std::unique_ptr<task_t>>& task_list
)
{
    auto& idx = grace::variable_list::get().getinvspacings() ;
    auto coords = grace::coordinate_system::get().get_device_coord_system();

    bool rx = get_param<bool>("amr","reflection_symmetries", "x") ;
    bool ry = get_param<bool>("amr","reflection_symmetries", "y") ;
    bool rz = get_param<bool>("amr","reflection_symmetries", "z") ;

    GRACE_TRACE("Registering phys-bc face-ext task (tid {}) number of faces {} staggering {}",
        task_counter, qid_h.size(), static_cast<int>(stag)) ;

    Kokkos::View<size_t*> qid_d{"qid_ext", qid_h.size()};
    Kokkos::View<uint8_t*> eid_d{"eid_ext", qid_h.size()} ;
    Kokkos::View<int8_t*[3]> dir_d{"dir_ext", qid_h.size()} ;
    Kokkos::View<int*[3]> ext_d{"extension_ext", qid_h.size()} ;   // always zero
    Kokkos::View<int*[3]> off_d{"offset_ext", qid_h.size()} ;      // always zero
    Kokkos::View<uint8_t*> guard_d{"guard_mask", qid_h.size()} ;

    grace::deep_copy_vec_to_view(qid_d, qid_h) ;
    grace::deep_copy_vec_to_view(eid_d, eid_h) ;
    grace::deep_copy_vec_to_2D_view(dir_d, dir_h) ;
    grace::deep_copy_vec_to_view(guard_d, guard_mask_h) ;

    // Precompute per-face geometry on host so the GPU kernel can skip
    // `compute_bounds` per-thread. Use the same staggered (nx,ny,nz) the
    // functor sees.
    auto const off = get_index_staggerings(stag) ;
    size_t const snx = nx + off[0] ;
    size_t const sny = ny + off[1] ;
    size_t const snz = nz + off[2] ;

    Kokkos::View<int*[3]> bnd_lmin_d   {"bnd_lmin_ext",    qid_h.size()} ;
    Kokkos::View<int*[3]> bnd_idir_d   {"bnd_idir_ext",    qid_h.size()} ;
    Kokkos::View<int*[3]> bnd_extents_d{"bnd_extents_ext", qid_h.size()} ;
    Kokkos::View<int*[3]> bnd_pdim_d   {"bnd_pdim_ext",    qid_h.size()} ;
    Kokkos::View<int*[3]> bnd_npdim_d  {"bnd_npdim_ext",   qid_h.size()} ;

    auto bnd_lmin_h    = Kokkos::create_mirror_view(bnd_lmin_d) ;
    auto bnd_idir_h    = Kokkos::create_mirror_view(bnd_idir_d) ;
    auto bnd_extents_h = Kokkos::create_mirror_view(bnd_extents_d) ;
    auto bnd_pdim_h    = Kokkos::create_mirror_view(bnd_pdim_d) ;
    auto bnd_npdim_h   = Kokkos::create_mirror_view(bnd_npdim_d) ;

    for (size_t iq=0; iq<qid_h.size(); ++iq) {
        int8_t const d[3] = {dir_h[iq][0], dir_h[iq][1], dir_h[iq][2]} ;
        int lmin[3], idir[3], extents[3], pdim[3]={0,0,0}, npdim[3]={0,0,0} ;
        // FACE_EXT = FACE elem, FACE bc, extended=true
        compute_bounds_classical_host(
            amr::element_kind_t::FACE, amr::element_kind_t::FACE,
            /*extended*/ true,
            d, eid_h[iq], snx, sny, snz, ngz,
            lmin, idir, extents, pdim, npdim) ;
        for (int ii=0; ii<3; ++ii) {
            bnd_lmin_h(iq,ii)    = lmin[ii] ;
            bnd_idir_h(iq,ii)    = idir[ii] ;
            bnd_extents_h(iq,ii) = extents[ii] ;
            bnd_pdim_h(iq,ii)    = pdim[ii] ;
            bnd_npdim_h(iq,ii)   = npdim[ii] ;
        }
    }
    Kokkos::deep_copy(bnd_lmin_d,    bnd_lmin_h) ;
    Kokkos::deep_copy(bnd_idir_d,    bnd_idir_h) ;
    Kokkos::deep_copy(bnd_extents_d, bnd_extents_h) ;
    Kokkos::deep_copy(bnd_pdim_d,    bnd_pdim_h) ;
    Kokkos::deep_copy(bnd_npdim_d,   bnd_npdim_h) ;

    auto exec_space = grace::make_exec_space(stream) ;

    gpu_task_t task{} ;
    amr::phys_bc_op<amr::element_kind_t::FACE,amr::element_kind_t::FACE,decltype(data_array),true> functor{
       data_array, data_array, idx, coords, qid_d, eid_d, dir_d,
       ext_d, off_d, var_refl, var_bc,
       VEC(snx, sny, snz), ngz, nv,
       /*is_cbuf*/ false, stag, rx, ry, rz,
       guard_d
    } ;
    // Wire precomputed bounds Views (struct fields are public).
    functor.bnd_lmin    = bnd_lmin_d ;
    functor.bnd_idir    = bnd_idir_d ;
    functor.bnd_extents = bnd_extents_d ;
    functor.bnd_pdim    = bnd_pdim_d ;
    functor.bnd_npdim   = bnd_npdim_d ;

    int const N_faces = static_cast<int>(qid_h.size()) ;
    int const N_vars  = static_cast<int>(nv) ;

    // MDRange bound for the two parallel in-face axes. FACE_EXT sweeps
    // `n + 2*ngz` cells along each parallel axis, so the loose upper bound
    // that covers every face regardless of which (pdim[0], pdim[1]) slot
    // lands where is max(snx, sny, snz) + 2*ngz. Smaller faces are handled
    // by the ragged-extent predicate inside the kernel.
    int const max_n = static_cast<int>(std::max({snx, sny, snz})) ;
    int const max_ext_par = max_n + 2 * static_cast<int>(ngz) ;

    Kokkos::MDRangePolicy<Kokkos::Rank<4>, amr::phys_bc_face_ext_md_tag> bc_policy(
        exec_space,
        {0, 0, 0, 0},
        {max_ext_par, max_ext_par, N_vars, N_faces}
    ) ;

    #ifdef GRACE_ENABLE_Z4C_METRIC
    constexpr bool need_z4c_constr = (stag == STAG_CENTER) ;
    #else
    constexpr bool need_z4c_constr = false ;
    #endif

    if constexpr (need_z4c_constr) {
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, amr::phys_bc_z4c_constr_tag> constr_policy(
            exec_space,
            {0, 0, 0},
            {max_ext_par, max_ext_par, N_faces}
        ) ;
        task._run = [functor, bc_policy, constr_policy] (view_alias_t alias) mutable {
            functor.template set_data_ptr<stag>(alias) ;
            #ifdef INSERT_FENCE_DEBUG_TASKS_
            GRACE_TRACE("Fill phys face-ext start") ;
            #endif
            Kokkos::parallel_for("fill_phys_ghostzones_face_ext", bc_policy, functor) ;
            Kokkos::parallel_for("fill_phys_ghostzones_face_ext_z4c_constr",
                                 constr_policy, functor) ;
            #ifdef INSERT_FENCE_DEBUG_TASKS_
            Kokkos::fence() ;
            GRACE_TRACE("Fill phys face-ext done") ;
            #endif
        };
    } else {
        task._run = [functor, bc_policy] (view_alias_t alias) mutable {
            functor.template set_data_ptr<stag>(alias) ;
            #ifdef INSERT_FENCE_DEBUG_TASKS_
            GRACE_TRACE("Fill phys face-ext start") ;
            #endif
            Kokkos::parallel_for("fill_phys_ghostzones_face_ext", bc_policy, functor) ;
            #ifdef INSERT_FENCE_DEBUG_TASKS_
            Kokkos::fence() ;
            GRACE_TRACE("Fill phys face-ext done") ;
            #endif
        };
    }

    task.stream = &stream ;
    auto tid = task_counter++ ;
    task.task_id = tid ;

    for (auto const dep_id : deps) {
        ASSERT(dep_id < task_list.size(), "Dep-id out-of-range") ;
        task._dependencies.push_back(dep_id) ;
        task_list[dep_id]->_dependents.push_back(tid) ;
    }

    task_list.push_back(std::make_unique<gpu_task_t>(std::move(task))) ;
    return tid ;
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
    uint8_t isign ; 
    for( uint8_t i=0; i<3; ++i) {
        if ( dir[i] ) {
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
    task_id_t restrict_tid,
    F&& insert_dependencies 
) 
{
    using namespace amr ;
    static const int other_dirs[3][2] = {
                        {1,2}, {0,2}, {0,1}
                    } ;
    if (kind == FACE) { // kind here is the kind of element, type is the type of BC 
        // nothing to do here 
        if ( is_cbuf ) {
            insert_dependencies(FACE,FACE,restrict_tid,is_cbuf) ;
        }
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
                // problem here: if we are in cbufs this might be 
                // virtual.. in which case the tid is not stored.
                // We need to somehow still retrieve the dependency! 
                if( !edge.filled ) {
                    int edge_dir = edge_idx / 4;
                    int isides[2] = { (edge_dir>>0)&1, (edge_dir>>1)&1 } ; 
                    for ( int iff=0; iff<2; ++iff) {
                        int fdir = other_dirs[edge_dir][iff] ; 
                        int fidx = 2 * fdir + isides[iff] ; 
                        auto const& face=ghost_array[std::get<0>(d)].faces[fidx] ; 
                        if(face.level_diff==COARSER) {
                            insert_dependencies(CORNER,FACE,face.data.full.task_id[stag], is_cbuf) ;
                        }
                    }
                } else {
                    insert_dependencies(CORNER,FACE,edge.data.full.task_id[stag], is_cbuf) ;
                }
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
    Kokkos::View<double*[3]> var_parities,
    device_stream_t& stream, 
    VEC(size_t nx, size_t ny, size_t nz), size_t ngz, size_t nv,
    task_id_t restrict_tid,
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

    // --- FACE_EXT fused bucket ---
    // Fuses non-cbuf FACE-FACE + (absorbed) EDGE-FACE + CORNER-FACE launches
    // whose cluster is entirely non-cbuf.  Cluster = (face, its 4 adjacent
    // type=FACE edges, its 4 adjacent type=FACE corners).  Cbuf-tainted
    // clusters fall through to the classical path.
    std::vector<size_t>                     fused_qid;
    std::vector<uint8_t>                    fused_eid;
    std::vector<std::array<int8_t,3>>       fused_dir;
    std::vector<uint8_t>                    fused_guard;
    std::unordered_set<uint64_t>            absorbed_faces;    // qid*6 + eid
    std::unordered_set<uint64_t>            absorbed_edges;    // qid*12 + eid
    std::unordered_set<uint64_t>            absorbed_corners;  // qid*8 + eid

    // Deps accumulated for the fused FACE_EXT kernel: union of the deps of
    // all absorbed edges/corners (the face itself has no phys-bc predecessors).
    std::unordered_set<task_id_t> fused_deps;

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

    auto insert_fused_dep = [&] (int /*elem*/, int /*bc*/, task_id_t const& tid, bool /*is_cbuf*/) {
        if ( tid == UNSET_TASK_ID ) {
            ERROR("Unset task_id") ;
        } else {
            fused_deps.insert(tid) ;
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

    // --- Pre-pass: classify face clusters as fused or classical. ---
    //
    // A face-cluster is eligible for FACE_EXT iff:
    //   - the face is PHYS, non-cbuf (type is always FACE for PHYS faces);
    //   - every adjacent type=FACE edge and corner is also non-cbuf.
    // Otherwise, the whole cluster falls through to the classical path.
    {
        auto const is_elem_cbuf_phys_face = [&] (size_t qid, int idx, int which) -> std::tuple<bool,bool> {
            // returns {is_phys_face_type, is_cbuf}
            if (which == 0) {
                auto const& e = ghost_array[qid].edges[idx];
                if (e.kind != interface_kind_t::PHYS) return {false, false};
                return {e.data.phys.type == amr::FACE, e.data.phys.in_cbuf};
            } else {
                auto const& c = ghost_array[qid].corners[idx];
                if (c.kind != interface_kind_t::PHYS) return {false, false};
                return {c.phys.type == amr::FACE, c.phys.in_cbuf};
            }
        };

        for (auto const& d : phys_bc_tasks[amr::FACE]) {
            auto const _qid = std::get<0>(d);
            auto const _eid = std::get<1>(d);
            auto const& face = ghost_array[_qid].faces[_eid];
            if (face.kind != interface_kind_t::PHYS) continue;
            if (face.data.phys.in_cbuf) continue;          // cbuf → classical
            // adj type=FACE edges
            uint8_t mask = 0;
            bool cluster_clean = true;
            uint8_t const fe = _eid;
            int adj_edges[4]    = { grace::amr::detail::f2e[fe][0], grace::amr::detail::f2e[fe][1],
                                    grace::amr::detail::f2e[fe][2], grace::amr::detail::f2e[fe][3] };
            int adj_corners[4]  = { grace::amr::detail::f2c[fe][0], grace::amr::detail::f2c[fe][1],
                                    grace::amr::detail::f2c[fe][2], grace::amr::detail::f2c[fe][3] };
            for (int b = 0; b < 4; ++b) {
                auto [is_face_t, is_cbuf] = is_elem_cbuf_phys_face(_qid, adj_edges[b], /*edge*/0);
                if (is_face_t && is_cbuf) { cluster_clean = false; break; }
                if (is_face_t) mask |= static_cast<uint8_t>(1u << b);
            }
            if (!cluster_clean) continue;
            for (int b = 0; b < 4; ++b) {
                auto [is_face_t, is_cbuf] = is_elem_cbuf_phys_face(_qid, adj_corners[b], /*corner*/1);
                if (is_face_t && is_cbuf) { cluster_clean = false; break; }
                if (is_face_t) mask |= static_cast<uint8_t>(1u << (4 + b));
            }
            if (!cluster_clean) continue;

            // Commit fused cluster.
            absorbed_faces.insert(uint64_t(_qid) * 6 + _eid);
            for (int b = 0; b < 4; ++b) {
                if ((mask >> b) & 1u)
                    absorbed_edges.insert(uint64_t(_qid) * 12 + adj_edges[b]);
            }
            for (int b = 0; b < 4; ++b) {
                if ((mask >> (4 + b)) & 1u)
                    absorbed_corners.insert(uint64_t(_qid) * 8 + adj_corners[b]);
            }
            fused_qid.push_back(_qid);
            fused_eid.push_back(_eid);
            fused_dir.emplace_back(std::array<int8_t,3>{
                face.data.phys.dir[0], face.data.phys.dir[1], face.data.phys.dir[2]});
            fused_guard.push_back(mask);
        }
    }

    auto const is_absorbed = [&] (int kind, size_t _qid, uint8_t _eid) -> bool {
        if (kind == amr::FACE)   return absorbed_faces.count(uint64_t(_qid) * 6 + _eid) > 0;
        if (kind == amr::EDGE)   return absorbed_edges.count(uint64_t(_qid) * 12 + _eid) > 0;
        return absorbed_corners.count(uint64_t(_qid) * 8 + _eid) > 0;
    };

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

            // Items absorbed into FACE_EXT: route their deps to fused_deps
            // and skip classical bucket insertion.
            bool const absorbed = (!is_cbuf && type == amr::FACE && is_absorbed(kind, _qid, _eid));
            if (absorbed) {
                grace::detail::unpack_dependencies<stag>(kind, d, ghost_array, is_cbuf, restrict_tid, insert_fused_dep);
                continue;
            }

            auto is_deferred = grace::detail::unpack_dependencies<stag>(kind, d, ghost_array, is_cbuf, restrict_tid, insert_dependencies);

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
    // FACE_EXT fused kernel: faces whose full cluster is non-cbuf.  Replaces
    // FACE-FACE + absorbed EDGE-FACE + absorbed CORNER-FACE for those quads.
    task_id_t tid ;
    if (!fused_qid.empty()) {
        tid = make_gpu_phys_bc_face_ext_task<stag>(
            fused_qid, fused_eid, fused_dir, fused_guard,
            fused_deps,
            var_bc, var_parities, stream, task_counter,
            state, nx, ny, nz, nv, ngz, task_list
        ) ;
        // write back tid to face, absorbed edges, absorbed corners
        for (size_t i = 0; i < fused_qid.size(); ++i) {
            auto _qid = fused_qid[i];
            auto _feid = fused_eid[i];
            uint8_t const mask = fused_guard[i];
            ghost_array[_qid].faces[_feid].data.phys.task_id[stag] = tid;
            for (int b = 0; b < 4; ++b) {
                if ((mask >> b) & 1u) {
                    auto eid_adj = grace::amr::detail::f2e[_feid][b];
                    ghost_array[_qid].edges[eid_adj].data.phys.task_id[stag] = tid;
                }
            }
            for (int b = 0; b < 4; ++b) {
                if ((mask >> (4 + b)) & 1u) {
                    auto cid_adj = grace::amr::detail::f2c[_feid][b];
                    ghost_array[_qid].corners[cid_adj].phys.task_id[stag] = tid;
                }
            }
        }
        // EDGE-EDGE depends on EDGE-FACE (absorbed into FACE_EXT)
        dependencies[EDGE][EDGE].insert(tid) ;
        // CORNER-EDGE depends on EDGE-FACE (absorbed into FACE_EXT)
        dependencies[CORNER][EDGE].insert(tid) ;
    }

    // face face is safe to schedule
    if ( qid[FACE][FACE].size() >0 ) {
        tid =make_gpu_phys_bc_task<FACE,FACE,stag>(
            qid[FACE][FACE],
            eid[FACE][FACE],
            dir[FACE][FACE],
            dependencies[FACE][FACE],
            var_bc, var_parities, stream, task_counter,
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
            var_bc, var_parities, stream, task_counter,
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
            var_bc, var_parities, stream, task_counter,
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
            var_bc, var_parities, stream, task_counter,
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
            var_bc, var_parities, stream, task_counter,
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
            var_bc, var_parities, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid[CORNER][EDGE],eid[CORNER][EDGE],tid) ;
    }
    
    // now for cbufs 
    if (qid_cbuf[FACE][FACE].size() > 0 ) {
        // == 
        tid = make_gpu_phys_bc_task<FACE,FACE,stag>(
            cid_cbuf[FACE][FACE],
            eid_cbuf[FACE][FACE],
            dir_cbuf[FACE][FACE],
            dependencies_cbuf[FACE][FACE],
            var_bc, var_parities, stream, task_counter,
            coarse_buffers,nx/2,ny/2,nz/2,nv,ngz,task_list,true
        ) ; 
        // write back tid 
        set_task_id(FACE,qid_cbuf[FACE][FACE],eid_cbuf[FACE][FACE],tid) ;
        dependencies_cbuf[EDGE][EDGE].insert(tid) ; 
    }
    if (qid_cbuf[EDGE][FACE].size() > 0 ) {
        // == 
        tid = make_gpu_phys_bc_task<EDGE,FACE,stag>(
            cid_cbuf[EDGE][FACE],
            eid_cbuf[EDGE][FACE],
            dir_cbuf[EDGE][FACE],
            dependencies_cbuf[EDGE][FACE],
            var_bc, var_parities, stream, task_counter,
            coarse_buffers,nx/2,ny/2,nz/2,nv,ngz,task_list,true
        ) ; 
        // write back tid 
        set_task_id(EDGE,qid_cbuf[EDGE][FACE],eid_cbuf[EDGE][FACE],tid) ;
        dependencies_cbuf[CORNER][EDGE].insert(tid) ;
        // just for cbufs! 
        dependencies_cbuf[CORNER][FACE].insert(tid) ; 
    }
    if (qid_cbuf[CORNER][FACE].size() > 0) {
        // == 
        tid = make_gpu_phys_bc_task<CORNER,FACE,stag>(
            cid_cbuf[CORNER][FACE],
            eid_cbuf[CORNER][FACE],
            dir_cbuf[CORNER][FACE],
            dependencies_cbuf[CORNER][FACE],
            var_bc, var_parities, stream, task_counter,
            coarse_buffers,nx/2,ny/2,nz/2,nv,ngz,task_list,true
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid_cbuf[CORNER][FACE],eid_cbuf[CORNER][FACE],tid) ;
        // nothing depends on this 
    }   
    if (qid_cbuf[EDGE][EDGE].size() > 0 ) {
        // == 
        tid = make_gpu_phys_bc_task<EDGE,EDGE,stag>(
            cid_cbuf[EDGE][EDGE],
            eid_cbuf[EDGE][EDGE],
            dir_cbuf[EDGE][EDGE],
            dependencies_cbuf[EDGE][EDGE],
            var_bc, var_parities, stream, task_counter,
            coarse_buffers,nx/2,ny/2,nz/2,nv,ngz,task_list,true
        ) ; 
        // write back tid 
        set_task_id(EDGE,qid_cbuf[EDGE][EDGE],eid_cbuf[EDGE][EDGE],tid) ;
        dependencies_cbuf[CORNER][CORNER].insert(tid) ; 
    }
    if (qid_cbuf[CORNER][CORNER].size() > 0 ) {
        // == 
        tid = make_gpu_phys_bc_task<CORNER,CORNER,stag>(
            cid_cbuf[CORNER][CORNER],
            eid_cbuf[CORNER][CORNER],
            dir_cbuf[CORNER][CORNER],
            dependencies_cbuf[CORNER][CORNER],
            var_bc, var_parities, stream, task_counter,
            coarse_buffers,nx/2,ny/2,nz/2,nv,ngz,task_list,true
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid_cbuf[CORNER][CORNER],eid_cbuf[CORNER][CORNER],tid) ;
    }
    if (qid_cbuf[CORNER][EDGE].size() > 0 ) {
        // == 
        tid = make_gpu_phys_bc_task<CORNER,EDGE,stag>(
            cid_cbuf[CORNER][EDGE],
            eid_cbuf[CORNER][EDGE],
            dir_cbuf[CORNER][EDGE],
            dependencies_cbuf[CORNER][EDGE],
            var_bc, var_parities, stream, task_counter,
            coarse_buffers,nx/2,ny/2,nz/2,nv,ngz,task_list,true
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid_cbuf[CORNER][EDGE],eid_cbuf[CORNER][EDGE],tid) ;
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
    Kokkos::View<double*[3]> var_parities,
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
            auto dummy2 = grace::detail::unpack_dependencies<stag>(kind, d, ghost_array, false, UNSET_TASK_ID, insert_dependencies);

            qid[kind][type].push_back(_qid) ; 
            eid[kind][type].push_back(_eid) ; 
            dir[kind][type].emplace_back(std::array{dx,dy,dz}) ; 
        }
    }

    task_id_t tid ; 
    if (qid[FACE][FACE].size()>0){
        tid = make_gpu_phys_bc_task<FACE,FACE,stag>(
            qid[FACE][FACE],
            eid[FACE][FACE],
            dir[FACE][FACE],
            dependencies[FACE][FACE],
            var_bc, var_parities, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(FACE,qid[FACE][FACE],eid[FACE][FACE],tid) ;
        dependencies[EDGE][EDGE].insert(tid) ; 
    }
    if (qid[EDGE][FACE].size() > 0) {
        tid = make_gpu_phys_bc_task<EDGE,FACE,stag>(
            qid[EDGE][FACE],
            eid[EDGE][FACE],
            dir[EDGE][FACE],
            dependencies[EDGE][FACE],
            var_bc, var_parities, stream, task_counter,
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
            var_bc, var_parities, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid[CORNER][FACE],eid[CORNER][FACE],tid) ;
    }
    if (qid[EDGE][EDGE].size()>0){
        tid = make_gpu_phys_bc_task<EDGE,EDGE,stag>(
            qid[EDGE][EDGE],
            eid[EDGE][EDGE],
            dir[EDGE][EDGE],
            dependencies[EDGE][EDGE],
            var_bc, var_parities, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(EDGE,qid[EDGE][EDGE],eid[EDGE][EDGE],tid) ;
        dependencies[CORNER][CORNER].insert(tid) ; 
    }
    if (qid[CORNER][CORNER].size()>0){
        tid = make_gpu_phys_bc_task<CORNER,CORNER,stag>(
            qid[CORNER][CORNER],
            eid[CORNER][CORNER],
            dir[CORNER][CORNER],
            dependencies[CORNER][CORNER],
            var_bc, var_parities, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid[CORNER][CORNER],eid[CORNER][CORNER],tid) ;
    }
    if (qid[CORNER][EDGE].size()>0){
        tid = make_gpu_phys_bc_task<CORNER,EDGE,stag>(
            qid[CORNER][EDGE],
            eid[CORNER][EDGE],
            dir[CORNER][EDGE],
            dependencies[CORNER][EDGE],
            var_bc, var_parities, stream, task_counter,
            state,nx,ny,nz,nv,ngz,task_list
        ) ; 
        // write back tid 
        set_task_id(CORNER,qid[CORNER][EDGE],eid[CORNER][EDGE],tid) ;
    }
}

template< var_staggering_t stag >
void tag_bcs_in_cbuf(
    std::unordered_set<size_t> const& cbuf_qid,
    std::vector<quad_neighbors_descriptor_t>& ghost_array
)
{
    for (auto const& qid : cbuf_qid) {
        for(int8_t f=0; f<P4EST_FACES; ++f) {
            auto& face = ghost_array[qid].faces[f] ; 
            if (face.kind == interface_kind_t::PHYS) continue ;  
            if (!(face.level_diff == level_diff_t::COARSER)) continue ; 
            int af[4],ae[8],ac[4];
            grace::detail::get_face_prolong_dependencies(f,af,ae,ac) ;
            for( int iaf=0; iaf<4; ++iaf) {
                auto& adjacent_face = ghost_array[qid].faces[af[iaf]] ; 
                if ( adjacent_face.kind == interface_kind_t::PHYS ) {
                    adjacent_face.data.phys.in_cbuf = true ;
                }
            }
            for( int iae=0; iae<8; ++iae) {
                auto& adj_edge = ghost_array[qid].edges[ae[iae]] ;
                if ( !adj_edge.filled ) continue ; 
                if ( adj_edge.kind == interface_kind_t::PHYS ) {
                    adj_edge.data.phys.in_cbuf = true ;
                }
            }
            for( int iac=0; iac<4; ++iac) {
                auto& adj_corner = ghost_array[qid].corners[ac[iac]] ;
                if ( !adj_corner.filled ) continue ; 
                if ( adj_corner.kind == interface_kind_t::PHYS) {
                    adj_corner.phys.in_cbuf=true ;  
                    continue ; 
                }
            }
        }
        for( int8_t e=0; e<12; ++e){
            auto& edge = ghost_array[qid].edges[e] ; 
            bool need_neighbor_restrict = (!edge.filled) ; 
            if ( edge.filled) {
                need_neighbor_restrict |= (edge.level_diff == level_diff_t::COARSER) and (edge.kind != interface_kind_t::PHYS);
            }
            if (!need_neighbor_restrict) continue ; 
            int af[4],ae[4],ac[2];
            grace::detail::get_edge_prolong_dependencies(e,af,ae,ac) ;
            for( int iaf=0; iaf<4; ++iaf) {
                auto& adjacent_face = ghost_array[qid].faces[af[iaf]] ; 
                if ( adjacent_face.kind == interface_kind_t::PHYS ) {
                    adjacent_face.data.phys.in_cbuf = true ;
                }
            }
            for( int iae=0; iae<4; ++iae) {
                auto& adj_edge = ghost_array[qid].edges[ae[iae]] ;
                if ( !adj_edge.filled ) continue ; 
                if ( adj_edge.kind == interface_kind_t::PHYS ) {
                    adj_edge.data.phys.in_cbuf = true ;
                }
            }
            for( int iac=0; iac<2; ++iac) {
                auto& adj_corner = ghost_array[qid].corners[ac[iac]] ;
                if ( !adj_corner.filled ) continue ; 
                if ( adj_corner.kind == interface_kind_t::PHYS) {
                    adj_corner.phys.in_cbuf=true ;  
                    continue ; 
                }
            }
        }
        for( int8_t c=0; c<P4EST_CHILDREN; ++c){
            auto& corner = ghost_array[qid].corners[c] ; 
            bool need_neighbor_restrict = (!corner.filled) ; 
            if (corner.filled) {
                need_neighbor_restrict |= (corner.level_diff == level_diff_t::COARSER) and (corner.kind != interface_kind_t::PHYS);
            } 
            if (!need_neighbor_restrict) continue ; 
            int af[3],ae[3],ac[1];
            grace::detail::get_corner_prolong_dependencies(c,af,ae,ac) ;
            for( int iaf=0; iaf<3; ++iaf) {
                auto& adjacent_face = ghost_array[qid].faces[af[iaf]] ; 
                if ( adjacent_face.kind == interface_kind_t::PHYS ) {
                    adjacent_face.data.phys.in_cbuf = true ;
                }
            }
            for( int iae=0; iae<3; ++iae) {
                auto& adj_edge = ghost_array[qid].edges[ae[iae]] ;
                if ( !adj_edge.filled ) continue ; 
                if ( adj_edge.kind == interface_kind_t::PHYS ) {
                    adj_edge.data.phys.in_cbuf = true ;
                }  
            }
        }

    }
}

} /* namespace grace */
#endif 