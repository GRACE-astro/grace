/**
 * @file reflux_kernels.hh
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
#ifndef GRACE_AMR_BC_REFLUX_KERNELS_HH
#define GRACE_AMR_BC_REFLUX_KERNELS_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/ghostzone_kernels/index_helpers.hh>
#include <grace/amr/ghostzone_kernels/type_helpers.hh>

#include <Kokkos_Core.hpp>

namespace grace { namespace amr {

/**
 * Sum 2x2 fine subfaces into the correct quarter of the coarse face plane.
 * - Works for any face orientation (±x, ±y, ±z) via face ids and index_transformer_t.
 * - Uses child_id + view_to_cbuf_offsets<FACE> to place results in the right quarter.
 * - Writes a SUM (no averaging).
 *
 * Expected view indexing: (i, j, k, ivar, qid) on full 3D arrays (with ghosts).
 */
template<typename view_t>
struct reflux_sum_faces_op {
    view_t src_view, dest_view;                    // fine & coarse face registers (3D)
    readonly_view_t<std::size_t> src_qid, dest_qid;
    readonly_view_t<uint8_t>     src_face, dest_face;   // fine/coarse face ids (0..5)
    readonly_view_t<uint8_t>     child_id;              // fine child position on the coarse face (0..3)
    index_transformer_t          transf;
    std::size_t                  nx, ngz;               // needed for plane offsets

    reflux_sum_faces_op(view_t _src_view, view_t _dest_view,
                        Kokkos::View<std::size_t*> _src_qid,
                        Kokkos::View<std::size_t*> _dest_qid,
                        Kokkos::View<uint8_t*>     _src_face,
                        Kokkos::View<uint8_t*>     _dest_face,
                        Kokkos::View<uint8_t*>     _child_id,
                        VEC(std::size_t _nx, std::size_t _ny, std::size_t _nz),
                        std::size_t _ngz)
      : src_view(_src_view)
      , dest_view(_dest_view)
      , src_qid(_src_qid)
      , dest_qid(_dest_qid)
      , src_face(_src_face)
      , dest_face(_dest_face)
      , child_id(_child_id)
      , transf(VEC(_nx,_ny,_nz), _ngz)
      , nx(_nx)
      , ngz(_ngz)
    {}

    void set_data_ptr(view_alias_t alias) {
        src_view  = alias.get();
        dest_view = alias.get();
    }

    // Iterate: ig (depth along normal), j,k (coarse plane indices in the child's quarter),
    //          ivar, iq (interface)
    KOKKOS_INLINE_FUNCTION
    void operator()(std::size_t ig, std::size_t j, std::size_t k,
                    std::size_t ivar, std::size_t iq) const
    {
        const auto qf = src_qid(iq);                         // fine child qid
        const auto qc = dest_qid(iq);                        // coarse parent qid
        const int  fs = static_cast<int>(src_face(iq));      // fine face id
        const int  fc = static_cast<int>(dest_face(iq));     // coarse face id
        const uint8_t child = child_id(iq);                   // which quarter

        // Quarter offsets on the **coarse face plane** for this child
        std::size_t j_off, k_off, j_off_c{0UL}, k_off_c{0UL} ;
        // KEN: Too few arguments before 
        view_to_cbuf_offsets<element_kind_t::FACE>::get(
            j_off,k_off,j_off_c,k_off_c, nx, ngz, child
        );

        // 1) Map coarse face-plane (j_off + j, k_off + k) to 3D coarse indices (half resolution)
        std::size_t iC, jC, kC;
        transf.compute_indices<element_kind_t::FACE, true>(
            ig, j_off + j, k_off + k, iC, jC, kC, fc, /*half_ncells=*/true);

        // 2) Map the four fine subfaces **within this child** covering this coarse plane cell.
        //    In the child's local face plane, indices run on fine resolution, so just use 2*j + dj, 2*k + dk.
        std::size_t iF, jF, kF;
        double sum = 0.0;

        transf.compute_indices<element_kind_t::FACE, true>(
            ig, 2*j + 0, 2*k + 0, iF, jF, kF, fs, /*half_ncells=*/false);
        sum += src_view(iF, jF, kF, ivar, qf);

        transf.compute_indices<element_kind_t::FACE, true>(
            ig, 2*j + 1, 2*k + 0, iF, jF, kF, fs, /*half_ncells=*/false);
        sum += src_view(iF, jF, kF, ivar, qf);

        transf.compute_indices<element_kind_t::FACE, true>(
            ig, 2*j + 0, 2*k + 1, iF, jF, kF, fs, /*half_ncells=*/false);
        sum += src_view(iF, jF, kF, ivar, qf);

        transf.compute_indices<element_kind_t::FACE, true>(
            ig, 2*j + 1, 2*k + 1, iF, jF, kF, fs, /*half_ncells=*/false);
        sum += src_view(iF, jF, kF, ivar, qf);

        // 3) Write SUM into that coarse face location.
        //    Children write to **disjoint quarters**, so '=' is safe (no overlap).
        dest_view(iC, jC, kC, ivar, qc) = sum;
    }
};

}} // namespace grace::amr

#endif // GRACE_AMR_BC_REFLUX_KERNELS_HH