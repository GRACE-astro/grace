/**
 * @file boundary_outflow.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Mass-outflux accumulator at outer (non-periodic) domain boundaries.
 * @date 2026-05-21
 *
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 */

#ifndef GRACE_EVOLUTION_BOUNDARY_OUTFLOW_HH
#define GRACE_EVOLUTION_BOUNDARY_OUTFLOW_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/amr/amr_ghosts.hh>
#include <grace/amr/grace_amr.hh>

#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_indices.hh>

#include <grace/utils/reductions.hh>   // scalar_symmetry_multiplier()
#include <grace/parallel/mpi_wrappers.hh>

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace grace {

/**
 * @brief Singleton accumulator for mass flux through the outer physical
 *        boundary of the simulation domain.
 *
 * Lifecycle:
 *   - Construction: device scalar `outflow_mass` allocated and zeroed.
 *   - `accumulate(...)` is called from the evolution loop AFTER reflux at
 *     each RK substep.  A TeamPolicy walks the boundary face list
 *     (see `boundary_quads_t` in `amr/amr_ghosts.hh`); each team handles
 *     one (quadrant, face_id) pair, sums DENS flux over the face cells
 *     in a team_reduce, then one thread atomic-adds the weighted
 *     contribution `dt · dtfact · dA · Σ F · sign` to the global scalar.
 *   - `flush_to_host()` is called by the diagnostic at output time.
 *     It MPI-Allreduces the device scalar to a global host double, then
 *     resets the local device scalar to 0.  Each diagnostic line therefore
 *     reports "mass that left the grid during this output interval".
 *
 * Sign convention (matches `iterate_faces.cpp`):
 *   fidx = 2*axis + side, side=0 (low) → sign = -1, side=1 (high) → sign = +1.
 *   `sgn = (fidx & 1) ? +1 : -1`.  Mass leaving the grid → positive
 *   contribution to `outflow_mass`.
 */
class boundary_outflow_t {
public:
    static boundary_outflow_t& get() {
        static boundary_outflow_t instance ;
        return instance ;
    }

    /**
     * @brief Accumulate the outward DENS flux integrated over outer-boundary
     *        faces for ONE RK substep.  Called from the evolution loop after
     *        reflux has corrected the face fluxes at coarse-fine interfaces.
     *
     * @param dt       Step size of the full RK step.
     * @param dtfact   RK substage weight (same factor as for source-term application).
     */
    void accumulate(double dt, double dtfact) {
        auto& ghost = amr_ghosts::get() ;
        auto const& boundary = ghost.get_boundary_quads() ;
        size_t const nb = boundary.size ;
        if (nb == 0) return ;

        DECLARE_GRID_EXTENTS ;
        auto& vlist  = variable_list::get() ;
        auto& fluxes = vlist.getfluxesarray() ;
        auto& dx     = vlist.getspacings() ;

        auto bq = boundary.q ;
        auto bf = boundary.face_id ;
        auto outflow = _outflow_mass ;
        double const weight = dt * dtfact ;

        using team_policy_t = Kokkos::TeamPolicy<default_space::execution_space> ;
        using member_t      = team_policy_t::member_type ;

        team_policy_t policy(static_cast<int>(nb), Kokkos::AUTO) ;

        Kokkos::parallel_for(
            "boundary_outflow_accumulate",
            policy,
            KOKKOS_LAMBDA(member_t const& team) {
                int const idx  = team.league_rank() ;
                int const q    = bq(idx) ;
                int const fidx = bf(idx) ;
                int const axis = fidx / 2 ;
                double const sgn  = (fidx & 1) ? +1.0 : -1.0 ;
                // Logical face area on a Cartesian grid: product of the
                // two perpendicular cell spacings.  For curvilinear grids
                // this would need the induced 2D-metric determinant on
                // the face — extending later if needed.
                double const dx0 = dx(0, q) ;
                double const dx1 = dx(1, q) ;
                double const dx2 = dx(2, q) ;
                double const dA  =
                    (axis == 0) ? dx1 * dx2 :
                    (axis == 1) ? dx0 * dx2 :
                                  dx0 * dx1 ;

                // Face flux is stored at the staggered index:
                //   axis=0, high face (fidx=1): i_flux = nx+ngz
                //   axis=0, low face  (fidx=0): i_flux = ngz
                // Same for axis=1 (j) and axis=2 (k).  The flux array
                // is fluxes(i,j,k, var, dir, q); the staggered direction
                // is `axis`, and the read index is the boundary face plane.
                int const lo_flux = static_cast<int>(ngz) ;
                int const hi_flux_x = static_cast<int>(nx + ngz) ;
                int const hi_flux_y = static_cast<int>(ny + ngz) ;
                int const hi_flux_z = static_cast<int>(nz + ngz) ;

                // Inner extents (cells along the face): the two non-staggered
                // directions run over the interior cell range [ngz, n+ngz).
                int const n_inner_x = static_cast<int>(nx) ;
                int const n_inner_y = static_cast<int>(ny) ;
                int const n_inner_z = static_cast<int>(nz) ;
                int const lo = static_cast<int>(ngz) ;

                // Team-reduce: sum DENS face flux over face cells.
                double face_sum = 0.0 ;
                if (axis == 0) {
                    int const i_face = (fidx & 1) ? hi_flux_x : lo_flux ;
                    int const N = n_inner_y * n_inner_z ;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, N),
                        [&](int const lin, double& acc) {
                            int const j = lo + (lin % n_inner_y) ;
                            int const k = lo + (lin / n_inner_y) ;
                            acc += fluxes(i_face, j, k, DENS_, axis, q) ;
                        }, face_sum) ;
                } else if (axis == 1) {
                    int const j_face = (fidx & 1) ? hi_flux_y : lo_flux ;
                    int const N = n_inner_x * n_inner_z ;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, N),
                        [&](int const lin, double& acc) {
                            int const i = lo + (lin % n_inner_x) ;
                            int const k = lo + (lin / n_inner_x) ;
                            acc += fluxes(i, j_face, k, DENS_, axis, q) ;
                        }, face_sum) ;
                } else {
                    int const k_face = (fidx & 1) ? hi_flux_z : lo_flux ;
                    int const N = n_inner_x * n_inner_y ;
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, N),
                        [&](int const lin, double& acc) {
                            int const i = lo + (lin % n_inner_x) ;
                            int const j = lo + (lin / n_inner_x) ;
                            acc += fluxes(i, j, k_face, DENS_, axis, q) ;
                        }, face_sum) ;
                }

                // One thread per team atomic-adds to the global scalar.
                team.team_barrier() ;
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    Kokkos::atomic_add(&outflow(0), sgn * weight * dA * face_sum) ;
                }) ;
            }) ;
    }

    /**
     * @brief Read the accumulated outflux, MPI-reduce across ranks, return the
     *        global value, and reset the local accumulator to zero.
     *
     * Symmetry: multiplies by `scalar_symmetry_multiplier()` so a half-domain
     * simulation reports the full-domain physical outflux.
     */
    double flush_to_host() {
        auto outflow_h = Kokkos::create_mirror_view(_outflow_mass) ;
        Kokkos::deep_copy(outflow_h, _outflow_mass) ;
        double const local = outflow_h(0) ;
        double global = 0.0 ;
        parallel::mpi_allreduce(&local, &global, 1, sc_MPI_SUM) ;
        global *= scalar_symmetry_multiplier() ;

        // Reset device scalar to zero — next interval starts fresh.
        Kokkos::deep_copy(_outflow_mass, 0.0) ;

        return global ;
    }

private:
    boundary_outflow_t() : _outflow_mass("boundary_outflow_mass") {
        Kokkos::deep_copy(_outflow_mass, 0.0) ;
    }
    Kokkos::View<double[1], default_space> _outflow_mass ;
} ;

} // namespace grace

#endif // GRACE_EVOLUTION_BOUNDARY_OUTFLOW_HH
