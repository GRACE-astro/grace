/**
 * @file adm_integrals.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Implementation of the ADM-mass surface integral diagnostic.
 *        See adm_integrals.hh for the integral form.
 *
 * @date 2026-04-27
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference / Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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
 */

#include <grace_config.h>

#include <grace/IO/diagnostics/adm_integrals.hh>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/IO/spherical_surfaces.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/config/config_parser.hh>
#include <grace/amr/amr_functions.hh>

#include <Kokkos_Core.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <array>
#include <vector>


namespace grace {

#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4

namespace {

//-------------------------------------------------------------------------
// Variable layout for the ADM diagnostic.
//-------------------------------------------------------------------------
// The interpolator returns one value per requested var index. We pack the
// metric components first, then chi (Z4c only). The compute_local routine
// reads from these fixed positions.
enum adm_var_slot : int {
    SLOT_GTXX = 0,
    SLOT_GTXY,
    SLOT_GTXZ,
    SLOT_GTYY,
    SLOT_GTYZ,
    SLOT_GTZZ,
    SLOT_CHI,
    N_ADM_SLOTS
};

constexpr double ONE_OVER_16PI = 1.0 / (16.0 * M_PI);

}  // namespace


//-------------------------------------------------------------------------
// Construction.
//-------------------------------------------------------------------------

adm_integrals::adm_integrals() {
    auto names = get_param<std::vector<std::string>>("adm_integrals", "detector_names");

    auto& spheres = grace::spherical_surface_manager::get();
    for (auto const& n : names) {
        auto idx = spheres.get_index(n);
        if (idx < 0) {
            GRACE_WARN("ADM integral detector {} not found in spherical_surfaces", n);
        } else {
            sphere_indices.push_back(idx);
        }
    }
    std::sort(sphere_indices.begin(), sphere_indices.end());
    sphere_indices.erase(
        std::unique(sphere_indices.begin(), sphere_indices.end()),
        sphere_indices.end()
    );

    // We need values + gradients of the conformal metric and chi.
    var_interp_idx = std::vector<int>({
        GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_,
        CHI_
    });
}


//-------------------------------------------------------------------------
// File initialisation (rank 0 only).
//-------------------------------------------------------------------------

void adm_integrals::initialize_files() {
    if (parallel::mpi_comm_rank() != 0) return;

    auto& sphere_list   = grace::spherical_surface_manager::get();
    auto& grace_runtime = grace::runtime::get();
    std::filesystem::path bdir = grace_runtime.scalar_io_basepath();
    static constexpr size_t width = 20;

    for (auto const& sidx : sphere_indices) {
        auto const& detector = sphere_list.get(sidx);
        std::string pfname = grace_runtime.scalar_io_basename()
                             + "M_ADM_" + detector.name + ".dat";
        std::filesystem::path fname = bdir / pfname;
        std::ofstream outfile(fname.string(), std::ios::app);
        outfile << std::fixed << std::setprecision(15);
        outfile << std::left << std::setw(width) << "Iteration"
                << std::left << std::setw(width) << "Time"
                << std::left << std::setw(width) << "M_ADM" << '\n';
    }
}


//-------------------------------------------------------------------------
// Per-sphere Hermite interpolator setup.
//-------------------------------------------------------------------------

void adm_integrals::refresh_interpolator(size_t sphere_idx) {
    DECLARE_GRID_EXTENTS;
    auto& sphere_list = grace::spherical_surface_manager::get();
    auto const& detector = sphere_list.get(sphere_idx);

    // The Hermite cubic stencil needs 1 valid ghost zone on each side of
    // the cell for the FD halo; metric variables in Z4c carry the full
    // ngz ghost coverage so we pass ngz here (no biasing needed in the
    // common case).
    auto it = interpolators.find(sphere_idx);
    if (it == interpolators.end()) {
        it = interpolators.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(sphere_idx),
            std::forward_as_tuple(static_cast<int>(ngz))
        ).first;
    }

    it->second.compute_weights(
        detector.points_h,
        detector.intersecting_points_h,
        detector.intersected_cells_h
    );
}


//-------------------------------------------------------------------------
// Local (rank-owned) integrand sum.
//-------------------------------------------------------------------------

double adm_integrals::compute_local(size_t sphere_idx) {
    auto& sphere_list = grace::spherical_surface_manager::get();
    auto const& detector = sphere_list.get(sphere_idx);

    auto npoints_loc = detector.intersecting_points_h.size();
    if (npoints_loc == 0) return 0.0;

    auto& interp = interpolators.at(sphere_idx);
    auto state   = variable_list::get().getstate();

    Kokkos::View<double**,  grace::default_space> ivals_d   ("adm_metric_vals", 0, 0);
    Kokkos::View<double***, grace::default_space> ivals_grad("adm_metric_grad", 0, 0, 0);

    interp.interpolate_with_grad(state, var_interp_idx, ivals_d, ivals_grad, true);

    // Pull to host for the reduction. npoints_loc * 7 * 4 doubles - small.
    auto ivals_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_d);
    auto ivals_grad_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_grad);

    double const r        = detector.radius;
    double const r2       = r * r;
    auto const& center    = detector.center;
    bool const z_sym      = get_param<bool>("amr","reflection_symmetries","z");
    double const sym_factor = z_sym ? 2.0 : 1.0;

    double local_sum = 0.0;

    for (size_t i = 0; i < npoints_loc; ++i) {
        auto const ip      = detector.intersecting_points_h[i];
        auto const& xyz    = detector.points_h[ip].second;
        double const dOmega = detector.weights_h[ip];

        // Conformal metric (gamma~_ij) and chi at the point.
        double const gtxx = ivals_h(i, SLOT_GTXX);
        double const gtxy = ivals_h(i, SLOT_GTXY);
        double const gtxz = ivals_h(i, SLOT_GTXZ);
        double const gtyy = ivals_h(i, SLOT_GTYY);
        double const gtyz = ivals_h(i, SLOT_GTYZ);
        double const gtzz = ivals_h(i, SLOT_GTZZ);
        double const chi  = ivals_h(i, SLOT_CHI);

        double const inv_chi  = 1.0 / std::max(chi, 1e-100);
        double const inv_chi2 = inv_chi * inv_chi;
        double const inv_chi3 = inv_chi2 * inv_chi;

        // Physical metric gamma_ij = gamma~_ij / chi^2 (3x3 symmetric).
        double g[3][3];
        g[0][0] = gtxx * inv_chi2;
        g[1][1] = gtyy * inv_chi2;
        g[2][2] = gtzz * inv_chi2;
        g[0][1] = g[1][0] = gtxy * inv_chi2;
        g[0][2] = g[2][0] = gtxz * inv_chi2;
        g[1][2] = g[2][1] = gtyz * inv_chi2;

        // Inverse 3-metric and sqrt(det g).
        double const det_g =
              g[0][0] * (g[1][1]*g[2][2] - g[1][2]*g[1][2])
            - g[0][1] * (g[0][1]*g[2][2] - g[0][2]*g[1][2])
            + g[0][2] * (g[0][1]*g[1][2] - g[0][2]*g[1][1]);
        double const inv_det = 1.0 / std::max(det_g, 1e-300);
        double const sqrt_det_g = std::sqrt(std::max(det_g, 0.0));

        double gI[3][3];
        gI[0][0] = (g[1][1]*g[2][2] - g[1][2]*g[1][2]) * inv_det;
        gI[1][1] = (g[0][0]*g[2][2] - g[0][2]*g[0][2]) * inv_det;
        gI[2][2] = (g[0][0]*g[1][1] - g[0][1]*g[0][1]) * inv_det;
        gI[0][1] = gI[1][0] = (g[0][2]*g[1][2] - g[0][1]*g[2][2]) * inv_det;
        gI[0][2] = gI[2][0] = (g[0][1]*g[1][2] - g[0][2]*g[1][1]) * inv_det;
        gI[1][2] = gI[2][1] = (g[0][1]*g[0][2] - g[0][0]*g[1][2]) * inv_det;

        // d_k gamma_ij via product rule on gamma~_ij/chi^2 (3x3x3, symmetric in i,j).
        auto dgt = [&] (int slot, int k) { return ivals_grad_h(i, slot, k); };
        auto dchi = [&] (int k)          { return ivals_grad_h(i, SLOT_CHI, k); };
        auto dgamma_of = [&](int slot, double gt_val, int k) {
            return dgt(slot, k) * inv_chi2 - 2.0 * gt_val * dchi(k) * inv_chi3;
        };
        double dg[3][3][3];
        for (int k = 0; k < 3; ++k) {
            dg[0][0][k] = dgamma_of(SLOT_GTXX, gtxx, k);
            dg[1][1][k] = dgamma_of(SLOT_GTYY, gtyy, k);
            dg[2][2][k] = dgamma_of(SLOT_GTZZ, gtzz, k);
            dg[0][1][k] = dg[1][0][k] = dgamma_of(SLOT_GTXY, gtxy, k);
            dg[0][2][k] = dg[2][0][k] = dgamma_of(SLOT_GTXZ, gtxz, k);
            dg[1][2][k] = dg[2][1][k] = dgamma_of(SLOT_GTYZ, gtyz, k);
        }

        // Covariant ADM integrand (Eq. 1 in ADMMass thorn doc; O Murchadha &
        // York 1974). Free index l contracts with the surface element:
        //
        //   T^l = gamma^{ij} gamma^{kl} ( d_j gamma_{ik} - d_k gamma_{ij} )
        //
        // Factored as T^l = gamma^{kl} H_k with
        //
        //   H_k = sum_{ij} gamma^{ij} ( d_j gamma_{ik} - d_k gamma_{ij} )
        //
        // The integrand on the sphere is sqrt(det gamma) * T^l n_l r^2 dOmega;
        // at infinity gamma^{ij}->delta^{ij}, sqrt(det g)->1 and this reduces
        // to the asymptotically-flat L_i n^i form.
        double H[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k) {
            double sum = 0.0;
            for (int ii = 0; ii < 3; ++ii)
                for (int jj = 0; jj < 3; ++jj)
                    sum += gI[ii][jj] * (dg[ii][k][jj] - dg[ii][jj][k]);
            H[k] = sum;
        }

        double T[3] = {0.0, 0.0, 0.0};
        for (int l = 0; l < 3; ++l) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k)
                sum += gI[k][l] * H[k];
            T[l] = sum;
        }

        // Outward unit normal w.r.t. sphere centre (flat-space).
        double const xx = xyz[0] - center[0];
        double const yy = xyz[1] - center[1];
        double const zz = xyz[2] - center[2];
        double const r_pt = std::sqrt(xx*xx + yy*yy + zz*zz);
        double const inv_r_pt = 1.0 / std::max(r_pt, 1e-100);
        double const n_l[3] = {xx * inv_r_pt, yy * inv_r_pt, zz * inv_r_pt};

        double const integrand = sqrt_det_g * (T[0]*n_l[0] + T[1]*n_l[1] + T[2]*n_l[2]);

        // dA = r^2 dOmega
        local_sum += integrand * r2 * dOmega;
    }

    return ONE_OVER_16PI * sym_factor * local_sum;
}


//-------------------------------------------------------------------------
// Compute (programmatic).
//-------------------------------------------------------------------------

std::vector<double> adm_integrals::compute() {
    std::vector<double> result;
    result.reserve(sphere_indices.size());
    for (auto const& sidx : sphere_indices) {
        refresh_interpolator(sidx);
        double local_M = compute_local(sidx);
        double global_M = 0.0;
        parallel::mpi_allreduce(&local_M, &global_M, 1, MPI_SUM);
        result.push_back(global_M);
    }
    return result;
}


//-------------------------------------------------------------------------
// Compute and write.
//-------------------------------------------------------------------------

void adm_integrals::compute_and_write() {
    if (sphere_indices.empty()) return;

    auto& grace_runtime = grace::runtime::get();
    auto& sphere_list   = grace::spherical_surface_manager::get();
    size_t const iter   = grace_runtime.iteration();
    double const time   = grace_runtime.time();

    std::filesystem::path bdir = grace_runtime.scalar_io_basepath();

    auto const values = compute();

    if (parallel::mpi_comm_rank() == 0) {
        for (size_t i = 0; i < sphere_indices.size(); ++i) {
            auto const sidx      = sphere_indices[i];
            auto const& detector = sphere_list.get(sidx);
            std::string pfname = grace_runtime.scalar_io_basename()
                                 + "M_ADM_" + detector.name + ".dat";
            std::filesystem::path fname = bdir / pfname;
            std::ofstream outfile(fname.string(), std::ios::app);
            outfile << std::fixed << std::setprecision(15);
            outfile << std::left << iter << '\t'
                    << std::left << time << '\t'
                    << std::left << values[i] << '\n';
        }
    }
}

#else  // GRACE_METRIC_EVOL != GRACE_METRIC_EVOL_Z4

// The diagnostic is only meaningful for an evolved spacetime metric; with
// the Cowling stub built in, leave the methods as no-ops so the symbol is
// still linkable from output_diagnostics.cpp without ifdef sprinkling.
adm_integrals::adm_integrals() {}
void adm_integrals::initialize_files() {}
void adm_integrals::refresh_interpolator(size_t) {}
double adm_integrals::compute_local(size_t) { return 0.0; }
std::vector<double> adm_integrals::compute() { return {}; }
void adm_integrals::compute_and_write() {}

#endif  // GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4

}  // namespace grace
