/**
 * @file apparent_horizon.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Apparent horizon finder implementation.
 * @date 2026-04-11
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

#include <grace/IO/diagnostics/apparent_horizon.hh>

#include <grace/data_structures/variables.hh>
#include <grace/amr/forest.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/system/print.hh>
#include <grace/utils/metric_utils.hh>

#include <cmath>
#include <algorithm>
#include <numeric>

namespace grace {

// ============================================================================
// Constructor
// ============================================================================

apparent_horizon_finder_t::apparent_horizon_finder_t(ah_config_t const& cfg)
    : cfg_(cfg)
    , interpolator_(1)
{
    basis_ = std::make_unique<spherical_harmonic_basis_t>(cfg_.l_max) ;
    coeffs_.assign(basis_->n_coeffs(), 0.0) ;
    // initial guess: sphere of radius initial_radius → a_{0,0} = r0 / Y_{0,0}
    // Y_{0,0} = 1/sqrt(4pi), so a_{0,0} = r0 * sqrt(4pi)
    coeffs_[0] = cfg_.initial_radius * std::sqrt(4.0 * M_PI) ;
}

// ============================================================================
// Surface point generation
// ============================================================================

void apparent_horizon_finder_t::generate_surface_points(
    std::vector<point_host_t>& points,
    std::vector<std::array<double,2>>& angles,
    int& n_colloc,
    double delta
) const
{
    n_colloc = basis_->n_points() ;
    int n_total = n_colloc * (1 + N_STAG_DIRS) ; // center + 6 staggered

    // evaluate h(theta,phi) at collocation points
    std::vector<double> h_vals(n_colloc) ;
    std::vector<double> dh_dth(n_colloc), dh_dph(n_colloc) ;
    basis_->evaluate(coeffs_.data(), h_vals.data(), dh_dth.data(), dh_dph.data()) ;

    points.resize(n_total) ;
    angles.resize(n_colloc) ;

    auto const& cx = cfg_.center ;

    for (int it = 0; it < basis_->n_theta(); ++it) {
        for (int ip = 0; ip < basis_->n_phi(); ++ip) {
            int pt = basis_->point_index(it, ip) ;
            double th = basis_->theta(it) ;
            double ph = basis_->phi(ip) ;
            double r  = h_vals[pt] ;

            double sth = std::sin(th), cth = std::cos(th) ;
            double sph = std::sin(ph), cph = std::cos(ph) ;

            double x = cx[0] + r * sth * cph ;
            double y = cx[1] + r * sth * sph ;
            double z = cx[2] + r * cth ;

            // center point
            points[pt] = {static_cast<size_t>(pt), {x, y, z}} ;
            angles[pt] = {th, ph} ;

            // staggered points for metric gradients
            // +x, -x, +y, -y, +z, -z
            double offsets[6][3] = {
                {+delta, 0, 0}, {-delta, 0, 0},
                {0, +delta, 0}, {0, -delta, 0},
                {0, 0, +delta}, {0, 0, -delta}
            } ;
            for (int s = 0; s < N_STAG_DIRS; ++s) {
                int idx = n_colloc + pt * N_STAG_DIRS + s ;
                points[idx] = {
                    static_cast<size_t>(idx),
                    {x + offsets[s][0], y + offsets[s][1], z + offsets[s][2]}
                } ;
            }
        }
    }
}

// ============================================================================
// Interpolation setup
// ============================================================================

void apparent_horizon_finder_t::setup_interpolation(
    std::vector<point_host_t> const& points,
    std::vector<size_t>& intersecting_points_out
)
{
    DECLARE_GRID_EXTENTS ;

    auto p4est = grace::amr::forest::get().get() ;

    std::vector<intersected_cell_descriptor_t> intersected_cells_h ;
    std::vector<size_t> intersecting_points_h ;
    intersected_cell_set_t set{&intersected_cells_h, &intersecting_points_h} ;

    auto points_array = sc_array_new_data(
        const_cast<point_host_t*>(points.data()),
        sizeof(point_host_t),
        points.size()
    ) ;

    p4est->user_pointer = static_cast<void*>(&set) ;
    p4est_search_local(p4est, false, nullptr, &grace_search_points, points_array) ;

    interpolator_.compute_weights(points, intersecting_points_h, intersected_cells_h) ;
    intersecting_points_out = std::move(intersecting_points_h) ;
}

// ============================================================================
// Metric data extraction
// ============================================================================

void apparent_horizon_finder_t::extract_metric_data(
    Kokkos::View<double**, Kokkos::HostSpace> const& ivals_h,
    int n_colloc,
    double delta,
    std::vector<metric_data_t>& mdata
) const
{
    mdata.resize(n_colloc) ;
    double inv2delta = 0.5 / delta ;

    for (int pt = 0; pt < n_colloc; ++pt) {
        auto& md = mdata[pt] ;

        // extract conformal metric and curvature at center point
        double gtxx = ivals_h(pt, GTXXL), gtxy = ivals_h(pt, GTXYL),
               gtxz = ivals_h(pt, GTXZL), gtyy = ivals_h(pt, GTYYL),
               gtyz = ivals_h(pt, GTYZL), gtzz = ivals_h(pt, GTZZL) ;
        double chi  = ivals_h(pt, CHIL) ;
        double atxx = ivals_h(pt, ATXXL), atxy = ivals_h(pt, ATXYL),
               atxz = ivals_h(pt, ATXZL), atyy = ivals_h(pt, ATYYL),
               atyz = ivals_h(pt, ATYZL), atzz = ivals_h(pt, ATZZL) ;
        double khat  = ivals_h(pt, KHATL) ;
        double theta_z4c = ivals_h(pt, THETAL) ;

        // physical metric: gamma_ij = gtilde_ij / chi^2
        double ooW = 1.0 / std::max(1e-100, chi) ;
        double ooW2 = ooW * ooW ;

        md.gamma = {gtxx*ooW2, gtxy*ooW2, gtxz*ooW2,
                     gtyy*ooW2, gtyz*ooW2, gtzz*ooW2} ;

        // inverse metric
        metric_array_t metric_obj({gtxx,gtxy,gtxz,gtyy,gtyz,gtzz},
                                   chi, {0,0,0}, 1.0) ;
        md.gamma_inv = metric_obj._ginv ;
        md.sqrtg = metric_obj.sqrtg() ;

        // extrinsic curvature: K_ij = A_ij/chi^2 + (1/3)(Khat + 2*Theta)*gamma_ij
        double K_trace = khat + 2.0 * theta_z4c ;
        double third_K = K_trace / 3.0 ;
        md.K_ij = {
            atxx*ooW2 + third_K * md.gamma[0],
            atxy*ooW2 + third_K * md.gamma[1],
            atxz*ooW2 + third_K * md.gamma[2],
            atyy*ooW2 + third_K * md.gamma[3],
            atyz*ooW2 + third_K * md.gamma[4],
            atzz*ooW2 + third_K * md.gamma[5]
        } ;
        md.K_trace = K_trace ;

        // metric gradients via centered FD on staggered points
        // stagger layout: +x(0), -x(1), +y(2), -y(3), +z(4), -z(5)
        for (int k = 0; k < 3; ++k) { // derivative direction
            int i_plus  = n_colloc + pt * N_STAG_DIRS + 2*k ;
            int i_minus = n_colloc + pt * N_STAG_DIRS + 2*k + 1 ;
            for (int c = 0; c < 6; ++c) { // metric component
                double gt_plus  = ivals_h(i_plus,  c) ;  // GTXXL..GTZZL = 0..5
                double gt_minus = ivals_h(i_minus, c) ;
                double chi_plus  = ivals_h(i_plus,  CHIL) ;
                double chi_minus = ivals_h(i_minus, CHIL) ;
                // physical metric at staggered points
                double ooW2_plus  = 1.0 / (chi_plus  * chi_plus  + 1e-200) ;
                double ooW2_minus = 1.0 / (chi_minus * chi_minus + 1e-200) ;
                double g_plus  = gt_plus  * ooW2_plus ;
                double g_minus = gt_minus * ooW2_minus ;
                md.dgamma[k][c] = (g_plus - g_minus) * inv2delta ;
            }
        }
    }
}

// ============================================================================
// Expansion computation at a single collocation point
// ============================================================================

double apparent_horizon_finder_t::compute_expansion(
    int pt,
    metric_data_t const& md,
    double const* h_vals,
    double const* dh_dth,
    double const* dh_dph,
    double const* d2h_dth2,
    double const* d2h_dthdph,
    double const* d2h_dph2,
    double theta,
    double phi
) const
{
    // Surface F(x) = r - h(theta,phi) = 0
    // The outward-pointing normal covector in Cartesian coords:
    //   N_i = dF/dx^i = dr/dx^i - (dh/dtheta)(dtheta/dx^i) - (dh/dphi)(dphi/dx^i)
    //
    // Thornburg (2004) expansion formula:
    //   Theta = (gamma^{ij} - s^i s^j)(D_i s_j) + K_{ij} s^i s^j - K
    //         = P^{ij}(H_{ij} - Gamma^k_{ij} N_k)/|N| + s^i s^j K_{ij} - K
    // where P^{ij} = gamma^{ij} - s^i s^j is the projection operator,
    //       H_{ij} = d_i N_j is the Hessian of F, and s^i = gamma^{ij} N_j / |N|.

    double h   = h_vals[pt] ;
    double ht  = dh_dth[pt] ;
    double hp  = dh_dph[pt] ;
    double htt = d2h_dth2[pt] ;
    double htp = d2h_dthdph[pt] ;
    double hpp = d2h_dph2[pt] ;

    double sth = std::sin(theta), cth = std::cos(theta) ;
    double sph = std::sin(phi),   cph = std::cos(phi) ;

    // Jacobian: dx^a / d(r, theta, phi)
    //   x = r sin(th) cos(ph)   →  dx/dr = sth*cph, dx/dth = r*cth*cph, dx/dph = -r*sth*sph
    //   y = r sin(th) sin(ph)   →  etc.
    //   z = r cos(th)

    // Inverse Jacobian: d(r,th,ph)/dx^a (evaluated at r=h)
    double r = h ;
    double or1 = 1.0 / std::max(r, 1e-15) ;
    double or_sth = (std::abs(sth) > 1e-14) ? 1.0 / (r * sth) : 0.0 ;

    // dr/dx^i
    double dr_dx[3] = {sth*cph, sth*sph, cth} ;
    // dth/dx^i
    double dth_dx[3] = {cth*cph*or1, cth*sph*or1, -sth*or1} ;
    // dph/dx^i
    double dph_dx[3] = {-sph*or_sth, cph*or_sth, 0.0} ;

    // N_i = dF/dx^i = dr/dx^i - ht * dth/dx^i - hp * dph/dx^i
    double N[3] ;
    for (int i = 0; i < 3; ++i) {
        N[i] = dr_dx[i] - ht * dth_dx[i] - hp * dph_dx[i] ;
    }

    // |N|^2 = gamma^{ij} N_i N_j
    auto const& ginv = md.gamma_inv ;
    double N_sq = ginv[0]*N[0]*N[0] + ginv[3]*N[1]*N[1] + ginv[5]*N[2]*N[2]
               + 2.0*(ginv[1]*N[0]*N[1] + ginv[2]*N[0]*N[2] + ginv[4]*N[1]*N[2]) ;
    double inv_N = 1.0 / std::sqrt(std::max(N_sq, 1e-30)) ;

    // unit outward normal s^i = gamma^{ij} N_j / |N|
    double sU[3] = {
        (ginv[0]*N[0] + ginv[1]*N[1] + ginv[2]*N[2]) * inv_N,
        (ginv[1]*N[0] + ginv[3]*N[1] + ginv[4]*N[2]) * inv_N,
        (ginv[2]*N[0] + ginv[4]*N[1] + ginv[5]*N[2]) * inv_N
    } ;

    // Christoffel symbols Gamma^k_{ij} from metric gradients
    // Gamma^k_{ij} = 0.5 * gamma^{kl} (d_i gamma_{jl} + d_j gamma_{il} - d_l gamma_{ij})
    // metric component packing: 0=XX, 1=XY, 2=XZ, 3=YY, 4=YZ, 5=ZZ
    // We need gamma_{ab} with a,b in {0,1,2}:
    //   (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=1, (1,1)=3, (1,2)=4, (2,0)=2, (2,1)=4, (2,2)=5
    auto sym_idx = [](int a, int b) -> int {
        if (a > b) std::swap(a,b) ;
        if (a==0) return b ;        // 0,1,2
        if (a==1) return b + 2 ;    // 3,4
        return 5 ;                   // 5
    } ;

    // Compute Gamma^k_{ij} contracted with N_k and projected
    // D_i s_j = (1/|N|)(d_i N_j - Gamma^k_{ij} N_k)
    // But the Hessian of F is H_{ij} = d_i N_j, which requires second
    // derivatives of the coordinate transform. We use the chain rule approach.

    // Compute Christoffel*N: Gamma^k_{ij} N_k
    double GammaN[3][3] ;  // GammaN[i][j] = Gamma^k_{ij} N_k
    for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
            double val = 0 ;
            for (int k = 0; k < 3; ++k) {
                // Gamma^k_{ij} = 0.5 * ginv[k,l] * (dg[i][jl] + dg[j][il] - dg[l][ij])
                for (int l = 0; l < 3; ++l) {
                    double ginv_kl = ginv[sym_idx(k,l)] ;
                    double dg_i_jl = md.dgamma[i][sym_idx(j,l)] ;
                    double dg_j_il = md.dgamma[j][sym_idx(i,l)] ;
                    double dg_l_ij = md.dgamma[l][sym_idx(i,j)] ;
                    val += 0.5 * ginv_kl * (dg_i_jl + dg_j_il - dg_l_ij) * N[k] ;
                }
            }
            GammaN[i][j] = val ;
            GammaN[j][i] = val ;
        }
    }

    // Hessian of F in Cartesian coordinates: H_{ij} = d_i d_j F
    // F = r - h(th,ph), so d_i F = N_i computed above.
    // H_{ij} = d_i(dr/dx^j) - htt*dth/dx^i*dth/dx^j - hpp*dph/dx^i*dph/dx^j
    //        - htp*(dth/dx^i*dph/dx^j + dph/dx^i*dth/dx^j)
    //        - ht*d_i(dth/dx^j) - hp*d_i(dph/dx^j)
    //
    // The second derivatives of the coordinate transform are:
    //   d_i(dr/dx^j) = d/dx^i(dr/dx^j)
    // These are computed from the chain rule on the spherical→Cartesian Jacobian.
    // Rather than writing all these out, we compute the Hessian numerically
    // by finite differencing the gradient of F in the angular directions.
    //
    // Alternative: use the formula
    //   P^{ij}(H_{ij} - Gamma^k_{ij} N_k)/|N| =
    //     [1/(h^2 * |N|)] * { (1/sin^2(th))(h^2 + ht^2)*hpp - 2*ht*hp*htp*(h + ...) + ... }
    //
    // We use the direct Thornburg formula in Cartesian coordinates, computing
    // H_{ij} analytically from the chain rule.

    // Second derivatives of coordinate transform evaluated at r=h:
    // d(dr/dx^i)/dx^j = (delta_{ij} - dr/dx^i * dr/dx^j) / r   (for flat space)
    // For curved coordinates we need the full expression.
    // Since we work in Cartesian coords (no coordinate curvature),
    // only the embedding r(x,y,z), theta(x,y,z), phi(x,y,z) have nontrivial Hessians.

    // Hessians of r, theta, phi w.r.t. Cartesian x,y,z
    double d2r[3][3], d2th[3][3], d2ph[3][3] ;
    {
        double r2 = r*r ;
        double sth2 = sth*sth ;

        // d^2 r / dx^i dx^j
        // r = sqrt(x^2+y^2+z^2), dr/dx^i = x^i/r
        // d^2r/dx^i dx^j = (delta_{ij} - x^i x^j / r^2) / r
        double x_hat[3] = {sth*cph, sth*sph, cth} ; // x^i / r
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                d2r[i][j] = ((i==j ? 1.0 : 0.0) - x_hat[i]*x_hat[j]) * or1 ;

        // theta = acos(z/r), dtheta/dx = (z*x)/(r^3*sth), etc.
        // Full expressions for d^2 theta / dx^i dx^j are lengthy;
        // we compute them from the chain rule:
        // dth/dx^i = cth*cph*or1, cth*sph*or1, -sth*or1
        // d(dth/dx^i)/dx^j = d(dth/dx^i)/dr * dr/dx^j
        //                   + d(dth/dx^i)/dth * dth/dx^j
        //                   + d(dth/dx^i)/dph * dph/dx^j
        // This is the standard approach. Let's compute the needed partials.

        // dth_dx[i] = {cth*cph/r, cth*sph/r, -sth/r}
        // d/dr(dth_dx[0]) = -cth*cph/r^2
        // d/dth(dth_dx[0]) = -sth*cph/r
        // d/dph(dth_dx[0]) = -cth*sph/r

        // d(dth/dx^i)/dr
        double ddth_dr[3] = {-cth*cph*or1*or1, -cth*sph*or1*or1, sth*or1*or1} ;
        // d(dth/dx^i)/dth
        double ddth_dth[3] = {-sth*cph*or1, -sth*sph*or1, -cth*or1} ;
        // d(dth/dx^i)/dph
        double ddth_dph[3] = {-cth*sph*or1, cth*cph*or1, 0.0} ;

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                d2th[i][j] = ddth_dr[i]*dr_dx[j] + ddth_dth[i]*dth_dx[j] + ddth_dph[i]*dph_dx[j] ;

        // dph_dx = {-sph/(r*sth), cph/(r*sth), 0}
        // d(dph/dx^i)/dr
        double ddph_dr[3] = {sph*or_sth*or1, -cph*or_sth*or1, 0.0} ;
        // d(dph/dx^i)/dth
        double cth_over_rsth2 = (std::abs(sth) > 1e-14) ? cth / (r * sth2) : 0.0 ;
        double ddph_dth[3] = {sph*cth_over_rsth2, -cph*cth_over_rsth2, 0.0} ;
        // d(dph/dx^i)/dph
        double ddph_dph[3] = {-cph*or_sth, -sph*or_sth, 0.0} ;

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                d2ph[i][j] = ddph_dr[i]*dr_dx[j] + ddph_dth[i]*dth_dx[j] + ddph_dph[i]*dph_dx[j] ;
    }

    // H_{ij} = d_i N_j = d_i(dr/dx^j) - htt*dth_dx[i]*dth_dx[j]
    //        - hpp*dph_dx[i]*dph_dx[j]
    //        - htp*(dth_dx[i]*dph_dx[j] + dph_dx[i]*dth_dx[j])
    //        - ht*d2th[i][j] - hp*d2ph[i][j]
    double H[3][3] ;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            H[i][j] = d2r[i][j]
                     - htt * dth_dx[i] * dth_dx[j]
                     - hpp * dph_dx[i] * dph_dx[j]
                     - htp * (dth_dx[i]*dph_dx[j] + dph_dx[i]*dth_dx[j])
                     - ht * d2th[i][j]
                     - hp * d2ph[i][j] ;
        }
    }

    // Expansion: Theta = P^{ij}(H_{ij} - Gamma^k_{ij} N_k) / |N| + s^i s^j K_{ij} - K
    double div_s = 0 ;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double P_ij = ginv[sym_idx(i,j)] - sU[i]*sU[j] ;
            div_s += P_ij * (H[i][j] - GammaN[i][j]) ;
        }
    }
    div_s *= inv_N ;

    // s^i s^j K_{ij}
    double sKs = 0 ;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            sKs += sU[i] * sU[j] * md.K_ij[sym_idx(i,j)] ;

    return div_s + sKs - md.K_trace ;
}

// ============================================================================
// Residual computation (expansion at all collocation points)
// ============================================================================

void apparent_horizon_finder_t::compute_residual(
    std::vector<metric_data_t> const& mdata,
    double const* a,
    std::vector<double>& residual
) const
{
    int np = basis_->n_points() ;

    std::vector<double> h(np), dh_dth(np), dh_dph(np) ;
    std::vector<double> d2h_dth2(np), d2h_dthdph(np), d2h_dph2(np) ;
    basis_->evaluate(a, h.data(), dh_dth.data(), dh_dph.data(),
                     d2h_dth2.data(), d2h_dthdph.data(), d2h_dph2.data()) ;

    residual.resize(np) ;
    for (int it = 0; it < basis_->n_theta(); ++it) {
        for (int ip = 0; ip < basis_->n_phi(); ++ip) {
            int pt = basis_->point_index(it, ip) ;
            residual[pt] = compute_expansion(
                pt, mdata[pt],
                h.data(), dh_dth.data(), dh_dph.data(),
                d2h_dth2.data(), d2h_dthdph.data(), d2h_dph2.data(),
                basis_->theta(it), basis_->phi(ip)
            ) ;
        }
    }
}

// ============================================================================
// Jacobian computation via forward differences on spectral coefficients
// ============================================================================

void apparent_horizon_finder_t::compute_jacobian(
    std::vector<metric_data_t> const& mdata,
    double const* a,
    std::vector<double> const& residual,
    std::vector<double>& J
) const
{
    int np = basis_->n_points() ;
    int nc = basis_->n_coeffs() ;
    double eps = cfg_.jacobian_eps ;

    J.resize(np * nc) ;

    // perturb each coefficient and recompute expansion
    // metric data is cached — only the geometric terms change
    std::vector<double> a_pert(a, a + nc) ;
    std::vector<double> res_pert(np) ;

    for (int c = 0; c < nc; ++c) {
        double old_val = a_pert[c] ;
        a_pert[c] = old_val + eps ;

        compute_residual(mdata, a_pert.data(), res_pert) ;

        for (int p = 0; p < np; ++p) {
            J[p * nc + c] = (res_pert[p] - residual[p]) / eps ;
        }

        a_pert[c] = old_val ;
    }
}

// ============================================================================
// Solve normal equations: (J^T J) delta_a = -J^T residual
// ============================================================================

bool apparent_horizon_finder_t::solve_normal_equations(
    std::vector<double> const& J,
    std::vector<double> const& residual,
    int n_rows,
    int n_cols,
    std::vector<double>& delta_a
) const
{
    // J^T J (n_cols x n_cols)
    std::vector<double> JTJ(n_cols * n_cols, 0.0) ;
    // -J^T r (n_cols)
    std::vector<double> JTr(n_cols, 0.0) ;

    for (int i = 0; i < n_cols; ++i) {
        for (int j = i; j < n_cols; ++j) {
            double val = 0 ;
            for (int p = 0; p < n_rows; ++p) {
                val += J[p*n_cols + i] * J[p*n_cols + j] ;
            }
            JTJ[i*n_cols + j] = val ;
            JTJ[j*n_cols + i] = val ;
        }
        double rval = 0 ;
        for (int p = 0; p < n_rows; ++p) {
            rval += J[p*n_cols + i] * residual[p] ;
        }
        JTr[i] = -rval ;
    }

    // Cholesky factorization of JTJ (it's SPD when J has full column rank)
    // L L^T = JTJ, in-place in lower triangle of JTJ
    for (int i = 0; i < n_cols; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = JTJ[i*n_cols + j] ;
            for (int k = 0; k < j; ++k) {
                sum -= JTJ[i*n_cols + k] * JTJ[j*n_cols + k] ;
            }
            if (i == j) {
                if (sum <= 0) return false ; // not positive definite
                JTJ[i*n_cols + i] = std::sqrt(sum) ;
            } else {
                JTJ[i*n_cols + j] = sum / JTJ[j*n_cols + j] ;
            }
        }
    }

    // Forward substitution: L y = JTr
    delta_a.resize(n_cols) ;
    for (int i = 0; i < n_cols; ++i) {
        double sum = JTr[i] ;
        for (int k = 0; k < i; ++k) {
            sum -= JTJ[i*n_cols + k] * delta_a[k] ;
        }
        delta_a[i] = sum / JTJ[i*n_cols + i] ;
    }

    // Back substitution: L^T x = y
    for (int i = n_cols - 1; i >= 0; --i) {
        double sum = delta_a[i] ;
        for (int k = i + 1; k < n_cols; ++k) {
            sum -= JTJ[k*n_cols + i] * delta_a[k] ;
        }
        delta_a[i] = sum / JTJ[i*n_cols + i] ;
    }

    return true ;
}

// ============================================================================
// Surface diagnostics (area, irreducible mass)
// ============================================================================

ah_result_t apparent_horizon_finder_t::compute_diagnostics(double const* a) const
{
    ah_result_t res ;
    int np = basis_->n_points() ;

    std::vector<double> h(np), dh_dth(np), dh_dph(np) ;
    basis_->evaluate(a, h.data(), dh_dth.data(), dh_dph.data()) ;

    // area = integral of sqrt(q) dtheta dphi
    // where q is the determinant of the induced 2-metric on the surface.
    // For a surface r=h(th,ph) embedded in flat space:
    //   dA = h^2 sin(th) sqrt(1 + (ht/h)^2 + (hp/(h sin(th)))^2) dth dph
    // In curved space the area element involves the 3-metric, but for now
    // use a simplified estimate; the exact form would require the metric on
    // each collocation point (stored in mdata, which we don't have here).
    // Instead, compute centroid from the coefficients.

    // Centroid: average of surface point positions
    double cx = 0, cy = 0, cz = 0 ;
    for (int it = 0; it < basis_->n_theta(); ++it) {
        for (int ip = 0; ip < basis_->n_phi(); ++ip) {
            int pt = basis_->point_index(it, ip) ;
            double th = basis_->theta(it) ;
            double ph = basis_->phi(ip) ;
            double r = h[pt] ;
            cx += r * std::sin(th) * std::cos(ph) ;
            cy += r * std::sin(th) * std::sin(ph) ;
            cz += r * std::cos(th) ;
        }
    }
    cx /= np ; cy /= np ; cz /= np ;
    res.centroid = {cfg_.center[0] + cx, cfg_.center[1] + cy, cfg_.center[2] + cz} ;

    res.coefficients.assign(a, a + basis_->n_coeffs()) ;
    return res ;
}

// ============================================================================
// Stagger delta: use coordinate spacing of finest level near center
// ============================================================================

double apparent_horizon_finder_t::get_stagger_delta() const
{
    auto& coord_system = grace::coordinate_system::get() ;
    // use the finest grid spacing available; probe at center
    // We search for the quadrant containing the center, get its spacing
    point_host_t center_pt{0, cfg_.center} ;
    auto carr = sc_array_new_data(&center_pt, sizeof(point_host_t), 1) ;

    auto p4est = grace::amr::forest::get().get() ;
    std::vector<intersected_cell_descriptor_t> cells_h ;
    std::vector<size_t> pts_h ;
    intersected_cell_set_t set{&cells_h, &pts_h} ;
    p4est->user_pointer = static_cast<void*>(&set) ;
    p4est_search_local(p4est, false, nullptr, &grace_search_points, carr) ;

    double dx = 1.0 ; // fallback
    if (!cells_h.empty()) {
        dx = coord_system.get_spacing(cells_h[0].q) ;
    }
    // use MPI_MIN to get the finest spacing across all ranks
    double dx_global ;
    parallel::mpi_allreduce(&dx, &dx_global, 1, mpi_min) ;
    return dx_global ;
}

// ============================================================================
// Main finder routine
// ============================================================================

ah_result_t apparent_horizon_finder_t::find()
{
    int np = basis_->n_points() ;
    int nc = basis_->n_coeffs() ;

    // variable indices to interpolate
    std::vector<int> var_idx = {
        GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_,
        CHI_,
        ATXX_, ATXY_, ATXZ_, ATYY_, ATYZ_, ATZZ_,
        KHAT_, THETA_
    } ;

    ah_result_t result ;
    result.found = false ;

    double delta = get_stagger_delta() ;

    for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
        // 1. generate surface + staggered points
        std::vector<point_host_t> points ;
        std::vector<std::array<double,2>> angles ;
        int n_colloc ;
        generate_surface_points(points, angles, n_colloc, delta) ;

        // 2. p4est search + interpolation setup
        std::vector<size_t> local_point_indices ;
        setup_interpolation(points, local_point_indices) ;

        // 3. interpolate metric variables
        auto& state = grace::variable_list::get().getstate() ;
        Kokkos::View<double**,grace::default_space> ivals_d("ah_ivals", 0, 0) ;
        interpolator_.interpolate(state, var_idx, ivals_d) ;
        auto ivals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_d) ;

        // gather interpolated data from all ranks: each rank only
        // has values for points on its local partition. Scatter local
        // results into a global array indexed by global point id, then
        // allreduce so every rank has the full data.
        int n_total_pts = static_cast<int>(points.size()) ;
        int n_vars = static_cast<int>(var_idx.size()) ;
        std::vector<double> global_ivals(n_total_pts * n_vars, 0.0) ;

        for (size_t loc = 0; loc < local_point_indices.size(); ++loc) {
            size_t glob = local_point_indices[loc] ;
            for (int v = 0; v < n_vars; ++v) {
                global_ivals[glob * n_vars + v] = ivals_h(loc, v) ;
            }
        }

        // each point is owned by exactly one rank; sum recovers full data
        std::vector<double> reduced_ivals(n_total_pts * n_vars, 0.0) ;
        parallel::mpi_allreduce(global_ivals.data(), reduced_ivals.data(),
                                n_total_pts * n_vars, mpi_sum) ;

        // wrap into a host view for extract_metric_data
        Kokkos::View<double**, Kokkos::HostSpace> ivals_global("ah_global", n_total_pts, n_vars) ;
        for (int i = 0; i < n_total_pts; ++i)
            for (int v = 0; v < n_vars; ++v)
                ivals_global(i, v) = reduced_ivals[i * n_vars + v] ;

        // 4. extract metric data
        std::vector<metric_data_t> mdata ;
        extract_metric_data(ivals_global, n_colloc, delta, mdata) ;

        // 5. compute residual
        std::vector<double> residual ;
        compute_residual(mdata, coeffs_.data(), residual) ;

        double res_norm = 0 ;
        for (auto r : residual) res_norm += r*r ;
        res_norm = std::sqrt(res_norm / np) ;

        if (cfg_.verbose) {
            GRACE_INFO("AH finder [{}] iter {}: ||Theta|| = {:.6e}",
                       cfg_.name, iter, res_norm) ;
        }

        if (res_norm < cfg_.tolerance) {
            result.found = true ;
            result.iterations = iter ;
            result.residual = res_norm ;
            auto diag = compute_diagnostics(coeffs_.data()) ;
            result.centroid = diag.centroid ;
            result.coefficients = diag.coefficients ;
            // compute proper area via quadrature with metric
            compute_area_with_metric(mdata, result) ;
            last_result_ = result ;
            return result ;
        }

        // 6. compute Jacobian and update
        std::vector<double> J ;
        compute_jacobian(mdata, coeffs_.data(), residual, J) ;

        std::vector<double> delta_a ;
        bool ok = solve_normal_equations(J, residual, np, nc, delta_a) ;
        if (!ok) {
            GRACE_WARN("AH finder [{}]: normal equations singular at iter {}", cfg_.name, iter) ;
            break ;
        }

        for (int c = 0; c < nc; ++c) {
            coeffs_[c] += delta_a[c] ;
        }

        result.iterations = iter + 1 ;
        result.residual = res_norm ;
    }

    last_result_ = result ;
    return result ;
}

// ============================================================================
// Proper area computation using metric data at collocation points
// ============================================================================

void apparent_horizon_finder_t::compute_area_with_metric(
    std::vector<metric_data_t> const& mdata,
    ah_result_t& result
) const
{
    int np = basis_->n_points() ;
    std::vector<double> h(np), dh_dth(np), dh_dph(np) ;
    basis_->evaluate(result.coefficients.data(), h.data(), dh_dth.data(), dh_dph.data()) ;

    // Area = integral sqrt(det q) dtheta dphi where q is the induced 2-metric.
    // The GL quadrature weight w(it,ip) = w_GL * 2pi/nphi integrates over
    // the unit sphere: sum w_i f_i = integral f dOmega = integral f sin(th) dth dphi.
    // So: A = integral sqrt(det q) dth dphi = sum w_i sqrt(det q_i) / sin(th_i).
    double area = 0 ;
    for (int it = 0; it < basis_->n_theta(); ++it) {
        double sth = std::sin(basis_->theta(it)) ;
        double inv_sth = (std::abs(sth) > 1e-14) ? 1.0 / sth : 0.0 ;
        for (int ip = 0; ip < basis_->n_phi(); ++ip) {
            int pt = basis_->point_index(it, ip) ;
            double th = basis_->theta(it) ;
            double ph = basis_->phi(ip) ;
            double r = h[pt] ;
            double ht_v = dh_dth[pt] ;
            double hp_v = dh_dph[pt] ;

            double cth = std::cos(th), cph = std::cos(ph), sph = std::sin(ph) ;
            // tangent vectors e_theta, e_phi on the embedded surface
            double et[3] = {
                ht_v*sth*cph + r*cth*cph,
                ht_v*sth*sph + r*cth*sph,
                ht_v*cth     - r*sth
            } ;
            double ep[3] = {
                hp_v*sth*cph - r*sth*sph,
                hp_v*sth*sph + r*sth*cph,
                hp_v*cth
            } ;

            auto const& g = mdata[pt].gamma ;
            auto sym_idx = [](int a, int b) -> int {
                if (a > b) std::swap(a,b) ;
                if (a==0) return b ;
                if (a==1) return b + 2 ;
                return 5 ;
            } ;
            auto contract = [&](double const* u, double const* v) -> double {
                double val = 0 ;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        val += g[sym_idx(i,j)] * u[i] * v[j] ;
                return val ;
            } ;

            double q_tt = contract(et, et) ;
            double q_tp = contract(et, ep) ;
            double q_pp = contract(ep, ep) ;
            double detq = q_tt * q_pp - q_tp * q_tp ;

            area += basis_->weight(it, ip) * std::sqrt(std::max(detq, 0.0)) * inv_sth ;
        }
    }

    result.area = area ;
    result.irreducible_mass = std::sqrt(area / (16.0 * M_PI)) ;
}

// ============================================================================
// Surface flux computation
// ============================================================================

ah_flux_result_t apparent_horizon_finder_t::compute_surface_fluxes()
{
    ah_flux_result_t fluxes ;

    if (!last_result_.found) return fluxes ;

    int np = basis_->n_points() ;

    // We need metric vars + gauge vars + GRMHD primitives
    // For the surface flux we only need center points, no staggered
    std::vector<double> h(np), dh_dth(np), dh_dph(np) ;
    basis_->evaluate(coeffs_.data(), h.data(), dh_dth.data(), dh_dph.data()) ;

    // generate only center points on the surface (no staggered)
    std::vector<point_host_t> points(np) ;
    std::vector<std::array<double,2>> angles(np) ;
    for (int it = 0; it < basis_->n_theta(); ++it) {
        for (int ip = 0; ip < basis_->n_phi(); ++ip) {
            int pt = basis_->point_index(it, ip) ;
            double th = basis_->theta(it) ;
            double ph = basis_->phi(ip) ;
            double r = h[pt] ;
            double sth = std::sin(th), cth = std::cos(th) ;
            double sph = std::sin(ph), cph = std::cos(ph) ;
            double x = cfg_.center[0] + r * sth * cph ;
            double y = cfg_.center[1] + r * sth * sph ;
            double z = cfg_.center[2] + r * cth ;
            points[pt] = {0, {x, y, z}} ;
            angles[pt] = {th, ph} ;
        }
    }

    // Variables to interpolate:
    // Evolved: gtxx..gtzz(6), chi(1), betax..z(3), alp(1) = 11
    // Aux: rho, eps, press, zvecx..z(3), bx..z(3) = 9

    // evolved var indices
    std::vector<int> evol_idx = {
        GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_,
        CHI_, BETAX_, BETAY_, BETAZ_, ALP_
    } ;
    // aux var indices
    std::vector<int> aux_idx = {
        RHO_, EPS_, PRESS_, ZVECX_, ZVECY_, ZVECZ_, BX_, BY_, BZ_
    } ;

    // p4est search + interpolation
    std::vector<size_t> local_point_indices ;
    setup_interpolation(points, local_point_indices) ;

    // interpolate evolved variables
    auto& state = grace::variable_list::get().getstate() ;
    Kokkos::View<double**, grace::default_space> ivals_evol_d("ah_flux_evol", 0, 0) ;
    interpolator_.interpolate(state, evol_idx, ivals_evol_d) ;
    auto ivals_evol_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_evol_d) ;

    // interpolate auxiliary variables
    auto& aux = grace::variable_list::get().getaux() ;
    Kokkos::View<double**, grace::default_space> ivals_aux_d("ah_flux_aux", 0, 0) ;
    interpolator_.interpolate(aux, aux_idx, ivals_aux_d) ;
    auto ivals_aux_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_aux_d) ;

    // gather to all ranks via allreduce
    int n_evol = static_cast<int>(evol_idx.size()) ;
    int n_aux = static_cast<int>(aux_idx.size()) ;
    std::vector<double> global_evol(np * n_evol, 0.0) ;
    std::vector<double> global_aux(np * n_aux, 0.0) ;

    for (size_t loc = 0; loc < local_point_indices.size(); ++loc) {
        size_t glob = local_point_indices[loc] ;
        for (int v = 0; v < n_evol; ++v)
            global_evol[glob * n_evol + v] = ivals_evol_h(loc, v) ;
        for (int v = 0; v < n_aux; ++v)
            global_aux[glob * n_aux + v] = ivals_aux_h(loc, v) ;
    }

    std::vector<double> reduced_evol(np * n_evol, 0.0) ;
    std::vector<double> reduced_aux(np * n_aux, 0.0) ;
    parallel::mpi_allreduce(global_evol.data(), reduced_evol.data(),
                            np * n_evol, mpi_sum) ;
    parallel::mpi_allreduce(global_aux.data(), reduced_aux.data(),
                            np * n_aux, mpi_sum) ;
    global_evol.swap(reduced_evol) ;
    global_aux.swap(reduced_aux) ;

    // integrate fluxes over the AH surface
    double mdot = 0, edot = 0, ldot = 0, phi = 0 ;

    for (int it = 0; it < basis_->n_theta(); ++it) {
        double sth = std::sin(basis_->theta(it)) ;
        double inv_sth = (std::abs(sth) > 1e-14) ? 1.0 / sth : 0.0 ;
        for (int ip = 0; ip < basis_->n_phi(); ++ip) {
            int pt = basis_->point_index(it, ip) ;
            double th = basis_->theta(it) ;
            double ph = basis_->phi(ip) ;
            double r = h[pt] ;
            double ht_v = dh_dth[pt] ;
            double hp_v = dh_dph[pt] ;
            double cth = std::cos(th), cph = std::cos(ph), sph = std::sin(ph) ;

            // extract interpolated values
            double gtxx = global_evol[pt * n_evol + 0] ;
            double gtxy = global_evol[pt * n_evol + 1] ;
            double gtxz = global_evol[pt * n_evol + 2] ;
            double gtyy = global_evol[pt * n_evol + 3] ;
            double gtyz = global_evol[pt * n_evol + 4] ;
            double gtzz = global_evol[pt * n_evol + 5] ;
            double chi  = global_evol[pt * n_evol + 6] ;
            double betax = global_evol[pt * n_evol + 7] ;
            double betay = global_evol[pt * n_evol + 8] ;
            double betaz = global_evol[pt * n_evol + 9] ;
            double alp   = global_evol[pt * n_evol + 10] ;

            double rho   = global_aux[pt * n_aux + 0] ;
            double eps   = global_aux[pt * n_aux + 1] ;
            double press = global_aux[pt * n_aux + 2] ;
            double zx    = global_aux[pt * n_aux + 3] ;
            double zy    = global_aux[pt * n_aux + 4] ;
            double zz_v  = global_aux[pt * n_aux + 5] ;
            double bx    = global_aux[pt * n_aux + 6] ;
            double by    = global_aux[pt * n_aux + 7] ;
            double bz    = global_aux[pt * n_aux + 8] ;

            // build metric
            metric_array_t metric(
                {gtxx, gtxy, gtxz, gtyy, gtyz, gtzz}, chi,
                {betax, betay, betaz}, alp
            ) ;

            // surface tangent vectors
            double et[3] = {
                ht_v*sth*cph + r*cth*cph,
                ht_v*sth*sph + r*cth*sph,
                ht_v*cth     - r*sth
            } ;
            double ep[3] = {
                hp_v*sth*cph - r*sth*sph,
                hp_v*sth*sph + r*sth*cph,
                hp_v*cth
            } ;

            // induced 2-metric determinant
            auto sym_idx = [](int a, int b) -> int {
                if (a > b) std::swap(a,b) ;
                if (a==0) return b ;
                if (a==1) return b + 2 ;
                return 5 ;
            } ;
            auto gcontract = [&](double const* u, double const* v) -> double {
                double val = 0 ;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        val += metric.gamma(sym_idx(i,j)) * u[i] * v[j] ;
                return val ;
            } ;

            double q_tt = gcontract(et, et) ;
            double q_tp = gcontract(et, ep) ;
            double q_pp = gcontract(ep, ep) ;
            double detq = q_tt * q_pp - q_tp * q_tp ;
            double sqrt_detq = std::sqrt(std::max(detq, 0.0)) ;

            // outward unit normal s^i to the surface (within the spatial slice)
            // F = r - h(th,ph) → ∂F/∂x^i = N_i
            // N_i = dr/dx^i - dh/dth * dth/dx^i - dh/dph * dph/dx^i
            double x_loc = r * sth * cph ;
            double y_loc = r * sth * sph ;
            double z_loc = r * cth ;
            double r_inv = 1.0 / std::max(r, 1e-30) ;
            double rxy2 = x_loc*x_loc + y_loc*y_loc ;
            double rxy = std::sqrt(rxy2) ;
            double rxy_inv = (rxy > 1e-30) ? 1.0 / rxy : 0.0 ;

            // dr/dx^i
            double drdx[3] = { sth*cph, sth*sph, cth } ;
            // dth/dx^i
            double dthdx[3] = { cth*cph*r_inv, cth*sph*r_inv, -sth*r_inv } ;
            // dph/dx^i
            double dphdx[3] = { -sph*rxy_inv, cph*rxy_inv, 0.0 } ;

            double N[3] ;
            for (int i = 0; i < 3; ++i) {
                N[i] = drdx[i] - ht_v * dthdx[i] - hp_v * dphdx[i] ;
            }
            // |N| = sqrt(gamma^{ij} N_i N_j)
            double Nmag2 = 0 ;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    Nmag2 += metric.invgamma(sym_idx(i,j)) * N[i] * N[j] ;
            double Nmag = std::sqrt(std::max(Nmag2, 1e-30)) ;

            // s^i = gamma^{ij} N_j / |N| (contravariant unit outward normal)
            double s_up[3] = {0, 0, 0} ;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    s_up[i] += metric.invgamma(sym_idx(i,j)) * N[j] ;
            for (int i = 0; i < 3; ++i) s_up[i] /= Nmag ;

            // 4-velocity: z^i is the Valencia transport velocity
            // W = sqrt(1 + gamma_{ij} z^i z^j)
            double W = std::sqrt(1.0 + metric.square_vec({zx, zy, zz_v})) ;
            double ooalp = 1.0 / std::max(alp, 1e-30) ;
            double u0 = W * ooalp ;

            // u^i = z^i/alpha - beta^i/alpha (following bh_diagnostics convention)
            // Actually: u^mu = (W/alpha, z^i/alpha - beta^i W/alpha) from Valencia formulation
            // In bh_diagnostics: uU[i] = v_i - beta^i/alpha where v_i = z^i / W
            // Let me follow exactly the bh_diagnostics convention:
            double vx = zx / W, vy = zy / W, vz_v = zz_v / W ;
            std::array<double,4> uU {{
                u0,
                vx - ooalp * metric.beta(0),
                vy - ooalp * metric.beta(1),
                vz_v - ooalp * metric.beta(2)
            }} ;
            auto uD = metric.lower_4vec(uU) ;

            // magnetic 4-vector
            double b0 = uD[1]*bx + uD[2]*by + uD[3]*bz ;
            double b1 = (bx + b0*uU[1]) / u0 ;
            double b2 = (by + b0*uU[2]) / u0 ;
            double b3 = (bz + b0*uU[3]) / u0 ;
            auto bD = metric.lower_4vec({b0, b1, b2, b3}) ;
            double bsq = b0*bD[0] + b1*bD[1] + b2*bD[2] + b3*bD[3] ;

            // project velocity and B-field onto the surface normal
            // s_i = N_i / |N| (covariant unit normal)
            double s_down[3] ;
            for (int i = 0; i < 3; ++i) s_down[i] = N[i] / Nmag ;

            // u_s = u^i s_i (spatial velocity through the surface)
            u_s = uU[1]*s_down[0] + uU[2]*s_down[1] + uU[3]*s_down[2] ;
            double b_s = b1*s_down[0] + b2*s_down[1] + b3*s_down[2] ;

            // angular momentum: u_phi = x * u_y - y * u_x (z-component of angular momentum)
            double x_abs = cfg_.center[0] + x_loc ;
            double y_abs = cfg_.center[1] + y_loc ;
            double u_phi = x_abs * uD[2] - y_abs * uD[1] ;
            double b_phi = x_abs * bD[2] - y_abs * bD[1] ;

            double rhoh = rho + rho * eps + press ;

            // T^s_0 = (rho*h + b^2) u_s u_0 - b_s b_0
            double t_s_0 = (rhoh + bsq) * u_s * uD[0] - b_s * bD[0] ;
            // T^s_phi = (rho*h + b^2) u_s u_phi - b_s b_phi
            double t_s_phi = (rhoh + bsq) * u_s * u_phi - b_s * b_phi ;
            // magnetic flux through surface: |b^s u^0 - b^0 u^s| / 2
            // where b^s = b^i s^i (contravariant) and u^s = u^i s^i (contravariant)
            double b_s_up = b1*s_up[0] + b2*s_up[1] + b3*s_up[2] ;
            double u_s_up = uU[1]*s_up[0] + uU[2]*s_up[1] + uU[3]*s_up[2] ;
            double phi_l = 0.5 * std::abs(b_s_up * u0 - b0 * u_s_up) ;

            // area element: sqrt(det q) dtheta dphi
            // GL weight w = sin(theta) dtheta * (2pi/nphi), so dtheta dphi = w / sin(theta)
            double dA = sqrt_detq * basis_->weight(it, ip) * inv_sth ;

            // sqrt(gamma) factor for flux normalization
            double sqrtg = metric.sqrtg() ;

            mdot += -sqrtg * rho * u_s * dA ;
            edot += -sqrtg * t_s_0 * dA ;
            ldot +=  sqrtg * t_s_phi * dA ;
            phi  +=  sqrtg * phi_l * dA ;
        }
    }

    fluxes.mdot = mdot ;
    fluxes.edot = edot ;
    fluxes.ldot = ldot ;
    fluxes.phi  = phi ;

    return fluxes ;
}

// ============================================================================
// Output
// ============================================================================

void apparent_horizon_finder_t::initialize_output_file()
{
    if (parallel::mpi_comm_rank() != 0) return ;
    auto& grace_runtime = grace::runtime::get() ;
    std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;
    std::string pfname = grace_runtime.scalar_io_basename() + "ah_" + cfg_.name + ".dat" ;
    outfilepath_ = bdir / pfname ;
    if (!std::filesystem::exists(outfilepath_)) {
        std::ofstream outfile(outfilepath_.string()) ;
        static constexpr size_t w = 20 ;
        outfile << std::left << std::setw(w) << "Iteration"
                << std::left << std::setw(w) << "Time"
                << std::left << std::setw(w) << "Found"
                << std::left << std::setw(w) << "Iterations"
                << std::left << std::setw(w) << "Residual"
                << std::left << std::setw(w) << "Area"
                << std::left << std::setw(w) << "M_irr"
                << std::left << std::setw(w) << "Cx"
                << std::left << std::setw(w) << "Cy"
                << std::left << std::setw(w) << "Cz"
                << '\n' ;
    }
}

void apparent_horizon_finder_t::write_result(ah_result_t const& res)
{
    if (parallel::mpi_comm_rank() != 0) return ;
    auto& grace_runtime = grace::runtime::get() ;
    size_t const iter = grace_runtime.iteration() ;
    double const time = grace_runtime.time() ;
    std::ofstream outfile(outfilepath_.string(), std::ios::app) ;
    outfile << std::fixed << std::setprecision(15) ;
    outfile << std::left << iter << '\t'
            << std::left << time << '\t'
            << std::left << res.found << '\t'
            << std::left << res.iterations << '\t'
            << std::left << res.residual << '\t'
            << std::left << res.area << '\t'
            << std::left << res.irreducible_mass << '\t'
            << std::left << res.centroid[0] << '\t'
            << std::left << res.centroid[1] << '\t'
            << std::left << res.centroid[2] << '\n' ;
}

void apparent_horizon_finder_t::initialize_flux_output_file()
{
    if (parallel::mpi_comm_rank() != 0) return ;
    auto& grace_runtime = grace::runtime::get() ;
    std::filesystem::path bdir = grace_runtime.scalar_io_basepath() ;
    std::string pfname = grace_runtime.scalar_io_basename() + "ah_fluxes_" + cfg_.name + ".dat" ;
    flux_outfilepath_ = bdir / pfname ;
    if (!std::filesystem::exists(flux_outfilepath_)) {
        std::ofstream outfile(flux_outfilepath_.string()) ;
        static constexpr size_t w = 20 ;
        outfile << std::left << std::setw(w) << "Iteration"
                << std::left << std::setw(w) << "Time"
                << std::left << std::setw(w) << "Mdot"
                << std::left << std::setw(w) << "Edot"
                << std::left << std::setw(w) << "Ldot"
                << std::left << std::setw(w) << "Phi"
                << '\n' ;
    }
}

void apparent_horizon_finder_t::write_flux_result(ah_flux_result_t const& res)
{
    if (parallel::mpi_comm_rank() != 0) return ;
    auto& grace_runtime = grace::runtime::get() ;
    size_t const iter = grace_runtime.iteration() ;
    double const time = grace_runtime.time() ;
    std::ofstream outfile(flux_outfilepath_.string(), std::ios::app) ;
    outfile << std::fixed << std::setprecision(15) ;
    outfile << std::left << iter << '\t'
            << std::left << time << '\t'
            << std::left << res.mdot << '\t'
            << std::left << res.edot << '\t'
            << std::left << res.ldot << '\t'
            << std::left << res.phi << '\n' ;
}

} // namespace grace
