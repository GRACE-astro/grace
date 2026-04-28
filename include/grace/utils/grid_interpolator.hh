/**
 * @file grid_interpolator.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Unified grid-to-point interpolation for GRACE.
 *
 *        Provides a single class template
 *
 *            grid_interpolator_t<poly_kind Kind, int Degree>
 *
 *        with two specializations:
 *
 *          - poly_kind::lagrange   classical barycentric Lagrange of given
 *                                  order; values only.
 *          - poly_kind::hermite    tensor-product Hermite (currently cubic,
 *                                  Degree==3), with both value and analytic
 *                                  gradient. Knot derivatives are NOT input
 *                                  data: they are computed in-kernel by
 *                                  central FD on the same f stencil that
 *                                  Lagrange uses, so the data API is
 *                                  identical across kinds.
 *
 *        Each diagnostic picks its kind by template parameter at the call
 *        site. The shared search infrastructure (point_host_t,
 *        intersected_cell_descriptor_t, grace_search_points, ...) lives
 *        here and is reused by both specializations.
 *
 *        Backward compatibility: include/grace/utils/lagrange_interpolation.hh
 *        is a thin shim around this header that exposes
 *
 *            template<int Order>
 *            using lagrange_interpolator_t =
 *                grid_interpolator_t<poly_kind::lagrange, Order>;
 *
 *        so existing call sites keep working unchanged.
 *
 * @date 2026-04-27
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference / Volume
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
 */

#ifndef GRACE_UTILS_GRID_INTERPOLATOR_HH
#define GRACE_UTILS_GRID_INTERPOLATOR_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/math.hh>
#include <grace/errors/assert.hh>
#include <grace/IO/surface_IO_utils.hh>
#include <grace/coordinates/coordinate_systems.hh>

#include <Kokkos_Core.hpp>

#include <vector>
#include <array>
#include <utility>

namespace grace {

//=========================================================================
// Shared search / cell descriptor infrastructure (used by both kinds).
//=========================================================================

/**
 * @brief Point descriptor used to drive the p4est local search.
 */
using point_host_t = std::pair<size_t, std::array<double,3>>;

/**
 * @brief Local cell descriptor for the cell containing a point.
 *        (i,j,k) are zero-offset cell indices within quadrant q.
 */
struct intersected_cell_descriptor_t {
    size_t i, j, k;
    size_t q;
};

/**
 * @brief Set of cells plus identifiers tying them to the points they contain.
 */
struct intersected_cell_set_t {
    std::vector<intersected_cell_descriptor_t>* cells;
    std::vector<size_t>* point_idx;
};

/**
 * @brief p4est_search_local callback: finds which local quadrant contains a
 *        given point, then computes the (i,j,k) of the cell within that
 *        quadrant. The forest's user_pointer must be set to an
 *        intersected_cell_set_t* before invocation.
 */
static int
grace_search_points(
    p4est_t* forest,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant,
    p4est_locidx_t local_num,
    void* point
)
{
    DECLARE_GRID_EXTENTS;
    auto point_desc = static_cast<point_host_t*>(point);
    auto p_idx = point_desc->first;
    auto pcoords = point_desc->second;
    auto cube = amr::detail::make_cube(amr::quadrant_t{quadrant}, which_tree);
    bool contained = (
        pcoords[0] < cube.v[1][0] && pcoords[0] >= cube.v[0][0] &&
        pcoords[1] < cube.v[2][1] && pcoords[1] >= cube.v[0][1] &&
        pcoords[2] < cube.v[4][2] && pcoords[2] >= cube.v[0][2]
    );
    if (!contained) return 0;

    if (local_num >= 0) {
        auto quadid = local_num;
        double xoff = pcoords[0] - cube.v[0][0];
        double yoff = pcoords[1] - cube.v[0][1];
        double zoff = pcoords[2] - cube.v[0][2];
        double idx  = static_cast<double>(nx) / (cube.v[1][0] - cube.v[0][0]);
        size_t i = std::min(nx-1, std::max(0UL, size_t(xoff * idx)));
        size_t j = std::min(ny-1, std::max(0UL, size_t(yoff * idx)));
        size_t k = std::min(nz-1, std::max(0UL, size_t(zoff * idx)));
        intersected_cell_descriptor_t desc;
        desc.i = i; desc.j = j; desc.k = k; desc.q = quadid;
        auto intersected_cells = static_cast<intersected_cell_set_t*>(forest->user_pointer);
        intersected_cells->cells->push_back(desc);
        intersected_cells->point_idx->push_back(p_idx);
    }
    return 1;
}

//=========================================================================
// Polynomial kind tag.
//=========================================================================

enum class poly_kind { lagrange, hermite };

//=========================================================================
// Primary template (no general definition).
//=========================================================================

template <poly_kind Kind, int Degree>
struct grid_interpolator_t;

//=========================================================================
// Lagrange specialization.
//
// This is the existing grace::lagrange_interpolator_t<Order> logic, hoisted
// behind the unified template. Order == polynomial degree, stencil width
// Order+1, halo width (Order+1)/2.
//=========================================================================

/**
 * @brief Per-query barycentric weights for Lagrange interpolation of given
 *        order (Order+1 weights per axis).
 */
template <size_t order>
struct interp_weights_t {
    double w[order+1][3];
};

template <int Order>
struct grid_interpolator_t<poly_kind::lagrange, Order> {

    static constexpr poly_kind kind              = poly_kind::lagrange;
    static constexpr int       polynomial_degree = Order;
    static constexpr int       stencil_width     = Order + 1;
    /// Half-width of the stencil: H knots to the left of the query cell,
    /// (Order+1) - H to the right, where H = (Order+1)/2 (integer floor).
    static constexpr int       stencil_half      = (Order + 1) / 2;

    grid_interpolator_t(int valid_gz)
        : _valid_gz(valid_gz)
    {}

    /// Value-only interpolation at the previously-registered query points.
    void interpolate(
        grace::var_array_t data,
        std::vector<int> const& var_idx_h,
        Kokkos::View<double**, grace::default_space>& out
    ) const
    {
        DECLARE_GRID_EXTENTS;
        using namespace grace;
        using namespace Kokkos;

        auto nvars = var_idx_h.size();

        readonly_view_t<int> var_idx;
        deep_copy_vec_to_const_view(var_idx, var_idx_h);

        Kokkos::realloc(out, npoints, nvars);

        auto icells     = intersected_cells;
        auto iweights   = interp_weights;
        auto istencils  = interp_stencils;

        MDRangePolicy<Rank<2>> policy({0,0}, {npoints, static_cast<long>(nvars)});
        parallel_for(
            GRACE_EXECUTION_TAG("IO", "interp_to_sphere"),
            policy,
            KOKKOS_LAMBDA (int const& ip, int const& iv) {
                auto const& cell = icells(ip);
                auto q  = cell.q;
                auto w  = iweights(ip);
                auto u  = subview(data, ALL(), ALL(), ALL(), var_idx(iv), q);
                int bx = istencils(ip,0);
                int by = istencils(ip,1);
                int bz = istencils(ip,2);

                constexpr int H = stencil_half;
                double val{0};
                for (int i = 0; i < Order+1; ++i) {
                    for (int j = 0; j < Order+1; ++j) {
                        for (int k = 0; k < Order+1; ++k) {
                            int io = i - H + bx;
                            int jo = j - H + by;
                            int ko = k - H + bz;
                            int ic = static_cast<int>(ngz + cell.i) + io;
                            int jc = static_cast<int>(ngz + cell.j) + jo;
                            int kc = static_cast<int>(ngz + cell.k) + ko;
                            val += w.w[i][0] * w.w[j][1] * w.w[k][2] * u(ic,jc,kc);
                        }
                    }
                }
                out(ip,iv) = val;
            }
        );
    }

    void compute_weights(
        std::vector<point_host_t> const& points,
        std::vector<size_t> const& ipoints,
        std::vector<intersected_cell_descriptor_t> const& icells
    ) {
        DECLARE_GRID_EXTENTS;
        auto& coord_system = grace::coordinate_system::get();

        npoints = ipoints.size();
        weights.resize(npoints);
        stencils.resize(npoints);

        auto wj = get_barycentric_weights();
        for (int i = 0; i < npoints; ++i) {
            auto pidx = ipoints[i];
            auto const& point_coords = points[pidx].second;
            auto const ijkq = icells[i];
            double const dx = coord_system.get_spacing(ijkq.q);

            // Compute the legal stencil-bias range from the global stencil
            // index constraint
            //
            //     ngz + cell_d + (p - H) + bias_d  in
            //         [ngz - _valid_gz,  nx + ngz + _valid_gz - 1]
            //
            // for p in {0..Order}. This rearranges to
            //
            //     bias_d in [b_min, b_max]
            //     b_min = H - cell_d - _valid_gz
            //     b_max = nx + _valid_gz - 1 - cell_d - (Order - H)
            //
            // Pick the smallest |bias| that lies in [b_min, b_max]:
            // 0 if it's already inside, else clamp to whichever bound is
            // violated.
            constexpr int H = stencil_half;
            std::array<int,3> bias{{0,0,0}};
            {
                std::array<long,3> ijk_l{{
                    static_cast<long>(ijkq.i),
                    static_cast<long>(ijkq.j),
                    static_cast<long>(ijkq.k)
                }};
                std::array<long,3> nxyz{{
                    static_cast<long>(nx),
                    static_cast<long>(ny),
                    static_cast<long>(nz)
                }};
                for (int idir = 0; idir < 3; ++idir) {
                    long const b_min = static_cast<long>(H) - ijk_l[idir]
                                       - static_cast<long>(_valid_gz);
                    long const b_max = nxyz[idir] + static_cast<long>(_valid_gz)
                                       - 1L - ijk_l[idir]
                                       - static_cast<long>(Order - H);
                    ASSERT(b_min <= b_max,
                        "grid_interpolator_t<lagrange," << Order << ">: grid is too small "
                        "for the stencil along axis " << idir
                        << " (cell=" << ijk_l[idir] << ", nx=" << nxyz[idir]
                        << ", _valid_gz=" << _valid_gz
                        << ", b_min=" << b_min << ", b_max=" << b_max << ").");
                    long b = 0;
                    if (b_min > 0)      b = b_min;
                    else if (b_max < 0) b = b_max;
                    bias[idir] = static_cast<int>(b);
                }
            }
            stencils[i] = bias;

            interp_weights_t<Order> iweights;
            for (int idir = 0; idir < 3; ++idir) {
                double norm{0};
                for (int ic = 0; ic < Order+1; ++ic) {
                    int off = ic - H + bias[idir];
                    std::array<size_t,3> ijk{{
                        ijkq.i + off * (idir==0),
                        ijkq.j + off * (idir==1),
                        ijkq.k + off * (idir==2)
                    }};
                    auto pcoords = coord_system.get_physical_coordinates(
                        ijk, ijkq.q, {0.5,0.5,0.5}, false
                    );
                    double wL = dx * wj[ic] /
                        (point_coords[idir] - pcoords[idir]
                         + 1e-15 * std::copysign(1.0, point_coords[idir] - pcoords[idir]));
                    norm += wL;
                    iweights.w[ic][idir] = wL;
                }
                for (int ic = 0; ic < Order+1; ++ic) {
                    iweights.w[ic][idir] /= norm;
                }
            }
            weights[i] = iweights;
        }

        deep_copy_vec_to_const_view(intersected_cells, icells);
        deep_copy_vec_to_const_2D_view(interp_stencils, stencils);
        deep_copy_vec_to_const_view(interp_weights, weights);
    }

    std::array<double, Order+1> get_barycentric_weights() const {
        std::vector<double> nodes(Order+1);
        for (int inode = 0; inode < Order+1; ++inode) nodes[inode] = inode;
        std::array<double, Order+1> w;
        for (int i = 0; i < Order+1; ++i) {
            double ww = 1;
            for (int j = 0; j < Order+1; ++j) {
                if (i == j) continue;
                ww *= nodes[i] - nodes[j];
            }
            w[i] = 1.0 / ww;
        }
        return w;
    }

    int _valid_gz;

    // host storage
    std::vector<std::array<int,3>>          stencils;
    std::vector<interp_weights_t<Order>>    weights;

    // device storage
    int npoints;
    readonly_view_t<intersected_cell_descriptor_t> intersected_cells;
    readonly_view_t<interp_weights_t<Order>>       interp_weights;
    readonly_twod_view_t<int,3>                    interp_stencils;
};

//=========================================================================
// Hermite specialization (currently cubic only, Degree == 3).
//
// Tensor-product cubic Hermite over the cell formed by the two cell-centred
// knots that bracket the query point along each axis. At each of the 8
// corner knots we need
//
//   { f, fx, fy, fz, fxy, fxz, fyz, fxyz }
//
// (8 values per knot, so 64 doubles per query per variable). These knot
// derivatives are NOT input data: they are computed in-kernel by central
// FD on a 4-wide stencil along each axis (the 2 corner knots ± 1-cell halo
// for the FD). The data API is therefore identical to Lagrange order-3.
//
// Stencil width along each axis is 4. The Hermite cell sits between the
// two innermost stencil knots (relative indices 1 and 2 along each axis).
//
// "Half" flag per axis: the Hermite cell hosting the query depends on
// whether the query lies in the left half (h=0) or right half (h=1) of
// its containing grid cell:
//
//     stencil_index_along_axis(p) = ijk + p - 2 + h    (p in {0,1,2,3})
//
//   - h = 0:  Hermite cell = [center(ijk-1), center(ijk)]
//   - h = 1:  Hermite cell = [center(ijk),   center(ijk+1)]
//
// `bias` shifts the whole stencil by ±1 cell when the unshifted stencil
// would run outside the valid ghost-zone budget [ngz-_valid_gz,
// nx+ngz+_valid_gz-1]. Same idea as Lagrange.
//
// The host side is simpler than Lagrange: no barycentric weights to
// precompute. We only store, per query: the cell descriptor, the half
// flags, the bias, the normalised query coordinates u in [0,1]^3, and dx.
//=========================================================================

namespace detail {

/**
 * @brief 1D cubic Hermite basis on [0,1].
 *
 *   psi_0(s) = 2 s^3 - 3 s^2 + 1     value at s=0,  zero at s=1
 *   psi_1(s) = -2 s^3 + 3 s^2        zero at s=0,   value at s=1
 *   phi_0(s) = s^3 - 2 s^2 + s       deriv (in s) at s=0
 *   phi_1(s) = s^3 - s^2             deriv (in s) at s=1
 *
 * The polynomial in normalized coord s reproduces
 *   p(0)=fa, p'(0)=dx*fa', p(1)=fb, p'(1)=dx*fb'
 * so that derivatives input to the basis are knot-physical-x derivatives
 * scaled by dx. We absorb dx scalings into the knot data tuple below.
 *
 * Returned arrays are indexed by the corner index (0=left, 1=right):
 *   psi[a]      : value-of-basis-a at s
 *   phi[a]      : deriv-of-basis-a (in s) at s, divided by dx -- i.e.
 *                 the basis paired with knot data already multiplied by dx.
 *                 We keep the raw phi here and apply dx scaling when
 *                 we form the knot data tuple, see below.
 *   dpsi_ds[a]  : d/ds of psi_a at s
 *   dphi_ds[a]  : d/ds of phi_a at s
 */
GRACE_ALWAYS_INLINE GRACE_HOST_DEVICE
void cubic_hermite_basis_1d(
    double s,
    double psi[2], double phi[2],
    double dpsi_ds[2], double dphi_ds[2]
) {
    double s2 = s * s;
    double s3 = s2 * s;
    psi[0]     =  2.0*s3 - 3.0*s2 + 1.0;
    psi[1]     = -2.0*s3 + 3.0*s2;
    phi[0]     =  s3 - 2.0*s2 + s;
    phi[1]     =  s3 - s2;
    dpsi_ds[0] =  6.0*s2 - 6.0*s;
    dpsi_ds[1] = -6.0*s2 + 6.0*s;
    dphi_ds[0] =  3.0*s2 - 4.0*s + 1.0;
    dphi_ds[1] =  3.0*s2 - 2.0*s;
}

}  // namespace detail

template <int Degree>
struct grid_interpolator_t<poly_kind::hermite, Degree> {

    static_assert(Degree == 3,
                  "grid_interpolator_t<poly_kind::hermite,...> currently "
                  "supports only Degree==3 (cubic). Quintic is a future "
                  "extension; the data API would still take raw f.");

    static constexpr poly_kind kind          = poly_kind::hermite;
    static constexpr int       polynomial_degree = Degree;
    static constexpr int       n_knots_per_axis  = 2;        // cubic
    // 4-wide stencil per axis: 2 Hermite-cell knots + 1-cell FD halo each side.
    static constexpr int       stencil_width = n_knots_per_axis + 2;

    /// Per-query host descriptor.
    struct query_state_t {
        // Cell and stencil-bias bookkeeping (same idea as Lagrange).
        std::array<int,3> bias;
        // Half flags: which half of the containing cell holds the query
        // along each axis (0 = left half, 1 = right half).
        std::array<int,3> half;
        // Normalised query coord within the Hermite cell, in [0,1]^3.
        std::array<double,3> u;
        // Cell width of the containing quadrant.
        double dx;
    };

    grid_interpolator_t(int valid_gz)
        : _valid_gz(valid_gz)
    {}

    void compute_weights(
        std::vector<point_host_t> const& points,
        std::vector<size_t> const& ipoints,
        std::vector<intersected_cell_descriptor_t> const& icells
    ) {
        DECLARE_GRID_EXTENTS;
        auto& coord_system = grace::coordinate_system::get();

        npoints = ipoints.size();
        states.resize(npoints);

        for (int i = 0; i < npoints; ++i) {
            auto pidx = ipoints[i];
            auto const& point_coords = points[pidx].second;
            auto const ijkq = icells[i];
            double const dx = coord_system.get_spacing(ijkq.q);

            // Cell-centre physical coords for cell ijkq.
            std::array<size_t,3> ijk{{ijkq.i, ijkq.j, ijkq.k}};
            auto cell_centre = coord_system.get_physical_coordinates(
                ijk, ijkq.q, {0.5,0.5,0.5}, false
            );

            query_state_t qs;
            qs.dx = dx;

            // For each axis pick the half (which neighbouring cell-centre
            // pair brackets the query) and compute u in [0,1].
            for (int idir = 0; idir < 3; ++idir) {
                // s_local in [-0.5, 0.5] is the offset from cell centre, in
                // units of dx.
                double const s_local = (point_coords[idir] - cell_centre[idir]) / dx;
                if (s_local >= 0.0) {
                    // Right half: Hermite cell = [center(ijk), center(ijk+1)]
                    qs.half[idir] = 1;
                    qs.u[idir]    = s_local;        // already in [0, 0.5]
                } else {
                    // Left half: Hermite cell = [center(ijk-1), center(ijk)]
                    qs.half[idir] = 0;
                    qs.u[idir]    = s_local + 1.0;  // shift into [0.5, 1.0]
                }
            }

            // Bias: keep the 4-wide stencil within the valid ghost-zone
            // budget. Per-axis global stencil index for p in {0,1,2,3} is
            //
            //     ic = ngz + cell_d + (p - 2 + half_d) + bias_d
            //
            // Valid-ghost constraint  ic in [ngz-_valid_gz, nx+ngz+_valid_gz-1]
            // gives, after rearrangement,
            //
            //     bias_d in [b_min, b_max]
            //     b_min  = 2 - half_d - cell_d - _valid_gz
            //     b_max  = nx + _valid_gz - 2 - cell_d - half_d
            //
            // Pick the smallest |bias| in that range. NB: when bias != 0
            // the Hermite cell (between stencil indices 1 and 2) shifts
            // away from the cell-pair containing the query, so the query
            // coordinate u may fall outside [0,1] and the evaluation
            // becomes extrapolation. This is unavoidable near a domain
            // edge with a tight _valid_gz; for the common case
            // (_valid_gz == ngz) the legal range always contains 0 and no
            // such extrapolation occurs.
            std::array<long,3> nxyz{{
                static_cast<long>(nx),
                static_cast<long>(ny),
                static_cast<long>(nz)
            }};
            for (int idir = 0; idir < 3; ++idir) {
                long const cell_l = static_cast<long>(ijk[idir]);
                long const half_l = static_cast<long>(qs.half[idir]);
                long const b_min = 2L - half_l - cell_l - static_cast<long>(_valid_gz);
                long const b_max = nxyz[idir] + static_cast<long>(_valid_gz)
                                   - 2L - cell_l - half_l;
                ASSERT(b_min <= b_max,
                    "grid_interpolator_t<hermite,3>: grid is too small for the "
                    "4-wide stencil along axis " << idir
                    << " (cell=" << cell_l << ", nx=" << nxyz[idir]
                    << ", _valid_gz=" << _valid_gz
                    << ", b_min=" << b_min << ", b_max=" << b_max << ").");
                long b = 0;
                if (b_min > 0)      b = b_min;
                else if (b_max < 0) b = b_max;
                qs.bias[idir] = static_cast<int>(b);
            }

            states[i] = qs;
        }

        deep_copy_vec_to_const_view(intersected_cells, icells);
        deep_copy_vec_to_const_view(query_states, states);
    }

    /// Value-only interpolation. Falls through to evaluate_with_grad and
    /// drops the gradient -- the gradient is essentially free given the
    /// knot derivatives we already FD'd.
    void interpolate(
        grace::var_array_t data,
        std::vector<int> const& var_idx_h,
        Kokkos::View<double**, grace::default_space>& out
    ) const
    {
        Kokkos::View<double***, grace::default_space> dummy_grad("hermite_grad_dummy", 0,0,0);
        interpolate_with_grad(data, var_idx_h, out, dummy_grad, /*want_grad=*/false);
    }

    /**
     * @brief Interpolate values and Cartesian gradients at the query points.
     *
     * @param data      The (i,j,k,ivar,q) variable array.
     * @param var_idx_h Indices of the variables to interpolate.
     * @param out       Output values, shape (npoints, nvars).
     * @param out_grad  Output gradients, shape (npoints, nvars, 3) with last
     *                  index (0,1,2) -> (d/dx, d/dy, d/dz). Only filled when
     *                  want_grad == true.
     */
    void interpolate_with_grad(
        grace::var_array_t data,
        std::vector<int> const& var_idx_h,
        Kokkos::View<double**,  grace::default_space>& out,
        Kokkos::View<double***, grace::default_space>& out_grad,
        bool want_grad = true
    ) const
    {
        DECLARE_GRID_EXTENTS;
        using namespace grace;
        using namespace Kokkos;

        auto nvars = var_idx_h.size();

        readonly_view_t<int> var_idx;
        deep_copy_vec_to_const_view(var_idx, var_idx_h);

        Kokkos::realloc(out, npoints, nvars);
        if (want_grad) {
            Kokkos::realloc(out_grad, npoints, nvars, 3);
        }

        auto icells = intersected_cells;
        auto qs_d   = query_states;
        bool const grad_flag = want_grad;

        MDRangePolicy<Rank<2>> policy({0,0}, {npoints, static_cast<long>(nvars)});
        parallel_for(
            GRACE_EXECUTION_TAG("IO", "interp_hermite_to_sphere"),
            policy,
            KOKKOS_LAMBDA (int const& ip, int const& iv) {
                auto const& cell = icells(ip);
                auto const& qs   = qs_d(ip);
                auto q_id        = cell.q;
                auto u_view      = subview(data, ALL(), ALL(), ALL(), var_idx(iv), q_id);

                double const dx  = qs.dx;
                double const inv_dx = 1.0 / dx;

                // Load the 4x4x4 stencil of f values into a local buffer.
                // Index mapping along axis d for stencil index p in {0,1,2,3}:
                //     ic_d(p) = ngz + cell.[ijk]_d + (p - 2 + qs.half[d] + qs.bias[d])
                int const off_x = static_cast<int>(ngz) + static_cast<int>(cell.i)
                                   - 2 + qs.half[0] + qs.bias[0];
                int const off_y = static_cast<int>(ngz) + static_cast<int>(cell.j)
                                   - 2 + qs.half[1] + qs.bias[1];
                int const off_z = static_cast<int>(ngz) + static_cast<int>(cell.k)
                                   - 2 + qs.half[2] + qs.bias[2];

                double F[4][4][4];
                for (int kp = 0; kp < 4; ++kp)
                for (int jp = 0; jp < 4; ++jp)
                for (int ip_ = 0; ip_ < 4; ++ip_) {
                    F[ip_][jp][kp] = u_view(off_x + ip_, off_y + jp, off_z + kp);
                }

                // Build the 8 corner knot tuples by central FD on F.
                // Corner (a,b,c) in {0,1}^3 maps to stencil index (1+a, 1+b, 1+c).
                //
                // Knot data, with axis dx already absorbed (so the entries
                // are derivatives in normalized cell coord s_d = (x-x_a)/dx):
                //     K[a][b][c][o]
                //   o = 0          : f
                //   o = 1 (bit 0=x): dx * fx
                //   o = 2 (bit 1=y): dx * fy
                //   o = 3          : dx^2 * fxy
                //   o = 4 (bit 2=z): dx * fz
                //   o = 5          : dx^2 * fxz
                //   o = 6          : dx^2 * fyz
                //   o = 7          : dx^3 * fxyz
                //
                // Central-FD formulae (knot at stencil index (P,Q,R)):
                //   fx   = (F[P+1] - F[P-1]) / (2 dx)
                //   fxy  = (F[P+1,Q+1] - F[P-1,Q+1] - F[P+1,Q-1] + F[P-1,Q-1]) / (4 dx dy)
                //   fxyz = sum_corners +/- F / (8 dx dy dz)
                // Multiplying by the appropriate dx-power gives the
                // s-derivative entries directly:
                //   K[1] = dx * fx     = (F[P+1] - F[P-1]) / 2
                //   K[3] = dx^2 * fxy  = (F[++] - F[-+] - F[+-] + F[--]) / 4
                //   K[7] = dx^3 *fxyz  = sum / 8
                // i.e. the dx factors cancel.
                double K[2][2][2][8];
                for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                for (int c = 0; c < 2; ++c) {
                    int const P = 1 + a;
                    int const Q = 1 + b;
                    int const R = 1 + c;

                    K[a][b][c][0] = F[P  ][Q  ][R  ];
                    K[a][b][c][1] = 0.5  * (F[P+1][Q  ][R  ] - F[P-1][Q  ][R  ]);
                    K[a][b][c][2] = 0.5  * (F[P  ][Q+1][R  ] - F[P  ][Q-1][R  ]);
                    K[a][b][c][4] = 0.5  * (F[P  ][Q  ][R+1] - F[P  ][Q  ][R-1]);
                    K[a][b][c][3] = 0.25 * (
                          F[P+1][Q+1][R  ] - F[P-1][Q+1][R  ]
                        - F[P+1][Q-1][R  ] + F[P-1][Q-1][R  ]);
                    K[a][b][c][5] = 0.25 * (
                          F[P+1][Q  ][R+1] - F[P-1][Q  ][R+1]
                        - F[P+1][Q  ][R-1] + F[P-1][Q  ][R-1]);
                    K[a][b][c][6] = 0.25 * (
                          F[P  ][Q+1][R+1] - F[P  ][Q-1][R+1]
                        - F[P  ][Q+1][R-1] + F[P  ][Q-1][R-1]);
                    K[a][b][c][7] = 0.125 * (
                          F[P+1][Q+1][R+1] - F[P-1][Q+1][R+1]
                        - F[P+1][Q-1][R+1] + F[P-1][Q-1][R+1]
                        - F[P+1][Q+1][R-1] + F[P-1][Q+1][R-1]
                        + F[P+1][Q-1][R-1] - F[P-1][Q-1][R-1]);
                }

                // 1D cubic Hermite basis evaluated at the query coords.
                double psi_x[2], phi_x[2], dpsi_x[2], dphi_x[2];
                double psi_y[2], phi_y[2], dpsi_y[2], dphi_y[2];
                double psi_z[2], phi_z[2], dpsi_z[2], dphi_z[2];
                detail::cubic_hermite_basis_1d(qs.u[0], psi_x, phi_x, dpsi_x, dphi_x);
                detail::cubic_hermite_basis_1d(qs.u[1], psi_y, phi_y, dpsi_y, dphi_y);
                detail::cubic_hermite_basis_1d(qs.u[2], psi_z, phi_z, dpsi_z, dphi_z);

                // Tensor-product evaluation. For each corner (a,b,c) and
                // each derivative-order tuple (o_x,o_y,o_z), the basis is a
                // product of one of {psi_a, phi_a} per axis. The bit
                // encoding is: o & 1 -> x deriv, (o>>1) & 1 -> y deriv,
                // (o>>2) & 1 -> z deriv.  Loop is 8*8 = 64 iterations,
                // fully unrollable.
                double f_eval = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
                for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                for (int c = 0; c < 2; ++c) {
                    for (int o = 0; o < 8; ++o) {
                        int const bit_x = (o     ) & 1;
                        int const bit_y = (o >> 1) & 1;
                        int const bit_z = (o >> 2) & 1;
                        double const Bx_val =  bit_x ? phi_x[a] : psi_x[a];
                        double const By_val =  bit_y ? phi_y[b] : psi_y[b];
                        double const Bz_val =  bit_z ? phi_z[c] : psi_z[c];
                        double const Bx_dgs =  bit_x ? dphi_x[a] : dpsi_x[a];
                        double const By_dgs =  bit_y ? dphi_y[b] : dpsi_y[b];
                        double const Bz_dgs =  bit_z ? dphi_z[c] : dpsi_z[c];

                        double const k = K[a][b][c][o];
                        f_eval += k * (Bx_val * By_val * Bz_val);
                        if (grad_flag) {
                            gx += k * (Bx_dgs * By_val * Bz_val);
                            gy += k * (Bx_val * By_dgs * Bz_val);
                            gz += k * (Bx_val * By_val * Bz_dgs);
                        }
                    }
                }

                out(ip, iv) = f_eval;
                if (grad_flag) {
                    // d/dx = (1/dx) d/ds for each axis.
                    out_grad(ip, iv, 0) = gx * inv_dx;
                    out_grad(ip, iv, 1) = gy * inv_dx;
                    out_grad(ip, iv, 2) = gz * inv_dx;
                }
            }
        );
    }

    int _valid_gz;

    // host storage
    std::vector<query_state_t> states;

    // device storage
    int npoints;
    readonly_view_t<intersected_cell_descriptor_t> intersected_cells;
    readonly_view_t<query_state_t>                 query_states;
};

}  // namespace grace

#endif  // GRACE_UTILS_GRID_INTERPOLATOR_HH
