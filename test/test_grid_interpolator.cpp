/**
 * @file test_grid_interpolator.cpp
 * @brief Unit tests for grace::grid_interpolator_t<poly_kind, Degree>.
 *
 *        The valid-ghost-zone test poisons the cells outside
 *        the [ngz - _valid_gz, nx + ngz + _valid_gz - 1] budget with NaN
 *        and fills the rest with a known polynomial. A correctly-biased
 *        stencil only ever reads the polynomial-filled region, so the
 *        output is finite AND equal to the analytic value. A bug that
 *        ignores _valid_gz, or caps |bias| too aggressively (as the old
 *        Lagrange code did), turns the output into NaN at the boundary
 *        cells.
 *
 *        Polynomial-exactness targets:
 *          - Lagrange order O reproduces a per-axis-degree-O polynomial
 *            exactly anywhere in the grid, regardless of bias.
 *          - Hermite cubic reproduces a per-axis-degree-2 polynomial
 *            exactly: central-FD is 2nd-order, hence exact for quadratics
 *            and their derivatives, so the tensor-product cubic Hermite
 *            constructed from FD'd knot data IS the input quadratic.
 *
 * @date 2026-04-28
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace_config.h>

#include <Kokkos_Core.hpp>

#include <grace/amr/grace_amr.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <grace/utils/grid_interpolator.hh>

#include <array>
#include <cmath>
#include <vector>


namespace {

//---------------------------------------------------------------------
// Polynomial test functions.
//---------------------------------------------------------------------

/// Per-axis-degree-4 polynomial -- exactly reproducible by any
/// tensor-product Lagrange of order >= 4.
double poly_deg4(double x, double y, double z) {
    return 1.7
         + 0.31 * x   - 0.22 * y   + 0.13 * z
         + 0.05 * x*x - 0.04 * y*y + 0.03 * z*z
         + 0.011 * x*y - 0.013 * y*z + 0.017 * x*z
         + 0.0021 * x*x*x - 0.0019 * y*y*y + 0.0011 * z*z*z
         + 7e-5 * x*x*x*x + 6e-5 * y*y*y*y + 5e-5 * z*z*z*z
         + 9e-4 * x*x*y - 7e-4 * y*y*z + 8e-4 * z*z*x;
}

/// Per-axis-degree-2 polynomial -- exactly reproducible by Hermite cubic
/// because central-FD of a quadratic is exact.
double poly_deg2(double x, double y, double z) {
    return 2.1
         - 0.41 * x   + 0.33 * y   - 0.27 * z
         + 0.052 * x*x - 0.044 * y*y + 0.031 * z*z
         + 0.013 * x*y - 0.017 * y*z + 0.019 * x*z;
}

/// Analytic gradient of poly_deg2.
std::array<double,3> grad_poly_deg2(double x, double y, double z) {
    return {
        -0.41 + 2.0 * 0.052 * x + 0.013 * y + 0.019 * z,
         0.33 - 2.0 * 0.044 * y + 0.013 * x - 0.017 * z,
        -0.27 + 2.0 * 0.031 * z - 0.017 * y + 0.019 * x
    };
}

//---------------------------------------------------------------------
// Fill the state View with `f(x,y,z)` inside the valid ghost-zone
// budget and stomp NaN everywhere outside it. Each axis is treated
// independently: a cell counts as valid iff all three of its global
// indices fall in [ngz - valid_gz, n_axis + ngz + valid_gz - 1].
//---------------------------------------------------------------------
template <typename Func>
void fill_state_with_polynomial_and_nan_fence(
    int var_slot, int valid_gz, Func const& f
) {
    using namespace grace;
    DECLARE_GRID_EXTENTS;
    auto& state = variable_list::get().getstate();
    auto state_h = Kokkos::create_mirror_view(state);

    long const lo_valid_x = static_cast<long>(ngz) - valid_gz;
    long const hi_valid_x = static_cast<long>(nx + ngz) + valid_gz;
    long const lo_valid_y = static_cast<long>(ngz) - valid_gz;
    long const hi_valid_y = static_cast<long>(ny + ngz) + valid_gz;
    long const lo_valid_z = static_cast<long>(ngz) - valid_gz;
    long const hi_valid_z = static_cast<long>(nz + ngz) + valid_gz;

    double const nan = std::nan("");
    auto& coord_system = coordinate_system::get();

    for (size_t q = 0; q < static_cast<size_t>(nq); ++q) {
        for (size_t i = 0; i < nx + 2*ngz; ++i)
        for (size_t j = 0; j < ny + 2*ngz; ++j)
        for (size_t k = 0; k < nz + 2*ngz; ++k) {
            bool const valid =
                static_cast<long>(i) >= lo_valid_x && static_cast<long>(i) < hi_valid_x &&
                static_cast<long>(j) >= lo_valid_y && static_cast<long>(j) < hi_valid_y &&
                static_cast<long>(k) >= lo_valid_z && static_cast<long>(k) < hi_valid_z;
            if (valid) {
                auto pc = coord_system.get_physical_coordinates(
                    {i, j, k}, q, {0.5, 0.5, 0.5}, /*use_ghostzones=*/true
                );
                state_h(i, j, k, var_slot, q) = f(pc[0], pc[1], pc[2]);
            } else {
                state_h(i, j, k, var_slot, q) = nan;
            }
        }
    }
    Kokkos::deep_copy(state, state_h);
}

//---------------------------------------------------------------------
// Build a list of query points spanning one row along the x axis of a
// single quadrant -- cells {0, ..., nx-1}, drives the bias path at both
// edges when valid_gz < stencil_half.
//
// Queries are intentionally placed AWAY from the cell centre (and away
// from cell-corner / cell-face values that coincide with stencil knots)
// so the interpolator's basis-function evaluation is non-trivial:
//
//   - Lagrange: at a knot, all but one barycentric weight collapse to
//     zero, so the test would only verify "data at the knot equals the
//     polynomial at the knot" -- vacuously true by construction.
//   - Hermite cubic: a query at u=0 or u=1 picks only one tensor-product
//     knot value (psi_a(0)=1, phi_a(0)=0 etc.); the cubic basis is never
//     exercised.
//
// The offset (sx, sy, sz) is a fraction of dx in [-0.5, +0.5]. The same
// offset is applied at every cell. A second test pass with a sign-flipped
// offset covers the other Hermite-half dispatch.
//---------------------------------------------------------------------
struct query_set_t {
    std::vector<grace::point_host_t> points;
    std::vector<size_t> ipoints;
    std::vector<grace::intersected_cell_descriptor_t> icells;
};

query_set_t build_x_row_queries(size_t test_q,
                                std::array<double,3> const& cell_offset) {
    using namespace grace;
    DECLARE_GRID_EXTENTS;
    auto& coord_system = coordinate_system::get();

    query_set_t qs;
    size_t const j0 = ny / 2;
    size_t const k0 = nz / 2;

    double const dx = coord_system.get_spacing(test_q);

    for (size_t i = 0; i < nx; ++i) {
        // Cell centre, then shift by the requested fraction-of-dx offset.
        auto pc = coord_system.get_physical_coordinates(
            {i, j0, k0}, test_q, {0.5, 0.5, 0.5}, /*use_ghostzones=*/false
        );
        pc[0] += cell_offset[0] * dx;
        pc[1] += cell_offset[1] * dx;
        pc[2] += cell_offset[2] * dx;

        size_t const pidx = qs.points.size();
        qs.points.push_back({pidx, {pc[0], pc[1], pc[2]}});
        qs.ipoints.push_back(pidx);
        // Cell descriptor is unchanged: shifting by < dx/2 keeps the
        // query in the same cell that the search would have placed it.
        qs.icells.push_back({i, j0, k0, test_q});
    }
    return qs;
}

// Off-centre offsets used by the tests below. Picked irrationally so the
// queries don't accidentally land on any special point (cell centre,
// face, or corner) of the stencil. The two variants flip the sign on
// each axis so Hermite's left-half / right-half dispatch is both
// exercised.
constexpr std::array<double,3> offset_pos = { 0.27, -0.19,  0.31};
constexpr std::array<double,3> offset_neg = {-0.27,  0.19, -0.31};

}  // namespace


//=====================================================================
// Lagrange order 4 -- NaN-fenced ghost budget.
//=====================================================================

TEST_CASE("grid_interpolator_t<lagrange,4>: NaN-fenced ghost budget, "
          "polynomial exactness", "[grid_interpolator][lagrange]")
{
    using namespace grace;
    DECLARE_GRID_EXTENTS;
    REQUIRE(nq > 0);

    // Lagrange<4> stencil is 5-wide. With basic_config (nx=16, ngz=2),
    // valid_gz can range over [0, ngz]:
    //   valid_gz=0  -> cell.i=0 needs bias=+2, cell.i=1 needs bias=+1.
    //                  Old code's |bias|<=1 cap would have failed here.
    //   valid_gz=1  -> cell.i=0 needs bias=+1.
    //   valid_gz=ngz=2 -> bias=0 for every cell.
    std::vector<int> valid_gz_values;
    for (int v = 1; v <= static_cast<int>(ngz); ++v) valid_gz_values.push_back(v);

    for (auto const& offset : {offset_pos, offset_neg}) {
        INFO("Lagrange<4> with cell offset ("
             << offset[0] << ", " << offset[1] << ", " << offset[2] << ")");

        auto queries = build_x_row_queries(/*test_q=*/0, offset);
        REQUIRE(queries.points.size() == static_cast<size_t>(nx));

        for (int valid_gz : valid_gz_values) {
            INFO("Lagrange<4> NaN-fenced with _valid_gz = " << valid_gz);

            fill_state_with_polynomial_and_nan_fence(0, valid_gz, poly_deg4);

            grid_interpolator_t<poly_kind::lagrange, 4> interp(valid_gz);
            interp.compute_weights(queries.points, queries.ipoints, queries.icells);

            Kokkos::View<double**, default_space> out("out_lagr", 0, 0);
            auto& state = variable_list::get().getstate();
            interp.interpolate(state, std::vector<int>{0}, out);

            auto out_h = Kokkos::create_mirror_view(out);
            Kokkos::deep_copy(out_h, out);

            for (size_t i = 0; i < queries.ipoints.size(); ++i) {
                auto const& pc = queries.points[queries.ipoints[i]].second;
                double const expected = poly_deg4(pc[0], pc[1], pc[2]);
                INFO("cell.i = " << queries.icells[i].i
                     << " coords = (" << pc[0] << ", " << pc[1] << ", " << pc[2] << ")"
                     << " valid_gz = " << valid_gz);
                // No NaN leakage from the stomped ghosts.
                REQUIRE(std::isfinite(out_h(i, 0)));
                // Polynomial-exact result.
                REQUIRE_THAT(out_h(i, 0),
                             Catch::Matchers::WithinAbs(expected, 1e-9));
            }
        }
    }
}


//=====================================================================
// Hermite cubic -- NaN-fenced ghost budget, value + analytic gradient.
//=====================================================================

TEST_CASE("grid_interpolator_t<hermite,3>: NaN-fenced ghost budget, "
          "value and analytic gradient",
          "[grid_interpolator][hermite]")
{
    using namespace grace;
    DECLARE_GRID_EXTENTS;
    REQUIRE(nq > 0);

    // Hermite<3> stencil is 4-wide with the half-aware shift. Stencil
    // half is effectively 2 on each side at half=1 (or the mirror at
    // half=0), so the bias path activates similarly to Lagrange<4>.
    // Two passes (offset_pos / offset_neg) make sure both half-dispatch
    // branches are exercised: a positive x-offset puts the query in the
    // right half (half=1, u in [0.5, 1.0)), a negative x-offset puts it
    // in the left half (half=0, u in (0.0, 0.5]).
    std::vector<int> valid_gz_values;
    for (int v = 1; v <= static_cast<int>(ngz); ++v) valid_gz_values.push_back(v);

    for (auto const& offset : {offset_pos, offset_neg}) {
        INFO("Hermite<3> with cell offset ("
             << offset[0] << ", " << offset[1] << ", " << offset[2] << ")");

        auto queries = build_x_row_queries(/*test_q=*/0, offset);
        REQUIRE(queries.points.size() == static_cast<size_t>(nx));

        for (int valid_gz : valid_gz_values) {
            INFO("Hermite<3> NaN-fenced with _valid_gz = " << valid_gz);

            fill_state_with_polynomial_and_nan_fence(0, valid_gz, poly_deg2);

            grid_interpolator_t<poly_kind::hermite, 3> interp(valid_gz);
            interp.compute_weights(queries.points, queries.ipoints, queries.icells);

            Kokkos::View<double**,  default_space> out     ("out_hrm",      0, 0);
            Kokkos::View<double***, default_space> out_grad("out_hrm_grad", 0, 0, 0);
            auto& state = variable_list::get().getstate();
            interp.interpolate_with_grad(state, std::vector<int>{0}, out, out_grad, true);

            auto out_h    = Kokkos::create_mirror_view(out);
            auto out_g_h  = Kokkos::create_mirror_view(out_grad);
            Kokkos::deep_copy(out_h,   out);
            Kokkos::deep_copy(out_g_h, out_grad);

            for (size_t i = 0; i < queries.ipoints.size(); ++i) {
                auto const& pc = queries.points[queries.ipoints[i]].second;
                double const expected = poly_deg2(pc[0], pc[1], pc[2]);
                auto const expected_grad = grad_poly_deg2(pc[0], pc[1], pc[2]);

                INFO("cell.i = " << queries.icells[i].i
                     << " coords = (" << pc[0] << ", " << pc[1] << ", " << pc[2] << ")"
                     << " valid_gz = " << valid_gz);

                REQUIRE(std::isfinite(out_h(i, 0)));
                REQUIRE_THAT(out_h(i, 0),
                             Catch::Matchers::WithinAbs(expected, 1e-9));

                for (int d = 0; d < 3; ++d) {
                    INFO("axis = " << d);
                    REQUIRE(std::isfinite(out_g_h(i, 0, d)));
                    REQUIRE_THAT(out_g_h(i, 0, d),
                                 Catch::Matchers::WithinAbs(expected_grad[d], 1e-9));
                }
            }
        }
    }
}
