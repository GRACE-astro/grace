#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/amr/amr_ghosts.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/gridloop.hh>
#include <grace/evolution/refluxing.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <cmath>
#include <iostream>

namespace {

// Per-quadrant perturbation, seeded by the quadrant centre's physical
// coordinates so that the SAME global quadrant gets the SAME
// perturbation regardless of which rank owns it (i.e. it is consistent
// across the up-to-4 sides that each shared edge sees).
//
// Without a perturbation, a setup based purely on f(physical_coords)
// gives bit-identical EMFs at every shared edge — first-pass reflux
// would then be a trivial no-op and the idempotence check would not
// exercise the apply kernels at all.
inline double quad_perturbation(size_t q)
{
    DECLARE_GRID_EXTENTS;
    (void)nq;
    auto& cs = grace::coordinate_system::get();
    auto pc = cs.get_physical_coordinates(
        {VEC(nx/2 + ngz, ny/2 + ngz, nz/2 + ngz)},
        static_cast<int64_t>(q),
        {VEC(0.5, 0.5, 0.5)},
        /*use_ghostzones=*/true);
    return std::sin(7.3 * pc[0] + 4.1 * pc[1] + 2.9 * pc[2]) * 1e-3;
}

// Initialise the EMF array with closed-form analytic functions of the
// edge physical coordinates, plus a per-quadrant FP-scale perturbation
// so that adjacent sides at every shared edge disagree.  Gives reflux
// real work to do on the first pass.
void setup_emfs()
{
    DECLARE_GRID_EXTENTS;
    (void)nq;
    using namespace grace;
    using namespace Kokkos;

    auto& cs  = coordinate_system::get();
    auto& emf = variable_list::get().getemfarray();
    Kokkos::fence();

    auto emf_h = create_mirror_view(emf);

    auto fill = [&](int dir,
                    std::array<double, GRACE_NSPACEDIM> lcoord,
                    std::array<bool,   GRACE_NSPACEDIM> stag)
    {
        host_grid_loop<true>(
            [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
                auto p = cs.get_physical_coordinates(
                    {VEC(i, j, k)}, static_cast<int64_t>(q), lcoord,
                    /*use_ghostzones=*/true);
                double v = 0.0;
                if      (dir == 0) v = p[0] * (SQR(p[1]) - 1.333 * SQR(p[2]));
                else if (dir == 1) v = p[1] * (SQR(p[0]) - 4.333 * p[2]);
                else               v = p[2] * (SQR(p[0]) + p[1]);
                emf_h(VEC(i, j, k), dir, q) = v + quad_perturbation(q);
            },
            stag, /*include_ghosts=*/true);
    };
    fill(0, {VEC(0.5, 0.0, 0.0)}, {VEC(false, true,  true )});
    fill(1, {VEC(0.0, 0.5, 0.0)}, {VEC(true,  false, true )});
    fill(2, {VEC(0.0, 0.0, 0.5)}, {VEC(true,  true,  false)});
    deep_copy(emf, emf_h);
    Kokkos::fence();
}

// Run one full reflux EMF pass: post the MPI traffic, wait for it
// inside reflux_correct_emfs, run all the apply kernels.
void apply_reflux_pass()
{
    auto ctx = grace::reflux_fill_emf_buffers();
    Kokkos::fence();
    grace::reflux_correct_emfs(ctx);
    Kokkos::fence();
    parallel::mpi_barrier();
}

} // namespace

// ---------------------------------------------------------------------------
// Reflux is supposed to enforce that EMFs at shared edges/faces agree
// across all sides.  After a first pass that property holds by
// construction, so a second pass MUST be a bit-exact no-op — its output
// is a fixed point of the algorithm.
//
// Any FP-level difference between the post-first-pass and post-second-
// pass EMF arrays means the apply kernels are inconsistent with their
// own input (typically: indexing, dispatch order, or buffer-management
// bug).  The test reports per-cell mismatches with full coordinates so
// the failure pattern (only one staggering, only at rank boundaries,
// only at corner edges, …) is immediately visible.
//
// On unigrid the relevant kernels that DO run are:
//   - reflux_emf_apply_coarse_face       (every same-level face)
//   - reflux_coarse_emf_compute_coarse_edge + apply_coarse_edge
//                                        (every same-level edge)
// (The hanging-face / hanging-edge variants are no-ops because their
// descriptor lists are empty on unigrid.)
// ---------------------------------------------------------------------------
TEST_CASE("Reflux is idempotent on unigrid (double-pass)", "[refluxing]")
{
    using namespace grace;
    Kokkos::fence();
    parallel::mpi_barrier();

    auto& emf = variable_list::get().getemfarray();
    int const rank = parallel::mpi_comm_rank();

    setup_emfs();

    // Snapshot pre-reflux state.  Use create_mirror (NOT
    // create_mirror_view) so we get an independent allocation even on
    // CPU builds where HostSpace == default_space.
    auto emf_pre = Kokkos::create_mirror(emf);
    Kokkos::deep_copy(emf_pre, emf);

    // First reflux pass: makes shared-edge / shared-face values agree
    // across sides.
    apply_reflux_pass();

    auto emf_after_first = Kokkos::create_mirror(emf);
    Kokkos::deep_copy(emf_after_first, emf);

    // Sanity report: how many cells did the first pass actually change?
    // Zero would mean the perturbation never produced a side-asymmetry
    // for reflux to act on, in which case the idempotence check below
    // is trivially satisfied without exercising the apply kernels.
    size_t n_changed_first = 0;
    for (size_t q = 0; q < emf.extent(4); ++q)
    for (size_t d = 0; d < emf.extent(3); ++d)
    for (size_t k = 0; k < emf.extent(2); ++k)
    for (size_t j = 0; j < emf.extent(1); ++j)
    for (size_t i = 0; i < emf.extent(0); ++i)
        if (emf_after_first(i,j,k,d,q) != emf_pre(i,j,k,d,q))
            ++n_changed_first;

    std::cout << "Rank " << rank
              << ": first reflux pass changed " << n_changed_first
              << " EMF cells" << std::endl;
    INFO("First pass changed " << n_changed_first
         << " cells on rank " << rank
         << ".  If 0, the perturbation was ineffective and the "
            "idempotence check below is trivial.");

    // Second reflux pass.  Bit-exact no-op expected.
    apply_reflux_pass();

    auto emf_after_second = Kokkos::create_mirror(emf);
    Kokkos::deep_copy(emf_after_second, emf);

    // Cell-by-cell bit-exact comparison.  CHECK (not REQUIRE) so all
    // violations surface on a single run; cap detailed INFO messages
    // so the log doesn't explode on a large grid.
    size_t n_violations = 0;
    constexpr size_t MAX_REPORTS = 50;
    for (size_t q = 0; q < emf.extent(4); ++q)
    for (size_t d = 0; d < emf.extent(3); ++d)
    for (size_t k = 0; k < emf.extent(2); ++k)
    for (size_t j = 0; j < emf.extent(1); ++j)
    for (size_t i = 0; i < emf.extent(0); ++i) {
        double const a = emf_after_first (i,j,k,d,q);
        double const b = emf_after_second(i,j,k,d,q);
        if (a != b) {
            if (n_violations < MAX_REPORTS) {
                INFO("Idempotence violation rank=" << rank
                     << " q=" << q
                     << " dir=" << d
                     << " ijk=(" << i << "," << j << "," << k << ")"
                     << " first="  << a
                     << " second=" << b
                     << " diff="   << (b - a));
                CHECK(a == b);
            }
            ++n_violations;
        }
    }
    std::cout << "Rank " << rank << ": "
              << n_violations << " idempotence violations (out of "
              << (emf.extent(0) * emf.extent(1) * emf.extent(2)
                * emf.extent(3) * emf.extent(4))
              << " cells)" << std::endl;
    CHECK(n_violations == 0);
}
