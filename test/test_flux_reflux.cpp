// ---------------------------------------------------------------------------
// Conservation test for the face-flux reflux pipeline on AMR.
//
// A finite-volume substep updates each cell by
//   state[cell] -= dt * idx * (F[hi] - F[lo])
// where F is read from the cell-local flux array.  Summed over all cells
// (with cell-volume weight), interior face contributions telescope and
// only outer-boundary face fluxes survive — the discrete divergence
// theorem.  At a coarse-fine interface the telescoping fails: the
// coarse-side cell sees the locally-computed coarse flux while the
// fine-side cells see fine fluxes that average to a different value,
// so mass is created or destroyed at the interface.  reflux_correct_fluxes
// overwrites the coarse-side flux at every fine-coarse face with the
// area-averaged fine flux IN PLACE (in the flux array, before the
// divergence step consumes it), so the subsequent state update is
// consistent on both sides and the global integral vanishes under
// periodic BCs.
//
// Test recipe:
//   1. state = 0 everywhere.
//   2. Fill the flux array with a smooth analytic pattern that is
//      (a) NOT per-quadrant-perturbed and (b) EXACTLY periodic with
//      the test domain — see setup_fluxes for why both constraints
//      matter.  The pattern is non-linear in physical coordinates, so
//      the area-averaged fine flux at any coarse-fine face differs
//      from the coarse-face-center flux by O(h^2).  Same-level
//      quadrant boundaries evaluate v at the same physical point and
//      get bit-identical fluxes; the periodic wrap contribution is
//      exactly zero by construction.  Only the fine-coarse mismatch
//      survives — that's the thing reflux is supposed to heal.
//   3. Apply reflux_correct_fluxes — overwrites coarse-side fluxes at
//      fine-coarse interfaces with the area-average of the four fine
//      fluxes.  Must run BEFORE step 4 (mirrors the EMF reflux pattern).
//   4. Apply a "fake" divergence update to every interior cell using
//      the now-corrected flux array.
//   5. Reduce two diagnostics with MPI_Allreduce:
//        (a) sum(state * cell_volume) — must be ~ 0 under periodic BCs.
//            A sign error in reflux, a missing area weight, or a child-
//            indexing bug produces a net imbalance even if individual
//            cells happen to cancel.
//        (b) max|state| — magnitude baseline so the tolerance for (a)
//            can be scaled relative to the typical update magnitude.
//
// Without reflux the integral would be O(N_interface_cells · dt · F_mismatch),
// far above any plausible FP tolerance — so the test has real
// discriminating power.
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/amr/amr_ghosts.hh>
#include <grace/amr/forest.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/gridloop.hh>
#include <grace/evolution/refluxing.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

// Fill the face-centered flux array with a smooth analytic function
// of physical coordinates only.  TWO design constraints distinguish
// this setup from the EMF/CT tests:
//
// (1) NO per-quadrant constant offset.  reflux_correct_fluxes operates
//     ONLY on fine-coarse interfaces; it has no descriptor for
//     same-level quadrant boundaries (in production those are made
//     consistent by the Riemann solver, not by reflux).  A per-quadrant
//     additive perturbation would create artificial mismatches at
//     same-level boundaries that reflux cannot fix, breaking
//     conservation for reasons unrelated to AMR refluxing.
//
// (2) The function MUST be periodic with the domain.  Under periodic
//     BCs the substep's outer-boundary contribution at the wrap is
//     F(x = xmin, ...) on one side and F(x = xmax, ...) on the other;
//     these must agree for the global integral of state to vanish.
//     A polynomial v(p) like p[0]*(p[1]^2 - p[2]^2) is wildly NOT
//     periodic (sign-flipped at opposite domain edges) and produces an
//     O(1) systematic imbalance at the periodic wrap that has nothing
//     to do with refluxing.  Sinusoidal functions with wavelengths
//     matching the domain extent are exactly periodic at the wrap.
//
// With both constraints honoured: adjacent same-level quadrants
// evaluate v at the same physical point and get bit-identical fluxes
// (no spurious same-level mismatch); the periodic wrap contribution is
// exactly zero (no spurious wrap imbalance); only the natural O(h^2)
// fine-vs-coarse area-averaging error survives, and that's exactly what
// reflux is designed to heal.
void setup_fluxes()
{
    DECLARE_GRID_EXTENTS;
    (void)nq;
    using namespace grace;
    using namespace Kokkos;

    auto& cs     = coordinate_system::get();
    auto& fluxes = variable_list::get().getfluxesarray();
    int const nvars_hrsc = variables::get_n_hrsc();
    Kokkos::fence();

    auto fluxes_h = create_mirror_view(fluxes);

    // Wavelength = 2 in physical coordinates; domain extents in the
    // shared ct_flux_conservation_test.yaml are 2 in every direction
    // (xmin=-1, xmax=1, etc.), so sin(kπp), cos(kπp) with k integer
    // are exactly periodic at the wrap. Use k=1 (one full wave across
    // each axis) — gives strong curvature for the AMR mismatch while
    // staying well-resolved on the coarse grid.
    constexpr double K = M_PI;

    auto fill = [&](int idir,
                    std::array<double, GRACE_NSPACEDIM> lcoord,
                    std::array<bool,   GRACE_NSPACEDIM> stag)
    {
        host_grid_loop<true>(
            [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
                auto p = cs.get_physical_coordinates(
                    {VEC(i, j, k)}, static_cast<int64_t>(q), lcoord,
                    /*use_ghostzones=*/true);
                double const sx = std::sin(K * p[0]);
                double const cx = std::cos(K * p[0]);
                double const sy = std::sin(K * p[1]);
                double const cy = std::cos(K * p[1]);
                double const sz = std::sin(K * p[2]);
                double const cz = std::cos(K * p[2]);
                for (int ivar = 0; ivar < nvars_hrsc; ++ivar) {
                    double const phase = 0.37 * (ivar + 1);
                    double v = 0.0;
                    if      (idir == 0) v = sx * cy * cz;
                    else if (idir == 1) v = cx * sy * cz;
                    else                v = cx * cy * sz;
                    // Constant phase is exactly periodic and contributes
                    // nothing to FD differences — kept only so different
                    // ivars produce visibly distinct state magnitudes in
                    // the diagnostic output.
                    fluxes_h(VEC(i, j, k), ivar, idir, q) = v + 0.1 * phase * (sx + sy + sz);
                }
            },
            stag, /*include_ghosts=*/true);
    };
    fill(0, {VEC(0.0, 0.5, 0.5)}, {VEC(true,  false, false)});
    fill(1, {VEC(0.5, 0.0, 0.5)}, {VEC(false, true,  false)});
    fill(2, {VEC(0.5, 0.5, 0.0)}, {VEC(false, false, true )});
    deep_copy(fluxes, fluxes_h);
    Kokkos::fence();
}

// Canonically-ordered global volume-weighted sum.
//
// Why: a plain `parallel_reduce` + `MPI_Allreduce(SUM)` accumulates in
// rank- and thread-decomposition-dependent order. Under FP non-
// associativity that means the conservation diagnostic flips by O(eps *
// l1) every time we change rank count, which is benign for tolerance
// asserts but ruins the test's use as a *symmetry* / regression check:
// 1-rank vs 2-rank runs should be bit-identical for an exactly
// conservative scheme.
//
// What this does: per-quadrant local sums are computed on host in fixed
// (k,j,i) order; results are slotted into a global qid-indexed buffer
// at the rank's `global_first_quadrant` offset; `MPI_Allreduce(SUM)`
// then performs an exact identity-add per slot (each global qid is
// owned by exactly one rank, so every slot has at most one nonzero
// contributor); the final reduction walks the global buffer in qid
// order on every rank. That makes the result bit-identical regardless
// of MPI partition.
//
// The caller passes a host-side functor `f_cell_q(VEC(i,j,k), q)`
// returning the per-cell contribution (already volume-weighted if
// needed).
template <typename HostCellFn>
double canonical_global_volume_sum(HostCellFn const& f_cell_q)
{
    DECLARE_GRID_EXTENTS;
    auto& fst   = grace::amr::forest::get();
    auto* p4est = fst.get();
    uint64_t const nq_glob  = static_cast<uint64_t>(p4est->global_num_quadrants);
    int const rank          = parallel::mpi_comm_rank();
    uint64_t const q_offset = static_cast<uint64_t>(fst.global_quadrant_offset(rank));

    std::vector<double> partials(nq_glob, 0.0);
    for (size_t q = 0; q < nq; ++q) {
        double s = 0.0;
        for (size_t k = ngz; k < nz + ngz; ++k) {
            for (size_t j = ngz; j < ny + ngz; ++j) {
                for (size_t i = ngz; i < nx + ngz; ++i) {
                    s += f_cell_q(VEC(i, j, k), q);
                }
            }
        }
        partials[q_offset + q] = s;
    }

    std::vector<double> global_partials(nq_glob, 0.0);
    parallel::mpi_allreduce(partials.data(),
                            global_partials.data(),
                            static_cast<int>(nq_glob),
                            sc_MPI_SUM);

    double total = 0.0;
    for (uint64_t qg = 0; qg < nq_glob; ++qg) total += global_partials[qg];
    return total;
}

} // namespace

TEST_CASE("Face-flux reflux preserves mass conservation on AMR (periodic BCs)",
          "[flux-conservation]")
{
    using namespace grace;
    Kokkos::fence();
    parallel::mpi_barrier();

    auto& ghost_layer = amr_ghosts::get();
    auto desc = ghost_layer.get_reflux_face_descriptors();
    int const rank = parallel::mpi_comm_rank();

    int has_local_interfaces = desc.coarse_qid.extent(0) > 0 ? 1 : 0;
    int has_any_interfaces = 0;
    parallel::mpi_allreduce(&has_local_interfaces, &has_any_interfaces, 1, sc_MPI_MAX);

    if (!has_any_interfaces) {
        WARN("No fine-coarse interfaces on any rank; flux conservation test is "
             "vacuous for this parfile. Use an FMR-enabled config.");
        SUCCEED("vacuous");
        return;
    }

    DECLARE_GRID_EXTENTS;
    (void)nq;

    auto& vlist  = variable_list::get();
    auto& state  = vlist.getstate();
    auto& fluxes = vlist.getfluxesarray();
    auto& idx    = vlist.getinvspacings();
    auto& dx_arr = vlist.getspacings();

    int const nvars_hrsc = variables::get_n_hrsc();

    // 1. state = 0 everywhere.
    Kokkos::deep_copy(state, 0.0);
    Kokkos::fence();

    // 2. Fill fluxes.
    setup_fluxes();

    // 3. Apply reflux on the flux array BEFORE the divergence update.
    //    `reflux_correct_fluxes` now overwrites the coarse-side flux at
    //    every fine-coarse face with the area-averaged fine flux, so the
    //    subsequent divergence kernel naturally uses the conservative
    //    value.  This is mathematically equivalent to the old post-update
    //    state patch but is race-free (each (qid,idir,side,i,j,ivar) slot
    //    in the flux array is written by at most one descriptor) and
    //    bit-invariant under MPI repartition.
    {
        auto ctx = reflux_fill_flux_buffers();
        Kokkos::fence();
        reflux_correct_fluxes(ctx);
        Kokkos::fence();
        parallel::mpi_barrier();
    }

    // 4. Fake divergence update on every interior cell, using the now-
    //    corrected flux array:
    //      state[v] -= dt * idx[d] * (F[d, hi] - F[d, lo])  for d = 0,1,2
    //    Only hydro/HRSC variables (indices 0..nvars_hrsc-1) are touched,
    //    matching what the production substep would do.
    constexpr double dt = 1.0, dtfact = 1.0;
    {
        Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>>
            interior_policy({VEC(ngz, ngz, ngz), 0},
                            {VEC(nx+ngz, ny+ngz, nz+ngz), nq});
        Kokkos::parallel_for(
            "fake_div_F_update", interior_policy,
            KOKKOS_LAMBDA(VEC(int const& i, int const& j, int const& k),
                          int const& q) {
                for (int v = 0; v < nvars_hrsc; ++v) {
                    double const dFx = (fluxes(VEC(i+1, j,   k  ), v, 0, q)
                                      - fluxes(VEC(i,   j,   k  ), v, 0, q)) * idx(0, q);
                    double const dFy = (fluxes(VEC(i,   j+1, k  ), v, 1, q)
                                      - fluxes(VEC(i,   j,   k  ), v, 1, q)) * idx(1, q);
                    double const dFz = (fluxes(VEC(i,   j,   k+1), v, 2, q)
                                      - fluxes(VEC(i,   j,   k  ), v, 2, q)) * idx(2, q);
                    state(VEC(i, j, k), v, q) -= dt * dtfact * (dFx + dFy + dFz);
                }
            });
        Kokkos::fence();
    }

    // 5. Per-variable conservation reductions.
    //    sum(state * dx*dy*dz) — should be ~0 under periodic BCs.
    //    max|state|             — magnitude baseline.
    //    l1(state * dx*dy*dz)   — for relative-tolerance scaling.
    //
    //  Sum/l1 use the canonical-order global sum so the result is bit-
    //  identical across MPI partitions; max uses the standard device
    //  parallel_reduce + MPI_MAX (max is order-independent under FP).
    Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>>
        all_policy({VEC(ngz, ngz, ngz), 0},
                   {VEC(nx+ngz, ny+ngz, nz+ngz), nq});

    auto state_h  = Kokkos::create_mirror_view(state);
    auto dx_arr_h = Kokkos::create_mirror_view(dx_arr);
    Kokkos::deep_copy(state_h, state);
    Kokkos::deep_copy(dx_arr_h, dx_arr);

    bool any_var_failed = false;
    for (int v = 0; v < nvars_hrsc; ++v) {
        double local_max_abs  = 0.0;
        Kokkos::parallel_reduce(
            "flux_conservation_max_v", all_policy,
            KOKKOS_LAMBDA(VEC(int const& i, int const& j, int const& k),
                          int const& q,
                          double& maxabs) {
                double const as = Kokkos::fabs(state(VEC(i, j, k), v, q));
                if (as > maxabs) maxabs = as;
            },
            Kokkos::Max<double>(local_max_abs));
        Kokkos::fence();

        double const integral = canonical_global_volume_sum(
            [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
                double const vol = dx_arr_h(0, q) * dx_arr_h(1, q) * dx_arr_h(2, q);
                return state_h(VEC(i, j, k), v, q) * vol;
            });
        double const l1_norm = canonical_global_volume_sum(
            [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
                double const vol = dx_arr_h(0, q) * dx_arr_h(1, q) * dx_arr_h(2, q);
                return std::fabs(state_h(VEC(i, j, k), v, q)) * vol;
            });
        double max_abs = 0.0;
        parallel::mpi_allreduce(&local_max_abs, &max_abs, 1, sc_MPI_MAX);

        // Relative tolerance: a properly-conservative AMR scheme should
        // give |integral| at the FP roundoff floor of the global sum-of-
        // many-numbers, which the canonical-ordered sum bounds as
        //   |integral|_fp ≲ eps · l1_norm  ≈  2.2e-16 · l1_norm.
        // The 1e-14·l1_norm threshold is ~50× margin over strict eps to
        // absorb cross-platform FMA / op-order variability without giving
        // up bug-catching power.  A real conservation bug (sign error in
        // reflux, missing area weight, child-flux summation ordering, …)
        // produces |integral| ~ N_interface · |Δ_flux| · dt — many orders
        // above this bound.
        double const eps_rel  = 1e-14;
        double const tol_int  = eps_rel * (l1_norm + 1e-30);

        // Vacuity guard: a flux setup that produces uniform/zero updates
        // would trivially pass the integral check.  Require max|Δstate|
        // to be safely above the per-cell FP roundoff so we know the
        // fake-divergence kernel actually did meaningful work.
        double const max_floor = 1e-12;

        if (rank == 0) {
            std::cout << "Flux conservation v=" << v
                      << " : integral = " << integral
                      << " ; max|Δstate| = " << max_abs
                      << " ; l1 = " << l1_norm
                      << " ; tol_int = " << tol_int << std::endl;
        }
        INFO("v=" << v
             << " integral=" << integral
             << " max|Δstate|=" << max_abs
             << " l1=" << l1_norm
             << " tol_int=" << tol_int);

        REQUIRE(max_abs > max_floor);  // catches a vacuously-empty setup

        if (Kokkos::fabs(integral) >= tol_int) {
            any_var_failed = true;
            CHECK(Kokkos::fabs(integral) < tol_int);
        }
    }

    CHECK_FALSE(any_var_failed);
}
