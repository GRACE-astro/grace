// ---------------------------------------------------------------------------
// Magnetic-flux conservation test for CT + EMF refluxing on AMR.
//
// The CT operator B_face <- B_face - dt * curl(E_edge) is structurally
// divergence-free at the discrete level: each EMF edge is shared by
// exactly four B-faces of two cells with opposite orientations, so
// div(curl E) telescopes to zero on a per-cell basis. This holds at
// finite precision so long as the EMF value at every shared edge is
// THE SAME on every cell that touches that edge. At a coarse-fine
// interface this requires reflux to replace the inconsistent
// coarse-side EMF with the area-averaged fine-side EMFs — without it,
// curl(E)'s discrete divergence picks up a per-cell bias proportional
// to the EMF mismatch at the interface, and div B grows.
//
// Test recipe:
//   1. Zero out face-staggered B (trivially div-free).
//   2. Set up EMFs with a smooth analytic pattern plus a per-quadrant
//      FP-scale perturbation, so coarse and fine see DIFFERENT EMFs at
//      every coarse-fine edge (giving reflux real work).
//   3. Apply EMF reflux (corrects coarse-fine EMFs).
//   4. Apply the production CT update (B <- B - dt * curl(E)).
//   5. Compute div B per cell over the interior of every quadrant.
//   6. Reduce two diagnostics with MPI_Allreduce:
//        (a) sum(div B * cell_volume)   — proxy for B-flux conservation.
//                                          A bias from a mis-weighted AMR
//                                          interface shows up as a net
//                                          imbalance even if individual
//                                          cells happen to cancel.
//        (b) max |div B|                 — per-cell preservation. Catches
//                                          local CT/reflux index/sign bugs
//                                          that integrate to ~0 by accident.
//
// To make (a) testable against a known value (zero) without having to
// compute the physical-boundary contribution, the parfile uses periodic
// BCs in all three directions. With no global outer boundary the
// surface flux integral vanishes identically, and the only remaining
// contribution to the sum is the (hopefully bit-zero) per-cell residual.
//
// On a grid with no fine-coarse interfaces the test is vacuous and
// self-skips.
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/amr/amr_ghosts.hh>
#include <grace/amr/forest.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/utils/gridloop.hh>
#include <grace/evolution/evolve.hh>
#include <grace/evolution/refluxing.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

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

// Fill the edge-staggered EMF array with an exactly-periodic sinusoidal
// pattern + per-quadrant FP-scale perturbation.
//
// Strictly speaking, divB = div(curl E) is structurally zero per cell
// regardless of whether E is periodic (each cell's divB is computed
// from face-B values local to that cell's quadrant, and each face-B
// was built by a curl(E) restricted to that same quadrant's EMFs).
// So this test would pass with a non-periodic E too.  We use a
// periodic pattern anyway, for two reasons:
//   (1) Uniformity with the face-flux conservation test, which DOES
//       require strict periodicity at the wrap.  Same parfile + same
//       analytic shape across both tests means a discrepancy between
//       them is unambiguously a test-side issue, not a grid/setup
//       accident.
//   (2) Same-level EMF reflux DOES act across periodic wrap edges
//       (via the periodic-aware descriptor build).  Using a periodic
//       E removes any artificial mismatch reflux would otherwise have
//       to "fix" at the wrap, isolating the test to AMR-interface
//       behaviour alone.
//
// The per-quadrant perturbation stays — it's constant within a
// quadrant, so FD differences in the curl cancel it cell-by-cell, but
// it creates an O(1e-3) jump at every QUADRANT boundary (same-level
// and fine-coarse) for EMF reflux to chew on.
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

    // Wavelength = 2 in physical coordinates; the shared parfile has
    // domain extents = 2 in every direction, so K = pi gives exactly
    // one full wave per axis, periodic at the wrap to ulp.
    constexpr double K = M_PI;

    auto fill = [&](int dir,
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
                double v = 0.0;
                if      (dir == 0) v = sx * cy * cz;
                else if (dir == 1) v = cx * sy * cz;
                else               v = cx * cy * sz;
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

void apply_emf_reflux_pass()
{
    auto ctx = grace::reflux_fill_emf_buffers();
    Kokkos::fence();
    grace::reflux_correct_emfs(ctx);
    Kokkos::fence();
    parallel::mpi_barrier();
}

// Canonically-ordered global volume-weighted sum.  See test_flux_reflux.cpp
// for the rationale; in short: per-quadrant local sums are slotted into a
// global qid-indexed buffer at the rank's `global_first_quadrant` offset,
// MPI_Allreduce(SUM) becomes an exact identity-add (each global qid is
// owned by exactly one rank), then the final reduction walks the global
// buffer in qid order on every rank.  Bit-identical across MPI partitions.
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

TEST_CASE("CT + EMF reflux preserve magnetic-flux conservation",
          "[ct-flux-conservation]")
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
        WARN("No fine-coarse interfaces on any rank; CT-flux conservation test "
             "is vacuous for this parfile. Use an FMR-enabled config.");
        SUCCEED("vacuous");
        return;
    }

    DECLARE_GRID_EXTENTS;
    (void)nq;

    auto& vlist  = variable_list::get();
    auto& state  = vlist.getstate();
    auto& stag   = vlist.getstaggeredstate();
    auto& dx_arr = vlist.getspacings();

    // 1. Zero face-B slots.
    Kokkos::deep_copy(Kokkos::subview(stag.face_staggered_fields_x,
        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
        static_cast<size_t>(BSX_), Kokkos::ALL()), 0.0);
    Kokkos::deep_copy(Kokkos::subview(stag.face_staggered_fields_y,
        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
        static_cast<size_t>(BSY_), Kokkos::ALL()), 0.0);
    Kokkos::deep_copy(Kokkos::subview(stag.face_staggered_fields_z,
        Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
        static_cast<size_t>(BSZ_), Kokkos::ALL()), 0.0);
    Kokkos::fence();

    // 2. Fill EMFs with smooth + perturbed pattern.
    setup_emfs();

    // 3. EMF reflux — coarse-fine EMFs agree at the interface after this.
    apply_emf_reflux_pass();

    // 4. Production CT update. update_CT only writes face-B; state,
    //    old_state, and old_stag are unused in its body so we pass the
    //    live arrays as aliases for the "old" slots.
    constexpr double t = 0.0, dt = 1.0, dtfact = 1.0;
    update_CT(t, dt, dtfact, state, state, stag, stag);
    Kokkos::fence();
    parallel::mpi_barrier();

    // 5+6. div B reductions over the interior of every quadrant.
    //  sum/l1 use the canonical-order global sum so the result is bit-
    //  identical across MPI partitions; max uses the standard device
    //  parallel_reduce + MPI_MAX (max is order-independent under FP).
    auto Bx = stag.face_staggered_fields_x;
    auto By = stag.face_staggered_fields_y;
    auto Bz = stag.face_staggered_fields_z;

    auto Bx_h     = Kokkos::create_mirror_view(Bx);
    auto By_h     = Kokkos::create_mirror_view(By);
    auto Bz_h     = Kokkos::create_mirror_view(Bz);
    auto dx_arr_h = Kokkos::create_mirror_view(dx_arr);
    Kokkos::deep_copy(Bx_h, Bx);
    Kokkos::deep_copy(By_h, By);
    Kokkos::deep_copy(Bz_h, Bz);
    Kokkos::deep_copy(dx_arr_h, dx_arr);

    auto divB_at = [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
        double const dx_phys = dx_arr_h(0, q);
        double const dy_phys = dx_arr_h(1, q);
        double const dz_phys = dx_arr_h(2, q);
        double const dBx = (Bx_h(VEC(i+1, j,   k  ), BSX_, q) - Bx_h(VEC(i, j, k), BSX_, q)) / dx_phys;
        double const dBy = (By_h(VEC(i,   j+1, k  ), BSY_, q) - By_h(VEC(i, j, k), BSY_, q)) / dy_phys;
        double const dBz = (Bz_h(VEC(i,   j,   k+1), BSZ_, q) - Bz_h(VEC(i, j, k), BSZ_, q)) / dz_phys;
        return dBx + dBy + dBz;
    };

    double const integral = canonical_global_volume_sum(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            double const vol = dx_arr_h(0, q) * dx_arr_h(1, q) * dx_arr_h(2, q);
            return divB_at(VEC(i, j, k), q) * vol;
        });
    double const l1_norm = canonical_global_volume_sum(
        [&] (VEC(size_t i, size_t j, size_t k), size_t q) {
            double const vol = dx_arr_h(0, q) * dx_arr_h(1, q) * dx_arr_h(2, q);
            return std::fabs(divB_at(VEC(i, j, k), q)) * vol;
        });

    // max|divB| via device reduction + MPI_MAX (FP-order-independent).
    double local_max_abs = 0.0;
    Kokkos::MDRangePolicy<Kokkos::Rank<GRACE_NSPACEDIM+1>>
        interior_policy({VEC(ngz, ngz, ngz), 0},
                        {VEC(nx+ngz, ny+ngz, nz+ngz), nq});
    Kokkos::parallel_reduce(
        "ct_flux_conservation_maxdivB", interior_policy,
        KOKKOS_LAMBDA(VEC(int const& i, int const& j, int const& k),
                      int const& q,
                      double& maxabs)
        {
            double const dx_phys = dx_arr(0, q);
            double const dy_phys = dx_arr(1, q);
            double const dz_phys = dx_arr(2, q);
            double const dBx = (Bx(VEC(i+1, j,   k  ), BSX_, q) - Bx(VEC(i, j, k), BSX_, q)) / dx_phys;
            double const dBy = (By(VEC(i,   j+1, k  ), BSY_, q) - By(VEC(i, j, k), BSY_, q)) / dy_phys;
            double const dBz = (Bz(VEC(i,   j,   k+1), BSZ_, q) - Bz(VEC(i, j, k), BSZ_, q)) / dz_phys;
            double const ad = Kokkos::fabs(dBx + dBy + dBz);
            if (ad > maxabs) maxabs = ad;
        },
        Kokkos::Max<double>(local_max_abs));
    Kokkos::fence();

    double max_abs = 0.0;
    parallel::mpi_allreduce(&local_max_abs, &max_abs, 1, sc_MPI_MAX);

    if (rank == 0) {
        std::cout << "CT-flux conservation:"
                  << " integral(divB*dV) = " << integral
                  << " ; max|divB| = " << max_abs
                  << " ; l1(divB*dV) = " << l1_norm << std::endl;
    }
    INFO("integral(divB*dV) = " << integral
         << " ; max|divB| = " << max_abs
         << " ; l1(divB*dV) = " << l1_norm);

    // Per-cell preservation: the CT curl operator is divergence-free
    // by construction, so max|divB| should sit at the FP floor.  A bug
    // in CT indexing/signs or in the EMF reflux apply kernel would
    // produce a non-ulp value at the AMR interface specifically.
    //
    // FP-floor model: divB = Σ_d (B+ − B-)/dx_d.  Each B_face has roundoff
    //   ~ eps · |B|; the (1/dx) amplification and 3-axis accumulation give
    //   |divB|_fp ≈ 3 · eps · |B| / dx_min.  With |B|_max ≈ 1 from the
    //   sin·cos·cos pattern + the 1e-3 quadrant bias, and dx_min ≈ 0.05
    //   on a typical FMR fine level, this is ≈ 1.3e-14.  The 1e-13 bound
    //   below is one order of margin over that estimate to absorb
    //   dx-dependent variation across parfiles.
    constexpr double eps_max = 1e-13;
    CHECK(max_abs < eps_max);

    // Global B-flux conservation: with periodic BCs the surface flux
    // integral vanishes identically, so the signed sum over the local
    // domain (MPI-summed across all ranks) should also be ~0.  A
    // mis-weighted AMR interface produces a NET signed imbalance that
    // doesn't cancel; this shows up here even if it happens to escape
    // the max|divB| check via per-cell sign accidents.
    //
    // FP-floor model: divergence theorem makes the volume integral cancel
    // exactly in real arithmetic.  FP residual is dominated by non-
    // cancelling contributions at fine-coarse interfaces: per F-C cell
    // ~ eps · |B_face| · dy·dz, summed over the interface plane (N² cells)
    // gives ~ eps · |B| · L²  — independent of resolution.  For |B|≈1,
    // L≈2 (this parfile's domain extent), bound is ≈ 9e-16.  Canonical
    // sum accumulation adds ≲ eps · l1_norm ≈ 2e-28 (negligible).  The
    // 1e-13 bound is two orders of margin to absorb FMA variability and
    // any extra contribution from quadrants > 1 F-C interface deep.
    constexpr double eps_integ = 1e-13;
    CHECK(Kokkos::fabs(integral) < eps_integ);
}
