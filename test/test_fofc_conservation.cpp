// ---------------------------------------------------------------------------
// Conservation test for the FOFC + reflux composition on AMR.
//
// Recipe:
//   1. `grace::initialize` (driven by configs/fofc_conservation_test.yaml)
//      loads a minkowski_vacuum ID — constant ρ, p, v_x — on an FMR grid
//      with periodic BCs.  State + primitives are valid for the production
//      `apply_fofc_correction` path.
//   2. Capture the initial canonical-ordered integral of every HRSC
//      conservative variable.
//   3. Run `compute_fluxes` + `compute_emfs` (production HO paths).
//   4. Manually inject FOFC face-slot tags for a small set of cells.
//      Preferred targets, derived from the reflux face descriptors:
//        - one COARSE cell adjacent to each fine-coarse interface on this
//          rank, and
//        - one FINE cell on the opposite side of each F-C interface.
//      If no F-C interfaces exist locally (unigrid or no descriptor),
//      fall back to a fixed pattern of interior cells so the test path
//      is exercised.
//   5. `apply_fofc_correction` (production) — overwrites the flagged face
//      fluxes with donor + LLF.
//   6. Standard `reflux_correct_fluxes` + `add_fluxes_and_source_terms`
//      (production).
//   7. Recompute integrals; assert
//        |∫(state_final − state_initial) · dV|_v  <  ε_rel · l1_norm[v]
//      for every HRSC variable v.
//
// What this test catches:
//   - Mis-indexed FOFC flux writes (would break per-face flux consistency
//     and produce a non-cancelling drift in the conservation integral).
//   - Composition bugs between FOFC and `reflux_correct_fluxes` at a
//     fine-coarse interface — the case where one cell's "corrected face"
//     IS the same face reflux targets.  This is the unique value of the
//     F-C target-cell selection above.
//   - MPI non-determinism: integrals computed by `canonical_global_volume_sum`
//     are bit-invariant across partitions, so the same parfile run at np=1
//     and np>1 must give identical printed values.  (Asserted via the
//     observed values in `[[conservation-fp-floor-2026-05]]`; this test
//     adds the FOFC-active case to that catalogue.)
//
// Notes:
//   - With constant primitives the production fluxes are uniform per
//     direction, so divergence is zero in exact arithmetic and the
//     conservation residual is at the FP floor.  This is the *interesting*
//     state for verifying that the composition itself doesn't introduce
//     drift — any non-zero residual is a real bug.
//   - This test does NOT flag any EMF edges (fofc_edge_cnt left at 0).
//     EMF refluxing is exercised by `test_ct_flux_conservation.cpp`; the
//     FOFC edge-correction path will be covered by an extension.
//
// Author: carlo.musolino@aei.mpg.de

#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/amr/amr_ghosts.hh>
#include <grace/amr/forest.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/data_structures/variable_properties.hh>
#include <grace/evolution/evolve.hh>
#include <grace/evolution/refluxing.hh>
#include <grace/physics/eos/eos_storage.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#ifdef GRACE_ENABLE_FOFC

namespace {

// ----------------------------------------------------------------------------
// Canonical-ordered, MPI-bit-invariant volume sum.  Same shape as the
// helpers in test_flux_reflux.cpp / test_ct_flux_conservation.cpp.
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// p4est face-id convention: 0=-x, 1=+x, 2=-y, 3=+y, 4=-z, 5=+z.
// Returns the (i,j,k) of the interior cell whose `face_id` boundary is the
// F-C interface — i.e. the cell sitting just inside the quadrant on that face.
// Uses a midpoint in the two transverse axes so the flagged cell is well
// inside the block (not at a corner where ghost ranges may overlap).
// ----------------------------------------------------------------------------
struct boundary_cell_ijk_t { int i, j, k; };

boundary_cell_ijk_t boundary_cell_for_face(int face_id,
                                           int nx, int ny, int nz, int ngz)
{
	int const ic = nx/2 + ngz;
	int const jc = ny/2 + ngz;
	int const kc = nz/2 + ngz;
	switch (face_id) {
		case 0: return { ngz,         jc,          kc          };
		case 1: return { nx+ngz-1,    jc,          kc          };
		case 2: return { ic,          ngz,         kc          };
		case 3: return { ic,          ny+ngz-1,    kc          };
		case 4: return { ic,          jc,          ngz         };
		case 5: return { ic,          jc,          nz+ngz-1    };
		default: return { ic, jc, kc };
	}
}

// face_id <-> opposite face_id (flip sign of axis).
inline int opposite_face(int f) { return f ^ 1; }

} // namespace

// =============================================================================
// Test
// =============================================================================

TEST_CASE("FOFC + reflux preserves conservation on AMR (manually flagged cells)",
          "[fofc][conservation][amr]")
{
	using namespace grace;
	using eos_t = hybrid_eos_t<piecewise_polytropic_eos_t>;

	Kokkos::fence();
	parallel::mpi_barrier();

	DECLARE_GRID_EXTENTS;
	(void)nq;
	int const rank = parallel::mpi_comm_rank();

	auto& vlist  = variable_list::get();
	auto& state  = vlist.getstate();
	auto& stag   = vlist.getstaggeredstate();
	auto& dx_arr = vlist.getspacings();
	int const nvars_hrsc = variables::get_n_hrsc();

	auto dx_arr_h = Kokkos::create_mirror_view(dx_arr);
	Kokkos::deep_copy(dx_arr_h, dx_arr);

	// -------------------------------------------------------------------
	// 1. Capture initial integrals.
	// -------------------------------------------------------------------
	auto state_h_initial = Kokkos::create_mirror_view(state);
	Kokkos::deep_copy(state_h_initial, state);

	std::vector<double> initial(nvars_hrsc), l1(nvars_hrsc);
	for (int v = 0; v < nvars_hrsc; ++v) {
		initial[v] = canonical_global_volume_sum(
			[&] (VEC(size_t i, size_t j, size_t k), size_t q) {
				double const vol = dx_arr_h(0,q) * dx_arr_h(1,q) * dx_arr_h(2,q);
				return state_h_initial(VEC(i,j,k), v, q) * vol;
			});
		l1[v] = canonical_global_volume_sum(
			[&] (VEC(size_t i, size_t j, size_t k), size_t q) {
				double const vol = dx_arr_h(0,q) * dx_arr_h(1,q) * dx_arr_h(2,q);
				return std::fabs(state_h_initial(VEC(i,j,k), v, q)) * vol;
			});
	}

	// -------------------------------------------------------------------
	// 2. Production HO flux + EMF computation.
	// -------------------------------------------------------------------
	constexpr double t = 0.0, dt = 1.0, dtfact = 1.0;
	compute_fluxes<eos_t>(t, dt, dtfact, state, state, stag, stag);
	compute_emfs(t, dt, dtfact, state, state, stag, stag);
	Kokkos::fence();

	// -------------------------------------------------------------------
	// 3. Identify target cells.  Prefer F-C interface cells (one on each
	//    side per descriptor) — they exercise the FOFC × reflux composition
	//    most strongly.  Fall back to interior cells if no F-C exists.
	// -------------------------------------------------------------------
	auto& ghost_layer = amr_ghosts::get();
	auto desc         = ghost_layer.get_reflux_face_descriptors();
	int const n_local_fc = (int)desc.coarse_qid.extent(0);

	auto coarse_qid_h     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, desc.coarse_qid);
	auto coarse_face_id_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, desc.coarse_face_id);
	auto coarse_is_rem_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, desc.coarse_is_remote);
	auto fine_qid_h       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, desc.fine_qid);
	auto fine_is_rem_h    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, desc.fine_is_remote);

	std::vector<std::array<int,4>> targets;  // {q, i, j, k}
	for (int n = 0; n < n_local_fc; ++n) {
		int const face_id = coarse_face_id_h(n);
		// Coarse-side target (only if owned by this rank).
		if (!coarse_is_rem_h(n)) {
			auto cb = boundary_cell_for_face(face_id, (int)nx, (int)ny, (int)nz, (int)ngz);
			targets.push_back({ coarse_qid_h(n), cb.i, cb.j, cb.k });
		}
		// Fine-side targets: first sub-quadrant that's local; opposite face.
		int const ofid = opposite_face(face_id);
		for (int m = 0; m < 4; ++m) {
			if (!fine_is_rem_h(n, m)) {
				auto fb = boundary_cell_for_face(ofid, (int)nx, (int)ny, (int)nz, (int)ngz);
				targets.push_back({ fine_qid_h(n, m), fb.i, fb.j, fb.k });
				break;  // one fine cell per descriptor is enough
			}
		}
	}

	// Unigrid / no-F-C fallback: 4 interior cells from different quadrants.
	if (targets.empty() && nq > 0) {
		int const ic = (int)nx/2 + (int)ngz;
		int const jc = (int)ny/2 + (int)ngz;
		int const kc = (int)nz/2 + (int)ngz;
		int const n_target = std::min<int>(4, (int)nq);
		for (int q = 0; q < n_target; ++q) {
			targets.push_back({ q, ic, jc, kc });
		}
	}

	// -------------------------------------------------------------------
	// 4. Reset FOFC tags + counts, populate face-slot lists.
	// -------------------------------------------------------------------
	auto& fofc_faces    = vlist.getfofcfacetags();
	auto& fofc_edges    = vlist.getfofcedgetags();
	auto& fofc_face_cnt = vlist.getfofcfcnt();
	auto& fofc_edge_cnt = vlist.getfofcecnt();
	auto& fofc_fx       = vlist.getfofcfx();
	auto& fofc_fy       = vlist.getfofcfy();
	auto& fofc_fz       = vlist.getfofcfz();

	Kokkos::deep_copy(fofc_faces, 0);
	Kokkos::deep_copy(fofc_edges, 0);
	Kokkos::deep_copy(fofc_face_cnt, 0);
	Kokkos::deep_copy(fofc_edge_cnt, 0);

	auto fofc_fx_h = Kokkos::create_mirror_view(fofc_fx);
	auto fofc_fy_h = Kokkos::create_mirror_view(fofc_fy);
	auto fofc_fz_h = Kokkos::create_mirror_view(fofc_fz);
	auto cnt_h     = Kokkos::create_mirror_view(fofc_face_cnt);
	cnt_h(0) = cnt_h(1) = cnt_h(2) = 0;

	for (auto const& tgt : targets) {
		int const q = tgt[0];
		int const i = tgt[1], j = tgt[2], k = tgt[3];
		// 6 faces of cell (i,j,k): ±x at i and i+1, ±y at j and j+1, ±z at k and k+1.
		fofc_fx_h(cnt_h(0)++) = { q, i,   j,   k   };
		fofc_fx_h(cnt_h(0)++) = { q, i+1, j,   k   };
		fofc_fy_h(cnt_h(1)++) = { q, i,   j,   k   };
		fofc_fy_h(cnt_h(1)++) = { q, i,   j+1, k   };
		fofc_fz_h(cnt_h(2)++) = { q, i,   j,   k   };
		fofc_fz_h(cnt_h(2)++) = { q, i,   j,   k+1 };
	}
	Kokkos::deep_copy(fofc_fx,       fofc_fx_h);
	Kokkos::deep_copy(fofc_fy,       fofc_fy_h);
	Kokkos::deep_copy(fofc_fz,       fofc_fz_h);
	Kokkos::deep_copy(fofc_face_cnt, cnt_h);
	Kokkos::fence();

	// -------------------------------------------------------------------
	// 5. apply_fofc_correction — overwrites flagged face fluxes with
	//    donor + LLF.  Edge counts are 0, so the EMF-correction parallel_fors
	//    iterate over an empty range (no-ops).
	// -------------------------------------------------------------------
	apply_fofc_correction<eos_t>(t, dt, dtfact, state, state, stag, stag);
	Kokkos::fence();

	// -------------------------------------------------------------------
	// 6. Production reflux + state update.
	// -------------------------------------------------------------------
	{
		auto ctx = reflux_fill_flux_buffers();
		Kokkos::fence();
		reflux_correct_fluxes(ctx);
		Kokkos::fence();
	}
	add_fluxes_and_source_terms<eos_t>(t, dt, dtfact, state, state, stag, stag);
	Kokkos::fence();
	parallel::mpi_barrier();

	// -------------------------------------------------------------------
	// 7. Final integrals and conservation check.
	//    Tolerance: 1e-13 · l1.  This is ~50× strict eps · l1, leaving
	//    headroom for FMA / op-order variability across platforms while
	//    still being tight enough to catch a real conservation bug (which
	//    would produce a drift many orders above eps).
	// -------------------------------------------------------------------
	auto state_h_final = Kokkos::create_mirror_view(state);
	Kokkos::deep_copy(state_h_final, state);

	int n_local_targets = (int)targets.size();
	int n_global_targets = 0;
	parallel::mpi_allreduce(&n_local_targets, &n_global_targets, 1, sc_MPI_SUM);

	bool any_failed = false;
	for (int v = 0; v < nvars_hrsc; ++v) {
		double final_v = canonical_global_volume_sum(
			[&] (VEC(size_t i, size_t j, size_t k), size_t q) {
				double const vol = dx_arr_h(0,q) * dx_arr_h(1,q) * dx_arr_h(2,q);
				return state_h_final(VEC(i,j,k), v, q) * vol;
			});
		double const delta = final_v - initial[v];
		double const tol   = 1e-13 * (l1[v] + 1e-30);

		if (rank == 0) {
			std::cout << "FOFC conservation v=" << v
			          << " : initial=" << initial[v]
			          << " final="     << final_v
			          << " delta="     << delta
			          << " l1="        << l1[v]
			          << " tol="       << tol << std::endl;
		}
		INFO("v=" << v
		     << " initial=" << initial[v]
		     << " final="   << final_v
		     << " delta="   << delta
		     << " tol="     << tol);

		if (std::fabs(delta) >= tol) {
			any_failed = true;
			CHECK(std::fabs(delta) < tol);
		}
	}
	CHECK_FALSE(any_failed);

	if (rank == 0) {
		std::cout << "FOFC conservation: " << n_global_targets
		          << " target cells flagged globally"
		          << " (this rank: " << n_local_targets
		          << "; face slots: x=" << cnt_h(0)
		          << " y="    << cnt_h(1)
		          << " z="    << cnt_h(2) << ")" << std::endl;
	}
}

#else  // GRACE_ENABLE_FOFC

TEST_CASE("FOFC + reflux preserves conservation (disabled)",
          "[fofc][conservation][.disabled]")
{
	SUCCEED("FOFC not enabled at compile time (GRACE_ENABLE_FOFC undefined). "
	        "Rebuild with -DGRACE_ENABLE_FOFC=ON to exercise this test.");
}

#endif // GRACE_ENABLE_FOFC
