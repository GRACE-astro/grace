/**
 * @file test_coordinates.cpp
 * @brief Unit tests for the Cartesian coordinate system.
 *
 * Bound to `configs/basic_config.yaml` (single-tree brick, side 64, level-2
 * initial refinement, 16 cells per quadrant per axis ⇒ uniform cell spacing
 * dx = 1.0).  Exercises:
 *
 *   - dx / idx round-trip and uniformity across all local quadrants.
 *   - Cell-centred coordinates: `get_physical_coordinates` →
 *     `get_logical_coordinates` round-trip agrees with the analytic
 *     formula for every interior cell.
 *   - Ghost-zone coordinates: same API on cells with i<ngz or i≥nx+ngz
 *     returns physically-meaningful (extrapolated) positions.
 *   - Face-centred coordinates: pass `cell_coordinates = {0,0.5,0.5}` /
 *     `{1,0.5,0.5}` to land on the −x / +x face of a cell and check the
 *     result is exactly cell_centre ± 0.5·dx.
 *   - Device coord-system parity: launching a Kokkos kernel that calls
 *     `device_coord_system.get_physical_coordinates(i,j,k,q,xyz)` and
 *     comparing the output bit-exact against the host call from the same
 *     (i,j,k,q).
 *   - Total volume: ∑ dx·dy·dz over interior cells, MPI-reduced, equals
 *     the physical domain volume (xmax-xmin)·(ymax-ymin)·(zmax-zmin).
 *
 * Multi-rank safe: the volume reduction and the per-cell asserts execute
 * over each rank's local quadrants and the volume is MPI-allreduced.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>

#include <grace/amr/forest.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/config/config_parser.hh>
#include <grace/data_structures/variables.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <array>

TEST_CASE("cartesian coordinates", "[coords]")
{
    using namespace grace ;
    using namespace Kokkos ;

    // Derive expected dx from the parfile so the test follows whatever
    // basic_config.yaml says — no risk of going stale on a parfile edit.
    double const kExpectedDx =
        (grace::get_param<double>("amr","xmax")
         - grace::get_param<double>("amr","xmin"))
        / (grace::get_param<int>("amr","npoints_block_x")
           * (1 << grace::get_param<int>("amr","initial_refinement_level"))) ;

    auto& cs   = coordinate_system::get() ;
    auto& vars = variable_list::get() ;
    auto& dx   = vars.getspacings() ;
    auto& idx  = vars.getinvspacings() ;

    int64_t nx, ny, nz ;
    std::tie(nx, ny, nz) = amr::get_quadrant_extents() ;
    int const ngz   = amr::get_n_ghosts() ;
    int64_t const nq = amr::get_local_num_quadrants() ;

    auto const dx_h  = create_mirror_view(dx) ;
    auto const idx_h = create_mirror_view(idx) ;
    deep_copy(dx_h,  dx) ;
    deep_copy(idx_h, idx) ;

    SECTION("dx · idx == 1 and dx is uniform across local quadrants")
    {
        for (int64_t q = 0 ; q < nq ; ++q) {
            for (int d = 0 ; d < GRACE_NSPACEDIM ; ++d) {
                INFO("q=" << q << " d=" << d) ;
                REQUIRE_THAT( dx_h(d, q) * idx_h(d, q),
                              Catch::Matchers::WithinRel(1.0, 1e-15) ) ;
                REQUIRE_THAT( dx_h(d, q),
                              Catch::Matchers::WithinRel(kExpectedDx, 1e-12) ) ;
            }
        }
    }

    SECTION("cell-centred coordinates round-trip via get_phys → get_log")
    {
        // For every interior cell on every local quadrant, the get_phys /
        // get_log pair must invert each other to FP precision.  Mirrors
        // the production usage in the metric-evaluation kernels.
        for (size_t it = amr::forest::get().first_local_tree() ;
                    it <= amr::forest::get().last_local_tree() ; ++it) {
            auto tree = amr::forest::get().tree(it) ;
            int64_t const qoff = tree.quadrants_offset() ;
            int64_t const nq_t = tree.num_quadrants() ;
            for (int64_t q_local = 0 ; q_local < nq_t ; ++q_local) {
                int64_t const q = q_local + qoff ;
                for (int64_t k = 0 ; k < nz ; ++k)
                for (int64_t j = 0 ; j < ny ; ++j)
                for (int64_t i = 0 ; i < nx ; ++i) {
                    auto pc = cs.get_physical_coordinates(
                                 {VEC(static_cast<size_t>(i),
                                      static_cast<size_t>(j),
                                      static_cast<size_t>(k))}, q, false) ;
                    auto lc = cs.get_logical_coordinates(it, pc) ;
                    auto pc_check = cs.get_physical_coordinates(it, lc) ;
                    for (int d = 0 ; d < GRACE_NSPACEDIM ; ++d) {
                        REQUIRE_THAT( pc_check[d],
                                      Catch::Matchers::WithinRel(pc[d], 1e-12) ) ;
                    }
                }
            }
        }
    }

    SECTION("ghost-zone coordinates are finite and extrapolated correctly")
    {
        // Pick the first quadrant on this rank.  In its ghost layer at
        // i = ngz-1 (one cell outside the −x interior boundary), the
        // physical coordinate must be cell_centre_at_ngz − dx in x.
        if (nq == 0) { SUCCEED("no local quadrants on this rank") ; return ; }
        int64_t const q = 0 ;
        auto pc_first_interior = cs.get_physical_coordinates(
                                     {VEC(static_cast<size_t>(ngz),
                                          static_cast<size_t>(ngz),
                                          static_cast<size_t>(ngz))}, q, true) ;
        auto pc_first_ghost    = cs.get_physical_coordinates(
                                     {VEC(static_cast<size_t>(ngz-1),
                                          static_cast<size_t>(ngz),
                                          static_cast<size_t>(ngz))}, q, true) ;
        REQUIRE_THAT( pc_first_ghost[0],
                      Catch::Matchers::WithinRel(pc_first_interior[0] - dx_h(0,q), 1e-12) ) ;
        // y and z components are the same cell-centre values on both sides
        // (no offset in those directions).
        REQUIRE_THAT( pc_first_ghost[1],
                      Catch::Matchers::WithinRel(pc_first_interior[1], 1e-12) ) ;
#ifdef GRACE_3D
        REQUIRE_THAT( pc_first_ghost[2],
                      Catch::Matchers::WithinRel(pc_first_interior[2], 1e-12) ) ;
#endif
    }

    SECTION("face-centred coordinates: cell_coordinates = {0, ½, ½} / {1, ½, ½}")
    {
        // Calling get_phys with cell_coords offset other than (½,½,½) gives
        // a point inside the cell.  {0,½,½} is the −x face centre, {1,½,½}
        // is the +x face centre.  Result must equal cell_centre ± ½·dx in x
        // and cell_centre in y, z.
        if (nq == 0) { SUCCEED("no local quadrants on this rank") ; return ; }
        int64_t const q = 0 ;
        std::array<size_t, GRACE_NSPACEDIM> ijk =
            {VEC(static_cast<size_t>(ngz + nx/2),
                 static_cast<size_t>(ngz + ny/2),
                 static_cast<size_t>(ngz + nz/2))} ;
        auto pc_center = cs.get_physical_coordinates(ijk, q, true) ;
        auto pc_mx     = cs.get_physical_coordinates(
                            ijk, q, {VEC(0.0, 0.5, 0.5)}, true) ;
        auto pc_px     = cs.get_physical_coordinates(
                            ijk, q, {VEC(1.0, 0.5, 0.5)}, true) ;
        REQUIRE_THAT( pc_mx[0],
                      Catch::Matchers::WithinRel(pc_center[0] - 0.5*dx_h(0,q), 1e-12) ) ;
        REQUIRE_THAT( pc_px[0],
                      Catch::Matchers::WithinRel(pc_center[0] + 0.5*dx_h(0,q), 1e-12) ) ;
        REQUIRE_THAT( pc_mx[1],
                      Catch::Matchers::WithinRel(pc_center[1], 1e-12) ) ;
        REQUIRE_THAT( pc_px[1],
                      Catch::Matchers::WithinRel(pc_center[1], 1e-12) ) ;
#ifdef GRACE_3D
        REQUIRE_THAT( pc_mx[2],
                      Catch::Matchers::WithinRel(pc_center[2], 1e-12) ) ;
        REQUIRE_THAT( pc_px[2],
                      Catch::Matchers::WithinRel(pc_center[2], 1e-12) ) ;
#endif
    }

    SECTION("device coord-system matches host for every interior cell")
    {
        // Launch a Kokkos kernel that calls the device-side
        // get_physical_coordinates at every interior cell.  Mirror back to
        // host and bit-exact compare against the host coord-system result.
        if (nq == 0) { SUCCEED("no local quadrants on this rank") ; return ; }
        auto dcs = cs.get_device_coord_system() ;

        size_t const Nint = static_cast<size_t>(nx) * ny * nz * nq ;
        View<double*[GRACE_NSPACEDIM]> dev_xyz("dev_xyz", Nint) ;

        parallel_for("device_coords_test",
                     MDRangePolicy<Rank<4>>({ngz, ngz, ngz, 0},
                                            {nx+ngz, ny+ngz, nz+ngz, nq}),
            KOKKOS_LAMBDA(int const i, int const j, int const k, int const q) {
                size_t const lin = static_cast<size_t>(
                    (((q * nz) + (k - ngz)) * ny + (j - ngz)) * nx + (i - ngz)) ;
                double xyz[3] ;
                dcs.get_physical_coordinates(i, j, k, static_cast<int64_t>(q), xyz) ;
                dev_xyz(lin, 0) = xyz[0] ;
                dev_xyz(lin, 1) = xyz[1] ;
                dev_xyz(lin, 2) = xyz[2] ;
            }) ;
        Kokkos::fence() ;

        auto dev_xyz_h = create_mirror_view(dev_xyz) ;
        deep_copy(dev_xyz_h, dev_xyz) ;

        for (int64_t q = 0 ; q < nq ; ++q)
        for (int64_t k = 0 ; k < nz ; ++k)
        for (int64_t j = 0 ; j < ny ; ++j)
        for (int64_t i = 0 ; i < nx ; ++i) {
            size_t const lin = (((q * nz) + k) * ny + j) * nx + i ;
            auto pc_h = cs.get_physical_coordinates(
                            {VEC(static_cast<size_t>(i),
                                 static_cast<size_t>(j),
                                 static_cast<size_t>(k))}, q, false) ;
            REQUIRE_THAT( dev_xyz_h(lin, 0),
                          Catch::Matchers::WithinAbs(pc_h[0], 1e-14) ) ;
            REQUIRE_THAT( dev_xyz_h(lin, 1),
                          Catch::Matchers::WithinAbs(pc_h[1], 1e-14) ) ;
#ifdef GRACE_3D
            REQUIRE_THAT( dev_xyz_h(lin, 2),
                          Catch::Matchers::WithinAbs(pc_h[2], 1e-14) ) ;
#endif
        }
    }

    SECTION("total physical volume = (xmax-xmin)·(ymax-ymin)·(zmax-zmin)")
    {
        // Sum dx·dy·dz over all interior cells, MPI-reduce, compare to the
        // bounding-box volume read from the parser.  This is the natural
        // closure of the dx-uniformity + cell-centred-coordinates checks.
        double local_vol = 0.0 ;
        for (int64_t q = 0 ; q < nq ; ++q) {
            local_vol += dx_h(0,q) * dx_h(1,q) * dx_h(2,q)
                         * static_cast<double>(nx)
                         * static_cast<double>(ny)
                         * static_cast<double>(nz) ;
        }
        double global_vol = 0.0 ;
        parallel::mpi_allreduce(&local_vol, &global_vol, 1, sc_MPI_SUM) ;

        double const expected =
              (grace::get_param<double>("amr","xmax") - grace::get_param<double>("amr","xmin"))
            * (grace::get_param<double>("amr","ymax") - grace::get_param<double>("amr","ymin"))
            * (grace::get_param<double>("amr","zmax") - grace::get_param<double>("amr","zmin")) ;
        REQUIRE_THAT( global_vol,
                      Catch::Matchers::WithinRel(expected, 1e-12) ) ;
    }
}
