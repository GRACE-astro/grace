/**
 * @file test_forest.cpp
 * @brief Unit tests for `grace::amr::forest` — the p4est forest singleton
 *        built from the user yaml during AMR initialisation.
 *
 * Covered:
 *   - The underlying `p4est_t*` handle is non-null.
 *   - The local-tree window `[first_local_tree, last_local_tree]` is
 *     well-formed: either empty, or contained in `[0, num_trees)`.
 *   - The MPI-reduced global quadrant count matches
 *     `n_trees · 2^(3·initial_refinement_level)` predicted by the yaml.
 *   - Every local quadrant has properly-initialised user-data:
 *       - non-null `p.user_data` (probed via `get_regrid_flag` /
 *         `get_min_level`, both of which read from that pointer);
 *       - `regrid_flag == DEFAULT_STATE` post-construction;
 *       - `min_level` in the valid range `[0, quad.level()]`.
 *     A null user_data pointer or junk-initialised memory would either
 *     crash the test or fail one of the value assertions.
 *
 * Bound to `configs/basic_config.yaml`: a single-tree brick of side 64
 * refined to level 2 → 2^(3·2) = 64 global quadrants.
 */
#include <catch2/catch_test_macros.hpp>

#include <grace/amr/forest.hh>
#include <grace/amr/connectivity.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/amr/amr_flags.hh>
#include <grace/amr/quadrant.hh>
#include <grace/config/config_parser.hh>
#include <grace/parallel/mpi_wrappers.hh>

TEST_CASE("amr_forest", "[amr][forest]")
{
    using namespace grace ;
    auto& forest = amr::forest::get() ;

    SECTION("p4est handle is non-null")
    {
        REQUIRE( forest.get() != nullptr ) ;
    }

    SECTION("local-tree window is well-formed")
    {
        // Either an empty range (rank with no local trees, which doesn't
        // happen for the trivial single-tree basic_config but might in
        // future configurations) or contained in [0, num_trees).
        size_t const first = forest.first_local_tree() ;
        size_t const last  = forest.last_local_tree() ;
        auto const num_trees =
            static_cast<size_t>( amr::connectivity::get().get()->num_trees ) ;

        bool const empty   = first  > last ;
        bool const in_range = (first < num_trees) && (last < num_trees) ;
        REQUIRE( (empty || in_range) ) ;
    }

    SECTION("global quadrant count matches initial-refinement prediction")
    {
        // basic_config: 1 tree × 2^(3·2) = 64 quadrants in 3D.  At
        // initial_refinement_level=L the count is num_trees · 8^L.
        auto const num_trees =
            static_cast<long long>( amr::connectivity::get().get()->num_trees ) ;
        // initial_refinement_level lives in the amr yaml block — pull from
        // the parser rather than hard-coding 2 here, so this test stays
        // valid even if basic_config.yaml is later edited.
        int const lvl =
            grace::get_param<int>("amr","initial_refinement_level") ;
        long long const expected = num_trees * (1LL << (3 * lvl)) ;

        long long const local_n =
            static_cast<long long>(forest.local_num_quadrants()) ;
        long long global_n = 0 ;
        parallel::mpi_allreduce(&local_n, &global_n, 1, sc_MPI_SUM) ;
        REQUIRE( global_n == expected ) ;
    }

    SECTION("local quadrants have initialised user_data")
    {
        // Walk every local quadrant and probe its user_data via the
        // accessor methods.  If `p.user_data` were null these calls would
        // dereference a null pointer; if it pointed at uninitialised
        // memory, `regrid_flag` would carry an unexpected value.
        size_t const first = forest.first_local_tree() ;
        size_t const last  = forest.last_local_tree() ;
        if (first > last) { SUCCEED("no local trees on this rank") ; return ; }

        size_t total_checked = 0 ;
        for (size_t it = first ; it <= last ; ++it) {
            auto tree = forest.tree(it) ;
            size_t const qoff = tree.quadrants_offset() ;
            size_t const nq   = tree.num_quadrants() ;
            INFO("tree " << it << ", " << nq << " local quadrants") ;
            for (size_t q = 0 ; q < nq ; ++q) {
                auto quad = amr::get_quadrant(it, q + qoff) ;
                REQUIRE( quad.get_regrid_flag() == amr::DEFAULT_STATE ) ;
                int const ml = quad.get_min_level() ;
                REQUIRE( ml >= 0 ) ;
                REQUIRE( ml <= quad.level() ) ;
                ++total_checked ;
            }
        }
        REQUIRE( total_checked == forest.local_num_quadrants() ) ;
    }
}
