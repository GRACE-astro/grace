/**
 * @file test_connectivities.cpp
 * @brief Unit tests for `grace::amr::connectivity` — the p4est connectivity
 *        singleton built from the user yaml's `amr` section.
 *
 * Covered:
 *   - `is_valid()` — p4est's structural validity check.
 *   - The underlying `p4est_connectivity_t*` is non-null and has the
 *     expected number of trees for the loaded parfile.
 *   - Tree-0 vertex coordinates and physical extents match the bounding box
 *     declared in the yaml (`xmin/xmax`, `ymin/ymax`, `zmin/zmax`).
 *   - Tree-to-tree polarity is 0 on every face — there's no coordinate flip
 *     across the (single) tree's boundaries in the trivial brick topology.
 *
 * The test is bound to `configs/basic_config.yaml` (`xmin = ymin = zmin =
 * -32`, `xmax = ymax = zmax = +32`, all extents equal), which builds a
 * single-tree brick of side 64.
 */
#include <catch2/catch_test_macros.hpp>

#include <grace/amr/connectivity.hh>
#include <grace/amr/p4est_headers.hh>

TEST_CASE("connectivity", "[amr][connectivity]")
{
    using namespace grace ;
    auto& conn = amr::connectivity::get() ;

    SECTION("is_valid")
    {
        REQUIRE( conn.is_valid() ) ;
    }

    SECTION("single-tree brick layout from basic_config.yaml")
    {
        // basic_config.yaml sets xmin/xmax = ymin/ymax = zmin/zmax = ±32.
        // All extents are equal → one tree of side 64 covers the domain.
        auto* pc = conn.get() ;
        REQUIRE( pc != nullptr ) ;
        REQUIRE( pc->num_trees == 1 ) ;
    }

    SECTION("tree-0 vertex coordinates and physical extents")
    {
        // Vertex 0 of tree 0 in p4est's Morton ordering is the
        // (xmin, ymin, zmin) corner.
        auto v0  = conn.vertex_coordinates(0, 0) ;
        auto ext = conn.tree_coordinate_extents(0) ;

        REQUIRE( v0[0] == -32.0 ) ;
        REQUIRE( v0[1] == -32.0 ) ;
        REQUIRE( ext[0] == 64.0 ) ;
        REQUIRE( ext[1] == 64.0 ) ;
#ifdef GRACE_3D
        REQUIRE( v0[2] == -32.0 ) ;
        REQUIRE( ext[2] == 64.0 ) ;
#endif
    }

    SECTION("tree-to-tree polarity is zero on every face")
    {
        // Trivial single-tree brick: no coordinate flips across any face.
        // A non-zero polarity would indicate a cubed-sphere-like topology
        // with rotated tree-tree gluing — not expected for basic_config.
        for (int f = 0 ; f < P4EST_FACES ; ++f) {
            INFO("face index " << f) ;
            REQUIRE( conn.tree_to_tree_polarity(0, f) == 0 ) ;
        }
    }
}
