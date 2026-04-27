/**
 * @file test_particle_owner_search.cpp
 * @brief Phase 1a validation: fluid-topology shadow against brute-force search.
 *
 * Strategy: build the grace forest from a parfile, refresh the topology
 * shadow, then for a random sample of points inside the global domain compare
 * the shadow's find_owner() against a brute-force scan of every quad on every
 * rank. find_owners are deterministic and rank-replicated, so we run the
 * brute-force on rank 0 by allgathering all per-rank bboxes.
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <catch2/catch_test_macros.hpp>

#include <grace/particles/particle_owner_search.hh>
#include <grace/amr/forest.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/config/config_parser.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <array>
#include <random>
#include <vector>

namespace {

struct global_quad_record {
    int    owner_rank;
    double xlo, ylo, zlo, xhi, yhi, zhi;
};

// Allgather every local quad's bbox + owner rank → a global record vector
// sufficient for the brute-force ground-truth comparison.
std::vector<global_quad_record> gather_all_quads()
{
    using namespace grace::particles;
    auto& sh = fluid_topology_shadow_t::get();
    const auto& local = sh.local_geometry();

    int rank = 0, nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int n_local = static_cast<int>(local.size());
    std::vector<int> counts(nproc), displs(nproc);
    MPI_Allgather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int n_global = 0;
    for (int r = 0; r < nproc; ++r) { displs[r] = n_global; n_global += counts[r]; }

    // Pack local: 6 doubles per quad.
    std::vector<double> local_pack(6 * static_cast<std::size_t>(n_local));
    for (int i = 0; i < n_local; ++i) {
        local_pack[6*i+0] = local[i].bbox.xlo;
        local_pack[6*i+1] = local[i].bbox.ylo;
        local_pack[6*i+2] = local[i].bbox.zlo;
        local_pack[6*i+3] = local[i].bbox.xhi;
        local_pack[6*i+4] = local[i].bbox.yhi;
        local_pack[6*i+5] = local[i].bbox.zhi;
    }
    std::vector<int> dcounts(nproc), ddispls(nproc);
    int total = 0;
    for (int r = 0; r < nproc; ++r) {
        dcounts[r] = 6 * counts[r];
        ddispls[r] = total;
        total += dcounts[r];
    }
    std::vector<double> global_pack(static_cast<std::size_t>(total));
    MPI_Allgatherv(local_pack.data(), 6 * n_local, MPI_DOUBLE,
                   global_pack.data(), dcounts.data(), ddispls.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    std::vector<global_quad_record> out;
    out.reserve(static_cast<std::size_t>(n_global));
    for (int r = 0; r < nproc; ++r) {
        for (int i = 0; i < counts[r]; ++i) {
            int idx = displs[r] + i;
            out.push_back(global_quad_record{
                r,
                global_pack[6*idx+0], global_pack[6*idx+1], global_pack[6*idx+2],
                global_pack[6*idx+3], global_pack[6*idx+4], global_pack[6*idx+5]});
        }
    }
    return out;
}

int brute_force_owner(const std::vector<global_quad_record>& all,
                      double x, double y, double z)
{
    for (const auto& q : all) {
        if (x < q.xlo || x >= q.xhi) continue;
        if (y < q.ylo || y >= q.yhi) continue;
        if (z < q.zlo || z >= q.zhi) continue;
        return q.owner_rank;
    }
    return -1;
}

} // namespace

TEST_CASE("particle_owner_search: shadow refresh + find_owner agrees with brute-force",
          "[particles][owner_search]")
{
    using namespace grace::particles;

    auto& sh = fluid_topology_shadow_t::get();
    sh.refresh();
    REQUIRE(sh.local_num_quads() == grace::amr::get_local_num_quadrants());

    auto all_quads = gather_all_quads();
    REQUIRE(!all_quads.empty());

    // Build a sample of points inside the global bbox.
    double gxlo =  std::numeric_limits<double>::infinity();
    double gylo =  std::numeric_limits<double>::infinity();
    double gzlo =  std::numeric_limits<double>::infinity();
    double gxhi = -std::numeric_limits<double>::infinity();
    double gyhi = -std::numeric_limits<double>::infinity();
    double gzhi = -std::numeric_limits<double>::infinity();
    for (const auto& q : all_quads) {
        gxlo = std::min(gxlo, q.xlo); gxhi = std::max(gxhi, q.xhi);
        gylo = std::min(gylo, q.ylo); gyhi = std::max(gyhi, q.yhi);
        gzlo = std::min(gzlo, q.zlo); gzhi = std::max(gzhi, q.zhi);
    }

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> ux(gxlo, gxhi);
    std::uniform_real_distribution<double> uy(gylo, gyhi);
    std::uniform_real_distribution<double> uz(gzlo, gzhi);

    constexpr int N = 200;
    int n_match = 0, n_outside = 0;
    for (int i = 0; i < N; ++i) {
        double x = ux(rng), y = uy(rng), z = uz(rng);
        // Pull strictly inside any face to avoid the half-open ambiguity at hi.
        x = std::min(x, gxhi - 1e-12);
        y = std::min(y, gyhi - 1e-12);
        z = std::min(z, gzhi - 1e-12);
        auto owner_shadow = sh.find_owner(x, y, z);
        int owner_brute = brute_force_owner(all_quads, x, y, z);
        if (owner_brute < 0) { n_outside++; continue; }
        REQUIRE(owner_shadow.rank == owner_brute);
        ++n_match;
    }
    INFO("n_match=" << n_match << " n_outside=" << n_outside << " of " << N);
    REQUIRE(n_match > 0);
}

TEST_CASE("particle_owner_search: fast_path_check + make_stencil for cell-center points",
          "[particles][owner_search]")
{
    using namespace grace::particles;

    auto& sh = fluid_topology_shadow_t::get();
    sh.refresh();

    auto extents = grace::amr::get_quadrant_extents();
    int nx = static_cast<int>(std::get<0>(extents));
    auto const ngz = grace::amr::get_n_ghosts();

    if (sh.local_num_quads() == 0) return;

    // Pick the first local quad and probe at its (0,0,0) cell center.
    const auto& g = sh.local_geometry()[0];
    const double xc = g.bbox.xlo + 0.5 * g.dx_cell;
    const double yc = g.bbox.ylo + 0.5 * g.dx_cell;
    const double zc = g.bbox.zlo + 0.5 * g.dx_cell;

    stencil_t st{};
    REQUIRE(sh.fast_path_check(0, xc, yc, zc, ngz, st));
    REQUIRE(st.base_i == 0);
    REQUIRE(st.base_j == 0);
    REQUIRE(st.base_k == 0);
    REQUIRE(std::abs(st.fx) < 1e-12);
    REQUIRE(std::abs(st.fy) < 1e-12);
    REQUIRE(std::abs(st.fz) < 1e-12);

    // Probe at the cell center of the (1,2,3) cell — fractional should still
    // be 0 because we land exactly on cell-center.
    const double x1 = g.bbox.xlo + 1.5 * g.dx_cell;
    const double y2 = g.bbox.ylo + 2.5 * g.dx_cell;
    const double z3 = g.bbox.zlo + 3.5 * g.dx_cell;
    if (nx > 3) {
        REQUIRE(sh.fast_path_check(0, x1, y2, z3, ngz, st));
        REQUIRE(st.base_i == 1);
        REQUIRE(st.base_j == 2);
        REQUIRE(st.base_k == 3);
        REQUIRE(std::abs(st.fx) < 1e-12);
        REQUIRE(std::abs(st.fy) < 1e-12);
        REQUIRE(std::abs(st.fz) < 1e-12);
    }

    // Halfway between two cell centers along x: fx should be 0.5.
    const double xh = g.bbox.xlo + 1.0 * g.dx_cell; // = (0.5+1.5)/2 * dx_cell
    REQUIRE(sh.fast_path_check(0, xh, yc, zc, ngz, st));
    REQUIRE(st.base_i == 0);
    REQUIRE(std::abs(st.fx - 0.5) < 1e-12);
}

#endif // GRACE_ENABLE_PARTICLES
