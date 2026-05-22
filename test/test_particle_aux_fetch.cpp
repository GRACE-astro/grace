/**
 * @file test_particle_aux_fetch.cpp
 * @brief Phase 1b validation: round-trip distribution_plan_t-based fetch.
 *
 * Strategy: fill aux[RHO_] with a known LINEAR function f(x,y,z) (trilinear
 * interp is exact for linear functions). Place tracers at random positions in
 * the global domain; some land on this rank (local fetch path), some on other
 * ranks (MPI path). Both go through the same Distributor pipeline. Verify
 * fetched values match f(pos) within tolerance.
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace/particles/particle_aux_fetch.hh>
#include <grace/particles/particle_owner_search.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/coordinates/coordinate_systems.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <random>
#include <vector>
#include <mpi.h>

namespace {

constexpr double A_COEF = 1.234;
constexpr double B_COEF = -0.567;
constexpr double C_COEF = 2.890;
constexpr double D_COEF = 0.111;

inline double linear_field(double x, double y, double z) {
    return A_COEF * x + B_COEF * y + C_COEF * z + D_COEF;
}

} // namespace

TEST_CASE("particle_aux_fetch: trilinear round-trip on aux[RHO_]",
          "[particles][aux_fetch]")
{
    using namespace grace::particles;
    using namespace grace::variables;

    // ------------------------------------------------------------------
    // 1. Refresh shadow + fill aux[RHO_] with a linear function (interior
    //    AND ghost cells, so trilinear at any halo position is exact).
    // ------------------------------------------------------------------
    auto& sh = fluid_topology_shadow_t::get();
    sh.refresh();

    auto& aux = grace::variable_list::get().getaux();
    long nx, ny, nz;
    std::tie(nx, ny, nz) = grace::amr::get_quadrant_extents();
    const std::size_t nq = grace::amr::get_local_num_quadrants();
    const int ngz = grace::amr::get_n_ghosts();
    const std::size_t ncells = (nx + 2*ngz) * (ny + 2*ngz) * (nz + 2*ngz) * nq;

    auto& coord_system = grace::coordinate_system::get();
    auto h_aux = Kokkos::create_mirror_view(aux);

    for (std::size_t icell = 0; icell < ncells; ++icell) {
        const std::size_t i = icell % (nx + 2*ngz);
        const std::size_t j = (icell / (nx + 2*ngz)) % (ny + 2*ngz);
        const std::size_t k = (icell / (nx + 2*ngz) / (ny + 2*ngz)) % (nz + 2*ngz);
        const std::size_t q = (icell / (nx + 2*ngz) / (ny + 2*ngz) / (nz + 2*ngz));
        auto pcoords = coord_system.get_physical_coordinates({i, j, k}, q, true);
        h_aux(i, j, k, RHO_, q) = linear_field(pcoords[0], pcoords[1], pcoords[2]);
    }
    Kokkos::deep_copy(aux, h_aux);

    // ------------------------------------------------------------------
    // 2. Sample N random positions strictly inside the global domain.
    // ------------------------------------------------------------------
    int rank = 0, nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    double gxlo =  std::numeric_limits<double>::infinity();
    double gylo =  std::numeric_limits<double>::infinity();
    double gzlo =  std::numeric_limits<double>::infinity();
    double gxhi = -std::numeric_limits<double>::infinity();
    double gyhi = -std::numeric_limits<double>::infinity();
    double gzhi = -std::numeric_limits<double>::infinity();
    for (const auto& g : sh.local_geometry()) {
        gxlo = std::min(gxlo, g.bbox.xlo); gxhi = std::max(gxhi, g.bbox.xhi);
        gylo = std::min(gylo, g.bbox.ylo); gyhi = std::max(gyhi, g.bbox.yhi);
        gzlo = std::min(gzlo, g.bbox.zlo); gzhi = std::max(gzhi, g.bbox.zhi);
    }
    parallel::mpi_allreduce_inplace(&gxlo, 1, sc_MPI_MIN);
    parallel::mpi_allreduce_inplace(&gylo, 1, sc_MPI_MIN);
    parallel::mpi_allreduce_inplace(&gzlo, 1, sc_MPI_MIN);
    parallel::mpi_allreduce_inplace(&gxhi, 1, sc_MPI_MAX);
    parallel::mpi_allreduce_inplace(&gyhi, 1, sc_MPI_MAX);
    parallel::mpi_allreduce_inplace(&gzhi, 1, sc_MPI_MAX);

    // Avoid the half-open hi-face ambiguity.
    const double pad = 1e-9 * std::max({gxhi-gxlo, gyhi-gylo, gzhi-gzlo});
    gxhi -= pad; gyhi -= pad; gzhi -= pad;

    const std::size_t N = 64;
    std::mt19937 rng(0xC0FFEE + rank);
    std::uniform_real_distribution<double> ux(gxlo, gxhi);
    std::uniform_real_distribution<double> uy(gylo, gyhi);
    std::uniform_real_distribution<double> uz(gzlo, gzhi);

    std::vector<std::array<double, 3>> h_positions(N);
    for (std::size_t i = 0; i < N; ++i) {
        h_positions[i] = {ux(rng), uy(rng), uz(rng)};
    }

    // ------------------------------------------------------------------
    // 3. Resolve owners.
    // ------------------------------------------------------------------
    std::vector<owner_t> owners;
    sh.find_owners_batch(h_positions, owners);

    // ------------------------------------------------------------------
    // 4. Allocate device views, populate, run fetch.
    // ------------------------------------------------------------------
    Kokkos::View<int*,        grace::default_space> d_owner_rank(
        Kokkos::ViewAllocateWithoutInitializing("owner_rank"), N);
    Kokkos::View<int*,        grace::default_space> d_owner_quad(
        Kokkos::ViewAllocateWithoutInitializing("owner_quad"), N);
    Kokkos::View<double*[3],  grace::default_space> d_pos(
        Kokkos::ViewAllocateWithoutInitializing("pos"), N);
    Kokkos::View<double**,    grace::default_space> d_out("out", N, 1);

    auto h_owner_rank = Kokkos::create_mirror_view(d_owner_rank);
    auto h_owner_quad = Kokkos::create_mirror_view(d_owner_quad);
    auto h_pos        = Kokkos::create_mirror_view(d_pos);
    for (std::size_t i = 0; i < N; ++i) {
        h_owner_rank(i) = owners[i].rank;
        h_owner_quad(i) = owners[i].local_quad;
        h_pos(i, 0) = h_positions[i][0];
        h_pos(i, 1) = h_positions[i][1];
        h_pos(i, 2) = h_positions[i][2];
    }
    Kokkos::deep_copy(d_owner_rank, h_owner_rank);
    Kokkos::deep_copy(d_owner_quad, h_owner_quad);
    Kokkos::deep_copy(d_pos,        h_pos);

    std::array<fetch_field_spec_t, 1> fields = {{
        {field_source::AUX, RHO_}
    }};
    fetch_at_positions<1>(MPI_COMM_WORLD, N, fields,
                          d_owner_rank, d_owner_quad, d_pos, d_out);

    // ------------------------------------------------------------------
    // 5. Compare to analytic. Trilinear is exact for linear → tight tol.
    // ------------------------------------------------------------------
    auto h_out = Kokkos::create_mirror_view(d_out);
    Kokkos::deep_copy(h_out, d_out);

    int n_checked = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (owners[i].rank < 0) continue;
        const double expected = linear_field(h_positions[i][0],
                                             h_positions[i][1],
                                             h_positions[i][2]);
        INFO("particle " << i << " pos=(" << h_positions[i][0] << ", "
             << h_positions[i][1] << ", " << h_positions[i][2] << ") owner="
             << owners[i].rank << " quad=" << owners[i].local_quad);
        REQUIRE_THAT(h_out(i, 0),
                     Catch::Matchers::WithinRel(expected, 1e-10) ||
                     Catch::Matchers::WithinAbs(expected, 1e-10));
        ++n_checked;
    }
    INFO("n_checked=" << n_checked << " of " << N);
    REQUIRE(n_checked > 0);
}

#endif // GRACE_ENABLE_PARTICLES
