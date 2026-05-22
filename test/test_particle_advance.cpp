/**
 * @file test_particle_advance.cpp
 * @brief Phase 1c validation: G2P + RK push round-trip.
 *
 * Strategy:
 *   1. Hand-set state and aux to a uniform analytic configuration:
 *        alpha = 1, beta^i = 0, gamma_ij = delta_ij, Z^x = Z0, Z^{y,z} = 0.
 *      Then the derived 3-velocity is v^x = Z0 / sqrt(1 + Z0^2), v^{y,z} = 0,
 *      W = sqrt(1 + Z0^2).
 *   2. Seed N tracers at random positions inside the global domain.
 *   3. Call advance_substep with dt=1, dtfact=1.
 *   4. Verify dst_pos = src_pos + (alpha * v - beta) * dt = src_pos + (v_x, 0, 0).
 *      Sample slices (rho, T, Y_e, ...) get the constants we hand-set.
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace/particles/particle_advance.hh>
#include <grace/particles/particle_owner_search.hh>
#include <grace/particles/particle_storage.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/data_structures/variable_indices.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <random>
#include <vector>
#include <mpi.h>

namespace {

constexpr double Z0 = 0.5;

inline double expected_vx() {
    return Z0 / std::sqrt(1.0 + Z0 * Z0);
}

inline double expected_W() {
    return std::sqrt(1.0 + Z0 * Z0);
}

constexpr double KNOWN_RHO     = 7.31;
constexpr double KNOWN_TEMP    = 12.5;
constexpr double KNOWN_YE      = 0.42;
constexpr double KNOWN_ENTROPY = 5.55;
constexpr double KNOWN_PRESS   = 1.23;
constexpr double KNOWN_EPS     = 0.91;

void fill_uniform_state(std::size_t ncells, long nx, long ny, long nz, int ngz)
{
    using namespace grace::variables;
    auto& aux   = grace::variable_list::get().getaux();
    auto& state = grace::variable_list::get().getstate();
    auto h_aux   = Kokkos::create_mirror_view(aux);
    auto h_state = Kokkos::create_mirror_view(state);

    for (std::size_t icell = 0; icell < ncells; ++icell) {
        const std::size_t i = icell % (nx + 2*ngz);
        const std::size_t j = (icell / (nx + 2*ngz)) % (ny + 2*ngz);
        const std::size_t k = (icell / (nx + 2*ngz) / (ny + 2*ngz)) % (nz + 2*ngz);
        const std::size_t q = (icell / (nx + 2*ngz) / (ny + 2*ngz) / (nz + 2*ngz));

        // Aux: Z = (Z0, 0, 0), constant samples.
        h_aux(i, j, k, ZVECX_, q)   = Z0;
        h_aux(i, j, k, ZVECY_, q)   = 0.0;
        h_aux(i, j, k, ZVECZ_, q)   = 0.0;
        h_aux(i, j, k, RHO_, q)     = KNOWN_RHO;
        h_aux(i, j, k, TEMP_, q)    = KNOWN_TEMP;
        h_aux(i, j, k, YE_, q)      = KNOWN_YE;
        h_aux(i, j, k, ENTROPY_, q) = KNOWN_ENTROPY;
        h_aux(i, j, k, PRESS_, q)   = KNOWN_PRESS;
        h_aux(i, j, k, EPS_, q)     = KNOWN_EPS;
        h_aux(i, j, k, BX_, q)      = 0.0;
        h_aux(i, j, k, BY_, q)      = 0.0;
        h_aux(i, j, k, BZ_, q)      = 0.0;

#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
        // Flat metric: gamma_ij = delta_ij, alpha = 1, beta = 0.
        h_state(i, j, k, GXX_, q)   = 1.0;
        h_state(i, j, k, GYY_, q)   = 1.0;
        h_state(i, j, k, GZZ_, q)   = 1.0;
        h_state(i, j, k, GXY_, q)   = 0.0;
        h_state(i, j, k, GXZ_, q)   = 0.0;
        h_state(i, j, k, GYZ_, q)   = 0.0;
        h_state(i, j, k, ALP_, q)   = 1.0;
        h_state(i, j, k, BETAX_, q) = 0.0;
        h_state(i, j, k, BETAY_, q) = 0.0;
        h_state(i, j, k, BETAZ_, q) = 0.0;
#endif
    }
    Kokkos::deep_copy(aux, h_aux);
    Kokkos::deep_copy(state, h_state);
}

} // namespace

TEST_CASE("particle_advance: uniform-Z push tracks v = Z/sqrt(1+Z^2)",
          "[particles][advance]")
{
    using namespace grace::particles;

    auto& sh = fluid_topology_shadow_t::get();
    sh.refresh();

    long nx, ny, nz;
    std::tie(nx, ny, nz) = grace::amr::get_quadrant_extents();
    const std::size_t nq = grace::amr::get_local_num_quadrants();
    const int         ngz = grace::amr::get_n_ghosts();
    const std::size_t ncells = (nx + 2*ngz) * (ny + 2*ngz) * (nz + 2*ngz) * nq;
    fill_uniform_state(ncells, nx, ny, nz, ngz);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Global bbox via reduction.
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

    const double pad = 1e-9 * std::max({gxhi-gxlo, gyhi-gylo, gzhi-gzlo});
    // Pull in further along x so the post-push dst stays inside too.
    const double v_step = expected_vx();
    gxhi -= (pad + std::abs(v_step) + 1e-6);
    gyhi -= pad; gzhi -= pad;

    const std::size_t N = 64;
    std::mt19937 rng(0xDECAF + rank);
    std::uniform_real_distribution<double> ux(gxlo, gxhi);
    std::uniform_real_distribution<double> uy(gylo, gyhi);
    std::uniform_real_distribution<double> uz(gzlo, gzhi);

    std::vector<std::array<double, 3>> h_positions(N);
    for (std::size_t i = 0; i < N; ++i) {
        h_positions[i] = {ux(rng), uy(rng), uz(rng)};
    }
    std::vector<owner_t> owners;
    sh.find_owners_batch(h_positions, owners);

    // Seed the tracer container.
    tracer_container_t<grace::default_space> tr;
    tr.resize(N);
    auto h_pos        = Kokkos::create_mirror_view(tr.pos);
    auto h_owner_rank = Kokkos::create_mirror_view(tr.owner_rank);
    auto h_owner_quad = Kokkos::create_mirror_view(tr.owner_local_quad);
    for (std::size_t i = 0; i < N; ++i) {
        h_pos(i, 0) = h_positions[i][0];
        h_pos(i, 1) = h_positions[i][1];
        h_pos(i, 2) = h_positions[i][2];
        h_owner_rank(i) = owners[i].rank;
        h_owner_quad(i) = owners[i].local_quad;
    }
    Kokkos::deep_copy(tr.pos, h_pos);
    Kokkos::deep_copy(tr.owner_rank, h_owner_rank);
    Kokkos::deep_copy(tr.owner_local_quad, h_owner_quad);

    // Advance once with dt=1, dtfact=1 (Forward Euler in place).
    advance_substep(MPI_COMM_WORLD, /*dt=*/1.0, /*dtfact=*/1.0,
                    tr.pos, tr.pos, tr);

    // Pull back results.
    auto h_dst_pos = Kokkos::create_mirror_view(tr.pos);
    Kokkos::deep_copy(h_dst_pos, tr.pos);
    auto h_v       = Kokkos::create_mirror_view(tr.sample_v);
    Kokkos::deep_copy(h_v, tr.sample_v);
    auto h_W       = Kokkos::create_mirror_view(tr.sample_W);
    Kokkos::deep_copy(h_W, tr.sample_W);
    auto h_alpha   = Kokkos::create_mirror_view(tr.sample_alpha);
    Kokkos::deep_copy(h_alpha, tr.sample_alpha);
    auto h_rho     = Kokkos::create_mirror_view(tr.sample_rho);
    Kokkos::deep_copy(h_rho, tr.sample_rho);
    auto h_temp    = Kokkos::create_mirror_view(tr.sample_temp);
    Kokkos::deep_copy(h_temp, tr.sample_temp);

    int n_checked = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (owners[i].rank < 0) continue;
        INFO("particle " << i << " src=(" << h_positions[i][0] << ", "
             << h_positions[i][1] << ", " << h_positions[i][2] << ") owner="
             << owners[i].rank);

        // Sample fields landed correctly.
        REQUIRE_THAT(h_alpha(i),    Catch::Matchers::WithinAbs(1.0,           1e-12));
        REQUIRE_THAT(h_v(i, 0),     Catch::Matchers::WithinAbs(expected_vx(), 1e-12));
        REQUIRE_THAT(h_v(i, 1),     Catch::Matchers::WithinAbs(0.0,           1e-12));
        REQUIRE_THAT(h_v(i, 2),     Catch::Matchers::WithinAbs(0.0,           1e-12));
        REQUIRE_THAT(h_W(i),        Catch::Matchers::WithinAbs(expected_W(),  1e-12));
        REQUIRE_THAT(h_rho(i),      Catch::Matchers::WithinAbs(KNOWN_RHO,     1e-12));
        REQUIRE_THAT(h_temp(i),     Catch::Matchers::WithinAbs(KNOWN_TEMP,    1e-12));

        // Position pushed by exactly v_x (alpha=1, beta=0, dt=dtfact=1).
        const double expected_x = h_positions[i][0] + expected_vx();
        REQUIRE_THAT(h_dst_pos(i, 0), Catch::Matchers::WithinAbs(expected_x,        1e-12));
        REQUIRE_THAT(h_dst_pos(i, 1), Catch::Matchers::WithinAbs(h_positions[i][1], 1e-12));
        REQUIRE_THAT(h_dst_pos(i, 2), Catch::Matchers::WithinAbs(h_positions[i][2], 1e-12));
        ++n_checked;
    }
    REQUIRE(n_checked > 0);
}

#endif // GRACE_ENABLE_PARTICLES
