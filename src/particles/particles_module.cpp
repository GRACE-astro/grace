/**
 * @file particles_module.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Top-level particle subsystem entry point.
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particles_module.hh>
#include <grace/particles/particle_storage.hh>
#include <grace/particles/particle_advance.hh>
#include <grace/particles/particle_owner_search.hh>

#include <grace/config/config_parser.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/runtime_functions.hh>
#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h>

namespace grace {
namespace particles {

class particles_module_impl_t {
  public:
    bool                 enabled               = false;
    int                  refresh_owners_every  = 1;
    int                  output_every          = 0;
    std::string          output_directory;
    std::string          output_basename;
    tracer_container_t<> tracers;
};

particles_module_t::particles_module_t()
  : _impl(new particles_module_impl_t())
{}

particles_module_t::~particles_module_t() {
    delete _impl;
}

particles_module_t& particles_module_t::get() {
    static particles_module_t instance;
    return instance;
}

namespace {

/// Deterministic per-rank seed: spread N tracers over the local quads,
/// using a low-discrepancy 3D offset inside each quad's bbox.
/// owner_rank/owner_local_quad are set so the first advance_step runs
/// without needing an immediate refresh.
void seed_local(tracer_container_t<>& tr, std::size_t n_per_rank) {
    auto& sh = fluid_topology_shadow_t::get();
    sh.refresh();
    const auto& geom = sh.local_geometry();
    if (geom.empty() || n_per_rank == 0) {
        tr.resize(0);
        return;
    }
    const int rank = parallel::mpi_comm_rank();

    tr.resize(n_per_rank);
    auto h_pos        = Kokkos::create_mirror_view(tr.pos);
    auto h_id         = Kokkos::create_mirror_view(tr.id);
    auto h_status     = Kokkos::create_mirror_view(tr.status);
    auto h_owner_rank = Kokkos::create_mirror_view(tr.owner_rank);
    auto h_owner_quad = Kokkos::create_mirror_view(tr.owner_local_quad);

    // Halton-ish offsets in (0,1) along each axis. Avoid edges so we don't
    // land on a quad face and risk dual-ownership ambiguity.
    auto halton = [](std::size_t i, int base) {
        double f = 1.0;
        double r = 0.0;
        std::size_t k = i + 1;
        while (k > 0) {
            f /= base;
            r += f * (k % base);
            k /= base;
        }
        return r;
    };

    for (std::size_t i = 0; i < n_per_rank; ++i) {
        const std::size_t q = i % geom.size();
        const std::size_t s = i / geom.size();
        const auto& g = geom[q];
        const double fx = halton(s, 2);
        const double fy = halton(s, 3);
        const double fz = halton(s, 5);
        h_pos(i, 0) = g.bbox.xlo + fx * (g.bbox.xhi - g.bbox.xlo);
        h_pos(i, 1) = g.bbox.ylo + fy * (g.bbox.yhi - g.bbox.ylo);
        h_pos(i, 2) = g.bbox.zlo + fz * (g.bbox.zhi - g.bbox.zlo);
        // Globally-unique id: (rank << 32) | local_index. Cheap, monotonic,
        // collision-free across ranks at this scale.
        h_id(i)         = (static_cast<uint64_t>(rank) << 32) | static_cast<uint32_t>(i);
        h_status(i)     = 0;
        h_owner_rank(i) = rank;
        h_owner_quad(i) = static_cast<int32_t>(q);
    }
    Kokkos::deep_copy(tr.pos,              h_pos);
    Kokkos::deep_copy(tr.id,               h_id);
    Kokkos::deep_copy(tr.status,           h_status);
    Kokkos::deep_copy(tr.owner_rank,       h_owner_rank);
    Kokkos::deep_copy(tr.owner_local_quad, h_owner_quad);
}

void refresh_owners(tracer_container_t<>& tr) {
    if (tr.size() == 0) return;
    auto& sh = fluid_topology_shadow_t::get();
    auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.pos);

    std::vector<std::array<double, 3>> positions(tr.size());
    for (std::size_t i = 0; i < tr.size(); ++i) {
        positions[i] = {h_pos(i, 0), h_pos(i, 1), h_pos(i, 2)};
    }
    std::vector<owner_t> owners;
    sh.find_owners_batch(positions, owners);

    auto h_owner_rank = Kokkos::create_mirror_view(tr.owner_rank);
    auto h_owner_quad = Kokkos::create_mirror_view(tr.owner_local_quad);
    for (std::size_t i = 0; i < tr.size(); ++i) {
        h_owner_rank(i) = owners[i].rank;
        h_owner_quad(i) = owners[i].local_quad;
    }
    Kokkos::deep_copy(tr.owner_rank,       h_owner_rank);
    Kokkos::deep_copy(tr.owner_local_quad, h_owner_quad);
}

void dump_text(const tracer_container_t<>& tr,
               const std::string& dir,
               const std::string& basename)
{
    if (tr.size() == 0) return;
    const int    rank = parallel::mpi_comm_rank();
    const size_t iter = grace::get_iteration();
    const double t    = grace::get_simulation_time();

    if (rank == 0) std::filesystem::create_directories(dir);
    MPI_Barrier(MPI_COMM_WORLD);

    auto h_pos    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.pos);
    auto h_id     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.id);
    auto h_v      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_v);
    auto h_W      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_W);
    auto h_alpha  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_alpha);
    auto h_rho    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_rho);
    auto h_temp   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_temp);
    auto h_ye     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_ye);

    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%s.%05d.txt",
                  dir.c_str(), basename.c_str(), rank);
    const bool fresh = !std::filesystem::exists(path);
    std::ofstream f(path, std::ios::app);
    if (fresh) {
        f << "# iter t id x y z vx vy vz W alpha rho temp ye\n";
    }
    f.precision(12);
    for (std::size_t i = 0; i < tr.size(); ++i) {
        f << iter << ' ' << t << ' ' << h_id(i) << ' '
          << h_pos(i,0) << ' ' << h_pos(i,1) << ' ' << h_pos(i,2) << ' '
          << h_v(i,0)   << ' ' << h_v(i,1)   << ' ' << h_v(i,2)   << ' '
          << h_W(i)     << ' ' << h_alpha(i) << ' '
          << h_rho(i)   << ' ' << h_temp(i)  << ' ' << h_ye(i)
          << '\n';
    }
}

} // namespace

void particles_module_t::initialize() {
    _impl->enabled = grace::get_param<bool>("particles", "enabled");
    if (!_impl->enabled) {
        GRACE_INFO("Particle subsystem disabled by config.");
        return;
    }
    _impl->refresh_owners_every =
        grace::get_param<int>("particles", "refresh_owners_every");
    _impl->output_every =
        grace::get_param<int>("particles", "output_every");
    _impl->output_directory =
        grace::get_param<std::string>("particles", "output_directory");
    _impl->output_basename =
        grace::get_param<std::string>("particles", "output_basename");

    const int n_per_rank =
        grace::get_param<int>("particles", "n_tracers_per_rank");
    seed_local(_impl->tracers, static_cast<std::size_t>(n_per_rank));

    GRACE_INFO("Particle subsystem enabled: {} tracers/rank seeded.",
               _impl->tracers.size());
}

void particles_module_t::finalize() {
    _impl->tracers = tracer_container_t<>{};
    _impl->enabled = false;
}

bool particles_module_t::enabled() const noexcept {
    return _impl->enabled;
}

std::size_t particles_module_t::local_count() const noexcept {
    return _impl->tracers.size();
}

tracer_container_t<>& particles_module_t::tracers() noexcept {
    return _impl->tracers;
}

const tracer_container_t<>& particles_module_t::tracers() const noexcept {
    return _impl->tracers;
}

void particles_module_t::advance_step(double dt) {
    if (!_impl->enabled) return;
    auto& tr = _impl->tracers;
    if (tr.size() == 0) return;

    const size_t iter = grace::get_iteration();

    // Owner refresh stands in for migration (Phase 2a). Without it, a tracer
    // that drifts into another rank's quad keeps fetching from its old owner
    // — which no longer contains its position — and silently gets garbage.
    if (_impl->refresh_owners_every > 0 &&
        iter % static_cast<size_t>(_impl->refresh_owners_every) == 0) {
        refresh_owners(tr);
    }

    advance_substep(MPI_COMM_WORLD, dt, /*dtfact=*/1.0, tr.pos, tr.pos, tr);

    if (_impl->output_every > 0 &&
        iter % static_cast<size_t>(_impl->output_every) == 0) {
        dump_text(tr, _impl->output_directory, _impl->output_basename);
    }
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
