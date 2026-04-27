/**
 * @file particle_storage.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Tracer particle SoA storage built directly on Kokkos.
 *
 * v1 layout: one Kokkos::View per field. Per-particle access via
 * `slice<F>()(i)`. AoSoA migration is a Phase 5 optimization and would
 * preserve this API; for tracers SoA is fast enough (G2P touches ~3 fields
 * per particle, push touches ~5).
 *
 * Sort key (for the partition): ascending (owner_rank, owner_local_quad,
 * intra_quad_idx). p4est SFC ordering of quads makes this Morton order on
 * particle positions at the quad's level — see doc/design/particles.md.
 */
#ifndef GRACE_PARTICLES_PARTICLE_STORAGE_HH
#define GRACE_PARTICLES_PARTICLE_STORAGE_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>

namespace grace {
namespace particles {

/// Field-index enum. Order matches the slices declared on tracer_container_t.
enum class particle_field : int {
    // Identity / topology
    pos              = 0,
    id               = 1,
    status           = 2,
    owner_rank       = 3,
    owner_local_quad = 4,
    // Advection inputs (sampled every substep)
    sample_alpha     = 5,
    sample_beta      = 6,
    sample_v         = 7,
    sample_W         = 8,
    // Hydro samples (output)
    sample_rho       = 9,
    sample_temp      = 10,
    sample_ye        = 11,
    sample_entropy   = 12,
    sample_press     = 13,
    sample_eps       = 14,
    sample_B         = 15
};

/// Number of scalar-equivalent slots per particle (counting vector fields as 3).
/// Used to size the per-substep aux fetch response payload.
constexpr int n_tracer_sample_scalars =
    1 /*alpha*/ + 3 /*beta*/ + 3 /*v*/ + 1 /*W*/
    + 1 /*rho*/ + 1 /*temp*/ + 1 /*ye*/ + 1 /*entropy*/
    + 1 /*press*/ + 1 /*eps*/ + 3 /*B*/;
static_assert(n_tracer_sample_scalars == 16,
              "Update aux-fetch N_FIELDS instantiation if this changes.");

/// SoA tracer container. Trivially resizable; lifetime managed by caller.
template <class MemorySpace = grace::default_space>
class tracer_container_t {
  public:
    // Identity / topology
    Kokkos::View<double*[3], MemorySpace> pos;
    Kokkos::View<uint64_t*,  MemorySpace> id;
    Kokkos::View<uint8_t*,   MemorySpace> status;
    Kokkos::View<int32_t*,   MemorySpace> owner_rank;
    Kokkos::View<int32_t*,   MemorySpace> owner_local_quad;
    // Advection inputs
    Kokkos::View<double*,    MemorySpace> sample_alpha;
    Kokkos::View<double*[3], MemorySpace> sample_beta;
    Kokkos::View<double*[3], MemorySpace> sample_v;
    Kokkos::View<double*,    MemorySpace> sample_W;
    // Hydro samples
    Kokkos::View<double*,    MemorySpace> sample_rho;
    Kokkos::View<double*,    MemorySpace> sample_temp;
    Kokkos::View<double*,    MemorySpace> sample_ye;
    Kokkos::View<double*,    MemorySpace> sample_entropy;
    Kokkos::View<double*,    MemorySpace> sample_press;
    Kokkos::View<double*,    MemorySpace> sample_eps;
    Kokkos::View<double*[3], MemorySpace> sample_B;

    tracer_container_t() = default;

    /// Resize all fields to n elements. Initializes to zero.
    void resize(std::size_t n) {
        _n = n;
        pos              = decltype(pos)             ("tracer_pos",          n);
        id               = decltype(id)              ("tracer_id",           n);
        status           = decltype(status)          ("tracer_status",       n);
        owner_rank       = decltype(owner_rank)      ("tracer_owner_rank",   n);
        owner_local_quad = decltype(owner_local_quad)("tracer_owner_quad",   n);
        sample_alpha     = decltype(sample_alpha)    ("tracer_alpha",        n);
        sample_beta      = decltype(sample_beta)     ("tracer_beta",         n);
        sample_v         = decltype(sample_v)        ("tracer_v",            n);
        sample_W         = decltype(sample_W)        ("tracer_W",            n);
        sample_rho       = decltype(sample_rho)      ("tracer_rho",          n);
        sample_temp      = decltype(sample_temp)     ("tracer_temp",         n);
        sample_ye        = decltype(sample_ye)       ("tracer_ye",           n);
        sample_entropy   = decltype(sample_entropy)  ("tracer_entropy",      n);
        sample_press     = decltype(sample_press)    ("tracer_press",        n);
        sample_eps       = decltype(sample_eps)      ("tracer_eps",          n);
        sample_B         = decltype(sample_B)        ("tracer_B",            n);
    }

    std::size_t size() const noexcept { return _n; }

  private:
    std::size_t _n = 0;
};

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_STORAGE_HH
