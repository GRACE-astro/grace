/**
 * @file particle_advance.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Per-substep particle advance (G2P sample + RK push).
 *
 * Calls into the aux-fetch protocol with the v1 tracer field set, splays the
 * fetched values into the tracer container's per-field sample slices, and
 * applies the position update for one RK substep:
 *     dst_pos = src_pos + dtfact * dt * (alpha * v - beta).
 *
 * v1 supports a single-stage advance per call (Forward Euler when called
 * once per step). Higher-order RK consistency = call once per substep with
 * the matching dtfact and src/dst pos selectors. Because the fluid sample
 * fields are computed at src and used in all subsequent substages, the
 * tracer fast-path stays correct for SSP-RK schemes.
 */
#ifndef GRACE_PARTICLES_PARTICLE_ADVANCE_HH
#define GRACE_PARTICLES_PARTICLE_ADVANCE_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_aux_fetch.hh>
#include <grace/particles/particle_storage.hh>
#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <mpi.h>

namespace grace {
namespace particles {

/// Order: alpha, beta[3], v[3], W, rho, temp, ye, entropy, press, eps, B[3].
/// Index map matches splay_into_samples() below; do not reorder without
/// updating both.
constexpr std::array<fetch_field_spec_t, 17> tracer_v1_field_specs() {
    return {{
        {field_source::STATE,       /*ALP_*/    -1}, // resolved below
        {field_source::STATE,       /*BETAX_*/  -1},
        {field_source::STATE,       /*BETAY_*/  -1},
        {field_source::STATE,       /*BETAZ_*/  -1},
        {field_source::DERIVED_V_X, 0},
        {field_source::DERIVED_V_Y, 0},
        {field_source::DERIVED_V_Z, 0},
        {field_source::DERIVED_W,   0},
        {field_source::AUX,         /*RHO_*/    -1},
        {field_source::AUX,         /*TEMP_*/   -1},
        {field_source::AUX,         /*YE_*/     -1},
        {field_source::AUX,         /*ENTROPY_*/-1},
        {field_source::AUX,         /*PRESS_*/  -1},
        {field_source::AUX,         /*EPS_*/    -1},
        {field_source::AUX,         /*BX_*/     -1},
        {field_source::AUX,         /*BY_*/     -1},
        {field_source::AUX,         /*BZ_*/     -1},
    }};
}

/// Fill the var_idx slots in the spec list at runtime. Called once at module
/// init (the indices depend on enabled code modules so they are not
/// constexpr).
std::array<fetch_field_spec_t, 17> tracer_v1_field_specs_resolved();

/// Splay a fetched (n_particles × 17) view into the tracer container's
/// per-field sample slices, in the order declared above.
template <class MemorySpace>
void splay_into_samples(
    tracer_container_t<MemorySpace>& tr,
    Kokkos::View<double**, MemorySpace> fetched);

/// One RK substep advance for tracers.
///   - Fetches the v1 field set at src_pos using fluid SRC state via the
///     aux-fetch protocol.
///   - Splays values into the tracer container's sample slices.
///   - Updates dst_pos = src_pos + dtfact * dt * (alpha * v - beta).
///
/// src_pos and dst_pos may alias (Forward Euler in-place is OK when only one
/// stage is needed). The other tracer-container slices are read/written
/// in-place; they hold "current state" semantically.
void advance_substep(
    MPI_Comm comm,
    double dt,
    double dtfact,
    Kokkos::View<double*[3], grace::default_space> src_pos,
    Kokkos::View<double*[3], grace::default_space> dst_pos,
    tracer_container_t<grace::default_space>&      tr);

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_ADVANCE_HH
