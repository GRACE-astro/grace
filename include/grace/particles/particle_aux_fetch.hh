/**
 * @file particle_aux_fetch.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Per-substep aux-fetch protocol via distribution_plan_t.
 *
 * Mechanism: forward distributor ships request POD tuples
 * (orig_rank, orig_idx, owner_local_quad, position) to fluid owners; owners
 * run trilinear interp on the requested cell-centered fields; reverse
 *
 * Templated on the field count so callers can specialize per particle
 * species (v1 tracers use n_tracer_sample_scalars from particle_storage.hh).
 */
#ifndef GRACE_PARTICLES_PARTICLE_AUX_FETCH_HH
#define GRACE_PARTICLES_PARTICLE_AUX_FETCH_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <mpi.h>

namespace grace {
namespace particles {

/// Source for a fetched scalar cell-centered field.
///
/// STATE/AUX read directly from the named state or aux variable index.
/// DERIVED_V_X/Y/Z and DERIVED_W are GRMHD-specific: the owner computes the
/// 3-velocity and Lorentz factor from interpolated Z^i and γ_ij in a single
/// per-request pass (cached across the four derived outputs). var_idx is
/// ignored for DERIVED_*.
enum class field_source : uint8_t {
    STATE        = 0,
    AUX          = 1,
    DERIVED_V_X  = 2,
    DERIVED_V_Y  = 3,
    DERIVED_V_Z  = 4,
    DERIVED_W    = 5,
};

struct fetch_field_spec_t {
    field_source source;
    int          var_idx; // ignored for DERIVED_*
};

/// Fetch N_FIELDS scalar cell-centered fields at each particle's position via
/// trilinear interpolation on the owning fluid rank.
///
/// All input/output views must be allocated on grace::default_space. Particles
/// with `owner_rank < 0` are skipped silently — `out` rows are left untouched
/// (caller should pre-zero if desired).
///
/// `fields[].source` selects the source array on the owner side (STATE or AUX);
/// `fields[].var_idx` is the variable index within that array.
///
/// Output `out` has shape (n_particles, N_FIELDS); row `i` receives the
/// trilinearly-interpolated values for particle `i` in the same order as
/// `fields`.
template <int N_FIELDS>
void fetch_at_positions(
    MPI_Comm comm,
    std::size_t n_particles,
    const std::array<fetch_field_spec_t, N_FIELDS>& fields,
    Kokkos::View<int*,         grace::default_space> owner_rank,
    Kokkos::View<int*,         grace::default_space> owner_local_quad,
    Kokkos::View<double*[3],   grace::default_space> positions,
    Kokkos::View<double**,     grace::default_space> out);

} // namespace particles
} // namespace grace

#include "particle_aux_fetch.tpp"

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_AUX_FETCH_HH
