/**
 * @file particle_compact.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Local-only container hygiene: compact dead tracers, append new ones.
 *
 * compact() drops every tracer whose status != PARTICLE_DEFAULT (handles
 * PARTICLE_OUTSIDE_DOMAIN, PARTICLE_INSIDE_BH, and any future cull flag
 * uniformly). It is local — no MPI — so cadence can be set independently
 * of rebalance.
 *
 * append() extends the container, drawing fresh ids from the per-rank
 * monotonic counter. The (rank << 32) | local_idx encoding ensures appended
 * tracers cannot collide with any previously-existing id, including dead
 * ids that were just compacted away.
 *
 * Sample fields on appended tracers are zero-initialised; they get filled
 * on the next advance_substep fetch.
 */
#ifndef GRACE_PARTICLES_PARTICLE_COMPACT_HH
#define GRACE_PARTICLES_PARTICLE_COMPACT_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_storage.hh>
#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace grace {
namespace particles {

/// Drop every tracer whose status != PARTICLE_DEFAULT. Returns the number of
/// tracers culled. Fields are reallocated to the survivor count.
std::size_t compact(tracer_container_t<>& tr);

/// Append n_new tracers at the given positions. Fresh ids are drawn from the
/// per-rank monotonic counter (rank<<32 | next_local_id..). Status defaults
/// to PARTICLE_DEFAULT. owner_rank is set to the supplied rank; owner_local_quad
/// is left -1 — the caller (or the next rebalance) is expected to resolve it.
///
/// new_positions must be a host-space Kokkos::View<double*[3]>.
void append(tracer_container_t<>&                      tr,
            int                                        rank,
            Kokkos::View<double*[3], Kokkos::HostSpace> new_positions);

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_COMPACT_HH
