/**
 * @file particle_rebalance.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Particle rebalancing: strategy + migration mechanism.
 *
 * Two concerns are split here:
 *
 *   1. Strategy. Given the current container state and the fluid topology,
 *      decide where each particle SHOULD live. Output is a per-particle
 *      destination-rank view (-1 to drop). Today we ship one strategy
 *      (quad_owner: send each particle to the rank owning the quad that
 *      contains its position). Future strategies — SFC-equal-count, weighted
 *      by per-quad fluid cost, PIC field-locality — slot in as additional
 *      compute_export_ranks_* functions returning the same shape.
 *
 *   2. Mechanism. migrate_topology() takes any export-ranks view and routes
 *      pos/id/status to their new homes via distribution_plan_t. Sample
 *      fields are NOT migrated — they get refilled by the next advance_substep
 *      fetch — so each rebalance only moves O(40 B/particle).
 *
 * Post-migration, owner_rank == self by construction, and owner_local_quad
 * is re-resolved from positions via fluid_topology_shadow_t. With non-quad-owner
 * strategies, owner_local_quad may be -1 (position not owned by self), and
 * the fetch protocol falls back to the partition-search slow path.
 */
#ifndef GRACE_PARTICLES_PARTICLE_REBALANCE_HH
#define GRACE_PARTICLES_PARTICLE_REBALANCE_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_storage.hh>
#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace grace {
namespace particles {

/// Strategy: send each tracer to the rank owning the quad that contains its
/// position. Tracers whose position falls outside the global domain get
/// destination -1 (the migration mechanism drops them; caller may also flag
/// them PARTICLE_OUTSIDE_DOMAIN before migration to surface them in output).
///
/// Implementation: host-side find_owners_batch on the position list, then
/// copies the resulting destination ranks back to device. Cheap for handfuls
/// of tracers; will need a device-side path before this scales.
Kokkos::View<int*, grace::default_space>
compute_export_ranks_quad_owner(const tracer_container_t<>& tr);

/// Migrate pos/id/status to new owners per the supplied export_ranks view.
/// Resizes `tr` in place to the post-migration count. After return:
///   - owner_rank(i) == this rank   for every surviving tracer
///   - owner_local_quad(i) is re-resolved from position; -1 if the local
///     fluid topology does not contain the tracer's position (only possible
///     under non-quad-owner strategies).
///   - sample_* are zero-initialised; the next advance_substep fetch fills
///     them.
void migrate_topology(MPI_Comm                                       comm,
                      tracer_container_t<>&                          tr,
                      Kokkos::View<int*, grace::default_space>       export_ranks);

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_REBALANCE_HH
