/**
 * @file particle_checkpoint.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Save / restore the singleton tracer container from a GRACE
 *        HDF5 checkpoint.
 *
 * Storage layout under group `/particles/` is partition-agnostic — the
 * file holds global datasets sized at N_global, plus a length-nproc_at_save
 * vector of per-birth-rank id counters. This means a checkpoint can be
 * restored under a different mpi_size without any post-processing: each
 * rank reads an equal slice of the global data, then the post-restore
 * rebalance migrates each tracer to the rank that currently owns its
 * containing quad.
 *
 * Sample fields are NOT saved — they are recomputable from the fluid state,
 * which the checkpoint already persists, and the first advance_step after
 * restart fetches them.
 */
#ifndef GRACE_PARTICLES_PARTICLE_CHECKPOINT_HH
#define GRACE_PARTICLES_PARTICLE_CHECKPOINT_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <hdf5.h>

namespace grace {
namespace particles {

/// Collective write of the singleton tracer container under `/particles/`
/// in the open checkpoint file. Does nothing (writes an empty group with
/// `n_global = 0`) if the module is disabled or empty, so the reader can
/// detect "no particles in this checkpoint" cheaply.
void save_particles_to_checkpoint(hid_t file_id, hid_t dxpl);

/// Collective read into the singleton tracer container. Each rank reads an
/// equal slice of the global data, restores the per-rank id counter (with
/// a high32-id scan fallback if mpi_size changed since save), then forces
/// an immediate rebalance to settle ownership under the current partition.
/// Returns true if a `/particles/` group was found and loaded; false if the
/// checkpoint predates the particle subsystem (caller should seed fresh).
/// No-op returning false if the module is disabled.
bool load_particles_from_checkpoint(hid_t file_id);

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_CHECKPOINT_HH
