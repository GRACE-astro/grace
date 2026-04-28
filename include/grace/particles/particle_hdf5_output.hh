/**
 * @file particle_hdf5_output.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Collective MPI-IO HDF5 snapshot writer for tracers + XDMF sidecar.
 *
 * Output model: one HDF5 file per snapshot (`<basename>_iterNNNNNNNN.h5`),
 * written collectively by all ranks; particles' ranks contribute their
 * local slice via per-rank hyperslabs into global datasets sized at
 * N_global (MPI_Exscan to compute offsets).
 *
 * A single master XDMF (`<basename>.xmf`) is rewritten on every snapshot,
 * accumulating a Temporal Collection of Polyvertex grids — ParaView opens
 * it as an animated point cloud. Each snapshot exposes `id` so the
 * "Temporal Particles To Pathlines" filter can join across timesteps to
 * render full trajectories as polylines.
 */
#ifndef GRACE_PARTICLES_PARTICLE_HDF5_OUTPUT_HH
#define GRACE_PARTICLES_PARTICLE_HDF5_OUTPUT_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_storage.hh>

#include <string>

namespace grace {
namespace particles {

/// Write one collective HDF5 snapshot of `tr` and update the master XMF.
/// Rank 0 creates `dir` if needed; all ranks participate in the collective
/// write. No-op if every rank reports n_local == 0.
void write_particle_snapshot(const tracer_container_t<>& tr,
                             const std::string& dir,
                             const std::string& basename);

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_HDF5_OUTPUT_HH
