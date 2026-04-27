/**
 * @file particle_checkpoint.cpp
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_checkpoint.hh>
#include <grace/particles/particles_module.hh>
#include <grace/particles/particle_storage.hh>
#include <grace/particles/particle_rebalance.hh>
#include <grace/particles/particle_owner_search.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/print.hh>
#include <grace/errors/error.hh>

#include <Kokkos_Core.hpp>
#include <hdf5.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#define HDF5_CALL(result, cmd)                                          \
    do {                                                                \
        if ((result = cmd) < 0) {                                       \
            ERROR("HDF5 API call failed with error code " << result);   \
        }                                                               \
    } while (false)

namespace grace {
namespace particles {

namespace {

/// Equal-slice partition for an arbitrary global count. Rank r owns
/// [r*N/P, (r+1)*N/P) — the standard "extras at the front" scheme.
struct slice_t { hsize_t offset; hsize_t count; };
slice_t equal_slice(hsize_t n_global, int rank, int nproc) {
    const hsize_t base   = n_global / static_cast<hsize_t>(nproc);
    const hsize_t extras = n_global % static_cast<hsize_t>(nproc);
    const hsize_t r      = static_cast<hsize_t>(rank);
    const hsize_t off    = r * base + std::min(r, extras);
    const hsize_t cnt    = base + (r < extras ? 1 : 0);
    return {off, cnt};
}

template <typename T>
void write_dataset_1d(hid_t file_id, hid_t dxpl, const char* name,
                      hid_t h5type, hsize_t n_global,
                      hsize_t local_offset, hsize_t n_local,
                      const T* host_data)
{
    herr_t  err;
    hid_t   global_space, local_space, dset_id;
    hsize_t gdims[1] = {n_global};
    hsize_t ldims[1] = {n_local};
    HDF5_CALL(global_space, H5Screate_simple(1, gdims, NULL));
    HDF5_CALL(local_space,  H5Screate_simple(1, ldims, NULL));
    HDF5_CALL(dset_id, H5Dcreate2(file_id, name, h5type, global_space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    if (n_local > 0) {
        hsize_t start[1] = {local_offset};
        hsize_t count[1] = {n_local};
        HDF5_CALL(err, H5Sselect_hyperslab(global_space, H5S_SELECT_SET,
                                           start, NULL, count, NULL));
    } else {
        HDF5_CALL(err, H5Sselect_none(global_space));
        HDF5_CALL(err, H5Sselect_none(local_space));
    }
    HDF5_CALL(err, H5Dwrite(dset_id, h5type, local_space, global_space,
                            dxpl, host_data));
    HDF5_CALL(err, H5Dclose(dset_id));
    HDF5_CALL(err, H5Sclose(local_space));
    HDF5_CALL(err, H5Sclose(global_space));
}

template <typename T>
void write_dataset_2d(hid_t file_id, hid_t dxpl, const char* name,
                      hid_t h5type, hsize_t n_global, hsize_t inner,
                      hsize_t local_offset, hsize_t n_local,
                      const T* host_data)
{
    herr_t  err;
    hid_t   global_space, local_space, dset_id;
    hsize_t gdims[2] = {n_global, inner};
    hsize_t ldims[2] = {n_local,  inner};
    HDF5_CALL(global_space, H5Screate_simple(2, gdims, NULL));
    HDF5_CALL(local_space,  H5Screate_simple(2, ldims, NULL));
    HDF5_CALL(dset_id, H5Dcreate2(file_id, name, h5type, global_space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    if (n_local > 0) {
        hsize_t start[2] = {local_offset, 0};
        hsize_t count[2] = {n_local,      inner};
        HDF5_CALL(err, H5Sselect_hyperslab(global_space, H5S_SELECT_SET,
                                           start, NULL, count, NULL));
    } else {
        HDF5_CALL(err, H5Sselect_none(global_space));
        HDF5_CALL(err, H5Sselect_none(local_space));
    }
    HDF5_CALL(err, H5Dwrite(dset_id, h5type, local_space, global_space,
                            dxpl, host_data));
    HDF5_CALL(err, H5Dclose(dset_id));
    HDF5_CALL(err, H5Sclose(local_space));
    HDF5_CALL(err, H5Sclose(global_space));
}

template <typename T>
void read_dataset_1d(hid_t file_id, hid_t dxpl, const char* name,
                     hid_t h5type, hsize_t local_offset, hsize_t n_local,
                     T* host_data)
{
    herr_t err;
    hid_t  dset_id, global_space, local_space;
    HDF5_CALL(dset_id,      H5Dopen2(file_id, name, H5P_DEFAULT));
    HDF5_CALL(global_space, H5Dget_space(dset_id));
    hsize_t ldims[1] = {n_local};
    HDF5_CALL(local_space,  H5Screate_simple(1, ldims, NULL));
    if (n_local > 0) {
        hsize_t start[1] = {local_offset};
        hsize_t count[1] = {n_local};
        HDF5_CALL(err, H5Sselect_hyperslab(global_space, H5S_SELECT_SET,
                                           start, NULL, count, NULL));
    } else {
        HDF5_CALL(err, H5Sselect_none(global_space));
        HDF5_CALL(err, H5Sselect_none(local_space));
    }
    HDF5_CALL(err, H5Dread(dset_id, h5type, local_space, global_space,
                           dxpl, host_data));
    HDF5_CALL(err, H5Dclose(dset_id));
    HDF5_CALL(err, H5Sclose(local_space));
    HDF5_CALL(err, H5Sclose(global_space));
}

template <typename T>
void read_dataset_2d(hid_t file_id, hid_t dxpl, const char* name,
                     hid_t h5type, hsize_t inner,
                     hsize_t local_offset, hsize_t n_local,
                     T* host_data)
{
    herr_t err;
    hid_t  dset_id, global_space, local_space;
    HDF5_CALL(dset_id,      H5Dopen2(file_id, name, H5P_DEFAULT));
    HDF5_CALL(global_space, H5Dget_space(dset_id));
    hsize_t ldims[2] = {n_local, inner};
    HDF5_CALL(local_space,  H5Screate_simple(2, ldims, NULL));
    if (n_local > 0) {
        hsize_t start[2] = {local_offset, 0};
        hsize_t count[2] = {n_local,      inner};
        HDF5_CALL(err, H5Sselect_hyperslab(global_space, H5S_SELECT_SET,
                                           start, NULL, count, NULL));
    } else {
        HDF5_CALL(err, H5Sselect_none(global_space));
        HDF5_CALL(err, H5Sselect_none(local_space));
    }
    HDF5_CALL(err, H5Dread(dset_id, h5type, local_space, global_space,
                           dxpl, host_data));
    HDF5_CALL(err, H5Dclose(dset_id));
    HDF5_CALL(err, H5Sclose(local_space));
    HDF5_CALL(err, H5Sclose(global_space));
}

} // namespace

void save_particles_to_checkpoint(hid_t file_id, hid_t dxpl) {
    auto& mod = particles_module_t::get();
    auto& tr  = mod.tracers();
    const int rank  = parallel::mpi_comm_rank();
    const int nproc = parallel::mpi_comm_size();
    const auto comm = parallel::get_comm_world();

    const long long n_local_ll = mod.enabled() ? static_cast<long long>(tr.size()) : 0LL;
    long long offset_ll  = 0;
    long long n_global_ll = 0;
    MPI_Exscan   (&n_local_ll, &offset_ll,   1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(&n_local_ll, &n_global_ll, 1, MPI_LONG_LONG, MPI_SUM, comm);
    if (rank == 0) offset_ll = 0;

    herr_t err;
    hid_t  grp_id;
    HDF5_CALL(grp_id, H5Gcreate2(file_id, "particles", H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT));

    // Global count + provenance attributes.
    {
        hid_t ds, attr;
        HDF5_CALL(ds,   H5Screate(H5S_SCALAR));
        HDF5_CALL(attr, H5Acreate2(grp_id, "n_global", H5T_NATIVE_INT64, ds,
                                   H5P_DEFAULT, H5P_DEFAULT));
        HDF5_CALL(err,  H5Awrite(attr, H5T_NATIVE_INT64, &n_global_ll));
        HDF5_CALL(err,  H5Aclose(attr));
        HDF5_CALL(err,  H5Sclose(ds));
    }
    {
        hid_t ds, attr;
        HDF5_CALL(ds,   H5Screate(H5S_SCALAR));
        HDF5_CALL(attr, H5Acreate2(grp_id, "nproc_at_save", H5T_NATIVE_INT, ds,
                                   H5P_DEFAULT, H5P_DEFAULT));
        HDF5_CALL(err,  H5Awrite(attr, H5T_NATIVE_INT, &nproc));
        HDF5_CALL(err,  H5Aclose(attr));
        HDF5_CALL(err,  H5Sclose(ds));
    }

    if (n_global_ll == 0) {
        HDF5_CALL(err, H5Gclose(grp_id));
        return;
    }

    // Mirror the persistent fields (no samples).
    auto h_pos    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.pos);
    auto h_id     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.id);
    auto h_status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.status);

    write_dataset_2d(grp_id, dxpl, "Position", H5T_NATIVE_DOUBLE, 3,
                     static_cast<hsize_t>(n_global_ll),
                     static_cast<hsize_t>(offset_ll),
                     static_cast<hsize_t>(n_local_ll), h_pos.data());
    write_dataset_1d(grp_id, dxpl, "id", H5T_NATIVE_UINT64,
                     static_cast<hsize_t>(n_global_ll),
                     static_cast<hsize_t>(offset_ll),
                     static_cast<hsize_t>(n_local_ll), h_id.data());
    write_dataset_1d(grp_id, dxpl, "status", H5T_NATIVE_UINT8,
                     static_cast<hsize_t>(n_global_ll),
                     static_cast<hsize_t>(offset_ll),
                     static_cast<hsize_t>(n_local_ll), h_status.data());

    // Per-rank id counter, gathered to one rank-indexed dataset. Allows
    // exact restore when mpi_size is unchanged; the loader has a fallback
    // for the changed-mpi_size case.
    {
        const uint32_t my_counter = tr.id_counter();
        std::vector<uint32_t> counters_all(static_cast<std::size_t>(nproc), 0);
        MPI_Gather(&my_counter, 1, MPI_UINT32_T,
                   counters_all.data(), 1, MPI_UINT32_T, 0, comm);
        // Only rank 0 needs to write — but collective HDF5 wants every rank
        // to participate. Write as a small length-nproc dataset; rank 0
        // contributes the data, others select-none.
        herr_t err2;
        hid_t  ds, dset, lspace;
        hsize_t gdims[1] = {static_cast<hsize_t>(nproc)};
        HDF5_CALL(ds,    H5Screate_simple(1, gdims, NULL));
        HDF5_CALL(dset,  H5Dcreate2(grp_id, "next_local_id_per_rank",
                                    H5T_NATIVE_UINT32, ds, H5P_DEFAULT,
                                    H5P_DEFAULT, H5P_DEFAULT));
        if (rank == 0) {
            HDF5_CALL(lspace, H5Screate_simple(1, gdims, NULL));
            HDF5_CALL(err2,   H5Dwrite(dset, H5T_NATIVE_UINT32, lspace, ds,
                                       dxpl, counters_all.data()));
            HDF5_CALL(err2,   H5Sclose(lspace));
        } else {
            HDF5_CALL(err2,   H5Sselect_none(ds));
            hsize_t zero[1] = {0};
            HDF5_CALL(lspace, H5Screate_simple(1, zero, NULL));
            HDF5_CALL(err2,   H5Sselect_none(lspace));
            HDF5_CALL(err2,   H5Dwrite(dset, H5T_NATIVE_UINT32, lspace, ds,
                                       dxpl, counters_all.data()));
            HDF5_CALL(err2,   H5Sclose(lspace));
        }
        HDF5_CALL(err2, H5Dclose(dset));
        HDF5_CALL(err2, H5Sclose(ds));
    }

    HDF5_CALL(err, H5Gclose(grp_id));
}

bool load_particles_from_checkpoint(hid_t file_id) {
    auto& mod = particles_module_t::get();
    if (!mod.enabled()) return false;
    auto& tr  = mod.tracers();
    const int rank  = parallel::mpi_comm_rank();
    const int nproc = parallel::mpi_comm_size();
    const auto comm = parallel::get_comm_world();

    // Older checkpoints (pre-particles) have no /particles group; caller
    // should fall back to seeding fresh.
    htri_t has_grp;
    HDF5_CALL(has_grp, H5Lexists(file_id, "particles", H5P_DEFAULT));
    if (!has_grp) return false;

    herr_t err;
    hid_t  grp_id;
    HDF5_CALL(grp_id, H5Gopen2(file_id, "particles", H5P_DEFAULT));

    int64_t n_global_ll  = 0;
    int     nproc_at_save = nproc;
    {
        hid_t attr, ds;
        HDF5_CALL(attr, H5Aopen(grp_id, "n_global", H5P_DEFAULT));
        HDF5_CALL(err,  H5Aread(attr, H5T_NATIVE_INT64, &n_global_ll));
        HDF5_CALL(err,  H5Aclose(attr));
        HDF5_CALL(attr, H5Aopen(grp_id, "nproc_at_save", H5P_DEFAULT));
        HDF5_CALL(err,  H5Aread(attr, H5T_NATIVE_INT, &nproc_at_save));
        HDF5_CALL(err,  H5Aclose(attr));
        (void)ds;
    }

    // Drop whatever the (just-init-seeded) tracers were and replace with
    // the checkpoint contents.
    if (n_global_ll == 0) {
        tr.resize(0);
        HDF5_CALL(err, H5Gclose(grp_id));
        return true;
    }

    // Equal slice for this rank.
    const slice_t s = equal_slice(static_cast<hsize_t>(n_global_ll), rank, nproc);
    tr.resize(static_cast<std::size_t>(s.count));

    auto h_pos    = Kokkos::create_mirror_view(tr.pos);
    auto h_id     = Kokkos::create_mirror_view(tr.id);
    auto h_status = Kokkos::create_mirror_view(tr.status);

    // Use a parallel dxpl matching the file's parallel access mode.
    hid_t dxpl;
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER));
    HDF5_CALL(err,  H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));

    read_dataset_2d(grp_id, dxpl, "Position", H5T_NATIVE_DOUBLE, 3,
                    s.offset, s.count, h_pos.data());
    read_dataset_1d(grp_id, dxpl, "id",       H5T_NATIVE_UINT64,
                    s.offset, s.count, h_id.data());
    read_dataset_1d(grp_id, dxpl, "status",   H5T_NATIVE_UINT8,
                    s.offset, s.count, h_status.data());

    // Restore id counter. Same-mpi_size: index by rank; different size:
    // scan loaded particles for the highest low32(id) tagged with this
    // rank's high32 and bump from there. No collisions either way.
    uint32_t restored_counter = 0;
    if (nproc == nproc_at_save) {
        std::vector<uint32_t> counters_all(static_cast<std::size_t>(nproc), 0);
        if (rank == 0) {
            hid_t dset;
            HDF5_CALL(dset, H5Dopen2(grp_id, "next_local_id_per_rank", H5P_DEFAULT));
            HDF5_CALL(err,  H5Dread(dset, H5T_NATIVE_UINT32, H5S_ALL, H5S_ALL,
                                    H5P_DEFAULT, counters_all.data()));
            HDF5_CALL(err,  H5Dclose(dset));
        }
        MPI_Scatter(counters_all.data(), 1, MPI_UINT32_T,
                    &restored_counter,   1, MPI_UINT32_T, 0, comm);
    } else {
        // Different mpi_size since save: scan loaded particles for the
        // highest low32(id) bearing our rank tag, Allreduce, and bump.
        // Note "max=0" is ambiguous (could mean "no particle with my tag"
        // or "the only particle has lo=0"), so we Allreduce a presence
        // flag separately.
        const uint32_t my_tag = static_cast<uint32_t>(rank);
        uint32_t local_max_lo  = 0;
        bool     any_with_tag  = false;
        for (std::size_t i = 0; i < s.count; ++i) {
            const uint64_t id = h_id(i);
            if (static_cast<uint32_t>(id >> 32) == my_tag) {
                any_with_tag = true;
                local_max_lo = std::max(local_max_lo,
                                        static_cast<uint32_t>(id & 0xFFFFFFFFull));
            }
        }
        uint32_t global_max_lo = 0;
        int      any_local     = any_with_tag ? 1 : 0;
        int      any_global    = 0;
        MPI_Allreduce(&local_max_lo, &global_max_lo, 1,
                      MPI_UINT32_T, MPI_MAX, comm);
        MPI_Allreduce(&any_local,    &any_global,    1,
                      MPI_INT,       MPI_LOR, comm);
        restored_counter = any_global ? global_max_lo + 1 : 0;
    }
    tr.set_id_counter(restored_counter);

    Kokkos::deep_copy(tr.pos,    h_pos);
    Kokkos::deep_copy(tr.id,     h_id);
    Kokkos::deep_copy(tr.status, h_status);

    // owner_rank/owner_local_quad and samples are intentionally not
    // persisted; the rebalance below resolves the first two from
    // positions, and the next advance_step refills samples.
    Kokkos::deep_copy(tr.owner_rank, -1);
    Kokkos::deep_copy(tr.owner_local_quad, -1);

    // Settle ownership under the current partition (which may differ from
    // the saved one). Uses the configured rebalance strategy.
    auto export_ranks = compute_export_ranks_quad_owner(tr);
    migrate_topology(MPI_COMM_WORLD, tr, export_ranks);

    HDF5_CALL(err, H5Pclose(dxpl));
    HDF5_CALL(err, H5Gclose(grp_id));

    GRACE_INFO("Particles restored from checkpoint: {} global, {} on this rank.",
               n_global_ll, tr.size());
    return true;
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
