/**
 * @file particle_hdf5_output.cpp
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_hdf5_output.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/runtime_functions.hh>
#include <grace/system/print.hh>
#include <grace/errors/error.hh>
#include <grace/config/config_parser.hh>

#include <Kokkos_Core.hpp>
#include <hdf5.h>
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
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

/// One emitted snapshot. Tracked in process memory so the master XMF can be
/// rewritten with the full temporal collection on every new snapshot.
struct snapshot_record_t {
    int64_t     iter     = 0;
    double      time     = 0.0;
    int64_t     n_global = 0;
    std::string h5_basename; // file name only, master xmf lives next to it
};

std::vector<snapshot_record_t>& snapshots() {
    static std::vector<snapshot_record_t> s;
    return s;
}

/// Single-axis (1D) and per-row vector (1D extent + fixed inner dim) helpers
/// for collective hyperslab writes. Each rank ships its [n_local] slice into
/// global slot [local_offset, local_offset+n_local).
struct collective_writer_t {
    hid_t     file_id;
    hid_t     dxpl;
    hsize_t   n_global;
    hsize_t   local_offset;
    hsize_t   n_local;
    hsize_t   chunk;
    int       compression;

    template <typename T>
    void write_1d(const std::string& name, hid_t h5type,
                  const T* host_data) const
    {
        herr_t  err;
        hid_t   global_space, local_space, dset_id, prop;
        hsize_t global_dims[1] = {n_global};
        hsize_t local_dims[1]  = {n_local};
        hsize_t chunk_dims[1]  = {std::min<hsize_t>(chunk, std::max<hsize_t>(n_global, 1))};
        HDF5_CALL(global_space, H5Screate_simple(1, global_dims, NULL));
        HDF5_CALL(local_space,  H5Screate_simple(1, local_dims,  NULL));
        HDF5_CALL(prop, H5Pcreate(H5P_DATASET_CREATE));
        if (n_global > 0) {
            HDF5_CALL(err, H5Pset_chunk(prop, 1, chunk_dims));
            if (compression > 0)
                HDF5_CALL(err, H5Pset_deflate(prop, compression));
        }
        HDF5_CALL(dset_id, H5Dcreate2(file_id, name.c_str(), h5type,
                                      global_space, H5P_DEFAULT, prop, H5P_DEFAULT));
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
        HDF5_CALL(err, H5Pclose(prop));
    }

    template <typename T>
    void write_2d(const std::string& name, hid_t h5type, hsize_t inner,
                  const T* host_data) const
    {
        herr_t  err;
        hid_t   global_space, local_space, dset_id, prop;
        hsize_t global_dims[2] = {n_global, inner};
        hsize_t local_dims[2]  = {n_local,  inner};
        hsize_t chunk_dims[2]  = {std::min<hsize_t>(chunk, std::max<hsize_t>(n_global, 1)), inner};
        HDF5_CALL(global_space, H5Screate_simple(2, global_dims, NULL));
        HDF5_CALL(local_space,  H5Screate_simple(2, local_dims,  NULL));
        HDF5_CALL(prop, H5Pcreate(H5P_DATASET_CREATE));
        if (n_global > 0) {
            HDF5_CALL(err, H5Pset_chunk(prop, 2, chunk_dims));
            if (compression > 0)
                HDF5_CALL(err, H5Pset_deflate(prop, compression));
        }
        HDF5_CALL(dset_id, H5Dcreate2(file_id, name.c_str(), h5type,
                                      global_space, H5P_DEFAULT, prop, H5P_DEFAULT));
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
        HDF5_CALL(err, H5Pclose(prop));
    }
};

void write_scalar_attr_double(hid_t loc, const char* name, double value) {
    herr_t err;
    hid_t  attr_id, ds_id;
    HDF5_CALL(ds_id,   H5Screate(H5S_SCALAR));
    HDF5_CALL(attr_id, H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, ds_id,
                                  H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CALL(err,     H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &value));
    HDF5_CALL(err,     H5Aclose(attr_id));
    HDF5_CALL(err,     H5Sclose(ds_id));
}

void write_scalar_attr_int64(hid_t loc, const char* name, int64_t value) {
    herr_t err;
    hid_t  attr_id, ds_id;
    HDF5_CALL(ds_id,   H5Screate(H5S_SCALAR));
    HDF5_CALL(attr_id, H5Acreate2(loc, name, H5T_NATIVE_INT64, ds_id,
                                  H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CALL(err,     H5Awrite(attr_id, H5T_NATIVE_INT64, &value));
    HDF5_CALL(err,     H5Aclose(attr_id));
    HDF5_CALL(err,     H5Sclose(ds_id));
}

/// Write the master temporal-collection XDMF listing every snapshot emitted
/// in this run so far. Rewritten by rank 0 on each new snapshot.
void rewrite_master_xmf(const std::string& xmf_path) {
    std::ofstream out(xmf_path);
    out << R"(<?xml version="1.0" encoding="utf-8"?>)" << "\n"
        << R"(<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">)" << "\n"
        << "<Domain>\n"
        << R"(<Grid CollectionType="Temporal" GridType="Collection" Name="ParticleTrajectories">)" << "\n";

    static const struct {
        const char* xmf_name;
        const char* h5_name;
        const char* type;
        int         inner; // 0 = scalar, else vector inner extent
    } fields[] = {
        {"id",       "id",             "Scalar", 0},
        {"status",   "status",         "Scalar", 0},
        {"alpha",    "sample_alpha",   "Scalar", 0},
        {"W",        "sample_W",       "Scalar", 0},
        {"rho",      "sample_rho",     "Scalar", 0},
        {"temp",     "sample_temp",    "Scalar", 0},
        {"ye",       "sample_ye",      "Scalar", 0},
        {"entropy",  "sample_entropy", "Scalar", 0},
        {"press",    "sample_press",   "Scalar", 0},
        {"eps",      "sample_eps",     "Scalar", 0},
        {"beta",     "sample_beta",    "Vector", 3},
        {"v",        "sample_v",       "Vector", 3},
        {"B",        "sample_B",       "Vector", 3},
    };

    for (const auto& s : snapshots()) {
        out << R"(<Grid Name="Particles_)" << s.iter << R"(">)" << "\n";
        out << R"(<Time Value=")" << s.time << R"("/>)" << "\n";
        // Geometry: list of (x,y,z) points.
        out << R"(<Geometry Type="XYZ">)" << "\n"
            << R"(<DataItem DataType="Float" Dimensions=")" << s.n_global << " 3"
            << R"(" Format="HDF" Precision="8">)" << s.h5_basename << ":/Position</DataItem>" << "\n"
            << R"(</Geometry>)" << "\n";
        // Topology: Polyvertex (one cell per point, no connectivity).
        out << R"(<Topology Type="Polyvertex" NumberOfElements=")" << s.n_global << R"("/>)" << "\n";
        for (const auto& f : fields) {
            out << R"(<Attribute Name=")" << f.xmf_name
                << R"(" Center="Node" Type=")" << f.type << R"(">)" << "\n";
            if (f.inner == 0) {
                out << R"(<DataItem DataType="Float" Dimensions=")" << s.n_global
                    << R"(" Format="HDF" Precision="8">)"
                    << s.h5_basename << ":/" << f.h5_name << "</DataItem>" << "\n";
            } else {
                out << R"(<DataItem DataType="Float" Dimensions=")" << s.n_global
                    << " " << f.inner << R"(" Format="HDF" Precision="8">)"
                    << s.h5_basename << ":/" << f.h5_name << "</DataItem>" << "\n";
            }
            out << R"(</Attribute>)" << "\n";
        }
        out << R"(</Grid>)" << "\n";
    }
    out << R"(</Grid>)" << "\n"
        << R"(</Domain>)" << "\n"
        << R"(</Xdmf>)" << "\n";
}

} // namespace

void write_particle_snapshot(const tracer_container_t<>& tr,
                             const std::string& dir,
                             const std::string& basename)
{
    const int  rank = parallel::mpi_comm_rank();
    const auto comm = parallel::get_comm_world();

    // Collective scan + reduce. Each rank's slice in the global file starts
    // at the prefix sum of n_local across ranks 0..rank-1.
    const long long n_local_ll = static_cast<long long>(tr.size());
    long long       offset_ll  = 0;
    long long       n_global_ll = 0;
    MPI_Exscan   (&n_local_ll, &offset_ll,   1, MPI_LONG_LONG, MPI_SUM, comm);
    parallel::mpi_allreduce(&n_local_ll, &n_global_ll, 1, sc_MPI_SUM, comm);
    if (rank == 0) offset_ll = 0;
    if (n_global_ll == 0) return; // nobody has tracers; skip the snapshot entirely

    // File path. Iter is in the filename so snapshots sort lexicographically.
    const int64_t iter = static_cast<int64_t>(grace::get_iteration());
    const double  t    = grace::get_simulation_time();
    char fname_c[256];
    std::snprintf(fname_c, sizeof(fname_c), "%s_iter%010lld.h5",
                  basename.c_str(), static_cast<long long>(iter));
    const std::string h5_basename = fname_c;
    const std::filesystem::path base_path(dir);
    if (rank == 0) std::filesystem::create_directories(base_path);
    parallel::mpi_barrier();
    const std::string out_path = (base_path / h5_basename).string();

    const size_t compression = grace::get_param<size_t>("IO", "hdf5_compression_level");
    const size_t chunk_param = grace::get_param<size_t>("IO", "hdf5_chunk_size");

    // Mirror device views to host. For O(N) tracers, this dominates only
    // briefly; collective HDF5 then writes from host buffers.
    //
    // 2-D vector fields (pos, beta, v, B) MUST be staged through explicit
    // LayoutRight host buffers: the device default layout on CUDA/HIP is
    // LayoutLeft, but HDF5 (and the [N,3] dimensions in the XMF) interpret
    // the buffer as row-major. Using create_mirror_view_and_copy preserves
    // the device layout, which would silently scramble x/y/z into diagonal
    // patterns at the visualization stage. Kokkos::deep_copy handles the
    // cross-layout transpose for us.
    using h_vec_t = Kokkos::View<double*[3], Kokkos::LayoutRight, Kokkos::HostSpace>;
    h_vec_t h_pos ("h_pos",  tr.size());
    h_vec_t h_beta("h_beta", tr.size());
    h_vec_t h_v   ("h_v",    tr.size());
    h_vec_t h_B   ("h_B",    tr.size());
    Kokkos::deep_copy(h_pos,  tr.pos);
    Kokkos::deep_copy(h_beta, tr.sample_beta);
    Kokkos::deep_copy(h_v,    tr.sample_v);
    Kokkos::deep_copy(h_B,    tr.sample_B);

    // 1-D fields are layout-agnostic; the simple mirror is fine.
    auto h_id       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.id);
    auto h_status   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.status);
    auto h_alpha    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_alpha);
    auto h_W        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_W);
    auto h_rho      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_rho);
    auto h_temp     = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_temp);
    auto h_ye       = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_ye);
    auto h_entropy  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_entropy);
    auto h_press    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_press);
    auto h_eps      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tr.sample_eps);

    // Open file with parallel access.
    herr_t err;
    hid_t  fapl, dxpl, file_id;
    HDF5_CALL(fapl, H5Pcreate(H5P_FILE_ACCESS));
    HDF5_CALL(err,  H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL));
    HDF5_CALL(file_id, H5Fcreate(out_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl));
    HDF5_CALL(err,  H5Pclose(fapl));
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER));
    HDF5_CALL(err,  H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));

    write_scalar_attr_double(file_id, "Time",      t);
    write_scalar_attr_int64 (file_id, "Iteration", iter);
    write_scalar_attr_int64 (file_id, "NParticles", n_global_ll);

    collective_writer_t w{file_id, dxpl,
                          static_cast<hsize_t>(n_global_ll),
                          static_cast<hsize_t>(offset_ll),
                          static_cast<hsize_t>(n_local_ll),
                          static_cast<hsize_t>(chunk_param),
                          static_cast<int>(compression)};

    w.write_2d("Position",       H5T_NATIVE_DOUBLE, 3, h_pos.data());
    w.write_2d("sample_beta",    H5T_NATIVE_DOUBLE, 3, h_beta.data());
    w.write_2d("sample_v",       H5T_NATIVE_DOUBLE, 3, h_v.data());
    w.write_2d("sample_B",       H5T_NATIVE_DOUBLE, 3, h_B.data());
    w.write_1d("id",             H5T_NATIVE_UINT64, h_id.data());
    w.write_1d("status",         H5T_NATIVE_UINT8,  h_status.data());
    w.write_1d("sample_alpha",   H5T_NATIVE_DOUBLE, h_alpha.data());
    w.write_1d("sample_W",       H5T_NATIVE_DOUBLE, h_W.data());
    w.write_1d("sample_rho",     H5T_NATIVE_DOUBLE, h_rho.data());
    w.write_1d("sample_temp",    H5T_NATIVE_DOUBLE, h_temp.data());
    w.write_1d("sample_ye",      H5T_NATIVE_DOUBLE, h_ye.data());
    w.write_1d("sample_entropy", H5T_NATIVE_DOUBLE, h_entropy.data());
    w.write_1d("sample_press",   H5T_NATIVE_DOUBLE, h_press.data());
    w.write_1d("sample_eps",     H5T_NATIVE_DOUBLE, h_eps.data());

    HDF5_CALL(err, H5Pclose(dxpl));
    HDF5_CALL(err, H5Fclose(file_id));

    // Track this snapshot and rewrite the master xmf (rank 0 only).
    snapshots().push_back({iter, t, n_global_ll, h5_basename});
    if (rank == 0) {
        rewrite_master_xmf((base_path / (basename + ".xmf")).string());
    }
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
