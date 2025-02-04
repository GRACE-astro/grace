/**
 * @file checkpoint_handler.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-01-31
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023 Carlo Musolino
 *                                    
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *   
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *   
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * 
 */

#include <grace/system/checkpoint_handler.hh>

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/config/config_parser.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/system/grace_system.hh>
#include <grace/IO/hdf5_output.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/profiling/profiling_runtime.hh>
#include <grace/data_structures/variables.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/system/runtime_functions.hh>

#include <grace/amr/p4est_headers.hh>

#include <hdf5.h>
#include <omp.h>
#include <filesystem>
#include <regex>
#include <vector> 
#include <algorithm>

#define HDF5_CALL(result,cmd) \
    do {  \
        if((result=cmd)<0) { \
            ERROR("HDF5 API call failed with error code " << result ) ; \
        } \
    } while(false)


namespace grace {

namespace detail {

/**
 * @brief Generate filename given basepath, prefix iteration and extension
 * \cond grace_detail
 * @param dir base path
 * @param base_name name prefix
 * @param iteration iteration
 * @param extension file extension
 * @return std::filesystem::path File path where the iteration number is padded by zeros and inserted
 */
std::filesystem::path inline get_filename(
    std::filesystem::path const& dir,
    std::string const& base_name, 
    int iteration,
    std::string const& extension )
{
    std::stringstream ss ; 
    ss << base_name << "_" << std::setw(6) << std::setfill('0') << iteration << extension ; 
    return dir / ss.str() ; 
}

void read_data_hdf5(
    hid_t file_id,
    hid_t dxpl, 
    std::string const& dset_name,
    grace::var_array_t<GRACE_NSPACEDIM>& data
)
{
    DECLARE_GRID_EXTENTS ; 
    
    herr_t err ; 

    auto rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ; 
    unsigned long const local_quad_offset = _p4est->global_first_quadrant[rank] ;
    /* Number of datapoints/quadrant */ 
    unsigned long const npts_quad = EXPR(data.extent(0),*data.extent(1),*data.extent(2)) ;
    /* Global dataset dimension */
    unsigned long const dim_loc  = npts_quad * data.extent(GRACE_NSPACEDIM) * nq ; 
    unsigned long const dim_glob = npts_quad * data.extent(GRACE_NSPACEDIM) * nq_glob ;

    // If there are no variables return 
    // FIXME if the code was somehow compiled with different variables than the code that wrote the checkpoint this is a bug! 
    if (dim_glob == 0) return ; 

    // Open the dataset
    hid_t dset_id, space_id; 
    HDF5_CALL(
        dset_id, H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT)
    ) ; 
    HDF5_CALL(
        space_id, H5Dget_space(dset_id)
    ) ; 

    // Get dataset dimensions
    hsize_t dim[1] ; 
    HDF5_CALL(
        err, H5Sget_simple_extent_dims(space_id, dim, NULL)
    ) ;

    // Check that they match
    ASSERT(dim[0] == dim_glob, "Dataset " << dset_name << " dimensions do not match: " << dim[0] << " != " << dim_glob) ;

    // Select hyperslab for this rank's output
    hid_t memspace_id ; 
    hsize_t count[1] = {dim_loc} ;
    hsize_t offset[1] = {local_quad_offset * npts_quad * data.extent(GRACE_NSPACEDIM)} ;
     HDF5_CALL(memspace_id, H5Screate_simple(1, count, NULL)) ; 
    HDF5_CALL(
        err, H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL)
    ) ;


    // Read data 
    double* buffer = (double*) Kokkos::kokkos_malloc<Kokkos::HostSpace>("staging_buffer", sizeof(double) * dim_loc) ; 
    HDF5_CALL(
        err, H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace_id, space_id, dxpl, buffer)
    ) ;

    // We need to shuffle the data a bit since in general the 
    // target View will have a different layout than the 
    // HDF5 buffer due to padding 

    // Create unmanaged host view 
    Kokkos::View<double EXPR(*,*,*)**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
     buffer_view( buffer, VEC(data.extent(0), data.extent(1), data.extent(2)), data.extent(GRACE_NSPACEDIM), data.extent(GRACE_NSPACEDIM+1) ) ;
    auto h_mirror = Kokkos::create_mirror_view(data)  ;
    // Copy h2h 
    Kokkos::deep_copy(h_mirror, buffer_view) ;
    // Copy h2d 
    Kokkos::deep_copy(data, h_mirror) ;

    // Close everything
    H5Sclose(memspace_id);
    H5Sclose(space_id);
    H5Dclose(dset_id);

}

void write_data_hdf5(
    hid_t file_id, 
    hid_t dxpl,
    std::string const& dset_name,
    grace::var_array_t<GRACE_NSPACEDIM> data 
)
{
    DECLARE_GRID_EXTENTS ; 

    herr_t err ; 

    static constexpr unsigned int chunk_size = 128 ; 
    static constexpr unsigned int compression_level = 6 ;
    auto rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ; 
    unsigned long const local_quad_offset = _p4est->global_first_quadrant[rank] ;
    /* Number of datapoints/quadrant */ 
    unsigned long const npts_quad = EXPR(data.extent(0),*data.extent(1),*data.extent(2)) ;
    /* Global dataset dimension */
    unsigned long const dim_loc  = npts_quad * data.extent(GRACE_NSPACEDIM) * nq ; 
    unsigned long const dim_glob = npts_quad * data.extent(GRACE_NSPACEDIM) * nq_glob ;

    // If there are no variables return 
    if (dim_glob == 0) return ; 

    /* Global space for dataset */
    hid_t space_id_glob ; 
    hsize_t dset_dims_glob[1] = {dim_glob} ;
    HDF5_CALL(space_id_glob, H5Screate_simple(1, dset_dims_glob, NULL)) ;
    /* Dataset properties */
    hid_t prop_id ; 
    HDF5_CALL(prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t chunk_dim[1] = {chunk_size} ;
    HDF5_CALL(err, H5Pset_chunk(prop_id,1,chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(prop_id, compression_level)) ;  
    /* Create local space for this rank */
    hid_t space_id ; 
    hsize_t dset_dims[1] = {dim_loc} ;
    HDF5_CALL(space_id, H5Screate_simple(1, dset_dims, NULL)) ;
    /* Start data transfer */
    auto h_mirror = Kokkos::create_mirror_view(data) ; 
    Kokkos::deep_copy(grace::default_execution_space{},h_mirror,data) ; 
    /* Create dataset */
    hid_t dset_id ; 
        HDF5_CALL( dset_id
                , H5Dcreate2( file_id
                            , dset_name.c_str()
                            , H5T_NATIVE_DOUBLE
                            , space_id_glob
                            , H5P_DEFAULT
                            , prop_id
                            , H5P_DEFAULT) ) ;
    /* Select hyperslab for this rank's output */
    hsize_t slab_start[1]  = {local_quad_offset * npts_quad * data.extent(GRACE_NSPACEDIM)} ; 
    hsize_t slab_count[1]  = {dim_loc} ;
    HDF5_CALL( err
            , H5Sselect_hyperslab( space_id_glob
                                , H5S_SELECT_SET
                                , slab_start
                                , NULL
                                , slab_count 
                                , NULL )) ;
    Kokkos::fence() ; // Wait for data transfer here! 

    /* write to dataset */
    HDF5_CALL( err
                , H5Dwrite( dset_id
                        , H5T_NATIVE_DOUBLE
                        , space_id
                        , space_id_glob 
                        , dxpl 
                        , reinterpret_cast<void*>(h_mirror.data()) )) ;

    /* Close dataset */
    HDF5_CALL(err, H5Dclose(dset_id)) ; 
    /*****************************************************************************************/
    /*                                  Close data space                                     */
    /*****************************************************************************************/
    HDF5_CALL(err, H5Sclose(space_id)) ; 
    HDF5_CALL(err, H5Sclose(space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(prop_id)) ;

}  

}

void checkpoint_handler_impl_t::save_checkpoint()  
{

    DECLARE_GRID_EXTENTS ;
    unsigned int const iter = grace::get_iteration() ;
    double const time = grace::get_simulation_time() ; 
    /* Save the current state to a checkpoint file */
    GRACE_INFO("Saving checkpoint at iteration {} simulation time {}.", iter, time ) ;

    // first write the forest to file 
    auto forest_file = detail::get_filename(checkpoint_dir, "checkpoint_grid", iter, ".bin") ; 
    p4est_save(
        forest_file.string().c_str(),
        amr::forest::get().get(),
        1
    ) ; 

    // Now we write the state data to an hdf5 file 
    auto state_file = detail::get_filename(checkpoint_dir, "checkpoint_data", iter, ".h5") ;
    herr_t err ; 
    
    // Create property list for parallel file access
    hid_t plist_id ; 
    HDF5_CALL(plist_id,H5Pcreate(H5P_FILE_ACCESS)) ; 
    HDF5_CALL(err,H5Pset_fapl_mpio(plist_id, mpi_comm_world, MPI_INFO_NULL)) ; 
    // Create a new file 
    hid_t file_id ; 
    HDF5_CALL(file_id,H5Fcreate(state_file.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) ;
    hid_t attr_id, attr_dataspace_id;
    std::string file_attr_name = "Time";
    const double file_attr_data = time ; 
    // Create a dataspace for the attribute
    HDF5_CALL(attr_dataspace_id,H5Screate(H5S_SCALAR));
    // Create the attribute
    HDF5_CALL(attr_id,H5Acreate2(file_id, file_attr_name.c_str(), H5T_NATIVE_DOUBLE, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT));
    // Write the attribute data
    HDF5_CALL(err,H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &file_attr_data));
    // Close the attribute and dataspace
    HDF5_CALL(err,H5Aclose(attr_id));
    HDF5_CALL(err,H5Sclose(attr_dataspace_id));
    file_attr_name = "Iteration" ;  
    HDF5_CALL(attr_dataspace_id,H5Screate(H5S_SCALAR));
    // Create the attribute
    HDF5_CALL(attr_id,H5Acreate2(file_id, file_attr_name.c_str(), H5T_NATIVE_UINT, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT));
    // Write the attribute data
    HDF5_CALL(err,H5Awrite(attr_id, H5T_NATIVE_UINT, &iter));
    // Close the attribute and dataspace
    HDF5_CALL(err,H5Aclose(attr_id));
    HDF5_CALL(err,H5Sclose(attr_dataspace_id));
    // Write coordinates 
    if ( parallel::mpi_comm_rank() == 0 ) {
        // x extent 
        {
            hid_t space_id; 
            hsize_t dset_dims[1] = {2} ;
            HDF5_CALL(space_id, H5Screate_simple(1, dset_dims, NULL)) ;
            /* Create dataset */
            hid_t dset_id ; 
            HDF5_CALL( dset_id
                    , H5Dcreate2( file_id
                                , "grid_extent_x"
                                , H5T_NATIVE_DOUBLE
                                , space_id
                                , H5P_DEFAULT
                                , H5P_DEFAULT
                                , H5P_DEFAULT) ) ;
            double x_extent[2] = {
                grace::get_param<double>("amr", "xmin"),
                grace::get_param<double>("amr", "xmax")
            } ;
            /* write to dataset */
            HDF5_CALL( err
                        , H5Dwrite( dset_id
                                , H5T_NATIVE_DOUBLE
                                , H5S_ALL
                                , H5S_ALL 
                                , H5P_DEFAULT 
                                , reinterpret_cast<void*>(x_extent) )) ;
            HDF5_CALL(err, H5Dclose(dset_id)) ;
            HDF5_CALL(err, H5Sclose(space_id)) ;
        }
        // y extent 
        {
            hid_t space_id; 
            hsize_t dset_dims[1] = {2} ;
            HDF5_CALL(space_id, H5Screate_simple(1, dset_dims, NULL)) ;
            /* Create dataset */
            hid_t dset_id ; 
            HDF5_CALL( dset_id
                    , H5Dcreate2( file_id
                                , "grid_extent_y"
                                , H5T_NATIVE_DOUBLE
                                , space_id
                                , H5P_DEFAULT
                                , H5P_DEFAULT
                                , H5P_DEFAULT) ) ;
            double y_extent[2] = {
                grace::get_param<double>("amr", "ymin"),
                grace::get_param<double>("amr", "ymax")
            } ;
            /* write to dataset */
            HDF5_CALL( err
                        , H5Dwrite( dset_id
                                , H5T_NATIVE_DOUBLE
                                , H5S_ALL
                                , H5S_ALL 
                                , H5P_DEFAULT 
                                , reinterpret_cast<void*>(y_extent) )) ;
            HDF5_CALL(err, H5Dclose(dset_id)) ;
            HDF5_CALL(err, H5Sclose(space_id)) ;
        }
        // z extent 
        {
            hid_t space_id; 
            hsize_t dset_dims[1] = {2} ;
            HDF5_CALL(space_id, H5Screate_simple(1, dset_dims, NULL)) ;
            /* Create dataset */
            hid_t dset_id ; 
            HDF5_CALL( dset_id
                    , H5Dcreate2( file_id
                                , "grid_extent_z"
                                , H5T_NATIVE_DOUBLE
                                , space_id
                                , H5P_DEFAULT
                                , H5P_DEFAULT
                                , H5P_DEFAULT) ) ;
            double z_extent[2] = {
                grace::get_param<double>("amr", "zmin"),
                grace::get_param<double>("amr", "zmax")
            } ;
            /* write to dataset */
            HDF5_CALL( err
                        , H5Dwrite( dset_id
                                , H5T_NATIVE_DOUBLE
                                , H5S_ALL
                                , H5S_ALL 
                                , H5P_DEFAULT 
                                , reinterpret_cast<void*>(z_extent) )) ;
            HDF5_CALL(err, H5Dclose(dset_id)) ;
            HDF5_CALL(err, H5Sclose(space_id)) ;
        } 
    }
    /* Create parallel dataset properties */
    hid_t dxpl ; 
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ; 
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ;
    // write state data 
    auto state = grace::variable_list::get().getstate() ; 
    detail::write_data_hdf5(file_id, dxpl, "CellCenteredData", state) ; 
    // write staggered state data
    auto sstate = grace::variable_list::get().getstaggeredstate() ; 
    detail::write_data_hdf5(file_id, dxpl, "CornerCenteredData", sstate.corner_staggered_fields) ;
    detail::write_data_hdf5(file_id, dxpl, "EdgeCenteredDataXY", sstate.edge_staggered_fields_xy) ;
    detail::write_data_hdf5(file_id, dxpl, "EdgeCenteredDataXZ", sstate.edge_staggered_fields_xz) ;
    detail::write_data_hdf5(file_id, dxpl, "EdgeCenteredDataYZ", sstate.edge_staggered_fields_yz) ;
    detail::write_data_hdf5(file_id, dxpl, "FaceCenteredDataX", sstate.face_staggered_fields_x) ;
    detail::write_data_hdf5(file_id, dxpl, "FaceCenteredDataY", sstate.face_staggered_fields_y) ;
    detail::write_data_hdf5(file_id, dxpl, "FaceCenteredDataZ", sstate.face_staggered_fields_z) ;
    // Block all processes until all data is written
    parallel::mpi_barrier() ;
    // Cleanup 
    HDF5_CALL(err, H5Pclose(dxpl)) ;
    /* Close the file */
    HDF5_CALL(err,H5Fclose(file_id)) ; 
    HDF5_CALL(err,H5Pclose(plist_id)) ; 

    // Append the checkpoint to the list 
    checkpoint_list.push_back(iter) ;
    if ( checkpoint_list.size() > max_n_checkpoints ) {
        delete_checkpoint() ;
    }

    next_checkpoint_wtime = grace::get_total_runtime() + checkpoint_wtime_interval * 3600 ; 
    next_checkpoint_iter  = grace::get_iteration() + checkpoint_iter_interval ;
    next_checkpoint_time  = grace::get_simulation_time() + checkpoint_time_interval ;
}



void checkpoint_handler_impl_t::load_checkpoint(int64_t iter )  
{
    if ( iter < 0 ) {
        iter = checkpoint_list.back() ;
    }
    /**********************************************************************/
    /* We do the following operations here:                               */
    /* 1) Load the forest file and setup the grid                         */
    /* 2) Once the grid is set up we read the data                        */
    /**********************************************************************/
    GRACE_INFO("Loading checkpoint from iteration {}", iter) ; 

    auto grid_fname = detail::get_filename(checkpoint_dir, "checkpoint_grid", iter, ".bin") ; 

    p4est_connectivity_t * conn = nullptr; 
    p4est_t* p4est  = p4est_load( 
        grid_fname.string().c_str(), 
        sc_MPI_COMM_WORLD, 
        sizeof(amr::amr_flags_t), 
        1, 
        nullptr, 
        &conn
    ) ; 
    ASSERT( p4est != nullptr, "Could not load forest file " << grid_fname ) ;
    ASSERT( conn != nullptr, "Could not load connectivity file " << grid_fname ) ;

    amr::connectivity::initialize(conn) ;
    amr::forest::initialize(p4est) ; 
    /**********************************************************************/
    /* Now we set these static variables from the parameter file          */
    /* Later we will need to check that they haven't changed since        */
    /* when the checkpoint was written.                                   */
    /**********************************************************************/
    grace::amr::detail::_nx = 
        grace::config_parser::get()["amr"]["npoints_block_x"].as<int64_t>() ;
    grace::amr::detail::_ny =
        grace::config_parser::get()["amr"]["npoints_block_y"].as<int64_t>() ;
    grace::amr::detail::_nz = 
        grace::config_parser::get()["amr"]["npoints_block_z"].as<int64_t>() ;
    grace::amr::detail::_ngz = 
        grace::config_parser::get()["amr"]["n_ghostzones"].as<int>() ;
    /**********************************************************************/
    GRACE_INFO("Allocating memory...");
    /**********************************************************************/
    grace::variable_list::initialize() ;
    grace::runtime::initialize() ; 
    grace::coordinate_system::initialize() ;
    /**********************************************************************/
    auto data_fname = detail::get_filename(checkpoint_dir, "checkpoint_data", iter, ".h5") ;
    /**********************************************************************/
    herr_t err ;
    // Create property list for parallel file access
    hid_t plist_id ; 
    HDF5_CALL(plist_id,H5Pcreate(H5P_FILE_ACCESS)) ; 
    HDF5_CALL(err,H5Pset_fapl_mpio(plist_id, mpi_comm_world, MPI_INFO_NULL)) ; 
    // Open the file
    hid_t file_id ;
    HDF5_CALL(file_id,H5Fopen(data_fname.string().c_str(), H5F_ACC_RDONLY, plist_id)) ;
    /**********************************************************************/
    /* Read the iteration and time attributes                             */
    /**********************************************************************/
    unsigned int iter_read ;
    double time_read ;
    hid_t attr_id, attr_dataspace_id ;
    HDF5_CALL(attr_id,H5Aopen(file_id, "Iteration", H5P_DEFAULT)) ;
    HDF5_CALL(attr_dataspace_id,H5Aget_space(attr_id)) ;
    HDF5_CALL(err,H5Aread(attr_id, H5T_NATIVE_UINT, &iter_read)) ;
    HDF5_CALL(err,H5Sclose(attr_dataspace_id)) ;    
    HDF5_CALL(err,H5Aclose(attr_id)) ;
    HDF5_CALL(attr_id,H5Aopen(file_id, "Time", H5P_DEFAULT)) ;
    HDF5_CALL(attr_dataspace_id,H5Aget_space(attr_id)) ;
    HDF5_CALL(err,H5Aread(attr_id, H5T_NATIVE_DOUBLE, &time_read)) ;
    HDF5_CALL(err,H5Sclose(attr_dataspace_id)) ;
    HDF5_CALL(err,H5Aclose(attr_id)) ;
    /**********************************************************************/
    ASSERT(iter == iter_read, "Iterations don't match in checkpoint file " << iter << " != " << iter_read) ;
    /**********************************************************************/
    // Set iteration and time in grace runtime 
    grace::set_iteration(iter) ;
    grace::set_simulation_time(time_read) ;
    /**********************************************************************/
    /* Read the grid extents and check they match the parfile             */
    /**********************************************************************/
    if( parallel::mpi_comm_rank() == 0 ) {
        // x extent 
        {
            hid_t dset_id, space_id;
            HDF5_CALL(dset_id,H5Dopen(file_id, "grid_extent_x", H5P_DEFAULT)) ;
            HDF5_CALL(space_id,H5Dget_space(dset_id)) ;

            hsize_t dim[1] ; 
            HDF5_CALL(err, H5Sget_simple_extent_dims(space_id, dim, NULL)) ;

            ASSERT(dim[0] == 2, "Dimension of grid_extent_x is not 2") ;

            double x_extent_read[2] ; 
            HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,x_extent_read)) ;

            double x_extent_param[2] = {
                grace::get_param<double>("amr", "xmin"),
                grace::get_param<double>("amr", "xmax")
            } ; 

            ASSERT(
                x_extent_read[0] == x_extent_param[0] && x_extent_read[1] == x_extent_param[1],
                "x extents do not match in checkpoint: " << x_extent_read[0] << " != " << x_extent_param[0] << " or " << x_extent_read[1] << " != " << x_extent_param[1]
            ) ; 
        }
        // y extent 
        {
            hid_t dset_id, space_id;
            HDF5_CALL(dset_id,H5Dopen(file_id, "grid_extent_y", H5P_DEFAULT)) ;
            HDF5_CALL(space_id,H5Dget_space(dset_id)) ;

            hsize_t dim[1] ; 
            HDF5_CALL(err, H5Sget_simple_extent_dims(space_id, dim, NULL)) ;

            ASSERT(dim[0] == 2, "Dimension of grid_extent_y is not 2") ;

            double y_extent_read[2] ; 
            HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,y_extent_read)) ;

            double y_extent_param[2] = {
                grace::get_param<double>("amr", "ymin"),
                grace::get_param<double>("amr", "ymax")
            } ; 

            ASSERT(
                y_extent_read[0] == y_extent_param[0] && y_extent_read[1] == y_extent_param[1],
                "y extents do not match in checkpoint: " << y_extent_read[0] << " != " << y_extent_param[0] << " or " << y_extent_read[1] << " != " << y_extent_param[1]
            ) ; 
        }
        // z extent 
        {
            hid_t dset_id, space_id;
            HDF5_CALL(dset_id,H5Dopen(file_id, "grid_extent_z", H5P_DEFAULT)) ;
            HDF5_CALL(space_id,H5Dget_space(dset_id)) ;

            hsize_t dim[1] ; 
            HDF5_CALL(err, H5Sget_simple_extent_dims(space_id, dim, NULL)) ;

            ASSERT(dim[0] == 2, "Dimension of grid_extent_z is not 2") ;

            double z_extent_read[2] ; 
            HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,z_extent_read)) ;

            double z_extent_param[2] = {
                grace::get_param<double>("amr", "zmin"),
                grace::get_param<double>("amr", "zmax")
            } ; 

            ASSERT(
                z_extent_read[0] == z_extent_param[0] && z_extent_read[1] == z_extent_param[1],
                "z extents do not match in checkpoint: " << z_extent_read[0] << " != " << z_extent_param[0] << " or " << z_extent_read[1] << " != " << z_extent_param[1]
            ) ; 
        }
    }
    /**********************************************************************/
    /* Read the data from the hdf5 file                                   */
    /**********************************************************************/
    hid_t dxpl ;
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ;
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ;
    /**********************************************************************/
    /* Read the state data                                                */
    /**********************************************************************/
    auto state = grace::variable_list::get().getstate() ; 
    auto sstate = grace::variable_list::get().getstaggeredstate() ; 
    detail::read_data_hdf5(file_id, dxpl, "CellCenteredData", state) ; 
    detail::read_data_hdf5(file_id, dxpl, "CornerCenteredData", sstate.corner_staggered_fields) ;
    detail::read_data_hdf5(file_id, dxpl, "EdgeCenteredDataXY", sstate.edge_staggered_fields_xy) ;
    detail::read_data_hdf5(file_id, dxpl, "EdgeCenteredDataXZ", sstate.edge_staggered_fields_xz) ;
    detail::read_data_hdf5(file_id, dxpl, "EdgeCenteredDataYZ", sstate.edge_staggered_fields_yz) ;
    detail::read_data_hdf5(file_id, dxpl, "FaceCenteredDataX", sstate.face_staggered_fields_x) ;
    detail::read_data_hdf5(file_id, dxpl, "FaceCenteredDataY", sstate.face_staggered_fields_y) ;
    detail::read_data_hdf5(file_id, dxpl, "FaceCenteredDataZ", sstate.face_staggered_fields_z) ;
    /**********************************************************************/
    /* Close the file                                                     */
    /**********************************************************************/
    HDF5_CALL(err,H5Fclose(file_id)) ;
    /**********************************************************************/
    /* Cleanup                                                            */
    /**********************************************************************/
    HDF5_CALL(err,H5Pclose(dxpl)) ;
    /**********************************************************************/
    next_checkpoint_time += grace::get_simulation_time() ; 
    next_checkpoint_iter += grace::get_iteration() ;
    GRACE_INFO("Checkpoint loaded successfully.") ;
}

void checkpoint_handler_impl_t::delete_checkpoint() 
{
    auto iter = checkpoint_list.front() ; 
    // first the forest file 
    auto fname = detail::get_filename(checkpoint_dir, "checkpoint_grid", iter, ".bin") ;
    std::filesystem::remove(fname) ;
    // then the data 
    auto data_fname = detail::get_filename(checkpoint_dir, "checkpoint_data", iter, ".h5") ;
    std::filesystem::remove(data_fname) ;
    // remove from the list
    checkpoint_list.pop_front() ;
}

void checkpoint_handler_impl_t::detect_checkpoints() 
{
    /**********************************************************************/
    /* Clear the list of checkpoints                                      */
    /**********************************************************************/
    checkpoint_list.clear() ; 
    /**********************************************************************/
    /* We search for checkpoints in the directory specified by the        */
    /* parameter <code>checkpoint_dir</code>.                             */
    /**********************************************************************/
    std::filesystem::path dir(checkpoint_dir) ;
    if ( !std::filesystem::exists(dir) ) {
        GRACE_VERBOSE("Checkpoint directory {} does not exist. No checkpoints found.", dir.string()) ;
        return ; 
    }
    /**********************************************************************/
    /* We search for files with the pattern checkpoint_data_*.h5          */
    /**********************************************************************/
    std::regex data_pattern(R"(checkpoint_data_(\d+)\.h5)") ;
    std::regex grid_pattern(R"(checkpoint_grid_(\d+)\.bin)") ;

    std::vector<int64_t> data_iters ; 
    std::vector<int64_t> grid_iters ; 

    for( auto const& entry: std::filesystem::directory_iterator(dir) ) {
        std::string filename = entry.path().filename().string() ; 
        GRACE_INFO("checking {}...", filename) ; 
        std::smatch match ;

        if ( std::regex_match(filename, match, data_pattern)) {
            GRACE_INFO("Hit! It's a data file") ; 
            data_iters.push_back(std::stoll(match[1].str())) ; 
        } else if ( std::regex_match(filename, match, grid_pattern)) {
            GRACE_INFO("Hit! It's a grid file") ;
            grid_iters.push_back(std::stoll(match[1].str())) ; 
        }
    }
    std::sort(data_iters.begin(), data_iters.end()); std::sort(grid_iters.begin(),grid_iters.end()) ; 
    /**********************************************************************/
    /* Check if there are any data files without corresponding grid files */
    /**********************************************************************/
    for( int i=0; i<data_iters.size(); ++i) {
        if ( std::find(grid_iters.begin(), grid_iters.end(), data_iters[i]) == grid_iters.end() ) {
            GRACE_WARN("Checkpoint data file {} does not have a corresponding grid file.", data_iters[i]) ;
        } else {
            checkpoint_list.push_back(data_iters[i]) ;
        }
    }
    /**********************************************************************/
    GRACE_INFO("Found {} checkpoints in directory {}", checkpoint_list.size(), dir.string()) ;
}

bool checkpoint_handler_impl_t::need_checkpoint()  {
    if ( checkpoint_interval_type == "iteration" ) {
        return grace::get_iteration() > next_checkpoint_iter ;
    } else if ( checkpoint_interval_type == "time" ) {
        return grace::get_simulation_time() > next_checkpoint_time ;
    } else {
        return grace::get_total_runtime() > next_checkpoint_wtime ;
    } 
}

checkpoint_handler_impl_t::checkpoint_handler_impl_t() {
    // Read parameters for max number of checkpoints 
    max_n_checkpoints = grace::get_param<unsigned int>("checkpoints", "max_n_checkpoints") ; 
    // And for the directory 
    checkpoint_dir = grace::get_param<std::string>("checkpoints", "checkpoint_dir") ;
    // If the directory does not exist rank 0 should create it 
    if ( parallel::mpi_comm_rank() == 0 ) {
        if ( !std::filesystem::exists(checkpoint_dir) ) {
            std::filesystem::create_directory(checkpoint_dir) ;
        }
    }
    // Find the method to save checkpoints
    checkpoint_interval_type = grace::get_param<std::string>("checkpoints", "interval_kind") ;
    std::vector<std::string> methods{ "iteration", "time", "walltime" } ;
    if (std::find(methods.begin(), methods.end(), checkpoint_interval_type) == methods.end()) {
        ERROR("Invalid checkpointing method " << checkpoint_interval_type << " valid methods are 'iteration', 'time' and 'walltime'.") ;
    }
    if ( checkpoint_interval_type == "iteration" ) {
        checkpoint_iter_interval = grace::get_param<int64_t>("checkpoints", "iteration_interval") ;
        ASSERT(checkpoint_iter_interval > 0, "Iteration interval must be greater than zero!") ;
        next_checkpoint_iter = checkpoint_iter_interval ;
    } else if ( checkpoint_interval_type == "time" ) {
        checkpoint_time_interval = grace::get_param<double>("checkpoints", "time_interval") ;
        ASSERT(checkpoint_time_interval > 0, "Time interval must be greater than zero!") ;
        next_checkpoint_time = checkpoint_time_interval ;
    } else if ( checkpoint_interval_type == "walltime" ) {
        checkpoint_wtime_interval = grace::get_param<double>("checkpoints", "walltime_interval") ;
        ASSERT(checkpoint_wtime_interval > 0, "Walltime interval must be greater than zero!") ;
        // This is in hours! 
        next_checkpoint_wtime = checkpoint_wtime_interval * 3600;
    }
    // Detect checkpoints
    detect_checkpoints() ;

    have_checkpoint_ = checkpoint_list.size() > 0 ; 
}   


checkpoint_handler_impl_t::~checkpoint_handler_impl_t() {
    save_checkpoint(); 
}

} // namespace grace 

#undef HDF5_CALL
