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

#include <grace/amr/p4est_headers.hh>

#define HDF5_CALL(result,cmd) \
    do {  \
        if((result=cmd)<0) { \
            ERROR("HDF5 API call failed with error code " << result ) ; \
        } \
    } while(false)


namespace grace {

namespace detail {

std::filesystem::path inline get_filename(
    std::filesystem::path const& dir,
    std::string const& base_name, 
    int iteration,
    std::string const& extension )
{

}

void write_data_hdf5(
    hid_t file_id, 
    hid_t dxpl,
    std::string const& dset_name,
    grace::var_array_t<GRACE_NSPACEDIM> data 
)
{
    DECLARE_GRID_EXTENTS ; 


    static constexpr unsigned int chunk_size = 128 ; 
    static constexpr unsigned int compression_level = 6 ;

     /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
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

void checkpoint_handler_impl_t::save_checkpoint() const 
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
        grace::forest::get().get(),
        1
    ) ; 

    // Now we write the state data to an hdf5 file 
    auto state_file = detail::get_filename(checkpoint_dir, "checkpoint_data", iter, ".h5") ;
    herr_t err ; 
    
    // Create property list for parallel file access
    hid_t plist_id ; 
    HDF5_CALL(plist_id,H5Pcreate(H5P_FILE_ACCESS)) ; 
    HDF5_CALL(err,H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL)) ; 
    // Create a new file 
    hid_t file_id ; 
    HDF5_CALL(file_id,H5Fcreate(out_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) ;
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
    /* Create parallel dataset properties */
    hid_t dxpl ; 
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ; 
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ;
    // write state data 
    write_data_hdf5(file_id, dxpl, "CellCenteredData", state) ; 
    // write staggered state data
    write_data_hdf5(file_id, dxpl, "CornerCenteredData", sstate.corner_staggered_fields) ;
    write_data_hdf5(file_id, dxpl, "EdgeCenteredDataXY", sstate.edge_staggered_fields_xy) ;
    write_data_hdf5(file_id, dxpl, "EdgeCenteredDataXZ", sstate.edge_staggered_fields_xz) ;
    write_data_hdf5(file_id, dxpl, "EdgeCenteredDataYZ", sstate.edge_staggered_fields_yz) ;
    write_data_hdf5(file_id, dxpl, "FaceCenteredDataX", sstate.face_staggered_fields_x) ;
    write_data_hdf5(file_id, dxpl, "FaceCenteredDataY", sstate.face_staggered_fields_y) ;
    write_data_hdf5(file_id, dxpl, "FaceCenteredDataZ", sstate.face_staggered_fields_z) ;
    // Block all processes until all data is written
    parallel::mpi_barrier() ;
    // Cleanup 
    HDF5_CALL(err, H5Pclose(dxpl)) ;
    /* Close the file */
    HDF5_CALL(err,H5Fclose(file_id)) ; 
    HDF5_CALL(err,H5Pclose(plist_id)) ; 
}



void checkpoint_handler_impl_t::load_checkpoint(int64_t iter ) const 
{
    /**********************************************************************/
    /* We do the following operations here:                               */
    /* 1) Load the forest file and setup the grid                         */
    /* 2) Once the grid is set up we read the data                        */
    /**********************************************************************/
    GRACE_INFO("Loading checkpoint from iteration {}", iter) ; 

    auto grid_fname = detail::get_filename(checkpoint_dir, "checkpoint_grid", iter, ".bin") ; 

    p4est_connectivity_t * conn = nullptr; 
    p4est_t* p4est  = p4est_load( 
        grid_fname, 
        sc_MPI_COMM_WORLD, 
        sizeof(amr_flags_t), 
        1, 
        nullptr, 
        &conn
    ) ; 
    ASSERT( p4est != nullptr, "Could not load forest file " << grid_fname ) ;
    ASSERT( conn != nullptr, "Could not load connectivity file " << grid_fname ) ;

    grace::connectivity::initialize(conn) ;
    grace::forest::initialize(p4est) ; 
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
    hid_t file_id ;
    HDF5_CALL(file_id,H5Fopen(data_fname.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)) ;
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
    /* Read the data from the hdf5 file                                   */
    /**********************************************************************/
    hid_t dxpl ;
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ;
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ;
    /**********************************************************************/
    /* Read the state data                                                */
    /**********************************************************************/
    auto state = grace::variables::get().getstate() ; 
    read_data_hdf5(file_id, dxpl, "CellCenteredData", state) ; // TODO ! 


}



}

#undef HDF5_CALL
