/**
 * @file hdf5_output.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-23
 * 
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
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

#ifndef GRACE_IO_HDF5_SLICED_OUTPUT_HH
#define GRACE_IO_HDF5_SLICED_OUTPUT_HH

#include <hdf5.h>

#include <string>
#include <vector>
#include <set>
#include <grace/amr/octree_search_class.hh>


namespace grace { namespace IO {

namespace detail {
    extern std::vector<int64_t> _volume_output_sliced_iterations ;
    extern std::vector<double>  _volume_output_sliced_times ;
    extern std::vector<int64_t> _volume_output_sliced_ncells ; 
    extern std::vector<std::string> _volume_output_sliced_filenames ; 
}

/**
 * @brief Writes cell data to an HDF5 file.
 *
 * This function outputs cell data in various formats based on the provided flags.
 *
 * @param out_vol If true, output volume data.
 * @param out_plane If true, output plane data.
 * @param out_sphere If true, output sphere data.
 */
void write_cell_data_sliced_hdf5(bool out_vol, bool out_plane, bool out_sphere) ; 

/**
 * @brief Writes volume cell data to an HDF5 file.
 *
 * This function handles the output of volume cell data into an HDF5 format,
 * which is a widely used data model, library, and file format for storing
 * and managing data.
 */
void write_volume_cell_data_sliced_hdf5(std::string plane_dir) ; 

/**
 * @brief Write the grid structure to hdf5 output.
 * 
 * @param file_id HDF5 identifier of the open file where data is written in parallel.
 * @param compression_level Level of compression for hdf5 output.
 * @param chunk_size Size of chunks for low-level HDF5 compression API.
 */
void write_grid_structure_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::OctreeSlicer& octree_slicer) ; 

/**
 * @brief Write arrays of volume data to HDF file.
 * 
 * @param file_id HDF5 identifier of the open file where data is written in parallel.
 * @param compression_level Level of compression for hdf5 output.
 * @param chunk_size Size of chunks for low-level HDF5 compression API. 
 */
void write_volume_data_arrays_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::OctreeSlicer& octree_slicer) ; 

/**
 * @brief Writes variable arrays to an HDF5 file.
 *
 * This function writes the provided variable arrays to the specified HDF5 file
 * using the given HDF5 identifiers and properties.
 *
 * @param varlist A set of strings representing the list of variable names.
 * @param svarlist A set of strings representing the list of staggered variable names.
 * @param file_id The HDF5 file identifier.
 * @param dxpl The HDF5 data transfer property list identifier.
 * @param space_id_glob The global dataspace identifier for the primary variables.
 * @param space_id The dataspace identifier for the primary variables.
 * @param prop_id The dataset creation property list identifier for the primary variables.
 * @param sspace_id_glob The global dataspace identifier for the staggered variables.
 * @param sspace_id The dataspace identifier for the staggered variables.
 * @param sprop_id The dataset creation property list identifier for the staggered variables.
 * @param ncells The number of cells to write.
 * @param local_quad_offset The local quadrature offset.
 */
void write_var_arrays_sliced_hdf5(std::set<std::string> const& varlist, std::set<std::string> const& svarlist,
                           hid_t file_id, hid_t dxpl, hid_t space_id_glob, hid_t space_id, hid_t prop_id,
                           hid_t sspace_id_glob, hid_t sspace_id, hid_t sprop_id, hsize_t ncells, hsize_t local_quad_offset
                           , amr::OctreeSlicer& octree_slicer) ;

/**
 * @brief Writes vector variable arrays to an HDF5 file.
 *
 * This function writes the specified vector variable arrays to an HDF5 file using the provided HDF5 identifiers and properties.
 *
 * @param varlist A set of strings representing the list of variable names to be written.
 * @param svarlist A set of strings representing the list of staggered variable names to be written.
 * @param file_id The HDF5 file identifier.
 * @param dxpl The HDF5 data transfer property list identifier.
 * @param space_id_glob The HDF5 dataspace identifier for the global space.
 * @param space_id The HDF5 dataspace identifier for the local space.
 * @param prop_id The HDF5 dataset creation property list identifier.
 * @param sspace_id_glob The HDF5 dataspace identifier for the staggered global space.
 * @param sspace_id The HDF5 dataspace identifier for the staggered local space.
 * @param sprop_id The HDF5 dataset creation property list identifier for the staggered variables.
 * @param ncells The number of cells in the dataset.
 * @param local_quad_offset The local quadrature offset.
 */
void write_vector_var_arrays_sliced_hdf5( std::set<std::string> const& varlist 
                                 , std::set<std::string> const& svarlist
                                 , hid_t file_id 
                                 , hid_t dxpl
                                 , hid_t space_id_glob
                                 , hid_t space_id
                                 , hid_t prop_id
                                 , hid_t sspace_id_glob
                                 , hid_t sspace_id
                                 , hid_t sprop_id
                                 , hsize_t ncells
                                 , hsize_t local_quad_offset
                                 , amr::OctreeSlicer& octree_slicer) ; 


/**
 * @brief Writes tensor variable arrays to an HDF5 file.
 *
 * This function writes the specified tensor variable arrays to an HDF5 file using the provided HDF5 identifiers and properties.
 *
 * @param varlist A set of strings representing the list of variable names to be written.
 * @param svarlist A set of strings representing the list of staggered variable names to be written.
 * @param file_id The HDF5 file identifier.
 * @param dxpl The HDF5 data transfer property list identifier.
 * @param space_id_glob The HDF5 dataspace identifier for the global space.
 * @param space_id The HDF5 dataspace identifier for the local space.
 * @param prop_id The HDF5 dataset creation property list identifier.
 * @param sspace_id_glob The HDF5 dataspace identifier for the staggered global space.
 * @param sspace_id The HDF5 dataspace identifier for the staggered local space.
 * @param sprop_id The HDF5 dataset creation property list identifier for the staggered variables.
 * @param ncells The number of cells in the dataset.
 * @param local_quad_offset The local quadrature offset.
 */
void write_tensor_var_arrays_sliced_hdf5( std::set<std::string> const& varlist 
                                 , std::set<std::string> const& svarlist 
                                 , hid_t file_id 
                                 , hid_t dxpl
                                 , hid_t space_id_glob
                                 , hid_t space_id
                                 , hid_t prop_id
                                 , hid_t sspace_id_glob
                                 , hid_t sspace_id
                                 , hid_t sprop_id
                                 , hsize_t ncells
                                 , hsize_t local_quad_offset 
                                 , amr::OctreeSlicer& octree_slicer) ;



/**
 * @brief Writes extra arrays to an HDF5 file.
 *
 * This function writes additional arrays to an HDF5 file specified by the file identifier.
 * The arrays are written with the specified compression level and chunk size.
 *
 * @param file_id The HDF5 file identifier.
 * @param compression_level The level of compression to apply to the data.
 * @param chunk_size The size of the chunks to use for the data.
 */
void write_extra_arrays_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::OctreeSlicer& octree_slicer) ; 

/**
 * @brief Writes a scalar dataset to an HDF5 file.
 *
 * @param data Pointer to the data to be written.
 * @param mem_type_id HDF5 memory datatype identifier.
 * @param file_id HDF5 file identifier.
 * @param dxpl HDF5 data transfer property list identifier.
 * @param dset_size Size of the dataset.
 * @param dset_size_glob Global size of the dataset.
 * @param offset Offset in the dataset.
 * @param chunk_size Size of the chunks for chunked storage.
 * @param compression_level Compression level for the dataset.
 * @param dset_name Name of the dataset.
 */
void write_scalar_dataset_sliced( void* data, hid_t mem_type_id, hid_t file_id, hid_t dxpl
                         , hsize_t dset_size, hsize_t dset_size_glob, hsize_t offset
                         , size_t chunk_size, unsigned int compression_level
                         , std::string const& dset_name ) ; 

/**
 * @brief Writes a string attribute to an HDF5 dataset.
 *
 * This function writes a string attribute to the specified HDF5 dataset.
 *
 * @param dset_id The identifier of the HDF5 dataset to which the attribute will be written.
 * @param attr_name The name of the attribute to be written.
 * @param attr_data The string data of the attribute to be written.
 */
void write_dataset_string_attribute_sliced_hdf5(hid_t dset_id, std::string const& attr_name, std::string const& attr_data) ; 
}} /* namespace grace::IO */

#endif /* GRACE_IO_HDF5_OUTPUT_HH */