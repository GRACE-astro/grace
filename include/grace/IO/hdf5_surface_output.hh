/**
 * @file hdf5_surface_output.hh
 * @author Keneth Miler (miler@itp.uni-frankfurt.de)
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
 * @brief Writes plane surface data to an HDF5 file.
 * \ingroup IO 
 */
void write_plane_cell_data() ; 

/**
 * @brief Writes data from a plane slice of the grid to HDF5.
 * \ingroup IO
 * @param plane The plane along which the grid is sliced
 * NB: This function currently only supports axis-aligned planes.
 */
void write_plane_cell_data_impl(amr::plane_desc_t const& plane) ; 

/**
 * @brief Write the grid structure to hdf5 output.
 * 
 * @param file_id HDF5 identifier of the open file where data is written in parallel.
 * @param compression_level Level of compression for hdf5 output.
 * @param chunk_size Size of chunks for low-level HDF5 compression API.
 */
void write_grid_structure_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::oct_tree_plane_slicer_t& octree_slicer) ; 

/**
 * @brief Write arrays of volume data to HDF file.
 * 
 * @param file_id HDF5 identifier of the open file where data is written in parallel.
 * @param compression_level Level of compression for hdf5 output.
 * @param chunk_size Size of chunks for low-level HDF5 compression API. 
 */
void write_volume_data_arrays_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::oct_tree_plane_slicer_t& octree_slicer) ; 

/**
 * @brief Writes variable arrays to an HDF5 file.
 *
 * This function writes the provided variable arrays to the specified HDF5 file
 * using the given HDF5 identifiers and properties.
 *
 * @param varlist A set of strings representing the list of variable names.
 * @param file_id The HDF5 file identifier.
 * @param dxpl The HDF5 data transfer property list identifier.
 * @param space_id_glob The global dataspace identifier for the primary variables.
 * @param space_id The dataspace identifier for the primary variables.
 * @param prop_id The dataset creation property list identifier for the primary variables.
 * @param ncells The number of cells to write.
 * @param local_quad_offset The local quadrature offset.
 * @param is_aux Is variable auxiliary? 
 */
void write_var_arrays_sliced_hdf5(std::vector<std::string> const& varlist
                           , hid_t file_id, hid_t dxpl, hid_t space_id_glob
                           , hid_t space_id, hid_t prop_id
                           , hsize_t ncells, hsize_t local_quad_offset
                           , amr::oct_tree_plane_slicer_t& octree_slicer
                           , Kokkos::View<size_t*> quad 
                           , Kokkos::View<size_t*> cell 
                           , bool is_aux = false ) ;

/**
 * @brief Writes vector variable arrays to an HDF5 file.
 *
 * This function writes the specified vector variable arrays to an HDF5 file using the provided HDF5 identifiers and properties.
 *
 * @param varlist A set of strings representing the list of variable names to be written.
 * @param file_id The HDF5 file identifier.
 * @param dxpl The HDF5 data transfer property list identifier.
 * @param space_id_glob The HDF5 dataspace identifier for the global space.
 * @param space_id The HDF5 dataspace identifier for the local space.
 * @param prop_id The HDF5 dataset creation property list identifier.
 * @param ncells The number of cells in the dataset.
 * @param local_quad_offset The local quadrature offset.
 * @param is_aux Is variable auxiliary? 
 */
void write_vector_var_arrays_sliced_hdf5( std::vector<std::string> const& varlist 
                                 , hid_t file_id 
                                 , hid_t dxpl
                                 , hid_t space_id_glob
                                 , hid_t space_id
                                 , hid_t prop_id
                                 , hsize_t ncells
                                 , hsize_t local_quad_offset
                                 , amr::oct_tree_plane_slicer_t& octree_slicer
                                 , Kokkos::View<size_t*> quad 
                                 , Kokkos::View<size_t*> cell 
                                 , bool is_aux=false ) ; 

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
void write_extra_arrays_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::oct_tree_plane_slicer_t& octree_slicer) ; 

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

}} /* namespace grace::IO */

#endif /* GRACE_IO_HDF5_OUTPUT_HH */