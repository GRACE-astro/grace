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

#ifndef GRACE_IO_HDF5_OUTPUT_HH
#define GRACE_IO_HDF5_OUTPUT_HH

#include <hdf5.h>

#include <string>
#include <vector>

namespace grace { namespace IO {

namespace detail {
extern std::vector<int64_t> _volume_output_iterations ;
extern std::vector<double>  _volume_output_times ;
extern std::vector<int64_t> _volume_output_ncells ; 
extern std::vector<std::string> _volume_output_filenames ; 
}

void write_cell_data_hdf5(bool out_vol, bool out_plane, bool out_sphere) ; 

void write_volume_cell_data_hdf5() ; 

void write_grid_structure_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size) ; 

void write_volume_data_arrays_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size) ; 


}} /* namespace grace::IO */

#endif /* GRACE_IO_HDF5_OUTPUT_HH */