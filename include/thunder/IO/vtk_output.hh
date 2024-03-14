/**
 * @file vtk_output.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief add a thin c++ wrapper around mpi calls.
 * @version 0.1
 * @date 2023-03-01
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference 
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


#include <vector>
#include <string>

#ifndef THUNDER_IO_VTK_OUTPUT_HH
#define THUNDER_IO_VTK_OUTPUT_HH


namespace thunder { namespace IO {

void write_volume_data( std::string const& outfname 
                      , std::vector<std::string> const& cell_scalar_vars 
                      , std::vector<std::string> const& cell_vector_vars 
                      , std::vector<std::vector<std::string>> const& cell_vector_vnames 
                      , std::vector<std::string> const& cell_tensor_vars 
                      , std::vector<std::vector<std::string>> const& cell_tensor_vnames
                      , std::vector<std::string> const& point_scalar_vars 
                      , std::vector<std::string> const& point_vector_vars 
                      , std::vector<std::vector<std::string>> const& point_vector_vnames 
                      , std::vector<std::string> const& point_tensor_vars 
                      , std::vector<std::vector<std::string>> const& point_tensor_vnames)

}} /* namespace thunder::IO */

#endif /* THUNDER_IO_VTK_OUTPUT_HH */
