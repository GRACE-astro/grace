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

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPUnstructuredGridWriter.h>


#include <vector>
#include <string>

#ifndef THUNDER_IO_VTK_OUTPUT_HH
#define THUNDER_IO_VTK_OUTPUT_HH


namespace thunder { namespace IO {

enum thunder_vtk_output_t {
    VOLUME=0,
    PLANE_SURFACE,
    SPHERE_SURFACE
} ; 

void write_cell_output(bool volume_output, bool surface_output_plane, bool surface_output_sphere ) ; 

void setup_volume_cell_data(vtkSmartPointer<vtkUnstructuredGrid> grid, size_t which_output) ;

void write_pvd_file( std::string const& pvdFilename
                   , std::vector<std::string> const& vtkFilenames 
                   , std::vector<double> const& times ) ; 

vtkSmartPointer<vtkUnstructuredGrid> setup_vtk_volume_grid(bool) ;

void add_extra_output_quantities(vtkSmartPointer<vtkUnstructuredGrid>, bool) ;


}} /* namespace thunder::IO */

#endif /* THUNDER_IO_VTK_OUTPUT_HH */
