/**
 * @file vtk_surface_output.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-05-17
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Differences
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

#ifndef THUNDER_IO_VTK_SURFACE_OUTPUT_HH
#define THUNDER_IO_VTK_SURFACE_OUTPUT_HH

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkXMLPPolyDataWriter.h>

namespace thunder { namespace IO {

void write_plane_surface_cell_data( vtkSmartPointer<vtkUnstructuredGrid> grid
                                  , vtkSmartPointer<vtkXMLPPolyDataWriter> pwriter ) ;

void write_sphere_surface_cell_data( vtkSmartPointer<vtkUnstructuredGrid> grid
                                   , vtkSmartPointer<vtkXMLPPolyDataWriter> pwriter ) ; 

}}


#endif /* THUNDER_IO_VTK_SURFACE_OUTPUT_HH */