/**
 * @file vtk_volume_output_3D.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2024-03-15
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
#ifndef THUNDER_IO_VTK_VOLUME_OUTPUT_3D_HH
#define THUNDER_IO_VTK_VOLUME_OUTPUT_3D_HH

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPUnstructuredGridWriter.h>


#include <vector>

namespace thunder { namespace IO {

void write_volume_vtk_cell_data( vtkSmartPointer<vtkUnstructuredGrid> grid
                               , vtkSmartPointer<vtkXMLPUnstructuredGridWriter> pwriter) ; 

}} /* namespace thunder::IO */


#endif /* THUNDER_IO_VTK_VOLUME_OUTPUT_3D_HH */
