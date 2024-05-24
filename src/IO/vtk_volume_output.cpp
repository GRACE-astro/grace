/**
 * @file vtk_volume_output.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
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
#include <Kokkos_Core.hpp>
/* VTK includes */
/* grid type */
#include <vtkUnstructuredGrid.h>
/* points */
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPoints.h>
/* writers */
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
/* cell types */
#include <vtkCellData.h>
#include <vtkHexahedron.h>
#include <vtkBiQuadraticQuadraticHexahedron.h> 
#include <vtkQuad.h>
#include <vtkQuadraticLinearQuad.h> 
/* memory */
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkXMLDataElement.h>
#include <vtkXMLUtilities.h>
/* VTK MPI */
#include <vtkMPI.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
/* thunder includes */
#include <thunder/data_structures/variable_properties.hh>
#include <thunder/system/thunder_runtime.hh>
#include <thunder/coordinates/coordinate_systems.hh> 
#include <thunder/IO/vtk_output.hh>
#include <thunder/IO/vtk_volume_output.hh>
#include <thunder/IO/vtk_output_auxiliaries.hh>
#include <thunder/IO/vtk_output.tpp>
#include <thunder/amr/thunder_amr.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/errors/error.hh>
#include <thunder/errors/assert.hh> 
#include <thunder/data_structures/variables.hh>
#include <thunder/coordinates/coordinate_systems.hh>
/* Kokkos */
#include <Kokkos_Core.hpp>
/* stdlib includes */
#include <tuple>
#include <array>
#include <string>
#include <filesystem>


namespace thunder { namespace IO {


void write_volume_vtk_cell_data( vtkSmartPointer<vtkUnstructuredGrid> grid
                               , vtkSmartPointer<vtkXMLPUnstructuredGridWriter> pwriter ) 
{   
    Kokkos::Profiling::pushRegion("VTK volume cell output") ; 
    auto& runtime = thunder::runtime::get() ;
    std::filesystem::path base_path (runtime.volume_io_basepath()) ;
    std::string pfname = runtime.volume_io_basename() + "_" + utils::zero_padded(runtime.iteration(),3)
                                                            + ".pvtu" ;
     
    std::filesystem::path out_path = base_path / pfname ;

    setup_volume_cell_data(grid, thunder_vtk_output_t::VOLUME) ; 
    
    pwriter->SetFileName(out_path.string().c_str()) ; ; 
    pwriter->SetInputData( grid ) ;
    pwriter->Write() ; 
    
    detail::_volume_filenames.push_back(pfname) ; 
    detail::_volume_iterations.push_back(thunder::get_iteration()) ; 
    detail::_volume_times.push_back(thunder::get_simulation_time()) ; 
    if( parallel::mpi_comm_rank() == 0 ) {
        std::string pvd_basefilename = runtime.volume_io_basename() + ".pvd" ; 
        std::filesystem::path pvd_filename = base_path / pvd_basefilename ;
        write_pvd_file( pvd_filename.string()
                      , detail::_volume_filenames
                      , detail::_volume_times ) ;
    } 
    Kokkos::Profiling::popRegion() ;
}

}} /* namespace thunder::IO */

