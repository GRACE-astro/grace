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

#include <vtkCellData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkStructuredGrid.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>

#include <thunder/data_structures/variable_properties.hh>

namespace thunder { namespace IO {

void write_volume_cell_data(  std::string const& outfname 
                            , var_array_t<3> _data 
                            , coord_array_t<3> _cell_coordinates 
                            , std::vector<std::string> const& cell_scalar_vars
                            , std::vector<int> const& cell_scalar_idx  
                            , std::vector<std::string> const& cell_vector_vars 
                            , std::vector<std::vector<int>> const& cell_vector_idx 
                            , std::vector<std::string> const& cell_tensor_vars 
                            , std::vector<std::vector<int>> const& cell_tensor_idx
                            )
{
    size_t  nx{ _cell_coordinates.extent(0) }
          , ny{ _cell_coordinates.extent(1) }
          , nz{ _cell_coordinates.extent(2) }
          , nq{ _cell_coordinates.extent(4) } ; 
    
    size_t ncells{ nx * ny * nz * nq } ; 

    /* Set up scalar cell data array */
    vtkNew<vtkDoubleArray> scalar_values ;
    /* initialize number of components and number of elements per component */
    scalar_values->SetNumberOfComponents(cell_scalar_vars.size()) ; 
    scalar_values->SetNumberOfTuples( ncells ) ; 
    /* set variable names */
    for( int ivar=0; ivar<cell_scalar_vars.size(); ++ivar)
    {
        scalar_values->SetComponentName(ivar, cell_scalar_vars[i].c_str() ) ; 
    }
    /* fill data */
    for( int iq=0; iq<nq; ++iq) {
        for( int ik=0; ik<nz; ik++){ 
            for( int ij=0; ij<ny; ij++){ 
                for( int ii=0; ii<nx; ii++){
                    size_t ituple = ii + ny * ( ij + nz * ( ik + nq * iq ) ) ; 
                    for( int icomp=0; icomp<cell_scalar_vars.size(); ++icomp){
                        unsigned int iv = cell_scalar_idx[icomp] ; 
                        scalar_values->SetComponent(ituple, icomp, 
                                                    _data(ii,ij,ik,iv,iq)) ; 
                    }
                }
            }
        }
    }
    vtkNew<vtkPoints> points ; 
    vtkNew<vtkStructuredGrid> grid ; 
    vtkNew<vtkStructuredGridWriter> writer ;
}

}} /* namespace thunder::IO */