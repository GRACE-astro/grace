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
#include <vtkUnStructuredGrid.h>
/* points */
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
/* writers */
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
/* cell types */
#include <vtkCellData.h>
#include <vtkHexahedron.h>
#include <vtkTriQuadraticHexahedron.h> 
/* memory */
#include <vtkSmartPointer.h>
#include <vtkNew.h>
/* thunder includes */
#include <thunder/data_structures/variable_properties.hh>
#include <thunder/system/thunder_runtime.hh> 

#include <thunder/IO/vtk_volume_output_3D.hh>

#include <thunder/amr/amr_functions.hh>

#include <thunder/errors/error.hh>
#include <thunder/errors/assert.hh> 

#include <tuple>
#include <array>
#include <string>

namespace thunder { namespace IO {


namespace detail {

vtkSmartPointer<vtkUnstructuredGrid> 
setup_vtk_grid_cartesian()
{
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New() ;

    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    size_t ncells = nx*ny*nz*nq ; // these are cells not vertices 
    size_t nvertex = 8 ; 
    using cell_type = vtkHexahedron ; 

    vtkNew<vtkPoints> points ; 
    points->SetNumberOfPoints( nvertex * ncells ) ; 

    auto const get_cell_coordinates = [&] ( size_t icell
                                          , double lx=0
                                          , double ly=0
                                          , double lz=0 )
    {
        size_t n = ny*nz*nq ; 
        size_t const ix = icell%n ; 
        size_t idx = icell / n ;
        n = nz*nq ; 
        size_t const iy = idx % n ;
        idx /= n ;
        n = nq ; 
        size_t const iz = idx %  n ; 
        size_t const iq = idx / n ;
        size_t itree = thunder::amr::get_quadrant_owner(iq) ; 
        auto const tree_coords = thunder::amr::get_tree_vertex(itree, 0UL) ; 
        auto const dx_tree     = thunder::amr::get_tree_spacing(itree)[0]  ;
        quadrant_t const quadrant = thunder::amr::get_quadrant(itree, iq) ;  
        auto const dx_quad        = dx_tree / ( 1<<quadrant.level()) ; 
        auto const qcoords        = quadrant.qcoords() ; 
        auto const dx_cell        = dx_quad / nx ;
        auto const dy_cell        = dx_quad / ny ; 
        auto const dz_cell        = dx_quad / nz ; 
        return std::array<double,3> {
            tree_coords[0] + qcoords[0] * dx_quad + (ix + 0.5 + lx) * dx_cell, 
            tree_coords[1] + qcoords[1] * dx_quad + (iy + 0.5 + ly) * dy_cell, 
            tree_coords[2] + qcoords[2] * dx_quad + (iz + 0.5 + lz) * dz_cell 
        } ; 
    } ; 

    for( size_t icell=0UL; icell<ncells; icell+=1UL )
    {
        vtkNew<cell_type> cell ; 
        auto lcoords = cell->GetParametricCoords() ;
        for( int iv=0; iv<nvertex; ++iv) { 
            size_t ipoint = iv + icell * nvertex ; 
            auto const coords = get_cell_coordinates( icell 
                                                    . lcoords[0]
                                                    , lcoords[1]
                                                    , lcoords[2] ) ; 
            points->SetPoint(ipoint, coords[0], coords[1], coords[2]) ; 
            cell->GetPointIds()->SetId(iv, ipoint) ; 
        }
        grid->->InsertNextCell( cell->GetCellType(), cell->GetPointIds() ) ; 
    }

    grid->SetPoints(points); 
    return grid; 
}

} /* namespace detail */

vtkSmartPointer<vtkUnstructuredGrid> setup_vtk_grid()
{
    std::string coord_system = params["amr"]["physical_coordinates"].as<std::string>() ; 
    if( coord_system == "cartesian" ) {
        return detail::setup_vtk_grid_cartesian() ; 
    } else if (coord_system == "spherical" ) {
        return detail::setup_vtk_grid_spherical() ; 
    } else {
        ERROR("Should not be here.") ; 
    }
}

void write_volume_cell_data(vtkSmartPointer<vtkUnstructuredGrid>) ; 



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