/**
 * @file vtk_volume_output_2D.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
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
#include <grace_config.h>

#include <grace/IO/vtk_setup_grid.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/utils/grace_utils.hh>
/* VTK includes */
/* grid type */
#include <vtkUnstructuredGrid.h>
/* points */
#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPoints.h>
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

namespace grace{ namespace IO { 

vtkSmartPointer<vtkUnstructuredGrid> 
setup_vtk_volume_grid()
{
    vtkSmartPointer<vtkUnstructuredGrid> grid 
        = vtkSmartPointer<vtkUnstructuredGrid>::New() ;
    auto& coord_system = grace::coordinate_system::get() ; 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    int ngz = grace::amr::get_n_ghosts() ;
    size_t nq = grace::amr::get_local_num_quadrants() ; 

    size_t ncells = EXPR(nx,*ny,*nz)*nq ; // these are cells not vertices 
    #ifdef GRACE_3D 
    #ifdef GRACE_CARTESIAN_COORDINATES
    using cell_type = vtkHexahedron ; 
    size_t constexpr nvertex = 8 ;
    #elif defined(GRACE_SPHERICAL_COORDINATES)
    using cell_type = vtkBiQuadraticQuadraticHexahedron ; 
    size_t constexpr nvertex = 24 ;
    #endif 
    #else 
    #ifdef GRACE_CARTESIAN_COORDINATES
    size_t nvertex = 4 ; 
    using cell_type = vtkQuad ; 
    #elif defined(GRACE_SPHERICAL_COORDINATES)
    size_t nvertex = 6 ; 
    using cell_type = vtkQuadraticLinearQuad ;
    #endif 
    #endif 
    vtkNew<vtkPoints> points ; 
    points->SetNumberOfPoints( nvertex * ncells ) ; 
   
    auto const get_cell_coordinates = [&] ( size_t icell
                                          , double lx=0
                                          , double ly=0
                                          , double lz=0 )
    {
        /* unpack index assuming LayoutLeft */
        #ifdef GRACE_3D
        size_t const ix = icell%nx ; 
        size_t const iy = (icell/nx) % ny ;
        size_t const iz = (icell/nx/ny) % nz ; 
        size_t const iq = (icell/nx/ny/nz) ;
        #else
        size_t const ix = icell%nx ; 
        size_t const iy = (icell/nx) % ny ;
        size_t const iq = (icell/nx/ny) ;
        #endif 
        return coord_system.get_physical_coordinates(
            {VEC(ix,iy,iz)},iq,{VEC(lx,ly,lz)},false
        ) ; 
    } ; 
    vtkNew<cell_type> _tmpcell ;
    auto lcoords = _tmpcell->GetParametricCoords() ;
    for( size_t icell=0UL; icell<ncells; icell+=1UL )
    {
        vtkNew<cell_type> cell ; 
        for( int iv=0; iv<nvertex; ++iv) { 
            size_t ipoint = iv + icell * nvertex ; 
            auto const coords = get_cell_coordinates( icell 
                                                    , lcoords[3*iv + 0]
                                                    , lcoords[3*iv + 1]
                                                    #ifdef GRACE_3D 
                                                    , lcoords[3*iv + 2]
                                                    #endif 
                                                    ) ; 
            points->SetPoint( ipoint
                            , coords[0]
                            , coords[1]
                            #ifdef GRACE_3D
                            , coords[2]
                            #else 
                            , 0.0
                            #endif 
                            ) ; 
            cell->GetPointIds()->SetId(iv, ipoint) ; 
        }
        grid->InsertNextCell( cell->GetCellType(), cell->GetPointIds() ) ; 
    }

    grid->SetPoints(points); 
    return grid; 
}


} }  /* namespace grace::IO::detail */