/**
 * @file vtk_volume_output_3D.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-18
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

namespace thunder{ namespace IO { 


vtkSmartPointer<vtkUnstructuredGrid> 
setup_vtk_volume_grid(bool include_gzs)
{
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New() ;
    auto& coord_system = thunder::coordinate_system::get() ; 
    size_t _nx,_ny,_nz; 
    std::tie(_nx,_ny,_nz) = thunder::amr::get_quadrant_extents() ; 
    int ngz = thunder::amr::get_n_ghosts() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    size_t nx = include_gzs ? _nx + 2 * ngz : _nx ; 
    size_t ny = include_gzs ? _ny + 2 * ngz : _ny ; 
    size_t nz = include_gzs ? _nz + 2 * ngz : _nz ;
    size_t ncells = nx*ny*nz*nq ; // these are cells not vertices 
    #ifdef THUNDER_CARTESIAN_COORDINATES
    using cell_type = vtkHexahedron ; 
    size_t constexpr nvertex = 8 ;
    #elif defined(THUNDER_SPHERICAL_COORDINATES)
    using cell_type = vtkBiQuadraticQuadraticHexahedron ; 
    size_t constexpr nvertex = 24 ;
    #endif 

    vtkNew<vtkPoints> points ; 
    points->SetNumberOfPoints( nvertex * ncells ) ; 
     
    auto const get_cell_coordinates = [&] ( size_t icell
                                          , double lx=0
                                          , double ly=0
                                          , double lz=0 )
    {
        /* unpack index assuming LayoutLeft */
        size_t const ix = icell%nx ; 
        size_t const iy = (icell/nx) % ny ;
        size_t const iz = (icell/nx/ny) % nz ; 
        size_t const iq = (icell/nx/ny/nz) ;
        return coord_system.get_physical_coordinates(
            {ix,iy,iz},iq,{lx,ly,lz},include_gzs
        ) ;  
    } ; 

    for( size_t icell=0UL; icell<ncells; icell+=1UL )
    {
        vtkNew<cell_type> cell ; 
        auto par_coords = cell->GetParametricCoords() ;
        for( int iv=0; iv<nvertex; ++iv) { 
            size_t ipoint = iv + icell * nvertex ; 
            auto const coords = get_cell_coordinates( icell 
                                                    , par_coords[3*iv + 0]
                                                    , par_coords[3*iv + 1]
                                                    , par_coords[3*iv + 2] ) ; 
            points->SetPoint(ipoint, coords[0], coords[1], coords[2]) ; 
            cell->GetPointIds()->SetId(iv, ipoint) ; 
        }
        grid->InsertNextCell( cell->GetCellType(), cell->GetPointIds() ) ; 
    }

    grid->SetPoints(points); 
    return grid; 
}

} } /* namespace thunder::IO::detail */