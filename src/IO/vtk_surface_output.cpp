/**
 * @file vtk_surface_output.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-17
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

#include <grace/IO/vtk_surface_output.hh>
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
#include <vtkXMLPPolyDataWriter.h>
/* cell types */
#include <vtkCellData.h>
#include <vtkHexahedron.h>
#include <vtkBiQuadraticQuadraticHexahedron.h> 
#include <vtkQuad.h>
#include <vtkQuadraticLinearQuad.h> 
/* slicing utils */
#include <vtkPlane.h>
#include <vtkSphere.h>
#include <vtkCutter.h>
#include <vtkGeometryFilter.h>
/* memory */
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkXMLDataElement.h>
#include <vtkXMLUtilities.h>
/* VTK MPI */
#include <vtkMPI.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
/* grace includes */
#include <grace/data_structures/variable_properties.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/coordinates/coordinate_systems.hh> 
#include <grace/IO/vtk_output.hh>
#include <grace/IO/vtk_surface_output.hh>
#include <grace/IO/vtk_output_auxiliaries.hh>
#include <grace/IO/vtk_output.tpp>
#include <grace/amr/grace_amr.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/errors/error.hh>
#include <grace/errors/assert.hh> 
#include <grace/data_structures/variables.hh>
#include <grace/coordinates/coordinate_systems.hh>
/* Kokkos */
#include <Kokkos_Core.hpp>
/* stdlib includes */
#include <tuple>
#include <array>
#include <string>
#include <filesystem>

namespace grace { namespace IO {

void write_plane_surface_vtk_cell_data( vtkSmartPointer<vtkUnstructuredGrid> grid
                                      , vtkSmartPointer<vtkXMLPPolyDataWriter> pwriter ) {
    Kokkos::Profiling::pushRegion("VTK plane output") ;                                    
    auto& runtime = grace::runtime::get() ;

    int n_planes = runtime.n_surface_output_planes() ; 

    auto plane_origins = runtime.cell_plane_surface_output_origins() ; 
    auto plane_dirs = runtime.cell_plane_surface_output_dirs() ; 
    auto plane_names   = runtime.cell_plane_surface_output_names()   ; 
    
    std::vector<std::array<double,3>> plane_normals(n_planes) ; 
    for ( int ip=0; ip<n_planes; ++ip) {
        for ( int id=0; id<3; ++id) {
            if ( plane_dirs[ip] == id ) {
                plane_normals[ip][id] = 1 ; 
            } else {
                plane_normals[ip][id] = 0 ;
            }
        }
    }

    std::filesystem::path base_path (runtime.surface_io_basepath()) ;

    setup_volume_cell_data(grid, grace_vtk_output_t::PLANE_SURFACE) ; 

    detail::_surface_plane_iterations.push_back(runtime.iteration())  ;
    detail::_surface_plane_times.push_back(runtime.time())            ; 

    for ( int iplane=0; iplane<n_planes; ++iplane ) {
        std::string pfname = runtime.surface_io_basename() + "_" + utils::zero_padded(runtime.iteration(),3)
                                                           + "_plane_" 
                                                           + plane_names[iplane] + ".pvtp" ;
         
        std::filesystem::path out_path = base_path / pfname ;

        vtkSmartPointer<vtkPlane> plane = vtkSmartPointer<vtkPlane>::New();
        plane->SetOrigin( plane_origins[iplane][0]
                        , plane_origins[iplane][1]
                        , plane_origins[iplane][2] ); // Center of the plane
        plane->SetNormal( plane_normals[iplane][0]
                        , plane_normals[iplane][1]
                        , plane_normals[iplane][2] ); // Normal to the plane

        // Set up the cutter with the plane
        vtkSmartPointer<vtkCutter> cutter = vtkSmartPointer<vtkCutter>::New();
        cutter->SetCutFunction(plane);
        cutter->SetInputData(grid);
        cutter->Update();

        pwriter->SetFileName(out_path.string().c_str()) ; 
        pwriter->SetInputData( cutter->GetOutput() ) ; 
        pwriter->Write() ;

        detail::_surface_plane_filenames[iplane].push_back(pfname)        ;
        if( parallel::mpi_comm_rank() == 0 ) {
            GRACE_TRACE("Plane {} npoints {}\n origin {} {} {}\n normal {} {} {}."
                         ,plane_names[iplane],cutter->GetOutput()->GetNumberOfPoints()
                         ,plane_origins[iplane][0],plane_origins[iplane][1],plane_origins[iplane][2]
                         ,plane_normals[iplane][0],plane_normals[iplane][1],plane_normals[iplane][2]);
            std::string pvd_basefilename = runtime.surface_io_basename() 
                                        + "_plane_" + plane_names[iplane]
                                        + ".pvd" ; 
            std::filesystem::path pvd_filename = base_path / pvd_basefilename ;
            write_pvd_file( pvd_filename.string()
                          , detail::_surface_plane_filenames[iplane]
                          , detail::_surface_plane_times ) ;
        }
    }
    Kokkos::Profiling::popRegion() ;
    return ; 
}

void write_sphere_surface_vtk_cell_data( vtkSmartPointer<vtkUnstructuredGrid> grid
                                       , vtkSmartPointer<vtkXMLPPolyDataWriter> pwriter ) {
    Kokkos::Profiling::pushRegion("VTK sphere output") ; 
    auto& runtime = grace::runtime::get() ;

    int n_spheres = runtime.n_surface_output_spheres() ; 

    auto sphere_centers = runtime.cell_sphere_surface_output_centers() ; 
    auto sphere_radii   = runtime.cell_sphere_surface_output_radii()   ; 
    auto sphere_names   = runtime.cell_sphere_surface_output_names()   ; 

    std::filesystem::path base_path (runtime.surface_io_basepath()) ;

    setup_volume_cell_data(grid, grace_vtk_output_t::SPHERE_SURFACE) ;
    detail::_surface_sphere_iterations.push_back(runtime.iteration())  ;
    detail::_surface_sphere_times.push_back(runtime.time())            ; 
    for ( int isphere=0; isphere<n_spheres; ++isphere ) {
        std::string pfname = runtime.surface_io_basename() + "_" + utils::zero_padded(runtime.iteration(),3)
                                                           + "_sphere_" 
                                                           + sphere_names[isphere] + ".pvtp" ;
        std::filesystem::path out_path = base_path / pfname ;

        vtkSmartPointer<vtkSphere> sphere = vtkSmartPointer<vtkSphere>::New();
        sphere->SetCenter( sphere_centers[isphere][0]
                         , sphere_centers[isphere][1]
                         , sphere_centers[isphere][2] ); // Center of the plane
        sphere->SetRadius( sphere_radii[isphere] ); // Normal to the plane

        // Set up the cutter with the plane
        vtkSmartPointer<vtkCutter> cutter = vtkSmartPointer<vtkCutter>::New();
        cutter->SetCutFunction(sphere);
        cutter->SetInputData(grid);
        cutter->Update();

        pwriter->SetFileName(out_path.string().c_str()) ; 
        pwriter->SetInputData( cutter->GetOutput() ) ; 
        pwriter->Write() ;

        detail::_surface_sphere_filenames[isphere].push_back(pfname)        ;
        
        if( parallel::mpi_comm_rank() == 0 ) {
            std::string pvd_basefilename = runtime.surface_io_basename() 
                                        + "_sphere_" + sphere_names[isphere]
                                        + ".pvd" ; 
            std::filesystem::path pvd_filename = base_path / pvd_basefilename ;
            write_pvd_file( pvd_filename.string()
                          , detail::_surface_sphere_filenames[isphere]
                          , detail::_surface_sphere_times ) ;
        }
    }
    Kokkos::Profiling::popRegion() ;
    return ; 
}



}}
