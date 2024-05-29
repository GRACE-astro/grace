/**
 * @file vtk_output.cpp
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
#include <grace_config.h>
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
#include <vtkXMLPPolyDataWriter.h>
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
/* grace includes */
#include <grace/data_structures/variable_properties.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/coordinates/coordinate_systems.hh> 
#include <grace/IO/vtk_output.tpp>
#include <grace/IO/vtk_output.hh>
#include <grace/IO/vtk_setup_grid.hh>
#include <grace/IO/vtk_volume_output.hh>
#include <grace/IO/vtk_output_auxiliaries.hh>
#include <grace/IO/vtk_surface_output.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/errors/error.hh>
#include <grace/errors/assert.hh> 
#include <grace/data_structures/variables.hh>
/* stdlib includes */
#include <tuple>
#include <array>
#include <string>
#include <filesystem>

namespace grace { namespace IO {


void write_cell_data_vtk(bool volume_output, bool surface_output_plane, bool surface_output_sphere )
{
    if(     (not volume_output)
        and (not surface_output_plane)
        and (not surface_output_sphere) )
    {
        return ; 
    }
    Kokkos::Profiling::pushRegion("VTK cell output") ; 
    vtkSmartPointer<vtkMPICommunicator> vtk_comm 
        = vtkSmartPointer<vtkMPICommunicator>::New();
    auto mpi_comm = parallel::get_comm_world() ;
    vtkMPICommunicatorOpaqueComm vtk_opaque_comm(&mpi_comm);
    vtk_comm->InitializeExternal(&vtk_opaque_comm);

    vtkSmartPointer<vtkMPIController> vtk_mpi_ctrl 
        = vtkSmartPointer<vtkMPIController>::New();
    vtk_mpi_ctrl->SetCommunicator(vtk_comm);

    vtkNew<vtkXMLPUnstructuredGridWriter> pwriter ;
    pwriter->SetController(vtk_mpi_ctrl);
    pwriter->SetNumberOfPieces( parallel::mpi_comm_size() ) ; 
    pwriter->SetStartPiece(parallel::mpi_comm_rank()) ;
    pwriter->SetEndPiece(parallel::mpi_comm_rank()) ;
    pwriter->SetDataModeToBinary() ; 
    pwriter->SetCompressorTypeToZLib();  

    vtkNew<vtkXMLPPolyDataWriter> ppolywriter ;
    ppolywriter->SetController(vtk_mpi_ctrl);
    ppolywriter->SetNumberOfPieces( parallel::mpi_comm_size() ) ; 
    ppolywriter->SetStartPiece(parallel::mpi_comm_rank()) ;
    ppolywriter->SetEndPiece(parallel::mpi_comm_rank()) ;
    ppolywriter->SetDataModeToBinary() ; 
    ppolywriter->SetCompressorTypeToZLib();  


    auto grid = setup_vtk_volume_grid() ; 

    if( volume_output ) {
        write_volume_vtk_cell_data(grid,pwriter) ; 
    }
    if( surface_output_plane ) {
        write_plane_surface_vtk_cell_data(grid,ppolywriter) ; 
    } 
    if ( surface_output_sphere ) {
        write_sphere_surface_vtk_cell_data(grid,ppolywriter) ; 
    }
    Kokkos::Profiling::popRegion() ;
    return ; 
}

void setup_volume_cell_data(vtkSmartPointer<vtkUnstructuredGrid> grid, size_t which_output) {

    // Clear any point data arrays
    grid->GetPointData()->Initialize();
    // Clear any cell data arrays
    grid->GetCellData()->Initialize();

    auto& runtime = grace::runtime::get() ;
    auto params  = grace::config_parser::get()["IO"] ; 
    
    bool output_extra = params["output_extra_quantities"].as<bool>() ; 
    std::vector<std::string> scalars, aux_scalars, vectors, aux_vectors ; 
    if( which_output == VOLUME ) {
        scalars     = runtime.cell_volume_output_scalar_vars() ; 
        aux_scalars = runtime.cell_volume_output_scalar_aux() ;
        vectors     = runtime.cell_volume_output_vector_vars() ; 
        aux_vectors = runtime.cell_volume_output_vector_aux() ; 
    } else if ( which_output == PLANE_SURFACE ) {
        scalars     = runtime.cell_plane_surface_output_scalar_vars() ; 
        aux_scalars = runtime.cell_plane_surface_output_scalar_aux()  ;
        vectors     = runtime.cell_plane_surface_output_vector_vars() ; 
        aux_vectors = runtime.cell_plane_surface_output_vector_aux()  ; 
    } else if ( which_output == SPHERE_SURFACE ) {
        scalars     = runtime.cell_sphere_surface_output_scalar_vars() ; 
        aux_scalars = runtime.cell_sphere_surface_output_scalar_aux()  ;
        vectors     = runtime.cell_sphere_surface_output_vector_vars() ; 
        aux_vectors = runtime.cell_sphere_surface_output_vector_aux()  ; 
    }
    size_t nx,ny,nz,nq; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = grace::amr::get_n_ghosts() ; 

    auto vars = grace::variable_list::get().getstate() ; 
    auto aux  = grace::variable_list::get().getaux()   ;
    using exec_space = decltype(vars)::execution_space ; 


    if( output_extra ) {
        add_extra_output_quantities(grid, false) ;  
    }
    auto h_mirror = Kokkos::create_mirror_view(vars) ;
    Kokkos::deep_copy(h_mirror, vars ) ; 
    auto aux_h_mirror = Kokkos::create_mirror_view(aux) ; 
    Kokkos::deep_copy(exec_space{}, aux_h_mirror, aux)  ; 
    for( int ivar=0; ivar<scalars.size(); ++ivar )
    {
        size_t varidx = grace::get_variable_index(scalars[ivar]) ; 
        
        auto h_sview = Kokkos::subview(h_mirror           , VEC( Kokkos::pair(ngz,nx+ngz)
                                                               , Kokkos::pair(ngz,ny+ngz)
                                                               , Kokkos::pair(ngz,nz+ngz) )
                                                          , ivar
                                                          , Kokkos::ALL()
                                                          ) ;  
        grid->GetCellData()->AddArray(vtk_create_cell_data(VEC(nx,ny,nz),nq,h_sview,scalars[ivar])) ;

    }
    
    for( int ivar=0; ivar<vectors.size(); ++ivar )
    {
        size_t varidx = grace::get_variable_index(vectors[ivar]+"[0]") ; 
        
        auto h_sview = Kokkos::subview(h_mirror           , VEC(  Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::pair(ivar,ivar+GRACE_NSPACEDIM)
                                                          , Kokkos::ALL()
                                                          ) ;  
        grid->GetCellData()->AddArray(vtk_create_vector_cell_data(VEC(nx,ny,nz),nq,h_sview,vectors[ivar])) ;

    }
    
    Kokkos::fence() ;
    for( int ivar=0; ivar<aux_scalars.size(); ++ivar )
    {
        size_t varidx = grace::get_variable_index(aux_scalars[ivar],true) ; 
        
        auto h_sview = Kokkos::subview(aux_h_mirror       , VEC( Kokkos::pair(ngz,nx+ngz)
                                                               , Kokkos::pair(ngz,ny+ngz)
                                                               , Kokkos::pair(ngz,nz+ngz) )
                                                          , ivar
                                                          , Kokkos::ALL()
                                                          ) ;  
        grid->GetCellData()->AddArray(vtk_create_cell_data(VEC(nx,ny,nz),nq,h_sview,aux_scalars[ivar])) ;

    }
    
    for( int ivar=0; ivar<aux_vectors.size(); ++ivar )
    {
        size_t varidx = grace::get_variable_index(aux_vectors[ivar]+"[0]",true) ; 
        
        auto h_sview = Kokkos::subview(aux_h_mirror       , VEC(  Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::pair(ivar,ivar+GRACE_NSPACEDIM)
                                                          , Kokkos::ALL()
                                                          ) ;  
        grid->GetCellData()->AddArray(vtk_create_vector_cell_data(VEC(nx,ny,nz),nq,h_sview,aux_vectors[ivar])) ;

    }
    
}; 

void write_pvd_file( std::string const& pvdFilename
                   , std::vector<std::string> const& vtkFilenames 
                   , std::vector<double> const& times ) {

    vtkNew<vtkXMLDataElement> collection;
    collection->SetName("Collection");
    for (size_t i = 0; i < times.size(); ++i) {
        vtkNew<vtkXMLDataElement> dataSet;
        dataSet->SetName("DataSet");
        dataSet->SetDoubleAttribute("timestep", times[i]);
        dataSet->SetAttribute("group", "");
        dataSet->SetAttribute("part", "0");
        dataSet->SetAttribute("file", vtkFilenames[i].c_str());
        collection->AddNestedElement(dataSet);
    }

    vtkNew<vtkXMLDataElement> root;
    root->SetName("VTKFile");
    root->SetAttribute("type", "Collection");
    root->SetAttribute("version", "0.1");
    root->SetAttribute("byte_order", "LittleEndian");
    root->SetAttribute("compressor", "vtkZLibDataCompressor");
    root->AddNestedElement(collection);

    vtkNew<vtkXMLUtilities> xmlUtilities;
    xmlUtilities->WriteElementToFile(root, pvdFilename.c_str(), nullptr);
}


void add_extra_output_quantities(vtkSmartPointer<vtkUnstructuredGrid> grid, bool include_gzs)
{
    using namespace grace ; 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    int ngz = grace::amr::get_n_ghosts() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    size_t ncells = EXPR(nx,*ny,*nz)*nq ;
    auto vol_mirror = Kokkos::create_mirror_view(grace::variable_list::get().getvolumes()) ; 
    Kokkos::deep_copy(vol_mirror,grace::variable_list::get().getvolumes()) ; 
    auto mpi_rank_array      = vtkSmartPointer<vtkIntArray>::New() ; 
    auto treeid_array        = vtkSmartPointer<vtkIntArray>::New() ; 
    auto quadid_array        = vtkSmartPointer<vtkIntArray>::New() ;
    auto reflevel_array      = vtkSmartPointer<vtkIntArray>::New() ;
    auto lcoords_array       = vtkSmartPointer<vtkDoubleArray>::New() ;
    auto pcoords_array       = vtkSmartPointer<vtkDoubleArray>::New() ;
    auto cell_volumes_array  = vtkSmartPointer<vtkDoubleArray>::New() ;

    mpi_rank_array->SetNumberOfTuples(ncells) ; 
    mpi_rank_array->SetNumberOfComponents(1) ;
    mpi_rank_array->SetName("owner_rank_id") ; 

    treeid_array->SetNumberOfTuples(ncells) ; 
    treeid_array->SetNumberOfComponents(1) ;
    treeid_array->SetName("tree_id") ;

    quadid_array->SetNumberOfTuples(ncells) ; 
    quadid_array->SetNumberOfComponents(1) ;
    quadid_array->SetName("quadrant_global_id") ;

    reflevel_array->SetNumberOfTuples(ncells) ; 
    reflevel_array->SetNumberOfComponents(1) ;
    reflevel_array->SetName("refinement_level") ;

    lcoords_array->SetNumberOfTuples(ncells*GRACE_NSPACEDIM) ; 
    lcoords_array->SetNumberOfComponents(GRACE_NSPACEDIM) ;
    lcoords_array->SetName("logical_coordinates") ;

    pcoords_array->SetNumberOfTuples(ncells*GRACE_NSPACEDIM) ; 
    pcoords_array->SetNumberOfComponents(GRACE_NSPACEDIM) ;
    pcoords_array->SetName("physical_coordinates") ;

    cell_volumes_array->SetNumberOfTuples(ncells) ; 
    cell_volumes_array->SetNumberOfComponents(1) ;
    cell_volumes_array->SetName("cell_volumes") ;

    auto mpi_rank = parallel::mpi_comm_rank() ; 

    auto& coord_system = coordinate_system::get() ; 
    
    for(size_t iq=0UL; iq<nq ; iq+=1UL) {
        for(size_t ix=0UL; ix<nx; ix+=1UL) {
            for(size_t iy=0UL; iy<ny; iy+=1UL) {
                #ifdef GRACE_3D
                for(size_t iz=0UL; iz<nz; iz+=1UL) {
                #endif 
                    #ifndef GRACE_3D 
                    size_t icell = ix + nx*(iy+ny*iq) ;
                    #else 
                    size_t icell = ix + nx*(iy+ny*(iz+nz*iq)) ;
                    #endif 
                    auto itree = amr::get_quadrant_owner(iq) ; 
                    auto quad  = amr::get_quadrant(itree,iq) ; 
                    auto level = quad.level() ; 
                    size_t iquad_glob = iq + amr::forest::get().global_quadrant_offset(mpi_rank) ; 
                    auto const lcoords = coord_system.get_logical_coordinates(
                        {VEC(ix,iy,iz)}, 
                        iq,
                        {VEC(0.5,0.5,0.5)},
                        false
                    ) ; 
                    auto const pcoords = coord_system.get_physical_coordinates(
                        {VEC(ix,iy,iz)}, 
                        iq,
                        {VEC(0.5,0.5,0.5)},
                        false
                    ) ;
                    mpi_rank_array->SetComponent(icell, 0, mpi_rank) ;
                    treeid_array->SetComponent(icell, 0, itree) ;
                    quadid_array->SetComponent(icell, 0, iquad_glob) ;
                    reflevel_array->SetComponent(icell, 0, level) ;
                    cell_volumes_array->SetComponent(icell, 0, vol_mirror(VEC(ix+ngz,iy+ngz,iz+ngz),iq)) ;
                    for(int idim=0; idim<GRACE_NSPACEDIM; ++idim){
                        lcoords_array->SetComponent(icell, idim, lcoords[idim]) ;
                        pcoords_array->SetComponent(icell, idim, pcoords[idim]) ;
                    }
                #ifdef GRACE_3D
                }
                #endif 
            }
        }
    }

    grid->GetCellData()->AddArray(mpi_rank_array) ; 
    grid->GetCellData()->AddArray(treeid_array) ; 
    grid->GetCellData()->AddArray(quadid_array) ; 
    grid->GetCellData()->AddArray(reflevel_array) ; 
    grid->GetCellData()->AddArray(lcoords_array) ;
    grid->GetCellData()->AddArray(pcoords_array) ;
    grid->GetCellData()->AddArray(cell_volumes_array) ;
}

}}

