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
#include <vtkPoints.h>
/* writers */
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
/* cell types */
#include <vtkCellData.h>
#ifdef THUNDER_3D 
#include <vtkHexahedron.h>
#include <vtkBiQuadraticQuadraticHexahedron.h> 
#else 
#include <vtkQuad.h>
#include <vtkQuadraticLinearQuad.h> 
#endif 
/* memory */
#include <vtkSmartPointer.h>
#include <vtkNew.h>
/* VTK MPI */
#include <vtkMPI.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
/* thunder includes */
#include <thunder/data_structures/variable_properties.hh>
#include <thunder/system/thunder_runtime.hh> 
#include <thunder/IO/vtk_volume_output.hh>
#ifdef THUNDER_3D 
#include <thunder/IO/vtk_output_3D.tpp>
#else 
#include <thunder/IO/vtk_output_2D.tpp>
#endif 
#include <thunder/amr/thunder_amr.hh>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/errors/error.hh>
#include <thunder/errors/assert.hh> 
#include <thunder/data_structures/variables.hh>
/* stdlib includes */
#include <tuple>
#include <array>
#include <string>
#include <filesystem>


namespace thunder { namespace IO {


vtkSmartPointer<vtkUnstructuredGrid> setup_vtk_volume_grid()
{
    auto& params = thunder::config_parser::get() ; 
    std::string coord_system = params["amr"]["physical_coordinates"].as<std::string>() ; 
    if( coord_system == "cartesian" ) {
        return detail::setup_vtk_grid_cartesian() ; 
    } else if (coord_system == "spherical" ) {
        std::cout << "Setting up spherical grid..." << std::endl ;
        return detail::setup_vtk_grid_spherical() ; 
    } else {
        ERROR("Should not be here.") ; 
    }
}



vtkSmartPointer<vtkUnstructuredGrid> setup_volume_cell_data() {

    auto& runtime = thunder::runtime::get() ;

    auto const scalars     = runtime.cell_volume_output_scalar_vars() ; 
    auto const aux_scalars = runtime.cell_volume_output_scalar_aux() ;
    auto const vectors     = runtime.cell_volume_output_vector_vars() ; 
    auto const aux_vectors = runtime.cell_volume_output_vector_aux() ; 
    size_t nx,ny,nz,nq; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    nq = thunder::amr::get_local_num_quadrants() ; 
    size_t ngz = thunder::amr::get_n_ghosts() ; 

    auto vars = thunder::variable_list::get().getstate() ; 
    auto aux  = thunder::variable_list::get().getaux()   ; 
    auto scalar_host_mirror = 
    variable_properties_t<THUNDER_NSPACEDIM>::view_t::HostMirror(
          "scalar_output_mirror"
        , VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz)
        ,  nq
        ,  scalars.size() 
    ) ; 
    auto vector_host_mirror = 
    variable_properties_t<THUNDER_NSPACEDIM>::view_t::HostMirror(
          "vector_output_mirror"
        , VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz)
        ,  nq 
        ,  3 * vectors.size() 
    ) ; 
    using exec_space = decltype(vars)::execution_space ; 
    auto grid = setup_vtk_volume_grid() ; 
    
    for( int ivar=0; ivar<scalars.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(scalars[ivar]) ; 
        auto h_sview = Kokkos::subview(scalar_host_mirror, VEC(  Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::ALL()
                                                          , ivar) ; 
        auto d_sview = Kokkos::subview(vars               , VEC( Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::ALL()
                                                          , varidx) ;
        Kokkos::deep_copy(h_sview, d_sview ) ;
    }
    for( int ivar=0; ivar<scalars.size(); ++ivar )
    {
        auto h_sview = Kokkos::subview(scalar_host_mirror, VEC(  Kokkos::pair(ngz, nx+ngz)
                                                               , Kokkos::pair(ngz, ny+ngz)
                                                               , Kokkos::pair(ngz, nz+ngz) )
                                                          , Kokkos::ALL()
                                                          , ivar
                                                          ) ;
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,scalars[ivar])) ; 
    }
    
    /* 
    * In the following we perform a series of 
    * calls to deep_copy for each variable group 
    * separately. These calls are asynchronous 
    * when the backend is CUDA or HIP and this 
    * allows us to overlap the data transfer 
    * from device to host and the writing of 
    * data to the vtkUnstructuredGrid object. 
    */
    /*
    for( int ivar=0; ivar<scalars.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(scalars[ivar]) ; 
        auto h_sview = Kokkos::subview(scalar_host_mirror, VEC(  Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::ALL()
                                                          , ivar) ; 
        auto d_sview = Kokkos::subview(vars               , VEC( Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::ALL()
                                                          , varidx) ;
        Kokkos::deep_copy(exec_space{}, h_sview, d_sview ) ;
    }
    Kokkos::fence() ; 
    std::cout << "Copied scalars...  " << std::endl ;
    for( int ivar=0; ivar<vectors.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(vectors[ivar]+"[0]") ; 
        auto h_sview = Kokkos::subview(vector_host_mirror, VEC(  Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(ivar, ivar+3)
                                                          ) ; 
        auto d_sview = Kokkos::subview(vars               , VEC( Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(varidx, varidx+3)
                                                          ) ;
        Kokkos::deep_copy(exec_space{}, h_sview, d_sview ) ;
    }

    for( int ivar=0; ivar<scalars.size(); ++ivar )
    {
        auto h_sview = Kokkos::subview(scalar_host_mirror, VEC(  Kokkos::pair(ngz, nx+ngz)
                                                               , Kokkos::pair(ngz, ny+ngz)
                                                               , Kokkos::pair(ngz, nz+ngz) )
                                                          , Kokkos::ALL()
                                                          , ivar
                                                          ) ;
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,scalars[ivar])) ; 
    }

    Kokkos::fence() ; 
    std::cout << "Copied vectors...  " << std::endl ;
    
    Kokkos::resize(  scalar_host_mirror
                  ,  VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz)
                  ,  nq 
                  ,  aux_scalars.size() 
                  ) ; 
    for( int ivar=0; ivar<aux_scalars.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(aux_scalars[ivar]) ; 
        auto h_sview = Kokkos::subview(scalar_host_mirror, VEC(  Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::ALL()
                                                          , ivar
                                                          ) ; 
        auto d_sview = Kokkos::subview(aux                , VEC( Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() ) 
                                                          , Kokkos::ALL()
                                                          , varidx
                                                          ) ;
        Kokkos::deep_copy(exec_space{}, h_sview, d_sview ) ;
    }

    for( int ivar=0; ivar<vectors.size(); ++ivar )
    {
        auto h_sview = Kokkos::subview(vector_host_mirror, VEC(  Kokkos::pair(ngz, nx+ngz)
                                                               , Kokkos::pair(ngz, ny+ngz)
                                                               , Kokkos::pair(ngz, nz+ngz) )
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(ivar, ivar+3)
                                                          ) ;
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,vectors[ivar])) ; 
    }
    Kokkos::fence() ; 
    
    Kokkos::resize(  vector_host_mirror
                  ,  VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz)
                  ,  nq
                  ,  3*aux_vectors.size() 
                ) ; 
    for( int ivar=0; ivar<aux_vectors.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(aux_vectors[ivar]+"[0]") ; 
        auto h_sview = Kokkos::subview(scalar_host_mirror, VEC(  Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(ivar, ivar+3)
                                                          ) ; 
        auto d_sview = Kokkos::subview(aux                , VEC( Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() ) 
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(varidx, varidx+3)
                                                          ) ;
        Kokkos::deep_copy(exec_space{}, h_sview, d_sview ) ;
    }

    for( int ivar=0; ivar<aux_scalars.size(); ++ivar )
    {
        auto h_sview = Kokkos::subview(scalar_host_mirror, VEC(  Kokkos::pair(ngz, nx+ngz)
                                                               , Kokkos::pair(ngz, ny+ngz)
                                                               , Kokkos::pair(ngz, nz+ngz) )
                                                          , Kokkos::ALL()
                                                          , ivar
                                                          ) ;
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,aux_scalars[ivar])) ; 
    }
    Kokkos::fence() ; 
    for( int ivar=0; ivar<aux_vectors.size(); ++ivar )
    {
        auto h_sview = Kokkos::subview(vector_host_mirror, VEC(  Kokkos::pair(ngz, nx+ngz)
                                                               , Kokkos::pair(ngz, ny+ngz)
                                                               , Kokkos::pair(ngz, nz+ngz) )
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(ivar, ivar+3)
                                                          ) ;
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,aux_vectors[ivar])) ; 
    }
    */
    return grid ; 
}; 

void write_volume_cell_data() 
{   
    auto& runtime = thunder::runtime::get() ;
    std::filesystem::path base_path (runtime.volume_io_basepath()) ;
    std::string pfname = runtime.volume_io_basename() + "_" + utils::zero_padded(runtime.iteration(),3)
                                                            + ".pvtu" ;
     
    std::filesystem::path out_path = base_path / pfname ;

    auto grid = setup_volume_cell_data() ; 

    vtkNew<vtkXMLPUnstructuredGridWriter> pwriter ;

    vtkSmartPointer<vtkMPICommunicator> vtk_comm 
        = vtkSmartPointer<vtkMPICommunicator>::New();
    auto mpi_comm = parallel::get_comm_world() ;
    vtkMPICommunicatorOpaqueComm vtk_opaque_comm(&mpi_comm);
    vtk_comm->InitializeExternal(&vtk_opaque_comm);

    vtkSmartPointer<vtkMPIController> vtk_mpi_ctrl 
        = vtkSmartPointer<vtkMPIController>::New();
    vtk_mpi_ctrl->SetCommunicator(vtk_comm);

    pwriter->SetController(vtk_mpi_ctrl);
    
    pwriter->SetFileName(out_path.string().c_str()) ; 
    pwriter->SetNumberOfPieces( parallel::mpi_comm_size() ) ; 
    pwriter->SetInputData( grid ) ;
    pwriter->SetStartPiece(parallel::mpi_comm_rank()) ;
    pwriter->SetEndPiece(parallel::mpi_comm_rank()) ;
    pwriter->SetDataModeToBinary() ; 
    pwriter->SetCompressorTypeToZLib();  
    pwriter->Write() ; 
    


}

}} /* namespace thunder::IO */

#ifdef THUNDER_3D 
#include "vtk_volume_output_3D.cpp"
#else 
#include "vtk_volume_output_2D.cpp"
#endif 