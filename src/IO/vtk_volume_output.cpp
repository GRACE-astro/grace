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
#include <thunder/coordinates/coordinate_systems.hh> 
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
#include <thunder/coordinates/coordinate_systems.hh>
/* stdlib includes */
#include <tuple>
#include <array>
#include <string>
#include <filesystem>


namespace thunder { namespace IO {



vtkSmartPointer<vtkUnstructuredGrid> setup_volume_cell_data() {

    auto& runtime = thunder::runtime::get() ;
    auto params  = thunder::config_parser::get()["IO"] ; 
    bool include_gzs = params["output_ghostzones"].as<bool>() ;
    include_gzs = false ; 
    bool output_extra = params["output_extra_quantities"].as<bool>() ; 
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
    using exec_space = decltype(vars)::execution_space ; 
    auto grid = setup_vtk_volume_grid(include_gzs) ; 
    if( include_gzs ) {
        flag_ghost_cells(grid) ; 
    }
    if( output_extra ) {
        add_extra_output_quantities(grid, include_gzs) ;  
    }
    auto h_mirror = Kokkos::create_mirror_view(vars) ;
    Kokkos::deep_copy(h_mirror, vars ) ; 
    auto aux_h_mirror = Kokkos::create_mirror_view(aux) ; 
    Kokkos::deep_copy(exec_space{}, aux_h_mirror, aux)  ; 
    for( int ivar=0; ivar<scalars.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(scalars[ivar]) ; 
        
        auto h_sview = Kokkos::subview(h_mirror           , VEC( Kokkos::pair(ngz,nx+ngz)
                                                               , Kokkos::pair(ngz,ny+ngz)
                                                               , Kokkos::pair(ngz,nz+ngz) )
                                                          , ivar
                                                          , Kokkos::ALL()
                                                          ) ;  
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,scalars[ivar])) ;

    }
    /*
    for( int ivar=0; ivar<vectors.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(vectors[ivar]+"[0]") ; 
        
        auto h_sview = Kokkos::subview(h_mirror           , VEC(  Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::pair(ivar,ivar+3)
                                                          , Kokkos::ALL()
                                                          ) ;  
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,vectors[ivar])) ;

    }
    */
    Kokkos::fence() ;
    for( int ivar=0; ivar<aux_scalars.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(aux_scalars[ivar]) ; 
        
        auto h_sview = Kokkos::subview(aux_h_mirror       , VEC( Kokkos::pair(ngz,nx+ngz)
                                                               , Kokkos::pair(ngz,ny+ngz)
                                                               , Kokkos::pair(ngz,nz+ngz) )
                                                          , ivar
                                                          , Kokkos::ALL()
                                                          ) ;  
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,aux_scalars[ivar])) ;

    }
    /*
    for( int ivar=0; ivar<aux_vectors.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(aux_vectors[ivar]+"[0]") ; 
        
        auto h_sview = Kokkos::subview(aux_h_mirror       , VEC(  Kokkos::ALL()
                                                               , Kokkos::ALL()
                                                               , Kokkos::ALL() )
                                                          , Kokkos::pair(ivar,ivar+3)
                                                          , Kokkos::ALL()
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

void flag_ghost_cells(vtkSmartPointer<vtkUnstructuredGrid> grid)
{
    using namespace thunder ; 
    size_t _nx,_ny,_nz; 
    std::tie(_nx,_ny,_nz) = thunder::amr::get_quadrant_extents() ; 
    int ngz = thunder::amr::get_n_ghosts() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    size_t nx = _nx + 2 * ngz ; 
    size_t ny = _ny + 2 * ngz ; 
    size_t nz = _nz + 2 * ngz ;
    size_t ncells = EXPR(nx,*ny,*nz)*nq ;
    int zeroExt[6] = {0,0,0,0,0,0};
    //grid->GenerateGhostArray(zeroExt,true); 
    auto flag_array = grid->GetCellGhostArray() ; 
    flag_array->SetNumberOfTuples(ncells) ; 
    flag_array->SetNumberOfComponents(1) ;
    auto const tag_ghost_cell = [&] (VEC(int const& i, int const& j, int const& k))
    {
        	return  EXPR(
                   i < ngz or i > _nx+ngz-1,
                or j < ngz or j > _ny+ngz-1,
                or k < ngz or k > _nz+ngz-1
            ) ; 
    } ;
    ASSERT_DBG( flag_array != nullptr, "Ghost array is null.") ; 
    for(size_t iq=0UL; iq<nq ; iq+=1UL) {
        EXPR(
        for(size_t ix=0UL; ix<nx; ix+=1UL) {,
            for(size_t iy=0UL; iy<ny; iy+=1UL) {,
                for(size_t iz=0UL; iz<nz; iz+=1UL) {
                    )
                    #ifndef THUNDER_3D 
                    size_t icell = ix + nx*(iy+ny*iq) ;
                    #else 
                    size_t icell = ix + nx*(iy+ny*(iz+nz*iq)) ;
                    #endif 
                    if(tag_ghost_cell(VEC(ix,iy,iz)))
                        flag_array->SetValue(icell, vtkDataSetAttributes::DUPLICATECELL) ; 
                EXPR(
                },
            },   
        })
    }
}

void add_extra_output_quantities(vtkSmartPointer<vtkUnstructuredGrid> grid, bool include_gzs)
{
    using namespace thunder ; 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    int ngz = thunder::amr::get_n_ghosts() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    size_t ncells = EXPR(nx,*ny,*nz)*nq ;
    auto vol_mirror = Kokkos::create_mirror_view(thunder::variable_list::get().getvolumes()) ; 
    Kokkos::deep_copy(vol_mirror,thunder::variable_list::get().getvolumes()) ; 
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

    lcoords_array->SetNumberOfTuples(ncells*THUNDER_NSPACEDIM) ; 
    lcoords_array->SetNumberOfComponents(THUNDER_NSPACEDIM) ;
    lcoords_array->SetName("logical_coordinates") ;

    pcoords_array->SetNumberOfTuples(ncells*THUNDER_NSPACEDIM) ; 
    pcoords_array->SetNumberOfComponents(THUNDER_NSPACEDIM) ;
    pcoords_array->SetName("physical_coordinates") ;

    cell_volumes_array->SetNumberOfTuples(ncells) ; 
    cell_volumes_array->SetNumberOfComponents(1) ;
    cell_volumes_array->SetName("cell_volumes") ;

    auto mpi_rank = parallel::mpi_comm_rank() ; 

    auto& coord_system = coordinate_system::get() ; 
    
    for(size_t iq=0UL; iq<nq ; iq+=1UL) {
        EXPR(
        for(size_t ix=0UL; ix<nx; ix+=1UL) {,
            for(size_t iy=0UL; iy<ny; iy+=1UL) {,
                for(size_t iz=0UL; iz<nz; iz+=1UL) {
                    )
                    #ifndef THUNDER_3D 
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
                    for(int idim=0; idim<THUNDER_NSPACEDIM; ++idim){
                        lcoords_array->SetComponent(icell, idim, lcoords[idim]) ;
                        pcoords_array->SetComponent(icell, idim, pcoords[idim]) ;
                    }
                EXPR(
                },
            },   
        })
    }

    grid->GetCellData()->AddArray(mpi_rank_array) ; 
    grid->GetCellData()->AddArray(treeid_array) ; 
    grid->GetCellData()->AddArray(quadid_array) ; 
    grid->GetCellData()->AddArray(reflevel_array) ; 
    grid->GetCellData()->AddArray(lcoords_array) ;
    grid->GetCellData()->AddArray(pcoords_array) ;
    grid->GetCellData()->AddArray(cell_volumes_array) ;
}

}} /* namespace thunder::IO */

#ifdef THUNDER_3D 
#include "vtk_volume_output_3D.cpp"
#else 
#include "vtk_volume_output_2D.cpp"
#endif 