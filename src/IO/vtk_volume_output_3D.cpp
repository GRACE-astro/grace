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
#include <vtkBiQuadraticHexahedron.h> 
/* memory */
#include <vtkSmartPointer.h>
#include <vtkNew.h>
/* thunder includes */
#include <thunder/data_structures/variable_properties.hh>
#include <thunder/system/thunder_runtime.hh> 

#include <thunder/IO/vtk_volume_output_3D.hh>

#include <thunder/amr/thunder_amr.hh>
#include <thunder/utils/thunder_utils.hh>

#include <thunder/errors/error.hh>
#include <thunder/errors/assert.hh> 

#include <thunder/data_structures/variables.hh>

#include <tuple>
#include <array>
#include <string>
#include <filesystem>


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
        /* unpack index assuming LayoutLeft */
        size_t const ix = icell%nx ; 
        size_t const iy = (icell/nx) % ny ;
        size_t const iz = (icell/nx/ny) % nz ; 
        size_t const iq = (icell/nx/ny/nz) ;
        /* Find quadrant index, level, coordinates, spacing */
        size_t itree              = thunder::amr::get_quadrant_owner(iq) ; 
        auto const tree_coords    = thunder::amr::get_tree_vertex(itree, 0UL) ; 
        auto const dx_tree        = thunder::amr::get_tree_spacing(itree)[0]  ;
        quadrant_t const quadrant = thunder::amr::get_quadrant(itree, iq) ;  
        auto const dx_quad        = dx_tree / ( 1<<quadrant.level()) ; 
        auto const qcoords        = quadrant.qcoords() ; 
        auto const dx_cell        = dx_quad / nx ;
        auto const dy_cell        = dx_quad / ny ; 
        auto const dz_cell        = dx_quad / nz ;
        /* return physical coordinates of point within cell */ 
        return std::array<double,3> {
            tree_coords[0] + qcoords[0] * dx_quad + (ix + lx) * dx_cell, 
            tree_coords[1] + qcoords[1] * dx_quad + (iy + ly) * dy_cell, 
            tree_coords[2] + qcoords[2] * dx_quad + (iz + lz) * dz_cell 
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

vtkSmartPointer<vtkUnstructuredGrid> 
setup_vtk_grid_spherical()
{
    auto& params = thunder::config_parser::get()  ;  
    auto const L = params["amr"][""].as<double>() ;
    auto const R = params["amr"][].as<double>()   ;
    auto const Rl = params["amr"][].as<double>()  ;
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New() ;

    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    size_t nq = thunder::amr::get_local_num_quadrants() ; 
    size_t ncells = nx*ny*nz*nq ; // these are cells not vertices 
    size_t nvertex = 24 ; 
    using cell_type = vtkBiQuadraticHexahedron ; 

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
        /* Find quadrant index, level */
        size_t itree              = thunder::amr::get_quadrant_owner(iq) ; 
        quadrant_t const quadrant = thunder::amr::get_quadrant(itree, iq) ;  
        auto const dx_quad        = 1.0 / ( 1<<quadrant.level()) ; 
        auto const qcoords        = quadrant.qcoords() ; 
        auto const dx_cell        = dx_quad / nx ;
        auto const dy_cell        = dx_quad / ny ; 
        auto const dz_cell        = dx_quad / nz ;
        /* return physical coordinates of point within cell */ 
        auto lcoords = std::array<double,3> {
            qcoords[0] * dx_quad + (ix + lx) * dx_cell, 
            qcoords[1] * dx_quad + (iy + ly) * dy_cell, 
            qcoords[2] * dx_quad + (iz + lz) * dz_cell 
        } ; 
        std::array<double,3> pcoords ;
        if( itree == CARTESIAN_TREE ){ 
            auto const tree_coords    = thunder::amr::get_tree_vertex(itree, 0UL) ; 
            auto const dx_tree        = thunder::amr::get_tree_spacing(itree)[0]  ;
            for(int idir=0; idir<3; ++idir) {
                pcoords[idir] = tree_coords[idir] + lcoords[idir] * dx_tree ;
            }
        } else {
            auto const H  = tan(M_PI/4. * (2.*lcoords[1]-1)) ; 
            auto const XI = tan(M_PI/4. * (2.*lcoords[2]-1)) ; 
            auto const rho = sqrt( 1 + math::int_pow<2>(H) + math::int_pow<2>(XI) ) ; 
            auto const zeta     = ((1.-lcoords[0]) * L + lcoords[0]*R/rho)  ;
            auto const zeta_log = sqrt( std::pow(R, 2*(1-lcoords[0])) * std::pow(Rlog, 2*lcoords[0]) ) / rho ;
            switch( itree )
            {
                case MX_TREE: 
                    pcoords[0] = -zeta     ; 
                    pcoords[1] = zeta * XI ;
                    pcoords[2] = zeta * H  ;
                    break ; 
                case PX_TREE: 
                    pcoords[0] = zeta      ; 
                    pcoords[1] = zeta * XI ;
                    pcoords[2] = zeta * H  ;
                    break; 
                case MY_TREE:
                    pcoords[0] =  zeta * H  ;
                    pcoords[1] = -zeta      ;
                    pcoords[2] =  zeta * XI ;
                    break;
                case PY_TREE:
                    pcoords[0] =  zeta * XI  ;
                    pcoords[1] =  zeta       ;
                    pcoords[2] =  zeta * H   ; 
                    break;
                case MZ_TREE:
                    pcoords[0] =  zeta * XI ;
                    pcoords[1] =  zeta * H  ;
                    pcoords[2] = -zeta      ;
                    break;
                case PZ_TREE:
                    pcoords[0] =  zeta * H  ;
                    pcoords[1] =  zeta * XI ;
                    pcoords[2] = -zeta      ;
                    break;
                case MXL_TREE: 
                    pcoords[0] = -zeta_log     ; 
                    pcoords[1] = zeta_log * XI ;
                    pcoords[2] = zeta_log * H  ;
                    break ; 
                case PXL_TREE: 
                    pcoords[0] = zeta_log      ; 
                    pcoords[1] = zeta_log * XI ;
                    pcoords[2] = zeta_log * H  ;
                    break; 
                case MYL_TREE:
                    pcoords[0] =  zeta_log * H  ;
                    pcoords[1] = -zeta_log      ;
                    pcoords[2] =  zeta_log * XI ;
                    break;
                case PYL_TREE:
                    pcoords[0] =  zeta_log * XI  ;
                    pcoords[1] =  zeta_log       ;
                    pcoords[2] =  zeta_log * H   ; 
                    break;
                case MZL_TREE:
                    pcoords[0] =  zeta_log * XI ;
                    pcoords[1] =  zeta_log * H  ;
                    pcoords[2] = -zeta_log      ;
                    break;
                case PZL_TREE:
                    pcoords[0] =  zeta_log * H  ;
                    pcoords[1] =  zeta_log * XI ;
                    pcoords[2] = -zeta_log      ;
                    break;

            }
        }
        return pcoords ;
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

vtkSmartPointer<vtkUnstructuredGrid> setup_vtk_volume_grid()
{
    std::string coord_system = params["amr"]["pcoords"].as<std::string>() ; 
    if( coord_system == "cartesian" ) {
        return detail::setup_vtk_grid_cartesian() ; 
    } else if (coord_system == "spherical" ) {
        return detail::setup_vtk_grid_spherical() ; 
    } else {
        ERROR("Should not be here.") ; 
    }
}

template< typename ViewT > 
vtkSmartPointer<vtkDoubleArray> vtk_create_cell_data(ViewT data_view, std::string const& name, bool is_vector=false)
{
    auto data = vtkSmartPointer<vtkDoubleArray>::New() ;
    data->SetNumberOfComponents( 1 + 2*is_vector ) ; 
    data->SetNumberOfTuples( 
        data_view.extent(0) * data_view.extent(1) * data_view.extent(2) * data_view.extent(THUNDER_NSPACEDIM+1) 
    ) ;
    if( is_vector ) {
        std::string comp_name = name + "[0]" ; 
        data->SetComponentName(0,comp_name) ; 
        comp_name = name + "[1]" ;
        data->SetComponentName(1,comp_name) ; 
        comp_name = name + "[2]" ;
        data->SetComponentName(2,comp_name) ; 
    } else {
        data->SetComponentName(0,name.c_str()) ; 
    }   
    size_t const  nq{data_view.extent(THUNDER_NSPACEDIM+1) }
                , nz{data_view.extent(2)}
                , ny{data_view.extent(1)}
                , nx{data_view.extent(0)} ; 

    for(size_t iq=0UL; iq<nq ; iq+=1UL) {
        for(size_t iz=0UL; iz<nz; iz+=1UL) {
            for(size_t iy=0UL; iy<ny; iy+=1UL) {
                for(size_t ix=0UL; ix<nx; ix+=1UL) {
                    size_t icell = ix + nx*(iy+ny*(iz+nz*iq)) ; 
                    for( int icomp=0; icom<1+2*is_vector; icomp++) 
                        data->SetComponent(icell,icomp,data_view(ix,iy,iz,icomp,iq)) ;
                }
            }
        }
    }
    return data ;
}

vtkSmartPointer<vtkUnstructuredGrid> saetup_volume_cell_data() {

    auto& runtime = thunder::runtime::get() ;

    auto const scalars     = runtime.cell_volume_output_scalar_vars() ; 
    auto const aux_scalars = runtime.cell_volume_output_scalar_vars() ;
    auto const vectors     = runtime.cell_volume_output_vector_vars() ; 
    auto const aux_vectors = runtime.cell_volume_output_vector_vars() ; 
    bool const trim_gzones = runtime.volume_oputput_trim_ghostzones() ;
    size_t nx,ny,nz,nq; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    nq = thunder::amr::get_local_num_quadrants() ; 
    int ngz = thunder::amr::get_n_ghostzones() ; 

    auto vars = thunder::variable_list::get().getstate() ; 
    auto aux  = thunder::variable_list::get().getaux()   ; 

    auto scalar_host_mirror = 
    Kokkos::View< double *****, decltype(vars)::host_mirror_space, decltype(vars)::array_layout >(
          "scalar_output_mirror"
        ,  nx + 2 * ( !trim_gzones ) * ngz
        ,  ny + 2 * ( !trim_gzones ) * ngz
        ,  nz + 2 * ( !trim_gzones ) * ngz
        ,  scalars.size() 
        ,  nq 
    ) ; 
    auto vector_host_mirror = 
    Kokkos::View< double *****, decltype(vars)::host_mirror_space, decltype(vars)::array_layout >(
          "vector_output_mirror"
        ,  nx + 2 * ( !trim_gzones ) * ngz
        ,  ny + 2 * ( !trim_gzones ) * ngz
        ,  nz + 2 * ( !trim_gzones ) * ngz
        ,  3 * vectors.size() 
        ,  nq 
    ) ; 
    
    
    auto grid = setup_vtk_volume_grid() ; 
    /* 
    * In the following we perform a series of 
    * calls to deep_copy for each variable group 
    * separately. These calls are asynchronous 
    * when the backend is CUDA or HIP and this 
    * allows us to overlap the data transfer 
    * from device to host and the writing of 
    * data to the vtkUnstructuredGrid object. 
    */
    for( int ivar=0; ivar<scalars.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(scalars[ivar]) ; 
        auto h_sview = Kokkos::subview(scalars_host_mirror, Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , ivar
                                                          , Kokkos::ALL()) ; 
        auto d_sview = Kokkos::subview(vars               , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , varidx
                                                          , Kokkos::ALL()) ;
        Kokkos::deep_copy(exec_space{}, h_sview, d_sview ) ;
    }
    Kokkos::fence() ; 

    for( int ivar=0; ivar<vectors.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(vectors[ivar]) ; 
        auto h_sview = Kokkos::subview(vectors_host_mirror, Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(ivar, ivar+3UL)
                                                          , Kokkos::ALL()) ; 
        auto d_sview = Kokkos::subview(vars               , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(varidx, varidx+3UL)
                                                          , Kokkos::ALL()) ;
        Kokkos::deep_copy(exec_space{}, h_sview, d_sview ) ;
    }

    for( int ivar=0; ivar<scalars.size(); ++ivar )
    {
        auto h_sview = Kokkos::subview(scalars_host_mirror, Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , ivar
                                                          , Kokkos::ALL()) ;
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,scalars[ivar])) ; 
    }

    Kokkos::fence() ; 
    /* recycle memory */
    Kokkos::resize(  scalars_host_mirror
                  ,  nx + 2 * ( !trim_gzones ) * ngz
                  ,  ny + 2 * ( !trim_gzones ) * ngz
                  ,  nz + 2 * ( !trim_gzones ) * ngz
                  ,  aux_scalars.size() 
                  ,  nq )
    for( int ivar=0; ivar<aux_scalars.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(aux_scalars[ivar]) ; 
        auto h_sview = Kokkos::subview(scalars_host_mirror, Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , ivar
                                                          , Kokkos::ALL()) ; 
        auto d_sview = Kokkos::subview(aux                , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , varidx
                                                          , Kokkos::ALL()) ;
        Kokkos::deep_copy(exec_space{}, h_sview, d_sview ) ;
    }

    for( int ivar=0; ivar<vectors.size(); ++ivar )
    {
        auto h_sview = Kokkos::subview(vectors_host_mirror, Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(ivar, ivar+3UL)
                                                          , Kokkos::ALL()) ;
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,vectors[ivar],true)) ; 
    }
    Kokkos::fence() ; 
    /* recycle memory */
    Kokkos::resize(  vectors_host_mirror
                  ,  nx + 2 * ( !trim_gzones ) * ngz
                  ,  ny + 2 * ( !trim_gzones ) * ngz
                  ,  nz + 2 * ( !trim_gzones ) * ngz
                  ,  3UL*aux_vectors.size() 
                  ,  nq )
    for( int ivar=0; ivar<aux_vectors.size(); ++ivar )
    {
        size_t varidx = thunder::get_variable_index(aux_vectors[ivar]) ; 
        auto h_sview = Kokkos::subview(scalars_host_mirror, Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(ivar, ivar+3UL)
                                                          , Kokkos::ALL()) ; 
        auto d_sview = Kokkos::subview(aux                , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(varidx, varidx+3UL)
                                                          , Kokkos::ALL()) ;
        Kokkos::deep_copy(exec_space{}, h_sview, d_sview ) ;
    }

    for( int ivar=0; ivar<aux_scalars.size(); ++ivar )
    {
        auto h_sview = Kokkos::subview(scalars_host_mirror, Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , ivar
                                                          , Kokkos::ALL()) ;
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,aux_scalars[ivar])) ; 
    }
    Kokkos::fence() ; 
    for( int ivar=0; ivar<aux_vectors.size(); ++ivar )
    {
        auto h_sview = Kokkos::subview(vectors_host_mirror, Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::ALL()
                                                          , Kokkos::pair(ivar, ivar+3UL)
                                                          , Kokkos::ALL()) ;
        grid->GetCellData()->AddArray(vtk_create_cell_data(h_sview,aux_vectors[ivar],true)) ; 
    }
}; 

void write_volume_cell_data() 
{   
    auto& runtime = thunder::runtime::get() ;
    std::filesystem::path base_path (runtime.volume_io_dir()) ;
    std::string fname = runtime.volume_io_base_filename() + std::to_string(parallel::mpi_comm_rank()) + ".vtu";
    std::filesystem::path out_path = base_path / fname ; 

    auto grid = setup_volume_cell_data() ; 

    vtkNew<XMLUnstructuredGridWriter> writer ;
    writer->SetFileName(out_filename.string().c_str()) ; 
    writer->SetInputData(grid)               ;
    writer->write() ;

    if( parallel::mpi_comm_rank() == runtime.master() ) 
    {
        std::string pfname = runtime.volume_io_base_filename() + ".pvtu" ;
        vtkNew<XMLPUnstructuredGridWriter> pwriter ; 
        out_path = base_path / pfname ;
        pwriter->SetFileName(out_path.string().c_str()) ; 
        pwriter->SetNumberOfPieces( parallel::mpi_comm_size() ) ; 
        pwriter->SetInputData( grid ) ; 
        pwriter->Update() ; 
    }


}

}} /* namespace thunder::IO */