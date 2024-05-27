/**
 * @file hdf5_output.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-23
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

#include <thunder_config.h>
#include <thunder/utils/thunder_utils.hh>
#include <thunder/amr/thunder_amr.hh>
#include <thunder/config/config_parser.hh>
#include <thunder/coordinates/coordinate_systems.hh>
#include <thunder/system/thunder_system.hh>
#include <thunder/IO/hdf5_output.hh>

#include <thunder/parallel/mpi_wrappers.hh>

#include <hdf5.h>
#include <omp.h>
/* cell types */
#include <vtkCellData.h>
#include <vtkHexahedron.h>
#include <vtkBiQuadraticQuadraticHexahedron.h> 
#include <vtkQuad.h>
#include <vtkQuadraticLinearQuad.h> 
/* memory */
#include <vtkSmartPointer.h>
#include <vtkNew.h>
/* xdmf */
#include <thunder/IO/xmf_utilities.hh>
/* stl */
#include <string>
#include <filesystem>

#define HDF5_CALL(result,cmd) \
    do {  \
        if((result=cmd)<0) { \
            ERROR("HDF5 API call failed with error code " << result ) ; \
        } \
    } while(false)

namespace thunder { namespace IO {

namespace detail {

std::vector<int64_t> _volume_output_iterations ;
std::vector<double>  _volume_output_times ;
std::vector<int64_t> _volume_output_ncells ; 
std::vector<std::string> _volume_output_filenames ; 

}

void write_cell_data_hdf5(bool out_vol, bool out_plane, bool out_sphere) {

    if( out_vol ) {
        write_volume_cell_data_hdf5() ; 
    }
    parallel::mpi_barrier() ; 
}


void write_volume_cell_data_hdf5() {
    Kokkos::Profiling::pushRegion("HDF5 volume output") ;

    detail::_volume_output_iterations.push_back(thunder::get_iteration())  ; 
    detail::_volume_output_times.push_back(thunder::get_simulation_time()) ;

    auto& runtime = thunder::runtime::get() ; 
    std::filesystem::path base_path (runtime.volume_io_basepath()) ;
    std::string pfname = runtime.volume_io_basename() + "_" + utils::zero_padded(runtime.iteration(),3)
                                                            + ".h5" ;
    std::filesystem::path out_path = base_path / pfname ;
    detail::_volume_output_filenames.push_back(pfname) ;

    auto& params = thunder::config_parser::get() ;
    size_t compression_level = params["IO"]["hdf5_compression_level"].as<size_t>() ;
    size_t chunk_size = params["IO"]["hdf5_chunk_size"].as<size_t>() ;

    auto comm = parallel::get_comm_world() ; 
    auto rank = parallel::mpi_comm_rank()  ; 
    auto size = parallel::mpi_comm_size()  ;

    herr_t err ;
    // Create property list for parallel file access
    hid_t plist_id ; 
    HDF5_CALL(plist_id,H5Pcreate(H5P_FILE_ACCESS)) ; 
    HDF5_CALL(err,H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL)) ; 
    // Create a new file 
    hid_t file_id ; 
    HDF5_CALL(file_id,H5Fcreate(out_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) ;
    /* Write grid structure to hdf5 file      */
    write_grid_structure_hdf5(file_id, compression_level,chunk_size) ; 
    /* Write requested variables to hdf5 file */
    write_volume_data_arrays_hdf5(file_id, compression_level,chunk_size) ;
    parallel::mpi_barrier() ; 
    /* Close the file */
    HDF5_CALL(err,H5Fclose(file_id)) ; 
    HDF5_CALL(err,H5Pclose(plist_id)) ; 
    /* Write xmf file */
    std::string pfname_xdmf = runtime.volume_io_basename() + ".xmf" ;
    std::filesystem::path out_path_xdmf = base_path / pfname_xdmf ;
    if( parallel::mpi_comm_rank() == 0)
        write_xmf_file( out_path_xdmf.string()
                      , detail::_volume_output_iterations
                      , detail::_volume_output_ncells 
                      , detail::_volume_output_times 
                      , detail::_volume_output_filenames ) ; 
    
    Kokkos::Profiling::popRegion() ; 
}

void write_grid_structure_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size) {
    herr_t err ; 

    auto& coord_system = thunder::coordinate_system::get() ; 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    int ngz = thunder::amr::get_n_ghosts() ;
    size_t nq = thunder::amr::get_local_num_quadrants() ; 

    #ifdef THUNDER_3D 
    #ifdef THUNDER_CARTESIAN_COORDINATES
    using cell_type = vtkHexahedron ; 
    size_t constexpr nvertex = 8 ;
    #elif defined(THUNDER_SPHERICAL_COORDINATES)
    using cell_type = vtkBiQuadraticQuadraticHexahedron ; 
    size_t constexpr nvertex = 24 ;
    #endif 
    #else 
    #ifdef THUNDER_CARTESIAN_COORDINATES
    size_t nvertex = 4 ; 
    #elif defined(THUNDER_SPHERICAL_COORDINATES)
    size_t nvertex = 6 ; 
    using cell_type = vtkQuadraticLinearQuad ;
    #endif 
    #endif 

    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = thunder::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ; 
    unsigned long const local_quad_offset = _p4est->global_first_quadrant[rank] ; 
    /* Get parametric coordinates of cells vertices */
    vtkNew<cell_type> _tmpcell ;
    auto lcoords = _tmpcell->GetParametricCoords() ;
    /* Number of cells per quadrant */
    unsigned long const ncells_quad = EXPR(nx,*ny,*nz) ; 
    /* Local number of cells   */
    unsigned long const ncells = ncells_quad * nq ; 
    /* Global number of cells  */
    unsigned long const ncells_glob = ncells_quad * nq_glob ; 
    /* Local number of points  */
    unsigned long const npoints = ncells * nvertex ; 
    /* Global number of points */
    unsigned long const npoints_glob = ncells_glob * nvertex ; 

    detail::_volume_output_ncells.push_back(ncells_glob) ; 

    double*  points = (double*)  malloc(sizeof(double)  * npoints * THUNDER_NSPACEDIM ) ; 
    unsigned int* cells  = (unsigned int*) malloc(sizeof(unsigned int) * ncells * nvertex ) ; 

    unsigned int icell  = 0L ; 
    unsigned int ipoint = 0U ; 
    #pragma omp parallel for collapse 4
    for(int64_t iq=0; iq<nq; ++iq) {
        for( size_t i=0; i<nx; ++i) {
            for( size_t j=0; j<ny; ++j) {
                #ifdef THUNDER_3D
                for(size_t k=0; k<nz; ++k){
                #endif 
                for( int iv=0; iv<nvertex; ++iv ) {
                    auto const pcoords = 
                        coord_system.get_physical_coordinates( {VEC(i,j,k)}
                                                             , iq
                                                             , {VEC( lcoords[3*iv+0]
                                                                   , lcoords[3*iv+1]
                                                                   , lcoords[3*iv+2])}
                                                             , false) ; 
                    points[THUNDER_NSPACEDIM*(nvertex*icell + iv) + 0 ] = pcoords[0] ; 
                    points[THUNDER_NSPACEDIM*(nvertex*icell + iv) + 1 ] = pcoords[1] ;
                    #ifdef THUNDER_3D 
                    points[THUNDER_NSPACEDIM*(nvertex*icell + iv) + 2 ] = pcoords[2] ;
                    #endif 
                    cells[nvertex * icell + iv] = ipoint ; 
                    ipoint ++ ; 
                }

                icell ++ ; 
                #ifdef THUNDER_3D
                }
                #endif 
            }
        }
    }
    ASSERT(icell == ncells, "Something went really wrong") ; 
    ASSERT(ipoint == npoints, "Something went really wrong") ; 
    /* Create parallel dataset properties */
    hid_t dxpl ; 
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ; 
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ; 

    /* Create/open datasets */
    hid_t points_space_id_glob ;
    hsize_t points_dset_dims_glob[2] = {npoints_glob, THUNDER_NSPACEDIM} ;   
    /* Create global space for points dataset */
    HDF5_CALL(points_space_id_glob, H5Screate_simple(2, points_dset_dims_glob, NULL)) ; 
    hid_t cells_space_id_glob ;
    hsize_t cells_dset_dims_glob[2] = {ncells_glob, nvertex} ;   
    /* Create global space for cells dataset */
    HDF5_CALL(cells_space_id_glob, H5Screate_simple(2, cells_dset_dims_glob, NULL)) ; 

    hid_t points_dset_id, cells_dset_id  ;
    hid_t points_prop_id, cells_prop_id  ;
    HDF5_CALL(points_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t points_chunk_dim[2] = {chunk_size,THUNDER_NSPACEDIM} ;
    HDF5_CALL(err, H5Pset_chunk(points_prop_id,2,points_chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(points_prop_id, compression_level)) ; 
    HDF5_CALL( points_dset_id
            , H5Dcreate2( file_id
                        , "/Points"
                        , H5T_NATIVE_DOUBLE
                        , points_space_id_glob
                        , H5P_DEFAULT
                        , points_prop_id
                        , H5P_DEFAULT) ) ;

    HDF5_CALL(cells_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ;
    hsize_t cells_chunk_dim[2] = {chunk_size,nvertex} ;
    HDF5_CALL(err, H5Pset_chunk(cells_prop_id,2,cells_chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(cells_prop_id, compression_level)) ; 
    HDF5_CALL( cells_dset_id
                , H5Dcreate2( file_id
                            , "/Cells"
                            , H5T_NATIVE_UINT
                            , cells_space_id_glob
                            , H5P_DEFAULT
                            , cells_prop_id
                            , H5P_DEFAULT) ) ;
      

    /* Write cells dataset */
    /* Create local space for this rank */
    hid_t cells_space_id ; 
    hsize_t cells_dset_dims[2] = {ncells, nvertex} ;
    HDF5_CALL(cells_space_id, H5Screate_simple(2, cells_dset_dims, NULL)) ; 
    /* Select hyperslab for this rank's output */
    hsize_t cells_slab_start[2]  = {local_quad_offset * ncells_quad,0} ; 
    hsize_t cells_slab_count[2]  = {ncells,nvertex} ;
    THUNDER_VERBOSE("Slab offset {}, size {}, total {}", cells_slab_start[0], ncells, ncells_glob) ;  
    HDF5_CALL( err
             , H5Sselect_hyperslab( cells_space_id_glob
                                  , H5S_SELECT_SET
                                  , cells_slab_start
                                  , NULL
                                  , cells_slab_count 
                                  , NULL )) ;


    HDF5_CALL( err
             , H5Dwrite( cells_dset_id
                       , H5T_NATIVE_UINT
                       , cells_space_id
                       , cells_space_id_glob 
                       , dxpl 
                       , reinterpret_cast<void*>(cells) )) ; 

    HDF5_CALL(err, H5Dclose(cells_dset_id)) ; 
    HDF5_CALL(err, H5Sclose(cells_space_id)) ; 
    HDF5_CALL(err, H5Sclose(cells_space_id_glob)) ; 
    HDF5_CALL(err, H5Pclose(cells_prop_id)) ;
    /* Release resources */
    free(cells) ; 

    /* Write points dataset */
    /* Create local space for this rank */
    hid_t points_space_id ; 
    hsize_t points_dset_dims[2] = {npoints, THUNDER_NSPACEDIM} ;
    HDF5_CALL(points_space_id, H5Screate_simple(2, points_dset_dims, NULL)) ; 
    /* Select hyperslab for this rank's output */
    hsize_t points_slab_start[2]  = {local_quad_offset * ncells_quad * nvertex,0} ;
    THUNDER_VERBOSE("Slab offset {}, size {}, total {}", points_slab_start[0], npoints, npoints_glob) ;  
    hsize_t points_slab_count[2]  = {npoints,THUNDER_NSPACEDIM} ;
    HDF5_CALL( err
             , H5Sselect_hyperslab( points_space_id_glob
                                  , H5S_SELECT_SET
                                  , points_slab_start
                                  , NULL
                                  , points_slab_count 
                                  , NULL )) ;
    /* Write data corresponding to this rank to disk */
    HDF5_CALL( err
             , H5Dwrite( points_dset_id
                       , H5T_NATIVE_DOUBLE
                       , points_space_id
                       , points_space_id_glob 
                       , dxpl 
                       , reinterpret_cast<void*>(points) )) ; 
    HDF5_CALL(err, H5Dclose(points_dset_id)) ; 
    HDF5_CALL(err, H5Sclose(points_space_id)) ; 
    HDF5_CALL(err, H5Sclose(points_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(points_prop_id)) ;

    HDF5_CALL(err, H5Pclose(dxpl)) ;
    /* Release resources*/
    free(points);
    
}

void write_volume_data_arrays_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size) {

    herr_t err ;

    auto& runtime = thunder::runtime::get() ;

    auto const scalars     = runtime.cell_volume_output_scalar_vars() ; 
    auto const aux_scalars = runtime.cell_volume_output_scalar_aux()  ;
    auto const vectors     = runtime.cell_volume_output_vector_vars() ; 
    auto const aux_vectors = runtime.cell_volume_output_vector_aux()  ;

    size_t nx,ny,nz,nq; 
    std::tie(nx,ny,nz) = thunder::amr::get_quadrant_extents() ; 
    nq = thunder::amr::get_local_num_quadrants() ; 
    size_t ngz = thunder::amr::get_n_ghosts() ; 

    auto vars = thunder::variable_list::get().getstate() ; 
    auto aux  = thunder::variable_list::get().getaux()   ;
    using exec_space = decltype(vars)::execution_space   ;

    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = thunder::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ; 
    unsigned long const local_quad_offset = _p4est->global_first_quadrant[rank] ; 
    /* Number of cells per quadrant */
    unsigned long const ncells_quad = EXPR(nx,*ny,*nz) ; 
    /* Local number of cells   */
    unsigned long const ncells = ncells_quad * nq ; 
    /* Global number of cells  */
    unsigned long const ncells_glob = ncells_quad * nq_glob ; 

    /* Create parallel dataset properties */
    hid_t dxpl ; 
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ; 
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ;
    /* Create/open datasets */
    hid_t sclars_space_id_glob ;
    hsize_t scalars_dset_dims_glob[1] = {ncells_glob} ;
    /* Create global space for points dataset */
    HDF5_CALL(sclars_space_id_glob, H5Screate_simple(1, scalars_dset_dims_glob, NULL)) ;
    hid_t scalars_dset_id, scalars_prop_id ;
    HDF5_CALL(scalars_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t scalars_chunk_dim[1] = {chunk_size} ;
    HDF5_CALL(err, H5Pset_chunk(scalars_prop_id,1,scalars_chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(scalars_prop_id, compression_level)) ;  
    /* Create local space for this rank */
    hid_t scalars_space_id ; 
    hsize_t scalars_dset_dims[1] = {ncells} ;
    HDF5_CALL(scalars_space_id, H5Screate_simple(1, scalars_dset_dims, NULL)) ; 
    /**********************************************/
    /* We need an extra device mirror because:    */
    /* 1) The view is not contiguous since we     */
    /*    cut out the ghost-zones.                */
    /* 2) The layout may differ from the          */
    /*    memory layout on device which           */
    /*    usually follows the FORTRAN convention. */
    /**********************************************/
    Kokkos::View<double EXPR(*,*,*)*, Kokkos::LayoutRight> 
        d_mirror("Device output mirror", VEC(nx,ny,nz), nq) ; 
    auto h_mirror = Kokkos::create_mirror_view(d_mirror) ; 
    for( int ivar=0; ivar<scalars.size(); ++ivar)
    {
        size_t varidx = thunder::get_variable_index(scalars[ivar]) ; 
        /* create HDF5 dataset */
        std::string dset_name = "/" + scalars[ivar] ; 
        HDF5_CALL( scalars_dset_id
                , H5Dcreate2( file_id
                            , dset_name.c_str()
                            , H5T_NATIVE_DOUBLE
                            , sclars_space_id_glob
                            , H5P_DEFAULT
                            , scalars_prop_id
                            , H5P_DEFAULT) ) ;
        for( int iq=0; iq<nq; ++iq){
            /* Copy data d2d */
            auto sview = Kokkos::subview( vars
                                        , Kokkos::pair<int,int>(ngz,nx+ngz)
                                        , Kokkos::pair<int,int>(ngz,ny+ngz)
                                        #ifdef THUNDER_3D
                                        , Kokkos::pair<int,int>(ngz,nz+ngz)
                                        #endif 
                                        , varidx 
                                        , iq ) ; 
            auto mirror_sview = Kokkos::subview( d_mirror
                                        , Kokkos::ALL()
                                        , Kokkos::ALL()
                                        #ifdef THUNDER_3D
                                        , Kokkos::ALL()
                                        #endif 
                                        , iq ) ; 
            /* This deep copy operation is asynchronous */
            Kokkos::deep_copy(mirror_sview, sview) ; 
        }
        /* Copy data d2h */
        Kokkos::deep_copy(thunder::default_execution_space{},h_mirror,d_mirror) ; 
        /* Select hyperslab for this rank's output */
        hsize_t sclars_slab_start[1]  = {local_quad_offset * ncells_quad} ; 
        hsize_t sclars_slab_count[1]  = {ncells} ;
        HDF5_CALL( err
                , H5Sselect_hyperslab( sclars_space_id_glob
                                    , H5S_SELECT_SET
                                    , sclars_slab_start
                                    , NULL
                                    , sclars_slab_count 
                                    , NULL )) ;
        Kokkos::fence() ; 
        /* write to dataset */
        HDF5_CALL( err
                    , H5Dwrite( scalars_dset_id
                            , H5T_NATIVE_DOUBLE
                            , scalars_space_id
                            , sclars_space_id_glob 
                            , dxpl 
                            , reinterpret_cast<void*>(h_mirror.data()) )) ;

        
        HDF5_CALL(err, H5Dclose(scalars_dset_id)) ; 
    }

    HDF5_CALL(err, H5Sclose(scalars_space_id)) ; 
    HDF5_CALL(err, H5Sclose(sclars_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(scalars_prop_id)) ;
    HDF5_CALL(err, H5Pclose(dxpl)) ;
     
}

}} /* namespace thunder::IO */
