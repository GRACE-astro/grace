/**
 * @file hdf5_output.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-05-23
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
#include <grace/utils/grace_utils.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/config/config_parser.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/system/grace_system.hh>
#include <grace/IO/hdf5_output.hh>

#include <grace/parallel/mpi_wrappers.hh>

#include "hdf5.h"
#include "H5public.h"
#include "H5Ppublic.h"  
#include <omp.h>
/* xdmf */
#include <grace/IO/xmf_utilities.hh>
/* stl */
#include <string>
#include <filesystem>
#include <sstream>
#include <iomanip>

#define HDF5_CALL(result,cmd) \
    do {  \
        if((result=cmd)<0) { \
            ERROR("HDF5 API call failed with error code " << result ) ; \
        } \
    } while(false)

namespace grace { namespace IO {

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

    detail::_volume_output_iterations.push_back(grace::get_iteration())  ; 
    detail::_volume_output_times.push_back(grace::get_simulation_time()) ;

    auto& runtime = grace::runtime::get() ; 
    std::filesystem::path base_path (runtime.volume_io_basepath()) ;
    std::ostringstream oss;
    oss << runtime.volume_io_basename() << "_" 
        << std::setw(6) << std::setfill('0') << grace::get_iteration() << ".h5";
    std::string pfname = oss.str();
    std::filesystem::path out_path = base_path / pfname ;
    detail::_volume_output_filenames.push_back(pfname) ;

    auto& params = grace::config_parser::get() ;
    size_t compression_level = params["IO"]["hdf5_compression_level"].as<size_t>() ;
    size_t chunk_size = params["IO"]["hdf5_chunk_size"].as<size_t>() ;

    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ;
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    unsigned long const ncells_quad = EXPR(nx,*ny,*nz) ; 
    /* Global number of cells  */
    unsigned long const ncells_glob = ncells_quad * nq_glob ; 
    if( chunk_size > ncells_glob ) {
        GRACE_WARN("Chunk size {} < number of cells {} will be overridden." , chunk_size, ncells_glob) ; 
        chunk_size = ncells_glob; 
    }

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
    hid_t attr_id, attr_dataspace_id;
    std::string file_attr_name = "Time";
    const double file_attr_data = grace::get_simulation_time() ;
    // Create a dataspace for the attribute
    HDF5_CALL(attr_dataspace_id,H5Screate(H5S_SCALAR));
    // Create the attribute
    HDF5_CALL(attr_id,H5Acreate2(file_id, file_attr_name.c_str(), H5T_NATIVE_DOUBLE, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT));
    // Write the attribute data
    HDF5_CALL(err,H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &file_attr_data));
    // Close the attribute and dataspace
    HDF5_CALL(err,H5Aclose(attr_id));
    HDF5_CALL(err,H5Sclose(attr_dataspace_id));
    file_attr_name = "Iteration" ; 
    const unsigned int iter = grace::get_iteration(); 
    HDF5_CALL(attr_dataspace_id,H5Screate(H5S_SCALAR));
    // Create the attribute
    HDF5_CALL(attr_id,H5Acreate2(file_id, file_attr_name.c_str(), H5T_NATIVE_UINT, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT));
    // Write the attribute data
    HDF5_CALL(err,H5Awrite(attr_id, H5T_NATIVE_UINT, &iter));
    // Close the attribute and dataspace
    HDF5_CALL(err,H5Aclose(attr_id));
    HDF5_CALL(err,H5Sclose(attr_dataspace_id));
    /* Write grid structure to hdf5 file      */
    write_grid_structure_hdf5(file_id, compression_level,chunk_size) ; 
    /* Write requested variables to hdf5 file */
    write_volume_data_arrays_hdf5(file_id, compression_level,chunk_size) ;
    /* Write extra quantities if requested */
    bool output_extra = grace::get_param<bool>("IO","output_extra_quantities") ; 
    if( output_extra ) {
        write_extra_arrays_hdf5(file_id, compression_level, chunk_size) ; 
    }
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

    auto& coord_system = grace::coordinate_system::get() ; 
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    int ngz = grace::amr::get_n_ghosts() ;
    size_t nq = grace::amr::get_local_num_quadrants() ; 

    #ifdef GRACE_3D 
    #ifdef GRACE_CARTESIAN_COORDINATES
    size_t constexpr nvertex = 8 ;
    #elif defined(GRACE_SPHERICAL_COORDINATES)
    size_t constexpr nvertex = 24 ;
    #endif 
    #else 
    #ifdef GRACE_CARTESIAN_COORDINATES
    size_t nvertex = 4 ; 
    #elif defined(GRACE_SPHERICAL_COORDINATES)
    size_t nvertex = 6 ; 
    #endif 
    #endif 

    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ; 
    unsigned long const local_quad_offset = _p4est->global_first_quadrant[rank] ; 
    /* Number of cells per quadrant */
    unsigned long const ncells_quad = EXPR(nx,*ny,*nz) ; 
    /* Local number of cells   */
    unsigned long const ncells = ncells_quad * nq ; 
    /* Global number of cells  */
    unsigned long const ncells_glob = ncells_quad * nq_glob ; 
    /* Number of unique points per quadrant */
    unsigned long const npoints_quad = (nx+1) * (ny+1) * (nz+1) ; 
    /* Local number of points  */
    unsigned long const npoints = npoints_quad * nq ;  
    /* Global number of points */
    unsigned long const npoints_glob = npoints_quad * nq_glob ;

    detail::_volume_output_ncells.push_back(ncells_glob) ; 

    double*  points = (double*)  malloc(sizeof(double)  * npoints * 3 ) ; 
    unsigned int* cells  = (unsigned int*) malloc(sizeof(unsigned int) * ncells * nvertex ) ; 
    const size_t global_point_offset = local_quad_offset * npoints_quad ;  
    unsigned int icell  = 0L ; 
    unsigned int ipoint = 0U ; 

    auto const get_point_index = 
    [&] 
    (
        VEC(int i, int j, int k), int64_t q
    ) 
    {
        #ifdef GRACE_3D 
        return i + (nx+1) * (j + (ny+1) * (k + (nz+1) * q)) ; 
        #else 
        return i + (nx+1) * (j + (ny+1) * q) ; 
        #endif
    } ; 

    auto const get_cell_vertex_indices = [&]
    (
        VEC(int i, int j, int k), int64_t q, int iv
    ) 
    {
        static constexpr std::array<std::array<int,3>,8> vertex_coords {{
            {0, 0, 0}, //
            {1, 0, 0}, //
            {1, 1, 0}, //
            {0, 1, 0}, //
            {0, 0, 1}, //
            {1, 0, 1}, //
            {1, 1, 1}, //
            {0, 1, 1}  //
        }} ; 
        return std::make_tuple(
            VEC(
                i+vertex_coords[iv][0],
                j+vertex_coords[iv][1],
                k+vertex_coords[iv][2]
            )
        ) ; 
    } ;
    #pragma omp parallel for collapse(GRACE_NSPACEDIM+1) 
    for(int64_t iq=0; iq<nq; ++iq) {
        #ifdef GRACE_3D
        for(size_t k=0; k<nz; ++k  ) {
        #endif  
            for( size_t j=0; j<ny; ++j) { 
                for(size_t i=0; i<nx; ++i){
                int64_t const icell = i + nx * ( j + ny * ( k  + nz * iq )) ; 
                for( int iv=0; iv<nvertex; ++iv ) {
                    int ip, jp, kp ; 
                    std::tie(ip,jp,kp) = get_cell_vertex_indices(VEC(i,j,k),iq,iv) ; 
                    cells[nvertex * icell + iv] = get_point_index(VEC(ip,jp,kp),iq+local_quad_offset); 
                }
                #ifdef GRACE_3D
                }
                #endif 
            }
        }
    }
    #pragma omp parallel for collapse(GRACE_NSPACEDIM+1) 
    for(int64_t iq=0; iq<nq; ++iq) {
        #ifdef GRACE_3D
        for(size_t k=0; k<nz+1; ++k  ) {
        #endif  
            for( size_t j=0; j<ny+1; ++j) { 
                for(size_t i=0; i<nx+1; ++i){
                    auto const pcoords = 
                        coord_system.get_physical_coordinates( {VEC(i,j,k)}
                                                             , iq
                                                             , {VEC( 0,0,0 )}
                                                             , false) ; 
                    auto const ipoint = get_point_index(VEC(i,j,k),iq) ; 
                    points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
                    points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
                    points[GRACE_NSPACEDIM*ipoint + 2 ] 
                    #ifdef GRACE_3D 
                        = pcoords[2] ;
                    #else 
                        = 0.0 ; 
                    #endif  
                }
            }
        #ifdef GRACE_3D
        }
        #endif 
    }
    /* Create parallel dataset properties */
    hid_t dxpl ; 
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ; 
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ; 

    /* Create/open datasets */
    hid_t points_space_id_glob ;
    hsize_t points_dset_dims_glob[2] = {npoints_glob, 3} ;   
    /* Create global space for points dataset */
    HDF5_CALL(points_space_id_glob, H5Screate_simple(2, points_dset_dims_glob, NULL)) ; 
    hid_t cells_space_id_glob ;
    hsize_t cells_dset_dims_glob[2] = {ncells_glob, nvertex} ;   
    /* Create global space for cells dataset */
    HDF5_CALL(cells_space_id_glob, H5Screate_simple(2, cells_dset_dims_glob, NULL)) ; 

    hid_t points_dset_id, cells_dset_id  ;
    hid_t points_prop_id, cells_prop_id  ;
    HDF5_CALL(points_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t points_chunk_dim[2] = {chunk_size,3} ;
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
    hid_t attr_id, attr_dataspace_id; 
    const std::string dataset_attr_name = "CellTopology";
    const char * dataset_attr_data = 
    #ifdef GRACE_3D
        "Hexahedron";
    #else 
        "Quadrilateral";
    #endif 
    hid_t str_type ; 
    HDF5_CALL(str_type,H5Tcopy(H5T_C_S1));
    HDF5_CALL(err,H5Tset_size(str_type, H5T_VARIABLE));
    HDF5_CALL(attr_dataspace_id, H5Screate(H5S_SCALAR));
    HDF5_CALL(attr_id,H5Acreate2(cells_dset_id, dataset_attr_name.c_str(), str_type, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CALL(err,H5Awrite(attr_id, str_type, &dataset_attr_data));
    HDF5_CALL(err,H5Aclose(attr_id));
    HDF5_CALL(err,H5Sclose(attr_dataspace_id));

    /* Write cells dataset */
    /* Create local space for this rank */
    hid_t cells_space_id ; 
    hsize_t cells_dset_dims[2] = {ncells, nvertex} ;
    HDF5_CALL(cells_space_id, H5Screate_simple(2, cells_dset_dims, NULL)) ; 
    /* Select hyperslab for this rank's output */
    hsize_t cells_slab_start[2]  = {local_quad_offset * ncells_quad,0} ; 
    hsize_t cells_slab_count[2]  = {ncells,nvertex} ;
    GRACE_VERBOSE("Slab offset {}, size {}, total {}", cells_slab_start[0], ncells, ncells_glob) ;  
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
    hsize_t points_dset_dims[2] = {npoints, GRACE_NSPACEDIM} ;
    HDF5_CALL(points_space_id, H5Screate_simple(2, points_dset_dims, NULL)) ; 
    /* Select hyperslab for this rank's output */
    hsize_t points_slab_start[2]  = {global_point_offset,0} ;
    GRACE_VERBOSE("Slab offset {}, size {}, total {}", points_slab_start[0], npoints, npoints_glob) ;  
    hsize_t points_slab_count[2]  = {npoints,GRACE_NSPACEDIM} ;
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

    auto& runtime = grace::runtime::get() ;

    auto const scalars     = runtime.cell_volume_output_scalar_vars() ; 
    auto const aux_scalars = runtime.cell_volume_output_scalar_aux()  ;
    auto const vectors     = runtime.cell_volume_output_vector_vars() ; 
    auto const aux_vectors = runtime.cell_volume_output_vector_aux()  ;

    size_t nx,ny,nz,nq; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = grace::amr::get_n_ghosts() ; 


    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
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

    /* Scalars */
    /* Create/open datasets */
    hid_t scalars_space_id_glob ;
    hsize_t scalars_dset_dims_glob[1] = {ncells_glob} ;

    /* Create global space for points dataset */
    HDF5_CALL(scalars_space_id_glob, H5Screate_simple(1, scalars_dset_dims_glob, NULL)) ;

    /* Create dataset properties and set chunking / compression mode */
    hid_t scalars_prop_id ;
    HDF5_CALL(scalars_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t scalars_chunk_dim[1] = {chunk_size} ;
    HDF5_CALL(err, H5Pset_chunk(scalars_prop_id,1,scalars_chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(scalars_prop_id, compression_level)) ;  

    /* Create local space for this rank */
    hid_t scalars_space_id ; 
    hsize_t scalars_dset_dims[1] = {ncells} ;
    HDF5_CALL(scalars_space_id, H5Screate_simple(1, scalars_dset_dims, NULL)) ; 

    /* Write data to file */
    write_var_arrays_hdf5( scalars,file_id,dxpl,scalars_space_id_glob
                         , scalars_space_id,scalars_prop_id,ncells,ncells_glob,local_quad_offset,false) ; 
    write_var_arrays_hdf5( aux_scalars,file_id,dxpl,scalars_space_id_glob
                         , scalars_space_id,scalars_prop_id,ncells,ncells_glob,local_quad_offset,true) ; 

    /* Close data spaces */
    HDF5_CALL(err, H5Sclose(scalars_space_id)) ; 
    HDF5_CALL(err, H5Sclose(scalars_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(scalars_prop_id)) ;

    /* Vectors */
    /* Create/open datasets */
    hid_t vectors_space_id_glob ;
    hsize_t vectors_dset_dims_glob[2] = {ncells_glob, 3} ;

    /* Create global space for points dataset */
    HDF5_CALL(vectors_space_id_glob, H5Screate_simple(2, vectors_dset_dims_glob, NULL)) ;

    /* Create dataset properties and set chunking / compression mode */
    hid_t vectors_prop_id ;
    HDF5_CALL(vectors_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t vectors_chunk_dim[2] = {chunk_size, 3} ;
    HDF5_CALL(err, H5Pset_chunk(vectors_prop_id,2,vectors_chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(vectors_prop_id, compression_level)) ;  

    /* Create local space for this rank */
    hid_t vectors_space_id ; 
    hsize_t vectors_dset_dims[2] = {ncells, 3} ;
    HDF5_CALL(vectors_space_id, H5Screate_simple(2, vectors_dset_dims, NULL)) ; 

    /* Write to file */
    write_vector_var_arrays_hdf5( vectors,file_id,dxpl,vectors_space_id_glob
                                , vectors_space_id,vectors_prop_id,ncells,ncells_glob,local_quad_offset,false) ; 
    write_vector_var_arrays_hdf5( aux_vectors,file_id,dxpl,vectors_space_id_glob
                                 , vectors_space_id,vectors_prop_id,ncells,ncells_glob,local_quad_offset,true) ; 
    
    /* Close data spaces */
    HDF5_CALL(err, H5Sclose(vectors_space_id)) ; 
    HDF5_CALL(err, H5Sclose(vectors_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(vectors_prop_id)) ;

    /* Cleanup and exit */
    HDF5_CALL(err, H5Pclose(dxpl)) ;
     
}

void write_var_arrays_hdf5( std::vector<std::string> const& varlist 
                          , hid_t file_id 
                          , hid_t dxpl
                          , hid_t space_id_glob
                          , hid_t space_id
                          , hid_t prop_id
                          , hsize_t ncells
                          , hsize_t ncells_glob
                          , hsize_t local_quad_offset
                          , bool isaux  ) 
{
    herr_t err ; 
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = grace::amr::get_n_ghosts() ;
    /* Number of cells per quadrant */
    unsigned long const ncells_quad = EXPR(nx,*ny,*nz) ; 

    /* Fetch variable arrays */
    auto vars = grace::variable_list::get().getstate() ; 
    auto aux  = grace::variable_list::get().getaux()   ;
    auto& view = isaux ? aux : vars ;

    /**********************************************/
    /* We need an extra device mirror because:    */
    /* 1) The view is not contiguous since we     */
    /*    cut out the ghost-zones.                */
    /* 2) The layout may differ from the          */
    /*    memory layout on device which           */
    /*    usually follows the FORTRAN convention. */
    /**********************************************/
    Kokkos::View<double EXPR(*,*,*)*, Kokkos::LayoutLeft> 
        d_mirror("Device output mirror", VEC(nx,ny,nz), nq) ; 
    auto h_mirror = Kokkos::create_mirror_view(d_mirror) ; 
    for( int ivar=0; ivar<varlist.size(); ++ivar)
    {
        size_t varidx = grace::get_variable_index(varlist[ivar],isaux) ; 
        GRACE_TRACE("Writing var {} to output. Variable index {} from auxiliaries? {}"
                   , varlist[ivar], varidx, isaux) ; 
        /* create HDF5 dataset */
        std::string dset_name = "/" + varlist[ivar] ; 
        hid_t dset_id ; 
        HDF5_CALL( dset_id
                , H5Dcreate2( file_id
                            , dset_name.c_str()
                            , H5T_NATIVE_DOUBLE
                            , space_id_glob
                            , H5P_DEFAULT
                            , prop_id
                            , H5P_DEFAULT) ) ;

        /* Write attributes to dataset */
        write_dataset_string_attribute_hdf5(dset_id, "VariableType", "Scalar");
        write_dataset_string_attribute_hdf5(dset_id, "VariableStaggering", "Cell");

        /* Shuffle data around to put it in the right form */
        auto sview = Kokkos::subview( view
                                    , Kokkos::pair<int,int>(ngz,nx+ngz)
                                    , Kokkos::pair<int,int>(ngz,ny+ngz)
                                    #ifdef GRACE_3D
                                    , Kokkos::pair<int,int>(ngz,nz+ngz)
                                    #endif 
                                    , varidx
                                    , Kokkos::ALL() ) ; 
 
        /* This deep copy operation is asynchronous */
        Kokkos::deep_copy(d_mirror, sview) ;
        /* Copy data d2h */
        Kokkos::deep_copy(grace::default_execution_space{},h_mirror,d_mirror) ; 

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[1]  = {local_quad_offset * ncells_quad} ; 
        hsize_t slab_count[1]  = {ncells} ;
        HDF5_CALL( err
                , H5Sselect_hyperslab( space_id_glob
                                    , H5S_SELECT_SET
                                    , slab_start
                                    , NULL
                                    , slab_count 
                                    , NULL )) ;
        Kokkos::fence() ; 
        /* write to dataset */
        HDF5_CALL( err
                    , H5Dwrite( dset_id
                            , H5T_NATIVE_DOUBLE
                            , space_id
                            , space_id_glob 
                            , dxpl 
                            , reinterpret_cast<void*>(h_mirror.data()) )) ;

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }
}

void write_vector_var_arrays_hdf5( std::vector<std::string> const& varlist 
                                 , hid_t file_id 
                                 , hid_t dxpl
                                 , hid_t space_id_glob
                                 , hid_t space_id
                                 , hid_t prop_id
                                 , hsize_t ncells
                                 , hsize_t ncells_glob
                                 , hsize_t local_quad_offset
                                 , bool isaux  ) 
{
    using namespace grace; 
    using namespace Kokkos; 
    herr_t err ;
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = grace::amr::get_n_ghosts() ;
    /* Number of cells per quadrant */
    unsigned long const ncells_quad = EXPR(nx,*ny,*nz) ; 

    /* Fetch variable arrays */
    auto vars = grace::variable_list::get().getstate() ; 
    auto aux  = grace::variable_list::get().getaux()   ;
    auto& view = isaux ? aux : vars ;

    /**********************************************/
    /* We need an extra device mirror because:    */
    /* 1) The view is not contiguous since we     */
    /*    cut out the ghost-zones.                */
    /* 2) The layout may differ from the          */
    /*    memory layout on device which           */
    /*    usually follows the FORTRAN convention. */
    /**********************************************/
    Kokkos::View<double *EXPR(*,*,*)*, Kokkos::LayoutLeft> 
        d_mirror("Device output mirror", 3, VEC(nx,ny,nz), nq) ; 
    auto h_mirror = Kokkos::create_mirror_view(d_mirror) ; 
    for( auto const& vname: varlist )
    {
        std::array<std::string, 3> const compnames 
                = {
                    vname + "[0]",
                    vname + "[1]",
                    vname + "[2]"
                } ;
        size_t varidx = grace::get_variable_index(compnames[0],isaux) ; 
        GRACE_TRACE("Writing vector var {} to output. Variable index {} from auxiliaries? {}"
                   , vname, varidx, isaux) ; 
        /* create HDF5 dataset */
        std::string dset_name = "/" + vname ; 
        hid_t dset_id ; 
        HDF5_CALL( dset_id
                , H5Dcreate2( file_id
                            , dset_name.c_str()
                            , H5T_NATIVE_DOUBLE
                            , space_id_glob
                            , H5P_DEFAULT
                            , prop_id
                            , H5P_DEFAULT) ) ;
        /* Write dataset attributes */
        write_dataset_string_attribute_hdf5(dset_id, "VariableType", "Vector");
        write_dataset_string_attribute_hdf5(dset_id, "VariableStaggering", "Cell");

        /* Shuffle data around to put it in the right form */
        
        /* Copy data d2d */
        auto sview = Kokkos::subview( view
                                    , Kokkos::pair<int,int>(ngz,nx+ngz)
                                    , Kokkos::pair<int,int>(ngz,ny+ngz)
                                    #ifdef GRACE_3D
                                    , Kokkos::pair<int,int>(ngz,nz+ngz)
                                    #endif 
                                    , Kokkos::pair<int,int>(varidx, varidx+3)
                                    , Kokkos::ALL() ) ; 
        auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>{
            {0,0,0,0,0}, {static_cast<long>(nx),static_cast<long>(ny),static_cast<long>(nz),3,static_cast<long>(nq)}
        } ; 
        Kokkos::parallel_for("copy_to_mirror", policy, 
            KOKKOS_LAMBDA (int i, int j, int k, int iv, int q) {
                d_mirror(iv, i,j,k,q) = sview(i,j,k,iv,q) ; 
            }
        ) ; 
            
        /************************************************/
        /* Transform variable to physical frame         */
        /************************************************/
        /* Fill coordinate jacobian matrices on device  */
        /* at cell centers                              */
        /************************************************/
        #if 0
        jacobian_array_t jac, invjac ; 
        fill_jacobian_matrices(jac,invjac) ;  
        parallel_for(GRACE_EXECUTION_TAG("IO","convert_to_physical")
                , MDRangePolicy<Rank<GRACE_NSPACEDIM+1>,default_execution_space>({VEC(0,0,0),0},{VEC(nx+2*ngz,ny+2*ngz,nz+2*ngz),nq})
                , KOKKOS_LAMBDA (VEC(int const& i, int const& j, int const& k), int const& q) 
                {
                    std::array<double,3> const vin
                    {
                          d_mirror(0, VEC(i,j,k), q)
                        , d_mirror(1, VEC(i,j,k), q)
                        , d_mirror(2, VEC(i,j,k), q)
                    } ; 
                    auto J = Kokkos::subview(jac, VEC(i,j,k), ALL(), ALL(),  q) ; 
                    auto const vout = ::grace::detail::apply_jacobian_vec(vin, J) ;
                    d_mirror(0, VEC(i,j,k), q) = vout[0] ; 
                    d_mirror(1, VEC(i,j,k), q) = vout[1] ; 
                    d_mirror(2, VEC(i,j,k), q) = vout[2] ;
                    }
        ) ;
        /* just to be sure add a fence */
        Kokkos::fence() ;
        #endif
        
        /* Copy data d2h */
        Kokkos::deep_copy(grace::default_execution_space{},h_mirror,d_mirror) ; 

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[2]  = {local_quad_offset * ncells_quad, 0} ; 
        hsize_t slab_count[2]  = {ncells, 3} ;
        HDF5_CALL( err
                , H5Sselect_hyperslab( space_id_glob
                                    , H5S_SELECT_SET
                                    , slab_start
                                    , NULL
                                    , slab_count 
                                    , NULL )) ;
        Kokkos::fence() ; 

        /* write to dataset */
        HDF5_CALL( err
                    , H5Dwrite( dset_id
                            , H5T_NATIVE_DOUBLE
                            , space_id
                            , space_id_glob 
                            , dxpl 
                            , reinterpret_cast<void*>(h_mirror.data()) )) ;
        

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }
}

void write_extra_arrays_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size) {
    herr_t err ;
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    nq = grace::amr::get_local_num_quadrants() ; 
    size_t ngz = grace::amr::get_n_ghosts() ;

    auto const rank_loc = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ; 
    unsigned long const local_quad_offset = _p4est->global_first_quadrant[rank_loc] ; 

    /* Number of cells per quadrant */
    unsigned long const ncells_quad = EXPR(nx,*ny,*nz) ; 
    /* Local number of cells   */
    unsigned long const ncells = ncells_quad * nq ; 
    /* Global number of cells  */
    unsigned long const ncells_glob = ncells_quad * nq_glob ; 

    unsigned int* lev   = (unsigned int*) malloc(sizeof(unsigned int) * ncells ) ;
    unsigned int* rank  = (unsigned int*) malloc(sizeof(unsigned int) * ncells ) ;
    unsigned int* tree_id    = (unsigned int*) malloc(sizeof(unsigned int) * ncells ) ;
    unsigned long long* qid  = (unsigned long long*) malloc(sizeof(unsigned long long) * ncells ) ;  

    unsigned int icell  = 0L ; 
    //#pragma omp parallel for collapse(GRACE_NSPACEDIM+1) reduction(+:icell)
    for(int64_t iq=0; iq<nq; ++iq) {
        #ifdef GRACE_3D
        for(size_t k=0; k<nz; ++k  ) {
        #endif  
            for( size_t j=0; j<ny; ++j) { 
                for(size_t i=0; i<nx; ++i){
                unsigned int itree = amr::get_quadrant_owner(iq) ; 
                auto quad  = amr::get_quadrant(itree,iq) ; 
                unsigned int level = quad.level() ;
                size_t iquad_glob = iq + amr::forest::get().global_quadrant_offset(rank_loc) ;

                rank[icell]    = rank_loc   ;
                lev[icell]     = level      ;
                tree_id[icell] = itree      ;
                qid[icell]     = iquad_glob ; 
                icell ++ ; 
                #ifdef GRACE_3D
                }
                #endif 
            }
        }
    }

    ASSERT(icell == ncells, "Something went really wrong") ; 

    /* Create parallel dataset properties */
    hid_t dxpl ; 
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ; 
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ; 
    auto const offset = local_quad_offset * ncells_quad ; 
    write_scalar_dataset( static_cast<void*>(rank),H5T_NATIVE_UINT,file_id,dxpl
                        , ncells,ncells_glob,offset,chunk_size,compression_level,"/Rank") ; 
    write_scalar_dataset( static_cast<void*>(lev),H5T_NATIVE_UINT,file_id,dxpl
                        , ncells,ncells_glob,offset,chunk_size,compression_level,"/Level") ; 
    write_scalar_dataset( static_cast<void*>(tree_id),H5T_NATIVE_UINT,file_id,dxpl
                        , ncells,ncells_glob,offset,chunk_size,compression_level,"/Tree_ID") ; 
    write_scalar_dataset( static_cast<void*>(qid),H5T_NATIVE_ULLONG,file_id,dxpl
                        , ncells,ncells_glob,offset,chunk_size,compression_level,"/Quad_ID") ;
    
    /* Release resources */
    free(rank) ;
    free(lev) ;
    free(tree_id) ;
    free(qid) ;

    /* Cleanup and exit */
    HDF5_CALL(err, H5Pclose(dxpl)) ;

    
}

void write_scalar_dataset( void* data, hid_t mem_type_id, hid_t file_id, hid_t dxpl
                         , hsize_t dset_size, hsize_t dset_size_glob, hsize_t offset
                         , size_t chunk_size, unsigned int compression_level
                         , std::string const& dset_name ) 
{
    herr_t err ;
    /* Create/open datasets */
    hid_t space_id_glob ;
    hsize_t dset_dims_glob[1] = {dset_size_glob} ;   
    /* Create global space for points dataset */
    HDF5_CALL(space_id_glob, H5Screate_simple(1, dset_dims_glob, NULL)) ; 

    hid_t dset_id ;
    hid_t prop_id ;
    HDF5_CALL(prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t chunk_dim[1] = {chunk_size} ;
    HDF5_CALL(err, H5Pset_chunk(prop_id,1,chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(prop_id, compression_level)) ; 
    HDF5_CALL( dset_id
             , H5Dcreate2( file_id
                         , dset_name.c_str()
                         , mem_type_id
                         , space_id_glob
                         , H5P_DEFAULT
                         , prop_id
                         , H5P_DEFAULT) ) ;;

    write_dataset_string_attribute_hdf5(dset_id, "VariableType", "Scalar");
    write_dataset_string_attribute_hdf5(dset_id, "VariableStaggering", "Cell");
    
    /* Write points dataset */
    /* Create local space for this rank */
    hid_t space_id ; 
    hsize_t dset_dims[1] = {dset_size} ;
    HDF5_CALL(space_id, H5Screate_simple(1, dset_dims, NULL)) ; 
    /* Select hyperslab for this rank's output */
    hsize_t slab_start[1]  = {offset} ;
    hsize_t slab_count[1]  = {dset_size} ;
    HDF5_CALL( err
             , H5Sselect_hyperslab( space_id_glob
                                  , H5S_SELECT_SET
                                  , slab_start
                                  , NULL
                                  , slab_count 
                                  , NULL )) ;
    /* Write data corresponding to this rank to disk */
    HDF5_CALL( err
             , H5Dwrite( dset_id
                       , mem_type_id
                       , space_id
                       , space_id_glob 
                       , dxpl 
                       , data )) ; 
    HDF5_CALL(err, H5Dclose(dset_id)) ; 
    HDF5_CALL(err, H5Sclose(space_id)) ; 
    HDF5_CALL(err, H5Sclose(space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(prop_id)) ;

}

void write_dataset_string_attribute_hdf5(hid_t dset_id, std::string const& attr_name, std::string const& attr_data)
{
    hid_t attr_id, attr_dataspace_id, str_type;
    herr_t err;

    // Create a scalar dataspace for the attribute
    HDF5_CALL(attr_dataspace_id, H5Screate(H5S_SCALAR));
    
    // Create a variable-length string datatype
    HDF5_CALL(str_type, H5Tcopy(H5T_C_S1));
    HDF5_CALL(err, H5Tset_size(str_type, H5T_VARIABLE));

    // Create the attribute
    HDF5_CALL(attr_id, H5Acreate2(dset_id, attr_name.c_str(), str_type, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT));

    // Write the attribute data
    const char* attr_data_cstr = attr_data.c_str();
    HDF5_CALL(err, H5Awrite(attr_id, str_type, &attr_data_cstr));

    // Close the attribute and dataspace
    HDF5_CALL(err, H5Aclose(attr_id));
    HDF5_CALL(err, H5Sclose(attr_dataspace_id));
}
}} /* namespace grace::IO */
