/**
 * @file hdf5_surface_output.cpp
 * @author Keneth Miler (miler@itp.uni-frankfurt.de) & Carlo Musolino (carlo.musolino@aei.mpg.de)
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
#include <grace/data_structures/variable_utils.hh>
#include <grace/data_structures/memory_defaults.hh>

#include <grace/system/grace_system.hh>
#include <grace/IO/hdf5_output.hh>
#include <grace/IO/hdf5_surface_output.hh>
#include <grace/IO/octree_search_class.hh>

#include <grace/parallel/mpi_wrappers.hh>

#include <hdf5.h>
#include <omp.h>
/* xmf */
#include <grace/IO/xmf_utilities.hh>
/* stl */
#include <string>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <Kokkos_UnorderedMap.hpp>

#define HDF5_CALL(result,cmd) \
    do {  \
        if((result=cmd)<0) { \
            ERROR("HDF5 API call failed with error code " << result ) ; \
        } \
    } while(false)


namespace grace { namespace IO {

namespace detail {
std::vector<int64_t> _volume_output_sliced_iterations ;
std::vector<double>  _volume_output_sliced_times ;
std::vector<int64_t> _volume_output_sliced_ncells ; 
std::vector<std::string> _volume_output_sliced_filenames ; 
}


void write_plane_cell_data() {
    Kokkos::Profiling::pushRegion("HDF5 plane output") ;
    GRACE_VERBOSE("Performing HDF5 output of surface data.") ; 
    auto& rt = grace::runtime::get() ; 

    auto n_planes = 3 ; 
    std::vector<std::string> plane_names = std::vector<std::string>( {"xy","xz","yz"} ) ; 
    auto plane_offsets = rt.cell_plane_surface_output_origins() ; 
    std::vector<amr::plane_axis> plane_axes{amr::plane_axis::XY, amr::plane_axis::XZ, amr::plane_axis::YZ } ; 
    for( int i=0 ; i<n_planes; ++i) {
        amr::plane_desc_t plane ; 
        plane.name  = plane_names[i] ; 
        plane.coord = plane_offsets[i] ;
        plane.axis  = plane_axes[i] ;  
        write_plane_cell_data_impl(plane) ; 
    }

    Kokkos::Profiling::popRegion() ; 
}

void write_plane_cell_data_impl(amr::plane_desc_t const& plane) {
    

    detail::_volume_output_sliced_iterations.push_back(grace::get_iteration())  ; 
    detail::_volume_output_sliced_times.push_back(grace::get_simulation_time()) ;

    /* figure out where to output */
    auto& runtime = grace::runtime::get() ; 
    std::filesystem::path base_path (runtime.surface_io_basepath()) ;
    std::ostringstream oss;
    oss << runtime.surface_io_basename() << "_plane" << "_" << plane.name << "_"
        << std::setw(6) << std::setfill('0') << grace::get_iteration() << ".h5";
    std::string pfname = oss.str();
    std::filesystem::path out_path = base_path / pfname ;
    detail::_volume_output_sliced_filenames.push_back(pfname) ;

    /* fetch some params */
    auto& params = grace::config_parser::get() ;
    size_t compression_level = params["IO"]["hdf5_compression_level"].as<size_t>() ;
    size_t chunk_size = params["IO"]["hdf5_chunk_size"].as<size_t>() ;


    /* number of cells per quad */
    size_t nx,ny,nz ;
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    size_t nq = amr::get_local_num_quadrants() ; 

    /* Get the sliced octants*/
    amr::oct_tree_plane_slicer_t octree_slicer(plane,nx,nq);
    octree_slicer.slice() ; 
    auto sliced_nq = octree_slicer.n_sliced_quads() ; 
    auto sliced_cells = sliced_nq * nx * nx ;
    // find out how many cells in total 
    size_t glob_sliced_nq ;
    parallel::mpi_allreduce(
        &sliced_nq, &glob_sliced_nq, 1, mpi_sum 
    ) ; 
    // write back some info for later use 
    octree_slicer.ncells = sliced_cells ; 
    octree_slicer.glob_ncells = glob_sliced_nq * nx * nx ; 
    octree_slicer.glob_nq = glob_sliced_nq ;  
    if( chunk_size > octree_slicer.glob_ncells ) {
        GRACE_WARN("Chunk size {} < number of cells {} will be overridden." , chunk_size, octree_slicer.glob_ncells) ; 
        chunk_size = math::max(1UL,octree_slicer.glob_ncells); 
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
    write_grid_structure_sliced_hdf5(file_id, compression_level,chunk_size, octree_slicer) ;
    /* Write requested variables to hdf5 file */
    write_volume_data_arrays_sliced_hdf5(file_id, compression_level,chunk_size, octree_slicer) ;
    /* Write extra quantities if requested */
    bool output_extra = grace::get_param<bool>("IO","output_extra_quantities") ; 
    if( output_extra ) {
        write_extra_arrays_sliced_hdf5(file_id, compression_level, chunk_size, octree_slicer) ; 
    }
    parallel::mpi_barrier() ; 
    /* Close the file */
    HDF5_CALL(err,H5Fclose(file_id)) ; 
    HDF5_CALL(err,H5Pclose(plist_id)) ; 
}

void write_grid_structure_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::oct_tree_plane_slicer_t& octree_slicer) {
    herr_t err ; 

    auto& coord_system = grace::coordinate_system::get() ; 
    size_t nx,ny,nz;
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    int ngz = grace::amr::get_n_ghosts() ;
    size_t nq = grace::amr::get_local_num_quadrants() ; 

    size_t nq_s = octree_slicer.sliced_quads.size();

    #ifdef GRACE_CARTESIAN_COORDINATES
    size_t constexpr nvertex = 4 ;
    #elif defined(GRACE_SPHERICAL_COORDINATES)
    ASSERT(0, "This does not work unless cartesian.") ; 
    #endif 

    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get local offset */
    size_t local_quad_offset; // has to be the same type as nq
    parallel::mpi_exscan_sum( &nq_s, &local_quad_offset, 1, parallel::get_comm_world() ) ;
    if (rank == 0) local_quad_offset = 0 ;

    octree_slicer.local_quad_offset = local_quad_offset ; 

    /* Number of unique points per quadrant */
    unsigned long npoints_quad_sliced = (nx+1) * (nx+1); 

    /* Local number of points  */
    unsigned long const npoints_sliced = npoints_quad_sliced * nq_s ;  
    /* Global number of points */
    unsigned long const npoints_glob_sliced = npoints_quad_sliced * octree_slicer.glob_nq ;

    /* Local number of cells */
    unsigned long const ncells_sliced = octree_slicer.ncells ;
    unsigned long const ncells_quad_sliced = nx*nx ; 
    unsigned long const ncells_glob = ncells_quad_sliced * octree_slicer.glob_nq ; 
 
    detail::_volume_output_sliced_ncells.push_back(octree_slicer.glob_nq) ; 

    double*  points      = (double      *) malloc(sizeof(double)  * npoints_sliced * 3 )                ; 
    unsigned long int* cells  = (unsigned long int*) malloc(sizeof(unsigned long int) * ncells_sliced * nvertex ) ; 
    const size_t global_point_offset_sliced = local_quad_offset * npoints_quad_sliced ;  
    unsigned long int icell  = 0UL ; 
    unsigned long int ipoint = 0UL ; 
    auto const get_point_index = [&] ( int i, int j, int64_t q ) 
    {
        return i + (nx+1) * (j + (nx+1) * q) ; 
    } ; 
    auto const get_cell_vertex_indices = [&] ( int i, int j, int iv) 
    {
            static constexpr std::array<std::array<int,3>,4> vertex_coords {{
                {0, 0, 0}, //
                {1, 0, 0}, //
                {1, 1, 0}, //
                {0, 1, 0}  //
            }} ; 

            return std::make_tuple(
                i+vertex_coords[iv][0],
                j+vertex_coords[iv][1]
            ) ; 
    } ;

    size_t ip, jp ; 
    #pragma omp parallel for collapse(3) private(ip,jp)
    for( int iq = 0; iq<octree_slicer.sliced_quads.size(); ++iq) {
        for( int i=0; i<nx; ++i) {
            for( int j=0; j<nx; ++j){
                size_t const icell = (i + nx * ( j + nx * iq )) ; 
                for( int iv=0; iv<nvertex; ++iv) {
                    std::tie(ip,jp) = get_cell_vertex_indices(i,j,iv) ; 
                    // this indexes into the global buffer 
                    cells[nvertex*icell + iv] = get_point_index(ip,jp,iq+local_quad_offset) ; 
                }
            }
        }
    }

    // coordinates 
    //#pragma omp parallel for 
    for( int iq = 0; iq<octree_slicer.sliced_quads.size(); ++iq) {
        for( int i=0; i<nx+1; ++i) for( int j=0; j<nx+1; ++j) {
            auto const q = octree_slicer.sliced_quads[iq] ; 
            auto const ipoint = get_point_index(i,j,iq) ; 
            std::array<size_t,3> ijk ; 
            if ( octree_slicer._plane.axis == amr::plane_axis::YZ ) {
                ijk[0] = octree_slicer.sliced_cell_offsets[iq] ; 
                ijk[1] = i ; ijk[2] = j ;
            } else if ( octree_slicer._plane.axis == amr::plane_axis::XZ ) {
                ijk[1] = octree_slicer.sliced_cell_offsets[iq] ; 
                ijk[0] = i ; ijk[2] = j ;
            } else {
                ijk[2] = octree_slicer.sliced_cell_offsets[iq] ; 
                ijk[0] = i ; ijk[1] = j ;
            }
            auto const pcoords = 
                        coord_system.get_physical_coordinates( ijk
                                                             , q /* note! */
                                                             , {VEC( 0,0,0 )}
                                                             , false) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ; 
        }
    }

    /* Create parallel dataset properties */
    hid_t dxpl ; 
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ; 
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ; 

    /* Create/open datasets */
    hid_t points_space_id_glob ;
    hsize_t points_dset_dims_glob[2] = {npoints_glob_sliced, 3} ;   
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
    if ( compression_level > 0 )
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
    if ( compression_level > 0 )
        HDF5_CALL(err, H5Pset_deflate(cells_prop_id, compression_level)) ; 
    HDF5_CALL( cells_dset_id
                , H5Dcreate2( file_id
                            , "/Cells"
                            , H5T_NATIVE_ULONG
                            , cells_space_id_glob
                            , H5P_DEFAULT
                            , cells_prop_id
                            , H5P_DEFAULT) ) ;
    hid_t attr_id, attr_dataspace_id; 
    const std::string dataset_attr_name = "CellTopology";
    const char * dataset_attr_data = "Quadrilateral";
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
    hsize_t cells_dset_dims[2] = {ncells_sliced, nvertex} ;
    HDF5_CALL(cells_space_id, H5Screate_simple(2, cells_dset_dims, NULL)) ; 
    /* Select hyperslab for this rank's output */
    hsize_t cells_slab_start[2]  = {local_quad_offset * ncells_quad_sliced,0} ; 
    hsize_t cells_slab_count[2]  = {ncells_sliced,nvertex} ;
    GRACE_VERBOSE("Slab offset {}, size {}, total {}", cells_slab_start[0], ncells_sliced, octree_slicer.glob_nq) ;  
    HDF5_CALL( err
             , H5Sselect_hyperslab( cells_space_id_glob
                                  , H5S_SELECT_SET
                                  , cells_slab_start
                                  , NULL
                                  , cells_slab_count 
                                  , NULL )) ;

    HDF5_CALL( err
             , H5Dwrite( cells_dset_id
                       , H5T_NATIVE_ULONG
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
    hsize_t points_dset_dims[2] = {npoints_sliced, GRACE_NSPACEDIM} ;
    HDF5_CALL(points_space_id, H5Screate_simple(2, points_dset_dims, NULL)) ; 
    /* Select hyperslab for this rank's output */
    hsize_t points_slab_start[2]  = {global_point_offset_sliced,0} ;
    GRACE_VERBOSE("Slab offset {}, size {}, total {}", points_slab_start[0], npoints_sliced, npoints_glob_sliced) ;  
    hsize_t points_slab_count[2]  = {npoints_sliced,GRACE_NSPACEDIM} ;
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

void write_volume_data_arrays_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::oct_tree_plane_slicer_t& octree_slicer) {

    herr_t err ;

    auto& runtime = grace::runtime::get() ;

    auto const scalars     = runtime.cell_plane_surface_output_scalar_vars() ; 
    auto const aux_scalars = runtime.cell_plane_surface_output_scalar_aux()  ;
    auto const vectors     = runtime.cell_plane_surface_output_vector_vars() ; 
    auto const aux_vectors = runtime.cell_plane_surface_output_vector_aux()  ;

    size_t nx,ny,nz ;
    std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
    size_t nq_s = octree_slicer.sliced_quads.size();
    size_t ngz = grace::amr::get_n_ghosts() ; 

    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob_sliced = octree_slicer.glob_nq ;
    auto local_quad_offset_sliced = octree_slicer.local_quad_offset ; 

    /* Number of cells per quadrant */
    unsigned long const ncells_quad_sliced = nx*nx ; 
    /* Local number of cells   */
    unsigned long const ncells_sliced = ncells_quad_sliced * nq_s; //octree_slicer.num_sliced_cells() ; 
    /* Global number of cells  */
    unsigned long const ncells_glob_sliced = ncells_quad_sliced * nq_glob_sliced ; 
    /* Number of unique points per quadrant */
    unsigned long ncorners_quad_sliced = (nx+1) * (nx+1); 

    /* Create parallel dataset properties */
    hid_t dxpl ; 
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ; 
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ;

    /*****************************************************************************************/
    /*                                     Scalars                                           */
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                               Create/open datasets                                    */
    /*****************************************************************************************/

    /* 1) Cell centered variables */
    hid_t scalars_space_id_glob ;
    hsize_t scalars_dset_dims_glob[1] = {ncells_glob_sliced} ;

    /* Create global space for points dataset */
    HDF5_CALL(scalars_space_id_glob, H5Screate_simple(1, scalars_dset_dims_glob, NULL)) ;

    /* Create dataset properties and set chunking / compression mode */
    hid_t scalars_prop_id ;
    HDF5_CALL(scalars_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t scalars_chunk_dim[1] = {chunk_size} ;
    HDF5_CALL(err, H5Pset_chunk(scalars_prop_id,1,scalars_chunk_dim)) ; 
    if ( compression_level > 0 )
        HDF5_CALL(err, H5Pset_deflate(scalars_prop_id, compression_level)) ;  

    /* Create local space for this rank */
    hid_t scalars_space_id ; 
    hsize_t scalars_dset_dims[1] = {ncells_sliced} ;
    HDF5_CALL(scalars_space_id, H5Screate_simple(1, scalars_dset_dims, NULL)) ; 

    /*****************************************************************************************/
    // Create host mirrors for keys and values.
    // Allocate the views in the default execution space (likely device memory)
    Kokkos::View<size_t*> d_quad_ids("plane_quad_ids", nq_s);
    Kokkos::View<size_t*> d_cell_offsets("plane_cell_offsets", nq_s);
    grace::deep_copy_vec_to_view(d_quad_ids, octree_slicer.sliced_quads) ; 
    grace::deep_copy_vec_to_view(d_cell_offsets, octree_slicer.sliced_cell_offsets) ; 
    /*****************************************************************************************/
    /*                                  Write to file                                        */
    /*****************************************************************************************/
    write_var_arrays_sliced_hdf5( 
        scalars, 
        file_id, dxpl,
        scalars_space_id_glob, scalars_space_id, scalars_prop_id,
        ncells_sliced, local_quad_offset_sliced, octree_slicer, d_quad_ids, d_cell_offsets) ; 
    write_var_arrays_sliced_hdf5( 
        aux_scalars, 
        file_id, dxpl,
        scalars_space_id_glob, scalars_space_id, scalars_prop_id,
        ncells_sliced, local_quad_offset_sliced, octree_slicer, d_quad_ids, d_cell_offsets, true ) ;
    /*****************************************************************************************/
    /*                                  Close data spaces                                    */
    /*****************************************************************************************/
    HDF5_CALL(err, H5Sclose(scalars_space_id)) ; 
    HDF5_CALL(err, H5Sclose(scalars_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(scalars_prop_id)) ;
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                                     Vectors                                           */
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                               Create/open datasets                                    */
    /*****************************************************************************************/

    /* 1) Cell centered variables */
    hid_t vectors_space_id_glob ;
    hsize_t vectors_dset_dims_glob[2] = {ncells_glob_sliced, 3} ;

    /* Create global space for points dataset */
    HDF5_CALL(vectors_space_id_glob, H5Screate_simple(2, vectors_dset_dims_glob, NULL)) ;

    /* Create dataset properties and set chunking / compression mode */
    hid_t vectors_prop_id ;
    HDF5_CALL(vectors_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t vectors_chunk_dim[2] = {chunk_size, 3} ;
    HDF5_CALL(err, H5Pset_chunk(vectors_prop_id,2,vectors_chunk_dim)) ; 
    if ( compression_level > 0 )
        HDF5_CALL(err, H5Pset_deflate(vectors_prop_id, compression_level)) ;  

    /* Create local space for this rank */
    hid_t vectors_space_id ; 
    hsize_t vectors_dset_dims[2] = {ncells_sliced, 3} ;
    HDF5_CALL(vectors_space_id, H5Screate_simple(2, vectors_dset_dims, NULL)) ; 
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                                  Write to file                                        */
    /*****************************************************************************************/
    write_vector_var_arrays_sliced_hdf5( 
        vectors,
        file_id,dxpl,
        vectors_space_id_glob,vectors_space_id,vectors_prop_id,
        ncells_sliced,local_quad_offset_sliced, octree_slicer, d_quad_ids, d_cell_offsets) ; 
    write_vector_var_arrays_sliced_hdf5( 
        aux_vectors,
        file_id,dxpl,
        vectors_space_id_glob,vectors_space_id,vectors_prop_id,
        ncells_sliced,local_quad_offset_sliced, octree_slicer, d_quad_ids, d_cell_offsets, true) ;
    /*****************************************************************************************/
    /*                                  Close data spaces                                    */
    /*****************************************************************************************/
    HDF5_CALL(err, H5Sclose(vectors_space_id)) ; 
    HDF5_CALL(err, H5Sclose(vectors_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(vectors_prop_id)) ;
    /*****************************************************************************************/
    /*****************************************************************************************/
    /*                                 Cleanup and exit                                      */
    /*****************************************************************************************/
    HDF5_CALL(err, H5Pclose(dxpl)) ;
    /*****************************************************************************************/
}

void write_var_arrays_sliced_hdf5( std::vector<std::string> const& varlist 
                          , hid_t file_id 
                          , hid_t dxpl
                          , hid_t space_id_glob
                          , hid_t space_id
                          , hid_t prop_id
                          , hsize_t ncells
                          , hsize_t local_quad_offset
                          , amr::oct_tree_plane_slicer_t& octree_slicer
                          , Kokkos::View<size_t*> d_quad_ids
                          , Kokkos::View<size_t*> d_cell_offsets
                          , bool isaux ) // had isaux 
{
    herr_t err ; 
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq;
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    auto nq_s = octree_slicer.sliced_quads.size() ; 
    nq = grace::amr::get_local_num_quadrants() ;
    size_t ngz = grace::amr::get_n_ghosts() ;
    /* Number of cells per quadrant */
    unsigned long const ncells_quad_sliced = nx*nx ; 
    /* Number of corners per quadrant */
    unsigned long ncorners_quad_sliced = (nx+1)*(nx+1) ;
    auto const dir = octree_slicer._plane.axis ; 
    //-----------
    auto vars = grace::variable_list::get().getstate() ; 
    auto aux  = grace::variable_list::get().getaux()   ;
    auto& view = isaux ? aux : vars ;
    //-----------
    /**********************************************/
    /* We need an extra device mirror because:    */
    /* 1) The view is not contiguous since we     */
    /*    cut out the ghost-zones.                */
    /* 2) The layout may differ from the          */
    /*    memory layout on device which           */
    /*    usually follows the FORTRAN convention. */
    /**********************************************/
    Kokkos::View<double ***, Kokkos::LayoutLeft>
        d_mirror("Device output mirror",nx,nx, nq_s) ;
    auto h_mirror = Kokkos::create_mirror_view(d_mirror) ;

    for( auto const& vname: varlist )
    {
        size_t varidx = grace::get_variable_index(vname,isaux) ; 
        GRACE_TRACE("Writing var {} to output.", vname) ;

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
        Kokkos::parallel_for(
            "copy_plane_data", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nx,nx,nq_s}),
            KOKKOS_LAMBDA( int i, int j, int iq) {
                int ijk[3] ; 
                if ( dir == amr::plane_axis::YZ ) {
                    ijk[0] = d_cell_offsets(iq) ; 
                    ijk[1] = i ; ijk[2] = j ; 
                } else if (dir == amr::plane_axis::XZ) {
                    ijk[1] = d_cell_offsets(iq) ; 
                    ijk[0] = i ; ijk[2] = j ; 
                } else {
                    ijk[2] = d_cell_offsets(iq) ; 
                    ijk[0] = i ; ijk[1] = j ; 
                }
                d_mirror(i,j,iq) = sview(
                    ijk[0],ijk[1],ijk[2],d_quad_ids(iq)
                ) ; 
            }) ; 

        Kokkos::deep_copy(grace::default_execution_space{},h_mirror,d_mirror) ;
        Kokkos::fence();

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[1]  = {local_quad_offset * ncells_quad_sliced} ; 
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

void write_vector_var_arrays_sliced_hdf5( std::vector<std::string> const& varlist 
                                 , hid_t file_id 
                                 , hid_t dxpl
                                 , hid_t space_id_glob
                                 , hid_t space_id
                                 , hid_t prop_id
                                 , hsize_t ncells
                                 , hsize_t local_quad_offset 
                                 , amr::oct_tree_plane_slicer_t& octree_slicer
                                 , Kokkos::View<size_t*> d_quad_ids
                                 , Kokkos::View<size_t*> d_cell_offsets
                                 , bool isaux ) // had isaux   
{
    using namespace grace; 
    using namespace Kokkos; 
    herr_t err ;
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq;
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    auto nq_s = octree_slicer.sliced_quads.size() ; 
    nq = grace::amr::get_local_num_quadrants() ;
    size_t ngz = grace::amr::get_n_ghosts() ;
    /* Number of cells per quadrant */
    unsigned long const ncells_quad_sliced = nx*nx ; 
    /* Number of corners per quadrant */
    unsigned long ncorners_quad_sliced = (nx+1)*(nx+1) ;
    auto const dir = octree_slicer._plane.axis ; 

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
    Kokkos::View<double ****, Kokkos::LayoutLeft> 
        d_mirror("Device output mirror", 3, nx, nx, nq) ; 
    auto h_mirror = Kokkos::create_mirror_view(d_mirror) ;
    for( auto const& vname: varlist )
    {
        GRACE_TRACE("Writing vector var {} to output.", vname) ; 
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
        
        std::array<std::string, 3> const compnames 
                = {
                    vname + "[0]",
                    vname + "[1]",
                    vname + "[2]"
                } ; 
        size_t varidx = grace::get_variable_index(compnames[0],isaux) ; 
        Kokkos::parallel_for(
            "copy_plane_data", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nx,nx,nq_s}),
            KOKKOS_LAMBDA( int i, int j, int iq) {
                int ijk[3] ; 
                if ( dir == amr::plane_axis::YZ ) {
                    ijk[0] = d_cell_offsets(iq) + ngz ; 
                    ijk[1] = i + ngz; ijk[2] = j + ngz; 
                } else if (dir==amr::plane_axis::XZ) {
                    ijk[1] = d_cell_offsets(iq) + ngz; 
                    ijk[0] = i + ngz; ijk[2] = j + ngz; 
                } else {
                    ijk[2] = d_cell_offsets(iq) + ngz; 
                    ijk[0] = i + ngz; ijk[1] = j + ngz; 
                }
                for( int icomp=0; icomp<3; ++icomp)
                    d_mirror(icomp,i,j,iq) = view(
                        ijk[0],ijk[1],ijk[2],varidx+icomp, d_quad_ids(iq)
                    ) ; 
        }) ;
    
        /* Copy data d2h */
        Kokkos::deep_copy(grace::default_execution_space{},h_mirror,d_mirror) ; 
        Kokkos::fence() ;

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[2]  = {local_quad_offset * ncells_quad_sliced, 0} ; 
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

void write_extra_arrays_sliced_hdf5(
    hid_t file_id, 
    size_t compression_level, 
    size_t chunk_size, 
    amr::oct_tree_plane_slicer_t& octree_slicer
) {
    herr_t err ;
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq;
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    auto nq_s = octree_slicer.sliced_quads.size() ; 
    nq = grace::amr::get_local_num_quadrants() ;
    size_t ngz = grace::amr::get_n_ghosts() ;

    /* Number of corners per quadrant */
    unsigned long ncorners_quad_sliced = (nx+1)*(nx+1) ;

    auto const rank_loc = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */

    unsigned long const nq_glob = octree_slicer.glob_nq ;
    auto local_quad_offset = octree_slicer.local_quad_offset ; 


    /* Number of cells per quadrant */
    unsigned long const ncells_quad = nx*nx ; 
    /* Local number of cells   */
    unsigned long const ncells = ncells_quad * nq_s ; 
    /* Global number of cells  */
    unsigned long const ncells_glob = ncells_quad * nq_glob ; 

    unsigned int* lev   = (unsigned int*) malloc(sizeof(unsigned int) * ncells ) ;
    unsigned int* rank  = (unsigned int*) malloc(sizeof(unsigned int) * ncells ) ;
    unsigned int* tree_id    = (unsigned int*) malloc(sizeof(unsigned int) * ncells ) ;
    unsigned long long* qid  = (unsigned long long*) malloc(sizeof(unsigned long long) * ncells ) ;  

    #pragma omp parallel for
    for(size_t idx = 0; idx < octree_slicer.sliced_quads.size(); ++idx) {
        for (size_t ii=0; ii<nx; ii++) {
            for (size_t jj=0; jj<nx; jj++) {
                unsigned long icell = ii + nx * ( jj + nx * idx ) ; 
                unsigned int itree = amr::get_quadrant_owner(octree_slicer.sliced_quads[idx]) ; 
                auto quad  = amr::get_quadrant(octree_slicer.sliced_quads[idx]) ; 
                unsigned int level = quad.level() ;
                size_t iquad_glob = octree_slicer.sliced_quads[idx] + amr::forest::get().global_quadrant_offset(rank_loc) ;

                rank[icell]    = rank_loc   ;
                lev[icell]     = level      ;
                tree_id[icell] = itree      ;
                qid[icell]     = iquad_glob ; 
            }
        }
    }

    /* Create parallel dataset properties */
    hid_t dxpl ; 
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ; 
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ; 
    auto const offset = local_quad_offset * ncells_quad ; 
    write_scalar_dataset_sliced( static_cast<void*>(rank),H5T_NATIVE_UINT,file_id,dxpl
                        , ncells,ncells_glob,offset,chunk_size,compression_level,"/Rank") ; 
    write_scalar_dataset_sliced( static_cast<void*>(lev),H5T_NATIVE_UINT,file_id,dxpl
                        , ncells,ncells_glob,offset,chunk_size,compression_level,"/Level") ; 
    write_scalar_dataset_sliced( static_cast<void*>(tree_id),H5T_NATIVE_UINT,file_id,dxpl
                        , ncells,ncells_glob,offset,chunk_size,compression_level,"/Tree_ID") ; 
    write_scalar_dataset_sliced( static_cast<void*>(qid),H5T_NATIVE_ULLONG,file_id,dxpl
                        , ncells,ncells_glob,offset,chunk_size,compression_level,"/Quad_ID") ;
    
    /* Release resources */
    free(rank) ;
    free(lev) ;
    free(tree_id) ;
    free(qid) ;

    /* Cleanup and exit */
    HDF5_CALL(err, H5Pclose(dxpl)) ;

    
}

void write_scalar_dataset_sliced( void* data, hid_t mem_type_id, hid_t file_id, hid_t dxpl
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
    if ( compression_level > 0 )
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

}} /* namespace grace::IO */
