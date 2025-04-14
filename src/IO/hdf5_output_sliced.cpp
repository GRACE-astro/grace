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
#include <grace/amr/octree_search_class.hh>
#include <grace/config/config_parser.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/IO/hdf5_output_sliced.hh>
#include <grace/utils/numerics/global_interpolators.hh>

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
#include <impl/Kokkos_UnorderedMap_impl.hpp>

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


void write_volume_cell_data_sliced_hdf5() {
    Kokkos::Profiling::pushRegion("HDF5 volume output") ;

    detail::_volume_output_sliced_iterations.push_back(grace::get_iteration())  ; 
    detail::_volume_output_sliced_times.push_back(grace::get_simulation_time()) ;

    auto& runtime = grace::runtime::get() ; 
    std::filesystem::path base_path (runtime.surface_io_basepath()) ;
    std::ostringstream oss;
    oss << runtime.volume_io_basename() << "_plane" << "_"
        << std::setw(6) << std::setfill('0') << grace::get_iteration() << ".h5";
    std::string pfname = oss.str();
    std::filesystem::path out_path = base_path / pfname ;
    detail::_volume_output_sliced_filenames.push_back(pfname) ;

    auto& params = grace::config_parser::get() ;
    size_t compression_level = params["IO"]["hdf5_compression_level"].as<size_t>() ;
    size_t chunk_size = params["IO"]["hdf5_chunk_size"].as<size_t>() ;

    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ;

    /* Get the sliced octants*/
    std::string plane_dir = "yz" ;
    amr::OctreeSlicer octree_slicer(plane_dir);
    octree_slicer.find_sliced_cells();
    octree_slicer.set_localToSlicedIdx();
    auto sliced_cells = octree_slicer.num_sliced_cells();
    auto num_sliced_quadrants = octree_slicer.sliced_quadrants().size();

    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    unsigned long const ncells_quad = ny*nz ; 
    /* Global number of cells  */
    unsigned long const ncells_glob = ncells_quad * num_sliced_quadrants ; 
    if( chunk_size > ncells_glob ) {
        GRACE_WARN("Chunk size {} < number of cells {} will be overridden." , chunk_size, ncells_glob) ; 
        chunk_size = ncells_glob/32.0; 
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
    /* Write xmf file */
    std::string pfname_xdmf = runtime.volume_io_basename() + "_plane"+ ".xmf" ;
    std::filesystem::path out_path_xdmf = base_path / pfname_xdmf ;
    if( parallel::mpi_comm_rank() == 0)
        write_xmf_file( out_path_xdmf.string()
                      , detail::_volume_output_sliced_iterations
                      , detail::_volume_output_sliced_ncells 
                      , detail::_volume_output_sliced_times 
                      , detail::_volume_output_sliced_filenames ) ; 
    
    Kokkos::Profiling::popRegion() ; 
}

void write_grid_structure_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::OctreeSlicer& octree_slicer) {
    herr_t err ; 

    auto& coord_system = grace::coordinate_system::get() ; 
    size_t nx_s,ny_s,nz_s; 
    std::tie(nx_s,ny_s,nz_s) = octree_slicer.get_quadrant_extents() ; 
    int ngz = grace::amr::get_n_ghosts() ;

    auto sliced_cells = octree_slicer.num_sliced_cells();
    auto localToSlicedIdx = octree_slicer.get_localToSlicedIdx();

    size_t nq_s = octree_slicer.sliced_quadrants().size();

    #ifdef GRACE_CARTESIAN_COORDINATES
    size_t constexpr nvertex = 8 ;
    #elif defined(GRACE_SPHERICAL_COORDINATES)
    size_t constexpr nvertex = 24 ;
    #endif 

    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long nq_glob_recv_buf; 
    parallel::mpi_allreduce(&nq_s, &nq_glob_recv_buf, 1, mpi_sum, parallel::get_comm_world()) ;
    unsigned long const nq_glob_sliced = nq_glob_recv_buf ;

    size_t local_quad_offset_recv_buf; // has to be the same type as nq
    parallel::mpi_exscan_sum( &nq_s, &local_quad_offset_recv_buf, 1, parallel::get_comm_world() ) ;
    if (rank == 0) local_quad_offset_recv_buf = 0 ;

    unsigned long const local_quad_offset_sliced = local_quad_offset_recv_buf ; 
    /* Number of cells per quadrant */
    unsigned long const ncells_quad_sliced = EXPR(nx_s,*ny_s,*nz_s) ; 
    /* Local number of cells   */
    unsigned long const ncells_sliced = ncells_quad_sliced * nq_s; //octree_slicer.num_sliced_cells() ; 
    /* Global number of cells  */
    unsigned long const ncells_glob_sliced = ncells_quad_sliced * nq_glob_sliced ; 
    /* Number of unique points per quadrant */
    unsigned long const npoints_quad_sliced = (nx_s+1) * (ny_s+1) * (nz_s+1); 
    /* Local number of points  */
    unsigned long const npoints_sliced = npoints_quad_sliced * nq_s ;  
    /* Global number of points */
    unsigned long const npoints_glob_sliced = npoints_quad_sliced * nq_glob_sliced ;

    /* Not sure what I need from here So is hust get all*/
    size_t nx,ny,nz; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    size_t nq = grace::amr::get_local_num_quadrants() ; 
    unsigned long const nq_glob = _p4est->global_num_quadrants ; 
    unsigned long const local_quad_offset = _p4est->global_first_quadrant[rank] ; 
    unsigned long const ncells_quad = EXPR(nx,*ny,*nz) ; 
    unsigned long const ncells = ncells_quad * nq ; 
    unsigned long const ncells_glob = ncells_quad * nq_glob ; 
    unsigned long const npoints_quad = (nx+1) * (ny+1) * (nz+1) ; 
    unsigned long const npoints = npoints_quad * nq ;  
    unsigned long const npoints_glob = npoints_quad * nq_glob ;

    detail::_volume_output_sliced_ncells.push_back(ncells_glob_sliced) ; 

    double*  points = (double*)  malloc(sizeof(double)  * npoints_sliced * 3 ) ; 
    unsigned int* cells  = (unsigned int*) malloc(sizeof(unsigned int) * ncells_sliced * nvertex ) ; 
    const size_t global_point_offset_sliced = local_quad_offset_sliced * npoints_quad_sliced ;  
    unsigned int icell  = 0L ; 
    unsigned int ipoint = 0U ; 
    auto const get_point_index = 
    [&] 
    (
        VEC(int i, int j, int k), int64_t q
    ) 
    {
        return i + (nx_s+1) * (j + (ny_s+1) * (k + (nz_s+1) * q)) ; 
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

    #pragma omp parallel for
    for(auto sliced_cell : octree_slicer.sliced_cells()) {
        auto const& q = sliced_cell.q ;
        auto i = sliced_cell.i ;
        auto const& j = sliced_cell.j ;
        auto const& k = sliced_cell.k ;
        auto const qlocal = q.localQuadrantIdx ; 
        auto const qsliced = localToSlicedIdx.at(qlocal) ;
        // nq = number of quadrants
        auto const nq_p4est = grace::amr::get_local_num_quadrants() ;
        unsigned long icell = 0 + nx_s * ( j + ny_s * ( k  + nz_s * qsliced )) ; 
        for( int iv=0; iv<nvertex; ++iv ) {
            int ip, jp, kp ; 
            std::tie(ip,jp,kp) = get_cell_vertex_indices(VEC(i,j,k),qsliced,iv) ; 
            cells[nvertex * icell + iv] = get_point_index(VEC(ip,jp,kp),qsliced+local_quad_offset_sliced); 
        }
    }
    #pragma omp parallel for
    for(auto sliced_cell : octree_slicer.sliced_cells()) {
        auto const& q = sliced_cell.q ;
        auto const& i = sliced_cell.i ;
        auto const& j = sliced_cell.j ;
        auto const& k = sliced_cell.k ;
        auto const& qglobal = q.globalIndex ;
        auto const qlocal = q.localQuadrantIdx ; 
        auto const qsliced = localToSlicedIdx.at(qlocal) ;

            auto pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i,j,k)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            Kokkos::printf("pcoords rank %d %d %f %f %f\n", rank, qlocal, pcoords[0], pcoords[1], pcoords[2]) ;
            auto ipoint = get_point_index(VEC(0,j,k),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;
            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i+1,j,k)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(1,j,k),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;
        if (((j+1) < ny_s ) and ((k+1) == nz_s )) {
            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i,j,k+1)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(0,j,k+1),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;
            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i+1,j,k+1)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(1,j,k+1),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;
        }
        else if (((j+1) == ny_s ) and ((k+1) < nz_s )) {
            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i,j+1,k)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(0,j+1,k),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;
            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i+1,j+1,k)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(1,j+1,k),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;
        }
        else if (((j+1) == ny_s ) and ((k+1) == nz_s )) {
            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i,j+1,k+1)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(0,j+1,k+1),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;
            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i+1,j+1,k+1)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(1,j+1,k+1),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;

            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i,j,k+1)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(0,j,k+1),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;
            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i+1,j, k+1)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(1,j,k+1),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;

            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i,j+1,k)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(0,j+1,k),qsliced) ;
            points[GRACE_NSPACEDIM*ipoint + 0 ] = pcoords[0] ; 
            points[GRACE_NSPACEDIM*ipoint + 1 ] = pcoords[1] ;
            points[GRACE_NSPACEDIM*ipoint + 2 ] = pcoords[2] ;
            pcoords = 
                            coord_system.get_physical_coordinates( {VEC(i+1,j+1,k)}
                                                                 , qglobal
                                                                 , {VEC( 0,0,0 )}
                                                                 , false) ; 
            ipoint = get_point_index(VEC(1,j+1,k),qsliced) ;
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
    hsize_t cells_dset_dims_glob[2] = {ncells_glob_sliced, nvertex} ;   
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
    hsize_t cells_dset_dims[2] = {ncells_sliced, nvertex} ;
    HDF5_CALL(cells_space_id, H5Screate_simple(2, cells_dset_dims, NULL)) ; 
    /* Select hyperslab for this rank's output */
    hsize_t cells_slab_start[2]  = {local_quad_offset_sliced * ncells_quad_sliced,0} ; 
    hsize_t cells_slab_count[2]  = {ncells_sliced,nvertex} ;
    GRACE_VERBOSE("Slab offset {}, size {}, total {}", cells_slab_start[0], ncells_sliced, ncells_glob_sliced) ;  
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

void write_volume_data_arrays_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::OctreeSlicer& octree_slicer) {

    herr_t err ;

    auto& runtime = grace::runtime::get() ;

    auto const scalars     = runtime.cell_volume_output_scalar_vars() ; 
    auto const sscalars     = runtime.corner_volume_output_scalar_vars() ; 
    auto const vectors     = runtime.cell_volume_output_vector_vars() ; 
    auto const svectors     = runtime.corner_volume_output_vector_vars() ;
    auto const tensors     = runtime.cell_volume_output_symm_tensor_vars() ;
    auto const stensors     = runtime.corner_volume_output_symm_tensor_vars() ;

    size_t nx_s,ny_s,nz_s,nq_s; 
    std::tie(nx_s,ny_s,nz_s) = octree_slicer.get_quadrant_extents() ; 
    nq_s = octree_slicer.sliced_quadrants().size() ; 
    size_t ngz = grace::amr::get_n_ghosts() ; 

    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */

    unsigned long nq_glob_recv_buf; 
    parallel::mpi_allreduce(&nq_s, &nq_glob_recv_buf, 1, mpi_sum, parallel::get_comm_world()) ;
    unsigned long const nq_glob_sliced = nq_glob_recv_buf ;

    size_t local_quad_offset_recv_buf; // has to be the same type as nq
    parallel::mpi_exscan_sum( &nq_s, &local_quad_offset_recv_buf, 1, parallel::get_comm_world() ) ;
    if (rank == 0) local_quad_offset_recv_buf = 0 ;

    unsigned long const local_quad_offset_sliced = local_quad_offset_recv_buf ; 
    /* Number of cells per quadrant */
    unsigned long const ncells_quad_sliced = EXPR(nx_s,*ny_s,*nz_s) ; 
    /* Local number of cells   */
    unsigned long const ncells_sliced = ncells_quad_sliced * nq_s; //octree_slicer.num_sliced_cells() ; 
    /* Global number of cells  */
    unsigned long const ncells_glob_sliced = ncells_quad_sliced * nq_glob_sliced ; 
    /* Number of unique points per quadrant */
    unsigned long const ncorners_quad_sliced = (nx_s+1) * (ny_s+1) * (nz_s+1); 
    /* Local number of points  */
    unsigned long const ncorners = ncorners_quad_sliced * nq_s ;  
    /* Global number of points */
    unsigned long const ncorners_glob_sliced = ncorners_quad_sliced * nq_glob_sliced ;


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
    HDF5_CALL(err, H5Pset_deflate(scalars_prop_id, compression_level)) ;  

    /* Create local space for this rank */
    hid_t scalars_space_id ; 
    hsize_t scalars_dset_dims[1] = {ncells_sliced} ;
    HDF5_CALL(scalars_space_id, H5Screate_simple(1, scalars_dset_dims, NULL)) ; 

    /* 2) Corner staggered variables */
    hid_t sscalars_space_id_glob ;
    hsize_t sscalars_dset_dims_glob[1] = {ncorners_glob_sliced} ;

    /* Create global space for points dataset */
    HDF5_CALL(sscalars_space_id_glob, H5Screate_simple(1, sscalars_dset_dims_glob, NULL)) ;

    /* Create dataset properties and set chunking / compression mode */
    hid_t sscalars_prop_id ;
    HDF5_CALL(sscalars_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t sscalars_chunk_dim[1] = {chunk_size} ;
    HDF5_CALL(err, H5Pset_chunk(sscalars_prop_id,1,sscalars_chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(sscalars_prop_id, compression_level)) ;  
    
    /* Create local space for this rank */
    hid_t sscalars_space_id ; 
    hsize_t sscalars_dset_dims[1] = {ncorners} ;
    HDF5_CALL(sscalars_space_id, H5Screate_simple(1, sscalars_dset_dims, NULL)) ; 
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                                  Write to file                                        */
    /*****************************************************************************************/
    write_var_arrays_sliced_hdf5( 
        scalars, sscalars, 
        file_id, dxpl,
        scalars_space_id_glob, scalars_space_id, scalars_prop_id,
        sscalars_space_id_glob, sscalars_space_id, sscalars_prop_id,
        ncells_sliced, local_quad_offset_sliced, octree_slicer) ; 
    /*****************************************************************************************/
    /*                                  Close data spaces                                    */
    /*****************************************************************************************/
    HDF5_CALL(err, H5Sclose(scalars_space_id)) ; 
    HDF5_CALL(err, H5Sclose(scalars_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(scalars_prop_id)) ;
    HDF5_CALL(err, H5Sclose(sscalars_space_id)) ; 
    HDF5_CALL(err, H5Sclose(sscalars_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(sscalars_prop_id)) ;
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
    HDF5_CALL(err, H5Pset_deflate(vectors_prop_id, compression_level)) ;  

    /* Create local space for this rank */
    hid_t vectors_space_id ; 
    hsize_t vectors_dset_dims[2] = {ncells_sliced, 3} ;
    HDF5_CALL(vectors_space_id, H5Screate_simple(2, vectors_dset_dims, NULL)) ; 

    /* 2) Corner staggered variables */
    hid_t svectors_space_id_glob ;
    hsize_t svectors_dset_dims_glob[2] = {ncorners_glob_sliced, 3} ;
    
    /* Create global space for points dataset */
    HDF5_CALL(svectors_space_id_glob, H5Screate_simple(2, svectors_dset_dims_glob, NULL)) ;

    /* Create dataset properties and set chunking / compression mode */
    hid_t svectors_prop_id ;
    HDF5_CALL(svectors_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t svectors_chunk_dim[2] = {chunk_size, 3} ;
    HDF5_CALL(err, H5Pset_chunk(svectors_prop_id,2,svectors_chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(svectors_prop_id, compression_level)) ;  

    /* Create local space for this rank */
    hid_t svectors_space_id ; 
    hsize_t svectors_dset_dims[2] = {ncorners, 3} ;
    HDF5_CALL(svectors_space_id, H5Screate_simple(2, svectors_dset_dims, NULL)) ; 
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                                  Write to file                                        */
    /*****************************************************************************************/
    write_vector_var_arrays_sliced_hdf5( 
        vectors,svectors,
        file_id,dxpl,
        vectors_space_id_glob,vectors_space_id,vectors_prop_id,
        svectors_space_id_glob,svectors_space_id,svectors_prop_id,
        ncells_sliced,local_quad_offset_sliced, octree_slicer) ; 
    /*****************************************************************************************/
    /*                                  Close data spaces                                    */
    /*****************************************************************************************/
    HDF5_CALL(err, H5Sclose(vectors_space_id)) ; 
    HDF5_CALL(err, H5Sclose(vectors_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(vectors_prop_id)) ;
    HDF5_CALL(err, H5Sclose(svectors_space_id)) ; 
    HDF5_CALL(err, H5Sclose(svectors_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(svectors_prop_id)) ;
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                                     Tensors                                           */
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                               Create/open datasets                                    */
    /*****************************************************************************************/

    /* 1) Cell centered variables */
    hid_t tensors_space_id_glob ;
    hsize_t tensors_dset_dims_glob[2] = {ncells_glob_sliced, 6} ;

    /* Create global space for points dataset */
    HDF5_CALL(tensors_space_id_glob, H5Screate_simple(2, tensors_dset_dims_glob, NULL)) ;

    /* Create dataset properties and set chunking / compression mode */
    hid_t tensors_prop_id ;
    HDF5_CALL(tensors_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t tensors_chunk_dim[2] = {chunk_size, 3} ;
    HDF5_CALL(err, H5Pset_chunk(tensors_prop_id,2,tensors_chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(tensors_prop_id, compression_level)) ;  

    /* Create local space for this rank */
    hid_t tensors_space_id ; 
    hsize_t tensors_dset_dims[2] = {ncells_sliced, 6} ;
    HDF5_CALL(tensors_space_id, H5Screate_simple(2, tensors_dset_dims, NULL)) ; 

    /* 2) Corner staggered variables */
    hid_t stensors_space_id_glob ;
    hsize_t stensors_dset_dims_glob[2] = {ncorners_glob_sliced, 6} ;

    /* Create global space for points dataset */
    HDF5_CALL(stensors_space_id_glob, H5Screate_simple(2, stensors_dset_dims_glob, NULL)) ;

    /* Create dataset properties and set chunking / compression mode */
    hid_t stensors_prop_id ;
    HDF5_CALL(stensors_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t stensors_chunk_dim[2] = {chunk_size, 3} ;
    HDF5_CALL(err, H5Pset_chunk(stensors_prop_id,2,stensors_chunk_dim)) ; 
    HDF5_CALL(err, H5Pset_deflate(stensors_prop_id, compression_level)) ;  

    /* Create local space for this rank */
    hid_t stensors_space_id ; 
    hsize_t stensors_dset_dims[2] = {ncorners, 6} ;
    HDF5_CALL(stensors_space_id, H5Screate_simple(2, stensors_dset_dims, NULL)) ;
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                                  Write to file                                        */
    /*****************************************************************************************/
    write_tensor_var_arrays_sliced_hdf5( 
        tensors,stensors,
        file_id,dxpl,
        tensors_space_id_glob,tensors_space_id,tensors_prop_id,
        stensors_space_id_glob,stensors_space_id,stensors_prop_id,
        ncells_sliced,local_quad_offset_sliced, octree_slicer) ; 
    /*****************************************************************************************/
    /*                                  Close data spaces                                    */
    /*****************************************************************************************/
    HDF5_CALL(err, H5Sclose(tensors_space_id)) ; 
    HDF5_CALL(err, H5Sclose(tensors_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(tensors_prop_id)) ;
    HDF5_CALL(err, H5Sclose(stensors_space_id)) ; 
    HDF5_CALL(err, H5Sclose(stensors_space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(stensors_prop_id)) ;
    /*****************************************************************************************/
    /*                                 Cleanup and exit                                      */
    /*****************************************************************************************/
    HDF5_CALL(err, H5Pclose(dxpl)) ;
    /*****************************************************************************************/
}

void write_var_arrays_sliced_hdf5( std::set<std::string> const& varlist 
                          , std::set<std::string> const& svarlist 
                          , hid_t file_id 
                          , hid_t dxpl
                          , hid_t space_id_glob
                          , hid_t space_id
                          , hid_t prop_id
                          , hid_t sspace_id_glob
                          , hid_t sspace_id
                          , hid_t sprop_id
                          , hsize_t ncells
                          , hsize_t local_quad_offset
                          , amr::OctreeSlicer& octree_slicer) // had isaux 
{
    herr_t err ; 
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq, nx_s,ny_s,nz_s,nq_s;
    std::tie(nx_s,ny_s,nz_s) = octree_slicer.get_quadrant_extents() ; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    nq_s = octree_slicer.sliced_quadrants().size() ; 
    nq = grace::amr::get_local_num_quadrants() ;
    size_t ngz = grace::amr::get_n_ghosts() ;
    /* Number of cells per quadrant */
    unsigned long const ncells_quad_sliced = EXPR(nx_s,*ny_s,*nz_s) ; 
    /* Number of corners per quadrant */
    unsigned long const ncorners_quad_sliced = EXPR((nx_s+1),*(ny_s+1),*(nz_s+1)) ;

    //-----------
    //amr::OctreeSlicer octree_slicer;
    //octree_slicer.find_sliced_cells();
    auto sliced_cells = octree_slicer.sliced_cells();
    auto sliced_quadrants = octree_slicer.sliced_quadrants();
    auto localToSlicedIdx = octree_slicer.get_localToSlicedIdx();

    using key_type = size_t;
    using value_type = size_t;
    using device_type = grace::default_execution_space;

    using map_type = Kokkos::UnorderedMap<key_type, value_type, device_type>;

    // Create host mirrors for keys and values.
    // Allocate the views in the default execution space (likely device memory)
    Kokkos::View<key_type*> d_h_keys("d_h_keys", nq_s);
    Kokkos::View<value_type*> d_h_values("d_h_values", nq_s);

    // Create a host mirror which is allocatable in host memory.
    auto h_keys = Kokkos::create_mirror_view(d_h_keys);
    auto h_values = Kokkos::create_mirror_view(d_h_values);

    // Fill the host mirror.
    size_t idx = 0;
    for (const auto& kv : localToSlicedIdx) {
      h_keys(idx)   = kv.first;
      h_values(idx) = kv.second;
      ++idx;
    }

    // Deep copy back to the device view.
    Kokkos::deep_copy(d_h_keys, h_keys);
    Kokkos::deep_copy(d_h_values, h_values);

    auto d_keys = d_h_keys;
    auto d_values = d_h_values;

    // Create the device unordered map with an initial capacity.
    map_type kokkosMap;
    kokkosMap.rehash(nq_s);
    // Parallel insertion into the device map.
    Kokkos::parallel_for("InsertIntoMap", Kokkos::RangePolicy<grace::default_execution_space>(0, nq_s),
      KOKKOS_LAMBDA(const int i) {
        // Note: insert returns the location or a status value.
        // You may also want to check for failure if your map becomes full.
        kokkosMap.insert(d_keys(i), d_values(i));
    });

    Kokkos::View<amr::OctreeSlicer::SlicedCellInfo*> d_sliced_cells("d_sliced_cells", sliced_cells.size());
    // Create a mirror on the host.
    auto h_sliced_cells = Kokkos::create_mirror_view(d_sliced_cells);
    // Copy data from your host container into the mirror.
    for (size_t i = 0; i < sliced_cells.size(); ++i) {
      h_sliced_cells(i) = sliced_cells[i];
    }

    // Deep copy to device.
    Kokkos::deep_copy(d_sliced_cells, h_sliced_cells);
    Kokkos::fence();
    //-----------
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
    Kokkos::View<double EXPR(*,*,*)*, Kokkos::LayoutLeft>
        d_temp_sliced("Device output mirror", VEC(nx_s,ny_s,nz_s), nq_s) ;
    auto h_mirror_sliced = Kokkos::create_mirror_view(d_temp_sliced) ;
    for( auto const& vname: varlist )
    {
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
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableType", "Scalar");
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableStaggering", "Cell");

        /* Shuffle data around to put it in the right form */
        int err; 
        auto props = variables::get_variable_properties(vname,err) ;
        ASSERT( err == 0, "Error retrieving variable properties for variable" << vname) ; 
        ASSERT( props.staggering == var_staggering_t::CELL_CENTER,
            "Something went wrong, corner staggered variable marked as cell-center for output." ) ;  

        auto sview = get_variable_subview(
                      vname
                    , Kokkos::pair<int,int>(ngz,nx+ngz)
                    , Kokkos::pair<int,int>(ngz,ny+ngz)
                    #ifdef GRACE_3D
                    , Kokkos::pair<int,int>(ngz,nz+ngz)
                    #endif 
                    , Kokkos::ALL()
                ) ;

        Kokkos::parallel_for("pack_quadrants", Kokkos::RangePolicy<>(0, d_sliced_cells.size()), KOKKOS_LAMBDA(const int it) {
            auto& sliced_cell = d_sliced_cells[it];
            auto const& q = sliced_cell.q ;
            auto const q_local = q.localQuadrantIdx ;
            auto const& i = sliced_cell.i ;
            auto const& j = sliced_cell.j ;
            auto const& k = sliced_cell.k ;
            auto sliced_q = kokkosMap.find(q_local);
            sliced_q = kokkosMap.value_at(sliced_q);
            d_temp_sliced(0, j, k, sliced_q) = sview(i, j, k, q_local);
        });
        /* Copy data d2h */
        Kokkos::fence();

        Kokkos::deep_copy(grace::default_execution_space{},h_mirror_sliced,d_temp_sliced) ;
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
                            , reinterpret_cast<void*>(h_mirror_sliced.data()) )) ;

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }

    /* Staggered variables */

    // Resize staging buffers 
    Kokkos::realloc(d_mirror,VEC(nx+1,ny+1,nz+1), nq) ; 
    Kokkos::realloc(d_temp_sliced,VEC(nx_s+1,ny_s+1,nz_s+1), nq_s) ;
    h_mirror_sliced = Kokkos::create_mirror_view(d_temp_sliced) ; 

    // Loop over variables 
    for( auto const& vname: svarlist )
    {
        GRACE_TRACE("Writing var {} to output.", vname) ; 
        /* create HDF5 dataset */
        std::string dset_name = "/" + vname ; 
        hid_t dset_id ; 
        HDF5_CALL( dset_id
                , H5Dcreate2( file_id
                            , dset_name.c_str()
                            , H5T_NATIVE_DOUBLE
                            , sspace_id_glob
                            , H5P_DEFAULT
                            , sprop_id
                            , H5P_DEFAULT) ) ;

        /* Write attributes to dataset */
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableType", "Scalar");
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableStaggering", "Node");

        /* Shuffle data around to put it in the right form */
        int err; 
        auto props = variables::get_variable_properties(vname,err) ;
        ASSERT( err == 0, "Error retrieving variable properties for variable" << vname) ; 
        ASSERT( props.staggering == var_staggering_t::CORNER,
            "Only corner staggered variables are supported for staggered variables output.") ;  

        auto sview = get_variable_subview(
                      vname
                    , Kokkos::pair<int,int>(ngz,nx+1+ngz)
                    , Kokkos::pair<int,int>(ngz,ny+1+ngz)
                    #ifdef GRACE_3D
                    , Kokkos::pair<int,int>(ngz,nz+1+ngz)
                    #endif 
                    , Kokkos::ALL()
                ) ;
        
        Kokkos::parallel_for("pack_quadrants", Kokkos::RangePolicy<>(0, d_sliced_cells.size()), KOKKOS_LAMBDA(const int it) {
            auto& sliced_cell = d_sliced_cells[it];
            auto const& q = sliced_cell.q ;
            auto const& q_local = q.localQuadrantIdx ;
            auto const& i = sliced_cell.i ;
            auto const& j = sliced_cell.j ;
            auto const& k = sliced_cell.k ;
            auto sliced_q = kokkosMap.find(q_local);
            sliced_q = kokkosMap.value_at(sliced_q);
            d_temp_sliced(0, j, k, sliced_q) = sview(i, j, k, q_local);
            d_temp_sliced(1, j, k, sliced_q) = sview(i+1, j, k, q_local);
            if (((j+1) < ny_s ) and ((k+1) == nz_s )) {
            d_temp_sliced(0, j, k+1, sliced_q) = sview(i, j, k+1, q_local);
            d_temp_sliced(1, j, k+1, sliced_q) = sview(i+1, j, k+1, q_local);
            }
            else if (((j+1) == ny_s ) and ((k+1) < nz_s )) {
            d_temp_sliced(0, j+1, k, sliced_q) = sview(i, j+1, k, q_local);
            d_temp_sliced(1, j+1, k, sliced_q) = sview(i+1, j+1, k, q_local);
            }
            else if (((j+1) == ny_s ) and ((k+1) == nz_s )) {
            d_temp_sliced(0, j+1, k+1, sliced_q) = sview(i, j+1, k+1, q_local);
            d_temp_sliced(1, j+1, k+1, sliced_q) = sview(i+1, j+1, k+1, q_local);
            d_temp_sliced(0, j, k+1, sliced_q) = sview(i, j, k+1, q_local);
            d_temp_sliced(1, j, k+1, sliced_q) = sview(i+1, j, k+1, q_local);
            d_temp_sliced(0, j+1, k, sliced_q) = sview(i, j+1, k, q_local);
            d_temp_sliced(1, j+1, k, sliced_q) = sview(i+1, j+1, k, q_local);
            }
        });
        /* Copy data d2h */
        Kokkos::fence() ;
        Kokkos::deep_copy(grace::default_execution_space{},h_mirror_sliced,d_temp_sliced) ; 
        Kokkos::fence() ;

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[1]  = {local_quad_offset * ncorners_quad_sliced} ; 
        hsize_t slab_count[1]  = {ncorners_quad_sliced * nq_s} ;
        HDF5_CALL( err
                , H5Sselect_hyperslab( sspace_id_glob
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
                            , sspace_id
                            , sspace_id_glob 
                            , dxpl 
                            , reinterpret_cast<void*>(h_mirror_sliced.data()) )) ;

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }
}

void write_vector_var_arrays_sliced_hdf5( std::set<std::string> const& varlist 
                                 , std::set<std::string> const& svarlist 
                                 , hid_t file_id 
                                 , hid_t dxpl
                                 , hid_t space_id_glob
                                 , hid_t space_id
                                 , hid_t prop_id
                                 , hid_t sspace_id_glob
                                 , hid_t sspace_id
                                 , hid_t sprop_id
                                 , hsize_t ncells
                                 , hsize_t local_quad_offset 
                                 , amr::OctreeSlicer& octree_slicer) // had isaux  
{
    using namespace grace; 
    using namespace Kokkos; 
    herr_t err ;
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq, nx_s,ny_s,nz_s,nq_s;
    std::tie(nx_s,ny_s,nz_s) = octree_slicer.get_quadrant_extents() ; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    nq_s = octree_slicer.sliced_quadrants().size() ; 
    nq = grace::amr::get_local_num_quadrants() ;
    size_t ngz = grace::amr::get_n_ghosts() ;
    /* Number of cells per quadrant */
    unsigned long const ncells_quad_sliced = EXPR(nx_s,*ny_s,*nz_s) ; 
    /* Number of corners per quadrant */
    unsigned long const ncorners_quad_sliced = EXPR((nx_s+1),*(ny_s+1),*(nz_s+1)) ;

    auto sliced_cells = octree_slicer.sliced_cells();
    auto sliced_quadrants = octree_slicer.sliced_quadrants();
    auto localToSlicedIdx = octree_slicer.get_localToSlicedIdx();

    using key_type = size_t;
    using value_type = size_t;
    using device_type = grace::default_execution_space;

    using map_type = Kokkos::UnorderedMap<key_type, value_type, device_type>;

    // Create host mirrors for keys and values.
    // Allocate the views in the default execution space (likely device memory)
    Kokkos::View<key_type*> d_h_keys("d_h_keys", nq_s);
    Kokkos::View<value_type*> d_h_values("d_h_values", nq_s);

    // Create a host mirror which is allocatable in host memory.
    auto h_keys = Kokkos::create_mirror_view(d_h_keys);
    auto h_values = Kokkos::create_mirror_view(d_h_values);

    // Fill the host mirror.
    size_t idx = 0;
    for (const auto& kv : localToSlicedIdx) {
      h_keys(idx)   = kv.first;
      h_values(idx) = kv.second;
      ++idx;
    }

    // Deep copy back to the device view.
    Kokkos::deep_copy(d_h_keys, h_keys);
    Kokkos::deep_copy(d_h_values, h_values);

    auto d_keys = d_h_keys;
    auto d_values = d_h_values;

    // Create the device unordered map with an initial capacity.
    map_type kokkosMap;
    kokkosMap.rehash(nq_s);
    // Parallel insertion into the device map.
    Kokkos::parallel_for("InsertIntoMap", Kokkos::RangePolicy<grace::default_execution_space>(0, nq_s),
      KOKKOS_LAMBDA(const int i) {
        // Note: insert returns the location or a status value.
        // You may also want to check for failure if your map becomes full.
        //kokkosMap.insert(d_keys(i), d_values(i));
        auto result = kokkosMap.insert(d_keys(i), d_values(i));
    });

    Kokkos::View<amr::OctreeSlicer::SlicedCellInfo*> d_sliced_cells("d_sliced_cells", sliced_cells.size());
    // Create a mirror on the host.
    auto h_sliced_cells = Kokkos::create_mirror_view(d_sliced_cells);
    // Copy data from your host container into the mirror.
    for (size_t i = 0; i < sliced_cells.size(); ++i) {
      h_sliced_cells(i) = sliced_cells[i];
    }

    // Deep copy to device.
    Kokkos::deep_copy(d_sliced_cells, h_sliced_cells);
    Kokkos::fence();
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
    Kokkos::View<double *EXPR(*,*,*)*, Kokkos::LayoutLeft>
        d_temp_sliced("Device output mirror", 3, VEC(nx_s,ny_s,nz_s), nq_s) ;
    auto h_mirror_sliced = Kokkos::create_mirror_view(d_temp_sliced) ;
    for( auto const& vname: varlist )
    {
        GRACE_TRACE("Writing vector var {} to output.") ; 
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
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableType", "Vector");
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableStaggering", "Cell");
        
        std::array<std::string, 3> const compnames 
                = {
                    vname + "[0]",
                    vname + "[1]",
                    vname + "[2]"
                } ; 
        int err ; 
        auto props = variables::get_variable_properties(compnames[0], err) ;
        ASSERT( err == 0, "Error retrieving variable properties for variable" << vname) ; 
        ASSERT( props.staggering == var_staggering_t::CELL_CENTER,
                "Something went wrong, corner staggered variable marked as cell-center for output." ) ;  
        bool var_is_staggered = props.staggering == var_staggering_t::CORNER ;
        for( int icomp=0; icomp<3; ++icomp){
            auto sview = get_variable_subview(
                      compnames[icomp]
                    , Kokkos::pair<int,int>(ngz,nx+ngz)
                    , Kokkos::pair<int,int>(ngz,ny+ngz)
                    #ifdef GRACE_3D
                    , Kokkos::pair<int,int>(ngz,nz+ngz)
                    #endif 
                    , Kokkos::ALL()
                ) ;


            Kokkos::parallel_for("pack_quadrants", Kokkos::RangePolicy<>(0, d_sliced_cells.size()), KOKKOS_LAMBDA(const int it) {
                auto& sliced_cell = d_sliced_cells[it];
                auto const& q = sliced_cell.q ;
                auto const& q_local = q.localQuadrantIdx ;
                auto const& i = sliced_cell.i ;
                auto const& j = sliced_cell.j ;
                auto const& k = sliced_cell.k ;
                auto sliced_q = kokkosMap.find(q_local);
                sliced_q = kokkosMap.value_at(sliced_q);
                d_temp_sliced(icomp, 0, j, k, sliced_q) = sview(i, j, k, q_local);
            });
        }
        /* Copy data d2h */
        Kokkos::fence() ;
        Kokkos::deep_copy(grace::default_execution_space{},h_mirror_sliced,d_temp_sliced) ; 
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
                            , reinterpret_cast<void*>(h_mirror_sliced.data()) )) ;
        

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }
    /* Staggered variables */
    // Resize staging buffers
    Kokkos::realloc(d_mirror,3,VEC(nx+1,ny+1,nz+1), nq) ; 
    Kokkos::realloc(d_temp_sliced,3,VEC(nx_s+1,ny_s+1,nz_s+1), nq_s) ;
    h_mirror_sliced = Kokkos::create_mirror_view(d_temp_sliced) ; 
    // Loop over variables
    for( auto const& vname: svarlist )
    {
        GRACE_TRACE("Writing vector var {} to output.") ; 
        /* create HDF5 dataset */
        std::string dset_name = "/" + vname ; 
        hid_t dset_id ; 
        HDF5_CALL( dset_id
                , H5Dcreate2( file_id
                            , dset_name.c_str()
                            , H5T_NATIVE_DOUBLE
                            , sspace_id_glob
                            , H5P_DEFAULT
                            , sprop_id
                            , H5P_DEFAULT) ) ;
        /* Write dataset attributes */
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableType", "Vector");
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableStaggering", "Node");
        
        std::array<std::string, 3> const compnames 
                = {
                    vname + "[0]",
                    vname + "[1]",
                    vname + "[2]"
                } ; 
        int err ; 
        auto props = variables::get_variable_properties(compnames[0], err) ;
        ASSERT( err == 0, "Error retrieving variable properties for variable" << vname) ; 
        ASSERT( props.staggering == var_staggering_t::CORNER,
                "Only corner staggered variables are supported for staggered variables output." ) ;  
        for( int icomp=0; icomp<3; ++icomp){
            auto sview = get_variable_subview(
                      compnames[icomp]
                    , Kokkos::pair<int,int>(ngz,nx+1+ngz)
                    , Kokkos::pair<int,int>(ngz,ny+1+ngz)
                    #ifdef GRACE_3D
                    , Kokkos::pair<int,int>(ngz,nz+1+ngz)
                    #endif 
                    , Kokkos::ALL()
                ) ;

            //Stupid GPU code, do not know how to do this in parallel really
            Kokkos::parallel_for("pack_quadrants", Kokkos::RangePolicy<>(0, d_sliced_cells.size()), KOKKOS_LAMBDA(const int it) {
                auto& sliced_cell = d_sliced_cells[it];
                auto const& q = sliced_cell.q ;
                auto const& q_local = q.localQuadrantIdx ;
                auto const& i = sliced_cell.i ;
                auto const& j = sliced_cell.j ;
                auto const& k = sliced_cell.k ;
                auto sliced_q = kokkosMap.find(q_local);
                sliced_q = kokkosMap.value_at(sliced_q);
                if (((j+1) < ny ) and ((k+1) < nz )) {
                d_temp_sliced(icomp, 0, j, k, sliced_q) = sview(i, j, k, q_local);
                d_temp_sliced(icomp, 1, j, k, sliced_q) = sview(i+1, j, k, q_local);
                }
                else if (((j+1) < ny ) and ((k+1) == nz )) {
                d_temp_sliced(icomp, 0, j, k+1, sliced_q) = sview(i, j, k+1, q_local);
                d_temp_sliced(icomp, 1, j, k+1, sliced_q) = sview(i+1, j, k+1, q_local);
                }
                else if (((j+1) == ny ) and ((k+1) < nz )) {
                d_temp_sliced(icomp, 0, j+1, k, sliced_q) = sview(i, j+1, k, q_local);
                d_temp_sliced(icomp, 1, j+1, k, sliced_q) = sview(i+1, j+1, k, q_local);
                }
                else if (((j+1) == ny ) and ((k+1) == nz )) {
                d_temp_sliced(icomp, 0, j+1, k+1, sliced_q) = sview(i, j+1, k+1, q_local);
                d_temp_sliced(icomp, 1, j+1, k+1, sliced_q) = sview(i+1, j+1, k+1, q_local);
                d_temp_sliced(icomp, 0, j, k+1, sliced_q) = sview(i, j, k+1, q_local);
                d_temp_sliced(icomp, 1, j, k+1, sliced_q) = sview(i+1, j, k+1, q_local);
                d_temp_sliced(icomp, 0, j+1, k, sliced_q) = sview(i, j+1, k, q_local);
                d_temp_sliced(icomp, 1, j+1, k, sliced_q) = sview(i+1, j+1, k, q_local);
                }
            });
        }
        /* Copy data d2h */
        Kokkos::deep_copy(grace::default_execution_space{},h_mirror_sliced,d_temp_sliced) ; 

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[2]  = {local_quad_offset * ncorners_quad_sliced, 0} ; 
        hsize_t slab_count[2]  = {ncorners_quad_sliced * nq_s, 3} ;
        HDF5_CALL( err
                , H5Sselect_hyperslab( sspace_id_glob
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
                            , sspace_id
                            , sspace_id_glob 
                            , dxpl 
                            , reinterpret_cast<void*>(h_mirror_sliced.data()) )) ;
        

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }
}

void write_tensor_var_arrays_sliced_hdf5( std::set<std::string> const& varlist 
                                 , std::set<std::string> const& svarlist 
                                 , hid_t file_id 
                                 , hid_t dxpl
                                 , hid_t space_id_glob
                                 , hid_t space_id
                                 , hid_t prop_id
                                 , hid_t sspace_id_glob
                                 , hid_t sspace_id
                                 , hid_t sprop_id
                                 , hsize_t ncells
                                 , hsize_t local_quad_offset 
                                 , amr::OctreeSlicer& octree_slicer) 
{
    using namespace grace; 
    using namespace Kokkos; 
    herr_t err ;
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq, nx_s,ny_s,nz_s,nq_s;
    std::tie(nx_s,ny_s,nz_s) = octree_slicer.get_quadrant_extents() ; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    nq_s = octree_slicer.sliced_quadrants().size() ; 
    nq = grace::amr::get_local_num_quadrants() ;
    size_t ngz = grace::amr::get_n_ghosts() ;
    /* Number of cells per quadrant */
    unsigned long const ncells_quad_sliced = EXPR(nx_s,*ny_s,*nz_s) ; 
    /* Number of corners per quadrant */
    unsigned long const ncorners_quad_sliced = EXPR((nx_s+1),*(ny_s+1),*(nz_s+1)) ;

    auto sliced_cells = octree_slicer.sliced_cells();
    auto sliced_quadrants = octree_slicer.sliced_quadrants();
    auto localToSlicedIdx = octree_slicer.get_localToSlicedIdx();

    using key_type = size_t;
    using value_type = size_t;
    using device_type = grace::default_execution_space;

    using map_type = Kokkos::UnorderedMap<key_type, value_type, device_type>;

    // Create host mirrors for keys and values.
    // Allocate the views in the default execution space (likely device memory)
    Kokkos::View<key_type*> d_h_keys("d_h_keys", nq_s);
    Kokkos::View<value_type*> d_h_values("d_h_values", nq_s);

    // Create a host mirror which is allocatable in host memory.
    auto h_keys = Kokkos::create_mirror_view(d_h_keys);
    auto h_values = Kokkos::create_mirror_view(d_h_values);

    // Fill the host mirror.
    size_t idx = 0;
    for (const auto& kv : localToSlicedIdx) {
      h_keys(idx)   = kv.first;
      h_values(idx) = kv.second;
      ++idx;
    }

    // Deep copy back to the device view.
    Kokkos::deep_copy(d_h_keys, h_keys);
    Kokkos::deep_copy(d_h_values, h_values);

    auto d_keys = d_h_keys;
    auto d_values = d_h_values;

    // Create the device unordered map with an initial capacity.
    map_type kokkosMap;
    kokkosMap.rehash(nq_s);
    // Parallel insertion into the device map.
    Kokkos::parallel_for("InsertIntoMap", Kokkos::RangePolicy<grace::default_execution_space>(0, nq_s),
      KOKKOS_LAMBDA(const int i) {
        // Note: insert returns the location or a status value.
        // You may also want to check for failure if your map becomes full.
        //kokkosMap.insert(d_keys(i), d_values(i));
        auto result = kokkosMap.insert(d_keys(i), d_values(i));
    });

    Kokkos::View<amr::OctreeSlicer::SlicedCellInfo*> d_sliced_cells("d_sliced_cells", sliced_cells.size());
    // Create a mirror on the host.
    auto h_sliced_cells = Kokkos::create_mirror_view(d_sliced_cells);
    // Copy data from your host container into the mirror.
    for (size_t i = 0; i < sliced_cells.size(); ++i) {
      h_sliced_cells(i) = sliced_cells[i];
    }

    // Deep copy to device.
    Kokkos::deep_copy(d_sliced_cells, h_sliced_cells);
    Kokkos::fence();
    /**********************************************/
    /* We need an extra device mirror because:    */
    /* 1) The view is not contiguous since we     */
    /*    cut out the ghost-zones.                */
    /* 2) The layout may differ from the          */
    /*    memory layout on device which           */
    /*    usually follows the FORTRAN convention. */
    /**********************************************/
    Kokkos::View<double *EXPR(*,*,*)*, Kokkos::LayoutLeft> 
        d_mirror("Device output mirror", 6, VEC(nx,ny,nz), nq) ; 
    Kokkos::View<double *EXPR(*,*,*)*, Kokkos::LayoutLeft>
        d_temp_sliced("Device output mirror", 3, VEC(nx_s,ny_s,nz_s), nq_s) ;
    auto h_mirror_sliced = Kokkos::create_mirror_view(d_temp_sliced) ;
    for( auto const& vname: varlist )
    {
        GRACE_TRACE("Writing tensor var {} to output.") ; 
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
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableType", "Tensor6");
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableStaggering", "Cell");
        
        std::array<std::string, 6> const compnames 
                = {
                    vname + "[0,0]",
                    vname + "[1,1]",
                    vname + "[2,2]",
                    vname + "[0,1]",
                    vname + "[0,2]",
                    vname + "[1,2]"
                } ; 
        int err ; 
        auto props = variables::get_variable_properties(compnames[0], err) ;
        ASSERT( err == 0, "Error retrieving variable properties for variable" << vname) ; 
        ASSERT( props.staggering == var_staggering_t::CELL_CENTER,
                "Something went wrong, corner staggered variable marked as cell-center for output." ) ;  
        for( int icomp=0; icomp<6; ++icomp){
            auto sview = get_variable_subview(
                      compnames[icomp]
                    , Kokkos::pair<int,int>(ngz,nx+ngz)
                    , Kokkos::pair<int,int>(ngz,ny+ngz)
                    #ifdef GRACE_3D
                    , Kokkos::pair<int,int>(ngz,nz+ngz)
                    #endif 
                    , Kokkos::ALL()
                ) ;
            Kokkos::parallel_for("pack_quadrants", Kokkos::RangePolicy<>(0, d_sliced_cells.size()), KOKKOS_LAMBDA(const int it) {
                auto& sliced_cell = d_sliced_cells[it];
                auto const& q = sliced_cell.q ;
                auto const q_local = q.localQuadrantIdx ;
                auto const& i = sliced_cell.i ;
                auto const& j = sliced_cell.j ;
                auto const& k = sliced_cell.k ;
                auto sliced_q = kokkosMap.find(q_local);
                sliced_q = kokkosMap.value_at(sliced_q);
                d_temp_sliced(icomp, 0, j, k, sliced_q) = sview(i, j, k, q_local);
            });
            
        }
        Kokkos::deep_copy(grace::default_execution_space{},h_mirror_sliced,d_temp_sliced) ; 

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[2]  = {local_quad_offset * ncells_quad_sliced, 0} ; 
        hsize_t slab_count[2]  = {ncells, 6} ;
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
                            , reinterpret_cast<void*>(h_mirror_sliced.data()) )) ;
        

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }
    /* Staggered variables */
    // Resize staging buffers
    Kokkos::realloc(d_mirror,6, VEC(nx+1,ny+1,nz+1), nq) ; 
    Kokkos::realloc(d_temp_sliced,3,VEC(nx_s+1,ny_s+1,nz_s+1), nq_s) ;
    h_mirror_sliced = Kokkos::create_mirror_view(d_temp_sliced) ; 
    // Loop over variables
    for( auto const& vname: svarlist )
    {
        GRACE_TRACE("Writing tensor var {} to output.") ; 
        /* create HDF5 dataset */
        std::string dset_name = "/" + vname ; 
        hid_t dset_id ; 
        HDF5_CALL( dset_id
                , H5Dcreate2( file_id
                            , dset_name.c_str()
                            , H5T_NATIVE_DOUBLE
                            , sspace_id_glob
                            , H5P_DEFAULT
                            , sprop_id
                            , H5P_DEFAULT) ) ;
        /* Write dataset attributes */
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableType", "Tensor6");
        write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableStaggering", "Node");
        
        std::array<std::string, 6> const compnames 
                = {
                    vname + "[0,0]",
                    vname + "[1,1]",
                    vname + "[2,2]",
                    vname + "[0,1]",
                    vname + "[0,2]",
                    vname + "[1,2]"
                } ; 
        int err ; 
        auto props = variables::get_variable_properties(compnames[0], err) ;
        ASSERT( err == 0, "Error retrieving variable properties for variable" << vname) ; 
        ASSERT( props.staggering == var_staggering_t::CORNER,
                "Only corner staggered variables are supported for staggered variables output." ) ;  
        for( int icomp=0; icomp<6; ++icomp){
            auto sview = get_variable_subview(
                      compnames[icomp]
                    , Kokkos::pair<int,int>(ngz,nx+1+ngz)
                    , Kokkos::pair<int,int>(ngz,ny+1+ngz)
                    #ifdef GRACE_3D
                    , Kokkos::pair<int,int>(ngz,nz+1+ngz)
                    #endif 
                    , Kokkos::ALL()
                ) ;
            Kokkos::parallel_for("pack_quadrants", Kokkos::RangePolicy<>(0, d_sliced_cells.size()), KOKKOS_LAMBDA(const int it) {
                auto& sliced_cell = d_sliced_cells[it];
                auto const& q = sliced_cell.q ;
                auto const& q_local = q.localQuadrantIdx ;
                auto const& i = sliced_cell.i ;
                auto const& j = sliced_cell.j ;
                auto const& k = sliced_cell.k ;
                auto sliced_q = kokkosMap.find(q_local);
                sliced_q = kokkosMap.value_at(sliced_q);
                if (((j+1) < ny ) and ((k+1) < nz )) {
                d_temp_sliced(icomp, 0, j, k, sliced_q) = sview(i, j, k, q_local);
                d_temp_sliced(icomp, 1, j, k, sliced_q) = sview(i+1, j, k, q_local);
                }
                else if (((j+1) < ny ) and ((k+1) == nz )) {
                d_temp_sliced(icomp, 0, j, k+1, sliced_q) = sview(i, j, k+1, q_local);
                d_temp_sliced(icomp, 1, j, k+1, sliced_q) = sview(i+1, j, k+1, q_local);
                }
                else if (((j+1) == ny ) and ((k+1) < nz )) {
                d_temp_sliced(icomp, 0, j+1, k, sliced_q) = sview(i, j+1, k, q_local);
                d_temp_sliced(icomp, 1, j+1, k, sliced_q) = sview(i+1, j+1, k, q_local);
                }
                else if (((j+1) == ny ) and ((k+1) == nz )) {
                d_temp_sliced(icomp, 0, j+1, k+1, sliced_q) = sview(i, j+1, k+1, q_local);
                d_temp_sliced(icomp, 1, j+1, k+1, sliced_q) = sview(i+1, j+1, k+1, q_local);
                d_temp_sliced(icomp, 0, j, k+1, sliced_q) = sview(i, j, k+1, q_local);
                d_temp_sliced(icomp, 1, j, k+1, sliced_q) = sview(i+1, j, k+1, q_local);
                d_temp_sliced(icomp, 0, j+1, k, sliced_q) = sview(i, j+1, k, q_local);
                d_temp_sliced(icomp, 1, j+1, k, sliced_q) = sview(i+1, j+1, k, q_local);
                }
            });
            
        }
        
        /* Copy data d2h */
        Kokkos::deep_copy(grace::default_execution_space{},h_mirror_sliced,d_mirror) ; 

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[2]  = {local_quad_offset * ncorners_quad_sliced, 0} ; 
        hsize_t slab_count[2]  = {ncorners_quad_sliced * nq_s, 6} ;
        HDF5_CALL( err
                , H5Sselect_hyperslab( sspace_id_glob
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
                            , sspace_id
                            , sspace_id_glob 
                            , dxpl 
                            , reinterpret_cast<void*>(h_mirror_sliced.data()) )) ;
        

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }
}

void write_extra_arrays_sliced_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, amr::OctreeSlicer& octree_slicer) {
    herr_t err ;
    /* Get cell and quadrant counts */
    size_t nx,ny,nz,nq, nx_s,ny_s,nz_s,nq_s;
    std::tie(nx_s,ny_s,nz_s) = octree_slicer.get_quadrant_extents() ; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ;
    nq_s = octree_slicer.sliced_quadrants().size() ; 
    nq = grace::amr::get_local_num_quadrants() ;
    size_t ngz = grace::amr::get_n_ghosts() ;
    auto const& localToSlicedIdx = octree_slicer.get_localToSlicedIdx();

    auto const rank_loc = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */

    unsigned long nq_glob_recv_buf; 
    parallel::mpi_allreduce(&nq_s, &nq_glob_recv_buf, 1, mpi_sum, parallel::get_comm_world()) ;
    unsigned long const nq_glob = nq_glob_recv_buf ;

    size_t local_quad_offset_recv_buf; // has to be the same type as nq
    parallel::mpi_exscan_sum( &nq_s, &local_quad_offset_recv_buf, 1, parallel::get_comm_world() ) ;
    if (rank_loc == 0) local_quad_offset_recv_buf = 0 ;

    unsigned long const local_quad_offset = local_quad_offset_recv_buf ; 

    /* Number of cells per quadrant */
    unsigned long const ncells_quad = EXPR(nx_s,*ny_s,*nz_s) ; 
    /* Local number of cells   */
    unsigned long const ncells = ncells_quad * nq_s ; 
    Kokkos::printf("Rank %d: Number of cells %lu, Number of quadrants %lu\n", rank_loc, ncells, nq_s) ;
    Kokkos::printf("octree_slicer.sliced_cells().size() %lu\n", octree_slicer.sliced_cells().size()) ;
    /* Global number of cells  */
    unsigned long const ncells_glob = ncells_quad * nq_glob ; 

    unsigned int* lev   = (unsigned int*) malloc(sizeof(unsigned int) * ncells ) ;
    unsigned int* rank  = (unsigned int*) malloc(sizeof(unsigned int) * ncells ) ;
    unsigned int* tree_id    = (unsigned int*) malloc(sizeof(unsigned int) * ncells ) ;
    unsigned long long* qid  = (unsigned long long*) malloc(sizeof(unsigned long long) * ncells ) ;  

    unsigned int icell_max  = 0L ; 
    
    #pragma omp parallel for reduction(max:icell_max)
    for(auto sliced_cell : octree_slicer.sliced_cells()) {
        auto const& q = sliced_cell.q ;
        auto i = sliced_cell.i ;
        auto const& j = sliced_cell.j ;
        auto const& k = sliced_cell.k ;
        auto const& qglobal = q.globalIndex ;
        auto const qlocal = q.localQuadrantIdx ; 
        auto const qsliced = localToSlicedIdx.at(qlocal) ;
        // nq = number of quadrants
        auto const nq_p4est = grace::amr::get_local_num_quadrants() ;
        unsigned long icell = 0 + nx_s * ( j + ny_s * ( k  + nz_s * qsliced )) ; 
        
        unsigned int itree = amr::get_quadrant_owner(qlocal) ; 
        auto quad  = amr::get_quadrant(itree,qlocal) ; 
        unsigned int level = quad.level() ;
        size_t iquad_glob = qlocal + amr::forest::get().global_quadrant_offset(rank_loc) ;

        rank[icell]    = rank_loc   ;
        lev[icell]     = level      ;
        tree_id[icell] = itree      ;
        qid[icell]     = qglobal ; 
        icell_max = max(icell,icell_max) ;    
    }
    Kokkos::printf("icell_max %lu, ncells %lu\n", icell_max, ncells) ;
    ASSERT((icell_max+1) == ncells, "Something went really wrong") ; 

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
    HDF5_CALL(err, H5Pset_deflate(prop_id, compression_level)) ; 
    HDF5_CALL( dset_id
             , H5Dcreate2( file_id
                         , dset_name.c_str()
                         , mem_type_id
                         , space_id_glob
                         , H5P_DEFAULT
                         , prop_id
                         , H5P_DEFAULT) ) ;;

    write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableType", "Scalar");
    write_dataset_string_attribute_sliced_hdf5(dset_id, "VariableStaggering", "Cell");
    
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

void write_dataset_string_attribute_sliced_hdf5(hid_t dset_id, std::string const& attr_name, std::string const& attr_data)
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
