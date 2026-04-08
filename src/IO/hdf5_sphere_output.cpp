/**
 * @file hdf5_sphere_output.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @version 0.1
 * @date 2025-12-05
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
#include <grace/IO/spherical_surfaces.hh>
#include <grace/IO/hdf5_output.hh>
#include <grace/IO/hdf5_sphere_output.hh>
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
#include <unordered_map>

#define HDF5_CALL(result,cmd) \
    do {  \
        if((result=cmd)<0) { \
            ERROR("HDF5 API call failed with error code " << result ) ; \
        } \
    } while(false)

namespace grace { namespace IO {

void write_sphere_cell_data_impl(const spherical_surface_iface&) ; 
void write_grid_structure_sphere_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, const spherical_surface_iface& sphere) ; 
void write_data_arrays_sphere_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, const spherical_surface_iface& sphere) ;


template< typename ViewT> 
void write_var_arrays_sphere_hdf5( 
    std::vector<std::string> const& varlist,
    std::unordered_map<std::string,int> imap,
    hid_t file_id, hid_t dxpl,
    hid_t space_id_glob, hid_t space_id,
    hid_t prop_id, hsize_t npoints, hsize_t local_offset,
    ViewT data ) 
{
    herr_t err ; 
    /**********************************************/
    /* We need an extra device mirror because:    */
    /* 1) The view is not contiguous since we     */
    /*    cut out the ghost-zones.                */
    /* 2) The layout may differ from the          */
    /*    memory layout on device which           */
    /*    usually follows the FORTRAN convention. */
    /**********************************************/    
    for( auto const& vname: varlist )
    {
        size_t varidx = imap[vname] ; 
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
        write_dataset_string_attribute_hdf5(dset_id, "VariableStaggering", "Node");        

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[1]  = {local_offset} ; 
        hsize_t slab_count[1]  = {npoints} ;
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
                            , reinterpret_cast<void*>(data.data() + npoints*varidx) )) ;

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }
}

template< typename ViewT>
void write_vector_var_arrays_sphere_hdf5(  
        std::vector<std::string> const& varlist,
        std::unordered_map<std::string,int> imap,
        hid_t file_id, hid_t dxpl,
        hid_t space_id_glob, hid_t space_id,
        hid_t prop_id, hsize_t npoints, hsize_t local_offset,
        ViewT data ) 
{
    using namespace grace; 
    using namespace Kokkos; 
    herr_t err ;

    /**********************************************/
    /* We need an extra device mirror because:    */
    /* 1) The view is not contiguous since we     */
    /*    cut out the ghost-zones.                */
    /* 2) The layout may differ from the          */
    /*    memory layout on device which           */
    /*    usually follows the FORTRAN convention. */
    /**********************************************/
    double *data_vec = (double*) malloc(sizeof(double) * npoints * 3) ; 
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
        write_dataset_string_attribute_hdf5(dset_id, "VariableStaggering", "Node");
        
        std::string vname_i = vname + "[" + std::to_string(0) + "]";
        size_t varidx = imap[vname_i] ; 
        for( int i=0; i<npoints;++i) {
            #pragma unroll 
            for( int icomp=0; icomp<3;++icomp)
                data_vec[i*3 + icomp] = data(i,varidx+icomp) ; 
        }
        

        /* Select hyperslab for this rank's output */
        hsize_t slab_start[2]  = {local_offset, 0} ; 
        hsize_t slab_count[2]  = {npoints, 3} ;
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
                            , reinterpret_cast<void*>(data_vec) )) ;
        

        /* Close dataset */
        HDF5_CALL(err, H5Dclose(dset_id)) ; 
    }
    // free memory 
    free(data_vec) ; 
}


static void get_sphere_indices(std::vector<size_t>& idx) {
    auto& rt = grace::runtime::get() ; 
    auto& sfm = grace::spherical_surface_manager::get() ;

    auto n_spheres = rt.n_surface_output_spheres() ;
    auto names = rt.cell_sphere_surface_output_names() ; /* fixme */
    idx.reserve(n_spheres) ; 
    for( auto const& n: names ) {
        auto _idx = sfm.get_index(n) ; 
        if ( _idx < 0 ) {
            GRACE_WARN("Spherical surface {} could not be retrieved", n) ; 
            continue ; 
        }
        idx.push_back(_idx) ; 
    }
} 

void write_sphere_cell_data() {
    Kokkos::Profiling::pushRegion("HDF5 sphere output") ;
    GRACE_VERBOSE("Performing HDF5 output of spherical surface data.") ; 
    auto& rt = grace::runtime::get() ; 
    auto& sfm = grace::spherical_surface_manager::get() ; 

    std::vector<size_t> sphere_indices ; 
    get_sphere_indices(sphere_indices) ; 

    for( int isph=0; isph<sphere_indices.size(); ++isph) {
        auto const& sphere = sfm.get(sphere_indices[isph]) ; 
        write_sphere_cell_data_impl(sphere) ; 
    }

    Kokkos::Profiling::popRegion() ;
}

void write_sphere_cell_data_impl(const spherical_surface_iface& sphere) {
    auto& rt = grace::runtime::get() ;

    std::filesystem::path base_path (rt.sphere_io_basepath()) ;
    std::ostringstream oss;
    oss << rt.surface_io_basename() << "_sphere" << "_" << sphere.name << "_"
        << std::setw(6) << std::setfill('0') << grace::get_iteration() << ".h5";
    std::string pfname = oss.str();
    std::filesystem::path out_path = base_path / pfname ;

    /* fetch some params */
    auto& params = grace::config_parser::get() ;
    size_t compression_level = params["IO"]["hdf5_compression_level"].as<size_t>() ;
    size_t chunk_size = params["IO"]["hdf5_chunk_size"].as<size_t>() ;

    if( chunk_size > sphere.npoints_glob ) {
        GRACE_WARN("Chunk size {} < number of cells {} will be overridden." , chunk_size, sphere.npoints_glob ) ; 
        chunk_size = math::max(1UL,sphere.npoints_glob ); 
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
    write_grid_structure_sphere_hdf5(file_id, compression_level,chunk_size, sphere) ;
    /* Write requested variables to hdf5 file */
    write_data_arrays_sphere_hdf5(file_id, compression_level,chunk_size, sphere) ;
    /* Wait for everyone to be done           */
    parallel::mpi_barrier() ; 
    /* Close the file */
    HDF5_CALL(err,H5Fclose(file_id)) ; 
    HDF5_CALL(err,H5Pclose(plist_id)) ; 

}; 

void write_grid_structure_sphere_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, const spherical_surface_iface& sphere) {
    DECLARE_GRID_EXTENTS ; 

    herr_t err ; 

    /* Number of points on the sphere */
    auto npoints_glob = sphere.npoints_glob ;
    auto npoints_loc = sphere.npoints_loc ; 

    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get local offset */
    size_t local_offset; // has to be the same type as nq
    parallel::mpi_exscan_sum( &npoints_loc, &local_offset, 1, parallel::get_comm_world() ) ;
    if (rank == 0) local_offset = 0 ;

    double*  points= (double*) malloc(sizeof(double) * npoints_loc * 3 ) ; 

    #pragma omp parallel for 
    for( int i=0; i<npoints_loc; ++i) {
        auto ip = sphere.intersecting_points_h[i];
        auto const& coords = sphere.points_h[ip].second ; 
        points[3*i + 0] = coords[0] ;
        points[3*i + 1] = coords[1] ;
        points[3*i + 2] = coords[2] ; 
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
    /* Set chunking / compression */
    hid_t points_dset_id, points_prop_id  ;
    HDF5_CALL(points_prop_id, H5Pcreate(H5P_DATASET_CREATE)) ; 
    hsize_t points_chunk_dim[2] = {chunk_size,3} ;
    HDF5_CALL(err, H5Pset_chunk(points_prop_id,2,points_chunk_dim)) ; 
    if ( compression_level > 0 )
        HDF5_CALL(err, H5Pset_deflate(points_prop_id, compression_level)) ;
    /* Create dataset */ 
    HDF5_CALL( points_dset_id
            , H5Dcreate2( file_id
                        , "/Points"
                        , H5T_NATIVE_DOUBLE
                        , points_space_id_glob
                        , H5P_DEFAULT
                        , points_prop_id
                        , H5P_DEFAULT) ) ;
    /* Write points dataset */
    /* Create local space for this rank */
    hid_t points_space_id ; 
    hsize_t points_dset_dims[2] = {npoints_loc, GRACE_NSPACEDIM} ;
    HDF5_CALL(points_space_id, H5Screate_simple(2, points_dset_dims, NULL)) ; 
    /* Select hyperslab for this rank's output */
    hsize_t points_slab_start[2]  = {local_offset,0} ;
    GRACE_VERBOSE("Slab offset {}, size {}, total {}", points_slab_start[0], npoints_loc, npoints_glob) ;  
    hsize_t points_slab_count[2]  = {npoints_loc,GRACE_NSPACEDIM} ;
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

static std::unordered_map<std::string,int> 
collect_variables_for_output(
    std::vector<int>& varidx, std::vector<int>& auxidx
)
{
    std::unordered_map<std::string,int> map ;  
    size_t ii=0UL ; 

    auto& runtime = grace::runtime::get() ;

    auto const scalars     = runtime.cell_sphere_surface_output_scalar_vars() ; 
    auto const aux_scalars = runtime.cell_sphere_surface_output_scalar_aux()  ;
    auto const vectors     = runtime.cell_sphere_surface_output_vector_vars() ; 
    auto const aux_vectors = runtime.cell_sphere_surface_output_vector_aux()  ;

    varidx.reserve(scalars.size() + 3*vectors.size()) ; 
    auxidx.reserve(aux_scalars.size() + 3*aux_vectors.size()) ; 
    for( auto const& vname: scalars) {
        varidx.push_back(grace::get_variable_index(vname,false)) ; 
        map[vname] = ii++ ; 
    }
    for(auto const& vname: vectors) {
        for(int i = 0; i < 3; ++i) {
            std::string vname_i = vname + "[" + std::to_string(i) + "]";
            varidx.push_back(grace::get_variable_index(vname_i, false));
            map[vname] = ii++ ; 
        }
    }

    for( auto const& vname: aux_scalars) {
        auxidx.push_back(grace::get_variable_index(vname,true)) ; 
        map[vname] = ii++ ; 
    }
    for(auto const& vname: aux_vectors) {
        for(int i = 0; i < 3; ++i) {
            std::string vname_i = vname + "[" + std::to_string(i) + "]";
            auxidx.push_back(grace::get_variable_index(vname_i, true));
            map[vname] = ii++ ; 
        }
    }
    return map ; 
}

void write_data_arrays_sphere_hdf5(hid_t file_id, size_t compression_level, size_t chunk_size, const spherical_surface_iface& sphere) 
{
    /******************************************************/
    /* First we collect all the variables and interpolate */
    /******************************************************/
    auto& runtime = grace::runtime::get() ;
    std::vector<int> varidx, auxidx; 
    auto map = collect_variables_for_output(varidx,auxidx) ; 
    Kokkos::View<double**,grace::default_space> interp_data_d ; 
    Kokkos::View<double**,grace::default_space> interp_data_aux_d ; 
    interpolate_on_sphere(sphere,varidx,auxidx,interp_data_d, interp_data_aux_d) ; 
    auto interp_data_h = Kokkos::create_mirror_view(interp_data_d) ; 
    Kokkos::deep_copy(interp_data_h,interp_data_d) ; 
    auto interp_data_aux_h = Kokkos::create_mirror_view(interp_data_d) ;
    Kokkos::deep_copy(interp_data_aux_h,interp_data_aux_d) ;  
    /******************************************************/
    /* Then we write them to file one at a time           */
    /******************************************************/
    herr_t err ; 
    /* Number of points on the sphere */
    auto npoints_glob = sphere.npoints_glob ;
    auto npoints_loc = sphere.npoints_loc ; 

    auto const rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = grace::amr::forest::get().get() ; 
    /* Get local offset */
    size_t local_offset; // has to be the same type as nq
    parallel::mpi_exscan_sum( &npoints_loc, &local_offset, 1, parallel::get_comm_world() ) ;
    if (rank == 0) local_offset = 0 ;

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
    hid_t scalars_space_id_glob ;
    hsize_t scalars_dset_dims_glob[1] = {npoints_glob} ;

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
    hsize_t scalars_dset_dims[1] = {npoints_loc} ;
    HDF5_CALL(scalars_space_id, H5Screate_simple(1, scalars_dset_dims, NULL)) ; 
    /*****************************************************************************************/
    /*                                  Write to file                                        */
    /*****************************************************************************************/
    auto const scalars     = runtime.cell_sphere_surface_output_scalar_vars() ; 
    auto const aux_scalars = runtime.cell_sphere_surface_output_scalar_aux()  ;
    write_var_arrays_sphere_hdf5( 
        scalars, map,
        file_id, dxpl,
        scalars_space_id_glob, scalars_space_id, scalars_prop_id,
        npoints_loc, local_offset, interp_data_h) ; 
    write_var_arrays_sphere_hdf5( 
        aux_scalars, map,
        file_id, dxpl,
        scalars_space_id_glob, scalars_space_id, scalars_prop_id,
        npoints_loc, local_offset, interp_data_aux_h) ;
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
    hsize_t vectors_dset_dims_glob[2] = {npoints_glob, 3} ;

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
    hsize_t vectors_dset_dims[2] = {npoints_loc, 3} ;
    HDF5_CALL(vectors_space_id, H5Screate_simple(2, vectors_dset_dims, NULL)) ; 
    /*****************************************************************************************/

    /*****************************************************************************************/
    /*                                  Write to file                                        */
    /*****************************************************************************************/
    auto const vectors     = runtime.cell_sphere_surface_output_vector_vars() ; 
    auto const aux_vectors = runtime.cell_sphere_surface_output_vector_aux()  ;
    write_vector_var_arrays_sphere_hdf5( 
        vectors,
        map,
        file_id,dxpl,
        vectors_space_id_glob,vectors_space_id,vectors_prop_id,
        npoints_loc, local_offset, interp_data_h) ; 
    write_vector_var_arrays_sphere_hdf5( 
        aux_vectors,
        map,
        file_id,dxpl,
        vectors_space_id_glob,vectors_space_id,vectors_prop_id,
        npoints_loc, local_offset, interp_data_aux_h) ;
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

}} /* namespace grace::IO */
