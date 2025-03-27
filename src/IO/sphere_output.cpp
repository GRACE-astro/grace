

/**
 * @file sphere_output.cpp
 * @authors Konrad Topolski, Kenneth Miller 
 * @brief 
 * @date 2025-03-21
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
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
#include <hdf5.h>

#include <array>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <cmath>

#include <grace/IO/sphere_output.hh>
#include <grace/healpix/detectors.hh>
#include <numeric>  // For std::iota

namespace grace { namespace IO {

    using namespace healpix;
    std::map<std::string, healpix_detector> detectors;




 
    void initialize_spherical_detectors(const int n_detectors, 
                                        const int nside,
                                        const std::vector<std::array<double,3>> output_spheres_centres,
                                        const std::vector<double> output_spheres_radii,
                                        const std::vector<std::string> output_spheres_names
                                        ){

            // construct the string-detector map
            for(size_t id_det=0; id_det<n_detectors; id_det++){
                    detectors.emplace(output_spheres_names[id_det], 
                                      healpix::healpix_detector(nside, output_spheres_radii[id_det], output_spheres_centres[id_det]));
            }
    }

    void update_spherical_detectors(){
        for (auto& [name, detector] : detectors) {
            detector.update_detector_info();
        }
    }

    void compute_multipoles( ){

    }

    void compute_spherical_surface_variable_data( ){
        auto& runtime = grace::runtime::get( ) ;
        const std::set<std::string> corner_scalar_vars = runtime.corner_sphere_surface_output_scalar_vars();
        const std::set<std::string> corner_vector_vars = runtime.corner_sphere_surface_output_vector_vars();
        const std::set<std::string> cell_scalar_vars = runtime.cell_sphere_surface_output_scalar_vars();
        const std::set<std::string> cell_vector_vars = runtime.cell_sphere_surface_output_vector_vars();
        
         for (auto& [name, detector] : detectors) {
            detector.update_detector_variable_data(corner_scalar_vars,
                                                   corner_vector_vars,
                                                   cell_scalar_vars,
                                                   cell_vector_vars,
                                                   INTERPOLATION_METHODS::LINEAR // this should be steerable by the parfile
                                                   //INTERPOLATION_METHODS::LAGRANGE3
                                                    );
        }
        //


    }

        

    void InitFile(const std::string& filename, 
                  const double& det_radius) { 
        // char *fn = nullptr;
        hid_t file_id, dspace_id, dset_id, group_id ;

        hsize_t size{1};
        herr_t h5err ;

        // the master rank takes care of HDF5 I/O
        // Here notice that it may not hold any information regarding
        // this particular surface. But it doesn't matter since we're
        // only writing its radius now.
        if ( parallel::mpi_comm_rank() == grace::master_rank() )
        {
            const char* fn = filename.c_str();
            // const char *fn = filename.c_str();
            // Util_asprintf(&fn, "%s/healpix_det_%d_surf.h5",out_dir,det_id) ;
            assert(fn) ; 

            hid_t is_hdf5;
            H5E_BEGIN_TRY { is_hdf5 = H5Fopen(fn,
                                                H5F_ACC_RDWR,
                                                H5P_DEFAULT ) ; }
            H5E_END_TRY ;
            // if the file exists we have nothing to initialize
            if ( is_hdf5 >= 0)
            return ;
            
            file_id = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) ; assert( file_id >= 0 );

            dspace_id = H5Screate_simple( 1, &size, nullptr) ; assert(dspace_id>=0);

            dset_id = H5Dcreate2( file_id, "/radius", H5T_NATIVE_DOUBLE, dspace_id,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) ; assert(dset_id>=0);

            h5err = H5Dwrite( dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, &det_radius ) ; assert(h5err>=0);

            h5err = H5Dclose(dset_id) ; assert(h5err>=0);
            
            h5err = H5Sclose(dspace_id) ; assert(h5err>=0);

            // create /data group
            group_id = H5Gcreate(file_id,"data",H5P_DEFAULT,H5P_DEFAULT,
                                H5P_DEFAULT); assert(group_id>=0);

            h5err = H5Gclose(group_id);assert(h5err>=0);

            h5err = H5Fclose(file_id) ;assert(h5err>=0);
            
        }

        return ; 
    }

    void CreateIter(const std::string& filename, const int& iteration, const double& time) {
        
        hid_t file_id, dspace_id, dset_id, group_id, dgroup_id ;
        herr_t h5err ;
     
        if ( parallel::mpi_comm_rank() == grace::master_rank() ){
            const char *fn = filename.c_str();

            file_id = H5Fopen( fn,
                    H5F_ACC_RDWR,
                    H5P_DEFAULT ) ; assert(file_id>=0);

            group_id = H5Gopen( file_id,
                    "data",
                    H5P_DEFAULT ) ; assert(group_id>=0);

            char grpname[200] ;
            sprintf( grpname, "%d", iteration ) ;

            dgroup_id = H5Gcreate( group_id,
                    grpname, H5P_DEFAULT,
                    H5P_DEFAULT, H5P_DEFAULT ) ; assert(dgroup_id>=0);

            hsize_t npoints = 1 ;
            dspace_id =  H5Screate_simple( 1, &npoints, nullptr ) ; assert(dspace_id>=0);

            dset_id   = H5Dcreate2( dgroup_id, "time", H5T_NATIVE_DOUBLE, dspace_id,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) ; assert(dset_id>=0);

            h5err = H5Dwrite( dset_id , H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &time ) ; 
            assert( h5err >= 0 ) ;

            h5err = H5Dclose(dset_id) ; assert( h5err >= 0 ) ;
            h5err = H5Sclose(dspace_id); assert( h5err >= 0 ) ;
            
            h5err = H5Gclose(group_id); assert( h5err >= 0 ) ;
            h5err = H5Gclose(dgroup_id); assert( h5err >= 0 ) ;
            h5err = H5Fclose(file_id); assert( h5err >= 0 ) ;
        }

        return ;
    }

    void WriteSingleField(const double* data_buffer, const size_t n_pts, std::string const& vname,
                          const int iter_num, std::string const& filename
                          ) {

        hid_t file_id, dspace_id, dset_id, group_id, dgroup_id ;
        hsize_t npoints{n_pts} ;
        herr_t h5err ;
        const char *fn = filename.c_str();

        file_id = H5Fopen(fn,
                H5F_ACC_RDWR,
                H5P_DEFAULT ) ; assert(file_id>=0);

        group_id = H5Gopen( file_id,
                "data",
                H5P_DEFAULT ) ; assert(group_id>=0) ;

        char grpname[200] ;
        
        sprintf( grpname, "%d", iter_num ) ;

        dgroup_id = H5Gopen( group_id,
                grpname,
                H5P_DEFAULT ) ; assert(dgroup_id>=0);

        dspace_id =  H5Screate_simple( 1, &npoints, nullptr ) ; assert(dspace_id>=0);


        dset_id   = H5Dcreate2( dgroup_id, vname.c_str(), H5T_NATIVE_DOUBLE, dspace_id,
                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ) ; assert(dset_id>=0) ;
        
        h5err = H5Dwrite( dset_id , H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, reinterpret_cast<void const *> (data_buffer) ) ;
        assert( h5err >= 0 ) ;

        h5err = H5Dclose(dset_id) ; assert( h5err >= 0 ) ;
        h5err = H5Sclose(dspace_id); assert( h5err >= 0 ) ;
        h5err = H5Gclose(dgroup_id); assert( h5err >= 0 ) ;
        h5err = H5Gclose(group_id); assert( h5err >= 0 ) ;
        h5err = H5Fclose(file_id); assert( h5err >= 0 ) ;

        return ;
        
    }


    void write_sphere_cell_data_hdf5( ){

        auto& runtime = grace::runtime::get( ) ;
    
        const auto comm = parallel::get_comm_world() ; 
        const int rank = parallel::mpi_comm_rank()  ; 
        const int world_size = parallel::mpi_comm_size()  ;
        const double current_time =  grace::get_simulation_time() ;
        const int current_iteration = grace::get_iteration() ;

        const int n_detectors = runtime.n_surface_output_spheres();
        const int nside = runtime.nside_surface_output_spheres();
        const std::vector<std::array<double,3>> output_spheres_centres  = runtime.cell_sphere_surface_output_centers();
        const std::vector<double> output_spheres_radii                  = runtime.cell_sphere_surface_output_radii()  ;
        const std::vector<std::string> output_spheres_names             = runtime.cell_sphere_surface_output_names();

        bool hdf5_files_created=true;

        // std::filesystem::path base_path (runtime.surface_io_basepath()) ;
        // std::ostringstream oss;

        if(n_detectors!=0){
            
            // if not yet initialized
            if(detectors.empty()){
                hdf5_files_created = false;
                GRACE_INFO("Initializing spherical surfaces.") ; 
                GRACE_VERBOSE( "There's {} detectors in total" , n_detectors) ; 

                initialize_spherical_detectors(n_detectors, nside,
                                                output_spheres_centres,
                                                output_spheres_radii,
                                                output_spheres_names ) ;
            }

            for (auto& [name, detector] : detectors) {
                // initialize HDF5 files if needed 
                std::filesystem::path base_path (runtime.surface_io_basepath()) ;
                const std::string filename =  "./healpix_det_" + name + "_surf.h5";
                std::filesystem::path out_path = base_path / filename ;
                // Resolve to absolute path
                std::filesystem::path absolute_path = std::filesystem::absolute(out_path.lexically_normal());

                if (rank == grace::master_rank()) {
                    //if(detectors.empty()){
                    if( !hdf5_files_created ){
                        GRACE_VERBOSE( "Initializing Surface HDF5 files" ) ; 
                        InitFile(absolute_path.string(), detector.radius_det_);
                    }
                    CreateIter(absolute_path.string(), current_iteration, current_time);
                }
            }
        } 

        update_spherical_detectors();
        GRACE_VERBOSE("Updated spherical surfaces info.") ; 

        compute_spherical_surface_variable_data();
        GRACE_VERBOSE("Interpolated variables on spherical surfaces.") ; 

        // IF MODE == SERIAL_WRITING 

        int detector_counter = 0;  // Initialize counter
        // this loop automatically omit ranks that do not have a detector assigned (e.g. no coordinate overlap)
        for (auto& [name, detector] : detectors) {
            std::vector<int> det_healpix_indices = detector.get_local_rank_healpix_indices();
            std::map<std::string,std::vector<double>> det_surface_data = detector.get_local_rank_detector_surface_data();

            std::filesystem::path base_path (runtime.surface_io_basepath()) ;
            const std::string filename =  "./healpix_det_" + name + "_surf.h5";
            std::filesystem::path out_path = base_path / filename ;
            // Resolve to absolute path
            std::filesystem::path absolute_path = std::filesystem::absolute(out_path.lexically_normal());

            // Gather the sizes first
            std::vector<int> recv_counts(world_size);  
            int local_size = det_healpix_indices.size(); // Each rank's send count
            parallel::mpi_gather(&local_size,              // send buffer
                                  1,                       // send count
                                  recv_counts.data(),      // recv buffer
                                  1,                       // recv count
                                  grace::master_rank());   // master rank

            // Compute displacements (only on root)
            std::vector<int> displacements(world_size, 0);
            if (rank == grace::master_rank()) {
                for (int i = 1; i < world_size; i++) {
                    displacements[i] = displacements[i - 1] + recv_counts[i - 1]; // Prefix sum
                }
            }

            std::vector<int> unsorted_healpix_indices ;
            unsorted_healpix_indices.resize(12 * nside * nside); 

            // all ranks send their data and local healpix detector indices
            // gatherv since different ranks might have different amount of data 
            parallel::mpi_gatherv(det_healpix_indices.data(),            // T* send_buffer
                                det_healpix_indices.size(),              // int send_count
                                unsorted_healpix_indices.data(),         // T* recv_buffer
                                recv_counts.data(),                      // int* recv_count   (arr)
                                displacements.data(),                    // int* recv_offsets (arr)
                                grace::master_rank()                     // root  
                                );

            // cannot const-qualify the auto& due to mpi_gather signature
            for ( auto& [var_name, var_data] : det_surface_data) {  
                std::vector<double> unsorted_var_data;
                unsorted_var_data.resize(12 * nside * nside); 

                parallel::mpi_gatherv(var_data.data(),          // T* send_buffer
                                var_data.size(),                // int send_count
                                unsorted_var_data.data(),       // T* recv_buffer
                                recv_counts.data(),             // int* recv_count   (arr)
                                displacements.data(),           // int* recv_offsets (arr)
                                grace::master_rank()            // root  
                                );

                // I/O operations with the master rank
                if(rank == grace::master_rank()){
                    std::vector<double> sorted_var_data;
                    sorted_var_data.resize(12 * nside * nside);
                    for(size_t i_px = 0; i_px < 12*nside*nside; i_px++){
                        sorted_var_data[unsorted_healpix_indices[i_px]] = unsorted_var_data[i_px];
                            // printf("i_px, unsortedhrealpixidx: %d,%d \n", i_px,unsorted_healpix_indices[i_px]);
                        }

                    // we are compliant with the original format https://arxiv.org/abs/2402.11009
                    // add the sorted_var_data to the HDF5 file:
                    printf("Some psi4 values %f, %f, %f:", sorted_var_data[10], sorted_var_data[50], sorted_var_data[100]);
                    // GRACE_VERBOSE("var_name, iter, abspath: {}, {}, {}",var_name, current_iteration,absolute_path.string());
                    WriteSingleField(sorted_var_data.data(), 
                                     sorted_var_data.size(), 
                                     var_name,
                                     current_iteration,
                                     absolute_path.string()
                                    ) ;
                }

            }
                detector_counter++;
        }  
        
        GRACE_VERBOSE("Saved spherical data.") ; 

    } 

    void write_multipole_and_integral_timeseries( ){

    } 


  }
}

