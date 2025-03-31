

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
#include <grace/utils/numerics/spherical_harmonics.hh>



namespace grace { namespace IO {

    using namespace healpix;
    std::map<std::string, healpix_detector> detectors;

    // access pattern to a s_Y_lm (pixel_id) spherical harmonic reads:
    // spherical_harmonics[ell][m][id_pixel]
    std::vector<std::vector<std::vector<double>>> spherical_harmonics_re; 
    std::vector<std::vector<std::vector<double>>> spherical_harmonics_im; 
        
    // HDF5 helper routines 
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

    // spherical detectors specific code:

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

    void compute_spherical_surface_variable_data(std::set<std::string> corner_scalar_vars,
                                                 std::set<std::string> corner_vector_vars, 
                                                 std::set<std::string> corner_tensor_vars, 
                                                 std::set<std::string> cell_scalar_vars, 
                                                 std::set<std::string> cell_vector_vars,
                                                 std::set<std::string> cell_tensor_vars){

         for (auto& [name, detector] : detectors) {
            detector.update_detector_variable_data(corner_scalar_vars,
                                                   corner_vector_vars,
                                                   corner_tensor_vars,
                                                   cell_scalar_vars,
                                                   cell_vector_vars,
                                                   cell_tensor_vars,
                                                   INTERPOLATION_METHODS::LINEAR // this should be steerable by the parfile
                                                   //INTERPOLATION_METHODS::LAGRANGE3
                                                    );
        }
        //


    }

    /**
     * @brief 
     * 
     * @note the 0-th iteration will feature data of pure zeroes for those grid functions
     *       which are only computed as auxiliary fields 
     */

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

        const std::set<std::string> corner_scalar_vars = runtime.corner_sphere_surface_output_scalar_vars();
        const std::set<std::string> corner_vector_vars = runtime.corner_sphere_surface_output_vector_vars();
        const std::set<std::string> corner_tensor_vars = runtime.corner_sphere_surface_output_tensor_vars();
        const std::set<std::string> cell_scalar_vars = runtime.cell_sphere_surface_output_scalar_vars();
        const std::set<std::string> cell_vector_vars = runtime.cell_sphere_surface_output_vector_vars();
        const std::set<std::string> cell_tensor_vars = runtime.cell_sphere_surface_output_tensor_vars();
        
        update_spherical_detectors();

        GRACE_VERBOSE("Updated spherical surfaces info.") ; 

        compute_spherical_surface_variable_data(corner_scalar_vars,corner_vector_vars,corner_tensor_vars,
                                                cell_scalar_vars,  cell_vector_vars,  cell_tensor_vars );

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
                        }

                    // we are compliant with the original format https://arxiv.org/abs/2402.11009
                    // add the sorted_var_data to the HDF5 file:
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




    void initialize_spherical_harmonics(const int spin_weight, const int max_ell, const int nside){
        // we will have #spin_weight many unused arrays (of interior sizes 0)
        // e.g. spin_weight=2 means l=0 and l=1 are redundant, but
        // it's more convenient to keep the indexing clean
        spherical_harmonics_re.resize(max_ell);
        spherical_harmonics_im.resize(max_ell);


        for (int idx_l = spin_weight; idx_l <= max_ell; ++idx_l) {
            int ell = idx_l; // these coincide
            spherical_harmonics_re[idx_l].resize(2*ell + 1);  // m ranges from -ell to ell
            spherical_harmonics_im[idx_l].resize(2*ell + 1);  // m ranges from -ell to ell
            for (int idx_m = 0; idx_m <= 2*idx_l; ++idx_m) {
                int m = -ell + idx_m ; // the true index is m = -ell + idx_m 

                spherical_harmonics_re[idx_l][idx_m].resize(12*nside*nside);  // Each [ell][idx_m] has 12 * nside**2  entries
                spherical_harmonics_im[idx_l][idx_m].resize(12*nside*nside);  // Each [ell][idx_m] has 12 * nside**2  entries

                for (int id_pixel; id_pixel < 12*nside*nside; id_pixel++){
                    double th, ph;
                    get_spherical_coord_from_healpix_index(nside, id_pixel, th, ph);

                    utils::multipole_spherical_harmonic(spin_weight, ell, m,
                                                 th, ph,
                                                spherical_harmonics_re[idx_l][idx_m][id_pixel],
                                                spherical_harmonics_im[idx_l][idx_m][id_pixel]);
                }

            }
        }


    }
/*================================================================================
    The multipoles .h5 data has the following layout: 

            Format: 
            multipoles.h5
            │-- /radius  (group)
            │-- /data  (group)
                │-- /data/time  (dataset, extensible, stores time entries)
                │-- /data/variable1  (group)
                │   │-- /data/variable1/0
                │   │   │-- /data/variable1/0/0/var_values (dataset, extensible)
                │   │   │-- /data/variable1/0/1/var_values (dataset, extensible)
                │   │-- /data/variable1/1
                │   │   │-- /data/variable1/1/-1/var_values (dataset, extensible)
                │   │   │-- /data/variable1/1/0/var_values (dataset, extensible)
                │   │   │-- /data/variable1/1/1/var_values (dataset, extensible)
                │-- /data/variable2  (group)
================================================================================*/


    void save_multipole_timeseries_hdf5_init(const std::string& abs_path,
                                             const double radius,
                                             const int spin_weight, 
                                             const int max_ell,
                                             std::set<std::string> vars_names){
        hid_t file_id, dspace_id, dset_id, group_id ;
        hid_t group_var_id, group_time_id, group_l_id, group_m_id;

        hsize_t size{1};
        herr_t h5err ;

        // the master rank takes care of HDF5 I/O
        // Here notice that it may not hold any information regarding
        // this particular surface. But it doesn't matter since we're
        // only writing its radius now.
        if ( parallel::mpi_comm_rank() == grace::master_rank() )
        {
            const char* fn = abs_path.c_str();
            assert(fn) ; 

            hid_t is_hdf5;
            H5E_BEGIN_TRY { is_hdf5 = H5Fopen(fn,
                                                H5F_ACC_RDWR,
                                                H5P_DEFAULT ) ; }
            H5E_END_TRY ;
            // if the file exists we have nothing to initialize
            if ( is_hdf5 >= 0)
            return ;
            
            file_id = H5Fcreate(fn, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            assert( file_id >= 0 );

            dspace_id = H5Screate_simple( 1, &size, nullptr);
            assert(dspace_id>=0);

            dset_id = H5Dcreate2( file_id, "/radius", H5T_NATIVE_DOUBLE, dspace_id,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
            assert(dset_id>=0);

            h5err = H5Dwrite( dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, &radius );
            assert(h5err>=0);

            h5err = H5Dclose(dset_id);
             assert(h5err>=0);
            
            h5err = H5Sclose(dspace_id);
            assert(h5err>=0);

            // create /data group
            group_id = H5Gcreate(file_id,"data",H5P_DEFAULT,H5P_DEFAULT,
                                H5P_DEFAULT);
            assert(group_id>=0);

            // Enable chunking property for time and individual /data/var/l/m datasets
            hid_t prop = H5Pcreate(H5P_DATASET_CREATE);
            hsize_t chunk_size = 10;
            H5Pset_chunk(prop, 1, &chunk_size);

            // create extensible /data/time dataset 
            // group_time_id = H5Gcreate(group_id,"time",H5P_DEFAULT,H5P_DEFAULT,
            //                     H5P_DEFAULT);
            // assert(group_time_id>=0);

            // settings for extensible datasets:
            hsize_t initial_size = 1, max_size = H5S_UNLIMITED;

            // Create /data/time as an extensible dataset
            //
            hid_t time_space = H5Screate_simple(1, &initial_size, &max_size);
            assert(time_space>=0);
            hid_t dset_time = H5Dcreate(group_id, "time", H5T_NATIVE_DOUBLE, time_space, H5P_DEFAULT, prop, H5P_DEFAULT);
            assert(dset_time>=0);
            h5eff=H5Dclose(dset_time);
            assert(h5err>=0);

            // create all /data/var/l/m groups
            for( const auto& var: vars_names ){
                        group_var_id = H5Gcreate(group_id, var.c_str() ,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
                        assert(group_var_id>=0);
                        for(int ell=spin_weight; ell<=max_ell; ell++){
                            group_l_id = H5Gcreate(group_var_id, std::to_string(ell).c_str(), H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
                            assert(group_l_id>=0);
                            for(int m=-ell; m<=ell; m++){
                                group_m_id = H5Gcreate(group_l_id, std::to_string(m).c_str() ,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
                                assert(group_m_id>=0);
                                // create /data/var/l/m/var_values datasets at the innermost level
                                hid_t var_space = H5Screate_simple(1, &initial_size, &max_size);
                                assert(var_space>=0);
                                // hid_t dset_var = H5Dcreate(group_m_id, "var_values", H5T_NATIVE_DOUBLE, var_space, H5P_DEFAULT, prop, H5P_DEFAULT);
                                // assert(dset_var>=0);
                                // create separate datasets for Real and Imag parts:
                                // real
                                hid_t dset_var = H5Dcreate(group_m_id, "Re", H5T_NATIVE_DOUBLE, var_space, H5P_DEFAULT, prop, H5P_DEFAULT);
                                assert(dset_var>=0);
                                h5err=H5Dclose(dset_var);
                                assert(h5err>=0);
                                // imag 
                                dset_var       = H5Dcreate(group_m_id, "Im", H5T_NATIVE_DOUBLE, var_space, H5P_DEFAULT, prop, H5P_DEFAULT);
                                assert(dset_var>=0);
                                h5err=H5Dclose(dset_var);
                                assert(h5err>=0);
                                // close m group
                                h5err=H5Gclose(group_m_id);
                                assert(h5err>=0);

                            }     
                            // close l group
                            h5err=H5Gclose(group_l_id);
                            assert(h5err>=0);

                        }
                        h5err=H5Gclose(group_var_id);
                        assert(h5err>=0);
  
            }

            // close properties
            h5err = H5Pclose(prop);
            assert(h5err>=0);
            // close /data group
            h5err = H5Gclose(group_id);
            assert(h5err>=0);
            // close file 
            h5err = H5Fclose(file_id);
            assert(h5err>=0);
        
        }

        return ; 

    }

    void save_multipole_timeseries_hdf5(const std::string& abs_path,
                                        const double spin_weight,
                                        const double max_ell,
                                        const std::set<std::string>& vars_names,
                                        const std::vector<double> &vars_vals_Re,
                                        const std::vector<double> &vars_vals_Im,
                                        const double current_time ){
        hid_t file_id, dspace_id, dset_id, group_id ;
        hid_t dgroup_id, dgroup_l_id, dgroup_m_id, dset_vars, dset_time;

        hsize_t size{1};
        herr_t h5err ;

    
        if ( parallel::mpi_comm_rank() == grace::master_rank() )
        {
            const char* fn = abs_path.c_str();
            assert(fn) ; 

            hid_t is_hdf5;
            H5E_BEGIN_TRY { is_hdf5 = H5Fopen(fn,
                                                H5F_ACC_RDWR,
                                                H5P_DEFAULT ) ; }
            H5E_END_TRY ;
            // file must already exist to write
            assert(is_hdf5 >= 0);
            // if successful, open:
            file_id = H5Fopen(fn,H5F_ACC_RDWR, H5P_DEFAULT ) ; assert(file_id>=0);

            // open /data group
            group_id = H5Gopen(file_id,"data",H5P_DEFAULT,H5P_DEFAULT,
                                H5P_DEFAULT);
            assert(group_id>=0);

            // append current time to the  extensible /data/time dataset 
            // Open dataset
            
            // restrict the scope for clarity 
            {
                hid_t dset_var = H5Dopen(group_id, "time", H5P_DEFAULT);
                assert(dset_var >= 0);

                // Get current dataset size
                hsize_t dims[1]; 
                H5Dget_space(dset_var);
                H5Dget_storage_size(dset_var);
                hsize_t new_size = dims[0] + 1;  // Increase by 1 time step

                // Extend dataset to new size
                H5Dset_extent(dset_var, &new_size);

                // Define memory space for new data
                hid_t mem_space = H5Screate_simple(1, &new_size, NULL);
                assert(mem_space >= 0);

                // Select the new data location in dataset
                hid_t file_space = H5Dget_space(dset_var);
                hsize_t start[1] = {dims[0]};  // Append at the end
                hsize_t count[1] = {1};
                H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);

                // Write new data
                double new_value = current_time;  // new_value to be written 
                H5Dwrite(dset_var, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, &new_value);

                // Cleanup
                h5err=H5Sclose(mem_space);assert(h5err>=0);
                h5err=H5Sclose(file_space);assert(h5err>=0);
                h5err=H5Dclose(dset_var);assert(h5err>=0);
            }

            // now time for all the multipole values: 
            // open all /data/var/l/m groups and append the latest vales to respective datasets
            int counter=0;
            for( const auto& var: vars_names){
                        group_var_id = H5Gopen(group_id, var.c_str() ,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
                        assert(group_var_id>=0);
                        for(int ell=spin_weight; ell<=max_ell; ell++){
                            group_l_id = H5Gopen(group_var_id, std::to_string(ell).c_str(), H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
                            assert(group_l_id>=0);
                            for(int m=-ell; m<=ell; m++){
                                group_m_id = H5Gopen(group_l_id, std::to_string(m).c_str() ,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
                                assert(group_m_id>=0);
                                // create /data/var/l/m/var_values datasets at the innermost level
                                // hid_t var_space = H5Screate_simple(1, &initial_size, &max_size);
                                // assert(var_space>=0);
                                // hid_t dset_var = H5Dcreate(group_m_id, "values", H5T_NATIVE_DOUBLE, time_space, H5P_DEFAULT, prop, H5P_DEFAULT);
                                // assert(dset_var>=0);
                                // append current time to the  extensible /data/time dataset 
                                // Open dataset
                                dset_var = H5Dopen(group_m_id, "values", H5P_DEFAULT);
                                assert(dset_var >= 0);

                                // Get current dataset size
                                hsize_t dims[1]; 
                                H5Dget_space(dset_var);
                                H5Dget_storage_size(dset_var);
                                hsize_t new_size = dims[0] + 1;  // Increase by 1 time step

                                // Extend dataset to new size
                                H5Dset_extent(dset_var, &new_size);

                                // Define memory space for new data
                                hid_t mem_space = H5Screate_simple(1, &new_size, NULL);
                                assert(mem_space >= 0);

                                // Select the new data location in dataset
                                hid_t file_space = H5Dget_space(dset_var);
                                hsize_t start[1] = {dims[0]};  // Append at the end
                                hsize_t count[1] = {1};
                                H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);

                                auto multipole_index = [=](const int idx_l, const int idx_m ){return idx_l * (2 * max_ell + 1) + idx_m;};

                                // Write new data
                                double new_value = vars_vals[counter];  // new_value to be written 
                                H5Dwrite(dset_var, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, &new_value);

                                // Cleanup
                                h5err=H5Sclose(mem_space);assert(h5err>=0);
                                h5err=H5Sclose(file_space);assert(h5err>=0);
                                h5err=H5Dclose(dset_var);assert(h5err>=0);
                                h5err=H5Gclose(group_m_id);assert(h5err>=0);

                            }     
                            h5err=H5Gclose(group_l_id);
                            assert(h5err>=0);

                        }
                        h5err=H5Gclose(group_var_id);
                        assert(h5err>=0);       
                counter++;
            }





            h5err = H5Gclose(group_id) ;assert(h5err>=0);

            h5err = H5Fclose(file_id) ;assert(h5err>=0);
            
        }

        return ; 

    }

        
    void write_multipole_timeseries(){

        auto& runtime = grace::runtime::get( ) ;
        const auto comm = parallel::get_comm_world() ; 
        const int rank = parallel::mpi_comm_rank()  ; 
        const int world_size = parallel::mpi_comm_size()  ;
        const double current_time =  grace::get_simulation_time() ;
        const int current_iteration = grace::get_iteration() ;

        const int n_detectors = runtime.n_surface_output_spheres();
        const int nside = runtime.nside_surface_output_spheres();
        const int max_ell = runtime.max_degree_multipoles_surface_output_spheres();

        // note this is hard-coded because we are only interested in
        // GW extraction at the moment
        // in the future, generic variables (Poynting flux, ang. momentum fluxes, etc)
        // will have to employ a similar philosophy with weight=0
        constexpr const int spin_weight=2;

        // the spin weights in that case will be provided at runtime:
        //const std::vector<int> spin_weights = runtime.multipole_spin_weights();

        if(spherical_harmonics_re.size()==0 && spherical_harmonics_im.size()==0){
            initialize_spherical_harmonics(spin_weight, max_ell, nside);
        }

        const std::set<std::string> corner_scalar_vars = runtime.corner_sphere_surface_multipole_output_scalar_vars();
        const std::set<std::string> corner_vector_vars = runtime.corner_sphere_surface_multipole_output_vector_vars();
        const std::set<std::string> corner_tensor_vars = runtime.corner_sphere_surface_multipole_output_tensor_vars();
        const std::set<std::string> cell_scalar_vars = runtime.cell_sphere_surface_multipole_output_scalar_vars();
        const std::set<std::string> cell_vector_vars = runtime.cell_sphere_surface_multipole_output_vector_vars();
        const std::set<std::string> cell_tensor_vars = runtime.cell_sphere_surface_multipole_output_tensor_vars();
        
    

        update_spherical_detectors();

        GRACE_VERBOSE("Updated spherical surfaces info - multipole computation.") ; 

        compute_spherical_surface_variable_data(corner_scalar_vars,corner_vector_vars,corner_tensor_vars,
                                                cell_scalar_vars,  cell_vector_vars,  cell_tensor_vars );

        GRACE_VERBOSE("Interpolated variables on spherical surfaces for multipole decomposition.") ; 

        // IF MODE == SERIAL_WRITING 
        int detector_counter = 0;  // Initialize counter
        // this loop automatically omit ranks that do not have a detector assigned (e.g. no coordinate overlap)
        // loop over detectors 
        for (auto& [name, detector] : detectors) {
            std::vector<int> det_healpix_indices = detector.get_local_rank_healpix_indices();
            std::map<std::string,std::vector<double>> det_surface_data = detector.get_local_rank_detector_surface_data();

            std::filesystem::path base_path (runtime.surface_io_basepath()) ;
            const std::string filename =  "./multipoles_det" + name + "_surf.h5";
            //const std::string filename =  "./multipoles_det" + name + "_spin_weight_" + str::to_string(spin_weight) + ".h5";
            std::filesystem::path out_path = base_path / filename ;
            // Resolve to absolute path
            std::filesystem::path absolute_path = std::filesystem::absolute(out_path.lexically_normal());

      
            auto multipole_index = [=](const int idx_l, const int idx_m ){return idx_l * (2 * max_ell + 1) + idx_m;};

            std::set<std::string, std::vector<double>> det_all_multipoles_re; 
            std::set<std::string, std::vector<double>> det_all_multipoles_im; 

            // here we do a rather dirty(!) trick 
            // since Psi4 is a complex variable, and we currently do not natively support complex fields, 
            // we check if the field is complex/has a conjugate counterpart 
            // just by searching for a substring (yuck!) of a variable name in det_surface_data
            // i.e. we know Psi4Re has a counterpart Psi4Im
            // and we then combine the 
            
            // the scalar product on S_2 reads: 
            // <f, s_Y_lm> = int_S2 conj(f) s_Y_lm * sinth * dth * dph
            // where conj(f) = Real(f) - i Imag(f)


            // loop over variables scheduled for multipole decomposition:
            for ( auto& [var_name, var_data] : det_surface_data) {  
                // local rank operations : 
                // partial sum arrays for the scalar product with each spherical harmonic:
                
                std::vector<double> local_scalar_products_re(max_ell * (2 * max_ell + 1), 0.0);
                std::vector<double> local_scalar_products_im(max_ell * (2 * max_ell + 1), 0.0);

                std::vector<double> global_scalar_products_re(max_ell * (2 * max_ell + 1), 0.0);
                std::vector<double> global_scalar_products_im(max_ell * (2 * max_ell + 1), 0.0);

                // lower ell than spin weight make no sense, clearly
                // TO DO: parallelize this loop 


                for(int idx_l=spin_weight; idx_l<max_ell; idx_l++){
                    for(int idx_m=0; idx_m <= 2*idx_l; idx_m++){

                        int idx_multipole = multipole_index(idx_l, idx_m);

                        local_scalar_products_re[idx_multipole] = 0.;
                        local_scalar_products_im[idx_multipole] = 0.;
                        for(int idx_pix=0; idx_pix < det_healpix_indices.size(); idx_pix++) {
                            const int pixel_index = det_healpix_indices[idx_pix];
                            local_scalar_products_re[idx_multipole] += var_data[pixel_index] * spherical_harmonics_re[idx_l][idx_m][pixel_index] ;
                            local_scalar_products_im[idx_multipole] += -var_data[pixel_index] * spherical_harmonics_im[idx_l][idx_m][pixel_index] ;
                        }
                    }
                }

                /*
                * At this point, all ranks have their local partial sums for (l>=s,m) 
                * We perform an MPI sum (harmonic-wise), and save on root 
                */
                
                parallel::mpi_reduce(local_scalar_products_re.data(), 
                                    global_scalar_products_re.data(),
                                    local_scalar_products_re.size(),
                                    mpi_sum,
                                    grace::master_rank()
                                    );
                parallel::mpi_reduce(local_scalar_products_im.data(), 
                                    global_scalar_products_im.data(),
                                    local_scalar_products_im.size(),
                                    mpi_sum,
                                    grace::master_rank()
                                    );
                /*
                *  Multiply by the healpix measure
                *  dA = 4 * pi / (12 NSIDE^2) 
                */

                const double dA = 4. * M_PI / ( 12 * nside * nside );
                
                std::for_each(global_scalar_products_re.begin(), global_scalar_products_re.end(),
                                [&](double& pt_val){ pt_val *= dA  ;});

                std::for_each(global_scalar_products_im.begin(), global_scalar_products_im.end(),
                                [&](double& pt_val){ pt_val *= dA ;});

                det_all_multipoles_re.insert({global_scalar_products_re});
                det_all_multipoles_im.insert({global_scalar_products_im});
            }

        }
                detector_counter++;
        
        GRACE_VERBOSE("Saved multipole decomposition data.") ; 

        }
                                 



  }
}

