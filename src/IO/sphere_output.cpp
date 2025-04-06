

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

// this fixes the compilation error I had, but ideally we'd fix the visibility of workspaces in a different way
#include <grace/errors/error.hh> 

namespace grace {
namespace utils {
    using ::utils::make_string;
}
}


namespace grace { namespace IO {

    using namespace healpix;
    // namespace utils = grace::utils;
    std::map<std::string, spherical_detector> detectors;

    // access pattern to a s_Y_lm (pixel_id) spherical harmonic reads:
    // spherical_harmonics[ell][m][id_pixel]
    std::vector<std::vector<std::vector<double>>> spherical_harmonics_re; 
    std::vector<std::vector<std::vector<double>>> spherical_harmonics_im; 
        
    //TODO:
    using Complex = Kokkos::complex<double>;
    using HostM   = Kokkos::HostSpace;

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
                                        const int ntheta,
                                        const int nphi, 
                                        const std::vector<std::array<double,3>> output_spheres_centres,
                                        const std::vector<double> output_spheres_radii,
                                        const std::vector<std::string> output_spheres_names,
                                        const std::vector<std::string> output_spheres_types
                                        ){

            // construct the string-detector map
            for(size_t id_det=0; id_det<n_detectors; id_det++){
                                    if(output_spheres_types[id_det]=="healpix"){
                                        detectors.emplace(output_spheres_names[id_det], healpix::spherical_detector(nside, output_spheres_radii[id_det], output_spheres_centres[id_det]));
                                    }
                                    else if(output_spheres_types[id_det]=="uniform"){
                                         detectors.emplace(output_spheres_names[id_det], healpix::spherical_detector(ntheta, nphi, output_spheres_radii[id_det], output_spheres_centres[id_det]));
                                    }
                                    else{
                                        //ERROR("Grid decomposition type of sphere no. " << std::to_string(id_det).c_str() << " not recognized.");
                                        ERROR("Grid decomposition type of sphere no not recognized.");
                                        
                                    }
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
                                                   INTERPOLATION_METHOD::LINEAR // this should be steerable by the parfile
                                                   //INTERPOLATION_METHOD::LAGRANGE3
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
        const int ntheta = runtime.ntheta_surface_output_spheres();
        const int nphi = runtime.nphi_surface_output_spheres();
        const std::vector<std::array<double,3>> output_spheres_centres  = runtime.cell_sphere_surface_output_centers();
        const std::vector<double> output_spheres_radii                  = runtime.cell_sphere_surface_output_radii()  ;
        const std::vector<std::string> output_spheres_names             = runtime.cell_sphere_surface_output_names();
        const std::vector<std::string> output_spheres_types             = runtime.cell_sphere_surface_output_types();
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
                                                ntheta,nphi,
                                                output_spheres_centres,
                                                output_spheres_radii,
                                                output_spheres_names,
                                                output_spheres_types ) ;
            }

            for (auto& [name, detector] : detectors) {

                // skip healpix output for non-healpix detectors 
                if(detector.grid_type!=SPHERICAL_GRID_TYPE::HEALPIX) continue;

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
            // skip healpix output for non-healpix detectors 
            if( detector.grid_type != SPHERICAL_GRID_TYPE::HEALPIX ) continue;

            std::vector<int> det_healpix_indices = detector.get_local_rank_sphere_indices();
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



    void update_spin_weighted_spherical_harmonics(Kokkos::View<Complex**, HostM>& sw_sph_harmonics, const int spin_weight, const int max_ell, const int ntheta, const int nphi){
        // we will have #spin_weight many unused arrays (of interior sizes 0)
        // e.g. spin_weight=2 means l=0 and l=1 are redundant, but
        // it's more convenient to keep the indexing clean
        // also, it's more convenient for indexing to 
        const int size_harmonics_vecspace = (max_ell + 1) * (max_ell + 1);
        const int size_det_indices        = (ntheta+1) * (nphi+1);

        Kokkos::View<Complex**, HostM> updated_spherical_harmonics("SPSH", size_harmonics_vecspace, size_det_indices);

        for (int idx_l = math::abs(spin_weight); idx_l <= max_ell; idx_l++) {
            // e.g. for s=0 SWSH: idx_l=0, ell=0 
            int ell = idx_l;  
            for (int idx_m = 0; idx_m <= 2*idx_l; idx_m++) {
                int m = -ell + idx_m ; // the physical index is m = -ell + idx_m 

                const int idx_multipole = utils::multipole_index(ell, m);  // 

                for (int idx_pixel = 0; idx_pixel < size_det_indices; idx_pixel++){
                    double th, ph;
                    get_spherical_coord_from_oned_index(ntheta,nphi, idx_pixel, th, ph);

                    utils::multipole_spherical_harmonic(spin_weight, ell, m,
                                                 th, ph,
                                                updated_spherical_harmonics(idx_multipole, idx_pixel).real(),
                                                updated_spherical_harmonics(idx_multipole, idx_pixel).imag()
                                                );
                }

            }
        }

        sw_sph_harmonics = updated_spherical_harmonics;
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
                │   │   │-- /data/variable1/0/0/ (group)
                │   │   |   │-- /data/variable1/0/0/Re (dataset, extensible)
                │   │   |   │-- /data/variable1/0/0/Im (dataset, extensible)
                │   │   │-- /data/variable1/0/1/ (group)
                |   |   | ... ... ... ... ... ... ... 
                │   │-- /data/variable1/1
                │   │   │-- /data/variable1/1/-1/(group)
                │   │   │-- /data/variable1/1/0/ (group)
                │   │   │-- /data/variable1/1/1/ (group)
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
            hsize_t initial_size = 0, max_size = H5S_UNLIMITED;

            // Create /data/time as an extensible dataset
            //
            hid_t time_space = H5Screate_simple(1, &initial_size, &max_size);
            assert(time_space>=0);
            hid_t dset_time = H5Dcreate(group_id, "time", H5T_NATIVE_DOUBLE, time_space, H5P_DEFAULT, prop, H5P_DEFAULT);
            assert(dset_time>=0);
            h5err=H5Dclose(dset_time);
            assert(h5err>=0);

            //create all /data/var/l/m groups
            for( const auto& var: vars_names ){
                        group_var_id = H5Gcreate(group_id, var.c_str() ,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
                        assert(group_var_id>=0);
                        for(int ell=math::abs(spin_weight); ell<=max_ell; ell++){
                            group_l_id = H5Gcreate(group_var_id, std::to_string(ell).c_str(), H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
                            assert(group_l_id>=0);
                            for(int m=-ell; m<=ell; m++){
                                group_m_id = H5Gcreate(group_l_id, std::to_string(m).c_str() ,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
                                assert(group_m_id>=0);
                                // create /data/var/l/m/var_values datasets at the innermost level
                                hid_t var_space = H5Screate_simple(1, &initial_size, &max_size);
                                assert(var_space>=0);
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
                                        const std::map<std::string, Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>> all_multipoles, 
                                        const double current_time ){
        hid_t file_id, dspace_id, dset_id, group_id ;
        hid_t dgroup_id, group_var_id, group_l_id, group_m_id, dset_var, dset_time;

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
            group_id = H5Gopen(file_id, "data" ,H5P_DEFAULT);
            assert(group_id>=0);

            // append current time to the  extensible /data/time dataset 
            // Get current dataset size
            {
                dset_var = H5Dopen(group_id, "time", H5P_DEFAULT);
                assert(dset_var >= 0);

                hsize_t dims[1];
                hid_t file_space = H5Dget_space(dset_var);
                H5Sget_simple_extent_dims(file_space, dims, NULL);  // Retrieve current size

                hsize_t new_size = dims[0] + 1;  // Increase by 1 time step
                H5Dset_extent(dset_var, &new_size);  // Extend dataset

                // Define memory space for just the new value
                hsize_t mem_dims[1] = {1};  
                hid_t mem_space = H5Screate_simple(1, mem_dims, NULL);
                assert(mem_space >= 0);

                // Reopen dataset space after extending
                file_space = H5Dget_space(dset_var);

                // Select hyperslab for the new data
                hsize_t start[1] = {dims[0]};  // Append at the end
                hsize_t count[1] = {1};
                H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);

                // Write new data
                double new_value = current_time;
                H5Dwrite(dset_var, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, &new_value);

                // Cleanup
                H5Sclose(mem_space);
                H5Sclose(file_space);
                H5Dclose(dset_var);
            }
            // now time for all the multipole values: 
            // open all /data/var/l/m groups and append the latest vales to respective datasets
            int counter=0;
            for( const auto& var_name: vars_names){
                        group_var_id = H5Gopen(group_id, var_name.c_str() ,H5P_DEFAULT);
                        assert(group_var_id>=0);
                        for(int ell=math::abs(spin_weight); ell<=max_ell; ell++){
                            group_l_id = H5Gopen(group_var_id, std::to_string(ell).c_str(), H5P_DEFAULT);
                            assert(group_l_id>=0);
                            for(int m=-ell; m<=ell; m++){
                                //int idx_multipole = utils::multipole_index(ell, m+ell); // m+ell because we need to go from 0 to 2*ell+1 in the indices
                                const int idx_multipole = utils::multipole_index(ell, m);  // 

                                group_m_id = H5Gopen(group_l_id, std::to_string(m).c_str() ,H5P_DEFAULT);
                                assert(group_m_id>=0);
                                // create /data/var/l/m/var_values datasets at the innermost level
                                         
                                // Real part 
                                {
                                    dset_var = H5Dopen(group_m_id, "Re", H5P_DEFAULT);
                                    assert(dset_var >= 0);

                                    hsize_t dims[1];
                                    hid_t file_space = H5Dget_space(dset_var);
                                    H5Sget_simple_extent_dims(file_space, dims, NULL);  // Retrieve current size

                                    hsize_t new_size = dims[0] + 1;  // Increase by 1 time step
                                    H5Dset_extent(dset_var, &new_size);  // Extend dataset

                                    // Define memory space for just the new value
                                    hsize_t mem_dims[1] = {1};  
                                    hid_t mem_space = H5Screate_simple(1, mem_dims, NULL);
                                    assert(mem_space >= 0);

                                    // Reopen dataset space after extending
                                    file_space = H5Dget_space(dset_var);

                                    // Select hyperslab for the new data
                                    hsize_t start[1] = {dims[0]};  // Append at the end
                                    hsize_t count[1] = {1};
                                    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);

                                    // Write new data
                                    double new_value = current_time;
                                    H5Dwrite(dset_var, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, &all_multipoles.at(var_name)(idx_multipole).real());

                                    // Cleanup
                                    H5Sclose(mem_space);
                                    H5Sclose(file_space);
                                    H5Dclose(dset_var);
                                }

                                // Imaginary part 
                                {
                                    dset_var = H5Dopen(group_m_id, "Im", H5P_DEFAULT);
                                    assert(dset_var >= 0);

                                    hsize_t dims[1];
                                    hid_t file_space = H5Dget_space(dset_var);
                                    H5Sget_simple_extent_dims(file_space, dims, NULL);  // Retrieve current size

                                    hsize_t new_size = dims[0] + 1;  // Increase by 1 time step
                                    H5Dset_extent(dset_var, &new_size);  // Extend dataset

                                    // Define memory space for just the new value
                                    hsize_t mem_dims[1] = {1};  
                                    hid_t mem_space = H5Screate_simple(1, mem_dims, NULL);
                                    assert(mem_space >= 0);

                                    // Reopen dataset space after extending
                                    file_space = H5Dget_space(dset_var);

                                    // Select hyperslab for the new data
                                    hsize_t start[1] = {dims[0]};  // Append at the end
                                    hsize_t count[1] = {1};
                                    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);

                                    // Write new data
                                    double new_value = current_time;
                                    H5Dwrite(dset_var, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, &all_multipoles.at(var_name)(idx_multipole).imag());

                                    // Cleanup
                                    H5Sclose(mem_space);
                                    H5Sclose(file_space);
                                    H5Dclose(dset_var);
                                }

                                // close m group
                                h5err=H5Gclose(group_m_id);assert(h5err>=0);

                            }     
                            h5err=H5Gclose(group_l_id);assert(h5err>=0);

                        }
                        h5err=H5Gclose(group_var_id);assert(h5err>=0);       
                counter++;
            }

            h5err = H5Gclose(group_id) ;assert(h5err>=0);

            h5err = H5Fclose(file_id) ;assert(h5err>=0);
            
        }

        return ; 

    }




    void save_multipole_timeseries_ascii(const std::string& parent_path,
                                        const double radius,
                                        const int spin_weight,
                                        const int max_ell,
                                        const std::set<std::string>& vars_names,
                                        const std::map<std::string, Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>> all_multipoles, 
                                        const double iter,
                                        const double current_time ){
    
        if ( parallel::mpi_comm_rank() == grace::master_rank() )
        {

            // now time for all the multipole values: 
            // open all /data/var/l/m groups and append the latest vales to respective datasets
            int counter=0;
            for( const auto& var_name: vars_names){
                for(int ell=math::abs(spin_weight); ell<=max_ell; ell++){
                    for(int m=-ell; m<=ell; m++){
                        //int idx_multipole = utils::multipole_index(ell, m+ell); // m+ell because we need to go from 0 to 2*ell+1 in the indices
                        const int idx_multipole = utils::multipole_index(ell, m);  // 

                        const std::string filename= ( parent_path+\
                                            "_" + var_name +\
                                            "_spin_weight_"+std::to_string(spin_weight)+\
                                            "_l_"+std::to_string(ell)+\
                                            "_m_"+std::to_string(m)+\
                                            ".ascii"  );

                    if (not std::filesystem::exists(filename)) {
                        std::ofstream outfile(filename,std::ios::app) ; 
                        outfile << std::setprecision(15) << std::defaultfloat ; 
                        outfile << std::left << "# detector at radius = " << std::setprecision(2) << radius << std::setprecision(15) << '\n'
                                << std::left << "# spin weight = " <<  spin_weight << '\n' 
                                << std::left << "# (l,m) = " <<  " (" << ell << "," << m << ")"  << '\n' 
                                << std::left << "# for variable = " << var_name  << '\n' 
                                << std::left << "# iter time Re Im " << '\n' ;
                    }
                    //else{
                        std::ofstream outfile(filename,std::ios::app) ; 
                        outfile << std::setprecision(15) << std::defaultfloat ; 
                        outfile << std::left << iter << '\t'
                                << std::left << current_time << '\t' 
                                << std::left << all_multipoles.at(var_name)(idx_multipole).real() << '\t'  
                                << std::left << all_multipoles.at(var_name)(idx_multipole).imag() << '\n' ; 
                    //}  
                    }
                }
            }
        }
        return ; 
    }




    /**
     * @brief write_multipole_timeseries
     * this function schedules the interpolation of variables registered for multipole computation,
     * computes the multipole decomposition and saves to HDF5 files
     * @warning the simulation must end GRACEfully [ ;-) ] for the HDF5 files to be readable; otherwise, the 
     * data becomes corrupted
     * @todo trivially extendible - loop over spin_weights (essentially just 2/3 interesting ones: s=-2,0,1) 
     * with their respective separate output 
     */

    // note: we currently do not use the HDF5 output, which also has a memory leak somewhere inside...                    

    std::map<std::string, View<Complex*,HostM>> 
    get_all_multipoles(const int spin_weight, 
                       const int max_ell,
                       const int ntheta,
                       const int nphi, 
                    //    const int nside,
                       const std::vector<int>& det_indices,
                       const View<Complex**, HostM>& sw_sph_harmonics,
                       const std::map< std::string, View<Complex*, HostM>>& complex_det_surface_data)
                    {
                        std::map<std::string, View<Complex*,HostM>> det_all_multipoles; 

                        for ( auto& [var_name, var_data] : complex_det_surface_data) {  
                                // local rank operations : 
                                // partial sum arrays for the scalar product with each spherical harmonic:
                                // we store them in this way and not as a complex Kokkos::View, 
                                // since we invoke mpi_reduce on MPI_DOUBLE type... 
                                std::vector<double> local_scalar_products_re((max_ell+1)*(max_ell+1), 0.0);
                                std::vector<double> local_scalar_products_im((max_ell+1)*(max_ell+1), 0.0);

                                std::vector<double> global_scalar_products_re((max_ell+1)*(max_ell+1), 0.0);
                                std::vector<double> global_scalar_products_im((max_ell+1)*(max_ell+1), 0.0);

                                // lower ell than spin weight make no sense, clearly
                                // TO DO: parallelize this loop 

                                for(int idx_l=math::abs(spin_weight); idx_l<=max_ell; idx_l++){
                                    for(int idx_m=0; idx_m <= 2*idx_l; idx_m++){

                                        const int ell = idx_l;
                                        const int m   = idx_m-ell; 
                                        // int idx_multipole = utils::multipole_index(idx_l, idx_m);
                                        const int idx_multipole = utils::multipole_index(ell, m);  // 

                                        for(int idx_pix=0; idx_pix < det_indices.size(); idx_pix++) {
                                            const int pixel_index = det_indices[idx_pix];


                                            double theta, phi; 

                                            get_spherical_coord_from_oned_index(ntheta, nphi, pixel_index,
                                                                                theta, phi);
                                            double sinth = std::sin(theta);

                                            // notabene:
                                            // idx_pix indexes the extent of the var_data
                                            // pixel_index is the 'true/physical' index, i.e. location according to the healpix convention

                                            // therefore, in the sum below, the `idx_pix'-th entry [for var_data] corresponds 
                                            // to the pixel_index location [for sw_sph_harmonics]

                                            // conj(F) * Y_lm = 
                                            //    Real(F)*Real(Y_lm) - Imag(F)*Imag(Y_lm)
                                            //+i*(Imag(F)*Real(Y_lm) + Real(F)*Imag(Y_lm))  

                                            Complex conjugate_var_data(var_data(idx_pix).real(), -var_data(idx_pix).imag());


                                            local_scalar_products_re[idx_multipole] 
                                                                //+=(var_data(pixel_index)*sw_sph_harmonics(idx_multipole,pixel_index)).real();
                                                                //+=(var_data(idx_pix)*sw_sph_harmonics(idx_multipole,pixel_index)).real();
                                                                += sinth * (conjugate_var_data*sw_sph_harmonics(idx_multipole,pixel_index)).real();
                                                                // += var_data(pixel_index).real() * spherical_harmonics_re[idx_l][idx_m][pixel_index]\
                                                                //   -var_data(pixel_index).imag() * spherical_harmonics_im[idx_l][idx_m][pixel_index];
                                            local_scalar_products_im[idx_multipole] 
                                                                //+=(var_data(pixel_index)*sw_sph_harmonics(idx_multipole,pixel_index)).imag();
                                                                //+=(var_data(idx_pix)*sw_sph_harmonics(idx_multipole,pixel_index)).imag();
                                                                += sinth * (conjugate_var_data*sw_sph_harmonics(idx_multipole,pixel_index)).imag();
                                                                // += var_data(pixel_index).imag() * spherical_harmonics_re[idx_l][idx_m][pixel_index]\ 
                                                                //   +var_data(pixel_index).real() * spherical_harmonics_im[idx_l][idx_m][pixel_index] ;
                                            
                                            // if(idx_multipole==6){
                                            //     printf("conjugate data on the %d th pixel: (%f,%f) \n ", conjugate_var_data.real(), conjugate_var_data.imag() );
                                                
                                            //     printf("6th harmonic: pixel idx, val: (%d, %f, %f) \n",pixel_index,
                                            //         sw_sph_harmonics(idx_multipole,pixel_index).real(),
                                            //         sw_sph_harmonics(idx_multipole,pixel_index).imag()  );
                                            // }

                                        }
                                    }
                                }

                    

                                /*
                                * At this point, all ranks have their local partial sums for (l>=s,m) 
                                * We perform an MPI sum (harmonic-wise), and save on root 
                                * Note : maybe use allreduce instead? 
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

                                // const double dA = 4. * M_PI / ( 12 * nside * nside );
                                const double dA = (M_PI / (ntheta + 1)) * ((2.*M_PI / (nphi + 1))) ;
                                
                                std::for_each(global_scalar_products_re.begin(), global_scalar_products_re.end(),
                                                [&](double& pt_val){ pt_val *= dA  ;});

                                std::for_each(global_scalar_products_im.begin(), global_scalar_products_im.end(),
                                                [&](double& pt_val){ pt_val *= dA ;});

                                Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace> single_var_multipoles("var_multipoles", (max_ell+1)*(max_ell+1) );
                                for(int idx_l=math::abs(spin_weight); idx_l<=max_ell; idx_l++){
                                    for(int idx_m=0; idx_m <= 2*idx_l; idx_m++){
                                        const int ell = idx_l;
                                        const int m   = idx_m-ell; 
                                        const int idx_multipole = utils::multipole_index(ell, m);  // 

                                        // finally, assign to the Kokkos::View...
                                        single_var_multipoles(idx_multipole) = Kokkos::complex<double>(global_scalar_products_re[idx_multipole],
                                                                                                    global_scalar_products_im[idx_multipole]);
                                    }
                                }
                                // operator[] for inserting a map pair:
                                det_all_multipoles[var_name] = single_var_multipoles;


                            }

                    return det_all_multipoles;
                }


    std::map<std::string, View<Complex*, HostM>> 
    complexify_detector_data(std::map<std::string,std::vector<double>> const& det_surface_data){

            std::map<std::string, View<Complex*, HostM>> complex_det_surface_data;
            std::set<std::string> all_var_names;    // has Psi4Re, Psi4Im
            std::set<std::string> unique_var_names; // complex ---> has only Psi4 
            std::map<std::string,bool> is_variable_complex; // keep track of whether the variable is complex... 
            int common_var_size;  
            for ( auto& [var_name, var_data] : det_surface_data) {
                all_var_names.insert(var_name);
                common_var_size=var_data.size(); // each one of the fields will have the same no. of entries
            }

            // fixed suffix size - if we e.g. decide to switch to Real and Imag in the future, we will need to change this:
            constexpr int suffix_size = sizeof("Re") - 1; 

            // we look if a variable has a name of the type XYZRe and if an equivalent XYZIm exists as well:
            for ( auto const& var_name : all_var_names) {
                // complicated syntax:
                if(var_name.size() > 1 &&                // if the var name is longer than 1 (clearly...)
                   all_var_names.count(var_name.substr(0, var_name.size() - suffix_size)+ "Re") && // if a key exists that has some root and Re at the end...
                   all_var_names.count(var_name.substr(0, var_name.size() - suffix_size)+ "Im"))  // and a key exists that shares the root but with Im at the end...
                    {   // then the variable is complex! 
                        // we can append the common root of the name:
                        std::string complex_name = var_name.substr(0, var_name.size() - suffix_size);
                        unique_var_names.insert(complex_name);
                        is_variable_complex[complex_name] = true; 
                    }
                    // note that since unique_var_names is a set, it will not insert Psi4 twice! great!
                else{  
                    unique_var_names.insert(var_name);
                    is_variable_complex[var_name] = false; 
                }
            }

            for ( auto const& var_name : unique_var_names) {
                Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace> complex_field("complex_var", common_var_size);
                
                for(int idx=0; idx<common_var_size;idx++){ 
                    if(!is_variable_complex[var_name]) { // variable purely real
                        complex_field(idx).real() =  det_surface_data.at(var_name)[idx];
                        complex_field(idx).imag() =  0.0;
                    }
                    else{ // complex variable 
                        complex_field(idx).real() =  det_surface_data.at(var_name+"Re")[idx];
                        complex_field(idx).imag() =  det_surface_data.at(var_name+"Im")[idx];
                    }
                }
                    complex_det_surface_data[var_name] = complex_field;  // assign to std::map finally 
            }  
        return complex_det_surface_data;
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
        const int ntheta = runtime.ntheta_surface_output_spheres();
        const int nphi = runtime.nphi_surface_output_spheres();
        const std::vector<std::array<double,3>> output_spheres_centres  = runtime.cell_sphere_surface_output_centers();
        const std::vector<double> output_spheres_radii                  = runtime.cell_sphere_surface_output_radii()  ;
        const std::vector<std::string> output_spheres_names             = runtime.cell_sphere_surface_output_names();
        const std::vector<std::string> output_spheres_types             = runtime.cell_sphere_surface_output_types();
       
        const int max_ell = runtime.max_degree_multipoles_surface_output_spheres();

        if(n_detectors!=0){
            // if not yet initialized
            if(detectors.empty()){
                GRACE_INFO("Initializing spherical surfaces.") ; 
                GRACE_VERBOSE( "There's {} detectors in total" , n_detectors) ; 

                initialize_spherical_detectors(n_detectors, nside, ntheta, nphi,
                                                output_spheres_centres,
                                                output_spheres_radii,
                                                output_spheres_names,
                                                output_spheres_types ) ;
            }
        }
        // note this is hard-coded because we are only interested in
        // GW extraction at the moment
        // in the future, generic variables (Poynting flux, ang. momentum fluxes, etc)
        // will have to employ a similar philosophy with weight=0
        // trivial to extend 

        constexpr const int spin_weight=-2;

        // the spin weights in such a case will be provided at runtime:
        //const std::vector<int> spin_weights = runtime.multipole_spin_weights();

        // if (sw_sph_harmonics.data()) {
         // If needed, explicitly deallocate or clear before reallocating
        Kokkos::View<Complex**, HostM> sw_sph_harmonics; //("spin_weighted_spherical_harmonics",1,1);

        sw_sph_harmonics = Kokkos::View<Complex**, HostM>();
        // }

        if(sw_sph_harmonics.extent(0)==0 and sw_sph_harmonics.extent(1)==0){  // if data=0
                //update_spin_weighted_spherical_harmonics(sw_sph_harmonics, spin_weight, max_ell, nside); // multipole output not adapted for HEALPIX convention
                update_spin_weighted_spherical_harmonics(sw_sph_harmonics, spin_weight, max_ell, ntheta, nphi);
                GRACE_VERBOSE("SWSH updated. Current size: {} x {}.", sw_sph_harmonics.extent(0),sw_sph_harmonics.extent(1) ) ; 
        }
        
        GRACE_VERBOSE("SWSH should be initialized. Current size: {} x {}.", sw_sph_harmonics.extent(0),sw_sph_harmonics.extent(1) ) ; 


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

            // skip detectors of HEALPIX type - we'd need proper integration weights 
            if( detector.grid_type != SPHERICAL_GRID_TYPE::UNIFORM ) continue;

            std::vector<int> det_indices = detector.get_local_rank_sphere_indices();
            std::map<std::string,std::vector<double>> det_surface_data = detector.get_local_rank_detector_surface_data();

            // here we do a rather dirty(!) trick 
            // since Psi4 is a complex variable, and we currently do not natively support complex fields, 
            // we check if the field is complex/has a conjugate counterpart 
            // just by searching for a substring (yuck!) of a variable name in det_surface_data
            // i.e. we know Psi4Re has a counterpart Psi4Im
            // and we then combine the 

            // the scalar product on S_2 reads: 
            // <f, s_Y_lm> = int_S2 conj(f) s_Y_lm * sinth * dth * dph
            // where conj(f) = Real(f) - i Imag(f)
            
            // access pattern:
            // complex_det_surface_data[var_name](healpix_index).real()
            std::map< std::string,
                      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace> > complex_det_surface_data; 

            complex_det_surface_data = complexify_detector_data(det_surface_data)  ; 

            // access pattern: e.g. for real part: 
            // det_all_multipoles.at(var_name)(cumulative_utils::multipole_index).real()
            // std::map<std::string, Kokkos::View<Kokkos::complex<double>*,Kokkos::HostSpace> > det_all_multipoles; 
            auto const det_all_multipoles  = get_all_multipoles(spin_weight, max_ell, ntheta, nphi, det_indices, sw_sph_harmonics, complex_det_surface_data);
            std::filesystem::path base_path (runtime.scalar_io_basepath()) ;
            const std::string ascii_filename =  "./multipoles_det" + name + "";
            std::filesystem::path ascii_out_path = base_path / ascii_filename ;
            // Resolve to absolute path
            std::filesystem::path ascii_absolute_path = std::filesystem::absolute(ascii_out_path.lexically_normal());

            std::set<std::string> unique_var_names; 
            for (const auto& [key, value] : det_all_multipoles) {
                unique_var_names.insert(key);
            }

            save_multipole_timeseries_ascii(ascii_absolute_path,
                                        detector.radius_det_,
                                        spin_weight,
                                        max_ell,
                                        unique_var_names,
                                        det_all_multipoles, 
                                        current_iteration,
                                         current_time );
        detector_counter++;
        }
        // reset to 0 to ensure nothing is allocated at a time when Kokkos::finalize() is called
        // TODO: make sw_sph_harmonics object persist between calls to write_multipole_timeseries 
        // how expensive is re-initializing them every sphere_surface_multipole_output_every?
        // for spin_weight=0 & spin_weight=-2? 

        sw_sph_harmonics = Kokkos::View<Complex**, HostM>();

        GRACE_VERBOSE("Saved multipole decomposition data.") ; 
        }
         


    void save_surface_integral_timeseries_ascii(const std::string& parent_path,
                                        const double radius,
                                        const std::map<std::string,double>& surface_integrals, 
                                        const int current_iter,
                                        const double current_time){
    
        if ( parallel::mpi_comm_rank() == grace::master_rank() )
        {
            // open all /data/var/l/m groups and append the latest vales to respective datasets
            int counter=0;
            for( const auto& [var_name, var_val] : surface_integrals){

                    const std::string filename= ( parent_path+\
                                            var_name +\
                                            ".ascii"  );

                    if (not std::filesystem::exists(filename)) {
                        std::ofstream outfile(filename,std::ios::app) ; 
                        outfile << std::setprecision(15) << std::defaultfloat ; 
                        outfile << std::left << "# detector at radius = " << std::setprecision(2) << radius << std::setprecision(15) << '\n'
                                << std::left << "# for variable = " << var_name  << '\n' 
                                << std::left << "# iter time values " << '\n' ;
                    }
                    //else{
                        std::ofstream outfile(filename,std::ios::app) ; 
                        outfile << std::setprecision(15) << std::defaultfloat ; 
                        outfile << std::left << current_iter << '\t'
                                << std::left << current_time << '\t' 
                               // << std::left << surface_integrals.at(var_name) << '\n' ; 
                                << std::left << var_val << '\n' ; 
                    //}  
                   
            }
        }
        return ; 
    }


    void write_spherical_integrals_timeseries(){

        auto& runtime = grace::runtime::get( ) ;
        const auto comm = parallel::get_comm_world() ; 
        const int rank = parallel::mpi_comm_rank()  ; 
        const int world_size = parallel::mpi_comm_size()  ;
        const double current_time =  grace::get_simulation_time() ;
        const int current_iteration = grace::get_iteration() ;

        const int n_detectors = runtime.n_surface_output_spheres();
        const int nside = runtime.nside_surface_output_spheres();
        
      

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

        GRACE_VERBOSE("Interpolated variables on spherical surfaces for integral computation.") ; 

        // IF MODE == SERIAL_WRITING 
        int detector_counter = 0;  // Initialize counter
        // this loop automatically omit ranks that do not have a detector assigned (e.g. no coordinate overlap)
        // loop over detectors 
        std::map<std::string, double> surface_integrals;

        for (auto& [name, detector] : detectors) {
            std::vector<int> det_healpix_indices = detector.get_local_rank_sphere_indices();
            std::map<std::string,std::vector<double>> det_surface_data = detector.get_local_rank_detector_surface_data();

            /*
            *  Multiply by the healpix measure
            *  dA = 4 * pi / (12 NSIDE^2) 
            */

            const double dA = 4. * M_PI / ( 12 * nside * nside );

            for ( auto& [var_name, var_data] : det_surface_data) {  
                    // local rank operations : 
                    // partial sum arrays for the scalar product with each spherical harmonic:
                    // we store them in this way and not as a complex Kokkos::View, 
                    // since we invoke mpi_reduce on MPI_DOUBLE type... 
                    double local_integral{0};
                    double global_integral{0};


                    for(int idx_pix=0; idx_pix < det_healpix_indices.size(); idx_pix++) {
                        // const int pixel_index = det_healpix_indices[idx_pix];
                        local_integral+=var_data[idx_pix] * dA ;
                    }
                    

                    /*
                    * At this point, all ranks have their local partial sums for (l>=s,m) 
                    * We perform an MPI sum (harmonic-wise), and save on root 
                    * Note : maybe use allreduce instead? 
                    */

                    parallel::mpi_reduce(&local_integral, 
                                        &global_integral,
                                        1,
                                        mpi_sum,
                                        grace::master_rank()
                                        );
                    surface_integrals[var_name] = global_integral;
                }


            std::filesystem::path base_path (runtime.scalar_io_basepath()) ;
            const std::string ascii_filename =  "./surface_integral_" + name + "";
            std::filesystem::path ascii_out_path = base_path / ascii_filename ;
            // Resolve to absolute path
            std::filesystem::path ascii_absolute_path = std::filesystem::absolute(ascii_out_path.lexically_normal());

            save_surface_integral_timeseries_ascii(ascii_absolute_path,
                                        detector.radius_det_,
                                        surface_integrals,
                                        current_iteration,
                                         current_time );

            detector_counter++;
            }
      
        GRACE_VERBOSE("Saved sphere surface integral data.") ; 
        }
         




  }
}

