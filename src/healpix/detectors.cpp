/**
 * @file detectors.cpp
 * @author Konrad Topolski (topolski@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-03-12
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

#include <grace/utils/grace_utils.hh>
#include <grace/utils/format_utils.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/system/grace_system.hh>
#include <grace/config/config_parser.hh>
#include <grace/amr/p4est_headers.hh> 
#include <grace/amr/grace_amr.hh>
// #include <grace/amr/octree_search.hh>
#include <grace/amr/forest.hh> 

#include <grace/healpix/detectors.hh> 



namespace grace {


/** @brief
 * forward declarations of two functions in <grace/amr/octree_search.h>
 * since the implementation there is not decoupled from declaration 
// */
int reset_user_data(p4est_t* p4est,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant,
    p4est_locidx_t local_num,
    void* point); 

int 
my_points_function(p4est_t* p4est,
    p4est_topidx_t which_tree,
    p4est_quadrant_t* quadrant,
    p4est_locidx_t local_num,
    void* point);
    
// TO DO: once octree_search decouples .cpp from .hh, remove the above!


namespace healpix {

    /** 
     * @brief    get the coordinate from the healpix ring index form, see arXiv:astro-ph/0409513
         idx should run from 0 to 6N**2+2N-1 (inclusive) for the northern hemisphere+equitor if zsymm
        idx should run from 0 to 12N**2-1 (inclusive) when z symmetry is not present 
    * 
    */

    // void GRACE_HOST 
    // get_coord_from_healpix_index(
    //         const size_t nside, const double radius, const int idx,
    //         double& x1, double& x2, double& x3){

    //         size_t  i, j, ss;
    //         double ph, z, phi, theta, aux, dummy;

    //         // calculate the northern polar cap
    //         if (idx<2*(nside-1)*nside){
    //             ph = (idx+1.)/2;
    //             aux = int(ph);
    //             aux = sqrt(ph-sqrt(aux));
    //             i = int(aux)+1;
    //             j = idx+1-2*i*(i-1);
    //             z = 1-i*i/(3.0*nside*nside);
    //             ss = 1;
    //             phi = M_PI*(j-ss/2.0)/(2*i);
    //         } // calculate the northern and southern belt
    //         else if (idx<10*nside*nside+2*nside){
    //             ph = idx-2*nside*(nside-1);
    //             i = int(ph/(4.*nside))+nside;
    //             j =  std::fmod(ph, 4.*nside)+1;
    //             z = 4.0/3.0-2*i/(3.0*nside);
    //             ss = (i-nside+1) % 2;
    //             phi = M_PI*(j+ss/2.0-1.0)/(2*nside);
    //         } // calculate the southern polar cap
    //         else if (idx<12*nside*nside){
    //             ph = ((12*nside*nside-idx-1)+1.)/2;
    //             aux = int(ph);
    //             aux = sqrt(ph-sqrt(aux));
    //             i = int(aux)+1;
    //             j = idx+1-2*i*(i-1);
    //             z = 1-i*i/(3.0*nside*nside);
    //             ss = 1;
    //             phi = 2*M_PI-M_PI*(j-ss/2.0)/(2*i);
    //         }
    //         else {
    //             ERROR("An index larger than 12*nside**2 is not allowed in healpix outflow!");
    //         }        
    //         // output coordinate according to coordinate type
            
    //         theta = std::acos(z);
    //         x1 = std::sin(theta) * std::cos(phi) * radius;
    //         x2 = std::sin(theta) * std::sin(phi) * radius;
    //         x3 = radius*z;
    // }


    void GRACE_HOST 
    get_spherical_coord_from_healpix_index(
            const size_t nside, const int idx,
            double& theta, double& phi){

            size_t  i, j, ss;
            double ph, z, aux, dummy;

            // calculate the northern polar cap
            if (idx<2*(nside-1)*nside){
                ph = (idx+1.)/2;
                aux = int(ph);
                aux = sqrt(ph-sqrt(aux));
                i = int(aux)+1;
                j = idx+1-2*i*(i-1);
                z = 1-i*i/(3.0*nside*nside);
                ss = 1;
                phi = M_PI*(j-ss/2.0)/(2*i);
            } // calculate the northern and southern belt
            else if (idx<10*nside*nside+2*nside){
                ph = idx-2*nside*(nside-1);
                i = int(ph/(4.*nside))+nside;
                j =  std::fmod(ph, 4.*nside)+1;
                z = 4.0/3.0-2*i/(3.0*nside);
                ss = (i-nside+1) % 2;
                phi = M_PI*(j+ss/2.0-1.0)/(2*nside);
            } // calculate the southern polar cap
            else if (idx<12*nside*nside){
                ph = ((12*nside*nside-idx-1)+1.)/2;
                aux = int(ph);
                aux = sqrt(ph-sqrt(aux));
                i = int(aux)+1;
                j = idx+1-2*i*(i-1);
                //z = 1-i*i/(3.0*nside*nside);
                z = 1-i*i/(3.0*nside*nside);
                z = -z; 
                ss = 1;
                phi = 2*M_PI-M_PI*(j-ss/2.0)/(2*i);
            }
            else {
                ERROR("An index larger than 12*nside**2 is not allowed in healpix outflow!");
            }        
            theta = std::acos(z);      
    }



    void GRACE_HOST 
    get_cartesian_coord_from_healpix_index(
            const size_t nside, const double radius, const int idx,
            double& x1, double& x2, double& x3){

            size_t  i, j, ss;
            double ph, z, phi, theta, aux, dummy;

            get_spherical_coord_from_healpix_index(nside, idx,
                                                   theta, phi);

            x1 = std::sin(theta) * std::cos(phi) * radius;
            x2 = std::sin(theta) * std::sin(phi) * radius;
            x3 = std::cos(theta) * radius;
    }


  /**
         * @brief get uniform-spherical-sampling detector coordinates for a given
         *  ntheta, nphi, radius and i_theta, j_phi indices
         * @note we use here uniform convention of pixel book-keeping
         */
        void GRACE_HOST_DEVICE 
        get_cartesian_coord_from_ith_jph_index(const size_t ntheta, const size_t nphi, double radius, const size_t i_th,   const size_t j_phi, 
                                                double& x1, double& x2, double& x3){
                
                double theta, phi;
                get_spherical_coord_from_ith_jph_index(ntheta,nphi,i_th,j_phi,
                                                      theta,phi);
                x1 = std::sin(theta) * std::cos(phi) * radius;
                x2 = std::sin(theta) * std::sin(phi) * radius;
                x3 = std::cos(theta) * radius;
            }

        void GRACE_HOST_DEVICE 
        get_spherical_coord_from_ith_jph_index(const size_t ntheta, const size_t nphi, const int i_th, const int j_phi,
                                              double& th, double& ph){
                // midpoint coordinates hard-coded
                const double dth  =    M_PI/(ntheta+1);
                const double dphi = 2.*M_PI/(nphi+1);
                th = i_th*dth    + 0.5*dth;
                ph = j_phi*dphi  + 0.5*dphi;
            }

        // same as above, just using the cumulative oned index 

        void GRACE_HOST_DEVICE
        get_spherical_coord_from_oned_index(const size_t ntheta, const size_t nphi,const int idx_oned,
                                            double& th, double& ph){
                int i_theta, j_phi;
                get_i_th_j_ph_from_oned_index(ntheta, nphi, idx_oned,
                                         i_theta, j_phi);
                get_spherical_coord_from_ith_jph_index(ntheta,nphi, i_theta, j_phi, th, ph);
            }




    void GRACE_HOST
    spherical_detector::clean_detector_info(){
            rank_coords_det_ = {};                
            indices_ = {};                        
            which_quadrants_ = {};               
            outflows_ = {};                      
    }

    void GRACE_HOST
    spherical_detector::update_detector_info(){
        // Need a p4est
        auto& forest = grace::amr::forest::get() ; 

        size_t first = forest.first_local_tree();
        size_t last = forest.last_local_tree();
        

        this->clean_detector_info(); 

        // Step 1.:
        // search for sthe quadrants that have a common intersection with detector's coordinates on this rank

        // Reset the user data to 0
        p4est_search_local_t reset_func = reset_user_data;
        p4est_search_local(forest.get(), false, reset_func, nullptr, nullptr);

        // TO DO 
        // rewrite the detector class so that we use sc_array_t for things from the start 
        auto convert_std_vector_to_sc_array_t 
                = [](const std::vector<std::array<double,3>>& coords) ->sc_array_t* {
                        using value_type = std::array<double, 3>;
                
                        sc_array_t* sc_array = new sc_array_t;
                        sc_array->elem_count = coords.size();
                        sc_array->elem_size = sizeof(value_type);
                        sc_array->array = static_cast<char*>(malloc(sc_array->elem_count * sc_array->elem_size));
                        value_type* data = reinterpret_cast<value_type*>(sc_array->array);
                        for (int i = 0; i < sc_array->elem_count; ++i) {
                                data[i] = coords[i];//{coords[3*i + 0], coords[3*i + 1], coords[3*i + 2]};
                        }
                    return sc_array;
                };

        p4est_search_local_t point_search_func = my_points_function;
        p4est_search_local(forest.get(), true , nullptr, point_search_func, convert_std_vector_to_sc_array_t(all_coords_det_));

        // how many quadrants are crossed in total?
        std::set<int> marked_quadrants;

        size_t it = 0;

        // GRACE_VERBOSE("first local tree idx {}", first);
        // GRACE_VERBOSE("last local tree idx {}", last);

        for (size_t i = first; i <= last; i++) // Loop from first to last local tree
        {
            auto tree = forest.tree(i);  // Assuming there is a function to access a tree by index

            size_t quadrant_offset = tree.quadrants_offset();  // Number of quadrants in the tree
            // GRACE_VERBOSE("Tree {}, quadrant offset{}", i,quadrant_offset );

            it=0;
            for (auto tree_quadrant : tree.quadrants())
            {
                auto quadrant = tree_quadrant;  // Get the j-th quadrant in the tree (adjust if needed)

                if (quadrant.p.user_int != 0)
                {
                    // push the cumulative local index of the quadrant 
                    // (i.e. the one counted across the trees and from 0 till <code>forest::local_num_quadrants()</code>)
                    // this represents all quadrants that will be used in interpolation
                    marked_quadrants.insert(quadrant_offset + it);

                    GRACE_VERBOSE("Quadrant {} in tree {} intersects the points {} times.\n", quadrant_offset + it, i, quadrant.p.user_int);

                }
                it++;
                }
            // GRACE_VERBOSE("Tree {} had {} quadrants ", i, it );
            }

    GRACE_VERBOSE("Detector coordinates cross {} quadrants on MPI rank {}", marked_quadrants.size(), parallel::mpi_comm_rank()) ; 

    size_t nx,ny,nz,nq; 
    std::tie(nx,ny,nz) = grace::amr::get_quadrant_extents() ; 
    nq = grace::amr::get_local_num_quadrants() ; 
    const int64_t q_global_offset = grace::amr::get_global_quadrant_offsets()[parallel::mpi_comm_rank()];

    // Retrieve the coordinate system from the grace library.
    auto& coord_system = grace::coordinate_system::get();
    std::array<size_t, 3> ijk = {0, 0, 0};

    assert(pix_end_ == all_coords_det_.size());

    // loop over all potential pixels 
    for(int idx_pt=0; idx_pt < pix_end_; idx_pt++){
        // loop over all quadrants that have been marked as relevant:
        // loc_q is global quadrant index
        for (auto const& loc_q : marked_quadrants) {

            // note:
            // this check is for coordinates of cell centres 
            // without the inclusion of ghost zones
            std::array<double, 3> lower = coord_system.get_physical_coordinates(ijk, loc_q, {VEC(0.0,0.0,0.0)}, false);
            std::array<double, 3> upper = coord_system.get_physical_coordinates(ijk, loc_q, {VEC(static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz))}, 0);

            // loop over all the coords:
            const std::array<double,3> point_coord = all_coords_det_[idx_pt];

            if (lower[0] <= point_coord[0] && upper[0] > point_coord[0] &&
                lower[1] <= point_coord[1] && upper[1] > point_coord[1] &&
                lower[2] <= point_coord[2] && upper[2] > point_coord[2]){
                // All conditions are satisfied, so the point is within the quadrant.
                indices_.push_back(idx_pt);
                rank_coords_det_.push_back(point_coord);  
                // local quadrant indices 
                which_quadrants_.push_back(loc_q);

                }   
            }
        }
    }

    double GRACE_HOST_DEVICE trilinearInterpolation(
    double F000, double F100, double F010, double F110,
    double F001, double F101, double F011, double F111,
    double a, double b, double c) 
    {
        return (1-a)*(1-b)*(1-c) * F000 +
            a*(1-b)*(1-c) * F100 +
            (1-a)*b*(1-c) * F010 +
            a*b*(1-c) * F110 +
            (1-a)*(1-b)*c * F001 +
            a*(1-b)*c * F101 +
            (1-a)*b*c * F011 +
            a*b*c * F111;
    }

    /**
     *  In this routine, we launch a GPU kernel that will interpolate at the required 
     *  coordinates 
     *  TODO: time this routine - is it more efficient to launch one GPU kernel per detector,
     *  or one global kernel for all detectors (in sphere_output)?
     */
    void GRACE_HOST
    spherical_detector::update_detector_variable_data( const std::set<std::string> corner_scalar_vars
                                                    ,const std::set<std::string> corner_vector_vars
                                                    ,const std::set<std::string> corner_tensor_vars
                                                    ,const std::set<std::string> cell_scalar_vars 
                                                    ,const std::set<std::string> cell_vector_vars
                                                    ,const std::set<std::string> cell_tensor_vars
                                                    ,const size_t interp_method
                                                ){
        
        using namespace grace ; 
        // using namespace Kokkos  ; 
        assert(indices_.size() == which_quadrants_.size());
        assert(indices_.size() == rank_coords_det_.size());
        
        int64_t nx,ny,nz ; 
        std::tie(nx,ny,nz) = amr::get_quadrant_extents() ; 
        int ngz = amr::get_n_ghosts() ; 
        int64_t nq = amr::get_local_num_quadrants() ;

        auto& state = variable_list::get().getstate()   ; 
        auto& sstate = variable_list::get().getstaggeredstate() ; 
        auto& aux = variable_list::get().getaux() ; 
        auto& saux = variable_list::get().getstaggeredaux() ; 

        auto& dx    = variable_list::get().getspacings() ; 
        auto& grace_runtime = grace::runtime::get() ; 

        // the MPI-rank specific coordinates of a detector (rank_coords_det_), pixel indices (indices_), and corresponding quadrants (which_quadrants_)
        // are all known - we will now fill out the outflow std::map for all variables
       
        std::set<std::string> corner_vars;

        #define COMBINE_VARS(COMBINED_SET, MEMBER)\
                COMBINED_SET.insert(MEMBER.begin(), MEMBER.end());


        COMBINE_VARS(corner_vars,corner_scalar_vars );
        COMBINE_VARS(corner_vars,corner_vector_vars );
        COMBINE_VARS(corner_vars,corner_tensor_vars );

        std::set<std::string> cell_vars;

        COMBINE_VARS(cell_vars,cell_scalar_vars );
        COMBINE_VARS(cell_vars,cell_vector_vars );
        COMBINE_VARS(cell_vars,cell_tensor_vars );


        // Step 1:
        // Copy data to Kokkos Views
        const size_t num_corner_variables = corner_vars.size(); 
        const size_t num_cell_centred_variables = cell_vars.size(); 

        std::set<std::string> all_vars = corner_vars;
        all_vars.insert(cell_vars.begin(),cell_vars.end()) ;


        // if(num_cell_centred_variables > 0) GRACE_ERROR("NOT YET IMPLEMENTED");


        Kokkos::View<double**, grace::default_space> d_rank_coords_det("rank_coords_det", indices_.size(), 3);                
        Kokkos::View<int*, grace::default_space>     d_indices("indices", indices_.size() );                
        Kokkos::View<int*, grace::default_space>             d_which_quadrants("quadrants", indices_.size() );                
        //Kokkos::View<double**, grace::default_space>         d_outflows("outflows", num_corner_variables+num_cell_centred_variables, indices_.size()); 
        Kokkos::View<double**, grace::default_space>         d_outflows("outflows", all_vars.size(), indices_.size()); 

        // Step 2: Mirror views to copy data from host to device
        auto h_rank_coords_det = create_mirror_view(d_rank_coords_det);
        auto h_indices = create_mirror_view(d_indices);
        auto h_which_quadrants = create_mirror_view(d_which_quadrants);
        auto h_outflows = create_mirror_view(d_outflows);

        // Step 3: Fill out the views on host:
        for(int i=0; i< indices_.size(); i++){
            h_indices(i)           = indices_[i];
            h_rank_coords_det(i,0) = rank_coords_det_[i][0];
            h_rank_coords_det(i,1) = rank_coords_det_[i][1];
            h_rank_coords_det(i,2) = rank_coords_det_[i][2];
            h_which_quadrants(i)   = which_quadrants_[i];    
        }

        // Step 4: Deep copy for use in the kernel
        deep_copy(d_rank_coords_det, h_rank_coords_det);
        deep_copy(d_which_quadrants, h_which_quadrants);
        deep_copy(d_indices, h_indices);
        
       // Step 5: define range policy and launch the kernel
       // the natural split here is just among the pixel indices

       // interpolation for the registered variables:
       // corner-centred variables 

       // Get coordinates of grid corners 
        auto& coord_system = coordinate_system::get() ; 
        coord_array_t<GRACE_NSPACEDIM> pcoords_corner ; 
        grace::fill_physical_coordinates(pcoords_corner, {VEC(true,true,true)} ) ;

        coord_array_t<GRACE_NSPACEDIM> pcoords_center ; 
        grace::fill_physical_coordinates(pcoords_center, {VEC(false,false,false)} ) ;

        size_t counter=0;

        GRACE_VERBOSE("Interpolating {} corner variables on the sphere" , corner_vars.size()) ; 

        // for(auto const& vname: corner_vars) {
        for(auto const& vname: all_vars) {
            GRACE_VERBOSE("Interpolating for variable: {} on {} points" , vname, indices_.size()) ; 

            auto policy = Kokkos::RangePolicy<grace::default_execution_space>(0, indices_.size());
            auto const u = get_variable_subview(vname) ; 
            auto it = variables::detail::_varprops.find(vname);
            if (it == variables::detail::_varprops.end()) {
                ERROR("In get_variable_subview variable " << vname << " does not exist.") ;
            }

            auto const& stagger_type = (it->second).staggering; 
            if(not ((stagger_type == var_staggering_t::CELL_CENTER) ||
                    (stagger_type == var_staggering_t::CORNER)        )){
                    ERROR("Unrecognized staggering for this routine");
            }

            const char* vname_cstr = vname.c_str();  // Still host-side

            parallel_for( GRACE_EXECUTION_TAG("IO","surface_interpolation") 
                    , policy 
                    , KOKKOS_LAMBDA(int idx_loc_pix){

                            // true pixel number in the RING format:
                            const size_t idx_pixel = d_indices(idx_loc_pix); // not used in this routine, only in the MPI-parallelized - healpix I/O

                            // if INTERPOLATION_METHOD::LINEAR
                            const size_t iq = d_which_quadrants(idx_loc_pix); 
                       

                            coord_array_t<GRACE_NSPACEDIM> pcoords;

                            if ( stagger_type == var_staggering_t::CORNER ) {
                                pcoords=pcoords_corner;
                            } 
                            else if( stagger_type == var_staggering_t::CELL_CENTER ){
                                pcoords=pcoords_center;
                            }
                            
                           
                            const std::array<double,3> pcoords_lower = { pcoords(VEC(ngz,ngz,ngz),0,iq),
                                                                         pcoords(VEC(ngz,ngz,ngz),1,iq),
                                                                         pcoords(VEC(ngz,ngz,ngz),2,iq) };
    
                            // between 0 and 1:                                
                            const double dst_x = std::fmod((d_rank_coords_det(idx_loc_pix,0) - pcoords_lower[0]) , dx(0,iq)) / dx(0,iq);
                            const double dst_y = std::fmod((d_rank_coords_det(idx_loc_pix,1) - pcoords_lower[1]) , dx(1,iq)) / dx(1,iq);
                            const double dst_z = std::fmod((d_rank_coords_det(idx_loc_pix,2) - pcoords_lower[2]) , dx(2,iq)) / dx(2,iq);
                            
                            const size_t i_lo = ngz; 
                            const size_t j_lo = ngz; 
                            const size_t k_lo = ngz; 
                            
                            const size_t i = (d_rank_coords_det(idx_loc_pix,0) - pcoords_lower[0]) / dx(0,iq) + i_lo;
                            const size_t j = (d_rank_coords_det(idx_loc_pix,1) - pcoords_lower[1]) / dx(1,iq) + j_lo;
                            const size_t k = (d_rank_coords_det(idx_loc_pix,2) - pcoords_lower[2]) / dx(2,iq) + k_lo;
                                
                            d_outflows(counter, idx_loc_pix) = trilinearInterpolation( u(VEC(i,j,k),iq),    u(VEC(i+1,j,k),iq),    u(VEC(i,j+1,k),iq),    u(VEC(i+1,j+1,k),iq),
                                                                                       u(VEC(i,j,k+1),iq),  u(VEC(i+1,j,k+1),iq),  u(VEC(i,j+1,k+1),iq),  u(VEC(i+1,j+1,k+1),iq),
                                                                                       dst_x, dst_y, dst_z); 

                            // printf("(x, y, z, val) = (%f,%f,%f,%f) \n", d_rank_coords_det(idx_loc_pix,0),
                            //                                           d_rank_coords_det(idx_loc_pix,1),
                            //                                           d_rank_coords_det(idx_loc_pix,2),
                            //                                           d_outflows(counter, idx_loc_pix)
                            //                                            )    ;                                                 
                            // printf("counter %d, idxloc %d, val %f",counter, idx_loc_pix, d_outflows(counter, idx_loc_pix));

                        }) ; 
            counter++;
        }

        Kokkos::deep_copy(h_outflows, d_outflows);

        counter=0;
        //for (auto const& vname : corner_vars) {
        for (auto const& vname : all_vars) {
                std::vector<double> outflow_;
                outflow_.resize(indices_.size());
                for(int i = 0 ; i < indices_.size() ; i++){
                            outflow_[i] = h_outflows(counter,i);
                        }
                outflows_[vname] = outflow_;
                counter++;
        }

    }

    /**
     * @brief compute surface fluxes for named quantities or vector variables 
     *        by contracting with the flat normal surface vector 
     * 
     */
    void GRACE_HOST_DEVICE
    spherical_detector::compute_surface_fluxes(){}
         // need metric interface for sqrt(-g)
         

    }
}