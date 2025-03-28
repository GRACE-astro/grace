/**
 * @file detectors.hh
 * @authors Konrad Topolski, Kenneth Miller, based on healpix implementations by 
 *          Carlo Musolino and Jin-Liang Jiang 
 * @brief 
 * @date 2025-02-27
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

#ifndef GRACE_DETECTORS
#define GRACE_DETECTORS

#include <grace_config.h> 
#include <hdf5.h>

#include <array>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/coordinates/coordinate_systems.hh>

#include <cmath>
// #include <grace/utils/spin_weighted_sph_harmonics.hh>

//**************************************************************************************************/

namespace grace{

    namespace healpix{


        enum INTERPOLATION_METHODS{
            LINEAR,
            LAGRANGE3
        };

        /**
         * @brief get detector coordinates for a given nside, radius and pixel index
         * @note we use the RING convention of pixel book-keeping
         */
        void GRACE_HOST 
        get_cartesian_coord_from_healpix_index(
            const size_t nside, const double radius, const int idx,
            double& x1, double& x2, double& x3);

        void GRACE_HOST 
        get_spherical_coord_from_healpix_index(
            const size_t nside, const int idx,
            double& th, double& ph);


        /** 
         * @brief healpix detector info 
         * @note we use the RING convention of pixel book-keeping
         */
        struct healpix_detector{
            // the following are immutable once initialized:
            int current_it_; // avoid calculating twice
            int nside_; // healpix parameter: nside
            int pix_end_; // largest pix number allowed by the nside
            int north_end_; // final index for the northern hemisphere
            int ncount_; // number of pixels in this mpirank
            double  radius_det_;   // radius of the detector
            std::vector<std::array<double,3>>  all_coords_det_; // all coordinates of the detector 
            // the following change as the AMR structure changes:
            std::vector<std::array<double,3>>  rank_coords_det_; // coordinates of the detector that exist on this MPI rank (i.e. their respective quadrants form the part of the tree governed by this MPI rank) 
            std::vector<int> indices_;  // internal healpix indices of the pixels in the detector that exist within this mpi rank
            std::vector<int> which_quadrants_; // stored positions of quadrants for each of the above indices
            std::map<std::string,std::vector<double>> outflows_; // store outflows in a two-D fashion (first index is healpix index value, second outflow type)

            // the following are unused 
            // std::array<double,3> detector_centre; // centre of the detector 
            // bool zsymm; ! if true, only use the northern hemisphere
        
            healpix_detector(int nside, double radius_det, std::array<double,3> detector_centre) : nside_(nside), radius_det_(radius_det){
                    pix_end_ = 12 * nside * nside;
                    all_coords_det_.resize(pix_end_);
                    for(size_t idx=0; idx<pix_end_ ; idx++){
                        double x1,x2,x3; 
                        get_cartesian_coord_from_healpix_index(nside_, radius_det_, idx, x1,x2,x3);
                        all_coords_det_[idx]={x1,x2,x3};
                    }
            }


        
            /** 
             *  @brief update_detector_info 
             *  @note find which detector healpix indices (and which coordinates on the sphere) 
             *        belong to the given MPI rank          
            **/
            void GRACE_HOST
            update_detector_info();
            
            /** 
             *  @brief clean_detector_data 
             *  @note clean the rank-specific coordinate and outflow detector data
            **/
            void GRACE_HOST
            clean_detector_info();


            /**
             * @brief interpolate registered variables 
             *      at the local (rank-specific) detector coordinates
             */
            void GRACE_HOST
            update_detector_variable_data(const std::set<std::string> corner_scalar_var
                                        ,const std::set<std::string> corner_vector_var
                                        ,const std::set<std::string> corner_tensor_var
                                        ,const std::set<std::string> cell_scalar_vars 
                                        ,const std::set<std::string> cell_vector_vars
                                        ,const std::set<std::string> cell_tensor_vars
                                        ,const size_t interp_method);

                /**  
             * @brief compute_surface_fluxes for registered variables
             * @note supports computing the contraction of a vector variable with the normal to sphere
             * 
             */
            //void GRACE_HOST_DEVICE 
            void GRACE_HOST_DEVICE
            compute_surface_fluxes(); 


            // void GRACE_HOST
            // compute_integrals(){
            // MPI-calls fetch all the data to root 
            //
            //
            // }
            // void GRACE_HOST
            // compute_multipoles(){
            // bring all the data onto 
            // }

            std::vector<int> GRACE_HOST_DEVICE
            get_local_rank_healpix_indices() const{
                return indices_;
            }

            std::vector<int> GRACE_HOST_DEVICE
            get_local_rank_quadrants() const{
                return which_quadrants_;
            }


            std::vector<std::array<double,3>> GRACE_HOST_DEVICE
            get_local_rank_detector_coordinates() const{
                return rank_coords_det_;
            }

            std::map<std::string,std::vector<double>> GRACE_HOST_DEVICE
            get_local_rank_detector_surface_data() const{
                return outflows_;
            }


        };


    }

}

#endif /* GRACE_DETECTORS */