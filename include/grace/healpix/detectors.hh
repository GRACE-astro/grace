/**
 * @file detectors.hh
 * @authors Konrad Topolski, Kenneth Miller 
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

#include <cmath>
// #include <grace/utils/spin_weighted_sph_harmonics.hh>

//**************************************************************************************************/
/**
 * \defgroup physics Physics Modules.
 */

namespace grace{

    namespace healpix{

        void GRACE_HOST 
        get_coord_from_healpix_index(
            const size_t nside, const double radius, const int idx,
            double& x1, double& x2, double& x3);


        // we need a wrapper around the healpix_detector class that will construct 
        // a global reference to the healpix state... 
        /** 
         * @brief initialize_detectors 
         * @note we make use of the GRACE singleton object to keep track of the created detectors
         */
        // void GRACE_HOST_DEVICE
        // static initialize_detectors() {
            
        // }


        /** 
         * @brief compute_fluxes 
         * @note scheduling the interpolation on detector surfaces on the device 
         */
        //void GRACE_HOST_DEVICE 
        void GRACE_HOST_DEVICE
        compute_surface_fluxes(); 

        
            // need to find and update the relevant quadrants in the forest anytime this routine is called
            // - the data interpolation takes place there
            // find_quadrants_from_coordinates()

            // start a kernel here to interpolate the 
            // interpolate_onto_detector_surfaces() 

            //

     
        // void GRACE_HOST
        // compute_integrals(){
            // MPI-calls fetch all the data to root 
            //
            //
        //}
        // void GRACE_HOST
        // compute_multipoles(){
            // bring all the data onto 
        // }
        



        /** 
         * @brief healpix detector info 
         * @note we use the RING convention of pixel book-keeping
         */
        struct healpix_detector{
            int current_it_; // avoid calculating twice
            int nside_; // healpix parameter: nside
            // bool zsymm; ! if true, only use the northern hemisphere
            int pix_end_; // largest pix number allowed by the nside
            int north_end_; // final index for the northern hemisphere
            int ncount_; // number of pixels in this mpirank
            double  radius_det_;   // radius of the detector
            std::vector<double>  all_coords_det_; // all coordinates of the detector 
            std::vector<double>  rank_coords_det_; // coordinates of the detector that exist on this MPI rank (i.e. their respective quadrants form the part of the tree governed by this MPI rank) (the coords are 3dim, in fact, so we pick some access pattern: i*Npts + j) 
            std::vector<int> indices_;  // internal healpix indices of the pixels in the detector that exist within this mpi rank
            std::vector<int> which_quadrants_; // stored positions of quadrants for each of the above indices
            std::vector<double> outflows_; // store outflows in a two-D fashion (first index is healpix index value, second outflow type)
            
            // For each detector, we fill out the all_coord_det_ vector on all MPI ranks
            // This is done one time only, at initialization 
            // Every subsequent computation - since the rank-tree assignment might have changed from the previous call -
            // requires us to find indices_ and rank_coord_det_ again. 

            healpix_detector() = delete; // delete empty constructor 

            healpix_detector(int nside, double radius_det) : nside_(nside), radius_det_(radius_det){
                pix_end_ = 12 * nside * nside;
                all_coords_det_.resize(pix_end_*GRACE_NSPACEDIM);
                for(size_t idx=0; idx<pix_end_ ; idx++){
                    double x1,x2,x3; 
                    //const size_t nside, const double radius, const int idx,
                    get_coord_from_healpix_index(nside_, radius_det_, idx, x1,x2,x3);
                    all_coords_det_[idx * 3 + 0] = x1;  // X coordinate for point i
                    all_coords_det_[idx * 3 + 1] = x2;  // Y coordinate for point i
                    all_coords_det_[idx * 3 + 2] = x3;  // Z coordinate for point i
                }

            }


       
        /** @brief update_detector_info 
         *  @note for the detector's array of coordinates and based on local forest information, 
         *        find which detector indices (and hence, which coordinates on the sphere) 
         *        belong to the given MPI rank 
         *  @credits to Kenneth Miller 
        **/
        void GRACE_HOST
        update_detector_info();
            
        
        void GRACE_HOST_DEVICE
        update_detector_fluxes();



        };




    }

}

#endif /* GRACE_DETECTORS */