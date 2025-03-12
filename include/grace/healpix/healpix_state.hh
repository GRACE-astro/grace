/**
 * @file healpix_state.hh
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

#ifndef GRACE_HEALPIX_STATE
#define GRACE_HEALPIX_STATE

#include <grace_config.h> 
#include <hdf5.h>

#include <array>
#include <grace/utils/grace_utils.hh>
#include <grace/system/grace_system.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/config/config_parser.hh>
#include <grace/utils/lifetime_tracker.hh>


#include <cmath>
#include <grace/healpix/detectors.hh>


//**************************************************************************************************/
/**
 * \defgroup physics Physics Modules.
 * 
 */

namespace grace{

    namespace healpix{

        using healpix_detectors_t = std::vector<healpix_detector>; 

        /** 
         * @ brief
         * wrapper class around the set of healpix-style detectors 
         * the main purpose of this class is to provide an adequate life-time control 
         * over the detectors by means of the singleton class  
         */

        class healpix_impl_t 
        {
        private:

        //*****************************************************************************************************
            healpix_detectors_t _healpix_detectors ; //!< vector of detectors 
            int num_of_detectors;
            std::vector<double> detectors_radii; 
        //*****************************************************************************************************
        public: 
         
            /**
             * @brief Get pointer to underlying detectors object. 
             */
            GRACE_ALWAYS_INLINE healpix_detectors_t&
            get_detectors() { return _healpix_detectors ; }

        //*****************************************************************************************************
        /**
         * @brief functions governing the initialization, scheduling and computation of detectors 
         * 
         */


        //*****************************************************************************************************
        private:
        //*****************************************************************************************************
            /**
             * @brief Never construct a new healpix_impl_t object directly, only through the singleton and longevity mechanism
             */
            healpix_impl_t() ; 
        //*****************************************************************************************************
            /**
             * @brief Never destroy the healpix_impl_t object directly, only through the singleton and longevity mechanism
             * 
             */
            ~healpix_impl_t() ; 
        //*****************************************************************************************************
            friend class utils::singleton_holder<healpix_impl_t, memory::default_create> ;          //!< Give access 
            friend class memory::new_delete_creator<healpix_impl_t, memory::new_delete_allocator> ; //!< Give access
            static constexpr unsigned int longevity = GRACE_HEALPIX; //!< Longevity of healpix_impl_t object. 
        //*****************************************************************************************************
        } ; 
    //*****************************************************************************************************
    /**
     * @brief GRACE singleton type. This 
     *        global object can be accessed from user code 
     *        to get information about the detectors
     * \ingroup healpix 
     */
    using healpix_state = utils::singleton_holder<healpix_impl_t > ; 

    }
}













#endif /* GRACE_HEALPIX_STATE */