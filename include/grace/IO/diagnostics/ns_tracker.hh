/**
 * @file puncture_tracker.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2026-01-27
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
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

#ifndef GRACE_IO_NS_TRACKER_HH
#define GRACE_IO_NS_TRACKER_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/data_structures/variable_indices.hh>

#include <grace/utils/metric_utils.hh>

#include <Kokkos_Core.hpp>


#include <grace/utils/singleton_holder.hh>
#include <grace/utils/lifetime_tracker.hh>

#include <vector>
#include <memory>

namespace grace {

/**
 * @brief Given an initial location of n neutron stars, integrates sqrtg D x^i to find the CoM
 *        of each star. The integral is restricted to rho > thresh and points are assigned to 
 *        the star whose last known position is closest. 
 */
struct ns_tracker_impl_t 
{
    public:

        void update_and_write() {

        }


    private:
        void update() {

        }

        void output() 
        {

        }

        void initialize_files() {

        }


} ; 


}

#endif /* GRACE_IO_NS_TRACKER_HH */