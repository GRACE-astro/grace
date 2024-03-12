/**
 * @file thunder_runtime.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @version 0.1
 * @date 2024-03-12
 * 
 * @copyright This file is part of Thunder.
 * Thunder is an evolution framework that uses Finite Difference
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
#ifndef INCLUDE_THUNDER_SYSTEM_THUNDER_RUNTIME
#define INCLUDE_THUNDER_SYSTEM_THUNDER_RUNTIME

#include <thunder_config.h>

#include <thunder/utils/singleton_holder.hh> 
#include <thunder/utils/creation_policies.hh>
#include <thunder/utils/lifetime_tracker.hh> 

#include <thunder/config/config_parser.hh>

#include <thunder/parallel/mpi_wrappers.hh>

#include <string> 
#include <iostream>

namespace thunder {

class thunder_runtime_impl_t 
{
 public:

    int master_rank ; //!< The master rank is the one which is allowed to print to stdout 
    int print_threshold ; //!< Maximum level warnings / messages printed 
 private:

    thunder_runtime_impl_t() {
        auto& params = thunder::config_parser::get() ; 
        master_rank = params["system"]["master_rank"].as<int>() ; 
        print_threshold = params["system"]["print_threshold"].as<int>() ; 
        if( parallel::mpi_comm_rank() == master_rank ) 
        {
            std::cout << THUNDER_BANNER ; 
        }
    }
    ~thunder_runtime_impl_t() {} 

    friend class utils::singleton_holder<thunder_runtime_impl_t,memory::default_create> ; 
    friend class memory::new_delete_creator<thunder_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access

    static constexpr size_t longevity = THUNDER_RUNTIME ; 

} ; 

using runtime = utils::singleton_holder<thunder_runtime_impl_t,memory::default_create> ;

} /* namespace thunder */

#endif /* INCLUDE_THUNDER_SYSTEM_THUNDER_RUNTIME */
