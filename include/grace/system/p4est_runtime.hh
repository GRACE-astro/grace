/**
 * @file p4est_runtime.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Singleton owning the p4est library lifetime (sc_init / p4est_init / sc_finalize) and bridging libsc log messages to the GRACE logger.
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
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
 */
#ifndef INCLUDE_GRACE_SYSTEM_P4EST_RUNTIME
#define INCLUDE_GRACE_SYSTEM_P4EST_RUNTIME

#include <grace/amr/p4est_headers.hh>
#include <grace_config.h>

#include <grace/utils/singleton_holder.hh> 
#include <grace/utils/creation_policies.hh>
#include <grace/utils/lifetime_tracker.hh> 
#include <grace/system/print.hh>
#include <spdlog/spdlog.h>
namespace grace {

namespace detail {
static void grace_sc_log_hijacker(FILE* log_stream,
                                    const char* filename,
                                    int lineno,
                                    int package,
                                    int category,
                                    int priority,
                                    const char* msg)
{
    GRACE_VERBOSE(msg) ;
}
}

class p4est_runtime_impl_t 
{
 private:
    
    p4est_runtime_impl_t() {
        p4est_init(detail::grace_sc_log_hijacker, SC_LP_DEFAULT) ; 
    }
    ~p4est_runtime_impl_t() { } 

    friend class utils::singleton_holder<p4est_runtime_impl_t,memory::default_create> ; //!< Give access
    friend class memory::new_delete_creator<p4est_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access

    static constexpr size_t longevity = P4EST_RUNTIME ; 

} ; 

using p4est_runtime = utils::singleton_holder<p4est_runtime_impl_t,memory::default_create> ;

} /* namespace grace */

#endif /* INCLUDE_p4est_SYSTEM_P4EST_RUNTIME */
