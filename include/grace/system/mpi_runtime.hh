/**
 * @file mpi_runtime.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Singleton owning the MPI lifetime (MPI_Init_thread / MPI_Finalize) and bookkeeping the master rank used by GRACE printing helpers.
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
#ifndef INCLUDE_GRACE_SYSTEM_MPI_RUNTIME
#define INCLUDE_GRACE_SYSTEM_MPI_RUNTIME

#include <grace_config.h>

#include <grace/parallel/mpi_wrappers.hh>
#include <grace/utils/singleton_holder.hh> 
#include <grace/utils/creation_policies.hh>
#include <grace/utils/lifetime_tracker.hh>
#include <grace/utils/inline.h>

#include <grace/config/config_parser.hh>

namespace grace {
//*****************************************************************************************************
//*****************************************************************************************************
/**
 * @brief Utility class that ensures MPI is initialized and finalized at appropriate times.
 * \ingroup system 
 */
class mpi_runtime_impl_t 
{
 private:
    int _master_rank ;     //!< The master rank is the one which is allowed to print to stdout 
 public:
    GRACE_ALWAYS_INLINE int master_rank() const { return _master_rank ; }
 private:
    //*****************************************************************************************************
    /**
     * @brief (Never) construct a new <code>mpi_runtime_impl_t</code> object
     */
    mpi_runtime_impl_t(int argc, char* argv[] ) {
        parallel::mpi_init(&argc, &argv) ; 
        auto& params = grace::config_parser::get() ; 
        _master_rank = params["system"]["master_rank"].as<int>() ; 
    }
    //*****************************************************************************************************
    /**
     * @brief (Never) destroy the <code>mpi_runtime_impl_t</code> object
     * 
     */
    ~mpi_runtime_impl_t() {
        parallel::mpi_finalize() ; 
    } 
    //*****************************************************************************************************
    friend class utils::singleton_holder<mpi_runtime_impl_t,memory::default_create> ;           //!< Give access
    friend class memory::new_delete_creator<mpi_runtime_impl_t, memory::new_delete_allocator> ; //!< Give access
    //*****************************************************************************************************
    static constexpr size_t longevity = MPI_RUNTIME ; //!< Schedule destruction
    //*****************************************************************************************************
} ; 
//*****************************************************************************************************
/**
 * @brief Proxy for mpi runtime
 */
using mpi_runtime = utils::singleton_holder<mpi_runtime_impl_t,memory::default_create> ;
//*****************************************************************************************************
//*****************************************************************************************************
} /* namespace grace */

#endif /* INCLUDE_GRACE_SYSTEM_MPI_RUNTIME */
